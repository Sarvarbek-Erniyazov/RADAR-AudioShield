"""e002 training: channel-consistency on a frozen (optionally top-k unfrozen)
WavLM backbone. ASVspoof5-only by default. No BMI / CALAS / prototypes.

Two arms, both via this one script (switch by config):
  arm A (augmentation-only):  consistency.lambda_kl = lambda_emb = 0
  arm B (consistency):        consistency.lambda_kl = 1.0, lambda_emb = 0.5

Held-out OOD corpora are NEVER loaded here -- selection uses an internal
degraded dev split so ITW/ReplayDF/AI4T stay honest test sets for cross_test.
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import random
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from audioshield.data.manifest import read_manifest
from audioshield.data.unified_dataset import UnifiedAudioDataset, collate_unified
from audioshield.models.detector import AudioShieldX
from audioshield.training.optim import build_optimizer
from audioshield.training.loop_e002 import train_one_epoch_e002, validate_e002
from audioshield.training.early_stopping import MeanCrossCorpusStopper, mean_cross_corpus_eer


def load_cfg(exp_path, model_path):
    cfg = yaml.safe_load(open(exp_path))
    cfg.update(yaml.safe_load(open(model_path)))
    return cfg


def maybe_unfreeze_top_k(model, k: int):
    """Unfreeze the top-k transformer layers of the WavLM backbone if k>0.
    Best-effort: tries common attribute paths; prints what it froze/unfroze."""
    if k <= 0:
        print("[e002] backbone fully frozen (unfreeze_top_k=0)")
        return
    enc = None
    for path in ["backbone.model.encoder.layers", "backbone.encoder.layers",
                 "ssl.model.encoder.layers", "ssl_backbone.model.encoder.layers"]:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            enc = obj; chosen = path; break
        except AttributeError:
            continue
    if enc is None:
        print("[e002][warn] could not locate encoder.layers; staying frozen. "
              "Inspect model and set the right path in maybe_unfreeze_top_k.")
        return
    n = len(enc)
    for i, layer in enumerate(enc):
        req = i >= (n - k)
        for p in layer.parameters():
            p.requires_grad = req
    print(f"[e002] unfroze top {k}/{n} backbone layers via {chosen}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-config", default="configs/experiments/e002_consistency_v1.yaml")
    ap.add_argument("--model-config", default="configs/models/audioshield_x_v1.yaml")
    ap.add_argument("--output-dir", default="runs/e002_consistency_v1")
    ap.add_argument("--max-train-batches", type=int, default=0)
    args = ap.parse_args()

    cfg = load_cfg(args.exp_config, args.model_config)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not cfg["train"].get("no_amp", False)

    exp = cfg["experiment"]
    train_corpora = exp["train_corpora"]
    md = exp["manifest_dir"]

    # ---- train rows (single corpus by default) ----
    train_rows = []
    for c in train_corpora:
        train_rows += read_manifest(Path(md) / f"{c}.csv", splits=["train"], corpora=[c])
    corpus_vocab = {c: i for i, c in enumerate(sorted({r.corpus for r in train_rows}))}
    bona_vocab = {s: i for i, s in enumerate(sorted({r.bona_fide_source for r in train_rows if r.target == 0}))}

    train_ds = UnifiedAudioDataset(
        train_rows, exp["data_root"], sample_rate=exp["sample_rate"],
        duration_seconds=exp["duration_seconds"], random_crop=True,
        corpus_vocab=corpus_vocab, bona_source_vocab=bona_vocab,
        degrade=True)                                   # <-- e002: degraded view on
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"].get("num_workers", 4),
        collate_fn=collate_unified, drop_last=True)

    # ---- internal dev: held-out slice of the SAME corpus, degraded too ----
    dev_loaders = {}
    for c in train_corpora:
        rows = read_manifest(Path(md) / f"{c}.csv", splits=["val"], corpora=[c])
        if not rows:
            continue
        cap = cfg["train"].get("max_val_items_per_corpus", 2000)
        if cap and len(rows) > cap:
            random.Random(13).shuffle(rows); rows = rows[:cap]
        # degraded dev: select on robustness, not clean accuracy
        ds_clean = UnifiedAudioDataset(rows, exp["data_root"], sample_rate=exp["sample_rate"],
                                       duration_seconds=exp["duration_seconds"], random_crop=False,
                                       corpus_vocab=corpus_vocab, bona_source_vocab=bona_vocab)
        dev_loaders[c + "_clean"] = DataLoader(ds_clean, batch_size=16, shuffle=False,
                                               num_workers=2, collate_fn=collate_unified)
        ds_deg = UnifiedAudioDataset(rows, exp["data_root"], sample_rate=exp["sample_rate"],
                                     duration_seconds=exp["duration_seconds"], random_crop=False,
                                     corpus_vocab=corpus_vocab, bona_source_vocab=bona_vocab,
                                     degrade=True)
        # validate_e002 reads "waveform"; for the degraded dev we want it to score the
        # degraded audio, so swap it in via a tiny wrapper collate.
        def _deg_collate(items):
            b = collate_unified(items)
            b["waveform"] = b["waveform_deg"]
            return b
        dev_loaders[c + "_deg"] = DataLoader(ds_deg, batch_size=16, shuffle=False,
                                             num_workers=2, collate_fn=_deg_collate)

    model = AudioShieldX(cfg).to(device)
    maybe_unfreeze_top_k(model, int(cfg["train"].get("unfreeze_top_k", 0)))
    optimizer = build_optimizer(model, head_lr=cfg["train"]["head_lr"],
                                weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    stopper = MeanCrossCorpusStopper(patience=cfg["train"]["early_stopping_patience"])

    print(f"device={device} amp={use_amp} train_rows={len(train_rows)} "
          f"corpora={list(corpus_vocab)} dev={list(dev_loaders)} "
          f"lambda_kl={cfg['consistency'].get('lambda_kl')} "
          f"lambda_emb={cfg['consistency'].get('lambda_emb')} "
          f"unfreeze_top_k={cfg['train'].get('unfreeze_top_k', 0)}")
    json.dump({"corpus_vocab": corpus_vocab, "bona_vocab": bona_vocab,
               "consistency": cfg["consistency"],
               "cfg": {k: cfg[k] for k in cfg if k != 'model'}},
              open(out / "run_config.json", "w"), indent=2)

    if args.max_train_batches > 0:
        from itertools import islice
        small = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                           num_workers=0, collate_fn=collate_unified, drop_last=True)
        t = train_one_epoch_e002(model, small, optimizer, scaler, device, cfg,
                                 use_amp=use_amp, max_steps=args.max_train_batches)
        print("DRY train terms:", {k: round(v, 4) for k, v in t.items()})
        per = validate_e002(model, dev_loaders, device, use_amp)
        print("DRY dev EER:", {k: round(v, 4) for k, v in per.items()})
        return

    best = float("inf")
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        terms = train_one_epoch_e002(model, train_loader, optimizer, scaler, device, cfg,
                                     use_amp=use_amp,
                                     max_steps=cfg["train"].get("max_steps_per_epoch", 0))
        per = validate_e002(model, dev_loaders, device, use_amp)
        # select on degraded-dev mean EER (robustness), fall back to all if naming differs
        deg = {k: v for k, v in per.items() if k.endswith("_deg")}
        mean_eer = mean_cross_corpus_eer(deg or per)
        improved, stop = stopper.update(mean_eer, epoch)
        print(f"epoch={epoch} dt={time.time()-t0:.0f}s loss={terms['loss']:.4f} "
              f"cls={terms['cls']:.4f} con={terms['con']:.4f} "
              f"mean_deg_dev_eer={mean_eer:.4f} per={ {k: round(v,4) for k,v in per.items()} }")
        ckpt = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                "per_corpus_eer": per, "mean_dev_eer": mean_eer,
                "corpus_vocab": corpus_vocab, "bona_vocab": bona_vocab}
        torch.save(ckpt, out / "last.pt")
        if improved:
            best = mean_eer
            torch.save(ckpt, out / "best.pt")
        if stop:
            print(f"early_stop epoch={epoch} best={best:.4f}")
            break


if __name__ == "__main__":
    main()
