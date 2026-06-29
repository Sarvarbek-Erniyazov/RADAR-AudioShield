"""UPC stage-1 training for AudioShield-X (e001)."""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from audioshield.data.manifest import read_manifest
from audioshield.data.unified_dataset import UnifiedAudioDataset, collate_unified
from audioshield.data.samplers import BMIQuotaSampler
from audioshield.models.detector import AudioShieldX
from audioshield.training.optim import build_optimizer
from audioshield.training.loop import train_one_epoch, validate
from audioshield.training.early_stopping import MeanCrossCorpusStopper, mean_cross_corpus_eer
from audioshield.losses.latent_aug import CalasController
from audioshield.utils.runtime import describe_device


def load_cfg(exp_path, model_path):
    cfg = yaml.safe_load(open(exp_path))
    cfg.update(yaml.safe_load(open(model_path)))   # adds "model"
    return cfg


def build_train_rows(cfg):
    rows = []
    for c in cfg["experiment"]["train_corpora"]:
        rows += read_manifest(Path(cfg["experiment"]["manifest_dir"]) / f"{c}.csv",
                              splits=["train"], corpora=[c])
    return rows


def build_val_loaders(cfg, vocab, device, max_items):
    loaders = {}
    BONA_ONLY = {"vctk"}
    for c in cfg["experiment"]["train_corpora"]:
        if c in BONA_ONLY:
            continue  # no spoof -> EER undefined; not a selection signal
        rows = read_manifest(Path(cfg["experiment"]["manifest_dir"]) / f"{c}.csv",
                             splits=["val"], corpora=[c])
        if not rows:
            continue
        if max_items and len(rows) > max_items:
            import random
            random.Random(13).shuffle(rows); rows = rows[:max_items]
        ds = UnifiedAudioDataset(rows, cfg["experiment"]["data_root"],
                                 sample_rate=cfg["experiment"]["sample_rate"],
                                 duration_seconds=cfg["experiment"]["duration_seconds"],
                                 random_crop=False,
                                 corpus_vocab=vocab["corpus"], bona_source_vocab=vocab["bona"])
        loaders[c] = DataLoader(ds, batch_size=16, shuffle=False,
                                num_workers=cfg["experiment"].get("num_workers", 4) if False else 2,
                                collate_fn=collate_unified)
    return loaders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-config", default="configs/experiments/e001_unified_v1.yaml")
    ap.add_argument("--model-config", default="configs/models/audioshield_x_v1.yaml")
    ap.add_argument("--output-dir", default="runs/e001_unified_v1")
    ap.add_argument("--max-train-batches", type=int, default=0, help="0 = full epoch; >0 = dry run")
    args = ap.parse_args()

    cfg = load_cfg(args.exp_config, args.model_config)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    describe_device(device)
    use_amp = device.type == "cuda" and not cfg["train"].get("no_amp", False)

    train_rows = build_train_rows(cfg)
    # build stable vocabularies across all train rows
    corpus_vocab = {c: i for i, c in enumerate(sorted({r.corpus for r in train_rows}))}
    bona_vocab = {s: i for i, s in enumerate(sorted({r.bona_fide_source for r in train_rows if r.target == 0}))}
    vocab = {"corpus": corpus_vocab, "bona": bona_vocab}

    train_ds = UnifiedAudioDataset(
        train_rows, cfg["experiment"]["data_root"],
        sample_rate=cfg["experiment"]["sample_rate"],
        duration_seconds=cfg["experiment"]["duration_seconds"],
        random_crop=True, corpus_vocab=corpus_vocab, bona_source_vocab=bona_vocab)
    sampler = BMIQuotaSampler(
        train_rows, batch_size=cfg["train"]["batch_size"],
        n_bona_domains=cfg["train"]["n_bona_domains"],
        min_per_domain=cfg["train"]["min_per_domain"],
        seed=cfg["experiment"]["seed"])
    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              num_workers=cfg["train"].get("num_workers", 4),
                              collate_fn=collate_unified)
    val_loaders = build_val_loaders(cfg, vocab, device, cfg["train"]["max_val_items_per_corpus"])

    model = AudioShieldX(cfg).to(device)
    optimizer = build_optimizer(model, head_lr=cfg["train"]["head_lr"],
                                weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    calas = CalasController(**cfg["calas"])
    stopper = MeanCrossCorpusStopper(patience=cfg["train"]["early_stopping_patience"])

    print(f"device={device} amp={use_amp} train_rows={len(train_rows)} "
          f"bona_domains={list(bona_vocab)} corpora={list(corpus_vocab)} "
          f"batches/epoch={len(sampler)} val_corpora={list(val_loaders)}")
    json.dump({"corpus_vocab": corpus_vocab, "bona_vocab": bona_vocab, "cfg": {k: cfg[k] for k in cfg if k != 'model'}},
              open(out / "run_config.json", "w"), indent=2)

    if args.max_train_batches > 0:
        # DRY RUN: a few batches + one quick val, no early stopping
        from itertools import islice
        import types
        small = list(islice(iter(sampler), args.max_train_batches))
        dry_loader = DataLoader(train_ds, batch_sampler=small,
                                num_workers=0, collate_fn=collate_unified)
        t = train_one_epoch(model, dry_loader, optimizer, scaler, device, cfg, calas,
                            grl_lambda=0.1, use_amp=use_amp)
        print("DRY train terms:", {k: round(v, 4) for k, v in t.items()})
        per = validate(model, val_loaders, device, use_amp)
        print("DRY per-corpus val EER:", {k: round(v, 4) for k, v in per.items()})
        print("DRY mean cross-corpus EER:", round(mean_cross_corpus_eer(per), 4))
        print("CALAS betas:", {c: round(calas.beta(c), 3) for c in corpus_vocab})
        return

    best_eer = float("inf")
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        ramp = min(1.0, epoch / max(1, cfg["train"]["grl_warmup_epochs"]))
        grl_lambda = cfg["train"]["grl_lambda_max"] * ramp
        terms = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, calas,
                                grl_lambda=grl_lambda, use_amp=use_amp,
                                max_steps=cfg['train'].get('max_steps_per_epoch', 0))
        per = validate(model, val_loaders, device, use_amp)
        mean_eer = mean_cross_corpus_eer(per)
        improved, stop = stopper.update(mean_eer, epoch)
        dt = time.time() - t0
        print(f"epoch={epoch} dt={dt:.0f}s loss={terms['loss']:.4f} "
              f"grl={grl_lambda:.3f} mean_dev_eer={mean_eer:.4f} per={ {k: round(v,4) for k,v in per.items()} } "
              f"betas={ {c: round(calas.beta(c),3) for c in corpus_vocab} }")

        ckpt = {"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                "per_corpus_eer": per, "mean_dev_eer": mean_eer,
                "corpus_vocab": corpus_vocab, "bona_vocab": bona_vocab}
        torch.save(ckpt, out / "last.pt")
        if improved:
            best_eer = mean_eer
            torch.save(ckpt, out / "best.pt")
        if stop:
            print(f"early_stop epoch={epoch} best_mean_dev_eer={best_eer:.4f}")
            break


if __name__ == "__main__":
    main()
