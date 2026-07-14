"""e002 training: channel-consistency on frozen (optionally top-k unfrozen)
WavLM. ASVspoof5-only by default. No BMI / CALAS / prototypes.
Arm A (aug-only): consistency.lambda_*=0. Arm B: lambda_kl=1.0, lambda_emb=0.5.
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
from torch.utils.data import DataLoader, WeightedRandomSampler

from audioshield.data.manifest import read_manifest
from audioshield.data.unified_dataset import UnifiedAudioDataset, collate_unified
from audioshield.models.detector import AudioShieldX
from audioshield.training.optim import (
    build_optimizer,
    build_optimizer_stage2,
    unfreeze_top_k,
    unfreeze_weighted_band,
)
from audioshield.training.loop_e002 import train_one_epoch_e002, validate_e002, probe_corpus_during_train
from audioshield.training.early_stopping import MeanCrossCorpusStopper, mean_cross_corpus_eer
from audioshield.utils.runtime import describe_device
from audioshield.utils.seeding import seed_everything, dataloader_seed_kwargs


def load_cfg(exp_path, model_path):
    def deep_update(base, override):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                deep_update(base[key], value)
            else:
                base[key] = value
        return base

    cfg = yaml.safe_load(open(model_path))
    return deep_update(cfg, yaml.safe_load(open(exp_path)))


def _deg_collate(items):
    """Module-level so Windows spawn workers can pickle it. Scores degraded audio."""
    b = collate_unified(items)
    b["waveform"] = b["waveform_deg"]
    return b


def _loader_kwargs(num_workers: int, device: torch.device, persistent: bool = True) -> dict:
    num_workers = int(num_workers)
    kwargs = {"num_workers": num_workers, "pin_memory": device.type == "cuda"}
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent
        kwargs["prefetch_factor"] = 2
    return kwargs


def _configure_backbone_finetuning(model, cfg: dict) -> bool:
    train_cfg = cfg["train"]
    k = int(train_cfg.get("unfreeze_top_k", 0))
    if k <= 0:
        print("[e002] backbone fully frozen (unfreeze_top_k=0)")
        return False

    strategy = str(train_cfg.get("unfreeze_strategy", "top_k")).lower()
    if strategy in {"weighted_band", "layer_weight_band", "ssl_band"}:
        model_cfg = cfg.get("model", {})
        band = train_cfg.get("unfreeze_layer_band", model_cfg.get("layer_weight_init_band", [8, 11]))
        center = int(model_cfg.get("layer_weight_init_center", (int(band[0]) + int(band[1])) // 2))
        plan = unfreeze_weighted_band(model, band, k=k, center=center)
        hs_lo, hs_hi = plan["hidden_state_window"]
        enc_layers = plan["encoder_indices"]
        print(
            "[e002] stage2: unfreeze_strategy=weighted_band "
            f"hidden_states={hs_lo}-{hs_hi} -> encoder.layers={enc_layers}"
        )
    elif strategy == "top_k":
        unfreeze_top_k(model, k)
        print(f"[e002] stage2: unfreeze_strategy=top_k top_layers={k}")
    else:
        raise ValueError(f"Unknown train.unfreeze_strategy={strategy!r}")

    return True


def make_balanced_sampler(train_rows, num_samples):
    """Per-sample weight = 1/(corpus size) * 1/(class size within corpus).
    Equalizes corpus contribution AND pulls classes toward 50/50."""
    from collections import Counter, defaultdict
    n_corpora = len(set(r.corpus for r in train_rows))
    cls_n = Counter((r.corpus, r.target) for r in train_rows)
    classes_in_corpus = defaultdict(set)
    for r in train_rows:
        classes_in_corpus[r.corpus].add(r.target)
    weights = [
        (1.0 / n_corpora)
        * (1.0 / len(classes_in_corpus[r.corpus]))
        * (1.0 / cls_n[(r.corpus, r.target)])
        for r in train_rows
    ]
    import torch as _t
    return WeightedRandomSampler(_t.DoubleTensor(weights),
                                 num_samples=num_samples, replacement=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-config", default="configs/experiments/e002_consistency_v1.yaml")
    ap.add_argument("--model-config", default="configs/models/audioshield_x_v1.yaml")
    ap.add_argument("--output-dir", default="runs/e002_consistency_v1")
    ap.add_argument("--max-train-batches", type=int, default=0)
    ap.add_argument("--resume", default="", help="path to checkpoint to resume from")
    ap.add_argument("--max-hours", type=float, default=0.0, help="wall-clock budget; checkpoint+exit when exceeded")
    ap.add_argument("--override", nargs="*", default=[],
                    help="dotted-key overrides, e.g. train.batch_size=8 train.unfreeze_top_k=4")
    args = ap.parse_args()

    cfg = load_cfg(args.exp_config, args.model_config)
    for ov in args.override:
        key, _, val = ov.partition("=")
        # parse value: int, float, bool, or string
        if val.lower() in ("true", "false"):
            pv = val.lower() == "true"
        else:
            try: pv = int(val)
            except ValueError:
                try: pv = float(val)
                except ValueError: pv = val
        d = cfg; parts = key.split(".")
        for k in parts[:-1]:
            d = d.setdefault(k, {})
        d[parts[-1]] = pv
        print(f"[override] {key} = {pv!r}")

    seed = int(cfg["experiment"]["seed"])
    seed_info = seed_everything(seed)
    print(f"[e002] seeded: {seed_info}")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    describe_device(device)
    use_amp = device.type == "cuda" and not cfg["train"].get("no_amp", False)

    exp = cfg["experiment"]
    train_corpora = exp["train_corpora"]
    md = exp["manifest_dir"]

    train_rows = []
    for c in train_corpora:
        train_rows += read_manifest(Path(md) / f"{c}.csv", splits=["train"], corpora=[c])
    corpus_vocab = {c: i for i, c in enumerate(sorted({r.corpus for r in train_rows}))}
    bona_vocab = {s: i for i, s in enumerate(sorted({r.bona_fide_source for r in train_rows if r.target == 0}))}

    train_ds = UnifiedAudioDataset(
        train_rows, exp["data_root"], sample_rate=exp["sample_rate"],
        duration_seconds=exp["duration_seconds"], random_crop=True,
        corpus_vocab=corpus_vocab, bona_source_vocab=bona_vocab, degrade=True)
    if cfg["train"].get("balanced_sampler", False):
        steps = cfg["train"].get("max_steps_per_epoch", 0) or (len(train_ds)//cfg["train"]["batch_size"])
        n_samp = steps * cfg["train"]["batch_size"]
        sampler = make_balanced_sampler(train_rows, n_samp)
        print(f"[e005] balanced sampler: {n_samp} samples/epoch across "
              f"{len(set(r.corpus for r in train_rows))} corpora")
        train_loader = DataLoader(
            train_ds, batch_size=cfg["train"]["batch_size"], sampler=sampler,
            collate_fn=collate_unified, drop_last=True,
            **_loader_kwargs(cfg["train"].get("num_workers", 4), device),
            **dataloader_seed_kwargs(seed))
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
            collate_fn=collate_unified, drop_last=True,
            **_loader_kwargs(cfg["train"].get("num_workers", 4), device),
            **dataloader_seed_kwargs(seed))

    dev_loaders = {}
    val_workers = int(cfg["train"].get("val_num_workers", min(2, int(cfg["train"].get("num_workers", 4)))))
    val_batch_size = int(cfg["train"].get("val_batch_size", 16))
    for c in train_corpora:
        rows = read_manifest(Path(md) / f"{c}.csv", splits=["val"], corpora=[c])
        if not rows:
            continue
        if len({r.target for r in rows}) < 2:
            print(f"[e002] skipping {c} dev EER: single-class validation split")
            continue
        cap = cfg["train"].get("max_val_items_per_corpus", 2000)
        if cap and len(rows) > cap:
            random.Random(13).shuffle(rows); rows = rows[:cap]
        ds_clean = UnifiedAudioDataset(rows, exp["data_root"], sample_rate=exp["sample_rate"],
                                       duration_seconds=exp["duration_seconds"], random_crop=False,
                                       corpus_vocab=corpus_vocab, bona_source_vocab=bona_vocab)
        dev_loaders[c + "_clean"] = DataLoader(
            ds_clean, batch_size=val_batch_size, shuffle=False,
            collate_fn=collate_unified,
            **_loader_kwargs(val_workers, device),
            **dataloader_seed_kwargs(seed))
        ds_deg = UnifiedAudioDataset(rows, exp["data_root"], sample_rate=exp["sample_rate"],
                                     duration_seconds=exp["duration_seconds"], random_crop=False,
                                     corpus_vocab=corpus_vocab, bona_source_vocab=bona_vocab,
                                     degrade=True)
        dev_loaders[c + "_deg"] = DataLoader(
            ds_deg, batch_size=val_batch_size, shuffle=False,
            collate_fn=_deg_collate,
            **_loader_kwargs(val_workers, device),
            **dataloader_seed_kwargs(seed))

    model = AudioShieldX(cfg).to(device)
    xc_cfg = cfg.get("xc_contrastive", {})
    if xc_cfg.get("enabled", False) and float(xc_cfg.get("lambda_xc", 0.0)) > 0.0:
        if xc_cfg.get("use_projection_head", False):
            if model.contrastive_proj is None:
                raise ValueError(
                    "xc_contrastive.use_projection_head=true requires "
                    "model.contrastive_proj_dim > 0"
                )
            print(
                "[e002] contrastive loss uses projection head "
                f"dim={cfg['model'].get('contrastive_proj_dim')}"
            )
        else:
            print("[e002] contrastive loss uses main embedding")
    if _configure_backbone_finetuning(model, cfg):
        try:
            if cfg["train"].get("gradient_checkpointing", True):
                model.ssl.backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
                print("[e002] gradient checkpointing enabled on backbone")
        except Exception as e:
            print(f"[e002][warn] could not enable gradient checkpointing: {e}")
        bb_lr = float(cfg["train"].get("backbone_lr", 1e-6))
        n_bb = sum(p.numel() for n,p in model.named_parameters()
                   if p.requires_grad and n.startswith("ssl.backbone"))
        print(f"[e002] stage2: backbone_lr={bb_lr}, trainable backbone params={n_bb:,}")
        optimizer = build_optimizer_stage2(
            model, head_lr=cfg["train"]["head_lr"],
            backbone_lr=bb_lr, weight_decay=cfg["train"]["weight_decay"])
    else:
        print("[e002] backbone fully frozen (unfreeze_top_k=0)")
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
               "cfg": {k: cfg[k] for k in cfg if k != 'model'},
               "model_cfg": cfg.get("model", {}),
               "seed_info": seed_info},
              open(out / "run_config.json", "w"), indent=2)

    if args.max_train_batches > 0:
        small = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                           num_workers=0, collate_fn=collate_unified, drop_last=True,
                           **dataloader_seed_kwargs(seed))
        t = train_one_epoch_e002(model, small, optimizer, scaler, device, cfg,
                                 use_amp=use_amp, max_steps=args.max_train_batches)
        print("DRY train terms:", {k: round(v, 4) for k, v in t.items()})
        per = validate_e002(model, dev_loaders, device, use_amp)
        print("DRY dev EER:", {k: round(v, 4) for k, v in per.items()})
        return

    best = float("inf")
    start_epoch = 1
    if args.resume:
        rk = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(rk["model"])
        if "optimizer" in rk:
            try:
                optimizer.load_state_dict(rk["optimizer"]); print(f"[e002] resumed optimizer state")
            except Exception as e:
                print(f"[e002][warn] optimizer resume failed ({e}); continuing with fresh optimizer")
        else:
            print("[e002][warn] checkpoint has no optimizer state; resuming model+epoch only")
        start_epoch = int(rk.get("epoch", 0)) + 1
        best = float(rk.get("mean_dev_eer", float("inf")))
        print(f"[e002] resumed from {args.resume} at epoch {start_epoch} (best={best:.4f})")
    wall_t0 = time.time()
    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        if args.max_hours > 0 and (time.time() - wall_t0) / 3600.0 >= args.max_hours:
            print(f"[e002] max-hours ({args.max_hours}) reached; checkpointing and exiting", flush=True)
            break
        t0 = time.time()
        terms = train_one_epoch_e002(model, train_loader, optimizer, scaler, device, cfg,
                                     use_amp=use_amp,
                                     max_steps=cfg["train"].get("max_steps_per_epoch", 0))
        per = validate_e002(model, dev_loaders, device, use_amp)
        probe_res = None
        if cfg["train"].get("probe_every_epoch", False):
            probe_res = probe_corpus_during_train(model, dev_loaders, device, use_amp)
        deg = {k: v for k, v in per.items() if k.endswith("_deg")}
        mean_eer = mean_cross_corpus_eer(deg or per)
        improved, stop = stopper.update(mean_eer, epoch)
        probe_str = ""
        if probe_res is not None:
            probe_str = (f"probe_bacc={probe_res['balanced_accuracy']:.4f} "
                         f"probe_base={probe_res['majority_baseline']:.4f} ")
        skip_str = (
            f"skip_loss={terms.get('skipped_nonfinite_loss', 0)} "
            f"skip_grad={terms.get('skipped_nonfinite_grad', 0)} "
        )
        xc_str = ""
        if cfg.get("xc_contrastive", {}).get("enabled", False):
            xc_str = (
                f"xc_npos={terms.get('xc_npos', 0.0):.1f} "
                f"xc_skip={terms.get('xc_skipped', 0.0):.3f} "
            )
        print(f"epoch={epoch} dt={time.time()-t0:.0f}s "
              f"steps={terms.get('steps','?')} loss={terms.get('loss',float('nan')):.4f} "
              f"cls={terms.get('cls',float('nan')):.4f} con={terms.get('con',float('nan')):.4f} "
              f"xc={terms.get('xc',float('nan')):.4f} "
              f"mean_deg_dev_eer={mean_eer:.4f} {probe_str}{skip_str}{xc_str}"
              f"per={ {k: round(v,4) for k,v in per.items()} }", flush=True)
        expected = cfg["train"].get("max_steps_per_epoch", 0) or (len(train_ds) // cfg["train"]["batch_size"])
        if terms.get("steps", 0) < 0.5 * expected:
            print(f"[e002][ABORT] epoch {epoch} ran {terms.get('steps')} steps, "
                  f"expected ~{expected}. Dataloader problem -- not saving checkpoint, stopping.", flush=True)
            break
        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "cfg": cfg, "epoch": epoch,
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
