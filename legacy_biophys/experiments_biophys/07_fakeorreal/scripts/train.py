"""Train BioPhys-HyperRADAR on DiffSSD."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from biophys_hyperradar.dataset import DiffSSDDataset, collate_batch
from biophys_hyperradar.labels import KNOWN_METHODS, TRANSFORM_STATES
from biophys_hyperradar.losses import LossWeights, MultiObjectiveLoss
from biophys_hyperradar.metrics import binary_metrics
from biophys_hyperradar.models import BioPhysHyperRADAR, ModelConfig
from biophys_hyperradar.transforms import make_augmented_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default=str(Path(__file__).resolve().parents[2] / "datasets" / "03_DiffSSD"),
    )
    parser.add_argument("--output-dir", default="implementation/runs/diffssd_biophys_hyperradar")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration-seconds", type=float, default=4.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--max-train-items", type=int)
    parser.add_argument("--max-val-items", type=int)
    parser.add_argument("--ssl-model-name")
    parser.add_argument("--unfreeze-ssl", action="store_true")
    parser.add_argument("--allow-single-class-debug", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--score-mode", choices=["logit", "prototype", "fused"], default="logit")
    parser.add_argument("--spoof-loss-weight", type=float, default=1.0)
    parser.add_argument("--method-loss-weight", type=float, default=0.2)
    parser.add_argument("--media-state-loss-weight", type=float, default=0.2)
    parser.add_argument("--target-prototype-loss-weight", type=float, default=0.3)
    parser.add_argument("--method-prototype-loss-weight", type=float, default=0.2)
    parser.add_argument("--bona-fide-compactness-weight", type=float, default=0.1)
    parser.add_argument("--energy-loss-weight", type=float, default=0.02)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.1)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--spoof-pos-weight", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--disable-early-stopping", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require_both = not args.allow_single_class_debug

    train_ds = DiffSSDDataset(
        args.dataset_root,
        split="train",
        sample_rate=args.sample_rate,
        duration_seconds=args.duration_seconds,
        max_items=args.max_train_items,
        require_both_classes=require_both,
        random_crop=True,
    )
    val_ds = DiffSSDDataset(
        args.dataset_root,
        split="val",
        sample_rate=args.sample_rate,
        duration_seconds=args.duration_seconds,
        max_items=args.max_val_items,
        require_both_classes=require_both,
        random_crop=False,
    )

    train_sampler = build_balanced_sampler(train_ds) if args.balanced_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    config = ModelConfig(
        sample_rate=args.sample_rate,
        ssl_model_name=args.ssl_model_name,
        freeze_ssl=not args.unfreeze_ssl,
    )
    model = BioPhysHyperRADAR(config).to(device)
    loss_weights = LossWeights(
        spoof=args.spoof_loss_weight,
        method=args.method_loss_weight,
        media_state=args.media_state_loss_weight,
        target_prototype=args.target_prototype_loss_weight,
        method_prototype=args.method_prototype_loss_weight,
        bona_fide_compactness=args.bona_fide_compactness_weight,
        energy=args.energy_loss_weight,
        consistency=args.consistency_loss_weight,
    )
    criterion = MultiObjectiveLoss(
        loss_weights,
        focal_gamma=args.focal_gamma,
        spoof_pos_weight=args.spoof_pos_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and not args.no_amp)
    use_amp = device.type == "cuda" and not args.no_amp

    print(
        f"device={device} amp={use_amp} train_items={len(train_ds)} val_items={len(val_ds)} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"batch_size={args.batch_size} workers={args.num_workers} "
        f"balanced_sampler={args.balanced_sampler} score_mode={args.score_mode}"
    )

    config_payload = vars(args).copy()
    config_payload.update({"methods": KNOWN_METHODS, "media_states": TRANSFORM_STATES})
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    best_eer = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_terms = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            sample_rate=args.sample_rate,
            augment=not args.no_augment,
            use_amp=use_amp,
            epoch=epoch,
            total_epochs=args.epochs,
            disable_progress=args.no_progress,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            score_mode=args.score_mode,
            epoch=epoch,
            total_epochs=args.epochs,
            disable_progress=args.no_progress,
        )
        elapsed = time.time() - started
        print(
            f"epoch={epoch} seconds={elapsed:.1f} "
            f"loss={train_terms['total']:.4f} val_eer={val_metrics['eer']:.4f} "
            f"val_ece={val_metrics['ece']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "config": config.__dict__,
            "epoch": epoch,
            "val_metrics": val_metrics,
            "methods": KNOWN_METHODS,
            "media_states": TRANSFORM_STATES,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        current_eer = val_metrics["eer"]
        improved = (
            epoch == 1
            or (not math.isnan(current_eer) and current_eer < best_eer - args.early_stopping_min_delta)
        )
        if improved:
            best_eer = current_eer
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        if (
            not args.disable_early_stopping
            and args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                f"early_stop epoch={epoch} best_val_eer={best_eer:.4f} "
                f"patience={args.early_stopping_patience}"
            )
            break


def train_one_epoch(
    model: BioPhysHyperRADAR,
    loader: DataLoader,
    criterion: MultiObjectiveLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    sample_rate: int,
    augment: bool,
    use_amp: bool,
    epoch: int,
    total_epochs: int,
    disable_progress: bool,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    steps = 0
    pbar = make_progress(
        loader,
        desc=f"train {epoch}/{total_epochs}",
        total=len(loader),
        disable=disable_progress,
    )
    for batch in pbar:
        batch = move_tensor_batch(batch, device)
        waveform = batch["waveform"]
        view_a, media_a = make_augmented_batch(waveform, sample_rate, enabled=augment)
        view_b, media_b = make_augmented_batch(waveform, sample_rate, enabled=augment)
        batch_a = dict(batch)
        batch_b = dict(batch)
        batch_a["media_state"] = media_a
        batch_b["media_state"] = media_b

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs_a = model(view_a)
            outputs_b = model(view_b)
            loss, terms = criterion(outputs_a, batch_a, outputs_b, batch_b)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0) + value
        steps += 1
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(
                loss=f"{totals['total'] / steps:.4f}",
                spoof=f"{totals['spoof'] / steps:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

    return {key: value / max(1, steps) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: BioPhysHyperRADAR,
    loader: DataLoader,
    device: torch.device,
    score_mode: str,
    epoch: int,
    total_epochs: int,
    disable_progress: bool,
) -> dict[str, float]:
    model.eval()
    labels = []
    scores = []
    pbar = make_progress(
        loader,
        desc=f"valid {epoch}/{total_epochs}",
        total=len(loader),
        disable=disable_progress,
    )
    for batch in pbar:
        batch = move_tensor_batch(batch, device)
        outputs = model(batch["waveform"])
        labels.extend(batch["target_long"].cpu().tolist())
        scores.extend(select_scores(outputs, score_mode).cpu().tolist())
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(items=len(labels))
    return binary_metrics(labels, scores)


def move_tensor_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def select_scores(outputs: dict[str, torch.Tensor], score_mode: str) -> torch.Tensor:
    logit_scores = torch.sigmoid(outputs["spoof_logit"])
    if score_mode == "logit":
        return logit_scores
    prototype_scores = torch.softmax(-outputs["target_distances"], dim=1)[:, 1]
    if score_mode == "prototype":
        return prototype_scores
    if score_mode == "fused":
        return 0.5 * (logit_scores + prototype_scores)
    raise ValueError(f"Unknown score mode: {score_mode}")


def build_balanced_sampler(dataset: DiffSSDDataset) -> WeightedRandomSampler:
    counts: dict[int, int] = {}
    for row in dataset.rows:
        counts[row.target] = counts.get(row.target, 0) + 1
    weights = [1.0 / counts[row.target] for row in dataset.rows]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def make_progress(iterable: object, desc: str, total: int, disable: bool) -> object:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, leave=False, disable=disable)


if __name__ == "__main__":
    main()
