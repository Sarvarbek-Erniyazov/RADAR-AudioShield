"""Evaluate a trained BioPhys-HyperRADAR checkpoint with grouped metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from biophys_hyperradar.dataset import DiffSSDDataset, collate_batch
from biophys_hyperradar.labels import ID_TO_TRANSFORM_STATE
from biophys_hyperradar.metrics import binary_metrics, grouped_metrics
from biophys_hyperradar.models import BioPhysHyperRADAR, ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default=str(Path(__file__).resolve().parents[2] / "datasets" / "03_DiffSSD"),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-json", default="implementation/runs/eval_metrics.json")
    parser.add_argument("--output-csv", default="implementation/runs/eval_records.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--score-mode", choices=["logit", "prototype", "fused"], default="logit")
    parser.add_argument("--allow-single-class-debug", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = ModelConfig(**checkpoint["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BioPhysHyperRADAR(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = DiffSSDDataset(
        args.dataset_root,
        split=args.split,
        sample_rate=config.sample_rate,
        require_both_classes=not args.allow_single_class_debug,
        random_crop=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    print(
        f"device={device} split={args.split} items={len(dataset)} batches={len(loader)} "
        f"batch_size={args.batch_size} workers={args.num_workers}"
    )
    records = collect_records(
        model,
        loader,
        device,
        score_mode=args.score_mode,
        disable_progress=args.no_progress,
    )
    metrics = {
        "overall": binary_metrics(
            [int(record["target"]) for record in records],
            [float(record["score"]) for record in records],
        ),
        "groups": grouped_metrics(
            records,
            group_keys=["method_name", "category", "source", "speaker_id", "accent", "media_state"],
        ),
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_json).open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with Path(args.output_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(json.dumps(metrics["overall"], indent=2))
    print(f"Wrote {args.output_json} and {args.output_csv}")


@torch.no_grad()
def collect_records(
    model: BioPhysHyperRADAR,
    loader: DataLoader,
    device: torch.device,
    score_mode: str,
    disable_progress: bool,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    pbar = make_progress(loader, desc="evaluate", total=len(loader), disable=disable_progress)
    for batch in pbar:
        tensor_batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        outputs = model(tensor_batch["waveform"])
        logit_scores = torch.sigmoid(outputs["spoof_logit"])
        prototype_scores = torch.softmax(-outputs["target_distances"], dim=1)[:, 1]
        if score_mode == "logit":
            scores = logit_scores
        elif score_mode == "prototype":
            scores = prototype_scores
        elif score_mode == "fused":
            scores = 0.5 * (logit_scores + prototype_scores)
        else:
            raise ValueError(f"Unknown score mode: {score_mode}")
        scores_list = scores.cpu().tolist()
        logit_scores_list = logit_scores.cpu().tolist()
        prototype_scores_list = prototype_scores.cpu().tolist()
        media_ids = outputs["media_logits"].argmax(dim=1).cpu().tolist()
        targets = tensor_batch["target_long"].cpu().tolist()
        for idx, score in enumerate(scores_list):
            records.append(
                {
                    "filename": batch["filename"][idx],
                    "target": int(targets[idx]),
                    "score": float(score),
                    "logit_score": float(logit_scores_list[idx]),
                    "prototype_score": float(prototype_scores_list[idx]),
                    "method_name": batch["method_name"][idx],
                    "category": batch["category"][idx],
                    "source": batch["source"][idx],
                    "speaker_id": batch["speaker_id"][idx],
                    "accent": batch["accent"][idx],
                    "media_state": ID_TO_TRANSFORM_STATE[int(media_ids[idx])],
                }
            )
        if hasattr(pbar, "set_postfix"):
            pbar.set_postfix(items=len(records))
    return records


def make_progress(iterable: object, desc: str, total: int, disable: bool) -> object:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, leave=False, disable=disable)


if __name__ == "__main__":
    main()
