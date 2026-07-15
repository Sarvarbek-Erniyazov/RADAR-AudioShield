"""Manifest-driven batch runner for paired waveform counterfactuals.

Reads a manifest slice (optionally filtered by corpus/split, capped by
--limit), applies one transform from audioshield.counterfactuals.registry at
one or more doses to each row's audio, and writes the counterfactual .wav
files plus a provenance.jsonl (one record per output file, joining the
transform's own provenance dict with the source row's utt_id/corpus/split/
target) to --out-dir.

NOT yet run against any real corpus -- this script exists and is tested
(tests/test_counterfactuals_batch_runner.py, synthetic manifest + audio in a
tmp_path) but Step 4's actual intervention battery run is a separate, later
step.

Usage:
    python scripts/build_counterfactuals.py \
        --manifest manifests/v2/asvspoof5.csv --data-root .. \
        --out-dir runs/counterfactuals/asvspoof5_noise \
        --transform noise --doses 20,10,0 --seed 13 \
        --corpora asvspoof5 --splits test --limit 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from audioshield.counterfactuals.codec import check_ffmpeg_available
from audioshield.counterfactuals.registry import get_transform
from audioshield.data.manifest import read_manifest
from audioshield.data.safe_audio import load_allowlist, load_audio_strict

TRANSFORM_EXTRA_ARGS = {
    "codec": ("codec",),
    "rir": ("rir_root",),
    "replay": ("rir_root",),
    "noise": ("noise_type",),
    "resample": (),
}


def _safe_filename(utt_id: str) -> str:
    stem = Path(utt_id).with_suffix("")  # drop the original extension; output is always .wav
    return str(stem).replace("/", "__").replace("\\", "__")


def _parse_doses(raw: str) -> list[float]:
    doses = [float(d) for d in raw.split(",") if d.strip()]
    if not doses:
        raise ValueError(f"--doses parsed to an empty list from {raw!r}")
    return doses


def build_counterfactuals(
    manifest_path: str | Path,
    data_root: str | Path,
    out_dir: str | Path,
    transform_name: str,
    doses: list[float],
    seed: int,
    corpora: list[str] | None = None,
    splits: list[str] | None = None,
    limit: int | None = None,
    codec: str = "opus",
    rir_root: str | None = None,
    noise_type: str = "white",
    allowlist_path: str = "configs/known_bad.txt",
) -> dict:
    """Run the batch job. Returns a summary dict (also useful for tests)."""
    if transform_name not in TRANSFORM_EXTRA_ARGS:
        raise ValueError(f"unknown transform {transform_name!r}; available: {sorted(TRANSFORM_EXTRA_ARGS)}")
    if transform_name == "codec":
        check_ffmpeg_available()  # fail loudly before touching any audio
    needed_extra = TRANSFORM_EXTRA_ARGS[transform_name]
    if "rir_root" in needed_extra and not rir_root:
        raise ValueError(f"--rir-root is required for transform={transform_name!r}")

    rows = read_manifest(manifest_path, splits=splits, corpora=corpora)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(
            f"0 manifest rows selected from {manifest_path} (corpora={corpora}, splits={splits}) "
            "-- refusing to run a batch job over nothing"
        )

    transform_fn = get_transform(transform_name)
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    allowlist = load_allowlist(allowlist_path)

    provenance_path = out_dir / "provenance.jsonl"
    n_written, n_skipped_allowlisted = 0, 0
    with provenance_path.open("w", encoding="utf-8") as prov_f:
        for row in rows:
            full_path = data_root / row.path
            loaded = load_audio_strict(full_path, row.utt_id, allowlist)
            if loaded is None:
                n_skipped_allowlisted += 1
                continue
            waveform, sr = loaded

            for dose in doses:
                kwargs = {}
                if "codec" in needed_extra:
                    kwargs["codec"] = codec
                if "rir_root" in needed_extra:
                    kwargs["rir_root"] = rir_root
                if "noise_type" in needed_extra:
                    kwargs["noise_type"] = noise_type

                aligned, provenance = transform_fn(waveform, sr, dose, seed, **kwargs)

                dose_dir = out_dir / transform_name / str(dose)
                dose_dir.mkdir(parents=True, exist_ok=True)
                out_path = dose_dir / f"{_safe_filename(row.utt_id)}.wav"
                sf.write(out_path, aligned, sr)

                record = dict(
                    utt_id=row.utt_id, corpus=row.corpus, split=row.split, target=row.target,
                    source_path=str(full_path), out_path=str(out_path), **provenance,
                )
                prov_f.write(json.dumps(record) + "\n")
                n_written += 1

    return dict(
        n_rows_selected=len(rows), n_written=n_written,
        n_skipped_allowlisted=n_skipped_allowlisted, provenance_path=str(provenance_path),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default="..")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--transform", required=True, choices=sorted(TRANSFORM_EXTRA_ARGS))
    ap.add_argument("--doses", required=True, help="comma-separated dose values")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--corpora", nargs="*", default=None)
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--codec", default="opus", choices=["opus", "aac", "mp3"])
    ap.add_argument("--rir-root", default=None)
    ap.add_argument("--noise-type", default="white", choices=["white", "pink"])
    ap.add_argument("--allowlist", default="configs/known_bad.txt")
    args = ap.parse_args()

    summary = build_counterfactuals(
        manifest_path=args.manifest, data_root=args.data_root, out_dir=args.out_dir,
        transform_name=args.transform, doses=_parse_doses(args.doses), seed=args.seed,
        corpora=args.corpora, splits=args.splits, limit=args.limit,
        codec=args.codec, rir_root=args.rir_root, noise_type=args.noise_type,
        allowlist_path=args.allowlist,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
