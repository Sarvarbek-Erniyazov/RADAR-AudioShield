"""Step 3 Phase B extraction preflight (step3_phaseB_preflight_brief.md).

Proves the extraction chain (scripts/extract_model_embeddings.py) is
correct on a small (few-hundred-clip) batch, measures throughput, and
projects the full 6-job (e007_A/B/C x DiffSSD/ReplayDF) wall-clock cost --
then STOPS. This script never launches the full extraction itself; that
is an explicit human decision downstream of this preflight's numbers (see
its own module docstring's "Definition of done" note, and
docs/phaseB_extraction_preflight_findings.md for why: the consumption
path -- turning these embeddings into populated w-metrics the gate reads
-- does not exist yet in scripts/run_reliance_battery.py, a real code gap
found while writing this preflight, not touched here).

Reuses scripts/extract_model_embeddings.py's own extraction primitives
(extract_checkpoint_corpus, embedding_dim_of, _build_model_from_checkpoint)
rather than reimplementing them -- this preflight validates that
machinery, it doesn't duplicate it. Deliberately imports NOTHING from
audioshield.reliance (which pulls in scipy.linalg at module level via
audioshield.reliance.metrics -> ._linalg) -- extraction is torch-only, and
gflow is known to segfault on scipy.linalg.eigh (see
extract_model_embeddings.py's own check_torch_scipy_eigh); keeping this
script's import chain scipy-free means it can't ever hit that crash
regardless of what else runs in the same environment.

The decisive check is `check_correctness`: proves the saved, on-disk,
(possibly float16-quantized) embedding really is the classifier's input --
`binary.fc(emb) == logit`, where `logit` comes from a FRESH forward pass
on the same waveforms, independent of the extraction call. If the script
extracted a different 256-d point (e.g. before the final LayerNorm, or a
different hook), this check fails loudly instead of silently producing
uninterpretable downstream reliance metrics.

IMPORTANT -- read docs/phaseB_extraction_preflight_findings.md before
running any of this for real: the extract-consume-w-metrics-gate chain
has a confirmed gap downstream of extraction (scripts/run_reliance_battery.py
cannot yet read a Phase B cache -- wrong array shape assumed, and its
whole per-checkpoint metric stack assumes one embedding space shared
across all checkpoints, which Phase B breaks by construction). This
preflight is still worth running now: it validates EXTRACTION correctness
in isolation (a prerequisite regardless of when the consumption gap is
closed) and is tiny (~500 clips, not the ~440k full run). The FULL 6-job
extraction should wait for a human decision on the consumption gap --
this script never launches it regardless.

COLLABORATOR PC (azimb@DESKTOP-ED3J7RM, TITAN RTX, `gflow` env -- NOT
`repro_2a`). Run in this exact order:

    # 0) Confirm the gflow interpreter path FIRST -- do not assume it
    #    matches repro_2a's path -- and that CUDA sees the TITAN.
    conda env list
    conda run -n gflow python -c \
        "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

    # 1) (Recommended first) the EXISTING, already-tested extraction
    #    preflight mode -- cheap, catches a broken checkpoint/config/
    #    missing audio before this script's own (slightly more expensive)
    #    validation run.
    conda run -n gflow python scripts/extract_model_embeddings.py --preflight \\
        --checkpoint /e/AI_voice_detection/checkpoint_backup/runs/e007_A_fresh/best.pt \\
        --corpus diffssd replaydf --data-root ..

    # 2) THIS preflight: one checkpoint x one corpus x ~500 clips, written
    #    under a _preflight subdirectory (never the real cache namespace,
    #    so a later full run can't mistake this for a completed shard).
    #    Repeat once per (checkpoint, corpus) pair of interest -- the
    #    brief's target set is e007_A/B/C x {diffssd, replaydf}, 6 pairs,
    #    but ONE pair is enough to validate correctness/throughput; this
    #    example uses e007_A_fresh x replaydf.
    conda run -n gflow python scripts/preflight_phaseB_extraction.py \\
        --checkpoint /e/AI_voice_detection/checkpoint_backup/runs/e007_A_fresh/best.pt \\
        --corpus replaydf \\
        --manifest-dir manifests/v2 \\
        --data-root .. \\
        --out-root analysis/step3/_embcache_modelspace/_preflight \\
        --n-clips 500 --batch-size 16 --dtype float32

    #    Prints every check's PASS/FAIL, measured clips/sec, and the
    #    projected full 6-job (e007_A/B/C x DiffSSD/ReplayDF) wall-clock.
    #    THE SEQUENCE STOPS HERE -- no full extraction is launched by
    #    this script or this brief.

    # 3) Clean up the tiny preflight shard once satisfied (it's already
    #    isolated under _preflight/, but remove it so nothing downstream
    #    ever mistakes it for a completed real shard):
    rm -rf analysis/step3/_embcache_modelspace/_preflight

Subset-extraction feasibility (informational, not implemented here): as
of this writing, scripts/extract_model_embeddings.py has no flag to
restrict extraction to a stratified/battery-referenced row subset --
`main()` always extracts every manifest row matching `--split` for a
requested corpus, regardless of run_reliance_battery.py's own
--max-rows-per-level=2000 cap. A stratified subset COULD cut the full
job's GPU cost several-fold, but implementing it (either a new CLI flag
reusing run_reliance_battery.py's own stratified sampling logic without
importing that module, or a separate script that pre-writes a reduced
manifest CSV) is a decision for the human, not made here.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_model_embeddings import (  # noqa: E402
    _build_model_from_checkpoint,
    check_torch_scipy_eigh,
    embedding_dim_of,
    extract_checkpoint_corpus,
)

from audioshield.data.manifest import ManifestRow, read_manifest  # noqa: E402
from audioshield.data.unified_dataset import UnifiedAudioDataset, collate_unified  # noqa: E402

# Real reference row counts, quoted from scripts/reproduce_eval.py's own
# docstring (C0's row-for-row comparison at full reference populations) --
# not guessed. Used only to project full-job wall-clock from a measured
# clips/sec; never used to decide anything by themselves.
CONFIRMED_ROW_COUNTS = dict(diffssd=94226, replaydf=52320)
CONFIRMED_CHECKPOINTS = ("e007_A_fresh", "e007_B_fresh", "e007_C_xlsr_fresh")
DEFAULT_PREFLIGHT_N_CLIPS = 500


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _log(msg: str) -> None:
    print(f"[{_timestamp()}] {msg}", flush=True)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


# ---------------------------------------------------------------------------
# Individual checks -- each pure/unit-testable where the brief asks for it.
# ---------------------------------------------------------------------------


def check_schema_and_dim(shard_path: Path, expected_dim: int) -> CheckResult:
    """Shard has the expected keys, emb is 2-D (n, expected_dim) -- NOT the
    3-D (n, n_layers, D) shape the existing XLS-R-300M cache uses (see
    docs/phaseB_extraction_preflight_findings.md §3a for why that
    distinction matters), and meta parses with the fields
    extract_checkpoint_corpus writes."""
    name = f"schema_and_dim[{shard_path.name}]"
    try:
        with np.load(shard_path, allow_pickle=True) as npz:
            missing = {"paths", "emb", "meta"} - set(npz.files)
            if missing:
                return CheckResult(name, False, f"missing keys {sorted(missing)}, found {sorted(npz.files)}")
            emb = npz["emb"]
            if emb.ndim != 2:
                return CheckResult(name, False, f"emb is {emb.ndim}-D, expected 2-D (n, dim)")
            if emb.shape[1] != expected_dim:
                return CheckResult(name, False, f"emb.shape[1]={emb.shape[1]}, expected {expected_dim}")
            meta = json.loads(str(npz["meta"]))
            required_meta_keys = {"checkpoint_sha256", "model_config_hash", "git_sha", "dtype",
                                   "checkpoint_path", "corpus", "corpus_dir", "n_rows"}
            missing_meta = required_meta_keys - set(meta)
            if missing_meta:
                return CheckResult(name, False, f"meta missing keys {sorted(missing_meta)}")
    except Exception as e:
        return CheckResult(name, False, f"{type(e).__name__}: {e}")
    return CheckResult(name, True, f"n_rows={emb.shape[0]} embedding_dim={emb.shape[1]} dtype={emb.dtype}")


def recompute_logit_from_embedding(fc: torch.nn.Linear, emb: torch.Tensor) -> torch.Tensor:
    """The exact algebra BinaryHead.forward performs (src/audioshield/models/
    heads.py:30-31), applied directly to `emb` -- if `emb` really is the
    classifier's input `z`, this reproduces the model's own logit."""
    return fc(emb).squeeze(-1)


def check_correctness(fc: torch.nn.Linear, emb: torch.Tensor, expected_logit: torch.Tensor,
                       atol: float = 1e-3, rtol: float = 1e-2) -> CheckResult:
    """The decisive check: does binary.fc(emb) == the model's own logit?
    Tolerances are loose enough to absorb float16 on-disk quantization
    (the default --dtype) without being loose enough to pass a genuinely
    wrong embedding -- a wrong hook point (e.g. pre-GELU, pre-final-
    LayerNorm) produces a systematically large, not marginal, discrepancy."""
    recomputed = recompute_logit_from_embedding(fc, emb.to(fc.weight.dtype))
    expected = expected_logit.to(recomputed.dtype)
    close = torch.allclose(recomputed, expected, atol=atol, rtol=rtol)
    max_abs_diff = float((recomputed - expected).abs().max()) if recomputed.numel() else float("nan")
    return CheckResult(
        "correctness_binary_fc_matches_logit", bool(close),
        f"max|binary.fc(emb) - logit|={max_abs_diff:.6g} (atol={atol}, rtol={rtol})",
    )


def check_sanity(emb: np.ndarray) -> CheckResult:
    finite = bool(np.isfinite(emb).all())
    norms = np.linalg.norm(emb.astype(np.float64), axis=1)
    reasonable = bool(finite and np.all(norms > 1e-6) and np.all(norms < 1e6))
    return CheckResult(
        "sanity_finite_and_reasonable_norms", reasonable,
        f"all_finite={finite} norm_min={norms.min():.4g} norm_max={norms.max():.4g} norm_mean={norms.mean():.4g}"
        if finite else "non-finite values present",
    )


def check_determinism(emb_a: np.ndarray, emb_b: np.ndarray, rtol: float = 1e-3, atol: float = 1e-5) -> CheckResult:
    """Same clips extracted twice must produce byte-identical-in-spirit
    embeddings (eval mode: no dropout, no random cropping -- see
    extract_checkpoint_corpus's UnifiedAudioDataset(..., random_crop=False)).
    Reliance metrics fit a subspace on these vectors; non-determinism would
    inject noise no seed could account for."""
    if emb_a.shape != emb_b.shape:
        return CheckResult("determinism_two_extractions_match", False,
                            f"shape mismatch {emb_a.shape} vs {emb_b.shape}")
    close = np.allclose(emb_a, emb_b, rtol=rtol, atol=atol)
    max_diff = float(np.abs(emb_a.astype(np.float64) - emb_b.astype(np.float64)).max())
    return CheckResult("determinism_two_extractions_match", bool(close), f"max_abs_diff={max_diff:.6g}")


def project_full_extraction_cost(clips_per_sec: float) -> dict:
    """Projects wall-clock for exactly the 6 (checkpoint, corpus) jobs the
    brief scopes -- e007_A/B/C x DiffSSD/ReplayDF -- nothing else. Row
    counts are CONFIRMED_ROW_COUNTS (quoted from reproduce_eval.py's own
    docstring), not estimates."""
    total_clips = len(CONFIRMED_CHECKPOINTS) * sum(CONFIRMED_ROW_COUNTS.values())
    seconds = total_clips / clips_per_sec if clips_per_sec > 0 else float("inf")
    return dict(
        n_jobs=len(CONFIRMED_CHECKPOINTS) * len(CONFIRMED_ROW_COUNTS),
        checkpoints=list(CONFIRMED_CHECKPOINTS), corpora=list(CONFIRMED_ROW_COUNTS),
        row_counts=dict(CONFIRMED_ROW_COUNTS), total_clips=total_clips,
        measured_clips_per_sec=clips_per_sec, projected_seconds=seconds,
        projected_hours=seconds / 3600.0,
    )


# ---------------------------------------------------------------------------
# Orchestration: run a tiny extraction + every check above.
# ---------------------------------------------------------------------------


def run_preflight_validation(
    checkpoint: Path, corpus: str, manifest_dir: Path, data_root: Path, out_root: Path,
    device: str, n_clips: int = DEFAULT_PREFLIGHT_N_CLIPS, batch_size: int = 16,
    dtype: str = "float32", splits: list[str] | None = None,
    build_model_fn=_build_model_from_checkpoint, run_eigh_check: bool = True,
) -> dict:
    """Runs the full preflight validation for ONE (checkpoint, corpus)
    pair on the first `n_clips` manifest rows, writing shards under
    out_root (the caller is responsible for pointing this at a
    `_preflight` subdirectory, never the real cache namespace -- see
    build_parser's --out-root default). Never raises: every check result
    is collected and returned even if an earlier one failed, so a single
    report always shows everything at once (matches extract_model_
    embeddings.py's own run_preflight convention).

    `build_model_fn` is dependency-injected (default: the real
    `_build_model_from_checkpoint`, which loads an actual AudioShieldX +
    HF backbone) -- mirrors extract_model_embeddings.py's own
    `run_preflight(..., build_model_fn=...)` convention, and is what lets
    this whole orchestration be unit-tested end-to-end with a fake model
    exposing exactly `.embed()`/`.binary.fc`/`__call__`, never a real
    backbone (out of scope for this environment, per
    tests/test_extract_model_embeddings.py's own precedent). `run_eigh_check`
    is skippable in tests for the same reason `check_torch_scipy_eigh`
    itself needs no fake -- it's a real subprocess check, cheap and
    side-effect-free, but pointless noise in a synthetic-only unit test."""
    results: list[CheckResult] = []

    if run_eigh_check:
        _log("preflight: checking torch+scipy.linalg.eigh in a subprocess (gflow segfault guard)")
        ok, detail = check_torch_scipy_eigh()
        results.append(CheckResult("torch_scipy_eigh_subprocess_safe", ok, detail))

    _log(f"preflight: building model from {checkpoint}")
    model, cfg, _sd = build_model_fn(checkpoint, device)
    expected_dim = embedding_dim_of(model)
    _log(f"preflight: model expects {expected_dim}-d embeddings (binary.fc.in_features)")

    _log(f"preflight: reading manifest for corpus={corpus!r}")
    rows = read_manifest(manifest_dir / f"{corpus}.csv", splits=splits)
    rows = sorted(rows, key=lambda r: r.utt_id)[:n_clips]
    if not rows:
        results.append(CheckResult("manifest_has_rows", False, "0 rows -- nothing to preflight"))
        return dict(results=[r.__dict__ for r in results], throughput=None, projection=None)
    _log(f"preflight: using {len(rows)} row(s) (capped at n_clips={n_clips})")

    ckpt_git_sha = "preflight"
    extraction_dir = out_root / checkpoint.stem
    exp = cfg.get("experiment", {})

    _log("preflight: running extraction (timed for throughput)")
    t0 = time.time()
    stats = extract_checkpoint_corpus(model, cfg, checkpoint, corpus, rows, data_root, extraction_dir,
                                       device, batch_size, shard_size=max(len(rows), 1), dtype=dtype,
                                       git_sha=ckpt_git_sha)
    elapsed = time.time() - t0
    clips_per_sec = len(rows) / elapsed if elapsed > 0 else float("inf")
    _log(f"preflight: extraction took {elapsed:.2f}s for {len(rows)} clips ({clips_per_sec:.2f} clips/sec)")

    shard_paths = sorted(extraction_dir.glob("shard_*.npz"))
    for shard_path in shard_paths:
        results.append(check_schema_and_dim(shard_path, expected_dim))

    # Correctness: fresh forward pass (independent of the extraction call
    # above) on the SAME rows, through the SAME dataset/loader pipeline
    # extraction itself uses (so preprocessing matches), then compare
    # against the SAVED shard's on-disk embedding.
    _log("preflight: running a fresh forward pass to check binary.fc(emb) == logit")
    ds = UnifiedAudioDataset(rows, data_root, sample_rate=exp.get("sample_rate", 16000),
                              duration_seconds=exp.get("duration_seconds", 4.0), random_crop=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                                       collate_fn=collate_unified)
    fresh_logits: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in dl:
            out = model(batch["waveform"].to(device))
            fresh_logits.append(out["spoof_logit"].cpu())
    fresh_logit_tensor = torch.cat(fresh_logits, dim=0)

    saved_emb = np.concatenate([np.load(p, allow_pickle=True)["emb"] for p in shard_paths], axis=0)
    saved_emb_tensor = torch.from_numpy(saved_emb.astype(np.float32))
    results.append(check_correctness(model.binary.fc, saved_emb_tensor, fresh_logit_tensor))
    results.append(check_sanity(saved_emb))

    # Determinism: extract the SAME rows again into a throwaway sibling
    # directory (never the shard we're validating, so resume-skip below
    # isn't accidentally exercised here instead).
    _log("preflight: re-extracting into a throwaway dir to check determinism")
    determinism_dir = out_root / f"{checkpoint.stem}_determinism_check"
    shutil.rmtree(determinism_dir, ignore_errors=True)
    extract_checkpoint_corpus(model, cfg, checkpoint, corpus, rows, data_root, determinism_dir,
                               device, batch_size, shard_size=max(len(rows), 1), dtype=dtype,
                               git_sha=ckpt_git_sha)
    second_shard_paths = sorted(determinism_dir.glob("shard_*.npz"))
    second_emb = np.concatenate([np.load(p, allow_pickle=True)["emb"] for p in second_shard_paths], axis=0)
    results.append(check_determinism(saved_emb, second_emb))
    shutil.rmtree(determinism_dir, ignore_errors=True)

    # Resume-safety: re-run extraction into the SAME dir as the first run
    # -- must skip every shard (extract_checkpoint_corpus's own, already
    # unit-tested resume logic; this just confirms it operationally here).
    _log("preflight: re-running extraction into the SAME dir to confirm resume-skip")
    resume_stats = extract_checkpoint_corpus(model, cfg, checkpoint, corpus, rows, data_root, extraction_dir,
                                               device, batch_size, shard_size=max(len(rows), 1), dtype=dtype,
                                               git_sha=ckpt_git_sha)
    resumed_ok = resume_stats["written"] == 0 and resume_stats["skipped"] == stats["n_shards"]
    results.append(CheckResult("resume_skips_completed_shard", resumed_ok,
                                f"first run: written={stats['written']} skipped={stats['skipped']}; "
                                f"second run: written={resume_stats['written']} skipped={resume_stats['skipped']}"))

    projection = project_full_extraction_cost(clips_per_sec)

    return dict(
        results=[r.__dict__ for r in results],
        throughput=dict(n_clips=len(rows), elapsed_seconds=elapsed, clips_per_sec=clips_per_sec),
        projection=projection,
        extraction_dir=str(extraction_dir),
    )


def print_report(report: dict) -> None:
    results = report["results"]
    name_w = max((len(r["name"]) for r in results), default=4)
    print(f"\n{'CHECK':<{name_w}}  STATUS  DETAIL")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['name']:<{name_w}}  {status:<6}  {r['detail']}")
    n_fail = sum(1 for r in results if not r["passed"])
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    if report["throughput"]:
        t = report["throughput"]
        print(f"\nthroughput: {t['n_clips']} clips in {t['elapsed_seconds']:.2f}s = {t['clips_per_sec']:.2f} clips/sec")
    if report["projection"]:
        p = report["projection"]
        print(f"\nprojected full extraction (6 jobs: {p['checkpoints']} x {list(p['row_counts'])}):")
        print(f"  total_clips={p['total_clips']} at {p['measured_clips_per_sec']:.2f} clips/sec "
              f"-> {p['projected_hours']:.2f} hours ({p['projected_seconds']:.0f}s)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--corpus", required=True, choices=sorted(CONFIRMED_ROW_COUNTS))
    ap.add_argument("--manifest-dir", default="manifests/v2")
    ap.add_argument("--data-root", default="..")
    ap.add_argument("--out-root", default="analysis/step3/_embcache_modelspace/_preflight",
                     help="a _preflight subdirectory by default -- never the real cache namespace, "
                          "so a full run later never mistakes this for a completed real shard")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--n-clips", type=int, default=DEFAULT_PREFLIGHT_N_CLIPS)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--split", nargs="*", default=None)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = run_preflight_validation(
        checkpoint=args.checkpoint, corpus=args.corpus, manifest_dir=Path(args.manifest_dir),
        data_root=Path(args.data_root), out_root=Path(args.out_root), device=args.device,
        n_clips=args.n_clips, batch_size=args.batch_size, dtype=args.dtype, splits=args.split,
    )
    print_report(report)
    n_fail = sum(1 for r in report["results"] if not r["passed"])
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
