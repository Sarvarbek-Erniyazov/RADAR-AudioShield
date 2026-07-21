"""Roadmap v3 Step 3: extract AudioShieldX.embed() model-space embeddings --
the space where the classifier weight w actually lives -- so the full
reliance battery (scripts/run_reliance_battery.py), including alignment/
r_var/task-direction metrics, can run without the dimension mismatch that
script's --w-metrics guards against. The cached XLS-R-300M embeddings are a
single raw backbone hidden-state layer, pre-pooling (1024-d); w is the
classifier weight over embed()'s pooled+projected output (256-d for the
e007 configs, but read from the model here, never hardcoded).

Two modes:

--preflight (must be cheap): PASS/FAIL table for everything a full run
depends on -- the chosen interpreter can import torch and run
scipy.linalg.eigh on a 1024x1024 matrix without crashing (run in a
SEPARATE subprocess: the gflow conda env is known to segfault on this
call, a native crash no in-process try/except could ever catch); each
checkpoint loads and constructs AudioShieldX from its own saved config;
raw audio exists for the requested corpora (resolved through the same
data_root/manifest-path join UnifiedAudioDataset uses); a forward pass on
4 real utterances produces embeddings of the model's own expected
dimensionality; enough free disk space for the output shards. Exits
non-zero if anything fails, after printing the full table.

(default) full extraction: streams manifest rows for the requested
corpora through model.embed() under torch.no_grad(), batched, sharded to
.npz with the SAME (paths, emb) key schema as the existing XLS-R-300M
cache, plus a "meta" entry (JSON-encoded: checkpoint sha256, model config
hash, git sha, dtype). One output directory per checkpoint
(<out-root>/<run-name>/<corpus-dir>/shard_*.npz), since embed() is
checkpoint-specific -- <run-name> is --run-name's entry for this
checkpoint (an explicit, deliberately-chosen identifier), or that
checkpoint path's own .stem if --run-name wasn't given. THE STEM FALLBACK
COLLIDES for checkpoints sharing a filename (e.g. runs/e007_A/best.pt and
runs/e007_B/best.pt both stem to "best" -- pass --run-name explicitly
whenever checkpoints share a filename, as in the usage example below).
Even under a collision, extract_checkpoint_corpus's checkpoint_sha256
guard refuses to treat one checkpoint's shard as another's completed
resume -- it raises rather than silently mixing embeddings from two
different checkpoints under one directory. Shard writes are atomic (temp
file + rename), so an existing shard_*.npz is always complete and is
never recomputed on rerun (resume, once its checkpoint_sha256 is
confirmed to match). Asserts non-zero rows per corpus before writing
anything.

Usage:
    python scripts/extract_model_embeddings.py --preflight \
        --checkpoint runs/e007_A/best.pt runs/e007_B/best.pt \
        --corpus diffssd replaydf vctk --data-root ..

    python scripts/extract_model_embeddings.py \
        --checkpoint runs/e007_A/best.pt runs/e007_B/best.pt \
        --run-name e007_A e007_B \
        --corpus diffssd replaydf vctk \
        --data-root .. --out-root analysis/step3/_embcache_modelspace
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader

from audioshield.data.manifest import ManifestRow, read_manifest
from audioshield.data.unified_dataset import UnifiedAudioDataset, collate_unified
from audioshield.models.detector import AudioShieldX
from audioshield.utils.runtime import describe_device

DEFAULT_MANIFEST_DIR = "manifests/v2"
DEFAULT_DURATION_SECONDS = 4.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_SHARD_SIZE = 2000
DEFAULT_SAMPLE_CHECK_N = 20  # raw-audio-exists preflight sample size, kept cheap
FORWARD_PASS_N = 4


# ---------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------


def _resolve_audio_path(data_root: Path, row_path: str) -> Path:
    """Same join rule as UnifiedAudioDataset._resolve /
    evaluation.cross_test._resolved_audio_path: `row_path` is joined onto
    data_root unless already absolute."""
    p = Path(row_path)
    return p if p.is_absolute() else (data_root / p)


def _corpus_dir_from_rows(rows: list[ManifestRow]) -> str:
    """Derive the dataset-root subdirectory name (e.g. "03_DiffSSD") from the
    manifest's OWN path column rather than a hardcoded corpus->dir table --
    every manifest's path is "datasets/<DIR>/..." (verified against every
    corpus in manifests/v2/*.csv), so this generalizes to any corpus without
    needing to know its directory name in advance."""
    dirs = {
        Path(r.path).parts[1] for r in rows
        if len(Path(r.path).parts) > 1 and Path(r.path).parts[0] == "datasets"
    }
    if len(dirs) != 1:
        raise ValueError(f"expected exactly one dataset dir prefix across {len(rows)} rows, got {sorted(dirs)}")
    return next(iter(dirs))


def _strip_dataset_prefix(path: str, corpus_dir: str) -> str:
    prefix = f"datasets/{corpus_dir}/"
    if not path.startswith(prefix):
        raise ValueError(f"{path!r} does not start with expected prefix {prefix!r}")
    return path[len(prefix):]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg.get("model", {}), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def embedding_dim_of(model) -> int:
    """The model's own expected embed() output dimensionality, read from the
    ACTUAL constructed module -- never hardcoded. BinaryHead.fc's
    in_features is exactly embed()'s output width
    (src/audioshield/models/heads.py: self.fc = nn.Linear(dim, 1),
    src/audioshield/models/detector.py: self.binary = BinaryHead(m["embedding_dim"]))."""
    return int(model.binary.fc.in_features)


def _build_model_from_checkpoint(ckpt_path: Path, device: str) -> tuple:
    """torch.load -> AudioShieldX(cfg) -> load_state_dict -> eval -- the
    established pattern (src/audioshield/evaluation/cross_test.py:313-323).
    Returns (model, cfg, raw_checkpoint_dict)."""
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(sd, dict) or "cfg" not in sd:
        raise RuntimeError(f"{ckpt_path}: checkpoint has no saved 'cfg' -- refusing to guess a model config")
    cfg = sd["cfg"]
    model = AudioShieldX(cfg).to(device).eval()
    model.load_state_dict(sd["model"])
    return model, cfg, sd


# ---------------------------------------------------------------------------
# preflight checks
# ---------------------------------------------------------------------------


def check_torch_scipy_eigh(python_exe: str | None = None, timeout: float = 60.0) -> tuple[bool, str]:
    """Runs a torch import + a 1024x1024 scipy.linalg.eigh call in a
    SEPARATE process. Deliberately subprocess-isolated, not a plain
    try/except: the gflow conda env is known to segfault (a native crash)
    on scipy.linalg.eigh, which no Python exception handler in THIS process
    could ever catch -- a segfault kills the whole process outright.
    Isolating it in a subprocess means a crash there is reported as a FAIL,
    not a crash of this preflight run."""
    python_exe = python_exe or sys.executable
    code = (
        "import torch\n"
        "import numpy as np\n"
        "from scipy.linalg import eigh\n"
        "rng = np.random.default_rng(0)\n"
        "M = rng.standard_normal((1024, 1024)).astype(np.float64)\n"
        "M = M + M.T\n"
        "w, v = eigh(M)\n"
        "print('EIGH_OK', w.shape[0])\n"
    )
    try:
        proc = subprocess.run([python_exe, "-c", code], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s (possible hang, not just a crash)"
    except OSError as e:
        return False, f"could not launch {python_exe!r}: {type(e).__name__}: {e}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-500:]
        return False, f"subprocess exited {proc.returncode} (torch/scipy import or eigh call failed or crashed): {tail}"
    if "EIGH_OK" not in proc.stdout:
        return False, f"subprocess exited 0 but did not report EIGH_OK -- unexpected output: {proc.stdout.strip()[-300:]}"
    return True, "torch import + scipy.linalg.eigh(1024x1024) succeeded in a subprocess"


def check_raw_audio_exists(rows: list[ManifestRow], data_root: Path, sample_n: int = DEFAULT_SAMPLE_CHECK_N) -> tuple[bool, str]:
    if not rows:
        return False, "no manifest rows to check"
    rng = np.random.default_rng(13)
    if len(rows) <= sample_n:
        sample = rows
    else:
        idx = rng.choice(len(rows), size=sample_n, replace=False)
        sample = [rows[i] for i in idx]
    missing = [r.utt_id for r in sample if not _resolve_audio_path(data_root, r.path).exists()]
    if missing:
        return False, f"{len(missing)}/{len(sample)} sampled files missing under {data_root} (e.g. {missing[:3]})"
    return True, f"{len(sample)}/{len(sample)} sampled files found under {data_root}"


def check_forward_pass(model, cfg: dict, rows: list[ManifestRow], data_root: Path, device: str, n: int = FORWARD_PASS_N) -> tuple[bool, str]:
    if len(rows) < n:
        return False, f"need >= {n} rows for a forward-pass check, got {len(rows)}"
    sample_rows = rows[:n]
    exp = cfg.get("experiment", {})
    try:
        ds = UnifiedAudioDataset(
            sample_rows, data_root,
            sample_rate=exp.get("sample_rate", 16000),
            duration_seconds=exp.get("duration_seconds", DEFAULT_DURATION_SECONDS),
            random_crop=False,
        )
        dl = DataLoader(ds, batch_size=n, shuffle=False, num_workers=0, collate_fn=collate_unified)
        batch = next(iter(dl))
        with torch.no_grad():
            emb = model.embed(batch["waveform"].to(device))
    except Exception as e:
        return False, f"forward pass failed ({type(e).__name__}: {e})"
    expected = embedding_dim_of(model)
    got = int(emb.shape[-1])
    if got != expected:
        return False, f"embed() produced {got}-d output, model expects {expected}-d (BinaryHead.fc.in_features)"
    return True, f"forward pass on {emb.shape[0]} utterances produced {got}-d embeddings (matches model)"


def check_disk_space(out_dir: Path, n_rows_total: int, embedding_dim: int, dtype_bytes: int = 4, safety_factor: float = 1.5) -> tuple[bool, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    estimated_bytes = n_rows_total * embedding_dim * dtype_bytes * safety_factor
    free_bytes = shutil.disk_usage(out_dir).free
    if free_bytes < estimated_bytes:
        return False, (f"need ~{estimated_bytes / 2**30:.2f}GB (with {safety_factor}x safety margin) under "
                        f"{out_dir}, only {free_bytes / 2**30:.2f}GB free")
    return True, f"{free_bytes / 2**30:.2f}GB free under {out_dir}, need ~{estimated_bytes / 2**30:.2f}GB"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def run_preflight(
    checkpoints: list[Path],
    corpora: list[str],
    manifest_dir: Path,
    data_root: Path,
    out_root: Path,
    device: str,
    splits: list[str] | None = None,
    embedding_dtype: str = "float32",
    python_exe: str | None = None,
    build_model_fn: Callable = _build_model_from_checkpoint,
) -> list[CheckResult]:
    """Runs every preflight check and returns the full list of results --
    every check that CAN run does, even if an earlier one failed (e.g. a
    missing checkpoint doesn't stop raw-audio-existence from being checked
    too), so a single table always shows everything wrong at once."""
    results: list[CheckResult] = []

    ok, detail = check_torch_scipy_eigh(python_exe)
    results.append(CheckResult("torch_scipy_eigh", ok, detail))

    corpus_rows: dict[str, list[ManifestRow]] = {}
    for corpus in corpora:
        try:
            rows = read_manifest(manifest_dir / f"{corpus}.csv", splits=splits)
            corpus_rows[corpus] = rows
            results.append(CheckResult(f"manifest_readable[{corpus}]", len(rows) > 0, f"{len(rows)} rows"))
        except Exception as e:
            corpus_rows[corpus] = []
            results.append(CheckResult(f"manifest_readable[{corpus}]", False, f"{type(e).__name__}: {e}"))

    models: dict[str, tuple] = {}
    dims: list[int] = []
    for ckpt_path in checkpoints:
        if not ckpt_path.exists():
            results.append(CheckResult(f"checkpoint_constructs[{ckpt_path.name}]", False, "file not found"))
            continue
        try:
            model, cfg, _sd = build_model_fn(ckpt_path, device)
        except Exception as e:
            results.append(CheckResult(f"checkpoint_constructs[{ckpt_path.name}]", False, f"{type(e).__name__}: {e}"))
            continue
        results.append(CheckResult(f"checkpoint_constructs[{ckpt_path.name}]", True, "AudioShieldX constructed OK"))
        models[ckpt_path.name] = (model, cfg)
        dims.append(embedding_dim_of(model))

    for corpus, rows in corpus_rows.items():
        if not rows:
            results.append(CheckResult(f"raw_audio_exists[{corpus}]", False, "no manifest rows to check against"))
            continue
        ok, detail = check_raw_audio_exists(rows, data_root)
        results.append(CheckResult(f"raw_audio_exists[{corpus}]", ok, detail))

    any_rows = next((rows for rows in corpus_rows.values() if rows), [])
    if models and any_rows:
        first_name, (first_model, first_cfg) = next(iter(models.items()))
        ok, detail = check_forward_pass(first_model, first_cfg, any_rows, data_root, device)
        results.append(CheckResult(f"forward_pass[{first_name}]", ok, detail))
    else:
        results.append(CheckResult("forward_pass", False, "no constructed model or no manifest rows to test with"))

    n_rows_total = sum(len(r) for r in corpus_rows.values()) * max(len(checkpoints), 1)
    dtype_bytes = 2 if embedding_dtype == "float16" else 4
    embedding_dim_for_estimate = dims[0] if dims else 256
    ok, detail = check_disk_space(out_root, n_rows_total, embedding_dim_for_estimate, dtype_bytes)
    results.append(CheckResult("disk_space", ok, detail))

    return results


def print_preflight_table(results: list[CheckResult]) -> None:
    name_w = max((len(r.name) for r in results), default=4)
    print(f"\n{'CHECK':<{name_w}}  STATUS  DETAIL")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"{r.name:<{name_w}}  {status:<6}  {r.detail}")
    n_fail = sum(1 for r in results if not r.passed)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")


# ---------------------------------------------------------------------------
# full extraction
# ---------------------------------------------------------------------------


def _write_shard_atomic(shard_path: Path, paths: np.ndarray, emb: np.ndarray, meta: dict) -> None:
    """Write to a .tmp file then atomically rename -- guarantees any
    shard_*.npz present at its final name is complete, never partial, which
    is what makes "shard exists -> skip" (extract_checkpoint_corpus) a safe
    resume check."""
    tmp_path = shard_path.with_name(shard_path.name + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez(f, paths=paths, emb=emb, meta=np.array(json.dumps(meta)))
    os.replace(tmp_path, shard_path)


def _read_shard_checkpoint_sha256(shard_path: Path) -> str | None:
    """Reads just the `checkpoint_sha256` field out of an existing shard's
    meta, for the collision guard below. Returns None if the shard can't
    be read as a valid shard at all (caller treats that as "can't verify,
    don't trust an existing directory here either")."""
    try:
        with np.load(shard_path, allow_pickle=True) as data:
            meta = json.loads(str(data["meta"]))
        return meta.get("checkpoint_sha256")
    except Exception:
        return None


def extract_checkpoint_corpus(
    model,
    cfg: dict,
    ckpt_path: Path,
    corpus: str,
    rows: list[ManifestRow],
    data_root: Path,
    out_dir: Path,
    device: str,
    batch_size: int,
    shard_size: int,
    dtype: str,
    git_sha: str,
) -> dict:
    """Stream `rows` through model.embed() in fixed-size shards, writing
    shard_{i:04d}.npz with the same (paths, emb) key schema as the existing
    XLS-R-300M cache, plus a "meta" entry. A shard file that already exists
    on disk is assumed complete (writes are atomic, see _write_shard_atomic)
    and is skipped without re-running the model on it (resume) -- UNLESS its
    recorded checkpoint_sha256 doesn't match the checkpoint being extracted
    now, in which case this is not a resume, it's a NAMING COLLISION (two
    different checkpoints mapped to the same out_dir, e.g. two nested
    runs/<run>/best.pt paths sharing the stem "best") -- fail loud rather
    than silently treat checkpoint B's rows as checkpoint A's completed
    shard."""
    if not rows:
        raise ValueError(f"{corpus}: 0 rows -- refusing to write an empty cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = _corpus_dir_from_rows(rows)

    ckpt_sha = _sha256_file(ckpt_path)
    cfg_hash = _model_config_hash(cfg)
    np_dtype = np.float16 if dtype == "float16" else np.float32
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    exp = cfg.get("experiment", {})

    n_shards = (len(rows) + shard_size - 1) // shard_size
    written, skipped = 0, 0
    for shard_i in range(n_shards):
        shard_path = out_dir / f"shard_{shard_i:04d}.npz"
        shard_rows = rows[shard_i * shard_size: (shard_i + 1) * shard_size]
        if shard_path.exists():
            existing_sha = _read_shard_checkpoint_sha256(shard_path)
            if existing_sha is not None and existing_sha != ckpt_sha:
                raise RuntimeError(
                    f"{shard_path}: NAMING COLLISION -- this shard was written for a checkpoint "
                    f"with sha256={existing_sha[:16]}..., but the checkpoint being extracted now "
                    f"({ckpt_path}) has sha256={ckpt_sha[:16]}... -- refusing to silently treat one "
                    "checkpoint's cached embeddings as another's. Pass a distinct --run-name (or "
                    "--out-root) for this checkpoint, or verify out_dir isn't shared by mistake."
                )
            skipped += 1
            print(f"[resume] {shard_path}: already complete (checkpoint sha256 matches), "
                  f"skipping ({len(shard_rows)} rows)")
            continue

        ds = UnifiedAudioDataset(
            shard_rows, data_root,
            sample_rate=exp.get("sample_rate", 16000),
            duration_seconds=exp.get("duration_seconds", DEFAULT_DURATION_SECONDS),
            random_crop=False,
        )
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_unified)

        all_paths: list[str] = []
        all_emb: list[np.ndarray] = []
        with torch.no_grad():
            for batch in dl:
                emb = model.embed(batch["waveform"].to(device))
                all_emb.append(emb.to(torch_dtype).cpu().numpy())
                all_paths.extend(_strip_dataset_prefix(p, corpus_dir) for p in batch["path"])

        emb_arr = np.concatenate(all_emb, axis=0).astype(np_dtype)
        paths_arr = np.array(all_paths)
        meta = dict(
            checkpoint_sha256=ckpt_sha, model_config_hash=cfg_hash, git_sha=git_sha,
            dtype=np.dtype(np_dtype).name, checkpoint_path=str(ckpt_path), corpus=corpus, corpus_dir=corpus_dir,
            n_rows=len(shard_rows),
        )
        _write_shard_atomic(shard_path, paths_arr, emb_arr, meta)
        written += 1
        print(f"[shard] {shard_path}: wrote {len(shard_rows)} rows")

    return dict(corpus=corpus, corpus_dir=corpus_dir, n_shards=n_shards, written=written, skipped=skipped,
                n_rows=len(rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", nargs="+", required=True, type=Path, help="one or more checkpoint .pt paths")
    ap.add_argument("--run-name", nargs="+", default=None,
                     help="explicit cache-directory identifier per --checkpoint, same order/count. "
                          "Defaults to each checkpoint path's own .stem, which COLLIDES for checkpoints "
                          "sharing a filename (e.g. runs/e007_A_fresh/best.pt and runs/e007_B_fresh/best.pt "
                          "both stem to \"best\") -- pass this explicitly (e.g. e007_A_fresh e007_B_fresh) "
                          "whenever checkpoints share a filename. A collision is still caught (never silently "
                          "corrupted) by the checkpoint_sha256 guard in extract_checkpoint_corpus regardless.")
    ap.add_argument("--corpus", nargs="+", required=True, help="corpus names matching <manifest-dir>/<corpus>.csv")
    ap.add_argument("--manifest-dir", default=DEFAULT_MANIFEST_DIR)
    ap.add_argument("--data-root", default="..", help="dataset root containing datasets/<CORPUS_DIR>/... "
                                                        "(same convention as evaluation.cross_test --data-root)")
    ap.add_argument("--out-root", default="analysis/step3/_embcache_modelspace")
    ap.add_argument("--split", nargs="*", default=None, help="restrict to these manifest splits (default: all)")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--python", default=sys.executable,
                     help="interpreter to check torch/scipy.linalg.eigh under in --preflight "
                          "(default: this script's own interpreter)")
    ap.add_argument("--preflight", action="store_true", help="run cheap PASS/FAIL checks and exit; no extraction")
    args = ap.parse_args(argv)

    manifest_dir = Path(args.manifest_dir)
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    describe_device(args.device)

    if args.preflight:
        results = run_preflight(
            args.checkpoint, args.corpus, manifest_dir, data_root, out_root, args.device,
            splits=args.split, embedding_dtype=args.dtype, python_exe=args.python,
        )
        print_preflight_table(results)
        sys.exit(0 if all(r.passed for r in results) else 1)

    if args.run_name is not None and len(args.run_name) != len(args.checkpoint):
        raise ValueError(f"--run-name given {len(args.run_name)} name(s) but --checkpoint has "
                          f"{len(args.checkpoint)} path(s) -- must match 1:1, never partially guessed")
    run_names = args.run_name if args.run_name is not None else [p.stem for p in args.checkpoint]

    git_sha = _git_sha()
    for ckpt_path, run_name in zip(args.checkpoint, run_names):
        print(f"[checkpoint] {ckpt_path} (run_name={run_name!r})")
        model, cfg, _sd = _build_model_from_checkpoint(ckpt_path, args.device)
        for corpus in args.corpus:
            rows = read_manifest(manifest_dir / f"{corpus}.csv", splits=args.split)
            if not rows:
                raise ValueError(f"{corpus}: 0 rows matched (manifest_dir={manifest_dir}, split={args.split}) "
                                  f"-- refusing to write an empty cache")
            rows = sorted(rows, key=lambda r: r.utt_id)
            out_dir = out_root / run_name / _corpus_dir_from_rows(rows)
            print(f"[extract] run_name={run_name} corpus={corpus} n_rows={len(rows)} -> {out_dir}")
            stats = extract_checkpoint_corpus(
                model, cfg, ckpt_path, corpus, rows, data_root, out_dir, args.device,
                args.batch_size, args.shard_size, args.dtype, git_sha,
            )
            print(f"[done] {stats}")


if __name__ == "__main__":
    main()
