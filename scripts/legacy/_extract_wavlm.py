"""WavLM-Large cache-space embedding extractor (gate criterion C1's input).

A deliberately minimal variation of scripts/legacy/_extract_xlsr.py: same shard
schema (shard_%04d.npz with keys {paths, emb, dur}), same naming, same float16
dtype, same 25 hidden-states x 1024 layout, same resume convention
(_done.txt/_skipped.txt), and the same OOM-halving forward. What makes the WavLM
cache joinable with the XLS-R cache for C1 is that both store the SAME stored-path
strings for the same file; this script guarantees that by deriving those strings
from the manifest exactly the way run_reliance_battery.py's own cache/manifest
join expects (see below).

FILE LISTING IS MANIFEST-DRIVEN, not _INVENTORY.tsv-driven (unlike the XLS-R
legacy script). Two GPU-machine facts force this (2026-07 preflight):
  - 04_ReplayDF/_INVENTORY.tsv has ~163,891 rows but manifests/v2/replaydf.csv
    needs only ~52,320 -- the corpus dir is a ~3.1x superset of what the
    batteries join, so inventory-driven extraction would waste ~2/3 of the GPU.
  - 03_DiffSSD/_INVENTORY.tsv does not exist (the corpus was re-downloaded after
    the original extraction era; the inventory was never recreated).
So we read manifests/v2/<corpus>.csv, resolve each row's `path` against
--data-root, and extract every manifest row exactly once -- the same convention
scripts/extract_model_embeddings.py already uses for exactly this problem.

PATH-FORMAT CONTRACT (C1 depends on this): run_reliance_battery.py's join
(strip_cache_prefix / join_cache_to_manifest) expects manifest `path` =
"datasets/<CORPUS_DIR>/<rel>" and cache stored `paths` = "<rel>". We store exactly
"<rel>" (the manifest path with the "datasets/<CORPUS_DIR>/" prefix stripped),
derived by string-slicing the manifest path -- byte-for-byte what the join
strips off the manifest side, so the inner-join is exact. Two runtime self-checks
(below) guard this before any GPU hours are spent.

Two things differ from the XLS-R extractor at the model call, both on purpose:

1. The model is microsoft/wavlm-large, pinned to the revision in
   configs/backbone_revisions.yaml and loaded with local_files_only=True. The
   weights already exist on the GPU machine from e005/e006, so a download attempt
   must fail loudly rather than silently fetch something unpinned.

2. WavLM-Large is fed WITHOUT an attention_mask -- unlike the XLS-R extractor,
   which passes one. This mirrors how the detector that trained e005/e006 actually
   feeds WavLM: src/audioshield/models/ssl_backbone.py (LayerWeightedSSL.forward)
   and scripts/layer_probe.py both call the backbone as
   `model(waveform, output_hidden_states=True)` with no attention_mask. We match
   that feed so the cached embeddings match what the trained model saw. We still
   mask the per-layer time-mean to each clip's true feature length (computed from
   the raw sample counts via _get_feat_extract_output_lengths) so batch
   zero-padding never enters the mean -- reproducing layer_probe's single-clip
   full-length mean while allowing batching.
"""

import os, sys, csv, time, argparse, contextlib
import numpy as np, soundfile as sf, torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample_poly
from math import gcd

BASE = Path(".").resolve()
OUT = BASE / "_embcache_wavlm_large"
SR = 16000
MAX_SEC = float(os.environ.get("MAX_SEC", 12))
BATCH = int(os.environ.get("EXTRACT_BATCH", 12))
WORKERS = int(os.environ.get("EXTRACT_WORKERS", 2))
SHARD = 20000
DEFAULT_MANIFEST_DIR = "manifests/v2"
# ReplayDF FIRST, deliberately: if the GPU is reclaimed mid-run, the ReplayDF
# cache alone already feeds both model-space batteries, so front-loading it means
# a partial run is still useful. diffssd follows. (corpus NAMES here, matching
# manifests/v2/<name>.csv; the on-disk cache directory is the CORPUS_DIR derived
# from each manifest's own path column -- 04_ReplayDF, 03_DiffSSD.)
ORDER = ["replaydf", "diffssd"]

# microsoft/wavlm-large, configs/backbone_revisions.yaml (kept in sync with the
# pin; _pinned_revision() prefers the committed yaml at runtime and only falls
# back to this literal when run from a directory without the config).
PINNED_REVISION = "c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c"


def _pinned_revision():
    p = BASE / "configs" / "backbone_revisions.yaml"
    if p.exists():
        import yaml
        revs = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        rev = revs.get("microsoft/wavlm-large")
        if rev:
            return rev
    return PINNED_REVISION


# --- manifest listing helpers (mirror scripts/extract_model_embeddings.py) ----

def read_manifest_rows(csv_path, splits=None):
    """Read manifests/v2/<corpus>.csv rows as dicts. `splits`, if given, keeps
    only rows whose `split` is in the set."""
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            if splits is not None and rec.get("split") not in splits:
                continue
            rows.append(rec)
    return rows

def corpus_dir_from_paths(paths):
    """Derive the dataset-root subdir (e.g. "04_ReplayDF") from the manifest's
    OWN path column -- every manifest path is "datasets/<DIR>/..." -- same rule
    as extract_model_embeddings._corpus_dir_from_rows."""
    dirs = {Path(p).parts[1] for p in paths
            if len(Path(p).parts) > 1 and Path(p).parts[0] == "datasets"}
    if len(dirs) != 1:
        raise ValueError(f"expected exactly one dataset dir prefix, got {sorted(dirs)}")
    return next(iter(dirs))

def strip_dataset_prefix(path, corpus_dir):
    """"datasets/<CORPUS_DIR>/<rel>" -> "<rel>". The cache stored-path format
    run_reliance_battery.strip_cache_prefix reconstructs on the manifest side."""
    prefix = f"datasets/{corpus_dir}/"
    if not path.startswith(prefix):
        raise ValueError(f"{path!r} does not start with expected prefix {prefix!r}")
    return path[len(prefix):]

def resolve_audio_path(data_root, row_path):
    """`row_path` joined onto data_root unless already absolute -- same rule as
    extract_model_embeddings._resolve_audio_path / evaluation.cross_test."""
    p = Path(row_path)
    return p if p.is_absolute() else (Path(data_root) / p)


class AudioDS(Dataset):
    def __init__(self, items):
        # items: list of (stored_rel, abs_audio_path)
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        rel, ap = self.items[i]
        try:
            x, sr = sf.read(str(ap), dtype="float32", always_2d=True)
            x = x.mean(axis=1)
            if sr != SR:
                g = gcd(sr, SR); x = resample_poly(x, SR // g, sr // g).astype(np.float32)
            n = len(x); dur = n / SR
            cap = int(MAX_SEC * SR)
            if n > cap:
                s = (n - cap) // 2; x = x[s:s + cap]
            if len(x) < 800: x = np.pad(x, (0, 800 - len(x)))
            return rel, x, dur
        except Exception as e:
            return rel, None, str(e)

def collate(items): return items

def forward_chunked(model, dev, batch):
    # autocast only on CUDA; nullcontext keeps the same code path CPU-runnable
    # (fp16 matmul is unsupported on CPU) for the offline smoke test.
    amp = torch.autocast("cuda", dtype=torch.float16) if str(dev) == "cuda" else contextlib.nullcontext()
    try:
        with torch.inference_mode(), amp:
            L = max(len(x) for _, x, _ in batch)
            wav = torch.zeros(len(batch), L)
            lengths = torch.zeros(len(batch))
            for i, (_, x, _) in enumerate(batch):
                wav[i, :len(x)] = torch.from_numpy(x); lengths[i] = len(x)
            # NO attention_mask -- mirror the e005/e006 detector's WavLM feed
            # (ssl_backbone.py / layer_probe.py). See module docstring.
            out = model(wav.to(dev), output_hidden_states=True)
            hs = torch.stack(out.hidden_states, 1)          # B,25,T,1024
            fl = model._get_feat_extract_output_lengths(lengths.to(dev)).long().clamp(min=1)
            emb = torch.stack([hs[i, :, :fl[i]].mean(1) for i in range(len(batch))])
            return emb.half().cpu().numpy()                  # B,25,1024
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(batch) == 1: raise
        h = len(batch) // 2
        return np.concatenate([forward_chunked(model, dev, batch[:h]),
                               forward_chunked(model, dev, batch[h:])])

def make_loader(ds, workers):
    return DataLoader(ds, batch_size=BATCH, num_workers=workers, collate_fn=collate,
                      pin_memory=True, persistent_workers=False,
                      prefetch_factor=(2 if workers > 0 else None))

def build_items(rows, corpus_dir, data_root):
    """Manifest rows -> deduped, deterministically-ordered list of
    (stored_rel, resolved_abs_path), extracting every manifest row exactly once."""
    seen, items = set(), []
    for r in sorted(rows, key=lambda r: r.get("utt_id", r["path"])):
        rel = strip_dataset_prefix(r["path"], corpus_dir)
        if rel in seen:
            continue
        seen.add(rel)
        items.append((rel, resolve_audio_path(data_root, r["path"])))
    return items

def process(model, dev, corpus, rows, data_root, out_root, stats):
    corpus_dir = corpus_dir_from_paths([r["path"] for r in rows])
    items = build_items(rows, corpus_dir, data_root)
    cdir = Path(out_root) / corpus_dir; cdir.mkdir(parents=True, exist_ok=True)
    done_p, skip_p = cdir / "_done.txt", cdir / "_skipped.txt"
    done = set(done_p.read_text().splitlines()) if done_p.exists() else set()

    # --- self-check 1: first 100 resolved audio paths must exist on disk ---
    sample = items[:100]
    missing = [str(ap) for _, ap in sample if not Path(ap).exists()]
    if missing:
        raise RuntimeError(
            f"[{corpus}] ABORT before extraction: {len(missing)}/{len(sample)} of the first "
            f"{len(sample)} resolved audio paths do not exist under --data-root={data_root}. "
            f"First missing: {missing[:3]}. Fix --data-root before committing GPU hours."
        )
    # --- self-check 2: stored paths must match the real XLS-R shard format
    #     EXACTLY -- corpus-dir-relative, forward slashes, no "datasets/" prefix,
    #     not absolute (empirically confirmed on the GPU machine, e.g.
    #     'wav/0129dbe27753/benign/de/02f79cc84bcf.wav'). A single mis-formatted
    #     stored path silently breaks run_reliance_battery's inner-join, so we
    #     abort loudly before any GPU hours if the format ever drifts.
    for rel, _ in items:
        if "\\" in rel or rel.startswith("datasets/") or rel.startswith("/") or Path(rel).is_absolute():
            raise RuntimeError(
                f"[{corpus}] ABORT: stored path {rel!r} is not in the required cache format "
                f"(corpus-dir-relative, forward slashes, no 'datasets/' prefix, not absolute) -- "
                f"this would break run_reliance_battery.py's cache/manifest join for C1."
            )
    # --- log 3 sample stored-path strings for the operator to eyeball against
    #     an existing _embcache_xlsr300m/<corpus_dir> shard ---
    print(f"[{corpus}] corpus_dir={corpus_dir} n_rows={len(items)} -> {cdir}", flush=True)
    print(f"[{corpus}] sample stored paths (eyeball vs _embcache_xlsr300m/{corpus_dir}):", flush=True)
    for rel, ap in items[:3]:
        print(f"    stored={rel!r}   <- resolved {ap}", flush=True)

    todo = [(rel, ap) for rel, ap in items if rel not in done]
    print(f"[{corpus}] audio={len(items)} done={len(done)} todo={len(todo)}", flush=True)
    if not todo: print(f"[{corpus}] EMBEDDINGS COMPLETE"); return
    ds = AudioDS(todo)
    try:
        loader = make_loader(ds, WORKERS); it = iter(loader); first = next(it)
    except Exception as e:
        print(f"  workers={WORKERS} failed ({type(e).__name__}: {e}) -> falling back to num_workers=0", flush=True)
        loader = make_loader(ds, 0); it = iter(loader); first = next(it)
    shard_i = len(list(cdir.glob("shard_*.npz")))
    P, E, D = [], [], []
    def flush():
        nonlocal shard_i, P, E, D
        if not P: return
        np.savez(cdir / f"shard_{shard_i:04d}.npz", paths=np.array(P),
                 emb=np.stack(E), dur=np.array(D, dtype=np.float32))
        with open(done_p, "a", encoding="utf-8") as f: f.write("\n".join(P) + "\n")
        shard_i += 1; P, E, D = [], [], []
    batch = first
    while True:
        good = [(r, x, d) for r, x, d in batch if x is not None]
        bad = [(r, d) for r, x, d in batch if x is None]
        if bad:
            with open(skip_p, "a", encoding="utf-8") as f:
                f.writelines(f"{r}\t{d}\n" for r, d in bad)
            with open(done_p, "a", encoding="utf-8") as f:
                f.writelines(r + "\n" for r, _ in bad)
        if good:
            emb = forward_chunked(model, dev, good)
            for j, (r, _, d) in enumerate(good):
                P.append(r); E.append(emb[j]); D.append(d)
            stats["n"] += len(good); stats["sec"] += sum(d for _, _, d in good)
        if len(P) >= SHARD: flush()
        if stats["n"] >= 2000 and not stats["cal"]:
            stats["cal"] = True
            xrt = stats["sec"] / max(time.time() - stats["t0"], 1)
            print(f"  CALIBRATION: {xrt:.0f}x realtime, avg {stats['sec']/stats['n']:.1f}s/file "
                  f"-> this corpus ETA ~{(len(todo)-stats['n'])*(stats['sec']/stats['n'])/xrt/3600:.1f} h", flush=True)
        if stats["n"] % 5000 < BATCH:
            el = time.time() - stats["t0"]
            print(f"  ... {stats['n']} files, {stats['sec']/3600:.1f} audio-h, "
                  f"{stats['sec']/max(el,1):.0f}x RT, elapsed {el/3600:.2f} h", flush=True)
        try: batch = next(it)
        except StopIteration: break
    flush()
    print(f"[{corpus}] EMBEDDINGS COMPLETE (skipped: "
          f"{len(skip_p.read_text().splitlines()) if skip_p.exists() else 0})", flush=True)

def load_model(dev):
    """Load microsoft/wavlm-large at the pinned revision, offline only.

    local_files_only=True: the weights already exist on the GPU machine from
    e005/e006; a missing local copy must raise, never trigger a silent unpinned
    download. Dependency-injected in the smoke test with a tiny random WavLM.
    """
    from transformers import WavLMModel
    revision = _pinned_revision()
    print(f"loading microsoft/wavlm-large @ {revision} (local_files_only)...", flush=True)
    return WavLMModel.from_pretrained(
        "microsoft/wavlm-large", revision=revision,
        local_files_only=True, torch_dtype=torch.float16,
    ).to(dev).eval()

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", default=ORDER,
                     help="corpus names matching <manifest-dir>/<corpus>.csv; "
                          "default is ReplayDF then DiffSSD (reclaim-anytime order)")
    ap.add_argument("--manifest-dir", default=DEFAULT_MANIFEST_DIR)
    ap.add_argument("--data-root", default="..",
                     help="dataset root containing datasets/<CORPUS_DIR>/... "
                          "(same semantics as extract_model_embeddings.py --data-root)")
    ap.add_argument("--out-root", default=str(OUT),
                     help="e.g. E:/AI_voice_detection/datasets/_embcache_wavlm_large")
    ap.add_argument("--split", nargs="*", default=None,
                     help="restrict to these manifest splits (default: all)")
    args = ap.parse_args(argv)

    assert torch.cuda.is_available(), "CUDA not available"
    free, _ = torch.cuda.mem_get_info()
    assert free > 3.5e9, f"only {free/1e9:.1f} GB VRAM free — close the other GPU job first"
    dev = "cuda"
    model = load_model(dev)
    torch.set_num_threads(4)

    manifest_dir = Path(args.manifest_dir)
    splits = set(args.split) if args.split else None
    targets = ORDER if args.corpus == ["all"] else args.corpus
    for corpus in targets:
        mp = manifest_dir / f"{corpus}.csv"
        if not mp.exists():
            print(f"[{corpus}] no manifest {mp} — skip"); continue
        rows = read_manifest_rows(mp, splits=splits)
        if not rows:
            print(f"[{corpus}] 0 rows (manifest={mp}, split={args.split}) — skip"); continue
        stats = {"n": 0, "sec": 0.0, "t0": time.time(), "cal": False}
        process(model, dev, corpus, rows, args.data_root, args.out_root, stats)
    print("ALL REQUESTED CORPORA DONE")

if __name__ == "__main__":
    main()
