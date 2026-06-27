#!/usr/bin/env python3
"""leakage_audit.py - Train/Test instance-level leakage audit for AudioShield.

Certifies that held-out OOD corpora (In-the-Wild, ReplayDF, AI4T) share no
INSTANCE-LEVEL origin with the training corpora (ASVspoof5, DiffSSD, FoR, VCTK),
while explicitly NOT penalizing acceptable domain similarity (same language,
codec, TTS family). CPU-only.

Checks:
  1. Metadata/path intersection  (video IDs, speaker IDs, utt IDs, file stems)
  2. Waveform-hash collision      (exact bytes + normalized-waveform SHA-256)
  3. Embedding nearest-neighbour  (frozen-WavLM cosine, dual calibrated threshold)

Usage (full audit on the machine that has the data):
  python scripts/leakage_audit.py --manifest-dir manifests \
      --train asvspoof5 diffssd fakeorreal vctk \
      --test inthewild replaydf ai4t --data-root .. \
      [--train-emb emb/train.npz --test-emb emb/test.npz] [--hash-max-files N]
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

try:
    import soundfile as sf
    from scipy.signal import resample_poly
    _AUDIO_OK = True
except Exception:
    _AUDIO_OK = False


def load_manifest(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------- CHECK 1: metadata / path intersection ----------
AI4T_VIDEO_RE = re.compile(r"ai4t/[^/]+/([A-Za-z0-9_\-]+?)_\d+\.wav$")

def extract_identifiers(rows):
    ids = defaultdict(set)
    for r in rows:
        p = (r.get("path") or "").replace("\\", "/")
        utt = r.get("utt_id") or ""
        m = AI4T_VIDEO_RE.search(p) or AI4T_VIDEO_RE.search(utt)
        if m:
            ids["video_id"].add(m.group(1))
        spk = r.get("speaker") or r.get("speaker_id")
        if spk:
            ids["speaker_id"].add(str(spk))
        stem = Path(p).stem if p else ""
        if stem:
            ids["file_stem"].add(stem)
        if utt:
            ids["utt_id"].add(utt)
    return ids

def check_metadata(train_ids, test_ids):
    out = {}
    for t in ("video_id", "speaker_id", "file_stem", "utt_id"):
        shared = sorted(train_ids.get(t, set()) & test_ids.get(t, set()))
        out[t] = {"n_shared": len(shared), "examples": shared[:20]}
    return out


# ---------- CHECK 2: waveform hashing ----------
def _norm_wave_hash(path: Path, target_sr=16000):
    if not _AUDIO_OK:
        return None
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr and len(wav) > 0:
            from math import gcd
            g = gcd(int(sr), target_sr)
            wav = resample_poly(wav, target_sr // g, sr // g)
        peak = np.max(np.abs(wav)) if wav.size else 0.0
        if peak > 0:
            wav = wav / peak
        q = np.round(wav * 32767.0).astype(np.int16)
        return hashlib.sha256(q.tobytes()).hexdigest()
    except Exception:
        return None

def _file_hash(path: Path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for ch in iter(lambda: f.read(1 << 20), b""):
                h.update(ch)
        return h.hexdigest()
    except Exception:
        return None

def check_hashes(train_rows, test_rows, data_root: Path, max_files=None):
    def resolve(r):
        return data_root / (r.get("path") or "").replace("\\", "/")
    def build(rows):
        exact, normd, seen = {}, {}, 0
        for r in rows:
            if max_files and seen >= max_files:
                break
            p = resolve(r)
            if not p.exists():
                continue
            seen += 1
            eh, nh = _file_hash(p), _norm_wave_hash(p)
            key = r.get("utt_id") or str(p)
            if eh:
                exact.setdefault(eh, []).append(key)
            if nh:
                normd.setdefault(nh, []).append(key)
        return exact, normd, seen
    tre, trn, n_tr = build(train_rows)
    tee, ten, n_te = build(test_rows)
    ex = [{"train": tre[h][:3], "test": tee[h][:3]} for h in tee if h in tre]
    nm = [{"train": trn[h][:3], "test": ten[h][:3]} for h in ten if h in trn]
    return {
        "audio_libs_available": _AUDIO_OK,
        "n_train_hashed": n_tr, "n_test_hashed": n_te,
        "exact_collisions": {"n": len(ex), "examples": ex[:20]},
        "normalized_waveform_collisions": {"n": len(nm), "examples": nm[:20]},
    }


# ---------- CHECK 3: embedding nearest-neighbour ----------
def _l2(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X / n

def calibrate(train_emb):
    rng = np.random.default_rng(13)
    Xn = _l2(train_emb.astype(np.float64))
    m = min(2000, len(Xn))
    s = Xn[rng.choice(len(Xn), m, replace=False)]
    dup = _l2(s + rng.normal(0, 0.01, s.shape))
    dup_floor = float(np.percentile(np.sum(s * dup, axis=1), 1))
    j = rng.permutation(m)
    pc = np.sum(s * s[j], axis=1); pc = pc[np.arange(m) != j]
    dom99 = float(np.percentile(pc, 99))
    thr = float(min(dup_floor, max(dom99 + 0.02, 0.90)))
    return {"duplicate_floor_p1": dup_floor, "domain_pair_p99": dom99, "leakage_threshold": thr}

def check_embeddings(train_emb, test_emb, test_keys, batch=512):
    cal = calibrate(train_emb); thr = cal["leakage_threshold"]
    Tr = _l2(train_emb.astype(np.float32)); Te = _l2(test_emb.astype(np.float32))
    sims = np.empty(len(Te), np.float32); idx = np.empty(len(Te), np.int64)
    for s in range(0, len(Te), batch):
        d = Te[s:s + batch] @ Tr.T
        idx[s:s + batch] = d.argmax(1); sims[s:s + batch] = d.max(1)
    flagged = [{"test_key": test_keys[i] if i < len(test_keys) else str(i),
                "nn_train_row": int(idx[i]), "cosine": float(sims[i])}
               for i in np.where(sims >= thr)[0]]
    hist, edges = np.histogram(sims, bins=20, range=(0.0, 1.0))
    return {"calibration": cal, "n_flagged": len(flagged),
            "flagged": sorted(flagged, key=lambda x: -x["cosine"])[:50],
            "nn_sim_histogram": {"counts": hist.tolist(), "bin_edges": edges.tolist()},
            "nn_sim_summary": {"mean": float(sims.mean()),
                               "p99": float(np.percentile(sims, 99)),
                               "max": float(sims.max())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--data-root", default="..")
    ap.add_argument("--train-emb"); ap.add_argument("--test-emb")
    ap.add_argument("--hash-max-files", type=int, default=None)
    ap.add_argument("--out", default="runs/leakage_audit_report.json")
    a = ap.parse_args()

    md, root = Path(a.manifest_dir), Path(a.data_root)
    train_rows = []
    for c in a.train:
        train_rows += load_manifest(md / f"{c}.csv")
    test_by = {c: load_manifest(md / f"{c}.csv") for c in a.test}

    rep = {"train_corpora": a.train, "test_corpora": a.test,
           "data_root": str(root), "checks": {"metadata": {}, "hashes": {}}}

    train_ids = extract_identifiers(train_rows)
    for c, rows in test_by.items():
        rep["checks"]["metadata"][c] = check_metadata(train_ids, extract_identifiers(rows))
        rep["checks"]["hashes"][c] = check_hashes(train_rows, rows, root, a.hash_max_files)

    if a.train_emb and a.test_emb:
        tr = np.load(a.train_emb)["emb"]; td = np.load(a.test_emb)
        rep["checks"]["embeddings"] = check_embeddings(
            tr, td["emb"], list(td["keys"]) if "keys" in td else [])
    else:
        rep["checks"]["embeddings"] = {"skipped": "no embeddings provided (Checks 1-2 only)"}

    print("\n================ LEAKAGE AUDIT SUMMARY ================")
    total_meta = sum(v[t]["n_shared"] for v in rep["checks"]["metadata"].values() for t in v)
    print(f"Check 1 (metadata)   : {total_meta} shared identifiers across all test corpora")
    for c, h in rep["checks"]["hashes"].items():
        warn = "" if h["audio_libs_available"] else "  [audio libs missing - install soundfile,scipy]"
        print(f"Check 2 (hashes) {c:11}: exact={h['exact_collisions']['n']} "
              f"norm-wave={h['normalized_waveform_collisions']['n']} "
              f"(train_hashed={h['n_train_hashed']}, test_hashed={h['n_test_hashed']}){warn}")
    emb = rep["checks"]["embeddings"]
    if "n_flagged" in emb:
        print(f"Check 3 (embeddings) : {emb['n_flagged']} flagged "
              f"(thr={emb['calibration']['leakage_threshold']:.4f})")
    else:
        print("Check 3 (embeddings) : skipped")
    print("======================================================\n")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"Full JSON report -> {a.out}")

if __name__ == "__main__":
    main()
