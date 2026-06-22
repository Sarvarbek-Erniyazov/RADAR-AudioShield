"""Literature-style frozen-SSL linear baseline (Combei-style, transplanted).

Train: logistic regression on frozen WavLM-large layer-K features, ASVspoof5 only.
Eval:  held-out corpora with EER / AUC / BAC@dev-threshold + Kwok bona-fide matrix
       + bona-fide-source probe. No BMI, no augmentation, no multi-corpus mix.
"""
from __future__ import annotations
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel

from audioshield.data.audio_io import load_audio, to_mono, resample_linear, crop_or_pad
from audioshield.data.manifest import read_manifest
from audioshield.evaluation.metrics import equal_error_rate


def balanced_subsample(rows, n_per_class, seed=13):
    rng = random.Random(seed)
    pos = [r for r in rows if r.target == 1]
    neg = [r for r in rows if r.target == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    k = min(n_per_class, len(pos), len(neg)) if n_per_class > 0 else min(len(pos), len(neg))
    return pos[:k] + neg[:k]


@torch.no_grad()
def extract_corpus(model, rows, data_root, layer, sr, dur, device, batch_size, cache_path):
    if cache_path.exists():
        d = np.load(cache_path)
        print(f"[cache] {cache_path.name}: {d['feats'].shape[0]} items")
        return d["feats"], d["labels"]
    n_samples = int(sr * dur)
    feats, labels, buf, buf_y = [], [], [], []

    def flush():
        if not buf: return
        batch = torch.stack(buf).to(device).half()
        out = model(batch, output_hidden_states=True)
        h = out.hidden_states[layer].mean(dim=1)          # [B, H]
        feats.append(h.float().cpu().numpy())
        labels.extend(buf_y)
        buf.clear(); buf_y.clear()

    for i, r in enumerate(rows):
        try:
            wav, osr = load_audio(Path(data_root) / r.path)
        except Exception as e:
            print(f"  SKIP unreadable {r.path}: {e}"); continue
        wav = resample_linear(to_mono(wav), osr, sr)
        buf.append(crop_or_pad(wav, n_samples, random_crop=False).clamp(-1, 1))
        buf_y.append(r.target)
        if len(buf) == batch_size: flush()
        if i % 2000 == 0 and i: print(f"  {i}/{len(rows)}")
    flush()
    feats = np.concatenate(feats); labels = np.array(labels)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, feats=feats, labels=labels)
    print(f"[cache] wrote {cache_path.name}  {feats.shape}")
    return feats, labels


def bac_at(labels, scores, thr):
    pred = (scores >= thr).astype(int)   # score = P(spoof); >= thr -> spoof
    tpr = ((pred == 1) & (labels == 1)).sum() / max((labels == 1).sum(), 1)
    tnr = ((pred == 0) & (labels == 0)).sum() / max((labels == 0).sum(), 1)
    return 0.5 * (tpr + tnr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="..")
    ap.add_argument("--manifest-dir", default="manifests")
    ap.add_argument("--train-corpus", default="asvspoof5")
    ap.add_argument("--eval-corpora", nargs="+", default=["inthewild", "replaydf", "ai4t"])
    ap.add_argument("--model", default="microsoft/wavlm-large")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=8000, help="per class; 0 = all")
    ap.add_argument("--n-dev", type=int, default=2000, help="per class; 0 = all")
    ap.add_argument("--max-eval", type=int, default=0, help="cap per eval corpus; 0 = all")
    ap.add_argument("--C", type=float, default=1e6, help="logreg C (Combei use weak reg)")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--cache-dir", default="features_cache/linear_baseline")
    ap.add_argument("--out", default="experiments/e003_linear_baseline/result.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(args.model).to(device).eval().half()
    tag = args.model.split("/")[-1]
    cdir = Path(args.cache_dir)

    def cache_p(corpus, split):
        return cdir / f"{tag}_L{args.layer}_{corpus}_{split}.npz"

    # --- train + dev features (ASVspoof5 only) ---
    mp = Path(args.manifest_dir) / f"{args.train_corpus}.csv"
    tr_rows = balanced_subsample(read_manifest(mp, splits=["train"], corpora=[args.train_corpus]), args.n_train)
    dv_rows = balanced_subsample(read_manifest(mp, splits=["val"], corpora=[args.train_corpus]), args.n_dev)
    print(f"[train] {args.train_corpus}: {len(tr_rows)} train / {len(dv_rows)} dev")
    Xtr, ytr = extract_corpus(model, tr_rows, args.data_root, args.layer, args.sample_rate,
                              args.duration, device, args.batch_size, cache_p(args.train_corpus, "train"))
    Xdv, ydv = extract_corpus(model, dv_rows, args.data_root, args.layer, args.sample_rate,
                              args.duration, device, args.batch_size, cache_p(args.train_corpus, "val"))

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=args.C, max_iter=5000).fit(scaler.transform(Xtr), ytr)

    s_dv = clf.predict_proba(scaler.transform(Xdv))[:, 1]
    dev_eer = float(equal_error_rate(ydv, s_dv))
    # dev EER threshold for BAC
    fprs = np.sort(s_dv[ydv == 0]); thr = float(np.quantile(s_dv[ydv == 0], 1 - dev_eer)) if len(fprs) else 0.5
    print(f"[dev] {args.train_corpus} EER={dev_eer:.4f}  thr={thr:.4f}")

    # --- eval corpora ---
    result = {"dev": {"corpus": args.train_corpus, "eer": dev_eer, "thr": thr},
              "layer": args.layer, "model": tag, "per_corpus": {}}
    pool = {}   # corpus -> (labels, scores) for Kwok matrix
    emb_real = {}  # corpus -> real embeddings for probe
    print(f"\n{'corpus':12s} {'EER':>8s} {'AUC':>8s} {'BAC':>8s}  n")
    for c in args.eval_corpora:
        mp = Path(args.manifest_dir) / f"{c}.csv"
        rows = read_manifest(mp, corpora=[c])
        if args.max_eval > 0 and len(rows) > args.max_eval:
            rng = random.Random(13); rows = rng.sample(rows, args.max_eval)
        X, y = extract_corpus(model, rows, args.data_root, args.layer, args.sample_rate,
                              args.duration, device, args.batch_size,
                              cache_p(c, "eval" if args.max_eval == 0 else f"eval{args.max_eval}"))
        s = clf.predict_proba(scaler.transform(X))[:, 1]
        eer = float(equal_error_rate(y, s)); auc = float(roc_auc_score(y, s)); bac = float(bac_at(y, s, thr))
        print(f"{c:12s} {eer:8.4f} {auc:8.4f} {bac:8.4f}  {len(y)}")
        result["per_corpus"][c] = {"eer": eer, "auc": auc, "bac": bac, "n": int(len(y))}
        pool[c] = (y, s); emb_real[c] = X[y == 0]

    # --- Kwok bona-fide matrix ---
    print("\n=== KWOK BONA-FIDE MATRIX (EER per spoof-set x bona-domain) ===")
    kwok = {}
    for cs, (ys, ss) in pool.items():
        spoof_s = ss[ys == 1]
        if len(spoof_s) == 0: continue
        row = {}
        for cb, (yb, sb) in pool.items():
            bona_s = sb[yb == 0]
            if len(bona_s) == 0: continue
            lab = np.concatenate([np.ones_like(spoof_s), np.zeros_like(bona_s)])
            sc = np.concatenate([spoof_s, bona_s])
            row[f"{cb}_real"] = round(float(equal_error_rate(lab, sc)), 4)
        vals = list(row.values())
        print(f"{cs:12s} mean={np.mean(vals):.4f} std={np.std(vals):.4f}  {row}")
        kwok[cs] = row
    result["kwok"] = kwok

    # --- bona-fide-source probe on raw L{layer} features ---
    doms = [c for c in emb_real if len(emb_real[c]) >= 50]
    if len(doms) >= 2:
        rng = np.random.default_rng(13)
        Xp, yp = [], []
        npc = min(min(len(emb_real[c]) for c in doms), 2000)
        for i, c in enumerate(doms):
            idx = rng.choice(len(emb_real[c]), npc, replace=False)
            Xp.append(emb_real[c][idx]); yp.extend([i] * npc)
        Xp = np.concatenate(Xp); yp = np.array(yp)
        sh = rng.permutation(len(yp)); Xp, yp = Xp[sh], yp[sh]
        k = int(0.7 * len(yp))
        sc2 = StandardScaler().fit(Xp[:k])
        pr = LogisticRegression(C=1.0, max_iter=2000).fit(sc2.transform(Xp[:k]), yp[:k])
        acc = float(pr.score(sc2.transform(Xp[k:]), yp[k:]))
        print(f"\n=== BONA-FIDE-SOURCE PROBE (raw L{args.layer}) ===")
        print(f"probe acc = {acc:.4f}  (chance = {1/len(doms):.4f}, {len(doms)} domains)")
        result["bonafide_probe"] = {"acc": acc, "chance": 1 / len(doms), "domains": doms}

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
