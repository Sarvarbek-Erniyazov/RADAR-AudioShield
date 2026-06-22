"""Held-out cross-test: per-corpus EER/BAC/ECE + Kwok bona-fide matrix + probe.

One forward pass over each held-out manifest yields embeddings + scores, reused
for all three analyses. Generic over corpora so the same tool runs Tier B now
and Tier C (Llama/WaveFake/CFAD) later with only different --corpora args.
"""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from audioshield.data.manifest import read_manifest
from audioshield.data.unified_dataset import UnifiedAudioDataset, collate_unified
from audioshield.models.detector import AudioShieldX
from audioshield.evaluation.metrics import equal_error_rate
from audioshield.evaluation.calibration import expected_calibration_error


@torch.no_grad()
def score_manifest(model, manifest_path, corpus, data_root, device, sr, dur,
                   max_items, use_amp):
    rows = read_manifest(manifest_path, corpora=[corpus])
    if max_items and len(rows) > max_items:
        import random
        random.Random(13).shuffle(rows); rows = rows[:max_items]
    ds = UnifiedAudioDataset(rows, data_root, sample_rate=sr, duration_seconds=dur,
                             random_crop=False,
                             corpus_vocab={corpus: 0},
                             bona_source_vocab={src: i for i, src in
                                                enumerate(sorted({x.bona_fide_source for x in rows}))})
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2,
                        collate_fn=collate_unified)
    labels, scores, embs, bona_src = [], [], [], []
    for batch in loader:
        wav = batch["waveform"].to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(wav, grl_lambda=0.0)
        labels += batch["target_long"].tolist()
        scores += torch.sigmoid(out["spoof_logit"]).float().cpu().tolist()
        embs.append(out["embedding"].float().cpu().numpy())
        bona_src += batch["bona_fide_source"]
    return (np.array(labels), np.array(scores),
            np.concatenate(embs, 0) if embs else np.zeros((0, 256)),
            np.array(bona_src))


def threshold_from_dev(model, dev_manifests, data_root, device, sr, dur, use_amp):
    """Pick the EER-threshold on pooled training-corpus dev (honest protocol)."""
    all_lab, all_sc = [], []
    for corpus, mp in dev_manifests.items():
        rows = read_manifest(mp, splits=["val"], corpora=[corpus])
        if not rows:
            continue
        import random
        random.Random(7).shuffle(rows); rows = rows[:1000]
        ds = UnifiedAudioDataset(rows, data_root, sample_rate=sr, duration_seconds=dur,
                                 random_crop=False, corpus_vocab={corpus: 0},
                                 bona_source_vocab={"x": 0})
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2,
                            collate_fn=collate_unified)
        with torch.no_grad():
            for batch in loader:
                wav = batch["waveform"].to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(wav, grl_lambda=0.0)
                all_lab += batch["target_long"].tolist()
                all_sc += torch.sigmoid(out["spoof_logit"]).float().cpu().tolist()
    lab = np.array(all_lab); sc = np.array(all_sc)
    # threshold at EER point on pooled dev
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(lab, sc)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(thr[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-config", default="configs/models/audioshield_x_v1.yaml")
    ap.add_argument("--data-root", default="..")
    ap.add_argument("--manifest-dir", default="manifests")
    ap.add_argument("--corpora", nargs="+", required=True,
                    help="held-out corpora to test, e.g. inthewild replaydf ai4t")
    ap.add_argument("--dev-corpora", nargs="+", default=["diffssd", "fakeorreal", "asvspoof5"])
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--max-items", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.model_config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    model = AudioShieldX(cfg).to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.checkpoint} (epoch {ckpt.get('epoch','?')})")

    dev_manifests = {c: Path(args.manifest_dir) / f"{c}.csv" for c in args.dev_corpora}
    thr = threshold_from_dev(model, dev_manifests, args.data_root, device,
                             args.sample_rate, args.duration, use_amp)
    print(f"dev-EER threshold = {thr:.4f}")

    # score every held-out corpus once; cache for all analyses
    cache = {}
    print("\n=== PER-CORPUS HELD-OUT METRICS ===")
    print(f"{'corpus':12s} {'EER':>8s} {'BAC':>8s} {'ECE':>8s}  n")
    table = {}
    for c in args.corpora:
        mp = Path(args.manifest_dir) / f"{c}.csv"
        if not mp.exists():
            print(f"{c:12s} NO MANIFEST"); continue
        lab, sc, emb, bsrc = score_manifest(model, mp, c, args.data_root, device,
                                             args.sample_rate, args.duration,
                                             args.max_items, use_amp)
        cache[c] = (lab, sc, emb, bsrc)
        if len(set(lab)) < 2:
            print(f"{c:12s} single-class (bona-only); skipping EER")
            continue
        eer = equal_error_rate(lab, sc)
        pred = (sc >= thr).astype(int)
        from sklearn.metrics import balanced_accuracy_score
        bac = balanced_accuracy_score(lab, pred)
        ece = expected_calibration_error(lab, sc)
        table[c] = {"eer": float(eer), "bac": float(bac), "ece": float(ece), "n": int(len(lab))}
        print(f"{c:12s} {eer:8.4f} {bac:8.4f} {ece:8.4f}  {len(lab)}")

    # Kwok bona-fide matrix: rows=spoof sets, cols=bona-fide domains
    print("\n=== KWOK BONA-FIDE MATRIX (EER per spoof-set x bona-domain) ===")
    bona_pools = {}
    for c, (lab, sc, emb, bsrc) in cache.items():
        for src in set(bsrc[lab == 0]):
            bona_pools.setdefault(src, []).append(sc[(lab == 0) & (bsrc == src)])
    bona_pools = {k: np.concatenate(v) for k, v in bona_pools.items()}
    bona_domains = sorted(bona_pools)
    print(f"bona-fide domains (columns): {bona_domains}")
    kwok = {}
    for c, (lab, sc, emb, bsrc) in cache.items():
        spoof = sc[lab == 1]
        if len(spoof) == 0:
            continue
        row = {}
        for bd in bona_domains:
            bona = bona_pools[bd]
            if len(bona) < 5:
                continue
            y = np.r_[np.ones(len(spoof)), np.zeros(len(bona))]
            s = np.r_[spoof, bona]
            row[bd] = float(equal_error_rate(y, s))
        if row:
            vals = list(row.values())
            kwok[c] = {"per_bona": row, "std": float(np.std(vals)), "mean": float(np.mean(vals))}
            print(f"{c:12s} mean={np.mean(vals):.4f} std={np.std(vals):.4f}  {({k: round(v,4) for k,v in row.items()})}")

    # bona-fide-source linear probe: can we still predict corpus from bona embeddings?
    print("\n=== BONA-FIDE-SOURCE LINEAR PROBE (lower = more invariant) ===")
    Xs, ys = [], []
    for c, (lab, sc, emb, bsrc) in cache.items():
        m = lab == 0
        if m.sum() == 0:
            continue
        Xs.append(emb[m]); ys += list(bsrc[m])
    probe_acc = None
    if Xs:
        X = np.concatenate(Xs, 0); y = np.array(ys)
        if len(set(y)) >= 2:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
            Xn = StandardScaler().fit_transform(X)
            clf = LogisticRegression(max_iter=2000, multi_class="auto")
            probe_acc = float(cross_val_score(clf, Xn, y, cv=3).mean())
            chance = 1.0 / len(set(y))
            print(f"probe acc = {probe_acc:.4f}  (chance = {chance:.4f}, {len(set(y))} domains)")
            print(f"  -> {'NEAR CHANCE: invariant (BMI worked)' if probe_acc < chance*1.5 else 'ABOVE CHANCE: residual corpus signal'}")

    result = {"checkpoint": args.checkpoint, "epoch": ckpt.get("epoch"),
              "threshold": thr, "per_corpus": table, "kwok": kwok,
              "bona_probe_acc": probe_acc}
    out = args.out or f"experiments/e001_unified_v1/crosstest_{Path(args.checkpoint).stem}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
