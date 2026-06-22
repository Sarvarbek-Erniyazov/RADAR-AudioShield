"""Frozen-SSL layer probe: pick backbone + layer by mean cross-corpus dev EER."""

from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import random
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
    rng.shuffle(pos)
    rng.shuffle(neg)
    k = min(n_per_class, len(pos), len(neg))
    return pos[:k] + neg[:k]


@torch.no_grad()
def pool_layers(model, wav, device, n_samples):
    mono = crop_or_pad(wav, n_samples, random_crop=False).clamp(-1, 1)
    out = model(mono[None].to(device).half(), output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=0).squeeze(1)  # [L+1, T, H]
    return hs.mean(dim=1).float().cpu().numpy()            # [L+1, H]


def extract(model_name, manifests, data_root, cache_dir, sr, dur, n_train, n_val, device):
    tag = model_name.split("/")[-1]
    cache = cache_dir / f"{tag}.npz"
    if cache.exists():
        print(f"[cache] {cache} exists, loading")
        d = np.load(cache, allow_pickle=True)
        return d["feats"], d["labels"], d["corpora"], d["splits"]

    model = AutoModel.from_pretrained(model_name).to(device).eval().half()
    n_samples = int(sr * dur)
    feats, labels, corpora, splits = [], [], [], []
    for corpus, mp in manifests.items():
        for split, cap in [("train", n_train), ("val", n_val)]:
            rows = read_manifest(mp, splits=[split], corpora=[corpus])
            if not rows:
                print(f"[warn] {corpus}/{split}: no rows")
                continue
            rows = balanced_subsample(rows, cap)
            print(f"[{tag}] {corpus}/{split}: {len(rows)} items")
            for i, r in enumerate(rows):
                wav, osr = load_audio(Path(data_root) / r.path)
                wav = resample_linear(to_mono(wav), osr, sr)
                feats.append(pool_layers(model, wav, device, n_samples))
                labels.append(r.target)
                corpora.append(corpus)
                splits.append(split)
                if i % 500 == 0 and i:
                    print(f"  {corpus}/{split} {i}/{len(rows)}")
    feats = np.stack(feats)
    labels = np.array(labels)
    corpora = np.array(corpora)
    splits = np.array(splits)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, feats=feats, labels=labels, corpora=corpora, splits=splits)
    print(f"[cache] wrote {cache}  feats={feats.shape}")
    del model
    torch.cuda.empty_cache()
    return feats, labels, corpora, splits


def fit(tag, feats, labels, corpora, splits):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    L = feats.shape[1]
    tr = splits == "train"
    dv = splits == "val"
    res = {}
    for layer in range(L):
        scaler = StandardScaler().fit(feats[tr, layer])
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(scaler.transform(feats[tr, layer]), labels[tr])
        per = {}
        for c in sorted(set(corpora[dv])):
            m = dv & (corpora == c)
            if m.sum() == 0 or len(set(labels[m])) < 2:
                continue
            s = clf.predict_proba(scaler.transform(feats[m, layer]))[:, 1]
            per[c] = float(equal_error_rate(labels[m], s))
        mean_cross = float(np.mean(list(per.values()))) if per else float("nan")
        res[layer] = {"mean_dev_eer": mean_cross, "per_corpus": per}
        print(f"[{tag}] layer {layer:2d}  mean_dev_eer={mean_cross:.4f}  {per}")
    valid = [l for l in res if not np.isnan(res[l]["mean_dev_eer"])]
    best = min(valid, key=lambda l: res[l]["mean_dev_eer"])
    print(f"[{tag}] BEST layer={best} mean_dev_eer={res[best]['mean_dev_eer']:.4f}")
    return {"best_layer": int(best), "best_mean_dev_eer": res[best]["mean_dev_eer"], "layers": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="..")
    ap.add_argument("--manifest-dir", default="manifests")
    ap.add_argument("--corpora", nargs="+", default=["diffssd", "fakeorreal", "asvspoof5"])
    ap.add_argument("--models", nargs="+",
                    default=["microsoft/wavlm-large", "facebook/wav2vec2-xls-r-300m"])
    ap.add_argument("--cache-dir", default="features_cache")
    ap.add_argument("--out", default="experiments/e000_layer_probe/probe_summary.json")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=2000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifests = {c: Path(args.manifest_dir) / f"{c}.csv" for c in args.corpora}
    summary = {}
    for model_name in args.models:
        tag = model_name.split("/")[-1]
        feats, labels, corpora, splits = extract(
            model_name, manifests, args.data_root, Path(args.cache_dir),
            args.sample_rate, args.duration, args.n_train, args.n_val, device)
        summary[tag] = fit(tag, feats, labels, corpora, splits)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("\nwrote", args.out)
    print("=== WINNER ===")
    best_tag = min(summary, key=lambda t: summary[t]["best_mean_dev_eer"])
    print(f"{best_tag} layer {summary[best_tag]['best_layer']} "
          f"mean_dev_eer={summary[best_tag]['best_mean_dev_eer']:.4f}")


if __name__ == "__main__":
    main()
