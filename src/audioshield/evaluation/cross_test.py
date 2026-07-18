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
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from audioshield.data.manifest import read_manifest
from audioshield.data.unified_dataset import UnifiedAudioDataset, collate_unified
from audioshield.models.detector import AudioShieldX
from audioshield.evaluation.metrics import equal_error_rate
from audioshield.evaluation.calibration import expected_calibration_error
from audioshield.utils.runtime import describe_device


def _loader_kwargs(num_workers: int, device: torch.device) -> dict:
    num_workers = int(num_workers)
    kwargs = {"num_workers": num_workers, "pin_memory": device.type == "cuda"}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def _fast_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """Efficient EER for bootstrap resampling."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def _cluster_key(row) -> str:
    """Stable bootstrap cluster key.

    AI4T is segmented from source videos, so strip the segment suffix
    (`_000.wav`, `_001.wav`, ...) and resample source videos. Other corpora
    currently lack an explicit source-video field in the unified manifest, so
    they fall back to utterance-level clusters.
    """
    if row.corpus == "ai4t":
        return re.sub(r"_[0-9]+(?=\.wav$)", "", row.utt_id)
    return row.utt_id


def _bootstrap_eer_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    clusters: np.ndarray,
    reps: int,
    seed: int,
) -> dict:
    uniq = np.unique(clusters)
    result = {"reps": int(reps), "seed": int(seed), "n_clusters": int(len(uniq))}
    if reps <= 0 or len(uniq) < 2 or len(np.unique(labels)) < 2:
        return {**result, "eer_p2_5": None, "eer_p50": None, "eer_p97_5": None}

    by_cluster = {c: np.flatnonzero(clusters == c) for c in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_cluster[c] for c in sampled])
        eer = _fast_eer(labels[idx], scores[idx])
        if np.isfinite(eer):
            vals.append(eer)
    if not vals:
        return {**result, "eer_p2_5": None, "eer_p50": None, "eer_p97_5": None}
    q = np.percentile(np.asarray(vals, dtype=np.float64), [2.5, 50.0, 97.5])
    return {
        **result,
        "valid_reps": int(len(vals)),
        "eer_p2_5": float(q[0]),
        "eer_p50": float(q[1]),
        "eer_p97_5": float(q[2]),
    }


@torch.no_grad()
def score_manifest(model, manifest_path, corpus, data_root, device, sr, dur,
                   max_items, use_amp, batch_size, num_workers):
    rows = read_manifest(manifest_path, corpora=[corpus])
    if max_items and len(rows) > max_items:
        import random
        random.Random(13).shuffle(rows); rows = rows[:max_items]
    ds = UnifiedAudioDataset(rows, data_root, sample_rate=sr, duration_seconds=dur,
                             random_crop=False,
                             corpus_vocab={corpus: 0},
                             bona_source_vocab={src: i for i, src in
                                                enumerate(sorted({x.bona_fide_source for x in rows}))})
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_unified,
        **_loader_kwargs(num_workers, device))
    labels, scores, embs, bona_src, clusters = [], [], [], [], []
    total = len(loader) if hasattr(loader, "__len__") else None
    for batch in tqdm(
        loader,
        total=total,
        desc=f"score {corpus}",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    ):
        wav = batch["waveform"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(wav, grl_lambda=0.0)
        start = len(labels)
        labels += batch["target_long"].tolist()
        scores += torch.sigmoid(out["spoof_logit"]).float().cpu().tolist()
        embs.append(out["embedding"].float().cpu().numpy())
        bona_src += batch["bona_fide_source"]
        clusters += [_cluster_key(rows[i]) for i in range(start, len(labels))]
    return (np.array(labels), np.array(scores),
            np.concatenate(embs, 0) if embs else np.zeros((0, 256)),
            np.array(bona_src),
            np.array(clusters))


def threshold_from_dev(model, dev_manifests, data_root, device, sr, dur, use_amp, batch_size, num_workers):
    """Pick the EER-threshold on pooled training-corpus dev (honest protocol)."""
    all_lab, all_sc = [], []
    for corpus, mp in dev_manifests.items():
        rows = read_manifest(mp, splits=["val"], corpora=[corpus])
        if not rows:
            continue
        if len({r.target for r in rows}) < 2:
            print(f"[cross_test] skipping {corpus} dev threshold rows: single-class split")
            continue
        import random
        random.Random(7).shuffle(rows); rows = rows[:1000]
        ds = UnifiedAudioDataset(rows, data_root, sample_rate=sr, duration_seconds=dur,
                                 random_crop=False, corpus_vocab={corpus: 0},
                                 bona_source_vocab={"x": 0})
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, collate_fn=collate_unified,
            **_loader_kwargs(num_workers, device))
        with torch.no_grad():
            total = len(loader) if hasattr(loader, "__len__") else None
            for batch in tqdm(
                loader,
                total=total,
                desc=f"threshold {corpus}",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            ):
                wav = batch["waveform"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
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


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser so callers can validate child commands in tests."""
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
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-items", type=int, default=0, help="0 = all")
    ap.add_argument("--bootstrap-reps", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=13)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true",
                    help="overwrite --out (or the derived default path) if it already exists")
    ap.add_argument(
        "--preflight",
        action="store_true",
        help="validate checkpoint/config/manifests and sample audio paths, then exit before model load",
    )
    return ap


def _resolved_audio_path(data_root: str | Path, manifest_path: str) -> Path:
    path = Path(manifest_path)
    return path if path.is_absolute() else Path(data_root) / path


def _run_preflight(args: argparse.Namespace) -> int:
    """Validate every runtime input class before allocating or loading a model."""
    table = []
    failures = []

    def record_file(role: str, name: str, path: Path) -> None:
        resolved = path.resolve(strict=False)
        try:
            stat = resolved.stat()
            if not resolved.is_file():
                raise OSError("not a regular file")
        except OSError as exc:
            table.append((role, name, "FAIL", "-", "0/1", str(resolved)))
            failures.append(f"{role}/{name}: {resolved}: {exc}")
            return
        table.append((role, name, "OK", "-", "1/1", f"{resolved} ({stat.st_size} bytes)"))

    def record_corpus(role: str, corpus: str, split: str | None) -> None:
        manifest = (Path(args.manifest_dir) / f"{corpus}.csv").resolve(strict=False)
        try:
            if not manifest.is_file():
                raise FileNotFoundError("manifest is not a regular file")
            rows = read_manifest(
                manifest,
                splits=[split] if split else None,
                corpora=[corpus],
            )
        except Exception as exc:
            table.append((role, corpus, "FAIL", "0", "0/0", str(manifest)))
            failures.append(f"{role}/{corpus}: {manifest}: {exc}")
            return

        if role == "dev":
            import random

            random.Random(7).shuffle(rows)
            rows = rows[:1000]
        elif args.max_items and len(rows) > args.max_items:
            import random

            random.Random(13).shuffle(rows)
            rows = rows[:args.max_items]

        if not rows:
            table.append((role, corpus, "FAIL", "0", "0/0", str(manifest)))
            failures.append(f"{role}/{corpus}: {manifest}: zero runtime rows")
            return

        checked = 0
        corpus_failures = []
        for row in rows[:25]:
            audio = _resolved_audio_path(args.data_root, row.path).resolve(strict=False)
            try:
                audio.stat()
                if not audio.is_file():
                    raise OSError("not a regular file")
            except OSError as exc:
                corpus_failures.append(f"{audio}: {exc}")
            else:
                checked += 1

        expected = min(25, len(rows))
        status = "OK" if not corpus_failures and checked == expected else "FAIL"
        table.append(
            (role, corpus, status, str(len(rows)), f"{checked}/{expected}", str(manifest))
        )
        failures.extend(f"{role}/{corpus}: {failure}" for failure in corpus_failures)

    record_file("config", "model", Path(args.model_config))
    record_file("checkpoint", "model", Path(args.checkpoint))
    for corpus in args.dev_corpora:
        record_corpus("dev", corpus, "val")
    for corpus in args.corpora:
        record_corpus("heldout", corpus, None)

    print("=== PREFLIGHT OK/FAIL TABLE ===")
    print(f"{'role':10s} {'name':14s} {'status':6s} {'rows':>8s} {'audio':>8s}  path")
    for role, name, status, rows, checked, path in table:
        print(f"{role:10s} {name:14s} {status:6s} {rows:>8s} {checked:>8s}  {path}")
    for failure in failures:
        print(f"PREFLIGHT FAILURE: {failure}")
    print("PREFLIGHT OK" if not failures else "PREFLIGHT FAIL")
    return 0 if not failures else 1


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.max_items:
        print(
            "[cross_test][sanity] --max-items is set; this output is capped "
            "and must not be reported as the full-corpus OOD result."
        )

    if args.preflight:
        return _run_preflight(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    describe_device(device)
    use_amp = device.type == "cuda"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "cfg" in ckpt:
        cfg = ckpt["cfg"]
        print("[cross_test] using model config saved in checkpoint")
    else:
        import yaml
        cfg = yaml.safe_load(open(args.model_config))
        print(f"[cross_test] using fallback model config {args.model_config}")
    model = AudioShieldX(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"loaded {args.checkpoint} (epoch {ckpt.get('epoch','?')})")

    dev_manifests = {c: Path(args.manifest_dir) / f"{c}.csv" for c in args.dev_corpora}
    thr = threshold_from_dev(model, dev_manifests, args.data_root, device,
                             args.sample_rate, args.duration, use_amp,
                             args.batch_size, args.num_workers)
    print(f"dev-EER threshold = {thr:.4f}")

    # score every held-out corpus once; cache for all analyses
    cache = {}
    print("\n=== PER-CORPUS HELD-OUT METRICS ===")
    print(f"{'corpus':12s} {'EER':>8s} {'BAC':>8s} {'ECE':>8s}  n")
    table = {}
    for c in tqdm(args.corpora, desc="held-out corpora", unit="corpus", dynamic_ncols=True):
        mp = Path(args.manifest_dir) / f"{c}.csv"
        if not mp.exists():
            print(f"{c:12s} NO MANIFEST"); continue
        lab, sc, emb, bsrc, clusters = score_manifest(model, mp, c, args.data_root, device,
                                                       args.sample_rate, args.duration,
                                                       args.max_items, use_amp,
                                                       args.batch_size, args.num_workers)
        cache[c] = (lab, sc, emb, bsrc, clusters)
        if len(set(lab)) < 2:
            print(f"{c:12s} single-class (bona-only); skipping EER")
            continue
        eer = equal_error_rate(lab, sc)
        pred = (sc >= thr).astype(int)
        from sklearn.metrics import balanced_accuracy_score
        bac = balanced_accuracy_score(lab, pred)
        ece = expected_calibration_error(lab, sc)
        ci = _bootstrap_eer_ci(
            lab,
            sc,
            clusters,
            reps=args.bootstrap_reps,
            seed=args.bootstrap_seed + len(table),
        )
        table[c] = {
            "eer": float(eer),
            "eer_ci95": ci,
            "bac": float(bac),
            "ece": float(ece),
            "n": int(len(lab)),
        }
        print(f"{c:12s} {eer:8.4f} {bac:8.4f} {ece:8.4f}  {len(lab)}")

    # Kwok bona-fide matrix: rows=spoof sets, cols=bona-fide domains
    print("\n=== KWOK BONA-FIDE MATRIX (EER per spoof-set x bona-domain) ===")
    bona_pools = {}
    for c, (lab, sc, emb, bsrc, clusters) in cache.items():
        for src in set(bsrc[lab == 0]):
            bona_pools.setdefault(src, []).append(sc[(lab == 0) & (bsrc == src)])
    bona_pools = {k: np.concatenate(v) for k, v in bona_pools.items()}
    bona_domains = sorted(bona_pools)
    print(f"bona-fide domains (columns): {bona_domains}")
    kwok = {}
    for c, (lab, sc, emb, bsrc, clusters) in cache.items():
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
    for c, (lab, sc, emb, bsrc, clusters) in cache.items():
        m = lab == 0
        if m.sum() == 0:
            continue
        Xs.append(emb[m]); ys += list(bsrc[m])
    probe_acc = None
    probe_error = None
    if Xs and os.environ.get("AUDIOSHIELD_SKIP_BONA_PROBE") == "1":
        probe_error = "skipped via AUDIOSHIELD_SKIP_BONA_PROBE=1"
        print(f"  bona-fide probe SKIPPED ({probe_error})")
    elif Xs:
        X = np.concatenate(Xs, 0); y = np.array(ys)
        if len(set(y)) >= 2:
            from audioshield.evaluation.grouped_probe import grouped_probe
            from sklearn.preprocessing import StandardScaler
            # NOTE: cache carries no source_id/speaker_id, so we cannot group by recording
            # here yet (audit §4.7). Until source_id is threaded into the cache tuple, we
            # report the HONEST-BASELINE version: balanced accuracy + macro-F1 vs the true
            # majority baseline (not 1/n_domains), ungrouped. This is a strict improvement
            # over raw-accuracy-vs-uniform-chance and does NOT overstate invariance.
            # Grouping upgrade tracked in docs/probe_wiring_todo.md.
            probe_res = grouped_probe(Xn := StandardScaler().fit_transform(X), y,
                                      meta=None, n_splits=3, seed=13)
            probe_acc = probe_res["balanced_accuracy"]
            base = probe_res["majority_baseline"]
            print(f"probe balanced_acc = {probe_acc:.4f}  (honest majority baseline = {base:.4f}, "
                  f"{probe_res.get('n_groups','?')} groups, {len(set(y))} domains)")
            print(f"  -> {'NEAR BASELINE: low residual corpus signal' if probe_acc < base + 0.10 else 'ABOVE BASELINE: residual corpus signal decodable'}")

    result = {"checkpoint": args.checkpoint, "epoch": ckpt.get("epoch"),
              "threshold": thr, "per_corpus": table, "kwok": kwok,
              "bona_probe_acc": probe_acc, "bona_probe_detail": (probe_res if "probe_res" in dir() else None),
              "reported_full_corpora": not bool(args.max_items),
              "max_items": int(args.max_items),
              "bootstrap_reps": int(args.bootstrap_reps)}
    # Default output lives beside the checkpoint and is namespaced by the checkpoint's
    # own parent dir name, not just its (often shared, e.g. "best") stem -- prevents
    # e007_A/B/C all colliding on one shared default path (report finding 3.5, High).
    ckpt_path = Path(args.checkpoint)
    default_name = f"crosstest_{ckpt_path.parent.name}_{ckpt_path.stem}.json"
    out = Path(args.out) if args.out else ckpt_path.parent / default_name
    if out.exists() and not args.force:
        raise SystemExit(
            f"[cross_test] refusing to overwrite existing {out} -- pass --force to overwrite, "
            "or choose a different --out (prevents silent last-writer-wins collisions)."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
