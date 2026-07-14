"""e002 train loop: channel-consistency, no BMI / CALAS / prototypes."""

from __future__ import annotations

from collections import defaultdict
from itertools import islice

import torch
from tqdm.auto import tqdm

from ..losses.classification import spoof_bce
from ..losses.consistency import consistency_loss
from ..losses.xc_contrastive import cross_corpus_supcon
from ..training.supcon_guard import supcon_batch_valid


def train_one_epoch_e002(model, loader, optimizer, scaler, device, cfg,
                         use_amp: bool, max_steps: int = 0):
    model.train()
    cw = cfg["consistency"]
    lam_kl = float(cw.get("lambda_kl", 1.0))
    lam_emb = float(cw.get("lambda_emb", 0.5))
    w_cls = float(cfg["loss_weights"].get("cls", 1.0))
    w_con = float(cfg["loss_weights"].get("con", 1.0))
    xc = cfg.get("xc_contrastive", {})
    xc_on = bool(xc.get("enabled", False))
    w_xc = float(xc.get("lambda_xc", 0.0))
    xc_temp = float(xc.get("temperature", 0.1))
    xc_cco = bool(xc.get("cross_corpus_only", True))
    xc_use_projection = bool(xc.get("use_projection_head", False))
    xc_min_corpora = int(xc.get("min_corpora_per_class", 2))
    totals = defaultdict(float); steps = 0
    skipped_nonfinite_loss = 0
    skipped_nonfinite_grad = 0
    xc_guard_skipped_batches = 0
    progress_every = int(cfg.get("train", {}).get("progress_every", 25))
    total = max_steps or (len(loader) if hasattr(loader, "__len__") else None)

    iterable = islice(loader, max_steps) if max_steps else loader
    progress = tqdm(
        enumerate(iterable),
        total=total,
        desc="train",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )
    for bi, batch in progress:
        wav = batch["waveform"].to(device, non_blocking=True)
        wav_deg = batch["waveform_deg"].to(device, non_blocking=True)
        target = batch["target_long"].to(device, non_blocking=True)
        corpus_id = batch["corpus_id"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(wav, grl_lambda=0.0)
            out_deg = model(wav_deg, grl_lambda=0.0)

            l_cls = 0.5 * (
                spoof_bce(out["spoof_logit"], target,
                          spoof_pos_weight=cfg["spoof_pos_weight"],
                          focal_gamma=cfg["focal_gamma"])
                + spoof_bce(out_deg["spoof_logit"], target,
                            spoof_pos_weight=cfg["spoof_pos_weight"],
                            focal_gamma=cfg["focal_gamma"])
            )
            l_con = consistency_loss(
                out["spoof_logit"], out_deg["spoof_logit"],
                out["embedding"], out_deg["embedding"],
                lambda_kl=lam_kl, lambda_emb=lam_emb, teacher=True)

            if xc_on and w_xc > 0.0:
                valid, _guard_diag = supcon_batch_valid(
                    corpus_id.tolist(), target.tolist(), min_corpora_per_class=xc_min_corpora)
                if valid:
                    xc_emb = out["contrastive_embedding"] if xc_use_projection else out["embedding"]
                    l_xc, xc_log = cross_corpus_supcon(
                        xc_emb, target, corpus_id,
                        temperature=xc_temp, cross_corpus_only=xc_cco)
                else:
                    # batch has a class spanning too few corpora for valid cross-corpus
                    # positives -- skip the term rather than silently computing it on a
                    # degenerate batch (audit §1).
                    l_xc = out["embedding"].sum() * 0.0
                    xc_log = {"xc_npos": 0.0, "xc_skipped": 1.0}
                    xc_guard_skipped_batches += 1
            else:
                l_xc = out["embedding"].sum() * 0.0; xc_log = {}
            loss = w_cls * l_cls + w_con * l_con + w_xc * l_xc

        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            skipped_nonfinite_loss += 1
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0)
        if not torch.isfinite(gnorm):
            skipped_nonfinite_grad += 1
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue
        scaler.step(optimizer); scaler.update()

        totals["loss"] += float(loss.detach())
        totals["cls"] += float(l_cls.detach())
        totals["con"] += float(l_con.detach())
        totals["xc"] += float(l_xc.detach())
        for k, v in xc_log.items():
            totals[k] += float(v)
        steps += 1
        if progress_every > 0 and (steps == 1 or steps % progress_every == 0):
            postfix = dict(
                loss=f"{float(loss.detach()):.4f}",
                skip_loss=skipped_nonfinite_loss,
                skip_grad=skipped_nonfinite_grad,
            )
            if xc_on and w_xc > 0.0:
                postfix["xc"] = f"{float(l_xc.detach()):.4f}"
                postfix["xc_npos"] = int(xc_log.get("xc_npos", 0))
            progress.set_postfix(**postfix)

    if steps == 0:
        print("[e002][warn] epoch ran 0 steps (dataloader exhausted or worker died)", flush=True)
        return {
            "loss": float("nan"),
            "cls": float("nan"),
            "con": float("nan"),
            "xc": float("nan"),
            "steps": 0,
            "skipped_nonfinite_loss": skipped_nonfinite_loss,
            "skipped_nonfinite_grad": skipped_nonfinite_grad,
            "xc_guard_skipped_batches": xc_guard_skipped_batches,
        }
    out = {k: v / steps for k, v in totals.items()}
    out["steps"] = steps
    out["skipped_nonfinite_loss"] = skipped_nonfinite_loss
    out["skipped_nonfinite_grad"] = skipped_nonfinite_grad
    out["xc_guard_skipped_batches"] = xc_guard_skipped_batches
    return out


@torch.no_grad()
def probe_corpus_during_train(model, loaders_by_corpus, device, use_amp, max_per_corpus=500):
    """Per-epoch corpus-identity probe. Reuses the same recipe as cross_test
    (StandardScaler -> LogisticRegression -> 3-fold CV accuracy) on dev embeddings.
    Returns probe accuracy, or None if sklearn unavailable / too few corpora."""
    import numpy as np
    model.eval()
    X, y = [], []
    with torch.no_grad():
        for corpus, loader in tqdm(
            loaders_by_corpus.items(),
            desc="probe corpora",
            unit="corpus",
            dynamic_ncols=True,
            leave=False,
        ):
            if corpus.endswith("_deg"):
                continue  # use clean dev only, one view per corpus
            n = 0
            total = len(loader) if hasattr(loader, "__len__") else None
            for batch in tqdm(
                loader,
                total=total,
                desc=f"probe {corpus}",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
            ):
                wav = batch["waveform"].to(device, non_blocking=True)
                cid = batch["corpus_id"].tolist()
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    out = model(wav, grl_lambda=0.0)
                emb = out["embedding"].float().cpu().numpy()
                X.append(emb); y.extend(cid)
                n += emb.shape[0]
                if n >= max_per_corpus:
                    break
    if not X:
        return None
    X = np.concatenate(X, axis=0); y = np.array(y)
    if len(set(y.tolist())) < 2:
        return None
    try:
        from sklearn.preprocessing import StandardScaler
        from audioshield.evaluation.grouped_probe import grouped_probe
        # Honest-baseline probe (audit §4.7): balanced accuracy + TRUE majority baseline,
        # replacing raw-accuracy-vs-uniform-chance. Grouping (source_id) is NOT available
        # in the probe loaders' batch dict {waveform, corpus_id}; recording-level grouping
        # is deferred to Step 2b (see docs/probe_wiring_todo.md). Ungrouped decodability is
        # an UPPER bound, so "probe pinned high" remains conservative.
        Xn = StandardScaler().fit_transform(X)
        res = grouped_probe(Xn, y, meta=None, n_splits=3, seed=13)
        return {"balanced_accuracy": res["balanced_accuracy"],
                "majority_baseline": res["majority_baseline"],
                "macro_f1": res.get("macro_f1")}
    except Exception as e:
        print(f"[probe][warn] {e}")
        return None


def validate_e002(model, loaders_by_corpus, device, use_amp: bool):
    from ..evaluation.metrics import binary_metrics
    model.eval()
    per_corpus = {}
    for corpus, loader in tqdm(
        loaders_by_corpus.items(),
        desc="validate corpora",
        unit="corpus",
        dynamic_ncols=True,
        leave=False,
    ):
        labels, scores = [], []
        total = len(loader) if hasattr(loader, "__len__") else None
        for batch in tqdm(
            loader,
            total=total,
            desc=f"validate {corpus}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        ):
            wav = batch["waveform"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(wav, grl_lambda=0.0)
            labels += batch["target_long"].tolist()
            scores += torch.sigmoid(out["spoof_logit"]).float().cpu().tolist()
        per_corpus[corpus] = binary_metrics(labels, scores)["eer"]
    return per_corpus
