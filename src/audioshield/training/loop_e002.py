"""e002 train loop: channel-consistency, no BMI / CALAS / prototypes."""

from __future__ import annotations

from collections import defaultdict

import torch

from ..losses.classification import spoof_bce
from ..losses.consistency import consistency_loss
from ..losses.xc_contrastive import cross_corpus_supcon


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
    totals = defaultdict(float); steps = 0

    for bi, batch in enumerate(loader):
        if max_steps and bi >= max_steps:
            break
        wav = batch["waveform"].to(device)
        wav_deg = batch["waveform_deg"].to(device)
        target = batch["target_long"].to(device)
        corpus_id = batch["corpus_id"].to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
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
                l_xc, xc_log = cross_corpus_supcon(
                    out["embedding"], target, corpus_id,
                    temperature=xc_temp, cross_corpus_only=xc_cco)
            else:
                l_xc = out["embedding"].sum() * 0.0; xc_log = {}
            loss = w_cls * l_cls + w_con * l_con + w_xc * l_xc

        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0)
        if not torch.isfinite(gnorm):
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue
        scaler.step(optimizer); scaler.update()

        totals["loss"] += float(loss.detach())
        totals["cls"] += float(l_cls.detach())
        totals["con"] += float(l_con.detach())
        totals["xc"] += float(l_xc.detach())
        steps += 1

    if steps == 0:
        print("[e002][warn] epoch ran 0 steps (dataloader exhausted or worker died)", flush=True)
        return {"loss": float("nan"), "cls": float("nan"), "con": float("nan"), "xc": float("nan"), "steps": 0}
    out = {k: v / steps for k, v in totals.items()}
    out["steps"] = steps
    return out


@torch.no_grad()
def probe_corpus_during_train(model, loaders_by_corpus, device, use_amp, max_per_corpus=500):
    """Per-epoch corpus-identity probe. Reuses the same recipe as cross_test
    (StandardScaler -> LogisticRegression -> 3-fold CV accuracy) on dev embeddings.
    Returns probe accuracy, or None if sklearn unavailable / too few corpora."""
    import numpy as np
    model.eval()
    X, y = [], []
    # collect a capped sample of embeddings + corpus ids across dev loaders
    seen = {}
    with torch.no_grad():
        for corpus, loader in loaders_by_corpus.items():
            if corpus.endswith("_deg"):
                continue  # use clean dev only, one view per corpus
            n = 0
            for batch in loader:
                wav = batch["waveform"].to(device)
                cid = batch["corpus_id"].tolist()
                with torch.cuda.amp.autocast(enabled=use_amp):
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
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        Xn = StandardScaler().fit_transform(X)
        clf = LogisticRegression(max_iter=2000, multi_class="auto")
        return float(cross_val_score(clf, Xn, y, cv=3).mean())
    except Exception as e:
        print(f"[probe][warn] {e}")
        return None


def validate_e002(model, loaders_by_corpus, device, use_amp: bool):
    from ..evaluation.metrics import binary_metrics
    model.eval()
    per_corpus = {}
    for corpus, loader in loaders_by_corpus.items():
        labels, scores = [], []
        for batch in loader:
            wav = batch["waveform"].to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(wav, grl_lambda=0.0)
            labels += batch["target_long"].tolist()
            scores += torch.sigmoid(out["spoof_logit"]).float().cpu().tolist()
        per_corpus[corpus] = binary_metrics(labels, scores)["eer"]
    return per_corpus
