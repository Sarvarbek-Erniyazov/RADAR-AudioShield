"""UPC train/validate loop with BMI + CALAS."""

from __future__ import annotations

from collections import defaultdict
from itertools import islice

import torch
from tqdm.auto import tqdm

from ..losses.classification import spoof_bce
from ..losses.prototype import aam_prototype_loss, spoof_prototype_regularizers
from ..losses.consistency import consistency_loss
from ..losses.bmi import bmi_loss
from ..losses.latent_aug import CalasController, augment_spoof_embeddings
from ..evaluation.metrics import binary_metrics


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, calas: CalasController,
                    grl_lambda: float, use_amp: bool, max_steps: int = 0):
    model.train()
    w = cfg["loss_weights"]
    totals = defaultdict(float); steps = 0
    corpus_margin_accum = defaultdict(list)
    total = max_steps or (len(loader) if hasattr(loader, "__len__") else None)
    iterable = islice(loader, max_steps) if max_steps else loader
    progress_every = int(cfg.get("train", {}).get("progress_every", 25))

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
        target = batch["target_long"].to(device, non_blocking=True)
        dom = batch["bona_source_id"].to(device, non_blocking=True)
        corpora = batch["corpus"]

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(wav, grl_lambda=grl_lambda)
            out_aug = model(wav + 0.01 * torch.randn_like(wav), grl_lambda=0.0)

            l_cls = spoof_bce(out["spoof_logit"], target,
                              spoof_pos_weight=cfg["spoof_pos_weight"],
                              focal_gamma=cfg["focal_gamma"])
            l_aam = aam_prototype_loss(out["bona_cos"], out["spoof_cos"], target,
                                       margin=cfg["model"]["prototypes"]["aam_margin"],
                                       scale=cfg["model"]["prototypes"]["aam_scale"])
            intra, inter = spoof_prototype_regularizers(model.prototypes.spoof, model.prototypes.bona)
            l_con = consistency_loss(out["spoof_logit"], out_aug["spoof_logit"],
                                     out["embedding"], out_aug["embedding"])

            bona = target == 0
            if bona.sum() >= 2 and "domain_logits" in out:
                n_dom = len(set(dom[bona].tolist()))
                if n_dom < 2:
                    print(f"[warn] batch has <2 bona domains (n={n_dom}); BMI Kwok will be ~0")
                l_bmi, bmi_log = bmi_loss(
                    out["bona_cos"][bona], out["domain_logits"][bona],
                    torch.sigmoid(out["spoof_logit"][bona]), dom[bona],
                    kwok_kind=cfg["bmi_kwok_kind"],
                    w_cent=w["bmi_cent"], w_grl=w["bmi_grl"], w_kwok=w["bmi_kwok"])
            else:
                l_bmi = out["embedding"].new_tensor(0.0); bmi_log = {}

            loss = (w["cls"] * l_cls + w["aam"] * l_aam
                    + w["proto_reg"] * (intra + inter)
                    + w["con"] * l_con + l_bmi)

        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            continue  # skip a non-finite batch instead of poisoning weights
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0)
        if not torch.isfinite(gnorm):
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue
        scaler.step(optimizer); scaler.update()

        # CALAS: accumulate per-corpus spoof margin (proto_spoof_score on spoof samples)
        sp = target == 1
        if sp.any():
            sc = out["proto_spoof_score"][sp].detach().float().cpu()
            for c, m in zip([corpora[i] for i in sp.nonzero(as_tuple=True)[0].tolist()], sc.tolist()):
                corpus_margin_accum[c].append(m)

        totals["loss"] += float(loss.detach()); totals["cls"] += float(l_cls.detach())
        totals["aam"] += float(l_aam.detach()); totals["con"] += float(l_con.detach())
        for k, v in bmi_log.items():
            totals[k] += v
        steps += 1
        if progress_every > 0 and (steps == 1 or steps % progress_every == 0):
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}", grl=f"{grl_lambda:.3f}")

    calas.update({c: sum(v)/len(v) for c, v in corpus_margin_accum.items() if v})
    return {k: v / max(1, steps) for k, v in totals.items()}


@torch.no_grad()
def validate(model, loaders_by_corpus, device, use_amp: bool):
    """loaders_by_corpus: {corpus: DataLoader}. Returns per-corpus EER dict."""
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
