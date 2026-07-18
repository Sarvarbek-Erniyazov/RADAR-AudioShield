"""Domain-Reliance Score on the six frozen-phase checkpoints (read-only).
DRS(w) = ||U^T w||^2 / ||w||^2 ; w = binary.fc.weight ; U = domain subspace (k=3).
Frozen models share one embedding space (U from a representative frozen model);
e004 (top-4 fine-tuned) gets its OWN embedding + OWN U to avoid space mismatch.
"""
import os, sys, json, random
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
sys.path.insert(0, "src")
import numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader
from audioshield.data.manifest import read_manifest
from audioshield.data.unified_dataset import UnifiedAudioDataset, collate_unified
from audioshield.models.detector import AudioShieldX

import warnings
warnings.warn(
    "scripts/compute_drs.py is out of 2a scope and known-broken: collect_embeddings() "
    "still pools all corpora (including VCTK, bona-fide-only) uniformly for "
    "domain-subspace estimation -- the audit §3.3 class-confound is NOT fixed here. "
    "This script is superseded by Roadmap v3 Step 3's DRS validity gate; "
    "see docs/2a_scope_notes.md.",
    DeprecationWarning, stacklevel=2,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAP = 400            # clean dev utts per corpus for embedding/subspace
SEED = 13

CKPTS = {
    "e002-A": "runs/e002_augonly_v1/best.pt",
    "e002-B": "runs/e002_consistency_v1/best.pt",
    "e004":   "runs/e004_aft_v1/best.pt",          # fine-tuned -> own subspace
    "e005-A": "runs/e005_multicorpus_frozen_v1/best.pt",
    "e005-C": "runs/e005_multicorpus_frozen_C/best.pt",
    "e006":   "runs/e006_xc_frozen_v1/best.pt",
}
FROZEN_REF = "e005-A"   # representative frozen model -> shared embedding/U for the 5 frozen
FROZEN_SET = {"e002-A","e002-B","e005-A","e005-C","e006"}

def build_model_from_ckpt(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    cfg = sd["cfg"]; cvocab = sd["corpus_vocab"]
    model = AudioShieldX(cfg).to(DEVICE)
    model.load_state_dict(sd["model"], strict=False)
    model.eval()
    return model, cfg, cvocab, sd

def collect_embeddings(model, cfg, cvocab):
    """Run clean dev through model, return X (N,256) embeddings + y (N,) corpus ids."""
    exp = cfg["experiment"]; md = exp["manifest_dir"]
    X, y = [], []
    for c in exp["train_corpora"]:
        rows = read_manifest(Path(md)/f"{c}.csv", splits=["val"], corpora=[c])
        if not rows: continue
        random.Random(SEED).shuffle(rows); rows = rows[:CAP]
        ds = UnifiedAudioDataset(rows, exp["data_root"], sample_rate=exp["sample_rate"],
                                 duration_seconds=exp["duration_seconds"], random_crop=False,
                                 corpus_vocab=cvocab, bona_source_vocab={})
        dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_unified)
        cid = cvocab[c]
        with torch.no_grad():
            for b in dl:
                out = model(b["waveform"].to(DEVICE), grl_lambda=0.0)
                X.append(out["embedding"].float().cpu().numpy())
                y += [cid]*out["embedding"].shape[0]
    return np.concatenate(X,0), np.array(y)

def domain_subspace(X, y, k=3):
    """Two estimates of the domain subspace, each orthonormal (256,k)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    Xs = StandardScaler().fit_transform(X)
    # A: probe weight rows -> orthonormalize
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xs, y)
    Wd = clf.coef_                      # (C,256)
    Ua,_ = np.linalg.qr(Wd.T)           # (256, C)
    Ua = Ua[:, :k]
    # B: centered class means -> SVD
    mus = np.stack([X[y==c].mean(0) for c in sorted(set(y))])  # (C,256)
    M = mus - mus.mean(0, keepdims=True)
    Ub,_,_ = np.linalg.svd(M.T, full_matrices=False)           # (256,C)
    Ub = Ub[:, :k]
    return Ua, Ub, float((clf.predict(Xs)==y).mean())

def drs(w, U):
    w = w / (np.linalg.norm(w) + 1e-12)
    return float(np.sum((U.T @ w)**2))   # ||U^T w_hat||^2 in [0,1]

def spoof_w(sd):
    return sd["model"]["binary.fc.weight"].squeeze().float().cpu().numpy()  # (256,)

def ood_eers(sd):
    pe = sd.get("per_corpus_eer", {})
    return pe  # may only hold train-corpus dev; OOD comes from your crosstest logs

# ---- shared frozen embedding + subspace (from FROZEN_REF) ----
print(f"[1] building shared frozen embedding from {FROZEN_REF} ...", flush=True)
m, cfg, cv, _ = build_model_from_ckpt(CKPTS[FROZEN_REF])
Xf, yf = collect_embeddings(m, cfg, cv)
print(f"    embeddings {Xf.shape}, corpora {sorted(set(yf.tolist()))}")
Ua_f, Ub_f, probe_f = domain_subspace(Xf, yf)
print(f"    shared subspace ready; probe acc on this embedding = {probe_f:.4f}")
del m; torch.cuda.empty_cache() if DEVICE=="cuda" else None

# ---- per-checkpoint DRS ----
rows_out = []
for name, path in CKPTS.items():
    sd = torch.load(path, map_location="cpu", weights_only=False)
    w = spoof_w(sd)
    # e004 trained on a single corpus -> no own domain subspace is definable.
    # Measure its classifier against the SHARED frozen subspace (approximate:
    # e004 backbone is top-4 fine-tuned, so its embedding space differs slightly).
    Ua, Ub = Ua_f, Ub_f
    if name not in FROZEN_SET:
        print(f"[2] {name}: single-corpus / fine-tuned -> DRS vs SHARED frozen subspace (approx).", flush=True)
    rows_out.append((name, drs(w, Ua), drs(w, Ub)))
    print(f"    {name}: DRS_A={rows_out[-1][1]:.4f}  DRS_B={rows_out[-1][2]:.4f}", flush=True)

# ---- table ----
print("\n=== DOMAIN-RELIANCE SCORE TABLE ===")
print(f"{'model':8} {'DRS_A':>8} {'DRS_B':>8}")
for n,a,b in rows_out:
    print(f"{n:8} {a:8.4f} {b:8.4f}")

# save for joining with your probe/EER numbers
with open("runs/drs_results.json","w") as f:
    json.dump({n:{"DRS_A":a,"DRS_B":b} for n,a,b in rows_out}, f, indent=2)
print("\nsaved -> runs/drs_results.json")
print("\nNEXT: paste this table + your known probe acc and ITW/AI4T EER per model,")
print("and we compute Spearman(DRS, EER) vs Spearman(probe, EER).")
