# Domain-Reliance Score (DRS) Analysis

## Motivation

The corpus probe answers *"is corpus information present in the embedding?"* — and across all six frozen models it stays at ~0.96–0.97. But high decodability does not prove the spoof classifier *uses* that information to decide real vs. fake. The Domain-Reliance Score asks the sharper question: **does the decision boundary rely on corpus-specific directions?**

This distinguishes our work from domain-invariance methods (which suppress corpus information from the representation). We instead measure, and propose to control, *reliance* while leaving corpus information intact.

## Definition

Let `w` be the spoof classifier's weight vector and `U` an orthonormal basis for the domain subspace (recovered from a corpus probe on the same embedding). The score is the fraction of the classifier's weight energy lying in the domain subspace:

```
DRS(w) = ||Uᵀ w||² / ||w||²        ∈ [0, 1]
```

- 0 → boundary orthogonal to the domain subspace (no linear reliance)
- 1 → boundary lies entirely in the domain subspace (maximal reliance)

The domain subspace `U` is estimated two ways for robustness: (A) orthonormalized corpus-probe weight rows; (B) SVD of centered per-corpus mean embeddings. DRS_A is treated as primary. (DRS captures *linear* reliance; non-linear reliance is out of scope for the linear head.)

## Result on the six frozen checkpoints

| Model | DRS_A | Probe acc | ITW EER | AI4T EER |
|-------|:-----:|:---------:|:-------:|:--------:|
| e002-A | 0.0137 | 0.962 | 0.127 | 0.417 |
| e002-B | 0.0041 | 0.960 | 0.149 | 0.484 |
| e004 | 0.0085 | 0.964 | 0.192 | 0.461 |
| e005-A | 0.0024 | 0.967 | 0.143 | 0.231 |
| e005-C | 0.0012 | 0.973 | 0.122 | 0.297 |
| e006 | 0.0026 | 0.972 | 0.135 | 0.260 |

(e004 is single-corpus, so it has no own multi-corpus subspace; its DRS is measured against the shared frozen subspace and is approximate. A secondary estimator, DRS_B, showed a numerical instability on e005-A and is not reported here pending a fix.)

## Interpretation

**Reliance is low and flat across all frozen models** — only 0.1–1.4% of the classifier's weight energy sits in the domain subspace, even though corpus identity is highly decodable (probe ~0.96). This directly explains the decoupling finding: the classifier was barely using the decodable corpus directions in the first place.

With n=6 there is no statistically meaningful correlation between DRS and OOD EER (significance at n=6 would require a rank correlation near 0.83+). The honest reading is therefore: **in the frozen regime, reliance is near-zero and is not the lever.**

## Decision

- The **Domain-Reliance Score** is retained as a *diagnostic contribution*: it measures reliance directly, which prior audio-deepfake work does not, and reveals decodability and reliance are different quantities.
- The **Domain-Reliance Loss** is **not** justified on frozen models — there is almost no reliance to regularize. It is moved to the **fine-tuning phase (e007)**, where the backbone can adapt and reliance may emerge (e004, the one fine-tuned model, already shows higher DRS than the best frozen multi-corpus models). e007 will log DRS per epoch; the loss is tested only if fine-tuning drives reliance up.

## Reproduce

```bash
python -u scripts/compute_drs.py     # writes drs_results.json here
```
