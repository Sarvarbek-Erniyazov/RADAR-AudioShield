# Observed learned layer weighting (e007 checkpoints, XLS-R-300M)

Recorded 2026-07-19, BEFORE any Step 3 battery results existed.
Source: softmax(ssl.layer_logits) from each checkpoint's state dict.
Config init: layer_weight_init_center=10, band=[8,11]; the forward pass
softmaxes over all 25 hidden states, so the learned weighting was free to
move anywhere.

| checkpoint | entropy | peak layer (w) | 2nd | 3rd | mass in [8,11] |
|---|---|---|---|---|---|
| e007_A_fresh      | 1.335 | 11 (0.396) | 10 (0.258) | 9 (0.199) | 0.998 |
| e007_B_fresh      | 1.337 | 10 (0.330) | 11 (0.283) | 9 (0.268) | 0.999 |
| e007_C_xlsr_fresh | 1.229 |  9 (0.512) | 10 (0.204) | 8 (0.156) | 0.999 |

All three retained >99.8% of mass inside the initialization band; the 5th-ranked
layer carries ~0.000 weight in every checkpoint. The heads differ in peak depth
(11 / 10 / 9) and concentration (C is most concentrated).

Implication for Step 3: --layer 9 is inside every checkpoint's effective band
and is used for the shared factor metrics (r_var, LEACE/INLP, prediction_change,
controls). Alignment is computed separately per checkpoint using
--layer-mode checkpoint-band, which pools with each checkpoint's own learned
softmax weights, so w and the embedding live in the same representation.
