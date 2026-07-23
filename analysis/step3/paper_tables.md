# Paper tables -- generated 2026-07-23T13:36:54.031Z

## Table 1 -- Cache-space factor encoding (FSS)

| Battery | Corpus | Factor | FSS (mean) | 95% CI | n_groups |
|---|---|---|---|---|---|
| diffssd_openvoicev2_accent_by_speaker | diffssd | language | 0.9922 | [0.9908, 0.9939] | 10 |

Only batteries with a readable, `ok` `headline_bootstrap` on this machine are listed here -- see this script's module docstring for which cache-space batteries exist only as prose elsewhere, not as files this script could open.

## Table 2 -- Model-space causal reliance per checkpoint

| Battery | Checkpoint | r_var | alignment | decision_flip_rate | exceeds_random |
|---|---|---|---|---|---|
| replaydf_language_by_channel | e007_A_fresh | 3.92e-16 | 5.97e-08 | 0 | False |
| replaydf_language_by_channel | e007_B_fresh | 3.31e-09 | 9.69e-05 | 0 | False |
| replaydf_language_by_channel | e007_C_xlsr_fresh | 2.15e-10 | 9.32e-05 | 0 | False |
| replaydf_generator_by_channel | e007_A_fresh | 1.58e-16 | 9.49e-08 | 0 | False |
| replaydf_generator_by_channel | e007_B_fresh | 1.96e-10 | 0.000385 | 0 | False |
| replaydf_generator_by_channel | e007_C_xlsr_fresh | 3.7e-11 | 5.81e-05 | 0 | False |

## Table 3 -- Step 4 gate verdict

| Criterion | Status | Evidence |
|---|---|---|
| C1 | pending_input | no second-backbone (WavLM-Large) battery input provided -- pass --phase-a-secondary-backbone once the collaborator machine's WavLM-Large embedding pass produces it |
| C2 | fail | per-battery association results: {'replaydf_language_by_channel': 'pass', 'replaydf_generator_by_channel': 'fail'} |
| C3 | pass | 2/2 decided batteries have a usable grouped bootstrap (status=ok, n_boot>=1000, n_groups>=8, non-degenerate grouping); 0 excluded/operational-gap |
| C4 | fail | per-checkpoint main-effect-vs-random and positive-control results, UNANIMOUS across checkpoints (docs/gate_prereg.md C4 amendment, 2026-07-22) -- see per_battery[name].per_checkpoint for each checkpoint's own verdict and per_battery[name].verdict_summary for a legible pass/fail breakdown |
| C5 | pass | 2/2 decided batteries have a stable rank window (>=2 consecutive ranks, including headline rank 1); 0 excluded/operational-gap |
| C6 | pass | 2/2 decided batteries show estimators_agree_sign=True (LDA-subspace vs cross-fitted linear-probe); 0 excluded/operational-gap |
| C7 | fail | clean-EER-residualized sign-survival per battery (n=3-ish checkpoints -- descriptive, not a powered significance test; see docs/gate_prereg.md §4 ambiguity 3) |
| C8 | pending_input | no head-replicate result file provided -- run scripts/head_replicates.py against the real cached embeddings (collaborator machine) and pass --head-replicates |

**Overall classification:** None
