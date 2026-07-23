# Paper tables -- generated 2026-07-23T14:32:05.341Z

## Table 1 -- Cache-space factor encoding (FSS)

| Battery | Corpus | Factor | FSS (mean) | 95% CI | n_groups |
|---|---|---|---|---|---|
| diffssd_openvoicev2_accent_by_speaker | diffssd | language | 0.9922 | [0.9908, 0.9939] | 10 |

Only batteries with a readable, `ok` `headline_bootstrap` on this machine are listed here -- see this script's module docstring for which cache-space batteries exist only as prose elsewhere, not as files this script could open.

## Table 2 -- Model-space causal reliance per checkpoint

| Battery | Checkpoint | r_var | alignment | decision_flip_rate | exceeds_random |
|---|---|---|---|---|---|
| replaydf_generator_by_channel | e007_A_fresh | 1.58e-16 | 9.49e-08 | 0 | False |
| replaydf_generator_by_channel | e007_B_fresh | 1.96e-10 | 0.000385 | 0 | False |
| replaydf_generator_by_channel | e007_C_xlsr_fresh | 3.7e-11 | 5.81e-05 | 0 | False |
| replaydf_language_by_channel | e007_A_fresh | 3.92e-16 | 5.97e-08 | 0 | False |
| replaydf_language_by_channel | e007_B_fresh | 3.31e-09 | 9.69e-05 | 0 | False |
| replaydf_language_by_channel | e007_C_xlsr_fresh | 2.15e-10 | 9.32e-05 | 0 | False |

## Table 4 -- Model-space three-tier intervention effect (behavioral control: prediction_change_control)

| Battery | Estimator | Checkpoint | factor true_effect | random mean | random std | random mean+2std | task_direction_effect | main exceeds_random | n_folds |
|---|---|---|---|---|---|---|---|---|---|
| replaydf_generator_by_channel | lda | e007_A_fresh | 3.95e-07 | 0.0353 | 0.0465 | 0.128 | 6.918 | False | 5 |
| replaydf_generator_by_channel | lda | e007_B_fresh | 0.00102 | 0.0331 | 0.0275 | 0.0882 | 4.442 | False | 5 |
| replaydf_generator_by_channel | lda | e007_C_xlsr_fresh | 0.000302 | 0.034 | 0.0258 | 0.0857 | 2.54 | False | 5 |
| replaydf_generator_by_channel | probe | e007_A_fresh | 7.38e-08 | 0.0353 | 0.0465 | 0.128 | 6.918 | False | 5 |
| replaydf_generator_by_channel | probe | e007_B_fresh | 0.000198 | 0.0239 | 0.023 | 0.0698 | 4.442 | False | 5 |
| replaydf_generator_by_channel | probe | e007_C_xlsr_fresh | 3.07e-05 | 0.0175 | 0.0168 | 0.0511 | 2.54 | False | 5 |
| replaydf_language_by_channel | lda | e007_A_fresh | 8.21e-08 | 0.0348 | 0.0542 | 0.143 | 7.806 | False | 5 |
| replaydf_language_by_channel | lda | e007_B_fresh | 0.000283 | 0.026 | 0.0259 | 0.0777 | 5.233 | False | 5 |
| replaydf_language_by_channel | lda | e007_C_xlsr_fresh | 0.000159 | 0.0135 | 0.0149 | 0.0433 | 2.653 | False | 5 |
| replaydf_language_by_channel | probe | e007_A_fresh | 1.01e-07 | 0.0348 | 0.0542 | 0.143 | 7.806 | False | 5 |
| replaydf_language_by_channel | probe | e007_B_fresh | 0.000501 | 0.0519 | 0.0393 | 0.131 | 5.233 | False | 5 |
| replaydf_language_by_channel | probe | e007_C_xlsr_fresh | 0.000152 | 0.0389 | 0.0327 | 0.104 | 2.653 | False | 5 |

The factor `true_effect` sits far below `random mean+2std` (so `main exceeds_random` is False everywhere) while `task_direction_effect` towers over the same bar -- the positive control is alive, the factor reliance is genuinely absent. This is the evidence behind C4's fail-WITH-live-control verdict (docs/gate_prereg.md 2026-07-23 #2).

## Table 5 -- Two-space factor decodability (cache-space FSS vs model-space, per fold)

| Battery | Estimator | Fold | cache-space FSS | model-space FSS | model-space LEACE decodability_before |
|---|---|---|---|---|---|
| replaydf_generator_by_channel | lda | 0 | 0.889 | 0.5059 | 0.365 |
| replaydf_generator_by_channel | lda | 1 | 0.889 | 0.5167 | 0.3917 |
| replaydf_generator_by_channel | lda | 2 | 0.889 | 0.5638 | 0.3118 |
| replaydf_generator_by_channel | lda | 3 | 0.889 | 0.4727 | 0.4006 |
| replaydf_generator_by_channel | lda | 4 | 0.889 | 0.5496 | 0.3641 |
| replaydf_generator_by_channel | probe | 0 | 0.889 | 0.3799 | 0.365 |
| replaydf_generator_by_channel | probe | 1 | 0.889 | 0.3648 | 0.3917 |
| replaydf_generator_by_channel | probe | 2 | 0.889 | 0.3401 | 0.3118 |
| replaydf_generator_by_channel | probe | 3 | 0.889 | 0.3655 | 0.4006 |
| replaydf_generator_by_channel | probe | 4 | 0.889 | 0.4091 | 0.3641 |
| replaydf_language_by_channel | lda | 0 | 0.949 | 0.8503 | 0.3226 |
| replaydf_language_by_channel | lda | 1 | 0.949 | 0.8401 | 0.2697 |
| replaydf_language_by_channel | lda | 2 | 0.949 | 0.8017 | 0.2984 |
| replaydf_language_by_channel | lda | 3 | 0.949 | 0.8384 | 0.3368 |
| replaydf_language_by_channel | lda | 4 | 0.949 | 0.8188 | 0.3204 |
| replaydf_language_by_channel | probe | 0 | 0.949 | 0.6698 | 0.3226 |
| replaydf_language_by_channel | probe | 1 | 0.949 | 0.5397 | 0.2697 |
| replaydf_language_by_channel | probe | 2 | 0.949 | 0.5729 | 0.2984 |
| replaydf_language_by_channel | probe | 3 | 0.949 | 0.6659 | 0.3368 |
| replaydf_language_by_channel | probe | 4 | 0.949 | 0.5905 | 0.3204 |

**Cache-space FSS provenance:** the cache-space FSS column for the ReplayDF batteries is carried from CURRENT_STATE.md prose (collaborator-machine bootstrap, not committed here) -- the per-battery cache-space bootstrap JSON is not committed on this machine (only the DiffSSD accent battery in Table 1 is). Each row's provenance is recorded in `cachespace_fss_source` in the JSON output; these values are never presented as file-derived.

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
