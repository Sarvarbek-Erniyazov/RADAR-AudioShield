# AudioShield / AI Voice Detection: Research, Novelty, and Publication Audit

**Audit date:** 2026-07-10  
**Audited state:** local `code_repo` working tree on branch `e007-optimized-layerband`, including ignored local `runs/` artifacts and uncommitted e007 changes  
**Scope:** source, manifests, experiment records, local e007 outputs, leakage analysis, reproducibility, and literature through July 2026

## Executive verdict

This is a serious research prototype addressing an important problem. Its strengths include cross-corpus evaluation, attention to bona-fide domain shift, a leakage audit, clustered uncertainty for segmented AI4T audio, and honest negative results. It is substantially stronger than a typical single-dataset classifier project.

However, **the current work is not ready for a top-conference submission**.

| Dimension | Rating | Reason |
|---|---:|---|
| Problem importance | **8/10** | Generalizable speech-deepfake detection is timely and security-relevant. |
| Demonstrated novelty | **3/10** | The trained system is mostly standard SSL features, pooling, multi-corpus sampling, augmentation, and SupCon. |
| Potential core novelty | **5/10** | “Decodability is not reliance” could be valuable, but the present DRS experiment is invalid. |
| Technical soundness | **3/10** | DRS has coordinate-system and confounding errors; several claimed modules are inactive. |
| Empirical rigor | **4/10** | Full OOD tests help, but there is one run per condition, weak baselines, and test-guided iteration. |
| Reproducibility | **3/10** | Seeds are not applied, dependencies are incomplete, tests are absent, and key results are ignored/untracked. |
| Top-conference readiness | **2/10 — strong reject** | The central claim is unsupported and named-dataset results trail published baselines. |

### Bottom line

- **Novelty now:** low to low-moderate.
- **Publishability now:** not suitable for a main track at NeurIPS, ICML, ICLR, AAAI, ACL, ACM MM, ICASSP, or Interspeech.
- **After cleanup only:** internal report, reproducibility/negative-results note, or possibly a workshop submission.
- **After major scientific revision:** potentially Interspeech/ICASSP or an applied security/multimedia venue.
- **For a top general ML venue:** the work needs a valid, general reliance framework with causal validation, stronger theory or controlled evidence, multiple tasks/domains, modern blind benchmarks, and stronger results.

The most promising story is not “a new detector architecture,” but:

> Domain information can remain highly decodable without being used by a detector; representation invariance is therefore an insufficient proxy for decision reliance. A valid reliance measure and reliance-targeted intervention should predict and improve failure under factorized domain shifts.

The repository does not yet prove this claim.

## 1. What is actually implemented and evaluated

The active experimental path is:

1. WavLM-Large or XLS-R-300M returns all hidden states.
2. A learned softmax-weighted sum combines hidden states, initialized around layers 8–11.
3. Attentive statistics pooling produces temporal mean and standard deviation.
4. A two-layer projection maps the feature to 256 dimensions.
5. A linear head predicts spoof versus bona fide.
6. Training uses BCE on clean and degraded views, optional detached-teacher consistency, and optional cross-corpus SupCon.
7. Multi-corpus runs sample ASVspoof5, DiffSSD, Fake-or-Real, and VCTK.
8. ITW, ReplayDF, and AI4T are used as OOD corpora.

Key implementation links:

- [SSL layer weighting](src/audioshield/models/ssl_backbone.py)
- [Detector](src/audioshield/models/detector.py)
- [Pooling](src/audioshield/models/pooling.py)
- [Active objective](src/audioshield/training/loop_e002.py)
- [Cross-corpus SupCon](src/audioshield/losses/xc_contrastive.py)
- [Training entry point](scripts/train_e002.py)
- [Evaluation](src/audioshield/evaluation/cross_test.py)

### Implemented does not mean evaluated

The repository also contains [BMI](src/audioshield/losses/bmi.py), [prototype losses](src/audioshield/losses/prototype.py), [CALAS](src/audioshield/losses/latent_aug.py), and a [BMI sampler](src/audioshield/data/samplers.py). These are **not called by the active reported e002–e007 training loop**. The evaluated objective is classification + consistency + optional SupCon. BMI, CALAS, AAM/prototype losses, and prototype regularizers therefore cannot be claimed as empirical contributions.

## 2. Reconstructed results, including undocumented e007 runs

All values are EER; lower is better. “Mean” is an unweighted arithmetic mean across the three OOD corpora and is only a compact summary.

| Run | Main change | ITW | ReplayDF | AI4T | Mean | Evidence status |
|---|---|---:|---:|---:|---:|---|
| e003 | Frozen WavLM linear baseline | **0.111** | 0.317 | 0.470 | 0.299 | Number only in `RESULTS.md`; artifact absent |
| e002-A | Neural head + channel augmentation, frozen | 0.1267 | 0.2956 | 0.4174 | 0.280 | Tracked log |
| e002-B | + consistency | 0.149 | 0.367 | 0.484 | 0.333 | Summary only; cross-test log absent |
| e004 | Single-corpus top-4 fine-tuning | 0.1921 | **0.2730** | 0.4611 | 0.309 | Tracked log |
| e005-A | Frozen, four-corpus sampling | 0.1431 | 0.2976 | **0.2306** | **0.224** | Strongest tracked overall run |
| e005-C | + spoof positive weight | 0.1219 | 0.3253 | 0.2970 | 0.248 | Tracked log |
| e006 | + cross-corpus SupCon, frozen | 0.135 | ~0.298 | 0.260 or ~0.231 | ~0.231 | Inconsistent summaries; log absent |
| e007-A fresh | Weighted-band WavLM fine-tuning, BCE | 0.1805 | 0.3327 | 0.2565 | 0.257 | Ignored local JSON; epoch 20 |
| e007-B fresh | e007-A + projected SupCon | **0.1167** | 0.3276 | 0.2629 | 0.236 | Ignored local JSON; epoch 8 |
| e007-C fresh | XLS-R-300M + projected SupCon | 0.2009 | 0.4530 | 0.3435 | 0.332 | Ignored local JSON; epoch 11 |

Local e007 evidence: [e007-A](runs/e007_A_fresh_crosstest.json), [e007-B](runs/e007_B_fresh_crosstest.json), and [e007-C](runs/e007_C_xlsr_fresh_crosstest.json). These files are ignored by `**/runs/`, so they are not in the shareable Git state.

### Supported conclusions

1. **Multi-corpus exposure is the clearest positive result.** Relative to e002-A, e005-A changes mean EER from about 28.0% to 22.4%, mainly through AI4T (41.7% to 23.1%).
2. **There is no uniformly best detector.** e003 wins ITW, e004 wins ReplayDF, and e005-A wins AI4T.
3. **Consistency regresses** on all three reported OOD corpora.
4. **Naive fine-tuning is not a general solution.** e007-A is worse than e005-A everywhere.
5. **Projected SupCon is domain-specific.** e007-B sharply helps ITW versus e007-A, barely changes ReplayDF, and slightly worsens AI4T.
6. **The XLS-R swap is a negative result.** e007-C is substantially worse than e007-B on all three corpora.
7. **Operational calibration is poor.** e007 ECE is roughly 0.15–0.45, while ReplayDF/AI4T balanced accuracy is often only 0.55–0.60.

### Unsupported conclusions

- The results do not show state of the art or statistically stable gains across seeds.
- They do not establish that SupCon improves aggregate OOD generalization.
- They do not establish a causal separation between domain decodability and reliance.
- They do not validate BMI, CALAS, prototypes, or reliance regularization.
- They are not final untouched-test results, because the OOD outcomes informed later design decisions.

## 3. The current Domain-Reliance Score is invalid

The intended score is

\[
\mathrm{DRS}(w,U)=\frac{\|U^T w\|_2^2}{\|w\|_2^2},
\]

where `w` is the binary classifier weight and `U` spans domain directions. The idea is promising; the implementation in [compute_drs.py](scripts/compute_drs.py) has fatal issues.

### 3.1 Incompatible coordinate systems

The script builds one shared subspace from e005-A, then projects weights from other models into it. Even with frozen WavLM, every model has independently trained layer weights, pooling, projection, and head. Their 256-D bases are not shared. A weight from one model has no meaningful angle with a subspace from another unless the representations are explicitly aligned.

### 3.2 Standardized versus raw coordinates

The domain probe is trained on `StandardScaler().fit_transform(X)`, so its coefficients live in standardized coordinates. The spoof-head weight lives in raw coordinates. If the feature scales are `sigma`, a standardized coefficient `w_s` maps to a raw normal proportional to `w_s / sigma`. The present direct projection is scale-dependent and uninterpretable.

### 3.3 Class-confounded “domain” subspace

`collect_embeddings()` uses all validation samples. VCTK is bona-fide-only, while other corpora have both classes, so domain prediction can exploit authenticity. The domain subspace can therefore include the exact task direction whose overlap DRS is supposed to measure.

### 3.4 Contradictory estimator

The checked-in JSON gives e005-A `DRS_A = 0.0024` but `DRS_B = 0.9002`. This is a qualitative contradiction, not a minor instability. Suppressing DRS_B does not validate DRS_A.

### 3.5 Inadequate inference

Six heterogeneous checkpoints, with no training replicates, cannot establish that reliance and OOD error are decoupled. “DRS explains the decoupling” is too strong.

### Required correction

For each checkpoint independently:

1. Extract balanced, class-controlled embeddings from that checkpoint.
2. Estimate the domain subspace in the same coordinates.
3. Cross-fit subspace estimation and reliance measurement.
4. Back-transform standardized coefficients or define an explicit whitened/covariance metric.
5. Use source-grouped resampling.
6. Validate causally by removing/swapping domain components and measuring logit, EER, and calibration changes.

For nonlinear heads, use an averaged projected decision-gradient score:

\[
\mathbb{E}_z\left[\frac{\|U^T \nabla_z f(z)\|_2^2}{\|\nabla_z f(z)\|_2^2}\right].
\]

For a linear head this reduces to a weight score, while extending to stronger back ends.

## 4. Experimental-design limitations

### Critical limitations

#### 4.1 The OOD test corpora have become development corpora

Every intervention is evaluated on ITW, ReplayDF, and AI4T, and the outcomes inform subsequent choices—for example, retaining `spoof_pos_weight=1.0` after the e005-C AI4T trade-off. The files never enter gradient training, but the test results influence model design. This is experiment-level test leakage.

Use three tiers:

- **Development domains:** available for iteration.
- **Validation domains:** used only for model-family selection.
- **Final blind domains:** evaluated once after freezing code and hypotheses, ideally through a new temporal cohort or challenge server.

#### 4.2 Single uncontrolled runs cannot support comparisons

No condition has multiple seeds. Although configs contain `seed: 13`, the active script never seeds Python, NumPy, Torch, workers, or the sampler. Initialization, weighted sampling, crop selection, and augmentation are uncontrolled.

Run at least five seeds for final conditions. Report mean, standard deviation, and paired source-cluster bootstrap intervals. A test-sample CI around one checkpoint does not capture training instability.

#### 4.3 Baselines are not competitive enough

The paper needs common-protocol versions of:

- frozen WavLM/XLS-R + logistic regression;
- W2V2-AASIST with RawBoost/RIR;
- canonical AASIST/RawNet2;
- SAM;
- GRL/CDANN, CORAL/MMD, GroupDRO, and class-conditional SupCon;
- a current data-centric mixture baseline;
- a recent SSL benchmark or audio-LLM reference where feasible.

Internal e002/e005 comparisons do not establish competitiveness.

#### 4.4 Class and corpus are not factorized

Training counts reconstructed from manifests are:

| Corpus | Train bona fide | Train spoof | Issue |
|---|---:|---:|---|
| ASVspoof5 | 18,797 | 163,560 | Strong imbalance |
| DiffSSD | 9,690 | 22,000 | Both classes |
| Fake-or-Real | 26,941 | 26,927 | Both classes |
| VCTK | 35,323 | 0 | Corpus perfectly predicts bona fide |

The sampler equalizes corpus, then class within corpus. Because VCTK has one class, the expected global mix is about 62.5% bona fide and 37.5% spoof—not 50/50. VCTK also creates label–corpus confounding. Use matched synthetic generations for each real source, or exclude bona-only sources from class-conditional domain objectives.

### High-priority limitations

#### 4.5 Leakage audit scope is narrower than claimed

The audit usefully finds no exact-byte or peak-normalized 16 kHz waveform collision. It does not yet prove speaker/source disjointness:

- manifests contain no speaker column, so the script cannot verify speaker IDs;
- embedding near-duplicate checks are pending;
- segment fingerprints and speaker-embedding checks are pending;
- re-synthesized, trimmed, or heavily processed common sources can evade hashing.

The defensible claim is only: “no exact or normalized-waveform collision was found.”

#### 4.6 Bootstrap clustering is adequate only for AI4T

AI4T suffixes are collapsed into 276 source-video clusters, which is good. ITW and ReplayDF use utterance-level clusters. ReplayDF contains related device/room recordings derived from common sources, so its intervals are likely too narrow. Add `source_id`, `speaker_id`, `generator_id`, `channel_id`, and `platform_id`, and cluster at the highest dependence level.

#### 4.7 The corpus probe is optimistic

The probe uses ordinary three-fold CV over segments/utterances, allowing related items across folds. It prints chance as `1/3`, despite severe domain imbalance; the majority baseline is much higher. Use grouped stratified folds and balanced accuracy or macro-F1.

#### 4.8 Modern robustness axes are missing

The evaluation lacks systematic breakdowns for unseen generator/release date, language/accent, speaker/demographics, neural codecs and social-platform laundering, SNR/reverb, adversarial attacks, short/streaming audio, partial/spliced fakes, silence removal, and worst-group risk. The manifest already has `attack`, but it is not used in reporting.

#### 4.9 One centered four-second crop is incomplete

Center-cropping long audio can miss partial manipulations; zero-padding short audio can create duration/silence shortcuts. Add multi-window aggregation, duration strata, voiced-only controls, and localization where relevant.

#### 4.10 Calibration is measured but not solved

OOD ECE is high and the dev threshold transfers poorly. Report Brier score, NLL, Cllr/actDCF where applicable, calibrated and uncalibrated results, source-grouped intervals, and selective risk with abstention. Calibration must be fitted only on development domains.

## 5. Source-code and reproducibility audit

### Positive findings

- `python -m compileall -q src scripts` passes.
- Newer checkpoints store resolved configs.
- The active config loader now deep-merges overrides.
- Weighted-band fine-tuning is more principled than blindly unfreezing final layers.
- Evaluation warns when data are capped.
- AI4T source-cluster bootstrap exists.
- Manifests and historical logs provide useful provenance.

### Major issues

| Severity | Finding | Impact | Fix |
|---|---|---|---|
| Critical | DRS shares a subspace across independently trained projection spaces | Invalid central values | Recompute per checkpoint or align spaces |
| Critical | Standardized probe weights are compared with raw head weights | Invalid angle | Back-transform or define a whitened metric |
| High | Fresh e007 JSONs/checkpoints are ignored | Best current evidence is not shareable | Commit lightweight result bundles and publish checkpoint hashes |
| High | README/`RESULTS.md` omit completed e007 | Public narrative is stale | Generate one canonical table from machine-readable artifacts |
| High | e006/e002-B logs are absent and e006 AI4T is reported two ways | Internal inconsistency | Recover scores or rerun |
| High | e003 artifact is absent | Headline ITW baseline is unverifiable | Commit config, sample hash, scores, and metrics |
| High | Config seed is never applied | Non-reproducible single runs | Seed all RNGs/workers/sampler and log settings |
| High | `scipy` is imported but absent from requirements | Fresh install failure | Add/pin SciPy and create lock/container |
| High | Dependencies/model revisions are unpinned; WavLM load warnings appear | Environment-dependent model behavior | Pin Torch, Transformers, and HF revision; test strict loading |
| High | Tests contain only `__init__.py` | No regression protection | Add unit/integration tests |
| High | Two training entry points have diverged | Invocation-dependent behavior | Keep one canonical entry point |
| Medium | RIR path is hard-coded; missing RIRs silently make reverb a no-op | Same config differs by machine | Configure/fingerprint assets; fail or clearly record absence |
| Medium | Loader silently substitutes one of the next 49 rows on read error | Requested sampling/label distribution can change | Prevalidate or fail fast; log every substitution |
| Medium | SupCon `min_corpora_per_class` is unused | Guard is not enforced | Implement/test or remove |
| Medium | EER implementation does full-array work at every threshold | Quadratic evaluation | Use a single ROC sort |
| Medium | README links a missing `docs/e007_finetuning_design.md` | Broken documentation | Restore or remove |
| Medium | Experiment README promises artifacts several runs lack | Inaccurate reproducibility claim | Add artifact validation in CI |
| Medium | Resolved configs/logs contain absolute local paths | Poor portability/privacy hygiene | Store logical IDs plus local overrides |

At audit time the branch is ahead of `origin/main`, with modified source/configs and an untracked XLS-R config. A paper release needs a clean tagged commit whose code, manifest hashes, environment, and result bundle agree.

## 6. Recent state of the art versus this project

This table focuses on directly relevant work. Values are **not one common leaderboard** because protocols differ.

| Work | Venue/status | Contribution | Implication here |
|---|---|---|---|
| [Pascu et al., calibrated SSL](https://www.isca-archive.org/interspeech_2024/pascu24_interspeech.html) | Interspeech 2024 | Frozen large SSL + logistic regression; mean EER 30.9% → 8.8% over eight datasets with <2k trained parameters | Frozen SSL + simple head is established; e003 needs exact reproduction under this protocol |
| [Li et al., Cross-Domain ADD](https://aclanthology.org/2024.emnlp-main.286/) | EMNLP 2024 | 300+ hours from five zero-shot TTS systems; attack augmentation; 4.1% Wav2Vec2 and 6.5% Whisper EER in their protocol; identifies codec weakness | Modern work contributes both controlled data and generator/channel shifts |
| [Xie et al., Codecfake](https://arxiv.org/abs/2405.04880) | 2024/2025 line | >1M English/Chinese samples focused on neural-codec/ALM generation; CSAM balancing | Neural-codec-specific evaluation is missing here |
| [Ren et al., Improving Generalization](https://ojs.aaai.org/index.php/AAAI/article/view/34221) | AAAI 2025 | Disentanglement, reconstruction, contrastive learning, mutual information, SAM, seen/unseen vocoder ablations | Cross-corpus SupCon alone is not novel enough |
| [Combei et al., real-world data-centric ADD](https://arxiv.org/abs/2506.09606) | Interspeech 2025 | Curation/pruning/augmentation; 1.7% ITW EER and 63% relative AI4T reduction; dataset mixing reaches 13.32% AI4T before later refinements | Best project results (11.1% ITW, 23.1% AI4T) are directionally behind; rerun common protocols before formal comparison |
| [Müller et al., ReplayDF](https://arxiv.org/abs/2505.14862) | Interspeech 2025 | 109 speaker–microphone combinations, six languages, four TTS models; W2V2-AASIST 18.2%, adaptive RIR 11.0% | Project best is 27.3% ReplayDF; current replay robustness is not competitive |
| [Huang et al., SpeechFake](https://aclanthology.org/2025.acl-long.493/) | ACL 2025 long | >3M deepfakes, >3,000 hours, 40 tools, 46 languages | Four-corpus training is not a data-scale novelty |
| [Kwok et al., Bona fide Cross Testing](https://www.isca-archive.org/interspeech_2025/kwok25_interspeech.html) | Interspeech 2025 | >150 synthesizers × nine bona-fide types; average EER near 10% hides worst-case >30% | The three-column project matrix is useful but borrowed and small |
| [Huang et al., sharpness and SAM](https://www.isca-archive.org/interspeech_2025/huang25e_interspeech.html) | Interspeech 2025 | Links loss sharpness to domain shift and improves robustness with SAM | A strong paper needs a validated explanatory mechanism |
| [Yang et al., Poin-HierNet](https://www.isca-archive.org/interspeech_2025/yang25l_interspeech.html) | Interspeech 2025 | Hyperbolic prototypes, hierarchy, feature whitening | Prototype/domain-invariance space is already crowded |
| [Nguyen et al., linguistic sensitivity](https://aclanthology.org/2025.emnlp-main.794/) | EMNLP 2025 | Linguistic adversarial attack and attribution; acoustic detectors can be bypassed | The project has no linguistic/adversarial analysis |
| [ASVspoof 5 evaluation](https://arxiv.org/abs/2601.03944) | 2026 challenge summary | 53 teams; adversarial attacks and neural compression remain hard; calibration emphasized | The project trains on ASVspoof5 but reports no official eval/minDCF/Cllr result |
| [Ciobanu et al., XMAD-Bench](https://aclanthology.org/2026.findings-eacl.162/) | Findings EACL 2026 | 668.8 hours; speakers, generators, and real sources disjoint; cross-domain can approach chance | “Cross-corpus” is insufficient without factor-level disjointness |
| [Ali et al., Spoof-SUPERB](https://arxiv.org/abs/2603.01482) | ICASSP 2026 accepted | Unified 20-SSL comparison; XLS-R mean 17.4%, WavLM Large 20.6%, XLS-R codec EER 13.5% | A common-protocol suite is needed; e007-C failure is training-specific, not a general XLS-R verdict |
| [Huang et al., data-centric SDD](https://aclanthology.org/2026.acl-long.796/) | ACL 2026 long | Scaling laws and DOSS; 3% of data beats naive aggregation; final 12k-hour curated pool | Equal corpus mixing is behind current data-centric novelty |
| [Li et al., ML-ITW](https://arxiv.org/abs/2603.05852) | 2026 preprint | 14 languages, seven platforms, 180 public figures; end-to-end, SSL, and audio-LLM systems degrade | Multilingual/platform evaluation is now expected |
| [Chou et al., ICLAD](https://aclanthology.org/2026.findings-acl.450/) | Findings ACL 2026 | OOD routing to an audio LM with comparison-guided reasoning and explanations; up to 2× relative macro-F1 gain | OOD routing/explainability is an uncovered frontier |
| [AT-ADD](https://www.at-add.com/) | ACM MM 2026 challenge | Robust speech plus all-type audio across speech, sound, singing, and music | The field is expanding beyond utterance-level binary speech |

### Direct performance perspective

The project’s best named-dataset EERs are 11.1% ITW, 27.3% ReplayDF, and 23.1% AI4T, achieved by three different models. Relevant papers report much lower values on those datasets—1.7% ITW, 18.2%/11.0% ReplayDF, and 13.32% AI4T in their respective protocols. Preprocessing, train data, segmentation, and aggregation can differ, so this is **not a formal leaderboard comparison**. It is enough to rule out a SOTA claim and makes common-protocol reproduction urgent.

## 7. Novelty by component

| Component | Novelty | Value now |
|---|---|---|
| Learnable SSL layer weighting | Low | Common downstream technique |
| Attentive statistics pooling | Low | Established audio/speaker technique |
| Frozen WavLM/XLS-R | Low | Established baseline family |
| Channel augmentation | Low | Standard and less complete than RawBoost/RIR/codec recipes |
| Detached consistency | Low | Standard and empirically regresses |
| Multi-corpus balancing | Low | Standard; current sampler is confounded and less principled than DOSS-like selection |
| Class-conditional cross-corpus SupCon | Low-moderate | Reasonable, but heavily explored and mixed here |
| Weighted-band fine-tuning | Low | Engineering heuristic with mixed/negative results |
| Kwok matrix | No method novelty | Useful adoption of prior evaluation |
| Cluster bootstrap | Low novelty, high rigor | Good practice, not an algorithmic contribution |
| Decodability vs reliance / DRS | Moderate potential | Most distinctive idea, but not validly demonstrated |
| BMI/CALAS/prototypes | Not demonstrated | Inactive code cannot support a claim |

## 8. Recommended novelty direction: Reliance-Aware Domain Generalization

Build a framework that separates:

1. **Presence:** can domain be decoded?
2. **Reliance:** does the decision change along domain directions?
3. **Robustness:** does reliance predict failure on held-out factor combinations?

### A. Valid metric

- Learn class-conditional domain subspaces with cross-fitting.
- Define linear and gradient-based reliance.
- Make it invariant to feature scaling/reparameterization, or specify the geometry.
- Quantify rank sensitivity and uncertainty.

### B. Causal validation

- Remove domain projections.
- Swap domain components between matched examples.
- Perturb only domain directions at controlled magnitude.
- Compare with equal-norm random and task-direction controls.

A reliance metric is convincing only if it predicts logit/EER/calibration sensitivity and real held-out failures.

### C. Reliance-control objective

Penalize decision sensitivity along domain directions rather than erasing all domain information. Compare with GRL/CDANN, CORAL/MMD, GroupDRO/IRM, SupCon, SAM, INLP/concept erasure, and no-regularization/data controls.

### D. Factorized evaluation

Construct `real source × speaker × language × generator × codec/channel × platform`. Evaluate unseen generator, unseen channel, unseen language, compound shifts, and a temporal holdout of newer/commercial generators. This makes the reliance claim falsifiable and relevant beyond one dataset naming scheme.

### Alternative: replay/media robustness

If the reliance program is too large, focus on ReplayDF/RADAR-style transformations: measured or differentiable speaker–microphone simulation, neural codecs, platform transcoding, source-grouped worst-device evaluation, paired clean/replay consistency, and streaming constraints. This direction must beat W2V2-AASIST and adaptive RIR baselines on the official protocol.

## 9. Minimum experiment package for a credible paper

### Baselines

1. Frozen WavLM Large + logistic regression.
2. Frozen XLS-R + logistic regression.
3. W2V2-AASIST + RawBoost/RIR.
4. Current e005-A neural head.
5. Current e007-A fine-tuning.
6. SupCon.
7. GRL/CDANN.
8. CORAL/MMD.
9. SAM.
10. Proposed reliance loss.

### Ablations

- domain estimator: probe weights vs between-domain covariance vs learned directions;
- subspace rank and raw vs whitened geometry;
- bona-only vs class-conditional estimation;
- decodability loss vs reliance loss;
- frozen vs partial vs adapters/LoRA/full tuning;
- with/without VCTK and with matched VCTK synthetic data;
- sampler/data mixture;
- clean, channel, codec, and replay augmentation;
- one window vs multi-window aggregation;
- SupCon projection head and temperature.

### Evaluation sets

At minimum: ASVspoof5 official eval, ITW, ReplayDF official protocol, AI4T official protocol, XMAD-Bench, Codecfake/current ALM data, a multilingual/platform set such as ML-ITW when licensing permits, and a genuinely blind temporal/commercial-generator set.

### Metrics

- EER with source-clustered 95% CI;
- minDCF and Cllr/actDCF where applicable;
- AUROC/AUPRC and TPR at low FPR;
- Brier, NLL, ECE/ACE;
- worst-generator/language/channel/intersection EER;
- selective risk/coverage;
- parameters, FLOPs, real-time factor, memory, and latency;
- paired tests against the strongest baseline;
- mean ± SD over at least five seeds.

## 10. Prioritized execution plan

### P0 — Repair validity before more large runs

1. Stop using ITW/ReplayDF/AI4T for hyperparameter decisions; treat them as exhausted development domains.
2. Fix DRS coordinates, scaling, and class confounding; rerun per checkpoint.
3. Add seed and deterministic worker handling.
4. Fix the sampler distribution and VCTK confounding.
5. Restore exact e003/e002-B/e006 artifacts or mark them unverifiable.
6. Commit lightweight e007 metrics/config/training summaries, score hashes, and environment metadata.
7. Pin dependencies/model revisions and add SciPy.
8. Add tests for EER, bootstrap grouping, sampler distribution, config merge, SupCon edges, and DRS invariances.

### P1 — Establish trustworthy baselines

1. Reproduce Pascu-style frozen SSL and W2V2-AASIST under official/common protocols.
2. Give all models identical preprocessing and score aggregation.
3. Add source and factor metadata to manifests.
4. Produce generator/language/channel/platform breakdowns.
5. Measure operational calibration.

### P2 — Test the hypothesis cheaply

Before another 300M-model run, use saved embeddings to test corrected DRS, domain-component removal, equal-norm controls, and whether reliance predicts cross-factor degradation. If this fails, abandon DRS as the main claim. If it succeeds, implement a reliance loss.

### P3 — Final training and ablations

- Use five seeds for the important 4–6 conditions.
- Pre-register the selection metric.
- Tune only on development domains.
- Freeze and commit code before blind evaluation.

### P4 — Blind evaluation and release

- Evaluate once on a temporal or challenge-server test.
- Release a clean tag, lockfile/container, licensed manifest hashes, lightweight scores, and checkpoint hashes.
- Preserve controlled negative results: consistency, naive fine-tuning, and XLS-R failures are useful evidence.

## 11. Suggested paper positioning

### Working title

**Decodability Is Not Reliance: Auditing and Controlling Domain Dependence in Speech Deepfake Detectors**

### Claims that could work after validation

1. Linear probe accuracy overstates decision dependence on domain information.
2. A reparameterization-aware reliance metric predicts failures under unseen generator/channel/language combinations better than probe accuracy.
3. Reliance-targeted regularization improves worst-group and cross-domain performance without the utility loss of full invariance.
4. The effect holds across multiple SSL front ends, back ends, and datasets—and ideally another domain-generalization task for a general ML venue.

### Claims to avoid now

- “state of the art,” “universal,” or “robust in the wild”;
- “DRS explains the decoupling”;
- “BMI/CALAS improves generalization”;
- “strictly held-out final test” for repeatedly inspected corpora;
- “multilingual generalization” without language-wise holdouts;
- “reproducible” until ignored results, RNGs, dependencies, and tests are fixed.

## 12. Venue assessment

| Venue class | Current fit | What would change it |
|---|---|---|
| NeurIPS / ICML / ICLR | Strong reject | General reliance theory/metric, causal validation, multiple tasks/modalities, extensive modern benchmarks |
| AAAI / IJCAI | Strong reject | Correct novel method, strong baselines, multi-seed gains, broader significance |
| ACL / EMNLP | Reject | Linguistic/multilingual dimension, modern generators, strong analysis or data contribution |
| ACM Multimedia | Reject | All-type/multimodal, localization, or strong forensic robustness |
| ICASSP / Interspeech | Reject now; plausible after major revision | Correct DRS, official baselines, factorized evaluation, statistically solid gains |
| Security venues | Weak fit | Explicit threat model, adaptive attacker, worst-case and deployment analysis |
| Workshop | Plausible after cleanup | Corrected diagnostic/negative-results paper with complete artifacts |

## 13. Final assessment

AudioShield is a promising research scaffold, not yet a top-conference contribution. Its strongest assets are broad OOD evaluation, recognition that bona-fide source matters, the leakage-audit mindset, and preserved negative results. Its strongest current empirical finding is simpler than the repository narrative:

> Balanced multi-corpus exposure helps AI4T substantially, while consistency, naive fine-tuning, projected SupCon, and a backbone swap do not produce uniform generalization.

The main novelty claim—decodability versus reliance—is where the work could become interesting, but it must be rebuilt with valid geometry and causal tests. Until then, DRS is an exploratory hypothesis, not a result.

The highest-value next action is **not another large training run**. Correct DRS, establish a common-protocol baseline suite, repair the train/test methodology, and test whether reliance predicts controlled domain-shift failure. That decision will determine whether to pursue a genuinely novel reliance-aware paper or pivot to a narrower replay/media-robustness contribution.

