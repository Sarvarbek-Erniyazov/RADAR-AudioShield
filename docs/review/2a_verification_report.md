# 2a-correctness Verification Report

Branch: `repair/2a-correctness`. Four read-only passes: (1) test/manifest execution, (2) audit-finding-to-commit traceability, (3) adversarial bug hunt, (4) final consistency check. No fixes were applied in any pass; this document only consolidates what was found.

---

## Pass 1 — Test suite and manifest validation

`python -m pytest -v`:
- **40 passed, 1 deselected**, 19.09s. 0 failed, 0 skipped, 0 warnings, no collection errors.
- Files: test_commit3.py (11), test_commit4.py (4), test_commit5.py (6), test_commit5b.py (3), test_commit6.py (5), test_determinism.py (4), test_environment.py (3), test_hf_loading.py (4).

`python scripts/validate_manifests.py`:
- Exit code **0**. 4 `[OK]`, 3 `[WARN]`, no `[FAIL]`.
- OK: asvspoof5.csv (323,307 rows), diffssd.csv (94,226), replaydf.csv (52,320), vctk.csv (44,242).
- WARN (same root cause — spoof rows without per-file `generator_id` label in source): ai4t.csv (919/3,148), fakeorreal.csv (34,695/69,300), inthewild.csv (11,816/31,779).

Both green. This says nothing about wiring — see Pass 2.

---

## Pass 2 — Audit-finding traceability table (verbatim)

For each finding: which commit/file claims to fix it, what tests it, and whether it is actually **wired** into `train_e002.py` / `cross_test.py` / `loop_e002.py`, verified by reading call sites rather than trusting commit messages.

| # | Audit finding | Commit / fix file | Test coverage | Wired into train_e002.py / cross_test.py / loop_e002.py? | Verdict |
|---|---|---|---|---|---|
| 1 | §4.2 seeds never applied | `66971f2` (2a-c3) → `src/audioshield/utils/seeding.py` (`seed_everything`, `worker_init_fn`) | `tests/test_determinism.py` | **No.** `train_e002.py`/`loop_e002.py` never import `audioshield.utils.seeding`. No `torch.manual_seed`, no `np.random.seed`, no `worker_init_fn=`/`generator=` on any DataLoader. Only stdlib `random.Random(13)` is used, to shuffle val-set caps — unrelated. | **PRESENT-BUT-UNWIRED** |
| 2 | §4.4 sampler class-corpus confound + VCTK | `e704033` (2a-c4) → `src/audioshield/data/balanced_weighting.py` (`compute_joint_weights`) | `tests/test_commit4.py` | **No.** `train_e002.py:93-110` defines its own `make_balanced_sampler`, called at `train_e002.py:163`, and reproduces verbatim the exact broken formula (`(1/n_corpora)·(1/n_classes_in_corpus)·(1/count)`) that the new module's own docstring names as the bug (VCTK double-weighted). `compute_joint_weights` is referenced only in its test. | **PRESENT-BUT-UNWIRED** |
| 3 | §4.7 probe protocol (grouping, chance baseline) | `bc07877`+`319a963`+`cb981fd` (2a-c5) → `src/audioshield/evaluation/grouped_probe.py` | `tests/test_commit5.py` | **Partially.** Honest majority-baseline + balanced accuracy IS wired at both `cross_test.py:310-311` and `loop_e002.py:179`. But both call sites hardcode `meta=None`, so `StratifiedGroupKFold` degenerates to ungrouped stratified folds — `docs/probe_wiring_todo.md` explicitly confirms grouping is deferred pending a `source_id` cache field, deferred to Step 2b. | **RESOLVED (chance baseline) / confirmed-deferred-by-design (grouping)** — matches the stated exception |
| 4 | §4.6 manifest factor columns + bootstrap clustering | `7fb20f3` (2a-c5b) → `ManifestRow` fields; `fbf66ff` (2a-c6) → `metrics_v2.py` (`clustered_bootstrap_eer`) | `tests/test_commit5b.py`, `tests/test_commit6.py` | **No, on both halves — but for different reasons.** Factor columns (source_id/speaker_id/etc.) are populated correctly in `manifests/v2/*.csv` and parsed into `ManifestRow`, but no production code reads them back out — `unified_dataset.py.__getitem__`/`collate_unified` never copy those fields into batches, so they can't reach `grouped_probe` or any clustering call. Separately, `metrics_v2.py` (which contains the real cluster-aware bootstrap) is imported nowhere outside `tests/test_commit6.py`; `cross_test.py` still calls its own pre-existing `_bootstrap_eer_ci`. | **Factor columns: PRESENT-BUT-UNWIRED (genuine gap).** **Bootstrap clustering: confirmed-unwired-by-design** — matches the stated `metrics_v2.py` exception (reproduction gate must run old path first) |
| 5 | §5 fail-fast loader | `66971f2` (2a-c3) → `src/audioshield/data/safe_audio.py` (`load_audio_strict`) | `tests/test_commit3.py` | **No.** The dataset actually used in training, `UnifiedAudioDataset.__getitem__` (`unified_dataset.py:72-92`), still uses the old `audio_io.load_audio` and reproduces the exact 50-row silent-substitution loop the fix targets. `load_audio_strict`/`AudioReadError` are referenced only in the test. | **PRESENT-BUT-UNWIRED** |
| 6 | §5 RIR silent no-op | `66971f2` (2a-c3) → `src/audioshield/data/aug_assets.py` (`resolve_aug_assets`, `fingerprint_asset_dir`) | `tests/test_commit3.py` | **No.** Real augmentation path (`channel_aug.py:21` hardcoded `RIR_DIR`, `:82-86` silent one-time-print no-op) is untouched; `resolve_aug_assets`/`fingerprint_asset_dir` are called nowhere outside the test — no fingerprint ever lands in `run_config.json`. | **PRESENT-BUT-UNWIRED** |
| 7 | §5 dependency pinning | `66971f2` (2a-c3) → `requirements.txt`, `requirements.lock.local4060`, `configs/backbone_revisions.yaml`, `hf_loading.py` | `tests/test_environment.py`, `tests/test_hf_loading.py` | **Partially.** Package pins (scipy, transformers bounds, exact torch/transformers/sklearn/numpy pins in the lockfile) are real and consumed at env setup. But the HF **revision** pin isn't wired: `load_backbone()` correctly reads `configs/backbone_revisions.yaml` and passes `revision=`, but the actual model class used in training, `models/ssl_backbone.py:30`, calls `AutoModel.from_pretrained(backbone_name, local_files_only=True)` directly — bypasses `load_backbone()` entirely, no `revision=` passed. | **PARTIALLY RESOLVED** (package pinning wired; HF revision pinning PRESENT-BUT-UNWIRED) |
| 8 | §5 quadratic EER | `fbf66ff` (2a-c6) → `metrics_v2.py:equal_error_rate` (single-sort, O(n log n)) | `tests/test_commit6.py` | **No.** `cross_test.py:27,243,285` still imports and calls the old quadratic `metrics.py:equal_error_rate`; `layer_probe.py`/`linear_baseline.py` also still use the old one. | **Confirmed-unwired-by-design** — matches the stated `metrics_v2.py` exception (same file/reasoning as the bootstrap-clustering half of #4) |
| 9 | §1 SupCon `min_corpora_per_class` | `fbf66ff` (2a-c6) → `src/audioshield/training/supcon_guard.py` (`supcon_batch_valid`) | `tests/test_commit6.py` | **No.** `loop_e002.py:70-72` calls `cross_corpus_supcon(...)` without importing `supcon_guard` or passing `min_corpora_per_class`. Worse: `cross_corpus_supcon` itself (`xc_contrastive.py:26`) declares a `min_corpora_per_class` parameter but never reads it in the function body — vestigial, not the guard. The audit's own tracking table (line 259) still lists this as open. Not on the stated exception list. | **PRESENT-BUT-UNWIRED** |
| 10 | §3.3 VCTK class-confound in domain estimation | *(no 2a-c* commit)* → `scripts/compute_drs.py:39` (`collect_embeddings`) | *(none)* | **N/A — untouched.** Verified via `git show --stat` on all 11 commits in the series: none touch `compute_drs.py`. `e704033`'s "VCTK policy" only affects training-sampler weights (`balanced_weighting.py`), not the domain-subspace embedding collection used for DRS. `collect_embeddings()` still pools all corpora, VCTK included, uniformly. | **UNRESOLVED** |

**Net picture:** of the 10 findings, Step 2a produced correctly-implemented, individually-tested fix modules for essentially all of them — but only two things actually reach the real execution path: the probe's honest chance-baseline (#3, partial) and package-level dependency pinning (#7, partial). Everything else (seeding, sampler/VCTK weighting, fail-fast loader, RIR fingerprinting, HF revision pinning, manifest-factor consumption, SupCon guard) is dead code reachable only from its own unit test. Two items (#4-bootstrap-clustering, #8) are unwired *by design*, confirmed via `docs/probe_wiring_todo.md` and the `reproduce_eval.py` reproduction-gate reasoning (commits `e9f46fb`/`d390cc1`). Finding #10 was never addressed by any commit in this series. Finding #9 is a genuine unwired gap not covered by any stated exception.

---

## Pass 3 — Adversarial bug hunt

Recurring bug class targeted: silent fallbacks, path assumptions, placeholder values treated as data. Headline meta-finding: the same shape repeats across nearly every category below — a correctly-written, individually-tested fix module exists, but the call site that would invoke it was never updated.

### 3.1 — try/except swallowing data-loading errors or substituting rows

| File:Line | Description | Severity | 2a/2b |
|---|---|---|---|
| `src/audioshield/data/unified_dataset.py:72-92` | `__getitem__` silently substitutes one of the next 50 rows on any load exception; `last_err` captured but never surfaced unless all 50 fail. `safe_audio.py`'s `load_audio_strict`/`AudioReadError` exist and are correct but are imported nowhere outside `tests/test_commit3.py`. | High | 2a-blocking |
| `src/audioshield/data/channel_aug.py:87-91` | Per-file RIR read failure inside `rir_reverb` caught by a bare `except Exception: return x` with no logging at all — stricter/more silent than the already-known dir-missing case. | High | 2a-blocking |
| `src/audioshield/data/converters/diffssd_style.py:83-86` | `except ValueError: continue` silently drops rows with a malformed `target` cell during manifest construction — no count, no log. Directly affects manifests the 2a schema work builds on. | Medium | 2a-blocking (manifest integrity, in-scope per roadmap's "manifest schema" item) |
| `src/audioshield/models/ssl_backbone.py:30` | `AutoModel.from_pretrained(backbone_name, local_files_only=True)` — no `revision=`, no `output_loading_info`/`validate_load_report` check. `hf_loading.load_backbone()` exists to close this and is dead code outside its own test. | Medium-High | 2a-blocking (audit §5 dependency-pinning item) |
| `scripts/linear_baseline.py:34-54`, `scripts/layer_probe.py:46` | `.npz` feature caches keyed only by filename, no manifest/config hash — stale caches silently reused after manifest edits (e.g. v1→v2). Skips are logged (transparent); stale-cache reuse isn't. | Low-Medium | 2b-deferrable (auxiliary scripts, not the e002-e007 loop) |
| `src/audioshield/training/loop_e002.py:183-185`, `scripts/train_e002.py:220-227,265-274` | Broad `except Exception` around probe/resume/grad-checkpointing — all log a `[warn]` and take an explicit, visible fallback. Checked and found fine. | — | not a finding |

### 3.2 — Hardcoded paths that silently no-op

| File:Line | Description | Severity | 2a/2b |
|---|---|---|---|
| `src/audioshield/data/channel_aug.py:21,79-86` | `RIR_DIR` hardcoded; missing/empty dir → one-time print, silent no-op. `aug_assets.py`'s `resolve_aug_assets`/`fingerprint_asset_dir` exist to fix it but are called from zero configs or code — no `configs/experiments/*.yaml` even has an `augmentation.rir_root` key. | High | 2a-blocking |
| `configs/known_bad.txt`, `configs/backbone_revisions.yaml` | Both well-formed, but consulted only by the unwired modules (`safe_audio.py`, `hf_loading.py`) and their own tests. Zero runtime references from `train_e002.py`/`compute_drs.py`. | High | 2a-blocking |
| `environment.local4060.json`, `requirements.lock.local4060` | Written once by `freeze_environment.py`; never re-checked by any training/eval entrypoint. Drift is invisible outside `pytest`. | Medium | 2b-deferrable |
| `scripts/leakage_audit.py:107` | `if not p.exists(): continue` — silently skips unresolvable files under a bad `--data-root`; a wrong root prints "0 collisions," indistinguishable from a clean audit. | Medium | 2b-deferrable, but flag before trusting any "no leakage" claim from this script |
| `scripts/reproduce_eval.py:10-11` | `CKPT_DIR` hardcoded to a collaborator's absolute path — has a real fallback chain and `sys.exit(1)`s on total failure. Fails loudly. | — | not a finding (correctly implemented) |

### 3.3 — `pd.read_csv` missing `keep_default_na=False`

No live bug: the runtime path (`manifest.py:read_manifest`, used by `train_e002.py`/`cross_test.py`/`compute_drs.py`) uses stdlib `csv.DictReader`, immune to this. The only two `pd.read_csv` call sites in scope (`extend_manifests.py:86`, `validate_manifests.py:12`) both correctly pass `keep_default_na=False`.

| File:Line | Description | Severity | 2a/2b |
|---|---|---|---|
| `manifests/v2/*.csv` (all 7 corpora) | Every unknown `source_id`/`speaker_id`/`generator_id`/`channel_id`/`platform_id` is literally the string `"NA"` (matches `ManifestRow` defaults). This is exactly one of pandas' default NA sentinels. Latent landmine: the moment anyone adds a `pd.read_csv` on these manifests without `keep_default_na=False` — e.g. to wire up §4.6's still-unwired factor-column consumption — a large fraction of rows for most corpora silently gets NaN in these columns. | Medium (latent, not currently triggered) | 2b-deferrable to fix now, but must be caught before §4.6 wiring lands |

### 3.4 — Tests asserting happy path only, not the failure mode they claim to guard

`test_commit3.py`, `test_commit4.py`, `test_commit5.py`, `test_commit5b.py`, `test_commit6.py` were checked line-by-line and **all genuinely exercise their claimed failure modes** (raises on corrupt file/missing asset dir, numeric before/after on the VCTK confound, planted-leakage inflation vs. suppression, missing-column rejection, EER-vs-sklearn equivalence, SupCon guard actually rejecting an invalid batch). No finding there.

| File | Description | Severity | 2a/2b |
|---|---|---|---|
| `tests/test_determinism.py:30-47` | Test itself is rigorous (real two-run end-to-end comparison) — but it only exercises `seed_everything()` in isolation. `train_e002.py`/`loop_e002.py` never import `audioshield.utils.seeding`. The test suite passing gives false confidence that training is reproducible; it isn't. | High | 2a-blocking |
| `tests/test_environment.py:7-24` | All three tests assert properties of the current, already-correct environment/lockfile/revisions file. No test constructs a drifted/truncated/mismatched fixture to prove the guard actually fires. Pure happy-path. | Medium | 2a-blocking (undermines confidence in the dependency-pinning fix itself) |
| `tests/test_hf_loading.py:16-23` | `test_unpinned_revision_refused` only tests "revisions YAML file missing" — not "file exists but has the wrong/stale hash for this model" (a distinct, untested branch in `hf_loading.py:36-37`). No test proves `load_backbone` actually threads the pinned revision into `from_pretrained(revision=...)`. | Medium | 2a-blocking |

### 3.5 — `cross_test.py` / `reproduce_eval.py` file-format assumptions

`CHECKPOINTS.md` actual format (verified by reading it): plain `sha256sum`-style coreutils output — `<64-hex sha256> *<path>`, one `#` comment line, no table/JSON.

| File:Line | Description | Severity | 2a/2b |
|---|---|---|---|
| `scripts/reproduce_eval.py:24-32` (`load_expected_hashes`) | Parses `CHECKPOINTS.md` correctly — but the function is never called anywhere else in the file. `main()` computes `sha256(ckpt)` and only **prints** it — never compares against `CHECKPOINTS.md`. The module docstring claims "verifies hashes"; it does not. Commit `d390cc1` ("finalize gate") does not fix this. | High | 2a-blocking — the reproduction gate itself fails to do the one thing it exists for |
| `scripts/reproduce_eval.py:44` vs `CHECKPOINTS.md` | Even if wired, the primary candidate path (`CKPT_DIR / f"runs_{run}_best.pt"`, flattened-underscore) wouldn't match `CHECKPOINTS.md`'s keys (`runs/e007_A_fresh/best.pt`, nested). Only the fallback candidate path format matches. | Medium | 2a-blocking (latent bug inside the gate) |
| `src/audioshield/evaluation/cross_test.py:318-327` | `--out` defaults to `experiments/e001_unified_v1/crosstest_{Path(checkpoint).stem}.json` and `write_text` silently overwrites with no existence check. All three real e007 checkpoints are literally named `best.pt` → identical default `stem` → identical default output path. Any invocation without `--out` collides across e007-A/B/C, last-writer-wins, no warning. | High | 2a-blocking |
| `git show d390cc1` (diff on `reproduce_eval.py` only) | Pre-fix, `--out` was never passed at all (`cmd += [...] if False else []`), so `reproduce_eval.py`'s own read-back logic always found `prod = None`. The fix only changes `reproduce_eval.py`'s own invocation (explicit `repro_{run}.json` for its 3 known runs) — it does **not** touch `cross_test.py`'s underlying default-path collision, which remains live for any other caller. | High (root cause unfixed in `cross_test.py`) / Low (reproduce_eval.py's own 3 runs now avoid it) | 2a-blocking — partial fix, general collision the commit message claims to "avoid" is still reachable |
| (absence) | No code anywhere detects a stale/foreign result already sitting at an `--out` target before overwriting it. No such artifacts currently exist in the checkout, so nothing is corrupted yet. | Medium | 2b-deferrable |

---

## Pass 4 — Final consistency check

**(1) Stale paths / non-v2 manifests**
- `"RADAR-AudioShield-clean"` — zero hits anywhere in the repo.
- Absolute `C:\`/`c:\`/`/c/` paths — zero hits in `src/`, `scripts/`, `configs/`, `docs/`, `tests/`, or top-level files. Only live hits are in `legacy_biophys/` (4 `train.sh` scripts) — out of scope, pre-existing.
- **`configs/experiments/e005_multicorpus_frozen_C.yaml:4`** — `manifest_dir: "manifests"` — still points at the old v1 manifest directory. Its sibling `e005_multicorpus_frozen_v1.yaml` and all of `e006_xc_frozen_v1.yaml`/`e007_A/B/C.yaml` were correctly repointed to `"manifests/v2"` by commit `bb1dda0`, but that commit's 5-file list omitted `e005_multicorpus_frozen_C.yaml`. Both `manifests/` and `manifests/v2/` exist on disk with identical filenames, so this **silently loads stale v1 data** rather than failing. Notable because e005-C's AI4T/ITW result directly drove `e007_A.yaml`'s `spoof_pos_weight: 1.0` decision — a rerun of e005-C won't reproduce on the manifest version everything else in the ladder now uses.

**(2) Backbone pinning**
`configs/backbone_revisions.yaml` exists with two entries: `facebook/wav2vec2-xls-r-300m` → `1a640f32...`, `microsoft/wavlm-large` → `c1423ed9...`. Both backbones actually referenced across e005/e006/e007 (WavLM-large default via `configs/models/audioshield_x_v1.yaml`, XLS-R explicit in `e007_C.yaml`) have matching entries — no gap. (The pin is still never consulted at runtime, per Pass 2/3 finding on `ssl_backbone.py:30`.)

**(3) Lockfile / environment JSON**
`requirements.lock.local4060` and `environment.local4060.json` agree exactly on all six shared packages: torch `2.5.1+cu121`, transformers `4.57.6`, numpy `2.5.1`, scipy `1.18.0`, scikit-learn `1.9.0`, soundfile `0.14.0`. No mismatches — consistent by construction (both written by one run of `freeze_environment.py`).

**(4) Dangling references**
Nothing broken by rename/typo. Only stale item: `docs/review/AI_VOICE_DETECTION_RESEARCH_AUDIT.md:261` (and its tracked root-level twin) still claims "README links a missing `docs/e007_finetuning_design.md`" — fixed in commit `e7eb024`; current `README.md` has no such link. Cosmetic, not functional. Everything else absent (`runs/*.pt`, `CHECKPOINTS.md`'s three checkpoints, `experiments/e000_layer_probe/`, `experiments/e003_linear_baseline/`) is gitignored/generated output, self-documented as absent in `README.md`/`RESULTS.md`/`experiments/README.md` — not a rename bug, just not yet produced.

---

## Final verdict

**Merge-ready: CONDITIONAL — NO in current state.**

Framing this as "consistent, pending only the reproduction-gate run on the GPU machine" undersells what's outstanding. Two different classes of problem exist, only one of which GPU execution would resolve.

### Conditions to satisfy before merge

1. **Fix `configs/experiments/e005_multicorpus_frozen_C.yaml:4`** — repoint `manifest_dir` to `"manifests/v2"` to match its siblings. No GPU required; checkable immediately.
2. **Wire the reproduction gate's own hash check** — `scripts/reproduce_eval.py`'s `load_expected_hashes()` must actually be called and compared against each checkpoint's computed sha256 before the gate can be trusted to catch a wrong/corrupted checkpoint. Also fix the path-key mismatch between `CKPT_DIR / f"runs_{run}_best.pt"` and `CHECKPOINTS.md`'s nested `runs/{run}/best.pt` keys.
3. **Fix `cross_test.py`'s `--out` collision at the source**, not just in `reproduce_eval.py`'s own invocation — any caller omitting `--out` and evaluating a checkpoint literally named `best.pt` (all three e007 checkpoints are) will silently overwrite a shared default path.
4. **Wire, or explicitly re-scope as 2b and document as known-unwired, the remaining PRESENT-BUT-UNWIRED items from Pass 2/3**: seeding (`seed_everything` never called), sampler/VCTK class-corpus weighting (`compute_joint_weights` never called), fail-fast audio loading (`load_audio_strict` never called), RIR asset fingerprinting (`resolve_aug_assets` never called), HF revision pinning (`load_backbone` never called), manifest factor-column consumption (fields populated but never read downstream), and the SupCon `min_corpora_per_class` guard (never called, and the underlying `cross_corpus_supcon` parameter is vestigial). As-is, the test suite passing does not mean these fixes are active in a real training/eval run.
5. **Add negative-path coverage** to `tests/test_environment.py` (drifted/mismatched env fixture) and `tests/test_hf_loading.py` (wrong/stale revision hash actually rejected) so the dependency-pinning claim is actually falsifiable by the test suite, not just self-consistent with the current correct state.
6. Cosmetic, non-blocking: update the stale "missing `docs/e007_finetuning_design.md`" claim in `docs/review/AI_VOICE_DETECTION_RESEARCH_AUDIT.md:261`.

Items already in good shape and not blocking: test suite (40/40 passing) and manifest validation (Pass 1); package-level dependency/environment pinning consistency (Pass 4 §3); backbone-revision coverage for both backbones in use (Pass 4 §2); no dangling/renamed file references (Pass 4 §4); the newly-added tests for commits 3-6 genuinely test their claimed failure modes, not just the happy path (Pass 3 §3.4); `pd.read_csv` NA-sentinel handling is correctly guarded everywhere it's currently used (Pass 3 §3.3, though flagged as a landmine for future §4.6 wiring work).

Confirmed-by-design exceptions (not blockers, per explicit scope note): `metrics_v2.py`'s cluster-aware bootstrap EER and single-sort EER are deliberately unwired pending the reproduction gate running on the original eval path first; probe grouping (`meta=None` in `grouped_probe` calls) is deliberately deferred per `docs/probe_wiring_todo.md`, pending a `source_id` cache field, to Step 2b.
