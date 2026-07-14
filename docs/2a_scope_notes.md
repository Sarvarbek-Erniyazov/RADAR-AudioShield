# 2a wiring-repair series -- explicit out-of-scope items

The 2a-w1..w11 commits wired the PRESENT-BUT-UNWIRED findings from
`docs/review/2a_verification_report.md` into the real execution path
(`train_e002.py` / `cross_test.py` / `loop_e002.py` / `reproduce_eval.py`). The
items below were deliberately left untouched. Each is documented here with the
reason it's out of scope, so it isn't mistaken for an oversight.

## scripts/compute_drs.py -- §3.3 VCTK class-confound in domain estimation

`collect_embeddings()` still pools all corpora, including VCTK (bona-fide-only),
uniformly when estimating the domain subspace used for DRS. This was flagged as
UNRESOLVED in Pass 2 (finding #10) and confirmed UNTOUCHED by any of the 2a-c*
commits.

It is out of 2a-w scope because the roadmap already plans to replace this script
wholesale in Roadmap v3 Step 3 (the DRS validity gate), which redesigns the
domain-subspace estimator with cross-fitting and class control rather than
patching the existing ad hoc script. Patching `collect_embeddings()` now would be
throwaway work. A `DeprecationWarning` was added at the top of
`scripts/compute_drs.py` pointing here, so anyone importing or running it is told
directly rather than discovering the confound by reading the code.

## Factor-column batch consumption (source_id/speaker_id/generator_id/channel_id/platform_id)

`ManifestRow` carries these fields and `manifests/v2/*.csv` populate them
correctly, but no production code path (`unified_dataset.py.__getitem__`,
`collate_unified`, the `grouped_probe` call sites) reads them back out of a row
into a batch. This is Step 2b work (threading a data-layer change through the
dataset/collate/probe chain) per `docs/probe_wiring_todo.md`, deliberately kept
out of the 2a correctness-fix set so it doesn't touch the training path the
reproduction gate has to validate.

**Landmine to check before wiring this in 2b**: every unknown factor value in
`manifests/v2/*.csv` is the literal uppercase string `"NA"`, which is one of
pandas' default NA sentinels. The current runtime path uses stdlib
`csv.DictReader` (immune to this), but any future `pd.read_csv` added to consume
these columns (e.g. a bootstrap-clustering or factor-analysis script) MUST pass
`keep_default_na=False` (as `extend_manifests.py` and `validate_manifests.py`
already do) or every `"NA"` cell will silently become `NaN`, corrupting the
column dtype for a large fraction of rows in most corpora.

## Environment drift runtime check

`environment.local4060.json` / `requirements.lock.local4060` are written once by
`scripts/freeze_environment.py` and only re-checked by `tests/test_environment.py`
under `pytest`. No training/eval entrypoint asserts the live environment matches
the fingerprint before a run. Adding that assertion to `train_e002.py`/
`cross_test.py` is a reasonable follow-up but is 2b: it doesn't fix a correctness
bug in the current results, it adds a new guard against future drift.

## linear_baseline.py / layer_probe.py feature-cache keys

Both scripts cache `.npz` embeddings keyed only by filename, with no hash of the
manifest/config that produced them, so a stale cache from before a manifest
change (e.g. v1 -> v2) is silently reused. These are auxiliary analysis scripts,
not part of the e002-e007 training/eval loop the reproduction gate covers, so
fixing their cache-invalidation logic is deferred to 2b.

## leakage_audit.py zero-input caveat

`check_hashes()` silently skips files it can't resolve under `--data-root`
(`if not p.exists(): continue`). A wrong `--data-root` therefore prints "0 shared
identifiers / 0 collisions" -- indistinguishable from a genuinely clean audit.
**Before trusting any "no leakage" claim from this script, manually confirm
`n_train_hashed`/`n_test_hashed` in its output are non-zero and roughly match the
expected corpus sizes.** Adding an explicit assertion for this is 2b (a defensive
tooling improvement, not a correctness fix to the trained/evaluated models
themselves).
