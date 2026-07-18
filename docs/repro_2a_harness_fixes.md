# Step 2a Reproduction Harness Fixes

This is the tracked runbook and provenance record for the reproduction-harness repair
that starts from `repair/2a-correctness` commit `777ff009`. It closes the same defect
class documented in [`docs/review/2a_verification_report.md`](review/2a_verification_report.md)
§3.5: a gate can look complete in isolation while its real call site remains unwired.

## Pinned environment recipe

From the repository root, install the exact third-party environment first and the local
package second:

```bash
pip install -r requirements.lock.local4060 \
  --extra-index-url https://download.pytorch.org/whl/cu121
pip install -e . --no-deps
```

`--no-deps` is deliberate. It makes the local `src`-layout package importable without
allowing the editable install to re-resolve or replace anything in the lockfile.
The extra index supplies the lockfile's exact `+cu121` Torch wheel; it does not change
the pinned version.

Before GPU evaluation, validate every child without loading a model:

```bash
python scripts/reproduce_eval.py --data-root E:/AI_voice_detection --preflight
```

Only after all three child preflights report `PREFLIGHT OK` should the EER gate run:

```bash
python scripts/reproduce_eval.py --data-root E:/AI_voice_detection
```

## Run 1 defects and fixes

### 1. The lockfile did not install the project package

Run 1 verified all three checkpoint hashes, but every child failed with:

```text
ModuleNotFoundError: No module named 'audioshield'
```

`requirements.lock.local4060` contains third-party distributions. AudioShield is a
local `src`-layout project defined by `pyproject.toml`, so installing the lockfile alone
cannot make it importable. The fix is the editable, dependency-free installation in the
recipe above. `tests/test_gate_runnable.py` now launches a fresh Python subprocess and
requires `import audioshield.evaluation.cross_test` to succeed.

Earlier local runs did not expose this because the developer environment already had
the project installed. That ambient installation was not represented by the lockfile,
so a fresh reproduction environment revealed the missing setup step.

### 2. The parent built an invalid child command

`cross_test.py` requires `--corpora`, but the Run 1 version of
`scripts/reproduce_eval.py` supplied no held-out corpora and relied on unrelated parser
defaults for other inputs. `build_cmd(run, ckpt, data_root)` now supplies the complete,
reviewable contract explicitly:

```text
--corpora inthewild replaydf ai4t
--manifest-dir manifests/v2
--dev-corpora diffssd fakeorreal asvspoof5
```

The command builder is pure. Tests parse its child arguments with the real
`cross_test.build_parser()` and also prove that removing `--corpora` is rejected.

### 3. Data-path failures were discovered too late

The cross-test estimates its threshold from development audio before held-out scoring.
Run 1 had no dry path that resolved those manifests and audio files before checkpoint
and model loading. `cross_test.py --preflight` now checks, in one table:

- the model config and checkpoint;
- all three development manifests;
- all three held-out manifests; and
- the first 25 audio paths that each corpus's real runtime selection would open.

Development selection mirrors the evaluator exactly: filter `val`, use a fresh
`random.Random(7)` per corpus, shuffle, and cap at 1,000. Preflight exits before device
selection, `torch.load`, backbone construction, or inference. The parent
`reproduce_eval.py --preflight` passes the flag to every child and stops after their
tables.

## C0 manifest-pin evidence

The historical README command used `--manifest-dir manifests`. C0 compared each legacy
file with its `manifests/v2` counterpart row-for-row over every shared column. All six
were identical:

| Corpus | Legacy rows | v2 rows | Shared-column result |
|---|---:|---:|---|
| In-the-Wild | 31,779 | 31,779 | IDENTICAL |
| ReplayDF | 52,320 | 52,320 | IDENTICAL |
| AI4T | 3,148 | 3,148 | IDENTICAL |
| ASVspoof5 | 323,307 | 323,307 | IDENTICAL |
| DiffSSD | 94,226 | 94,226 | IDENTICAL |
| FakeOrReal | 69,300 | 69,300 | IDENTICAL |

The shared columns were `attack`, `bona_fide_source`, `corpus`, `path`, `split`,
`target`, and `utt_id`. This proves row identity for the legacy evaluation fields; it
does not claim the schemas are identical because v2 additionally carries factor
columns. The held-out counts also equal the full populations recorded in each committed
`experiments/e007/*_crosstest.json`, so the reference EERs were not produced from a
capped subset. The reproduction pin is therefore `manifests/v2`.

## Assertion scope

The child records the development threshold and, per held-out corpus, EER, balanced
accuracy, expected calibration error, sample count, and bootstrap interval. The Step 2a
reproduction gate asserts only EER against the preregistered tolerance:

```text
abs(reproduced_eer - expected_eer) <= 0.002
```

Threshold, balanced accuracy, and ECE remain recorded diagnostics. They are not asserted
by this preservation gate, and their presence in the JSON must not be described as an
additional reproduction guarantee.

## Run #2 deviation

Gate run #1 assumed all three dev-tier corpora (`diffssd`, `fakeorreal`, `asvspoof5`)
were present on the machine running the gate. Between run #1 and run #2, ASVspoof5 and
DiffSSD raw audio were found absent from the Syncthing-synced `datasets` folder on both
machines:

- **Sarvarbek's machine**: verified absent. Only the XLS-R embedding cache and the
  preserved manifests remain; raw audio was deleted post-embedding, per the project's
  established storage pattern. Checksum manifests were taken 2026-07-12, immediately
  before pruning.
- **Collaborator machine**: verified absent by an exhaustive read-only sweep of all
  three drives (C/D/E) -- no directories, no archives.

Gate run #2 therefore uses `--dev-corpora fakeorreal` -- the one dev-tier corpus
confirmed present on both machines -- instead of the canonical three-corpus default.
`build_cmd` (`scripts/reproduce_eval.py:45`) now takes `dev_corpora` as an optional
parameter, and `main()` gained a `--dev-corpora` flag (`scripts/reproduce_eval.py:69-75`)
that threads a caller-supplied list through to it, so this is an explicit, reviewable
substitution rather than a silent one. The canonical default (`DEV_CORPORA` =
`diffssd, fakeorreal, asvspoof5`) is unchanged; an invocation without `--dev-corpora`
still uses all three.

EER is the only quantity the Step 2a reproduction gate asserts (see "Assertion scope"
above; the comparison loop is `scripts/reproduce_eval.py:133-141`), and it is computed
from the **held-out** (`inthewild`, `replaydf`, `ai4t`) corpora and the checkpoint's
already-fixed decision boundary -- threshold-free, and therefore unaffected by which
dev-tier corpora are available. Development audio is used only to fit the recorded
(non-asserted) classification threshold that the diagnostic balanced-accuracy and ECE
numbers depend on; the `threshold`, `bac`, and `ece` fields in `repro_*.json` for run #2
are **not** comparable to the committed `experiments/e007/*.json` references, since
they were fit against a single-corpus dev-tier set rather than the original
three-corpus default.

This is a data-availability deviation, not a code or methodology change. The XLS-R
embedding caches that Steps 3-5 (reliance instrumentation, the gate, the baseline
suite) consume are unaffected, since those steps operate on cached embeddings, not raw
audio. Raw ASVspoof5/DiffSSD audio re-acquisition from public sources is required only
before Step 6 fine-tuning, which needs raw waveforms for augmentation and backbone
adaptation.
