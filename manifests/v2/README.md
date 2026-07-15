# manifests/v2 -- unified schema

One CSV per corpus, identical columns (see `src/audioshield/data/manifest.py`,
`FIELDNAMES`):

```
utt_id, path, target, corpus, split, attack, bona_fide_source,
source_id, speaker_id, generator_id, channel_id, language, platform_id, hf_path
```

- `target` -- 1 = spoof, 0 = bona fide.
- `corpus` -- lowercase, no separators (`diffssd`, `asvspoof5`, `mlaad`, `odssclean`, `kwokbona`, ...).
- `split` -- `train` | `val` | `test`. Eval-only corpora (`inthewild`, `replaydf`, `ai4t`,
  `kwokbona`) simply never have a `train`/`val` row -- that is the "eval-only" marker; there
  is no separate role column.
- `attack` -- generator/attack tag. **Diagnostics only, never a train label.**
- `bona_fide_source` -- genuine-domain tag for bona-fide rows (used by BMI / Kwok
  cross-testing); `na` for spoof rows.
- `source_id` / `speaker_id` / `generator_id` / `channel_id` / `language` / `platform_id` --
  factor-metadata columns. `"NA"` (not blank) means "not derivable from what we have",
  never a silent omission.
- `hf_path` -- upstream HuggingFace-repo-relative path, for corpora whose raw audio is
  fetched from an HF dataset repo and later deleted post-embedding (see below). `"NA"`
  for every corpus that isn't sourced this way. New column, added alongside the three
  corpora below; every pre-existing manifest is untouched and simply lacks it (readers
  default missing `hf_path` to `"NA"`).

## New in this pass: mlaad, odssclean, kwokbona

Built by `scripts/build_mlaad_manifest.py` / `build_odss_manifest.py` /
`build_kwok_manifest.py`, using the converters under
`src/audioshield/data/converters/{mlaad,odss_clean,kwok_bona}.py`. All three are
**measurement-only for now** -- Step 3/4/5 material. They are not wired into any
training config or mix, per this pass's scope.

### mlaad (train-pool candidate, spoof-only)

456,000 rows, `spoof_frac == 1.0` (enforced by an assert in the build script -- MLAAD
has no bona-fide content, so any deviation from 1.0 means the manifest is broken).

MLAAD's raw audio is deleted language-by-language right after its frozen XLS-R-300M
embedding is cached (`_mlaad_pipeline.py` in the dataset root) -- disk-space lifecycle,
not a data-loss bug. The manifest is therefore built from `_MASTER_MANIFEST.tsv`
(rel_path, size, sha256, HF revision) rather than a directory walk. `hf_path` preserves
the exact `mueller91/MLAAD` repo-relative path per row (e.g.
`fake/en/Cartesia.ai (Sonic-3)/dorothy_and_wizard_oz_01_f000059.wav`), pinned to the
revision recorded in `10_MLAAD/_SOURCE_REVISION.json`. That is what makes a *targeted*
re-fetch possible later (`hf_hub_download(repo_id="mueller91/MLAAD", filename=hf_path,
revision=...)`) without re-pulling the full multi-hundred-GB dataset.

`generator_id`/`language`/`source_id` are derived from the path
(`fake/<lang>/<generator>/<book>_<chap>_fNNNNNN.wav`) by reusing
`scripts/extend_manifests.py`'s existing (already-tested, `tests/test_commit3.py`-pinned)
`derive()` "mlaad" branch, so this stays consistent with parsing logic that predates
this pass. 175 distinct generator names across 54 languages.

Per docs/mix_sweep_v2_findings.md (Decision Point A): MLAAD is *not* in the frozen-probe
base mix -- reserved for the multilingual evaluation axis (Step 5). Per-target effects
there are directional (helps AI4T, hurts ReplayDF, slightly hurts ITW), not a flat
"MLAAD hurts."

### odssclean (train-pool candidate, mixed)

26,954 rows, spoof_frac = 0.7046 (18,993 spoof / 7,961 bona), matching the number
already reported in `_mix_sweep_v2.log` / `docs/mix_sweep_v2_findings.md`. The build
script asserts this stays within +/-0.05 of 0.70 as a regression guard.

Raw layout (`12_ODSS/odss/{natural,fastpitch-hifigan,vits}/<source_dataset>/<speaker>/*.wav`):
`natural` = bona fide, `fastpitch-hifigan`/`vits` = two TTS systems applied over the
same source utterances. `<source_dataset>` in `{hifi-tts, hui-acg, openslr-es, vctk}`.

**Exclusion**: `odss/natural/vctk/**` (3,071 files) is dropped. Verified against
`_embcache_xlsr300m/12_ODSS_CLEAN/_done.txt` (26,954 rows vs. 30,025 in the unfiltered
`12_ODSS` cache) -- this is precisely the exclusion the project already applied when it
built the `12_ODSS_CLEAN` embedding cache used by the mix sweep; this manifest
reproduces it exactly rather than introducing a new policy. Those bona-fide files are
literally the standalone VCTK corpus's own recordings re-used as ODSS's TTS-prompt
source speech -- keeping them would duplicate bona-fide identity across two pool
corpora, the same §4.4 bona-only-confound concern already documented for VCTK's slight
negative inclusion effect. `odss/vits/vctk/**` (3,071 *spoof* files, synthetic audio) is
kept -- it isn't a duplicate of anything, and `odss/fastpitch-hifigan` never covered
vctk source text to begin with (7,961 files vs. 11,032 for natural/vits).

`generator_id` = `fastpitch-hifigan`/`vits` for spoof rows, `NA` for bona rows (matches
the rest of the schema's convention: generator_id is never populated for bona-fide
audio). `language`: hifi-tts -> en, hui-acg -> de, openslr-es -> es, vctk -> en.

### kwokbona (eval-only, bona-fide cross-testing)

4,200 rows, `spoof_frac == 0.0` (enforced by an assert -- this is a bona-fide-only
eval pool), `split == "test"` for every row (the eval-only marker).

Source: Kwok et al., "Bona fide Cross Testing Reveals Weak Spot in Audio Deepfake
Detection Systems" (Interspeech 2025). `13_KWOK_BONA/data/` ships 9-10 subdirectories,
one per bona-fide "style x synthesizer pairing" -- **these are not homogeneously bona
fide.** Each ships a `trial_metadata.txt` whose 6th field is the real per-file label;
several folders bundle a small (~600) bona-fide carrier set together with spoof
variants generated against it (that pairing is the point of the framework). Verified by
reading every subset's label distribution directly:

| subset | bonafide | spoof |
|---|---|---|
| ami_ihm | 600 | 0 |
| ami_sdm | 600 | 0 |
| librispeech_test_clean | 600 | 0 |
| librispeech_test_other | 600 | 0 |
| vctk | 600 | 0 |
| emofake | 600 | 3,000 |
| llamapartialspoof_r01tts0a | 600 | 3,600 |
| llamapartialspoof_r01tts0b | 0 | 3,600 |
| asvspoof2019_la (excluded) | 600 | 7,800 |
| asvspoof2021_df (excluded) | 600 | 65,400 |

The converter filters every subset on this label, keeping only rows marked
"bonafide" -- an earlier pass assumed folder-level homogeneity and would have
mislabeled ~10,200 emotion-converted / partial-spoof / TTS rows as bona fide.
`llamapartialspoof_r01tts0b` therefore contributes zero rows.

**Leakage-policy exclusion**: `asvspoof2019_la` and `asvspoof2021_df` are dropped
*entirely* (before their trial_metadata is even read) -- their audio filenames
(`LA_E_########.flac`, `DF_E_########.flac`) confirm they are literally
ASVspoof2019-LA / ASVspoof2021-DF eval-partition audio, and this project's pool
already carries ASVspoof5. The exclusion check matches on the substring "asvspoof" in
the subset name (`_is_asvspoof_derived`) rather than a hardcoded pair, so any future
re-pull that adds another ASVspoof generation is caught the same way.

**Known, NOT excluded here** (out of this exclusion's stated scope -- ASVspoof leakage
only -- but a real overlap worth flagging for whoever next touches bona-fide-confound
policy): `kwokbona`'s `vctk` subset duplicates the standalone VCTK corpus's own
recordings (same speaker codes, e.g. p227), and `llamapartialspoof_r01tts0a`'s
bona-fide carrier is built over LibriSpeech dev-clean/test-clean (same speaker/chapter
ids DiffSSD's `real_speech` pulls from). Both are candidates for the same kind of
exclusion already applied to ODSS_CLEAN's `natural/vctk`. `source_id`/`speaker_id`
deliberately reuse each upstream corpus's own identity namespace (`ls-<speaker>`,
`p<NNN>`) specifically so a future cross-corpus leakage audit can match on them
directly.

## Not wired into training

None of these three corpora are referenced by any training config or mix in this
pass. They exist for measurement (Step 3/4/5): MLAAD's multilingual axis, ODSS_CLEAN
as an additional train-pool candidate, and kwokbona as a bona-fide-diversity eval
probe.
