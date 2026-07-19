# manifests/v2 -- unified schema

One CSV per corpus, identical columns (see `src/audioshield/data/manifest.py`,
`FIELDNAMES`):

```
utt_id, path, target, corpus, split, attack, bona_fide_source,
source_id, speaker_id, generator_id, channel_id, language, platform_id
```

- `target` -- 1 = spoof, 0 = bona fide.
- `corpus` -- lowercase corpus id (`diffssd`, `asvspoof5`, `fakeorreal`, `vctk`, ...).
- `split` -- `train` | `val` | `test`.
- `attack` -- generator/attack tag as recorded in the source CSV. **Diagnostics only,
  never a train label.**
- `bona_fide_source` -- genuine-domain tag for bona-fide rows; `na` for spoof rows.
- `source_id` / `speaker_id` / `generator_id` / `channel_id` / `language` / `platform_id`
  -- factor-metadata columns backfilled by `scripts/extend_manifests.py`'s `derive()`.
  `"NA"` (not blank) means "not derivable from what we have," never a silent omission.

## diffssd: openvoicev2 speaker/accent convention

`diffssd`'s spoof audio lives under `generated_speech/<generator>/...`. For most
generators the whole spoof set is flat (`generated_speech/<generator>/sentence_N.wav`)
and `generator_id` comes straight from the source CSV's `attack` column.

`openvoicev2` is the one exception: it is nested one level deeper --
`generated_speech/openvoicev2/speaker_NNN/sentence_K_en-XX.wav` -- with 10 distinct
`speaker_NNN` ids and 5 distinct accent suffixes on the filename itself (2,500 rows per
speaker, 5,000 rows per accent, 25,000 total). `derive()` parses both directly from the
path/filename, since neither is present in the `attack` column:

- `speaker_id` = the `speaker_NNN` path component (e.g. `speaker_100`).
- `language` = the trailing `en-XX` accent suffix on the filename, kept **verbatim** --
  not normalized or restricted to a fixed set. The five values seen in the current
  manifest are `en-au`, `en-br`, `en-default`, `en-india`, `en-us`; any future accent
  suffix parses the same way without a code change.

`generator_id` for `openvoicev2` rows comes from the source CSV's `attack` column like
every other diffssd generator (`attack="openvoicev2"` is a real per-file generator
label here, not a placeholder -- see the `PLACEHOLDER_ATTACK_BY_CORPUS` scoping in
`extend_manifests.py`, which previously misapplied ITW's/ReplayDF's placeholder-attack
suppression to diffssd and silently mislabeled every diffssd openvoicev2 row
`generator_id="generated_speech"`).
