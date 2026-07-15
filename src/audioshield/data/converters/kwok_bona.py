"""Convert the Kwok bona-fide cross-testing corpus into unified manifest rows.

Source: Kwok et al., "Bona fide Cross Testing Reveals Weak Spot in Audio Deepfake
Detection Systems" (Interspeech 2025). `13_KWOK_BONA/data/` holds one subdirectory
per bona-fide "style x synthesizer pairing": ami_ihm, ami_sdm, asvspoof2019_la,
asvspoof2021_df, emofake, librispeech_test_clean, librispeech_test_other,
llamapartialspoof_r01tts0a, llamapartialspoof_r01tts0b, vctk.

IMPORTANT -- these directories are NOT homogeneously bona fide. Each ships a
`trial_metadata.txt` whose 6th whitespace-separated field is the real per-file label
(bonafide/spoof); several folders bundle a small bona-fide "carrier" set together
with spoof variants generated against it (that pairing is the whole point of the
framework -- one bona-fide style cross-tested against many synthesizers). Verified
by direct inspection of every subset's trial_metadata.txt label distribution:

    ami_ihm/ami_sdm/librispeech_test_clean/librispeech_test_other/vctk : 100% bonafide (600 each)
    emofake                                                            : 600 bonafide / 3,000 spoof
    llamapartialspoof_r01tts0a                                          : 600 bonafide / 3,600 spoof
    llamapartialspoof_r01tts0b                                          : 0 bonafide / 3,600 spoof
    asvspoof2019_la / asvspoof2021_df                                   : 600 bonafide / (7,800 / 65,400) spoof

A first pass of this converter assumed folder-level homogeneity and would have
mislabeled ~10,200 emotion-converted / partial-spoof / TTS rows as bona fide
(target=0). We now filter every subset on the metadata label, keeping only rows
marked "bonafide" -- llamapartialspoof_r01tts0b therefore contributes zero rows
(it is purely a synthesizer-side pairing, not its own bona-fide style).

Leakage policy: `asvspoof2019_la` and `asvspoof2021_df` are excluded entirely --
verified by inspecting their audio filenames (`LA_E_########.flac`, `DF_E_########.flac`,
the native ASVspoof2019-LA / ASVspoof2021-DF eval-partition naming) -- since this
project's pool already carries ASVspoof5, and these two subsets' own bona-fide carrier
is itself ASVspoof-derived. `_is_asvspoof_derived()` matches on subset name rather
than a hardcoded pair so any future re-pull that adds another ASVspoof generation
(2015, 2017, ...) is caught the same way. These two are skipped by name before their
trial_metadata.txt is even read.

Known, NOT excluded here (out of this exclusion's scope, but real overlap risk worth
flagging): `vctk` duplicates the standalone VCTK corpus's own recordings (same
speaker codes, e.g. p227), and `llamapartialspoof_r01tts0a`'s bona-fide carrier is
built over LibriSpeech dev-clean/test-clean (same speaker/chapter ids DiffSSD's
real_speech pulls from) -- both are bona-only-confound candidates in the same family
as ODSS_CLEAN's natural/vctk exclusion (see converters/odss_clean.py), but resolving
them is out of scope for this pass; `speaker_id`/`source_id` below intentionally
reuse each upstream corpus's own identity namespace (ls-<speaker>, p<NNN>) so a
future cross-corpus leakage audit can match on them directly.

Marked eval-only: every row gets split="test", matching how the other eval-only
corpora (inthewild, replaydf, ai4t) are marked in manifests/v2 -- an eval-only
corpus simply never has a train/val row.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..manifest import ManifestRow

AUDIO_EXTS = {".wav", ".flac"}
_ASVSPOOF_RE = re.compile(r"asvspoof", re.IGNORECASE)

_LIBRISPEECH_NATIVE_RE = re.compile(r"^(\d+)-(\d+)-\d+$")          # 1089-134686-0002
_LIBRISPEECH_UNDERSCORE_RE = re.compile(r"_(\d+)_(\d+)_\d+_\d+$")  # ..._1272_128104_000005_000007
_AMI_RE = re.compile(r"^eval_ami_(?P<session>[a-z0-9]+)_(?P<channel>h\d+|sdm)_(?P<speaker>[a-z0-9]+)_")
_EMOFAKE_RE = re.compile(r"^(\d+)_")
_VCTK_RE = re.compile(r"^(p\d+)_")

# Every included style is English per the corpus's own provenance (LibriSpeech, AMI,
# VCTK, and EmoFake's bona-fide speakers 0011-0020 == ESD's English-speaker range,
# confirmed by direct inspection: this pull's bonafide-labeled rows are only
# speakers 0014/0018).
LANGUAGE = "en"


def _is_asvspoof_derived(subset_name: str) -> bool:
    return bool(_ASVSPOOF_RE.search(subset_name))


def _identity_for(subset_name: str, stem: str) -> tuple[str, str, str]:
    """Return (source_id, speaker_id, channel_id), best-effort, "NA" if not derivable."""
    NA = "NA"
    if subset_name in ("librispeech_test_clean", "librispeech_test_other"):
        m = _LIBRISPEECH_NATIVE_RE.match(stem)
        if m:
            spk, chapter = m.group(1), m.group(2)
            return f"ls-{spk}-{chapter}", f"ls-{spk}", NA
    elif subset_name.startswith("llamapartialspoof_"):
        m = _LIBRISPEECH_UNDERSCORE_RE.search(stem)
        if m:
            spk, chapter = m.group(1), m.group(2)
            return f"ls-{spk}-{chapter}", f"ls-{spk}", NA
    elif subset_name in ("ami_ihm", "ami_sdm"):
        m = _AMI_RE.match(stem)
        if m:
            session, channel, speaker = m.group("session"), m.group("channel"), m.group("speaker")
            return f"ami-{session}-{speaker}", f"ami-{speaker}", channel
    elif subset_name == "emofake":
        m = _EMOFAKE_RE.match(stem)
        if m:
            spk = m.group(1)
            return f"esd-{spk}", f"esd-{spk}", NA
    elif subset_name == "vctk":
        m = _VCTK_RE.match(stem)
        if m:
            spk = m.group(1)
            return spk, spk, NA
    return NA, NA, NA


def _read_bonafide_ids(trial_metadata_path: Path) -> set[str]:
    """Return the set of audio-id stems trial_metadata.txt labels "bonafide"."""
    bonafide_ids: set[str] = set()
    with trial_metadata_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            cells = line.rstrip("\n").split(" ")
            if len(cells) > 5 and cells[5] == "bonafide":
                bonafide_ids.add(cells[1])
    return bonafide_ids


def convert(
    root: str | Path,
    path_prefix: str = "datasets/13_KWOK_BONA/data",
    corpus: str = "kwokbona",
    split: str = "test",
) -> list[ManifestRow]:
    """Read a `13_KWOK_BONA/data/<style>/**` tree and return unified ManifestRows.

    Args:
        root: path to the `13_KWOK_BONA/data` folder.
        path_prefix: dataset folder the manifest `path` column resolves from.
        corpus: unified corpus id.
        split: every row gets this split; Kwok is eval-only (see module docstring).
    """
    root = Path(root)
    subset_dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    if not subset_dirs:
        raise FileNotFoundError(f"No bona-fide-style subdirectories found under {root}")

    rows: list[ManifestRow] = []
    n_excluded_asvspoof = 0
    n_dropped_non_bonafide = 0
    excluded_subsets: list[str] = []
    empty_after_filter: list[str] = []
    for subset_dir in subset_dirs:
        subset_name = subset_dir.name
        if _is_asvspoof_derived(subset_name):
            n = sum(1 for p in subset_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
            n_excluded_asvspoof += n
            excluded_subsets.append(subset_name)
            continue

        meta_path = subset_dir / "trial_metadata.txt"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{subset_dir}: no trial_metadata.txt -- cannot verify which rows are "
                "genuinely bona fide (this corpus's folders are NOT homogeneously bona "
                "fide; see module docstring)"
            )
        bonafide_ids = _read_bonafide_ids(meta_path)

        audio_files = sorted(p for p in subset_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
        n_kept_here = 0
        for p in audio_files:
            if p.stem not in bonafide_ids:
                n_dropped_non_bonafide += 1
                continue
            source_id, speaker_id, channel_id = _identity_for(subset_name, p.stem)
            rel = p.relative_to(root).as_posix()
            full_path = f"{path_prefix.rstrip('/')}/{rel}"
            row = ManifestRow(
                utt_id=f"{corpus}/{rel}",
                path=full_path,
                target=0,
                corpus=corpus,
                split=split,
                attack="bonafide",
                bona_fide_source=subset_name,
                source_id=source_id,
                speaker_id=speaker_id,
                channel_id=channel_id,
                language=LANGUAGE,
            )
            assert not _is_asvspoof_derived(row.bona_fide_source), (
                f"{row.utt_id}: bona_fide_source {row.bona_fide_source!r} is ASVspoof-derived "
                "-- leakage-policy gate should have excluded this subset"
            )
            rows.append(row)
            n_kept_here += 1
        if n_kept_here == 0:
            empty_after_filter.append(subset_name)

    if not rows:
        raise ValueError(f"{root}: 0 usable rows -- refusing to write manifest")
    print(f"[kwok_bona] {root}: kept {len(rows)} bonafide-labeled rows; "
          f"dropped {n_dropped_non_bonafide} non-bonafide rows (spoof variants bundled in the "
          f"same folders) and {n_excluded_asvspoof} rows from ASVspoof-derived styles "
          f"{excluded_subsets}; styles with zero surviving rows: {empty_after_filter or 'none'}")
    return rows
