"""Convert the ODSS_CLEAN corpus into unified manifest rows.

Expected layout (raw dataset folder `12_ODSS`, per `_odss_real_files.txt` / hash
audit):

    odss/natural/<source_dataset>/<speaker>/<utt>.wav             (bona fide)
    odss/fastpitch-hifigan/<source_dataset>/<speaker>/<utt>.wav    (spoof)
    odss/vits/<source_dataset>/<speaker>/<utt>.wav                 (spoof)

`<source_dataset>` in {hifi-tts, hui-acg, openslr-es, vctk}. The "CLEAN" filter --
verified against the project's own `_embcache_xlsr300m/12_ODSS_CLEAN/_done.txt`
(26,954 rows vs. 30,025 in the unfiltered `12_ODSS` embedding cache) -- drops
`odss/natural/vctk/**` (3,071 files) only. Those bona-fide files are literally the
standalone VCTK corpus's own recordings re-used as ODSS's TTS-prompt source speech,
so keeping them would duplicate bona-fide identity across two pool corpora -- the
same §4.4 bona-only-confound concern already documented for VCTK's slight negative
inclusion effect in docs/mix_sweep_v2_findings.md. `odss/vits/vctk/**` (3,071 spoof
files, synthetic audio, not a duplicate of anything) is kept; `odss/fastpitch-hifigan`
never covered vctk source text to begin with (7,961 files vs. 11,032 for natural/vits).
"""

from __future__ import annotations

from pathlib import Path

from ..manifest import ManifestRow

AUDIO_EXTS = {".wav", ".flac"}
EXCLUDED_SOURCE_DATASETS = {"vctk"}  # bona-only confound vs. the standalone VCTK corpus
LANGUAGE_BY_SOURCE_DATASET = {"hifi-tts": "en", "hui-acg": "de", "openslr-es": "es", "vctk": "en"}
GENERATOR_DIRS_SPOOF = {"fastpitch-hifigan", "vits"}
GENERATOR_DIR_BONA = "natural"


def convert(
    root: str | Path,
    path_prefix: str = "datasets/12_ODSS",
    corpus: str = "odssclean",
    split: str = "train",
) -> list[ManifestRow]:
    """Read a `12_ODSS/odss/{natural,fastpitch-hifigan,vits}` tree and return rows.

    Args:
        root: path to the `12_ODSS` folder (containing `odss/`).
        path_prefix: dataset folder the manifest `path` column resolves from.
        corpus: unified corpus id.
        split: every row gets this split; ODSS_CLEAN has no external train/val/test file.
    """
    root = Path(root)
    odss_root = root / "odss"
    generator_dirs = sorted(p for p in odss_root.iterdir() if p.is_dir()) if odss_root.exists() else []
    if not generator_dirs:
        raise FileNotFoundError(f"No odss/<generator> directories found under {odss_root}")

    rows: list[ManifestRow] = []
    n_excluded_vctk = 0
    for gen_dir in generator_dirs:
        is_bona = gen_dir.name == GENERATOR_DIR_BONA
        if not is_bona and gen_dir.name not in GENERATOR_DIRS_SPOOF:
            continue
        for source_dir in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
            source_dataset = source_dir.name
            audio_files = sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
            if is_bona and source_dataset in EXCLUDED_SOURCE_DATASETS:
                # Confound is specifically duplicate BONA-FIDE identity vs. the
                # standalone VCTK corpus; the synthetic vits/vctk spoof audio is not
                # a duplicate of anything and is kept (verified against
                # _embcache_xlsr300m/12_ODSS_CLEAN/_done.txt: exactly the 3,071
                # natural/vctk files are absent, vits/vctk is untouched).
                n_excluded_vctk += len(audio_files)
                continue
            language = LANGUAGE_BY_SOURCE_DATASET.get(source_dataset, "NA")
            for p in audio_files:
                speaker = p.parent.name
                speaker_id = f"{source_dataset}-{speaker}"
                rel = p.relative_to(root).as_posix()
                full_path = f"{path_prefix.rstrip('/')}/{rel}"
                target = 0 if is_bona else 1
                rows.append(
                    ManifestRow(
                        utt_id=f"{corpus}/{rel}",
                        path=full_path,
                        target=target,
                        corpus=corpus,
                        split=split,
                        attack="bonafide" if is_bona else gen_dir.name,
                        bona_fide_source=source_dataset if is_bona else "na",
                        source_id=speaker_id,
                        speaker_id=speaker_id,
                        generator_id="NA" if is_bona else gen_dir.name,
                        language=language,
                    )
                )
    if not rows:
        raise ValueError(f"{odss_root}: 0 usable rows -- refusing to write manifest")
    print(f"[odss_clean] {odss_root}: kept {len(rows)} rows, "
          f"excluded {n_excluded_vctk} natural/vctk rows (confound policy)")
    return rows
