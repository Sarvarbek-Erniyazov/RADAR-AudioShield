"""Tests for scripts/build_counterfactuals.py -- synthetic manifest + audio in
tmp_path only, per the task: NOT run against any real corpus."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_counterfactuals import build_counterfactuals  # noqa: E402


@pytest.fixture
def synthetic_corpus(tmp_path):
    """A tiny manifest (mixed corpora/splits/targets) + matching synthetic
    16kHz wav files under a data root, both in tmp_path."""
    data_root = tmp_path / "data"
    (data_root / "toy").mkdir(parents=True)
    rng = np.random.default_rng(3)
    rows = []
    for i in range(6):
        rel = f"toy/f{i}.wav"
        sf.write(data_root / rel, (0.2 * rng.standard_normal(8000)).astype(np.float32), 16000)
        rows.append(dict(
            utt_id=rel, path=rel, target=i % 2, corpus="toyA" if i < 3 else "toyB",
            split="test" if i % 2 == 0 else "train", attack="bonafide" if i % 2 == 0 else "gen",
            bona_fide_source="toy" if i % 2 == 0 else "na",
        ))
    manifest_path = tmp_path / "toy_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path, data_root, rows


def _read_provenance(out_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (out_dir / "provenance.jsonl").read_text().splitlines()]


def test_writes_expected_files_and_provenance(synthetic_corpus, tmp_path):
    manifest_path, data_root, rows = synthetic_corpus
    out_dir = tmp_path / "out"
    doses = [20.0, 10.0, 0.0]
    summary = build_counterfactuals(
        manifest_path=manifest_path, data_root=data_root, out_dir=out_dir,
        transform_name="noise", doses=doses, seed=13,
    )
    assert summary["n_rows_selected"] == len(rows)
    assert summary["n_written"] == len(rows) * len(doses)

    wavs = sorted((out_dir / "noise").rglob("*.wav"))
    assert len(wavs) == len(rows) * len(doses)

    records = _read_provenance(out_dir)
    assert len(records) == len(rows) * len(doses)
    required = {"utt_id", "corpus", "split", "target", "source_path", "out_path",
                "transform", "family", "dose", "dose_unit", "seed", "sr", "noise_type"}
    for rec in records:
        assert required.issubset(rec)


def test_corpus_and_split_filters_reduce_selection(synthetic_corpus, tmp_path):
    manifest_path, data_root, rows = synthetic_corpus
    out_dir = tmp_path / "out"
    summary = build_counterfactuals(
        manifest_path=manifest_path, data_root=data_root, out_dir=out_dir,
        transform_name="noise", doses=[10.0], seed=13, corpora=["toyA"], splits=["test"],
    )
    expected_n = sum(1 for r in rows if r["corpus"] == "toyA" and r["split"] == "test")
    assert expected_n > 0
    assert summary["n_rows_selected"] == expected_n


def test_limit_caps_selection(synthetic_corpus, tmp_path):
    manifest_path, data_root, _ = synthetic_corpus
    out_dir = tmp_path / "out"
    summary = build_counterfactuals(
        manifest_path=manifest_path, data_root=data_root, out_dir=out_dir,
        transform_name="noise", doses=[10.0], seed=13, limit=2,
    )
    assert summary["n_rows_selected"] == 2


def test_empty_selection_raises_before_writing_anything(synthetic_corpus, tmp_path):
    manifest_path, data_root, _ = synthetic_corpus
    out_dir = tmp_path / "out"
    with pytest.raises(ValueError, match="0 manifest rows"):
        build_counterfactuals(
            manifest_path=manifest_path, data_root=data_root, out_dir=out_dir,
            transform_name="noise", doses=[10.0], seed=13, corpora=["does_not_exist"],
        )
    assert not out_dir.exists()


def test_unknown_transform_raises(synthetic_corpus, tmp_path):
    manifest_path, data_root, _ = synthetic_corpus
    with pytest.raises(ValueError, match="unknown transform"):
        build_counterfactuals(
            manifest_path=manifest_path, data_root=data_root, out_dir=tmp_path / "out",
            transform_name="bogus", doses=[10.0], seed=13,
        )


def test_rir_transform_requires_rir_root(synthetic_corpus, tmp_path):
    manifest_path, data_root, _ = synthetic_corpus
    with pytest.raises(ValueError, match="rir-root"):
        build_counterfactuals(
            manifest_path=manifest_path, data_root=data_root, out_dir=tmp_path / "out",
            transform_name="rir", doses=[0.5], seed=13,
        )


def test_reverb_transform_end_to_end(synthetic_corpus, synthetic_rir_root, tmp_path):
    manifest_path, data_root, rows = synthetic_corpus
    out_dir = tmp_path / "out"
    summary = build_counterfactuals(
        manifest_path=manifest_path, data_root=data_root, out_dir=out_dir,
        transform_name="rir", doses=[0.5], seed=13, rir_root=str(synthetic_rir_root),
    )
    assert summary["n_written"] == len(rows)
    records = _read_provenance(out_dir)
    assert all("chosen_rir_sha256" in r for r in records)


def test_codec_transform_checks_ffmpeg_before_reading_manifest(synthetic_corpus, tmp_path, monkeypatch):
    """The ffmpeg-availability check must fire before any manifest reading or
    directory creation -- fail loudly, before doing any work."""
    manifest_path, data_root, _ = synthetic_corpus
    out_dir = tmp_path / "out"

    import build_counterfactuals as bc_mod
    from audioshield.counterfactuals.codec import FfmpegNotAvailableError

    def fake_check():
        raise FfmpegNotAvailableError("ffmpeg not found (simulated)")

    monkeypatch.setattr(bc_mod, "check_ffmpeg_available", fake_check)

    with pytest.raises(FfmpegNotAvailableError):
        build_counterfactuals(
            manifest_path=manifest_path, data_root=data_root, out_dir=out_dir,
            transform_name="codec", doses=[16.0], seed=13,
        )
    assert not out_dir.exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_codec_transform_end_to_end(synthetic_corpus, tmp_path):
    manifest_path, data_root, rows = synthetic_corpus
    out_dir = tmp_path / "out"
    summary = build_counterfactuals(
        manifest_path=manifest_path, data_root=data_root, out_dir=out_dir,
        transform_name="codec", doses=[16.0], seed=13, codec="opus",
    )
    assert summary["n_written"] == len(rows)
    records = _read_provenance(out_dir)
    assert all(r["codec"] == "opus" for r in records)
