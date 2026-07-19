"""Tests for scripts/extract_model_embeddings.py -- synthetic fixtures only.

No real checkpoint, backbone, or GPU is used: constructing a real
AudioShieldX requires downloading/loading an HF backbone (network or a
local cache), which is out of scope here (see the module's own docstring
and scripts/run_reliance_battery.py's precedent -- real checkpoints live on
the collaborator machine). Model-dependent logic is instead tested through
a minimal fake object exposing exactly the attributes/methods the script
actually reads (model.embed(waveform), model.binary.fc.in_features),
dependency-injected via run_preflight's build_model_fn parameter.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_model_embeddings import (  # noqa: E402
    CheckResult,
    _corpus_dir_from_rows,
    _resolve_audio_path,
    _strip_dataset_prefix,
    _write_shard_atomic,
    check_disk_space,
    check_raw_audio_exists,
    check_torch_scipy_eigh,
    embedding_dim_of,
    extract_checkpoint_corpus,
    main,
    run_preflight,
)
from audioshield.data.manifest import ManifestRow


# ---------------------------------------------------------------------------
# small pure helpers
# ---------------------------------------------------------------------------


def test_resolve_audio_path_joins_relative_onto_data_root():
    assert _resolve_audio_path(Path("/root"), "datasets/03_DiffSSD/f.wav") == Path("/root/datasets/03_DiffSSD/f.wav")


def test_resolve_audio_path_leaves_absolute_path_unchanged():
    abs_path = str(Path("/somewhere/f.wav").resolve())
    assert _resolve_audio_path(Path("/root"), abs_path) == Path(abs_path)


def _row(path, corpus="diffssd", utt_id="u"):
    return ManifestRow(utt_id=utt_id, path=path, target=1, corpus=corpus, split="train",
                        attack="na", bona_fide_source="na")


def test_corpus_dir_from_rows_derives_from_path_column():
    rows = [_row("datasets/03_DiffSSD/a.wav"), _row("datasets/03_DiffSSD/sub/b.wav")]
    assert _corpus_dir_from_rows(rows) == "03_DiffSSD"


def test_corpus_dir_from_rows_raises_on_inconsistent_dirs():
    rows = [_row("datasets/03_DiffSSD/a.wav"), _row("datasets/04_ReplayDF/b.wav")]
    with pytest.raises(ValueError, match="exactly one"):
        _corpus_dir_from_rows(rows)


def test_strip_dataset_prefix_strips_matching_prefix():
    assert _strip_dataset_prefix("datasets/03_DiffSSD/a/b.wav", "03_DiffSSD") == "a/b.wav"


def test_strip_dataset_prefix_raises_on_mismatch():
    with pytest.raises(ValueError, match="does not start with"):
        _strip_dataset_prefix("datasets/OTHER/a.wav", "03_DiffSSD")


def test_embedding_dim_of_reads_from_model_not_hardcoded():
    """A deliberately non-256 dim proves the value comes from the model, not
    a hardcoded constant."""
    class _FakeBinary:
        class fc:
            in_features = 37

    class _FakeModel:
        binary = _FakeBinary()

    assert embedding_dim_of(_FakeModel()) == 37


# ---------------------------------------------------------------------------
# check_torch_scipy_eigh: subprocess isolation
# ---------------------------------------------------------------------------


def test_check_torch_scipy_eigh_passes_in_this_env():
    ok, detail = check_torch_scipy_eigh()
    assert ok is True
    assert "EIGH_OK" not in detail  # human-readable summary, not raw stdout
    assert "succeeded" in detail


def test_check_torch_scipy_eigh_reports_fail_without_raising_on_bad_interpreter(tmp_path):
    bad_python = tmp_path / "definitely_not_a_real_interpreter"
    ok, detail = check_torch_scipy_eigh(python_exe=str(bad_python))
    assert ok is False
    assert "could not launch" in detail




# ---------------------------------------------------------------------------
# check_raw_audio_exists / check_disk_space
# ---------------------------------------------------------------------------


def test_check_raw_audio_exists_passes_when_all_sampled_files_present(tmp_path):
    (tmp_path / "datasets" / "03_DiffSSD").mkdir(parents=True)
    rows = []
    for i in range(3):
        rel = f"datasets/03_DiffSSD/f{i}.wav"
        (tmp_path / rel).write_bytes(b"fake")
        rows.append(_row(rel, utt_id=f"u{i}"))
    ok, detail = check_raw_audio_exists(rows, tmp_path)
    assert ok is True
    assert "3/3" in detail


def test_check_raw_audio_exists_fails_when_some_missing(tmp_path):
    (tmp_path / "datasets" / "03_DiffSSD").mkdir(parents=True)
    (tmp_path / "datasets/03_DiffSSD/f0.wav").write_bytes(b"fake")
    rows = [_row("datasets/03_DiffSSD/f0.wav", utt_id="u0"),
            _row("datasets/03_DiffSSD/missing.wav", utt_id="u1")]
    ok, detail = check_raw_audio_exists(rows, tmp_path)
    assert ok is False
    assert "1/2" in detail
    assert "u1" in detail


def test_check_raw_audio_exists_no_rows_is_a_fail():
    ok, detail = check_raw_audio_exists([], Path("/nonexistent"))
    assert ok is False


def test_check_disk_space_fails_when_insufficient(tmp_path):
    ok, detail = check_disk_space(tmp_path, n_rows_total=10**15, embedding_dim=1024, dtype_bytes=4)
    assert ok is False
    assert "need" in detail


def test_check_disk_space_passes_for_tiny_estimate(tmp_path):
    ok, detail = check_disk_space(tmp_path, n_rows_total=10, embedding_dim=8, dtype_bytes=4)
    assert ok is True


# ---------------------------------------------------------------------------
# run_preflight: failure paths, no crashes, checks continue past failures
# ---------------------------------------------------------------------------


def _write_manifest(manifest_dir: Path, corpus: str, rows: list[dict]) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest_dir / f"{corpus}.csv", index=False)


def test_run_preflight_missing_checkpoint_reported_as_fail_without_crashing(tmp_path):
    manifest_dir = tmp_path / "manifests"
    _write_manifest(manifest_dir, "diffssd", [
        dict(utt_id=f"diffssd/f{i}", path=f"datasets/03_DiffSSD/f{i}.wav", target=1,
             corpus="diffssd", split="train", attack="na", bona_fide_source="na")
        for i in range(4)
    ])
    missing_ckpt = tmp_path / "nope.pt"

    results = run_preflight(
        [missing_ckpt], ["diffssd"], manifest_dir, tmp_path / "data_root", tmp_path / "out", "cpu",
    )

    by_name = {r.name: r for r in results}
    assert by_name["checkpoint_constructs[nope.pt]"].passed is False
    assert "not found" in by_name["checkpoint_constructs[nope.pt]"].detail
    # other checks still ran (didn't crash out because one checkpoint was missing)
    assert "manifest_readable[diffssd]" in by_name
    assert "raw_audio_exists[diffssd]" in by_name
    assert "disk_space" in by_name
    assert all(isinstance(r, CheckResult) for r in results)


def test_run_preflight_missing_manifest_reported_as_fail_without_crashing(tmp_path):
    results = run_preflight(
        [], ["does_not_exist"], tmp_path / "manifests", tmp_path / "data_root", tmp_path / "out", "cpu",
    )
    by_name = {r.name: r for r in results}
    assert by_name["manifest_readable[does_not_exist]"].passed is False


def test_run_preflight_uses_injected_build_model_fn_for_dim_and_forward_pass(tmp_path):
    """Dependency-injects a fake model builder so the forward-pass/dim checks
    can be exercised without a real backbone -- proves embedding_dim_of and
    check_forward_pass are wired correctly end-to-end through run_preflight."""
    manifest_dir = tmp_path / "manifests"
    data_root = tmp_path / "data"
    (data_root / "datasets" / "03_DiffSSD").mkdir(parents=True)
    rows = []
    for i in range(4):
        rel = f"datasets/03_DiffSSD/f{i}.wav"
        sf.write(data_root / rel, (np.zeros(1600, dtype="float32") + 0.01), 16000)
        rows.append(dict(utt_id=f"diffssd/f{i}", path=rel, target=1, corpus="diffssd",
                          split="train", attack="na", bona_fide_source="na"))
    _write_manifest(manifest_dir, "diffssd", rows)

    ckpt_path = tmp_path / "fake_ckpt.pt"
    ckpt_path.write_bytes(b"not a real checkpoint, never read by the fake builder")

    class _FakeBinary:
        class fc:
            in_features = 9

    class _FakeModel:
        binary = _FakeBinary()

        def embed(self, waveform):
            return torch.zeros(waveform.shape[0], 9)

    def fake_build_model_fn(path, device):
        cfg = {"experiment": {"sample_rate": 16000, "duration_seconds": 0.5}}
        return _FakeModel(), cfg, {}

    results = run_preflight(
        [ckpt_path], ["diffssd"], manifest_dir, data_root, tmp_path / "out", "cpu",
        build_model_fn=fake_build_model_fn,
    )
    by_name = {r.name: r for r in results}
    assert by_name["checkpoint_constructs[fake_ckpt.pt]"].passed is True
    assert by_name["forward_pass[fake_ckpt.pt]"].passed is True
    assert "9-d" in by_name["forward_pass[fake_ckpt.pt]"].detail


def test_main_preflight_exits_nonzero_when_checkpoint_missing(tmp_path):
    manifest_dir = tmp_path / "manifests"
    _write_manifest(manifest_dir, "diffssd", [
        dict(utt_id="diffssd/f0", path="datasets/03_DiffSSD/f0.wav", target=1,
             corpus="diffssd", split="train", attack="na", bona_fide_source="na"),
    ])
    with pytest.raises(SystemExit) as exc:
        main([
            "--preflight",
            "--checkpoint", str(tmp_path / "nope.pt"),
            "--corpus", "diffssd",
            "--manifest-dir", str(manifest_dir),
            "--data-root", str(tmp_path / "data"),
            "--out-root", str(tmp_path / "out"),
        ])
    assert exc.value.code != 0


# ---------------------------------------------------------------------------
# _write_shard_atomic: schema + atomicity
# ---------------------------------------------------------------------------


def test_write_shard_atomic_schema_and_no_leftover_tmp(tmp_path):
    shard_path = tmp_path / "shard_0000.npz"
    paths = np.array(["a.wav", "b.wav"])
    emb = np.zeros((2, 4), dtype=np.float32)
    meta = dict(checkpoint_sha256="abc", model_config_hash="def", git_sha="ghi", dtype="float32")

    _write_shard_atomic(shard_path, paths, emb, meta)

    assert shard_path.exists()
    assert not (tmp_path / "shard_0000.npz.tmp").exists()
    data = np.load(shard_path, allow_pickle=False)
    assert set(data.files) == {"paths", "emb", "meta"}
    np.testing.assert_array_equal(data["paths"], paths)
    np.testing.assert_array_equal(data["emb"], emb)
    loaded_meta = json.loads(str(data["meta"]))
    assert loaded_meta == meta


# ---------------------------------------------------------------------------
# extract_checkpoint_corpus: resume skips completed shards, real end-to-end
# audio -> embed() -> shard write, using a fake (no-backbone) model.
# ---------------------------------------------------------------------------


class _FakeEmbedModel:
    """Exposes exactly what extract_checkpoint_corpus needs: .embed(waveform)."""
    def embed(self, waveform):
        return torch.arange(waveform.shape[0] * 5, dtype=torch.float32).reshape(waveform.shape[0], 5)


def _write_diffssd_rows(data_root: Path, n: int) -> list[ManifestRow]:
    (data_root / "datasets" / "03_DiffSSD").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        rel = f"datasets/03_DiffSSD/f{i:02d}.wav"
        sf.write(data_root / rel, (np.zeros(1600, dtype="float32") + 0.01), 16000)
        rows.append(ManifestRow(utt_id=f"diffssd/f{i:02d}", path=rel, target=1, corpus="diffssd",
                                 split="train", attack="na", bona_fide_source="na"))
    return sorted(rows, key=lambda r: r.utt_id)


def test_extract_checkpoint_corpus_writes_expected_shard_count_and_schema(tmp_path):
    data_root = tmp_path / "data"
    rows = _write_diffssd_rows(data_root, n=6)
    ckpt_path = tmp_path / "fake_ckpt.pt"
    ckpt_path.write_bytes(b"fake checkpoint bytes")
    cfg = {"experiment": {"sample_rate": 16000, "duration_seconds": 0.5}}
    out_dir = tmp_path / "out"

    stats = extract_checkpoint_corpus(
        _FakeEmbedModel(), cfg, ckpt_path, "diffssd", rows, data_root, out_dir,
        device="cpu", batch_size=2, shard_size=3, dtype="float32", git_sha="deadbeef",
    )

    assert stats == dict(corpus="diffssd", corpus_dir="03_DiffSSD", n_shards=2, written=2, skipped=0, n_rows=6)
    shard_files = sorted(out_dir.glob("shard_*.npz"))
    assert [p.name for p in shard_files] == ["shard_0000.npz", "shard_0001.npz"]

    data = np.load(shard_files[0], allow_pickle=False)
    assert set(data.files) == {"paths", "emb", "meta"}
    assert data["emb"].shape == (3, 5)
    assert data["paths"].shape == (3,)
    assert all(not p.startswith("datasets/") for p in data["paths"])  # prefix stripped
    meta = json.loads(str(data["meta"]))
    assert meta["checkpoint_sha256"] == __import__("hashlib").sha256(ckpt_path.read_bytes()).hexdigest()
    assert meta["git_sha"] == "deadbeef"
    assert meta["dtype"] == "float32"
    assert meta["corpus_dir"] == "03_DiffSSD"
    assert meta["n_rows"] == 3


def test_extract_checkpoint_corpus_resume_skips_completed_shards(tmp_path):
    data_root = tmp_path / "data"
    rows = _write_diffssd_rows(data_root, n=6)
    ckpt_path = tmp_path / "fake_ckpt.pt"
    ckpt_path.write_bytes(b"fake checkpoint bytes")
    cfg = {"experiment": {"sample_rate": 16000, "duration_seconds": 0.5}}
    out_dir = tmp_path / "out"

    stats1 = extract_checkpoint_corpus(
        _FakeEmbedModel(), cfg, ckpt_path, "diffssd", rows, data_root, out_dir,
        device="cpu", batch_size=2, shard_size=3, dtype="float32", git_sha="deadbeef",
    )
    assert stats1["written"] == 2 and stats1["skipped"] == 0
    shard_files = sorted(out_dir.glob("shard_*.npz"))
    mtimes_before = {p.name: p.stat().st_mtime_ns for p in shard_files}
    contents_before = {p.name: np.load(p, allow_pickle=False)["emb"].copy() for p in shard_files}

    stats2 = extract_checkpoint_corpus(
        _FakeEmbedModel(), cfg, ckpt_path, "diffssd", rows, data_root, out_dir,
        device="cpu", batch_size=2, shard_size=3, dtype="float32", git_sha="deadbeef",
    )
    assert stats2["written"] == 0 and stats2["skipped"] == 2

    mtimes_after = {p.name: p.stat().st_mtime_ns for p in sorted(out_dir.glob("shard_*.npz"))}
    assert mtimes_before == mtimes_after  # not rewritten
    contents_after = {n: np.load(out_dir / n, allow_pickle=False)["emb"] for n in mtimes_after}
    for name in contents_before:
        np.testing.assert_array_equal(contents_before[name], contents_after[name])


def test_extract_checkpoint_corpus_raises_on_zero_rows(tmp_path):
    with pytest.raises(ValueError, match="0 rows"):
        extract_checkpoint_corpus(
            _FakeEmbedModel(), {"experiment": {}}, tmp_path / "ckpt.pt", "diffssd", [], tmp_path / "data",
            tmp_path / "out", device="cpu", batch_size=2, shard_size=3, dtype="float32", git_sha="x",
        )


def test_extract_checkpoint_corpus_partial_shard_write_is_not_resumed_from(tmp_path, monkeypatch):
    """A .tmp file left behind by a killed run must NOT be treated as a
    complete shard on the next run -- only the final (atomically renamed)
    shard_*.npz name counts."""
    data_root = tmp_path / "data"
    rows = _write_diffssd_rows(data_root, n=3)
    ckpt_path = tmp_path / "fake_ckpt.pt"
    ckpt_path.write_bytes(b"fake checkpoint bytes")
    cfg = {"experiment": {"sample_rate": 16000, "duration_seconds": 0.5}}
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    (out_dir / "shard_0000.npz.tmp").write_bytes(b"leftover partial write from a killed run")

    stats = extract_checkpoint_corpus(
        _FakeEmbedModel(), cfg, ckpt_path, "diffssd", rows, data_root, out_dir,
        device="cpu", batch_size=2, shard_size=3, dtype="float32", git_sha="deadbeef",
    )
    assert stats["written"] == 1 and stats["skipped"] == 0
    assert (out_dir / "shard_0000.npz").exists()
    assert not (out_dir / "shard_0000.npz.tmp").exists()
