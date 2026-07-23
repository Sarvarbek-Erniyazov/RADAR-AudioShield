"""CPU-only smoke test for scripts/legacy/_extract_wavlm.py -- no network, no
real WavLM weights, no GPU.

The model loader is dependency-injected with a tiny randomly-initialized WavLM
built from WavLMConfig (2 layers, hidden 32), so the whole extraction path --
manifest listing -> audio load -> masked per-layer time-mean forward -> sharded
.npz write -> resume -> corrupt-file skip -- runs end to end on CPU in seconds.
Also proves the produced shards are readable by run_reliance_battery.py's
load_corpus_embeddings (the binding cache-space consumer contract).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "legacy"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import _extract_wavlm as ew  # noqa: E402
from run_reliance_battery import load_corpus_embeddings  # noqa: E402

CORPUS = "replaydf"
CORPUS_DIR = "04_ReplayDF"          # what corpus_dir_from_paths derives for replaydf
MANIFEST_COLS = ["utt_id", "path", "target", "corpus", "split", "attack", "bona_fide_source"]


@pytest.fixture
def tiny_model():
    """A tiny, randomly-initialized WavLM: 2 hidden layers (-> 3 hidden states),
    hidden size 32, standard 7-layer conv frontend (so the ~320x downsample and
    _get_feat_extract_output_lengths are exercised for real)."""
    from transformers import WavLMConfig, WavLMModel
    cfg = WavLMConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        conv_dim=(32, 32, 32, 32, 32, 32, 32), conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
        num_buckets=16, max_bucket_distance=64,
    )
    return WavLMModel(cfg).eval(), cfg


def _write_corpus(tmp_path, n_good=6, add_corrupt=True):
    """Write synthetic wavs under a datasets/<CORPUS_DIR>/ tree plus a synthetic
    manifest CSV whose `path` column uses the real "datasets/<CORPUS_DIR>/<rel>"
    format. Returns (data_root, rows)."""
    data_root = tmp_path / "data"
    audio_dir = data_root / "datasets" / CORPUS_DIR / "wav"
    audio_dir.mkdir(parents=True)
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n_good):
        rel = f"wav/clip_{i:03d}.wav"
        n = 8000 + i * 500  # varying length -> exercises feat-length masking
        sf.write(data_root / "datasets" / CORPUS_DIR / rel,
                 rng.standard_normal(n).astype("float32") * 0.05, ew.SR)
        rows.append(dict(utt_id=f"{CORPUS}/{rel}", path=f"datasets/{CORPUS_DIR}/{rel}",
                         target=i % 2, corpus=CORPUS, split="test",
                         attack="na", bona_fide_source="na"))
    if add_corrupt:
        rel = "wav/bad.wav"
        (data_root / "datasets" / CORPUS_DIR / rel).write_bytes(b"not a real wav file")
        rows.append(dict(utt_id=f"{CORPUS}/{rel}", path=f"datasets/{CORPUS_DIR}/{rel}",
                         target=0, corpus=CORPUS, split="test",
                         attack="na", bona_fide_source="na"))
    return data_root, rows


def _run(model, data_root, rows, out_root):
    stats = {"n": 0, "sec": 0.0, "t0": 0.0, "cal": False}
    ew.process(model, "cpu", CORPUS, rows, data_root, out_root, stats)
    return stats


@pytest.fixture(autouse=True)
def _cpu_friendly_tunables(monkeypatch):
    # small batch/shard to exercise batching + multi-shard writes; 0 workers so
    # no DataLoader subprocess spawn on Windows CI.
    monkeypatch.setattr(ew, "WORKERS", 0)
    monkeypatch.setattr(ew, "BATCH", 2)
    monkeypatch.setattr(ew, "SHARD", 3)


def test_shards_written_with_expected_schema_dtype_and_layer_count(tmp_path, tiny_model):
    model, cfg = tiny_model
    data_root, rows = _write_corpus(tmp_path, n_good=6, add_corrupt=True)
    out_root = tmp_path / "_embcache_wavlm_large"

    _run(model, data_root, rows, out_root)

    cdir = out_root / CORPUS_DIR
    shards = sorted(cdir.glob("shard_*.npz"))
    assert shards, "no shard_*.npz written"

    total = 0
    for sp in shards:
        with np.load(sp, allow_pickle=False) as d:
            assert set(d.files) == {"paths", "emb", "dur"}
            emb = d["emb"]
            assert emb.ndim == 3
            assert emb.shape[1] == cfg.num_hidden_layers + 1  # 25 for real WavLM-Large
            assert emb.dtype == np.float16
            assert emb.shape[2] == cfg.hidden_size
            assert d["paths"].shape[0] == emb.shape[0] == d["dur"].shape[0]
            total += emb.shape[0]
    assert total == 6  # the 6 good clips; the corrupt one is not embedded

    # stored paths are the manifest-stripped rels (join contract), never the
    # "datasets/<CORPUS_DIR>/" prefixed manifest path.
    all_paths = np.concatenate([np.load(sp, allow_pickle=False)["paths"] for sp in shards])
    assert set(all_paths) == {f"wav/clip_{i:03d}.wav" for i in range(6)}
    # exact format the real XLS-R shard uses: corpus-dir-relative, forward
    # slashes, no "datasets/" prefix, not absolute.
    assert all(not p.startswith("datasets/") for p in all_paths)
    assert all("\\" not in p for p in all_paths)
    assert all(not p.startswith("/") for p in all_paths)


def test_startup_aborts_loudly_on_mis_formatted_stored_path(tmp_path, tiny_model):
    """The stored-path format self-check must fail before extraction if a
    manifest path would produce a stored rel that breaks the join (e.g. a
    backslash the battery's forward-slash join would never match)."""
    model, _ = tiny_model
    data_root, rows = _write_corpus(tmp_path, n_good=6, add_corrupt=False)
    # a path that strips to a backslash-bearing rel. Materialize the file so the
    # existence self-check (1) passes and it's specifically the format check (2)
    # that fires.
    weird_path = f"datasets/{CORPUS_DIR}/wav\\weird.wav"
    resolved = ew.resolve_audio_path(data_root, weird_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(resolved), np.zeros(8000, dtype="float32"), ew.SR)
    rows.append(dict(utt_id=f"{CORPUS}/weird", path=weird_path,
                     target=0, corpus=CORPUS, split="test", attack="na", bona_fide_source="na"))
    out_root = tmp_path / "_embcache_wavlm_large"
    with pytest.raises(RuntimeError, match="required cache format"):
        _run(model, data_root, rows, out_root)


def test_corrupt_file_is_skipped_without_aborting(tmp_path, tiny_model):
    model, _ = tiny_model
    data_root, rows = _write_corpus(tmp_path, n_good=6, add_corrupt=True)
    out_root = tmp_path / "_embcache_wavlm_large"

    _run(model, data_root, rows, out_root)

    skip_p = out_root / CORPUS_DIR / "_skipped.txt"
    assert skip_p.exists(), "corrupt file did not produce a _skipped.txt"
    skipped = skip_p.read_text(encoding="utf-8")
    assert "wav/bad.wav" in skipped
    # the corrupt clip is recorded as done (so resume never retries it forever)
    done = (out_root / CORPUS_DIR / "_done.txt").read_text(encoding="utf-8").splitlines()
    assert "wav/bad.wav" in done


def test_resume_second_run_adds_nothing(tmp_path, tiny_model):
    model, _ = tiny_model
    data_root, rows = _write_corpus(tmp_path, n_good=6, add_corrupt=True)
    out_root = tmp_path / "_embcache_wavlm_large"
    cdir = out_root / CORPUS_DIR

    _run(model, data_root, rows, out_root)
    shards_before = sorted(p.name for p in cdir.glob("shard_*.npz"))
    mtimes_before = {p.name: p.stat().st_mtime_ns for p in cdir.glob("shard_*.npz")}

    _run(model, data_root, rows, out_root)  # everything already in _done.txt
    shards_after = sorted(p.name for p in cdir.glob("shard_*.npz"))
    mtimes_after = {p.name: p.stat().st_mtime_ns for p in cdir.glob("shard_*.npz")}

    assert shards_after == shards_before
    assert mtimes_after == mtimes_before  # not rewritten


def test_load_corpus_embeddings_reads_the_produced_shards(tmp_path, tiny_model):
    model, cfg = tiny_model
    data_root, rows = _write_corpus(tmp_path, n_good=6, add_corrupt=True)
    out_root = tmp_path / "_embcache_wavlm_large"

    _run(model, data_root, rows, out_root)

    # the binding consumer contract: cache_root/<corpus_dir>/shard_*.npz, emb[:, layer, :]
    paths, emb = load_corpus_embeddings(out_root, CORPUS_DIR, layer=0)
    assert emb.shape == (6, cfg.hidden_size)
    assert emb.dtype == np.float32  # load_corpus_embeddings upcasts f16 -> f32
    assert set(paths) == {f"wav/clip_{i:03d}.wav" for i in range(6)}

    # a layer index past the cached layer count must fail loudly, per that
    # function's own contract (proves the shards carry the real layer axis).
    with pytest.raises(ValueError):
        load_corpus_embeddings(out_root, CORPUS_DIR, layer=cfg.num_hidden_layers + 5)


def test_model_loader_is_dependency_injectable(tmp_path, tiny_model, monkeypatch):
    """load_model is the single seam the GPU entrypoint uses; prove it can be
    swapped so no real weights/network are ever needed in the test path."""
    model, _ = tiny_model
    monkeypatch.setattr(ew, "load_model", lambda dev: model)
    assert ew.load_model("cpu") is model
