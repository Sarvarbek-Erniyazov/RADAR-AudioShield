"""Interop tests between scripts/extract_model_embeddings.py (the
extractor) and scripts/run_reliance_modelspace.py (the consumer) --
step3_modelspace_hardening_addendum.md Finding 1.

Drives the REAL extractor (with a fake, no-backbone model swapped in via
monkeypatch, matching tests/test_extract_model_embeddings.py's own
established convention) to write a tiny cache under the CANONICAL flat
checkpoint-file naming, then runs the REAL, unmodified consumer main()
pointed at that same output -- proving the two scripts' path conventions
actually meet (a battery populates, not skips), and that
--require-all-checkpoints converts a naming/cache miss into a loud,
actionable error instead of a silent skip.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import extract_model_embeddings as eme  # noqa: E402
import run_reliance_modelspace as rrm  # noqa: E402


class _FakeBinary:
    def __init__(self, dim):
        self.fc = torch.nn.Linear(dim, 1)


class _FakeModel:
    """Exposes exactly what extract_checkpoint_corpus needs (.embed()) and
    what embedding_dim_of reads (.binary.fc.in_features) -- a deterministic
    function of the waveform, so downstream reliance metrics are
    well-defined (not needed for this test's own assertions, but keeps the
    fixture reusable if that changes)."""

    def __init__(self, dim=6):
        self.binary = _FakeBinary(dim)
        self._dim = dim

    def embed(self, waveform):
        return waveform[:, : self._dim]


def _write_manifest(manifest_dir: Path, data_root: Path, corpus: str, corpus_dir: str,
                     n: int, n_groups: int, seed: int) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (data_root / "datasets" / corpus_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rel = f"datasets/{corpus_dir}/f{i:04d}.wav"
        sf.write(data_root / rel, rng.normal(scale=0.01, size=1600).astype("float32"), 16000)
        rows.append(dict(
            utt_id=f"{corpus}/f{i:04d}", path=rel, target=int(rng.integers(0, 2)), corpus=corpus,
            split="train", attack="na", bona_fide_source="na", source_id=f"src{i % n_groups}",
            speaker_id="NA", generator_id=f"gen{i % 4}", channel_id="NA", language="NA", platform_id="NA",
        ))
    pd.DataFrame(rows).to_csv(manifest_dir / f"{corpus}.csv", index=False)


def _write_flat_checkpoint(ckpt_dir: Path, run: str, w: np.ndarray) -> Path:
    """The CANONICAL flat naming both run_reliance_battery.py's
    load_all_checkpoints and run_reliance_modelspace.py's main() already
    use: <ckpt_dir>/runs_<run>_best.pt -- unique stem per run, no
    --run-name required for the extractor to meet the consumer's own
    lookup convention."""
    ckpt_path = ckpt_dir / f"runs_{run}_best.pt"
    state = {"binary.fc.weight": torch.tensor(w, dtype=torch.float32).reshape(1, -1),
             "binary.fc.bias": torch.tensor([0.0], dtype=torch.float32)}
    torch.save({"model": state, "cfg": {"model": {}, "experiment": {"sample_rate": 16000, "duration_seconds": 0.1}}},
               ckpt_path)
    return ckpt_path


@pytest.fixture
def interop_scene(tmp_path, monkeypatch):
    """Real manifest + real (fake-model) extraction output under the
    canonical flat naming, for two checkpoints -- everything the
    interop/strict-mode tests below share."""
    monkeypatch.chdir(tmp_path)
    dim, n, n_groups = 6, 60, 12
    manifest_dir = tmp_path / "manifests"
    data_root = tmp_path / "data"
    _write_manifest(manifest_dir, data_root, "diffssd", "03_DiffSSD", n=n, n_groups=n_groups, seed=7)

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    cache_root = tmp_path / "_embcache_modelspace"
    runs = ("e007_A_fresh", "e007_B_fresh")

    monkeypatch.setattr(eme, "_build_model_from_checkpoint",
                         lambda path, device: (_FakeModel(dim), {"experiment": {"sample_rate": 16000,
                                                                                "duration_seconds": 0.1}}, {}))

    for i, run in enumerate(runs):
        w = np.random.default_rng(i).normal(size=dim)
        ckpt_path = _write_flat_checkpoint(ckpt_dir, run, w)
        # THE CANONICAL, contract-satisfying invocation: extract from the
        # flat checkpoint file with NO --run-name -- its own .stem
        # ("runs_<run>_best") is already exactly what the consumer looks
        # for, so the two scripts' directory names meet without needing
        # to pass anything to disambiguate.
        eme.main([
            "--checkpoint", str(ckpt_path),
            "--corpus", "diffssd",
            "--manifest-dir", str(manifest_dir),
            "--data-root", str(data_root),
            "--out-root", str(cache_root),
        ])

    return dict(tmp_path=tmp_path, manifest_dir=manifest_dir, ckpt_dir=ckpt_dir, cache_root=cache_root, runs=runs)


def test_extractor_output_under_canonical_naming_is_findable_by_its_own_convention(interop_scene):
    """Sanity check on the fixture itself: the extractor really did write
    under <out-root>/runs_<run>_best/<corpus_dir>/, not some other layout."""
    for run in interop_scene["runs"]:
        shard_dir = interop_scene["cache_root"] / f"runs_{run}_best" / "03_DiffSSD"
        assert sorted(shard_dir.glob("shard_*.npz")), f"no shards written under {shard_dir}"


def test_consumer_populates_battery_from_canonically_named_extractor_output(interop_scene):
    """THE interop proof: the real consumer main(), pointed at the real
    extractor's real output (no hand-built synthetic shards), produces a
    POPULATED battery -- not a "fewer than 2 checkpoints, skipping" no-op
    -- proving the two scripts' path conventions actually meet."""
    out_path = interop_scene["tmp_path"] / "reliance_modelspace.json"
    rrm.main([
        "--model-space-cache-root", str(interop_scene["cache_root"]),
        "--manifest-dir", str(interop_scene["manifest_dir"]),
        "--ckpt-dir", str(interop_scene["ckpt_dir"]),
        "--checkpoints", *interop_scene["runs"],
        "--corpus", "diffssd",
        "--factor", "generator_id",
        "--ranks", "1", "2",
        "--n-boot", "0",
        "--seed", "13",
        "--out", str(out_path),
    ])

    assert out_path.exists()
    import json
    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(manifest["batteries"]) == 1
    battery = manifest["batteries"][0]
    assert "skipped" not in battery, f"battery was skipped, not populated: {battery}"
    per_ckpt = battery["estimators"]["lda"]["fold_results"][0]["effect"]["per_checkpoint"]
    assert set(per_ckpt) == set(interop_scene["runs"])


def test_require_all_checkpoints_raises_loudly_on_a_naming_mismatch(interop_scene):
    """The self-explaining-failure half of Finding 1: pointing the
    consumer's --checkpoints at a run name whose cache the extractor never
    actually produced (a naming mismatch) must raise, under
    --require-all-checkpoints, with the exact directory it looked for and
    the exact extraction command that would produce it -- never a silent
    skip that could shrink a battery below the human's notice."""
    # "e007_C_xlsr_fresh" has a checkpoint FILE (added below) but no
    # extracted cache -- exactly the "extraction never ran / wrong name"
    # scenario Finding 1 describes.
    missing_run = "e007_C_xlsr_fresh"
    _write_flat_checkpoint(interop_scene["ckpt_dir"], missing_run, np.random.default_rng(2).normal(size=6))

    out_path = interop_scene["tmp_path"] / "reliance_modelspace_strict.json"
    with pytest.raises(RuntimeError, match="require-all-checkpoints"):
        rrm.main([
            "--model-space-cache-root", str(interop_scene["cache_root"]),
            "--manifest-dir", str(interop_scene["manifest_dir"]),
            "--ckpt-dir", str(interop_scene["ckpt_dir"]),
            "--checkpoints", *interop_scene["runs"], missing_run,
            "--corpus", "diffssd",
            "--factor", "generator_id",
            "--require-all-checkpoints",
            "--out", str(out_path),
        ])


def test_require_all_checkpoints_raises_on_missing_checkpoint_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest_dir = tmp_path / "manifests"
    data_root = tmp_path / "data"
    _write_manifest(manifest_dir, data_root, "diffssd", "03_DiffSSD", n=10, n_groups=4, seed=1)

    with pytest.raises(RuntimeError, match="require-all-checkpoints"):
        rrm.main([
            "--model-space-cache-root", str(tmp_path / "_embcache_modelspace"),
            "--manifest-dir", str(manifest_dir),
            "--ckpt-dir", str(tmp_path / "checkpoints"),  # doesn't exist at all
            "--checkpoints", "e007_A_fresh",
            "--corpus", "diffssd",
            "--require-all-checkpoints",
            "--out", str(tmp_path / "out.json"),
        ])


def test_without_strict_mode_the_same_naming_mismatch_only_warns_and_skips(interop_scene):
    """Confirms the DEFAULT (lenient) behavior is unchanged -- strict mode
    is opt-in, so existing synthetic/preflight-style callers keep their
    tolerant skip-and-continue behavior."""
    missing_run = "e007_C_xlsr_fresh"
    _write_flat_checkpoint(interop_scene["ckpt_dir"], missing_run, np.random.default_rng(2).normal(size=6))

    out_path = interop_scene["tmp_path"] / "reliance_modelspace_lenient.json"
    rrm.main([
        "--model-space-cache-root", str(interop_scene["cache_root"]),
        "--manifest-dir", str(interop_scene["manifest_dir"]),
        "--ckpt-dir", str(interop_scene["ckpt_dir"]),
        "--checkpoints", *interop_scene["runs"], missing_run,
        "--corpus", "diffssd",
        "--factor", "generator_id",
        "--ranks", "1", "2",
        "--n-boot", "0",
        "--out", str(out_path),
    ])  # must not raise

    import json
    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    battery = manifest["batteries"][0]
    assert "skipped" not in battery
    # the missing run's cache was never found -- only the other two contribute
    per_ckpt = battery["estimators"]["lda"]["fold_results"][0]["effect"]["per_checkpoint"]
    assert set(per_ckpt) == set(interop_scene["runs"])
    assert missing_run not in per_ckpt
