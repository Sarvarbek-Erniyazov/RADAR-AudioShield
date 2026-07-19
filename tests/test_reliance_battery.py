"""Tests for scripts/run_reliance_battery.py -- synthetic fixtures only, no
real embedding cache or checkpoints (those live on the collaborator machine)."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_reliance_battery import (  # noqa: E402
    groups_from_column,
    join_cache_to_manifest,
    load_corpus_embeddings,
    load_task_direction,
    ranks_for_n_levels,
    run_battery,
    select_battery_rows,
    strip_cache_prefix,
)


# ---------------------------------------------------------------------------
# join logic: prefix stripping + cache-extras dropping
# ---------------------------------------------------------------------------


def test_strip_cache_prefix_strips_matching_prefix():
    assert strip_cache_prefix("datasets/03_DiffSSD/generated_speech/gradtts/s0.wav", "03_DiffSSD") == \
        "generated_speech/gradtts/s0.wav"


def test_strip_cache_prefix_returns_none_for_non_matching_prefix():
    assert strip_cache_prefix("datasets/OTHER_CORPUS/x.wav", "03_DiffSSD") is None


def _manifest_df(n=5, corpus_dir="03_DiffSSD"):
    return pd.DataFrame({
        "path": [f"datasets/{corpus_dir}/gen/f{i}.wav" for i in range(n)],
        "target": [1, 1, 1, 0, 0][:n],
        "generator_id": ["a", "b", "a", "NA", "NA"][:n],
        "source_id": ["s1", "s2", "s1", "NA", "NA"][:n],
    })


def test_join_drops_cache_extras_and_reports_stats():
    manifest_df = _manifest_df(n=5)
    # cache has f0..f3 plus two extras not present in the manifest
    cache_paths = np.array([f"gen/f{i}.wav" for i in range(4)] + ["gen/extra1.wav", "gen/extra2.wav"])
    cache_emb = np.arange(6 * 8).reshape(6, 8).astype(np.float32)

    joined_df, joined_emb, stats = join_cache_to_manifest(cache_paths, cache_emb, manifest_df, "03_DiffSSD")

    assert stats == dict(n_cache=6, n_manifest=5, n_joined=4, n_dropped=2)
    assert len(joined_df) == 4
    assert joined_emb.shape == (4, 8)
    # f4 (in manifest, not in cache) must not appear
    assert "datasets/03_DiffSSD/gen/f4.wav" not in set(joined_df["path"])
    # row alignment: f0's embedding must be cache row 0, not some other row
    f0_row = joined_df.index[joined_df["path"] == "datasets/03_DiffSSD/gen/f0.wav"][0]
    np.testing.assert_array_equal(joined_emb[f0_row], cache_emb[0])


def test_join_asserts_nonzero_when_nothing_matches():
    manifest_df = _manifest_df(n=3, corpus_dir="03_DiffSSD")
    cache_paths = np.array(["totally/different/path.wav"])
    cache_emb = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(AssertionError):
        join_cache_to_manifest(cache_paths, cache_emb, manifest_df, "03_DiffSSD")


def test_join_drops_manifest_rows_with_no_cache_match():
    manifest_df = _manifest_df(n=5)
    cache_paths = np.array(["gen/f0.wav", "gen/f1.wav"])  # only 2 of 5 manifest rows have embeddings
    cache_emb = np.arange(2 * 4).reshape(2, 4).astype(np.float32)
    joined_df, joined_emb, stats = join_cache_to_manifest(cache_paths, cache_emb, manifest_df, "03_DiffSSD")
    assert stats["n_joined"] == 2
    assert stats["n_dropped"] == 0  # no cache-extras here; the cache is a strict subset of the manifest


# ---------------------------------------------------------------------------
# NA-exclusion + row_filter
# ---------------------------------------------------------------------------


def test_select_battery_rows_excludes_na_factor_rows():
    df = pd.DataFrame({
        "generator_id": ["a", "NA", "b", "a"],
        "target": [1, 1, 0, 1],
        "source_id": ["s1", "s2", "NA", "s1"],
    })
    emb = np.arange(4 * 3).reshape(4, 3).astype(np.float32)
    spec = dict(factor="generator_id", grouping="source_id", corpus="diffssd")

    Z, factor, y, groups = select_battery_rows(df, emb, spec)

    assert len(Z) == 3
    assert "NA" not in factor
    assert list(factor) == ["a", "b", "a"]
    np.testing.assert_array_equal(Z[0], emb[0])
    np.testing.assert_array_equal(Z[1], emb[2])
    np.testing.assert_array_equal(Z[2], emb[3])


def test_select_battery_rows_applies_row_filter_before_na_exclusion():
    df = pd.DataFrame({
        "generator_id": ["openvoicev2", "gradtts", "openvoicev2", "openvoicev2"],
        "language": ["en-au", "en", "NA", "en-us"],
        "target": [1, 1, 1, 1],
        "speaker_id": ["sp1", "sp2", "sp1", "sp3"],
    })
    emb = np.arange(4 * 3).reshape(4, 3).astype(np.float32)
    spec = dict(factor="language", grouping="speaker_id", corpus="diffssd",
                row_filter=("generator_id", "openvoicev2"))

    Z, factor, y, groups = select_battery_rows(df, emb, spec)

    # row 1 (gradtts) excluded by row_filter; row 2 (openvoicev2 but language=NA) excluded by NA rule
    assert len(Z) == 2
    assert list(factor) == ["en-au", "en-us"]


def test_groups_from_column_replaces_na_with_unique_tokens():
    g = groups_from_column(np.array(["a", "NA", "b", "NA", "a"], dtype=object))
    assert g[0] == "a" and g[4] == "a"
    assert g[1] != g[3]  # the two NA rows must NOT co-cluster
    assert g[1] == "__ungrouped_1" and g[3] == "__ungrouped_3"


# ---------------------------------------------------------------------------
# rank capping at n_levels - 1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_levels,expected", [
    (10, [1, 2, 3, 5, 8]),
    (4, [1, 2, 3]),
    (6, [1, 2, 3, 5]),
    (109, [1, 2, 3, 5, 8, 12, 16, 24]),
    (2, [1]),
    (1, []),
])
def test_ranks_for_n_levels_caps_correctly(n_levels, expected):
    assert ranks_for_n_levels([1, 2, 3, 5, 8, 12, 16, 24], n_levels) == expected


# ---------------------------------------------------------------------------
# layer-index bounds error
# ---------------------------------------------------------------------------


def _write_shard(shard_dir: Path, n=20, n_layers=5, d=8, name="shard_0000.npz"):
    shard_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n, n_layers, d)).astype(np.float16)
    paths = np.array([f"gen/f{i}.wav" for i in range(n)])
    np.savez(shard_dir / name, emb=emb, paths=paths)


def test_load_corpus_embeddings_valid_layer(tmp_path):
    _write_shard(tmp_path / "03_DiffSSD", n_layers=5)
    paths, emb = load_corpus_embeddings(tmp_path, "03_DiffSSD", layer=2)
    assert paths.shape == (20,)
    assert emb.shape == (20, 8)
    assert emb.dtype == np.float32


def test_load_corpus_embeddings_raises_on_out_of_range_layer(tmp_path):
    _write_shard(tmp_path / "03_DiffSSD", n_layers=5)
    with pytest.raises(ValueError, match="out of range"):
        load_corpus_embeddings(tmp_path, "03_DiffSSD", layer=99)


def test_load_corpus_embeddings_raises_on_negative_layer(tmp_path):
    _write_shard(tmp_path / "03_DiffSSD", n_layers=5)
    with pytest.raises(ValueError, match="out of range"):
        load_corpus_embeddings(tmp_path, "03_DiffSSD", layer=-1)


def test_load_corpus_embeddings_raises_when_no_shards_found(tmp_path):
    (tmp_path / "03_DiffSSD").mkdir()
    with pytest.raises(FileNotFoundError):
        load_corpus_embeddings(tmp_path, "03_DiffSSD", layer=2)


def test_load_corpus_embeddings_concatenates_multiple_shards(tmp_path):
    _write_shard(tmp_path / "03_DiffSSD", n=10, n_layers=5, name="shard_0000.npz")
    _write_shard(tmp_path / "03_DiffSSD", n=7, n_layers=5, name="shard_0001.npz")
    paths, emb = load_corpus_embeddings(tmp_path, "03_DiffSSD", layer=1)
    assert paths.shape == (17,)
    assert emb.shape == (17, 8)


# ---------------------------------------------------------------------------
# checkpoint loading / layer-mismatch flagging
# ---------------------------------------------------------------------------


def _write_checkpoint(path: Path, layer_center=10, layer_band=(8, 11)):
    sd = dict(
        model={"binary.fc.weight": torch.randn(1, 32), "binary.fc.bias": torch.randn(1)},
        cfg={"model": {"layer_weight_init_center": layer_center, "layer_weight_init_band": list(layer_band)}},
    )
    torch.save(sd, path)


def test_load_task_direction_flags_mismatch(tmp_path):
    ckpt = tmp_path / "runs_e007_A_fresh_best.pt"
    _write_checkpoint(ckpt, layer_center=10)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = load_task_direction(ckpt, requested_layer=9)
    assert out["w_layer_mismatch"] is True
    assert out["ckpt_layer_center"] == 10
    assert out["w"].shape == (32,)
    assert any("layer" in str(w.message).lower() for w in caught)


def test_load_task_direction_no_mismatch_when_layer_matches(tmp_path):
    ckpt = tmp_path / "runs_e007_A_fresh_best.pt"
    _write_checkpoint(ckpt, layer_center=9)
    out = load_task_direction(ckpt, requested_layer=9)
    assert out["w_layer_mismatch"] is False


def test_load_task_direction_raises_when_no_classifier_key_found(tmp_path):
    ckpt = tmp_path / "bad.pt"
    torch.save(dict(model={"unrelated.weight": torch.randn(3, 3)}, cfg={}), ckpt)
    with pytest.raises(RuntimeError, match="no classifier weight"):
        load_task_direction(ckpt, requested_layer=9)


def test_load_task_direction_mismatch_when_no_layer_info_present(tmp_path):
    ckpt = tmp_path / "no_cfg.pt"
    torch.save(dict(model={"binary.fc.weight": torch.randn(1, 16)}, cfg={}), ckpt)
    out = load_task_direction(ckpt, requested_layer=9)
    assert out["w_layer_mismatch"] is True
    assert out["ckpt_layer_center"] is None


# ---------------------------------------------------------------------------
# every removal metric has a matching control entry
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_battery_data():
    rng = np.random.default_rng(13)
    d, n, k_factor, n_groups = 12, 240, 3, 40
    w_true = rng.normal(size=d)
    w_true /= np.linalg.norm(w_true)
    M = rng.normal(size=(d, k_factor))
    M = M - np.outer(w_true, w_true @ M)
    U_true, _, _ = np.linalg.svd(M, full_matrices=False)
    U_true = U_true[:, :k_factor]

    groups_raw = rng.integers(0, n_groups, size=n)
    group_offset = rng.normal(scale=0.5, size=(n_groups, d))[groups_raw]
    y = rng.integers(0, 2, size=n)
    factor_levels = rng.integers(0, 4, size=n)
    raw_centers = rng.normal(size=(4, k_factor))
    raw_centers -= raw_centers.mean(axis=0, keepdims=True)
    Uc, Sc, Vtc = np.linalg.svd(raw_centers, full_matrices=False)
    factor_centers = Uc @ np.diag(np.full_like(Sc, 3.0)) @ Vtc

    Z = (np.outer((y * 2 - 1).astype(float), w_true) * 3.0
         + factor_centers[factor_levels] @ U_true.T
         + group_offset
         + rng.normal(size=(n, d)))
    factor = np.array([f"gen{i}" for i in factor_levels], dtype=object)
    groups = np.array([f"grp{i}" for i in groups_raw], dtype=object)
    checkpoints = {
        "e007_A_fresh": dict(w=w_true, b=0.0, ckpt_layer_center=10, ckpt_layer_band=[8, 11], w_layer_mismatch=True),
    }
    return dict(Z=Z, factor=factor, y=y, groups=groups, checkpoints=checkpoints)


def test_every_removal_metric_has_a_matching_control(synthetic_battery_data):
    d = synthetic_battery_data
    spec = dict(name="synthetic_test", corpus="diffssd", factor="generator_id", grouping="source_id")
    result = run_battery(
        spec, d["Z"], d["factor"], d["y"], d["groups"], d["checkpoints"],
        ranks=[1, 2, 3], n_boot=30, seed=13,
    )

    assert "estimators" in result
    for estimator_name, estimator_result in result["estimators"].items():
        for fold in estimator_result["fold_results"]:
            effect = fold["effect"]

            # prediction_change (per checkpoint) -> prediction_change_control
            for run, ck_effect in effect["per_checkpoint"].items():
                assert "prediction_change" in ck_effect
                assert "prediction_change_control" in ck_effect, (
                    f"{estimator_name}/fold{fold['fold_id']}/{run}: prediction_change has no control entry"
                )
                control = ck_effect["prediction_change_control"]
                for key in ("true_effect", "random_effects", "random_mean", "random_std",
                            "task_direction_effect", "exceeds_random"):
                    assert key in control

            # LEACE / INLP erasure -> projection_removal_control (the shared,
            # U-parameterized removal-style control for this battery's factor subspace)
            assert "leace" in effect and "decodability_drop" in effect["leace"]
            assert "inlp" in effect and "decodability_drop" in effect["inlp"]
            assert "projection_removal_control" in effect
            control = effect["projection_removal_control"]
            for key in ("true_effect", "random_effects", "random_mean", "random_std",
                        "task_direction_effect", "exceeds_random"):
                assert key in control


def test_run_battery_reports_grouping_degenerate(synthetic_battery_data):
    d = synthetic_battery_data
    spec = dict(name="degenerate_test", corpus="vctk", factor="generator_id", grouping="generator_id")
    result = run_battery(
        spec, d["Z"], d["factor"], d["y"], d["factor"],  # grouping == factor column's own values
        d["checkpoints"], ranks=[1], n_boot=10, seed=13,
    )
    assert result["grouping_degenerate"] is True


def test_run_battery_skips_when_no_rank_survives_capping(synthetic_battery_data):
    d = synthetic_battery_data
    binary_factor = np.where(d["factor"] == "gen0", "gen0", "other")  # collapse to 2 levels
    spec = dict(name="tiny_test", corpus="diffssd", factor="generator_id", grouping="source_id")
    result = run_battery(
        spec, d["Z"], binary_factor, d["y"], d["groups"], d["checkpoints"],
        ranks=[5, 8], n_boot=10, seed=13,  # both > n_levels-1 == 1
    )
    assert "skipped" in result
