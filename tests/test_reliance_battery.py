"""Tests for scripts/run_reliance_battery.py -- synthetic fixtures only, no
real embedding cache or checkpoints (those live on the collaborator machine)."""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_reliance_battery as rrb  # noqa: E402 -- needed for monkeypatching module-level names
from run_reliance_battery import (  # noqa: E402
    LAYER_LOGITS_KEY,
    _describe_pooling_mismatch,
    _find_layer_logits_key,
    _guarded_call,
    _not_estimable,
    _removal_control_without_task_direction,
    _run_with_timeout,
    _uniform_band_weights,
    groups_from_column,
    join_cache_to_manifest,
    load_corpus_embeddings,
    load_task_direction,
    main,
    pool_band_embeddings,
    ranks_for_n_levels,
    resolve_w_metrics,
    run_battery,
    select_battery_rows,
    strip_cache_prefix,
    summarize_prereg_candidate,
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

    Z, factor, y, groups, Z_full = select_battery_rows(df, emb, spec)

    assert Z_full is None
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

    Z, factor, y, groups, Z_full = select_battery_rows(df, emb, spec)

    # row 1 (gradtts) excluded by row_filter; row 2 (openvoicev2 but language=NA) excluded by NA rule
    assert len(Z) == 2
    assert list(factor) == ["en-au", "en-us"]


def test_select_battery_rows_masks_emb_full_identically_to_emb():
    df = pd.DataFrame({
        "generator_id": ["a", "NA", "b", "a"],
        "target": [1, 1, 0, 1],
        "source_id": ["s1", "s2", "NA", "s1"],
    })
    emb = np.arange(4 * 3).reshape(4, 3).astype(np.float32)
    emb_full = np.arange(4 * 2 * 3).reshape(4, 2, 3).astype(np.float32)
    spec = dict(factor="generator_id", grouping="source_id", corpus="diffssd")

    Z, factor, y, groups, Z_full = select_battery_rows(df, emb, spec, emb_full=emb_full)

    assert Z_full.shape == (3, 2, 3)
    np.testing.assert_array_equal(Z_full[0], emb_full[0])
    np.testing.assert_array_equal(Z_full[1], emb_full[2])
    np.testing.assert_array_equal(Z_full[2], emb_full[3])


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


def test_load_task_direction_fixed_mode_default_matches_explicit(tmp_path):
    """layer_mode defaults to "fixed" -- must be byte-for-byte the same call
    as passing layer_mode="fixed" explicitly (Roadmap: "fixed mode output
    unchanged from before")."""
    ckpt = tmp_path / "runs_e007_A_fresh_best.pt"
    _write_checkpoint(ckpt, layer_center=10)
    out_default = load_task_direction(ckpt, requested_layer=9)
    out_explicit = load_task_direction(ckpt, requested_layer=9, layer_mode="fixed")
    assert out_default["w_layer_mismatch"] == out_explicit["w_layer_mismatch"] is True
    assert out_default["ckpt_layer_center"] == out_explicit["ckpt_layer_center"] == 10
    assert out_default["layer_pooling"] == out_explicit["layer_pooling"] == "fixed_layer"
    assert out_default["band_weights"] is out_explicit["band_weights"] is None
    np.testing.assert_array_equal(out_default["w"], out_explicit["w"])


# ---------------------------------------------------------------------------
# checkpoint-band pooling: learned-softmax layer weights, uniform fallback
# ---------------------------------------------------------------------------


def _write_checkpoint_with_layer_logits(path: Path, logits, layer_center=10, layer_band=(8, 11), key=LAYER_LOGITS_KEY):
    sd = dict(
        model={"binary.fc.weight": torch.randn(1, 32), "binary.fc.bias": torch.randn(1),
               key: torch.as_tensor(logits, dtype=torch.float32)},
        cfg={"model": {"layer_weight_init_center": layer_center, "layer_weight_init_band": list(layer_band)}},
    )
    torch.save(sd, path)


def test_find_layer_logits_key_exact_match():
    state = {LAYER_LOGITS_KEY: torch.zeros(5), "binary.fc.weight": torch.zeros(1)}
    assert _find_layer_logits_key(state) == LAYER_LOGITS_KEY


def test_find_layer_logits_key_suffix_fallback_when_wrapped():
    state = {f"module.{LAYER_LOGITS_KEY}": torch.zeros(5)}
    assert _find_layer_logits_key(state) == f"module.{LAYER_LOGITS_KEY}"


def test_find_layer_logits_key_returns_none_when_absent():
    state = {"binary.fc.weight": torch.zeros(1)}
    assert _find_layer_logits_key(state) is None


def test_find_layer_logits_key_returns_none_when_ambiguous():
    state = {f"a.{LAYER_LOGITS_KEY}": torch.zeros(5), f"b.{LAYER_LOGITS_KEY}": torch.zeros(5)}
    assert _find_layer_logits_key(state) is None


def test_uniform_band_weights_matches_manual_computation():
    w = _uniform_band_weights((8, 11), num_layers=25)
    assert w.shape == (25,)
    np.testing.assert_allclose(w.sum(), 1.0)
    assert np.all(w[8:12] == pytest.approx(0.25))
    assert np.all(w[:8] == 0.0) and np.all(w[12:] == 0.0)


def test_load_task_direction_checkpoint_band_uses_learned_softmax(tmp_path):
    ckpt = tmp_path / "runs_e007_A_fresh_best.pt"
    logits = torch.tensor([-10.0] * 8 + [1.0, 3.0, 2.0] + [-10.0] * 14)  # length 25, concentrated at 8-10
    _write_checkpoint_with_layer_logits(ckpt, logits)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # learned_softmax path must NOT warn
        out = load_task_direction(ckpt, requested_layer=9, layer_mode="checkpoint-band", num_cache_layers=25)
    assert out["layer_pooling"] == "learned_softmax"
    assert out["w_layer_mismatch"] is False
    expected = torch.softmax(logits, dim=0).numpy()
    np.testing.assert_allclose(out["band_weights"], expected, atol=1e-6)
    assert out["band_weights"].sum() == pytest.approx(1.0)


def test_load_task_direction_checkpoint_band_falls_back_to_uniform_with_warning(tmp_path):
    ckpt = tmp_path / "runs_e007_B_fresh_best.pt"
    _write_checkpoint(ckpt, layer_center=10, layer_band=(8, 11))  # no layer_logits key at all
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = load_task_direction(ckpt, requested_layer=9, layer_mode="checkpoint-band", num_cache_layers=25)
    assert out["layer_pooling"] == "uniform_band_fallback"
    assert out["w_layer_mismatch"] is True
    np.testing.assert_allclose(out["band_weights"], _uniform_band_weights((8, 11), 25))
    assert any("uniform" in str(w.message).lower() for w in caught)


def test_load_task_direction_checkpoint_band_raises_when_nothing_to_fall_back_to(tmp_path):
    ckpt = tmp_path / "no_band.pt"
    torch.save(dict(model={"binary.fc.weight": torch.randn(1, 16)}, cfg={}), ckpt)
    with pytest.raises(RuntimeError, match="refusing to guess"):
        load_task_direction(ckpt, requested_layer=9, layer_mode="checkpoint-band", num_cache_layers=25)


def test_load_task_direction_checkpoint_band_raises_on_logits_length_mismatch(tmp_path):
    ckpt = tmp_path / "runs_e007_C_xlsr_fresh_best.pt"
    _write_checkpoint_with_layer_logits(ckpt, torch.zeros(13))  # 13 != num_cache_layers=25
    with pytest.raises(RuntimeError, match="entries"):
        load_task_direction(ckpt, requested_layer=9, layer_mode="checkpoint-band", num_cache_layers=25)


# ---------------------------------------------------------------------------
# pool_band_embeddings
# ---------------------------------------------------------------------------


def test_pool_band_embeddings_matches_weighted_sum():
    rng = np.random.default_rng(3)
    Z_full = rng.standard_normal((5, 4, 3)).astype(np.float32)  # (n=5, L=4, D=3)
    weights = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    pooled = pool_band_embeddings(Z_full, weights)

    expected = sum(weights[l] * Z_full[:, l, :] for l in range(4))
    np.testing.assert_allclose(pooled, expected, atol=1e-5)
    assert pooled.shape == (5, 3)


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
        "e007_A_fresh": dict(w=w_true, b=0.0, ckpt_layer_center=10, ckpt_layer_band=[8, 11],
                              layer_pooling="fixed_layer", band_weights=None, w_layer_mismatch=True),
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


def test_run_battery_fixed_mode_still_computes_alignment_inline(synthetic_battery_data):
    """Roadmap: "fixed mode output unchanged from before" -- alignment must
    still be populated directly by the main crossfit run (not left null for
    a post-processing pass that only exists in checkpoint-band mode)."""
    d = synthetic_battery_data
    spec = dict(name="fixed_mode_test", corpus="diffssd", factor="generator_id", grouping="source_id")
    result = run_battery(
        spec, d["Z"], d["factor"], d["y"], d["groups"], d["checkpoints"],
        ranks=[1, 2], n_boot=10, seed=13,  # layer_mode defaults to "fixed"
    )
    assert result["layer_mode"] == "fixed"
    for estimator_result in result["estimators"].values():
        for fold in estimator_result["fold_results"]:
            per_ck = fold["effect"]["per_checkpoint"]["e007_A_fresh"]
            assert per_ck["alignment"] is not None
            assert np.isfinite(per_ck["alignment"])


# ---------------------------------------------------------------------------
# checkpoint-band mode: alignment recovers the TRUE task/factor geometry even
# when --layer points at a differently-rotated ("wrong") representation
# ---------------------------------------------------------------------------


def _random_orthogonal(dim, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return Q


@pytest.fixture
def band_pooling_fixture():
    rng = np.random.default_rng(21)
    d, n, k_factor, n_groups, n_cache_layers = 12, 300, 2, 40, 3

    w_true = rng.normal(size=d)
    w_true /= np.linalg.norm(w_true)
    M = rng.normal(size=(d, k_factor))
    Uo, _, _ = np.linalg.svd(M, full_matrices=False)
    U_true = Uo[:, :k_factor]

    # Force a KNOWN alignment between w_true and U_true's span, so the test
    # has a ground truth to check the recovered value against.
    target_alignment = 0.5
    u_part = U_true @ (U_true.T @ w_true)
    u_part /= np.linalg.norm(u_part)
    perp = w_true - U_true @ (U_true.T @ w_true)
    perp /= np.linalg.norm(perp)
    w_true = np.sqrt(target_alignment) * u_part + np.sqrt(1 - target_alignment) * perp
    w_true /= np.linalg.norm(w_true)
    assert abs(float(np.sum((U_true.T @ w_true) ** 2)) - target_alignment) < 1e-9

    groups_raw = rng.integers(0, n_groups, size=n)
    y = rng.integers(0, 2, size=n)
    factor_levels = rng.integers(0, 4, size=n)
    raw_centers = rng.normal(size=(4, k_factor))
    raw_centers -= raw_centers.mean(axis=0, keepdims=True)
    Uc, Sc, Vtc = np.linalg.svd(raw_centers, full_matrices=False)
    factor_centers = Uc @ np.diag(np.full_like(Sc, 3.0)) @ Vtc

    L = (np.outer((y * 2 - 1).astype(float), w_true) * 3.0
         + factor_centers[factor_levels] @ U_true.T
         + rng.normal(scale=0.3, size=(n, d)))

    # Layer 1 is the checkpoint's dominant pooling layer: a clean (identity)
    # view of the underlying task/factor geometry. Layers 0 and 2 are
    # independently rotated -- "a different learned representation", standing
    # in for a --layer choice whose coordinate frame does not match where w
    # was learned (exactly the real e007 scenario: layer 9 vs. the pooled
    # band around layer 10).
    R0, R2 = _random_orthogonal(d, seed=101), _random_orthogonal(d, seed=102)
    emb_full = np.zeros((n, n_cache_layers, d), dtype=np.float32)
    emb_full[:, 0, :] = L @ R0 + rng.normal(scale=0.3, size=(n, d))
    emb_full[:, 1, :] = L
    emb_full[:, 2, :] = L @ R2 + rng.normal(scale=0.3, size=(n, d))

    factor = np.array([f"gen{i}" for i in factor_levels], dtype=object)
    groups = np.array([f"grp{i}" for i in groups_raw], dtype=object)

    logits = torch.tensor([-10.0, 0.0, -10.0])  # softmax concentrates on layer 1
    band_weights = torch.softmax(logits, dim=0).numpy()
    checkpoints = {
        "e007_A_fresh": dict(
            w=w_true, b=0.0, ckpt_layer_center=10, ckpt_layer_band=[8, 11],
            layer_pooling="learned_softmax", band_weights=band_weights, w_layer_mismatch=False,
        ),
    }
    return dict(emb_full=emb_full, factor=factor, y=y, groups=groups, checkpoints=checkpoints,
                target_alignment=target_alignment)


def test_run_battery_checkpoint_band_alignment_recovers_true_geometry(band_pooling_fixture):
    d = band_pooling_fixture
    Z_full = d["emb_full"]
    Z_layer0 = Z_full[:, 0, :].astype(np.float32)  # simulate --layer=0: a rotated, mismatched representation
    spec = dict(name="band_test", corpus="diffssd", factor="generator_id", grouping="source_id")

    result = run_battery(
        spec, Z_layer0, d["factor"], d["y"], d["groups"], d["checkpoints"],
        ranks=[2], n_boot=10, seed=13, layer_mode="checkpoint-band", Z_full=Z_full,
    )

    aligns = []
    for estimator_result in result["estimators"].values():
        for fold in estimator_result["fold_results"]:
            per_ck = fold["effect"]["per_checkpoint"]["e007_A_fresh"]
            assert per_ck["alignment"] is not None
            assert np.isfinite(per_ck["alignment"])
            assert per_ck["layer_pooling"] == "learned_softmax"
            aligns.append(per_ck["alignment"])

    assert aligns
    mean_align = float(np.mean(aligns))
    # Recovers close to the true 0.5 despite --layer being a rotated/
    # mismatched representation, because alignment is refit in the
    # checkpoint's own (unrotated, layer-1-dominant) pooled space.
    assert abs(mean_align - d["target_alignment"]) < 0.3


def test_run_battery_fixed_mode_leaves_alignment_none_semantics_unaffected(band_pooling_fixture):
    """Sanity check that checkpoint-band-only fixture data still behaves
    correctly under plain fixed mode (alignment computed inline, no crash,
    regardless of which layer is picked)."""
    d = band_pooling_fixture
    Z_layer0 = d["emb_full"][:, 0, :].astype(np.float32)
    spec = dict(name="band_test_fixed", corpus="diffssd", factor="generator_id", grouping="source_id")

    result = run_battery(
        spec, Z_layer0, d["factor"], d["y"], d["groups"], d["checkpoints"],
        ranks=[2], n_boot=10, seed=13,  # layer_mode defaults to "fixed", Z_full omitted
    )
    for estimator_result in result["estimators"].values():
        for fold in estimator_result["fold_results"]:
            per_ck = fold["effect"]["per_checkpoint"]["e007_A_fresh"]
            assert per_ck["alignment"] is not None


# ---------------------------------------------------------------------------
# --w-metrics: dimension-mismatch detection in load_task_direction
# ---------------------------------------------------------------------------


def test_describe_pooling_mismatch_reads_real_proj_shapes_when_present():
    state = {"proj.0.weight": torch.randn(256, 2048), "proj.4.weight": torch.randn(256, 256)}
    reason = _describe_pooling_mismatch(state, "binary.fc.weight", 256)
    assert "(256, 2048)" in reason
    assert "(256, 256)" in reason
    assert "binary.fc.weight" in reason


def test_describe_pooling_mismatch_falls_back_when_proj_keys_absent():
    reason = _describe_pooling_mismatch({}, "binary.fc.weight", 256)
    assert "2-layer proj MLP" in reason
    assert "binary.fc.weight" in reason


def test_load_task_direction_auto_mode_flags_dim_mismatch_without_raising(tmp_path):
    ckpt = tmp_path / "runs_e007_A_fresh_best.pt"
    _write_checkpoint(ckpt, layer_center=9)  # w is 32-d, see _write_checkpoint
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = load_task_direction(ckpt, requested_layer=9, w_metrics_mode="auto", embedding_dim=1024)
    assert out["w_dim"] == 32
    assert out["w_dim_mismatch"] is True
    assert "32" in out["w_dim_mismatch_reason"] and "no linear pullback" in out["w_dim_mismatch_reason"]
    assert any("no linear pullback" in str(w.message) for w in caught)


def test_load_task_direction_auto_mode_no_mismatch_when_dims_match(tmp_path):
    ckpt = tmp_path / "runs_e007_A_fresh_best.pt"
    _write_checkpoint(ckpt, layer_center=9)
    out = load_task_direction(ckpt, requested_layer=9, w_metrics_mode="auto", embedding_dim=32)
    assert out["w_dim_mismatch"] is False
    assert out["w_dim_mismatch_reason"] is None


def test_load_task_direction_on_mode_raises_on_dim_mismatch(tmp_path):
    ckpt = tmp_path / "runs_e007_A_fresh_best.pt"
    _write_checkpoint(ckpt, layer_center=9)
    with pytest.raises(RuntimeError, match="w-metrics on"):
        load_task_direction(ckpt, requested_layer=9, w_metrics_mode="on", embedding_dim=1024)


def test_load_task_direction_off_mode_skips_w_entirely(tmp_path):
    ckpt = tmp_path / "bad.pt"
    torch.save(dict(model={"unrelated.weight": torch.randn(3, 3)}, cfg={}), ckpt)  # no classifier key at all
    out = load_task_direction(ckpt, requested_layer=9, w_metrics_mode="off")
    assert out["w"] is None and out["b"] is None
    assert out["w_dim"] is None and out["w_dim_mismatch"] is None
    assert out["w_layer_mismatch"] is None


# ---------------------------------------------------------------------------
# resolve_w_metrics
# ---------------------------------------------------------------------------


def test_resolve_w_metrics_off_mode_disabled_regardless_of_checkpoints():
    result = resolve_w_metrics({"e007_A_fresh": dict(w_dim_mismatch=False, w_dim=256)}, "off", embedding_dim=1024)
    assert result["enabled"] is False
    assert result["w_dim"] is None


def test_resolve_w_metrics_no_checkpoints_disabled():
    result = resolve_w_metrics({}, "auto", embedding_dim=1024)
    assert result["enabled"] is False


def test_resolve_w_metrics_any_mismatch_disables_whole_run():
    checkpoints = {
        "e007_A_fresh": dict(w_dim_mismatch=False, w_dim=1024, w_dim_mismatch_reason=None),
        "e007_B_fresh": dict(w_dim_mismatch=True, w_dim=256, w_dim_mismatch_reason="B mismatched"),
    }
    result = resolve_w_metrics(checkpoints, "auto", embedding_dim=1024)
    assert result["enabled"] is False
    assert result["reason"] == "B mismatched"
    assert result["w_dim"] == 256


def test_resolve_w_metrics_all_matched_enabled():
    checkpoints = {"e007_A_fresh": dict(w_dim_mismatch=False, w_dim=1024, w_dim_mismatch_reason=None)}
    result = resolve_w_metrics(checkpoints, "auto", embedding_dim=1024)
    assert result["enabled"] is True
    assert result["w_dim"] == 1024


# ---------------------------------------------------------------------------
# _removal_control_without_task_direction: matches removal_control_report's
# w-independent fields exactly when the effect_fn genuinely ignores w
# ---------------------------------------------------------------------------


def test_removal_control_without_task_direction_matches_full_report_when_effect_fn_ignores_w():
    from audioshield.reliance.metrics import removal_control_report

    rng = np.random.default_rng(7)
    d, k = 10, 2
    Z = rng.standard_normal((30, d))
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    U = Q[:, :k]
    w = rng.standard_normal(d)

    def effect_with_w(Z_, w_, U_):
        return float(np.mean(Z_ @ U_))

    def effect_without_w(Z_, U_):
        return float(np.mean(Z_ @ U_))

    full = removal_control_report(Z, w, U, effect_fn=effect_with_w, n_random=15, seed=13)
    partial = _removal_control_without_task_direction(Z, U, effect_fn=effect_without_w, n_random=15, seed=13)

    assert partial["true_effect"] == pytest.approx(full["true_effect"])
    assert partial["random_effects"] == pytest.approx(full["random_effects"])
    assert partial["random_mean"] == pytest.approx(full["random_mean"])
    assert partial["random_std"] == pytest.approx(full["random_std"])
    assert partial["exceeds_random"] == full["exceeds_random"]
    assert "task_direction_effect" not in partial  # caller's responsibility to add


# ---------------------------------------------------------------------------
# run_battery / summarize_prereg_candidate with w-metrics disabled
# ---------------------------------------------------------------------------


def test_run_battery_w_metrics_disabled_yields_not_estimable_and_completes(synthetic_battery_data):
    d = synthetic_battery_data
    spec = dict(name="w_disabled_test", corpus="diffssd", factor="generator_id", grouping="source_id")
    reason = "test reason: dims mismatch"

    result = run_battery(
        spec, d["Z"], d["factor"], d["y"], d["groups"], d["checkpoints"],
        ranks=[1, 2], n_boot=10, seed=13, w_metrics_enabled=False, w_metrics_reason=reason,
    )

    assert result["headline_bootstrap"]["metric"] == "factor_separation_score"
    assert result["rank_sensitivity"]["metric"] == "factor_separation_score"
    for estimator_result in result["estimators"].values():
        for fold in estimator_result["fold_results"]:
            effect = fold["effect"]
            per_ck = effect["per_checkpoint"]["e007_A_fresh"]
            for key in ("alignment", "r_var", "r_var_class_conditional",
                        "prediction_change", "prediction_change_control"):
                assert per_ck[key] == _not_estimable(reason), f"{key} not marked not_estimable"
            assert per_ck["w_layer_mismatch"] is None
            # factor-only metrics still run unchanged
            assert "decodability_drop" in effect["leace"]
            assert "decodability_drop" in effect["inlp"]
            prc = effect["projection_removal_control"]
            assert np.isfinite(prc["true_effect"])
            assert all(np.isfinite(v) for v in prc["random_effects"])
            assert prc["task_direction_effect"] == _not_estimable(reason)


def test_run_battery_w_metrics_enabled_default_uses_r_var_headline(synthetic_battery_data):
    d = synthetic_battery_data
    spec = dict(name="w_enabled_test", corpus="diffssd", factor="generator_id", grouping="source_id")
    result = run_battery(
        spec, d["Z"], d["factor"], d["y"], d["groups"], d["checkpoints"],
        ranks=[1, 2], n_boot=10, seed=13,  # w_metrics_enabled defaults to True
    )
    assert result["headline_bootstrap"]["metric"] == "r_var"
    assert result["rank_sensitivity"]["metric"] == "r_var"


def test_summarize_prereg_candidate_handles_disabled_w_metrics(synthetic_battery_data):
    d = synthetic_battery_data
    spec = dict(name="w_disabled_summary_test", corpus="diffssd", factor="generator_id", grouping="source_id")
    result = run_battery(
        spec, d["Z"], d["factor"], d["y"], d["groups"], d["checkpoints"],
        ranks=[1, 2], n_boot=10, seed=13, w_metrics_enabled=False, w_metrics_reason="dims mismatch",
    )
    summary = summarize_prereg_candidate(result)
    assert summary["headline_metric"] == "factor_separation_score"
    assert isinstance(summary["estimators_agree_sign"], bool)
    assert isinstance(summary["cis_overlap"], bool)


# ---------------------------------------------------------------------------
# main() end-to-end: a genuine dimension mismatch must not crash the run
# ---------------------------------------------------------------------------


def _write_synthetic_main_inputs(tmp_path, rng, n=120, n_layers=25, cache_dim=64, w_dim=16):
    manifest_dir = tmp_path / "manifests"
    cache_root = tmp_path / "cache"
    ckpt_dir = tmp_path / "ckpts"
    manifest_dir.mkdir()
    (cache_root / "03_DiffSSD").mkdir(parents=True)
    ckpt_dir.mkdir()

    generators = rng.choice(["gradtts", "xttsv2", "playht", "yourtts"], size=n)
    rows = []
    for i in range(n):
        rows.append(dict(
            utt_id=f"diffssd/generated_speech/{generators[i]}/s{i}.wav",
            path=f"datasets/03_DiffSSD/generated_speech/{generators[i]}/s{i}.wav",
            target=1, corpus="diffssd", split="train", attack=generators[i], bona_fide_source="na",
            source_id=f"src{i % 20}", speaker_id="NA", generator_id=generators[i],
            channel_id="NA", language="NA", platform_id="NA",
        ))
    pd.DataFrame(rows).to_csv(manifest_dir / "diffssd.csv", index=False)

    emb = rng.standard_normal((n, n_layers, cache_dim)).astype(np.float16)
    paths = np.array([f"generated_speech/{generators[i]}/s{i}.wav" for i in range(n)])
    np.savez(cache_root / "03_DiffSSD" / "shard_0000.npz", emb=emb, paths=paths)

    for run in ("e007_A_fresh", "e007_B_fresh", "e007_C_xlsr_fresh"):
        sd = dict(
            model={"binary.fc.weight": torch.randn(1, w_dim), "binary.fc.bias": torch.randn(1)},
            cfg={"model": {"layer_weight_init_center": 10, "layer_weight_init_band": [8, 11]}},
        )
        torch.save(sd, ckpt_dir / f"runs_{run}_best.pt")

    return manifest_dir, cache_root, ckpt_dir


def test_main_auto_mode_completes_with_not_estimable_on_dim_mismatch(tmp_path):
    """Regression guard for the crash this task fixes: a genuine w/embedding
    dimension mismatch must complete the run (exit 0), not die inside the
    grouped bootstrap's w-dependent headline metric."""
    rng = np.random.default_rng(21)
    manifest_dir, cache_root, ckpt_dir = _write_synthetic_main_inputs(tmp_path, rng, cache_dim=64, w_dim=16)
    out_path = tmp_path / "out.json"

    argv = [
        "--cache-root", str(cache_root), "--manifest-dir", str(manifest_dir), "--layer", "9",
        "--out", str(out_path), "--corpus", "diffssd", "--factor", "generator_id",
        "--ranks", "1", "2", "--n-boot", "10", "--seed", "13", "--ckpt-dir", str(ckpt_dir),
        "--w-metrics", "auto",
    ]
    main(argv)  # must not raise

    output = json.loads(out_path.read_text())
    assert output["w_metrics"]["enabled"] is False
    assert output["w_metrics"]["w_dim"] == 16
    assert output["w_metrics"]["embedding_dim"] == 64
    battery = output["batteries"][0]
    fold0 = battery["estimators"]["lda"]["fold_results"][0]
    per_ck = fold0["effect"]["per_checkpoint"]["e007_A_fresh"]
    assert per_ck["alignment"]["status"] == "not_estimable"
    assert battery["headline_bootstrap"]["metric"] == "factor_separation_score"


def test_main_on_mode_raises_on_dim_mismatch(tmp_path):
    rng = np.random.default_rng(22)
    manifest_dir, cache_root, ckpt_dir = _write_synthetic_main_inputs(tmp_path, rng, cache_dim=64, w_dim=16)
    out_path = tmp_path / "out.json"

    argv = [
        "--cache-root", str(cache_root), "--manifest-dir", str(manifest_dir), "--layer", "9",
        "--out", str(out_path), "--corpus", "diffssd", "--factor", "generator_id",
        "--ranks", "1", "2", "--n-boot", "10", "--seed", "13", "--ckpt-dir", str(ckpt_dir),
        "--w-metrics", "on",
    ]
    with pytest.raises(RuntimeError, match="w-metrics on"):
        main(argv)


def test_main_matched_dims_regression_guard(tmp_path):
    """A matched-dims synthetic case still computes w-metrics normally."""
    rng = np.random.default_rng(23)
    manifest_dir, cache_root, ckpt_dir = _write_synthetic_main_inputs(tmp_path, rng, cache_dim=32, w_dim=32)
    out_path = tmp_path / "out.json"

    argv = [
        "--cache-root", str(cache_root), "--manifest-dir", str(manifest_dir), "--layer", "9",
        "--out", str(out_path), "--corpus", "diffssd", "--factor", "generator_id",
        "--ranks", "1", "2", "--n-boot", "10", "--seed", "13", "--ckpt-dir", str(ckpt_dir),
        "--w-metrics", "auto",
    ]
    main(argv)

    output = json.loads(out_path.read_text())
    assert output["w_metrics"]["enabled"] is True
    battery = output["batteries"][0]
    fold0 = battery["estimators"]["lda"]["fold_results"][0]
    per_ck = fold0["effect"]["per_checkpoint"]["e007_A_fresh"]
    assert per_ck["alignment"] is not None and np.isfinite(per_ck["alignment"])
    assert battery["headline_bootstrap"]["metric"] == "r_var"


# ---------------------------------------------------------------------------
# per-(battery, estimator) timeout guard: _run_with_timeout / _guarded_call
# ---------------------------------------------------------------------------


def test_run_with_timeout_returns_value_on_success():
    completed, value, exc = _run_with_timeout(lambda: 42, timeout=5)
    assert (completed, value, exc) == (True, 42, None)


def test_run_with_timeout_captures_exception_without_raising():
    def _boom():
        raise ValueError("boom")

    completed, value, exc = _run_with_timeout(_boom, timeout=5)
    assert completed is True
    assert value is None
    assert isinstance(exc, ValueError)
    assert "boom" in str(exc)


def test_run_with_timeout_reports_not_completed_on_timeout():
    def _slow():
        time.sleep(1.0)
        return "done"

    start = time.time()
    completed, value, exc = _run_with_timeout(_slow, timeout=0.05)
    elapsed = time.time() - start

    assert completed is False
    assert value is None
    assert exc is None
    assert elapsed < 0.5, f"took {elapsed}s -- looks like it waited for the slow call instead of timing out"


def test_guarded_call_success_merges_status_ok():
    out = _guarded_call(lambda: {"x": 1}, timeout=5, fallback=lambda r: {"status": "failed", "error": r})
    assert out == {"x": 1, "status": "ok"}


def test_guarded_call_does_not_override_an_explicit_status():
    out = _guarded_call(lambda: {"x": 1, "status": "ok-already"}, timeout=5,
                         fallback=lambda r: {"status": "failed", "error": r})
    assert out["status"] == "ok-already"


def test_guarded_call_timeout_uses_fallback():
    def _slow():
        time.sleep(1.0)

    out = _guarded_call(_slow, timeout=0.05, fallback=lambda r: {"status": "failed", "error": r})
    assert out["status"] == "failed"
    assert "timed out after 0.05s" in out["error"]


def test_guarded_call_exception_uses_fallback():
    def _boom():
        raise RuntimeError("kaboom")

    out = _guarded_call(_boom, timeout=5, fallback=lambda r: {"status": "failed", "error": r})
    assert out["status"] == "failed"
    assert "kaboom" in out["error"]


# ---------------------------------------------------------------------------
# run_battery: the timeout guard integrated -- a hung cell degrades, the run
# completes
# ---------------------------------------------------------------------------


def test_run_battery_completes_normally_reports_status_ok(synthetic_battery_data):
    """With a generous timeout (the default), every guarded piece succeeds
    and reports status='ok' -- the additive schema change doesn't disturb
    the happy path."""
    d = synthetic_battery_data
    spec = dict(name="status_ok_test", corpus="diffssd", factor="generator_id", grouping="source_id")
    result = run_battery(
        spec, d["Z"], d["factor"], d["y"], d["groups"], d["checkpoints"],
        ranks=[1, 2], n_boot=10, seed=13,
    )
    for name, estimator_result in result["estimators"].items():
        assert estimator_result["status"] == "ok", f"{name}: {estimator_result}"
    assert result["headline_bootstrap"]["status"] == "ok"
    assert result["rank_sensitivity"]["status"] == "ok"


def test_run_battery_timeout_records_failed_and_continues(synthetic_battery_data, monkeypatch):
    """Simulates a hung (battery, estimator) cell via an artificially tiny
    timeout plus slowed-down subspace fits -- the practical, deterministic
    stand-in for reproducing a genuine ill-conditioned lbfgs/eigh stall
    (the overnight run that motivated this fix never raised an exception,
    so an exception handler alone couldn't have caught it -- only a
    wall-clock timeout can). Verifies run_battery still completes promptly
    (never hangs) and every guarded piece -- both estimators' crossfit, the
    headline bootstrap, the rank-sensitivity sweep -- is marked
    status='failed' rather than propagating or blocking the rest of the run."""
    d = synthetic_battery_data
    spec = dict(name="timeout_test", corpus="diffssd", factor="generator_id", grouping="source_id")

    def _slow(*a, **kw):
        time.sleep(0.5)
        return np.zeros((d["Z"].shape[1], 1))

    monkeypatch.setattr(rrb, "lda_subspace", _slow)
    monkeypatch.setattr(rrb, "crossfitted_probe_subspace", _slow)

    start = time.time()
    result = run_battery(
        spec, d["Z"], d["factor"], d["y"], d["groups"], d["checkpoints"],
        ranks=[1, 2], n_boot=10, seed=13, battery_timeout_seconds=0.05,
    )
    elapsed = time.time() - start

    assert elapsed < 3.0, f"run_battery took {elapsed}s -- looks like it waited out the slow calls"
    for name, estimator_result in result["estimators"].items():
        assert estimator_result["status"] == "failed", f"{name} estimator did not report failed"
        assert "timed out" in estimator_result["error"]
        assert estimator_result["fold_results"] == []
    assert result["headline_bootstrap"]["status"] == "failed"
    assert "timed out" in result["headline_bootstrap"]["error"]
    assert result["rank_sensitivity"]["status"] == "failed"
    assert "timed out" in result["rank_sensitivity"]["error"]


def test_run_battery_timeout_on_one_estimator_does_not_block_the_other(synthetic_battery_data, monkeypatch):
    """Only the probe estimator is slowed down -- lda must still complete
    and report status='ok', proving one hung cell degrades independently
    rather than taking the whole battery down with it."""
    d = synthetic_battery_data
    spec = dict(name="partial_timeout_test", corpus="diffssd", factor="generator_id", grouping="source_id")

    def _slow(*a, **kw):
        time.sleep(20.0)
        return np.zeros((d["Z"].shape[1], 1))

    monkeypatch.setattr(rrb, "crossfitted_probe_subspace", _slow)

    result = run_battery(
        spec, d["Z"], d["factor"], d["y"], d["groups"], d["checkpoints"],
        ranks=[1, 2], n_boot=10, seed=13, battery_timeout_seconds=5,
    )

    assert result["estimators"]["probe"]["status"] == "failed"
    assert result["estimators"]["lda"]["status"] == "ok"
    assert result["estimators"]["lda"]["fold_results"]
