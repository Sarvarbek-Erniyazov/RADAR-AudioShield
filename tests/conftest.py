import numpy as np
import pandas as pd
import pytest

@pytest.fixture(autouse=True)
def _clear_subspace_cache():
    """lda_subspace/crossfitted_probe_subspace cache their rank-independent
    computation (eigendecomposition / cross-fitted coefficient rows) keyed
    by a content hash of their inputs (audioshield.reliance.subspaces).
    Deterministically-seeded fixtures (e.g. planted_factor_data) can
    legitimately produce byte-identical arrays across unrelated test
    functions -- without resetting the cache between tests, a later test
    could silently hit an earlier, unrelated test's cache entry (e.g. a
    test asserting a scaler was called a specific number of times would
    see zero calls on a cache hit). Reset before every test; only affects
    performance, never correctness, in production (each battery task runs
    in its own freshly-spawned process there, so this is always empty at
    the start regardless)."""
    from audioshield.reliance.subspaces import clear_subspace_cache
    clear_subspace_cache()
    yield


@pytest.fixture
def synthetic_manifest(tmp_path):
    """Tiny manifest with the v3 factor-metadata schema (audit §4.6/§4.8)."""
    rng = np.random.default_rng(7)
    rows = []
    for i in range(120):
        corpus = ["cA", "cB", "cC"][i % 3]
        label = "bonafide" if (i % 3 == 2 or i % 2 == 0) else "spoof"  # deliberately confounded
        rows.append(dict(
            path=f"{corpus}/f{i:04d}.wav", label=label, corpus=corpus,
            source_id=f"src{i // 4}", speaker_id=f"spk{i // 6}",
            generator_id="NA" if label == "bonafide" else f"gen{i % 5}",
            channel_id=f"ch{i % 2}", platform_id="NA", attack="NA",
            duration=float(rng.uniform(1, 10)),
        ))
    df = pd.DataFrame(rows)
    p = tmp_path / "manifest.csv"
    df.to_csv(p, index=False)
    return p, df

@pytest.fixture
def deterministic_embeddings():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(120, 32)).astype("float32")
    y = (rng.uniform(size=120) > 0.5).astype("int64")
    groups = np.array([f"src{i // 4}" for i in range(120)])
    return X, y, groups

@pytest.fixture
def planted_factor_data():
    """Synthetic Gaussian embeddings with a planted task direction `w_true` and an
    orthogonal planted factor subspace `U_true` (rank 3), plus grouped structure
    (simulating source_id/speaker_id sessions) with within-group correlated
    offsets -- so grouped-fold discipline actually matters for these rows, not
    just in principle. Used across tests/test_reliance_*.py.
    """
    rng = np.random.default_rng(13)
    d, n, k_factor, n_groups = 20, 3000, 3, 600

    w_true = rng.normal(size=d)
    w_true /= np.linalg.norm(w_true)

    M = rng.normal(size=(d, k_factor))
    M = M - np.outer(w_true, w_true @ M)  # remove w_true component -> orthogonal by construction
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    U_true = U[:, :k_factor]

    groups = rng.integers(0, n_groups, size=n)
    group_offset = rng.normal(scale=0.5, size=(n_groups, d))[groups]

    y = rng.integers(0, 2, size=n)
    factor = rng.integers(0, 4, size=n)
    # Equalize separation strength across all k_factor directions -- a raw
    # rng.normal draw can (and, at seed=13, does) produce a poorly-conditioned
    # centers matrix with one near-degenerate direction (singular value ~10x
    # weaker than the other two), making that direction's recovery marginal for
    # any estimator regardless of correctness. Rescaling every singular value to
    # the same constant keeps the plant well-conditioned without hand-picking a
    # seed.
    raw_centers = rng.normal(size=(4, k_factor))
    raw_centers -= raw_centers.mean(axis=0, keepdims=True)
    Uc, Sc, Vtc = np.linalg.svd(raw_centers, full_matrices=False)
    factor_centers = Uc @ np.diag(np.full_like(Sc, 3.0)) @ Vtc

    Z = (
        np.outer((y * 2 - 1).astype(float), w_true) * 3.0
        + factor_centers[factor] @ U_true.T
        + group_offset
        + rng.normal(size=(n, d))
    )
    return dict(
        Z=Z, y=y, factor=factor, groups=groups.astype(str),
        w_true=w_true, U_true=U_true, d=d, k_factor=k_factor,
    )
