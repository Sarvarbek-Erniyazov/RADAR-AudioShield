import numpy as np
import pandas as pd
import pytest

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
