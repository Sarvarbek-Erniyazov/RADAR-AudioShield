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

@pytest.fixture
def synthetic_broadband_audio():
    """(waveform, sr) -- 2s of broadband (non-periodic) noise at 16kHz. Used by
    tests/test_counterfactuals_*.py for dose-response checks: a pure tone's
    periodic autocorrelation makes cross-correlation-based sample-alignment
    ambiguous (ties every period apart look identical), which is a property of
    correlation-based alignment in general, not a bug -- broadband signal
    avoids that ambiguity, matching real (non-periodic) speech audio."""
    rng = np.random.default_rng(7)
    sr = 16000
    x = (0.3 * rng.standard_normal(sr * 2)).astype(np.float32)
    return x, sr

@pytest.fixture
def synthetic_rir_root(tmp_path):
    """A tiny synthetic RIR asset directory: 12 short impulse-response-like
    .wav files, 16kHz -- enough to satisfy aug_assets.fingerprint_asset_dir's
    min_files=10 gate and to exercise reverb.py's asset resolution/selection
    without touching the real (large) RIRS_NOISES corpus.

    Direct-path peak (amplitude 1.0 at sample 0) with a FAST-decaying,
    AMPLITUDE-SUBORDINATE tail (tau in [80, 250] samples, tail scaled to 0.3x)
    -- a first draft used an undamped full-amplitude random tail (tau up to
    800, no tail scaling), which is closer to colored noise than a real RIR
    and produced a large, spurious non-monotonic dip in dose-response tests
    (chaotic phase interactions with the dry signal at intermediate wet/dry
    mixes). This shape -- direct path dominates, tail dies out quickly -- is
    both more physically realistic and what the dose-response tests need.
    """
    import soundfile as sf
    root = tmp_path / "rirs"
    root.mkdir()
    rng = np.random.default_rng(11)
    sr = 16000
    n = 4000
    for i in range(12):
        t = np.arange(n)
        decay = np.exp(-t / rng.uniform(80, 250))
        rir = (0.3 * rng.standard_normal(n) * decay).astype(np.float32)
        rir[0] = 1.0  # sharp direct-path peak at sample 0, dominates the tail
        sf.write(root / f"rir_{i:02d}.wav", rir, sr)
    return root
