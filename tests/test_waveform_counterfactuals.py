"""Tests for scripts/waveform_counterfactuals.py -- Task 4 (Step 4 gate
prep, waveform-level paired counterfactuals).

Uses a handful of tiny synthetic sine-wave wav files written to tmp_path
(not real corpus audio -- keeps this suite hermetic/CI-portable). Codec
tests are skipped if ffmpeg isn't on PATH (confirmed present on this
development machine with libmp3lame/libopus, but this must not be a hard
CI dependency).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from waveform_counterfactuals import (  # noqa: E402
    Condition,
    apply_condition,
    build_paired_manifest,
    build_parser,
    codec_roundtrip,
    parse_condition,
    resample_roundtrip,
    synthetic_rir,
    synthetic_rir_convolve,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None
requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not on PATH")

SR = 16000


def _sine_wav(path: Path, freq: float = 440.0, duration: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    wave = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, wave, sr)
    return wave


# ---------------------------------------------------------------------------
# Condition parsing
# ---------------------------------------------------------------------------


def test_parse_condition_codec_with_bitrate():
    c = parse_condition("codec_mp3_16k")
    assert c == Condition(name="codec_mp3_16k", family="codec", params=dict(codec="mp3", bitrate_kbps=16))


def test_parse_condition_codec_amr_nb_multiword():
    c = parse_condition("codec_amr_nb_8k")
    assert c.params == dict(codec="amr_nb", bitrate_kbps=8)


def test_parse_condition_resample():
    c = parse_condition("resample_8000")
    assert c == Condition(name="resample_8000", family="resample", params=dict(target_sr=8000))


def test_parse_condition_rir():
    c = parse_condition("rir_rt60_0.5")
    assert c == Condition(name="rir_rt60_0.5", family="rir", params=dict(rt60_seconds=0.5))


def test_parse_condition_unrecognized_raises():
    with pytest.raises(ValueError):
        parse_condition("not_a_real_condition")


# ---------------------------------------------------------------------------
# Resample round-trip -- dose-response check: a lower target rate should
# attenuate high-frequency content MORE than a higher target rate.
# ---------------------------------------------------------------------------


def test_resample_roundtrip_preserves_length():
    wav = _sine_wav_array(440.0, 0.5, SR)
    out = resample_roundtrip(wav, SR, 8000)
    assert out.shape[-1] == wav.shape[-1]


def _sine_wav_array(freq, duration, sr):
    t = np.arange(int(duration * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_resample_roundtrip_dose_response_attenuates_high_frequency_more():
    high_freq_tone = _sine_wav_array(6000.0, 0.5, SR)
    mild = resample_roundtrip(high_freq_tone, SR, 12000)   # nyquist 6000 -- borderline
    severe = resample_roundtrip(high_freq_tone, SR, 4000)  # nyquist 2000 -- well below 6000Hz tone

    mild_energy = float(np.mean(mild**2))
    severe_energy = float(np.mean(severe**2))
    original_energy = float(np.mean(high_freq_tone**2))

    assert severe_energy < mild_energy < original_energy


# ---------------------------------------------------------------------------
# Synthetic RIR -- dose-response check: higher RT60 spreads energy over a
# longer effective duration.
# ---------------------------------------------------------------------------


def test_synthetic_rir_length_matches_rt60():
    short_ir = synthetic_rir(0.2, sample_rate=SR, seed=13)
    long_ir = synthetic_rir(0.8, sample_rate=SR, seed=13)
    assert len(long_ir) > len(short_ir)
    assert len(short_ir) == pytest.approx(0.2 * SR, abs=1)
    assert len(long_ir) == pytest.approx(0.8 * SR, abs=1)


def test_synthetic_rir_convolve_preserves_input_length():
    wave = _sine_wav_array(440.0, 0.5, SR)
    wet = synthetic_rir_convolve(wave, rt60_seconds=0.3, sample_rate=SR, seed=13)
    assert len(wet) == len(wave)
    assert np.all(np.isfinite(wet))


def test_synthetic_rir_convolve_dose_response_more_energy_after_original_offset():
    """A higher RT60 should leave more reverberant tail energy in the
    clip's second half (since the impulse response itself decays more
    slowly) than a lower RT60, for the same dry input."""
    wave = _sine_wav_array(440.0, 0.5, SR)
    half = len(wave) // 2
    mild = synthetic_rir_convolve(wave, rt60_seconds=0.05, sample_rate=SR, seed=13)
    severe = synthetic_rir_convolve(wave, rt60_seconds=0.9, sample_rate=SR, seed=13)

    mild_tail_energy = float(np.mean(mild[half:] ** 2))
    severe_tail_energy = float(np.mean(severe[half:] ** 2))
    assert severe_tail_energy > mild_tail_energy


# ---------------------------------------------------------------------------
# apply_condition / build_paired_manifest -- end to end on tiny real files
# ---------------------------------------------------------------------------


def test_apply_condition_resample_writes_valid_wav(tmp_path):
    in_wav = tmp_path / "in.wav"
    _sine_wav(in_wav)
    out_wav = tmp_path / "out.wav"
    apply_condition(in_wav, out_wav, parse_condition("resample_8000"), sample_rate=SR)
    data, sr = sf.read(out_wav)
    assert sr == SR
    assert len(data) == pytest.approx(SR * 0.5, abs=2)


def test_apply_condition_rir_writes_valid_wav(tmp_path):
    in_wav = tmp_path / "in.wav"
    _sine_wav(in_wav)
    out_wav = tmp_path / "out.wav"
    apply_condition(in_wav, out_wav, parse_condition("rir_rt60_0.3"), sample_rate=SR)
    data, sr = sf.read(out_wav)
    assert sr == SR
    assert np.all(np.isfinite(data))


@requires_ffmpeg
def test_codec_roundtrip_mp3_produces_valid_wav(tmp_path):
    in_wav = tmp_path / "in.wav"
    _sine_wav(in_wav)
    out_wav = tmp_path / "out.wav"
    codec_roundtrip(in_wav, out_wav, codec="mp3", bitrate_kbps=16, sample_rate=SR)
    data, sr = sf.read(out_wav)
    assert sr == SR
    assert len(data) > 0
    assert np.any(data != 0)


@requires_ffmpeg
def test_codec_roundtrip_opus_produces_valid_wav(tmp_path):
    in_wav = tmp_path / "in.wav"
    _sine_wav(in_wav)
    out_wav = tmp_path / "out.wav"
    codec_roundtrip(in_wav, out_wav, codec="opus", bitrate_kbps=12, sample_rate=SR)
    data, sr = sf.read(out_wav)
    assert sr == SR
    assert len(data) > 0


@requires_ffmpeg
def test_build_paired_manifest_multiple_conditions(tmp_path):
    data_root = tmp_path / "data_root"
    data_root.mkdir()
    rows = []
    for i in range(3):
        rel = f"clip_{i}.wav"
        _sine_wav(data_root / rel, freq=300 + i * 50)
        rows.append(dict(path=rel))

    conditions = [parse_condition("resample_8000"), parse_condition("rir_rt60_0.3"),
                  parse_condition("codec_mp3_16k")]
    out_audio_dir = tmp_path / "out_audio"
    manifest = build_paired_manifest(rows, data_root, out_audio_dir, conditions, sample_rate=SR)

    assert manifest["n_rows"] == 3
    assert len(manifest["entries"]) == 3 * 3
    assert all(e["status"] == "ok" for e in manifest["entries"])
    for e in manifest["entries"]:
        assert Path(e["transformed_path"]).exists()


def test_build_paired_manifest_never_crashes_on_missing_file(tmp_path):
    data_root = tmp_path / "data_root"
    data_root.mkdir()
    rows = [dict(path="does_not_exist.wav")]
    conditions = [parse_condition("resample_8000")]
    manifest = build_paired_manifest(rows, data_root, tmp_path / "out_audio", conditions, sample_rate=SR)

    assert len(manifest["entries"]) == 1
    assert manifest["entries"][0]["status"] == "failed"
    assert "reason" in manifest["entries"][0]


def test_build_parser_requires_manifest_and_conditions():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--out-audio-dir", "x", "--out-manifest", "y.json"])
