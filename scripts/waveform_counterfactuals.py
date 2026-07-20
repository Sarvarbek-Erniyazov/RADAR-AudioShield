"""Task 4 (Step 4 gate prep, conditional on Tasks 1-3): waveform-level
paired counterfactuals for the Step 4 intervention battery (Roadmap v3:
"codec chains, RIR/replay simulation, resampling; dose-response; using
locally present raw audio + the manifest re-fetch subset").

Three transform families, each with a dose parameter (more dose = more
degradation):

  - codec_roundtrip: encode -> decode through a lossy codec (mp3/opus/
    amr_nb via ffmpeg, confirmed present on this machine's PATH with
    libmp3lame/libopus/libopencore_amrnb built in) at a given bitrate.
    Lower bitrate = higher dose.
  - resample_roundtrip: downsample to a target rate, then back up to the
    working rate (scipy.signal.resample_poly, no ffmpeg dependency).
    Lower target rate = higher dose. Uses scipy rather than torchaudio:
    torchaudio is used elsewhere in this project only as a lazily-imported,
    optional enhancement over soundfile (src/audioshield/data/audio_io.py)
    -- not a hard module-level dependency -- and isn't actually installed
    in this environment's Python (torch is, torchaudio isn't); scipy is
    already a guaranteed project dependency, so it's the more portable
    choice here too.
  - synthetic_rir_convolve: convolve with a synthetic exponential-decay
    noise impulse response (scipy.signal.fftconvolve), parameterized by
    RT60. No real RIR corpus is required to exercise dose-response; a
    real recorded/simulated RIR set (the roadmap's "manifest re-fetch
    subset") can replace this later without changing the paired-manifest
    machinery below. Higher RT60 = higher dose.

`build_paired_manifest` applies a list of named conditions (parsed by
`parse_condition`) to a handful of manifest rows, writes each transformed
file next to a run-specific output directory, and returns paired
(original, transformed) manifest rows -- never crashing on a single file's
failure (matches this project's established per-unit try/except
convention; a failed row is recorded with status="failed" and skipped,
not allowed to abort the whole manifest build).

Usage (a handful of locally-present files; full-scale runs happen later
wherever raw audio lives -- COLLABORATOR PC once the manifest re-fetch
subset exists):
    python scripts/waveform_counterfactuals.py \\
        --manifest manifests/v2/diffssd.csv --data-root .. \\
        --n-files 20 --conditions codec_mp3_16k codec_opus_12k resample_8000 rir_rt60_0.5 \\
        --out-audio-dir analysis/step4/counterfactuals/diffssd \\
        --out-manifest analysis/step4/counterfactuals/diffssd_manifest.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_SAMPLE_RATE = 16000


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _log(msg: str) -> None:
    print(f"[{_timestamp()}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Transform family 1: codec round-trip (ffmpeg subprocess)
# ---------------------------------------------------------------------------

_CODEC_ENCODERS = dict(mp3="libmp3lame", opus="libopus", amr_nb="libopencore_amrnb")
_CODEC_CONTAINER_EXT = dict(mp3="mp3", opus="opus", amr_nb="amr")
_CODEC_FORCED_SAMPLE_RATE = dict(amr_nb=8000)  # AMR-NB is defined only at 8kHz mono


def codec_roundtrip(in_wav: Path, out_wav: Path, codec: str, bitrate_kbps: int,
                     sample_rate: int = DEFAULT_SAMPLE_RATE, ffmpeg_bin: str = "ffmpeg") -> None:
    """Encode `in_wav` through `codec` at `bitrate_kbps`, then decode back
    to a `sample_rate`-Hz wav at `out_wav`. Two ffmpeg subprocess calls
    (encode, decode) rather than one piped command -- keeps the
    intermediate compressed file inspectable for debugging and avoids
    piping binary audio through subprocess stdout, which is fragile across
    platforms."""
    if codec not in _CODEC_ENCODERS:
        raise ValueError(f"unknown codec {codec!r}, expected one of {sorted(_CODEC_ENCODERS)}")
    encode_sr = _CODEC_FORCED_SAMPLE_RATE.get(codec, sample_rate)
    compressed = out_wav.with_suffix("." + _CODEC_CONTAINER_EXT[codec])
    encode_cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error", "-i", str(in_wav),
        "-ar", str(encode_sr), "-ac", "1", "-c:a", _CODEC_ENCODERS[codec], "-b:a", f"{bitrate_kbps}k",
        str(compressed),
    ]
    subprocess.run(encode_cmd, check=True, capture_output=True)
    decode_cmd = [
        ffmpeg_bin, "-y", "-loglevel", "error", "-i", str(compressed),
        "-ar", str(sample_rate), "-ac", "1", str(out_wav),
    ]
    subprocess.run(decode_cmd, check=True, capture_output=True)
    compressed.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Transform family 2: resample round-trip (torchaudio, no external process)
# ---------------------------------------------------------------------------


def resample_roundtrip(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Downsample orig_sr -> target_sr, then back up target_sr -> orig_sr.
    The lossy step is the downsample (discards frequency content above
    target_sr/2 via anti-aliasing); the upsample restores the original
    sample count/rate without restoring the discarded content -- a
    resampling-degradation dose parameterized by how low target_sr goes.
    scipy.signal.resample_poly does exact rational-ratio polyphase
    resampling (up/down reduced by their gcd); output length can differ
    from the input by a few samples due to that rounding, so the result is
    trimmed/zero-padded back to the original length."""
    from math import gcd

    from scipy.signal import resample_poly

    waveform = np.asarray(waveform, dtype=np.float32)
    g = gcd(orig_sr, target_sr)
    down = resample_poly(waveform, target_sr // g, orig_sr // g)
    back = resample_poly(down, orig_sr // g, target_sr // g)
    n = len(waveform)
    if len(back) >= n:
        return back[:n].astype(np.float32)
    return np.pad(back, (0, n - len(back))).astype(np.float32)


# ---------------------------------------------------------------------------
# Transform family 3: synthetic RIR convolution (scipy, no external corpus)
# ---------------------------------------------------------------------------


def synthetic_rir(rt60_seconds: float, sample_rate: int = DEFAULT_SAMPLE_RATE, seed: int = 13) -> np.ndarray:
    """A synthetic room impulse response: white noise shaped by an
    exponential decay envelope whose -60dB point lands at rt60_seconds --
    the standard synthetic-RIR construction used when no real
    recorded/simulated RIR corpus is available yet (the roadmap's
    "manifest re-fetch subset" can supply a real IR set later; this
    machinery is agnostic to where the IR array comes from)."""
    rng = np.random.default_rng(seed)
    n = max(1, int(rt60_seconds * sample_rate))
    t = np.arange(n) / sample_rate
    decay_rate = 3.0 * np.log(10.0) / rt60_seconds  # exponential(-decay_rate * t) reaches -60dB at rt60
    envelope = np.exp(-decay_rate * t)
    ir = rng.normal(size=n) * envelope
    ir /= np.sqrt(np.sum(ir**2)) + 1e-12  # unit energy, so convolution doesn't change overall loudness much
    return ir.astype(np.float32)


def synthetic_rir_convolve(waveform: np.ndarray, rt60_seconds: float, sample_rate: int = DEFAULT_SAMPLE_RATE,
                            seed: int = 13) -> np.ndarray:
    """Convolve `waveform` with a synthetic RIR (see synthetic_rir),
    trimmed back to the original length (`full` convolution, then
    truncated) so paired original/transformed files stay the same
    duration -- the intervention is the added reverberant tail energy
    within the clip, not a longer clip."""
    from scipy.signal import fftconvolve
    ir = synthetic_rir(rt60_seconds, sample_rate, seed)
    wet = fftconvolve(waveform, ir, mode="full")[: len(waveform)]
    peak = np.max(np.abs(wet)) + 1e-12
    orig_peak = np.max(np.abs(waveform)) + 1e-12
    return (wet * (orig_peak / peak)).astype(np.float32)  # renormalize peak so codec/quantization dose stays comparable


# ---------------------------------------------------------------------------
# Condition parsing -- a small string grammar so the CLI can name
# dose-response conditions without a config file.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    name: str
    family: str  # "codec" | "resample" | "rir"
    params: dict


def parse_condition(spec: str) -> Condition:
    parts = spec.split("_")
    if spec.startswith("codec_"):
        # codec_<name>_<bitrate>k  e.g. codec_mp3_16k, codec_opus_12k, codec_amr_nb (no bitrate -> default 12k)
        rest = parts[1:]
        if rest and rest[-1].endswith("k") and rest[-1][:-1].isdigit():
            bitrate = int(rest[-1][:-1])
            codec = "_".join(rest[:-1])
        else:
            bitrate = 12
            codec = "_".join(rest)
        return Condition(name=spec, family="codec", params=dict(codec=codec, bitrate_kbps=bitrate))
    if spec.startswith("resample_"):
        target_sr = int(parts[1])
        return Condition(name=spec, family="resample", params=dict(target_sr=target_sr))
    if spec.startswith("rir_rt60_"):
        rt60 = float(spec[len("rir_rt60_"):])
        return Condition(name=spec, family="rir", params=dict(rt60_seconds=rt60))
    raise ValueError(f"unrecognized condition spec {spec!r} -- expected codec_<name>_<bitrate>k, "
                      "resample_<hz>, or rir_rt60_<seconds>")


def apply_condition(in_wav: Path, out_wav: Path, condition: Condition, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    if condition.family == "codec":
        codec_roundtrip(in_wav, out_wav, condition.params["codec"], condition.params["bitrate_kbps"], sample_rate)
        return
    waveform, sr = sf.read(in_wav, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if condition.family == "resample":
        out = resample_roundtrip(waveform, sr, condition.params["target_sr"])
    elif condition.family == "rir":
        out = synthetic_rir_convolve(waveform, condition.params["rt60_seconds"], sr)
    else:
        raise ValueError(f"unknown condition family {condition.family!r}")
    sf.write(out_wav, out, sr)


# ---------------------------------------------------------------------------
# Paired-manifest construction
# ---------------------------------------------------------------------------


def build_paired_manifest(rows: list, data_root: Path, out_audio_dir: Path,
                           conditions: list[Condition], sample_rate: int = DEFAULT_SAMPLE_RATE) -> dict:
    """For every (row, condition) pair, write the transformed audio under
    out_audio_dir/<condition_name>/<row's own relative path> and record a
    paired manifest entry: {original_path, transformed_path, condition,
    dose_params, status}. A single file's transform failure is caught and
    recorded as status="failed" -- never aborts the remaining rows/
    conditions (this project's established per-unit-of-work convention,
    e.g. scripts/run_reliance_battery.py's per-battery try/except,
    scripts/extract_model_embeddings.py's per-shard resume)."""
    entries = []
    for condition in conditions:
        _log(f"waveform_counterfactuals: condition={condition.name} -- {len(rows)} file(s)")
        for row in rows:
            row_path = getattr(row, "path", None) or row["path"]
            in_wav = data_root / row_path
            out_wav = out_audio_dir / condition.name / Path(row_path).name
            try:
                apply_condition(in_wav, out_wav, condition, sample_rate)
                entries.append(dict(original_path=str(row_path), transformed_path=str(out_wav),
                                     condition=condition.name, family=condition.family,
                                     params=condition.params, status="ok"))
            except Exception as exc:
                entries.append(dict(original_path=str(row_path), transformed_path=None,
                                     condition=condition.name, family=condition.family,
                                     params=condition.params, status="failed", reason=str(exc)))
    n_ok = sum(e["status"] == "ok" for e in entries)
    _log(f"waveform_counterfactuals: {n_ok}/{len(entries)} pairs written successfully")
    return dict(schema_version=1, generated_at=_timestamp(), n_rows=len(rows),
                conditions=[c.name for c in conditions], entries=entries)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--data-root", default="..", type=Path)
    ap.add_argument("--n-files", type=int, default=20, help="handful of files to transform (dev-run scale)")
    ap.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    ap.add_argument("--conditions", nargs="+", required=True,
                     help="e.g. codec_mp3_16k codec_opus_12k resample_8000 rir_rt60_0.5")
    ap.add_argument("--out-audio-dir", required=True, type=Path)
    ap.add_argument("--out-manifest", required=True, type=Path)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from audioshield.data.manifest import read_manifest

    _log(f"waveform_counterfactuals: reading {args.manifest}")
    rows = read_manifest(args.manifest)[: args.n_files]
    _log(f"waveform_counterfactuals: {len(rows)} row(s) (capped at --n-files={args.n_files})")
    conditions = [parse_condition(c) for c in args.conditions]
    payload = build_paired_manifest(rows, args.data_root, args.out_audio_dir, conditions, args.sample_rate)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.out_manifest.with_suffix(args.out_manifest.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(args.out_manifest)
    _log(f"waveform_counterfactuals: wrote {args.out_manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
