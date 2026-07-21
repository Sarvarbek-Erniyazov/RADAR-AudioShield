"""Tests for src/audioshield/utils/hashing.py
(step3_modelspace_hardening_addendum.md Finding 2).

The whole point of this module is to be a stdlib-only shared home for
sha256_file, so scripts/run_reliance_modelspace.py (CPU-only reliance
analysis) can reuse the SAME hash implementation
scripts/extract_model_embeddings.py (GPU/model stack) uses without
transitively importing torch/AudioShieldX/UnifiedAudioDataset just to
reuse a five-line hash helper. The import-isolation test below is the
proof of that property -- it MUST run in a fresh subprocess, since torch
is almost always already loaded in the main pytest process by the time
this test file runs (other test modules import it first).
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from audioshield.utils.hashing import sha256_file


def test_sha256_file_matches_hashlib_reference(tmp_path):
    path = tmp_path / "some_bytes.bin"
    path.write_bytes(b"arbitrary checkpoint-shaped content, not a real .pt file")
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_sha256_file_reads_large_files_in_chunks(tmp_path):
    """Confirms the chunked read loop doesn't truncate/miscompute on
    something bigger than one chunk (1 MiB) -- a real checkpoint file is
    many times this size."""
    path = tmp_path / "big.bin"
    # ~2.5 chunks' worth, deterministic content (not all-zero, so a
    # naive implementation that stopped early would produce a different hash).
    import numpy as np
    rng = np.random.default_rng(0)
    path.write_bytes(rng.bytes(int(2.5 * 1024 * 1024)))
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_hashing_module_import_does_not_pull_in_torch_or_model_stack():
    """THE proof of Finding 2: a fresh process that imports ONLY
    audioshield.utils.hashing must never load torch or the AudioShieldX/
    UnifiedAudioDataset model stack -- confirming run_reliance_modelspace.py
    (which imports this module for sha256_file) is no longer contingent on
    that heavier stack importing cleanly in the CPU analysis environment."""
    code = (
        "import sys\n"
        "import audioshield.utils.hashing\n"
        "assert 'torch' not in sys.modules, sorted(sys.modules)\n"
        "assert 'audioshield.models' not in sys.modules, sorted(k for k in sys.modules if k.startswith('audioshield'))\n"
        "assert 'audioshield.models.detector' not in sys.modules\n"
        "assert 'audioshield.data.unified_dataset' not in sys.modules\n"
        "print('IMPORT_ISOLATION_OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "IMPORT_ISOLATION_OK" in result.stdout


def test_extract_model_embeddings_still_reuses_the_same_shared_function():
    """Both scripts must import the SAME implementation (not two
    independently-written copies that could silently drift) -- confirmed
    by identity, not just equal output."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import extract_model_embeddings

    assert extract_model_embeddings.sha256_file is sha256_file
