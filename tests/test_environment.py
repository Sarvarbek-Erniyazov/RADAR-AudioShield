"""Commit-1 correctness: frozen env is importable and internally consistent."""
import importlib, json, re
from pathlib import Path

CORE = ["numpy", "scipy", "sklearn", "pandas", "soundfile", "transformers", "yaml", "torch"]

def test_core_imports():
    for mod in CORE:
        importlib.import_module(mod)  # audit §5: scipy was missing — this catches it forever

def test_lockfile_exists_and_matches_env():
    locks = list(Path(".").glob("requirements.lock.*"))
    assert locks, "run scripts/freeze_environment.py <host_tag> before merging (audit §5 pinning)"
    env_jsons = list(Path(".").glob("environment.*.json"))
    assert env_jsons, "environment.<tag>.json missing"
    info = json.loads(env_jsons[0].read_text())
    import transformers
    assert info["transformers"] == transformers.__version__, "lock drifted from live env — re-freeze"

def test_backbone_revisions_pinned():
    p = Path("configs/backbone_revisions.yaml")
    assert p.exists(), "run scripts/pin_backbone_revisions.py once and commit (audit §5)"
    text = p.read_text()
    assert re.search(r"wav2vec2-xls-r-300m:\s*\S{20,}", text) and re.search(r"wavlm-large:\s*\S{20,}", text)
