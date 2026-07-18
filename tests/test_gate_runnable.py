from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from audioshield.evaluation.cross_test import build_parser
from scripts.reproduce_eval import build_cmd


def test_reproduction_child_command_parses() -> None:
    child_args = build_cmd("e007_A_fresh", Path("x.pt"), ".")[3:]

    args = build_parser().parse_args(child_args)

    assert args.checkpoint == "x.pt"
    assert args.corpora == ["inthewild", "replaydf", "ai4t"]
    assert args.manifest_dir == "manifests/v2"
    assert args.dev_corpora == ["diffssd", "fakeorreal", "asvspoof5"]
    assert args.data_root == "."
    assert args.out == "repro_e007_A_fresh.json"
    assert args.force is True


def test_cross_test_imports_in_fresh_subprocess() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import audioshield.evaluation.cross_test"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_reproduction_child_command_accepts_dev_corpora_override() -> None:
    child_args = build_cmd("e007_A_fresh", Path("x.pt"), ".", dev_corpora=("fakeorreal",))[3:]

    args = build_parser().parse_args(child_args)

    assert args.dev_corpora == ["fakeorreal"]
    assert args.checkpoint == "x.pt"
    assert args.corpora == ["inthewild", "replaydf", "ai4t"]
    assert args.manifest_dir == "manifests/v2"
    assert args.data_root == "."
    assert args.out == "repro_e007_A_fresh.json"
    assert args.force is True


def test_reproduction_child_without_corpora_fails_parse() -> None:
    child_args = build_cmd("e007_A_fresh", Path("x.pt"), ".")[3:]
    corpora_start = child_args.index("--corpora")
    manifest_start = child_args.index("--manifest-dir")
    stripped = child_args[:corpora_start] + child_args[manifest_start:]

    with pytest.raises(SystemExit):
        build_parser().parse_args(stripped)
