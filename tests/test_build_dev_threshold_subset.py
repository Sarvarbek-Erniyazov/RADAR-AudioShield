from __future__ import annotations

import csv
import hashlib
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from audioshield.data.manifest import FIELDNAMES, read_manifest
from scripts.build_dev_threshold_subset import SubsetBuildError, build_subset


@dataclass(frozen=True)
class Fixture:
    data_root: Path
    manifest_dir: Path
    checksum_dir: Path
    payloads: dict[str, bytes]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_fixture(tmp_path: Path) -> Fixture:
    data_root = tmp_path / "data"
    manifest_dir = tmp_path / "manifests" / "v2"
    checksum_dir = tmp_path / "manifests" / "checksums"
    payloads: dict[str, bytes] = {}

    corpora = {
        "asvspoof5": ("datasets/01_ASVspoof5", "01_ASVspoof5_SHA256.txt"),
        "diffssd": ("datasets/03_DiffSSD", "03_DiffSSD_SHA256.txt"),
    }
    manifest_dir.mkdir(parents=True)
    checksum_dir.mkdir(parents=True)

    for corpus, (prefix, checksum_name) in corpora.items():
        rows: list[dict[str, object]] = []
        checksum_lines: list[str] = []
        for index in range(8):
            relative = f"audio/{corpus}_{index:02d}.flac"
            archive_path = f"{prefix}/{relative}"
            payload = f"fixture audio: {corpus} #{index}\n".encode()
            source = data_root / archive_path
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(payload)
            payloads[archive_path] = payload
            rows.append(
                {
                    "utt_id": f"{corpus}/{index}",
                    "path": archive_path,
                    "target": index % 2,
                    "corpus": corpus,
                    "split": "val",
                    "attack": "fixture-spoof" if index % 2 else "bonafide",
                    "bona_fide_source": "na" if index % 2 else "fixture-real",
                }
            )
            checksum_lines.append(f"{_sha256(payload)}  {relative}\n")

        # A non-val row proves that selection is delegated to the real reader filters.
        rows.append(
            {
                "utt_id": f"{corpus}/train-only",
                "path": f"{prefix}/audio/train-only.flac",
                "target": 1,
                "corpus": corpus,
                "split": "train",
                "attack": "fixture-spoof",
                "bona_fide_source": "na",
            }
        )
        with (manifest_dir / f"{corpus}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        (checksum_dir / checksum_name).write_text(
            "".join(checksum_lines), encoding="utf-8", newline="\n"
        )

    return Fixture(data_root, manifest_dir, checksum_dir, payloads)


def _expected_names(fixture: Fixture) -> list[str]:
    selected: list[str] = []
    for corpus in ("asvspoof5", "diffssd"):
        rows = read_manifest(
            fixture.manifest_dir / f"{corpus}.csv",
            splits=["val"],
            corpora=[corpus],
        )
        random.Random(7).shuffle(rows)
        selected.extend(row.path for row in rows[:1000])
    return selected


def test_subset_archive_is_deterministic_and_preserves_dataset_paths(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    first = build_subset(
        fixture.data_root,
        tmp_path / "first" / "subset.tar.gz",
        manifest_dir=fixture.manifest_dir,
        checksum_dir=fixture.checksum_dir,
    )
    second = build_subset(
        fixture.data_root,
        tmp_path / "second" / "renamed.tar.gz",
        manifest_dir=fixture.manifest_dir,
        checksum_dir=fixture.checksum_dir,
    )

    assert first.archive_sha256 == second.archive_sha256
    assert first.archive_path.read_bytes() == second.archive_path.read_bytes()
    assert first.file_list_path.read_bytes() == second.file_list_path.read_bytes()
    assert first.n_selected == 16
    assert first.per_corpus["asvspoof5"].n_selected == 8
    assert first.per_corpus["diffssd"].n_selected == 8

    expected_names = _expected_names(fixture)
    expected_file_list = "".join(
        f"{_sha256(fixture.payloads[name])}  {name}\n" for name in expected_names
    )
    assert first.file_list_path.read_text(encoding="utf-8") == expected_file_list

    with tarfile.open(first.archive_path, "r:gz") as archive:
        members = archive.getmembers()
        assert [member.name for member in members] == expected_names
        for member in members:
            assert member.name.startswith("datasets/")
            assert member.mtime == 0
            assert member.uid == 0
            assert member.gid == 0
            assert member.uname == ""
            assert member.gname == ""
            assert member.mode & 0o777 == 0o644
            extracted = archive.extractfile(member)
            assert extracted is not None
            assert extracted.read() == fixture.payloads[member.name]


def test_subset_checksum_mismatch_publishes_nothing(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    selected = _expected_names(fixture)[0]
    (fixture.data_root / selected).write_bytes(b"corrupted after checksums were recorded")
    output = tmp_path / "output" / "subset.tar.gz"

    with pytest.raises(SubsetBuildError, match=r"asvspoof5: checksum mismatch"):
        build_subset(
            fixture.data_root,
            output,
            manifest_dir=fixture.manifest_dir,
            checksum_dir=fixture.checksum_dir,
        )

    assert not output.exists()
    assert not Path(f"{output}.files.sha256").exists()
    assert not Path(f"{output}.sha256").exists()
