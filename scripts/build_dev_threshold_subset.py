"""Build the checksum-verified development subset consumed by cross-test thresholding.

The selection is intentionally not configurable: it mirrors
``cross_test.threshold_from_dev`` exactly for ASVspoof5 and DiffSSD (v2 val rows,
a fresh ``random.Random(7)`` shuffle per corpus, then the first 1,000 rows).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import random
import re
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from audioshield.data.manifest import ManifestRow, read_manifest


PIN = Path("manifests/v2")
CHECKSUM_DIR = Path("manifests/checksums")
SEED = 7
LIMIT = 1000
_COREUTILS_LINE = re.compile(r"^([0-9A-Fa-f]{64}) ([ *])(.+)$")


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    archive_prefix: PurePosixPath
    checksum_name: str


CORPORA = (
    CorpusSpec(
        "asvspoof5",
        PurePosixPath("datasets/01_ASVspoof5"),
        "01_ASVspoof5_SHA256.txt",
    ),
    CorpusSpec(
        "diffssd",
        PurePosixPath("datasets/03_DiffSSD"),
        "03_DiffSSD_SHA256.txt",
    ),
)


class SubsetBuildError(RuntimeError):
    """The subset cannot be trusted or safely published."""


@dataclass(frozen=True)
class VerifiedFile:
    corpus: str
    archive_path: PurePosixPath
    source_path: Path
    sha256: str
    size: int


@dataclass(frozen=True)
class CorpusSummary:
    n_selected: int
    source_bytes: int


@dataclass(frozen=True)
class BuildResult:
    per_corpus: dict[str, CorpusSummary]
    n_selected: int
    source_bytes: int
    archive_bytes: int
    archive_sha256: str
    archive_path: Path
    file_list_path: Path
    archive_sha256_path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_rows(manifest_dir: Path, corpus: str) -> list[ManifestRow]:
    """Replicate the evaluator's development selection literally."""
    rows = read_manifest(
        Path(manifest_dir) / f"{corpus}.csv",
        splits=["val"],
        corpora=[corpus],
    )
    random.Random(SEED).shuffle(rows)
    rows = rows[:LIMIT]
    if not rows:
        raise SubsetBuildError(f"{corpus}: selected zero val rows from {manifest_dir}")
    return rows


def _archive_path(row: ManifestRow, spec: CorpusSpec) -> PurePosixPath:
    path = PurePosixPath(row.path.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise SubsetBuildError(f"{spec.name}: unsafe manifest path {row.path!r}")
    try:
        path.relative_to(spec.archive_prefix)
    except ValueError as exc:
        raise SubsetBuildError(
            f"{spec.name}: path {path} is outside required prefix {spec.archive_prefix}"
        ) from exc
    return path


def _required_checksums(checksum_path: Path, required: set[str]) -> dict[str, str]:
    if not checksum_path.is_file():
        raise SubsetBuildError(f"checksum manifest absent: {checksum_path}")

    found: dict[str, str] = {}
    with checksum_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.rstrip("\r\n")
            if not raw:
                continue
            match = _COREUTILS_LINE.fullmatch(raw)
            if match is None:
                raise SubsetBuildError(
                    f"{checksum_path}:{line_number}: malformed coreutils checksum line"
                )
            digest, _, name = match.groups()
            normalized = PurePosixPath(name.replace("\\", "/")).as_posix()
            if normalized not in required:
                continue
            if normalized in found:
                raise SubsetBuildError(
                    f"{checksum_path}:{line_number}: duplicate checksum entry {normalized}"
                )
            found[normalized] = digest.lower()

    missing = sorted(required - set(found))
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
        raise SubsetBuildError(
            f"{checksum_path}: {len(missing)} selected checksum entries absent: {preview}{suffix}"
        )
    return found


def _verify_selected(
    data_root: Path,
    manifest_dir: Path,
    checksum_dir: Path,
) -> tuple[list[VerifiedFile], dict[str, CorpusSummary]]:
    data_root = data_root.resolve(strict=False)
    verified: list[VerifiedFile] = []
    summaries: dict[str, CorpusSummary] = {}
    seen_archive_paths: set[str] = set()

    for spec in CORPORA:
        rows = select_rows(manifest_dir, spec.name)
        archive_paths = [_archive_path(row, spec) for row in rows]
        keys = {
            path.relative_to(spec.archive_prefix).as_posix()
            for path in archive_paths
        }
        if len(keys) != len(archive_paths):
            raise SubsetBuildError(f"{spec.name}: duplicate selected manifest paths")
        checksums = _required_checksums(checksum_dir / spec.checksum_name, keys)

        corpus_bytes = 0
        for archive_path in archive_paths:
            archive_name = archive_path.as_posix()
            if archive_name in seen_archive_paths:
                raise SubsetBuildError(f"duplicate archive member across corpora: {archive_name}")
            seen_archive_paths.add(archive_name)

            source_path = data_root.joinpath(*archive_path.parts).resolve(strict=False)
            try:
                source_path.relative_to(data_root)
            except ValueError as exc:
                raise SubsetBuildError(
                    f"{spec.name}: selected source escapes data root: {source_path}"
                ) from exc
            if not source_path.is_file():
                raise SubsetBuildError(f"{spec.name}: selected file absent: {source_path}")

            checksum_key = archive_path.relative_to(spec.archive_prefix).as_posix()
            expected = checksums[checksum_key]
            actual = sha256_file(source_path)
            if actual != expected:
                raise SubsetBuildError(
                    f"{spec.name}: checksum mismatch for {source_path}: "
                    f"expected {expected}, got {actual}"
                )
            size = source_path.stat().st_size
            corpus_bytes += size
            verified.append(
                VerifiedFile(spec.name, archive_path, source_path, actual, size)
            )

        summaries[spec.name] = CorpusSummary(len(archive_paths), corpus_bytes)

    return verified, summaries


def _temporary_path(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(name)


def _write_archive(path: Path, verified: list[VerifiedFile]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=raw,
            mtime=0,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.USTAR_FORMAT,
            ) as archive:
                for item in verified:
                    info = tarfile.TarInfo(item.archive_path.as_posix())
                    info.size = item.size
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with item.source_path.open("rb") as source:
                        archive.addfile(info, source)


def _verify_archive(path: Path, verified: list[VerifiedFile]) -> None:
    """Prove that the staged archive contains the bytes that were checksummed."""
    expected_names = [item.archive_path.as_posix() for item in verified]
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            actual_names = [member.name for member in members]
            if actual_names != expected_names:
                raise SubsetBuildError(
                    "staged archive member list differs from the verified selection"
                )

            for member, item in zip(members, verified, strict=True):
                if not member.isfile() or member.size != item.size:
                    raise SubsetBuildError(
                        f"staged archive metadata mismatch for {member.name}"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise SubsetBuildError(
                        f"cannot read staged archive member {member.name}"
                    )
                digest = hashlib.sha256()
                for chunk in iter(lambda: extracted.read(1024 * 1024), b""):
                    digest.update(chunk)
                actual = digest.hexdigest()
                if actual != item.sha256:
                    raise SubsetBuildError(
                        f"staged archive checksum mismatch for {member.name}: "
                        f"expected {item.sha256}, got {actual}"
                    )
    except (OSError, tarfile.TarError) as exc:
        raise SubsetBuildError(f"cannot verify staged archive {path}: {exc}") from exc


def build_subset(
    data_root: Path,
    output: Path,
    *,
    manifest_dir: Path = PIN,
    checksum_dir: Path = CHECKSUM_DIR,
    force: bool = False,
) -> BuildResult:
    """Verify and package the subset, publishing its digest marker last."""
    data_root = Path(data_root)
    output = Path(output)
    manifest_dir = Path(manifest_dir)
    checksum_dir = Path(checksum_dir)
    file_list_path = Path(f"{output}.files.sha256")
    archive_sha256_path = Path(f"{output}.sha256")
    published = (output, file_list_path, archive_sha256_path)
    existing = [path for path in published if path.exists()]
    if existing and not force:
        raise SubsetBuildError(
            "refusing to overwrite existing output(s): "
            + ", ".join(str(path) for path in existing)
        )

    verified, summaries = _verify_selected(data_root, manifest_dir, checksum_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = [_temporary_path(path) for path in published]
    temp_archive, temp_file_list, temp_archive_sha = temporary
    try:
        _write_archive(temp_archive, verified)
        _verify_archive(temp_archive, verified)
        archive_sha = sha256_file(temp_archive)
        temp_file_list.write_text(
            "".join(
                f"{item.sha256}  {item.archive_path.as_posix()}\n"
                for item in verified
            ),
            encoding="utf-8",
            newline="\n",
        )
        temp_archive_sha.write_text(
            f"{archive_sha}  {output.name}\n",
            encoding="utf-8",
            newline="\n",
        )
        temp_archive.replace(output)
        temp_file_list.replace(file_list_path)
        temp_archive_sha.replace(archive_sha256_path)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)

    return BuildResult(
        per_corpus=summaries,
        n_selected=len(verified),
        source_bytes=sum(item.size for item in verified),
        archive_bytes=output.stat().st_size,
        archive_sha256=archive_sha,
        archive_path=output,
        file_list_path=file_list_path,
        archive_sha256_path=archive_sha256_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=PIN)
    parser.add_argument("--checksum-dir", type=Path, default=CHECKSUM_DIR)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_subset(
            args.data_root,
            args.output,
            manifest_dir=args.manifest_dir,
            checksum_dir=args.checksum_dir,
            force=args.force,
        )
    except SubsetBuildError as exc:
        raise SystemExit(f"DEV THRESHOLD SUBSET: FAIL — {exc}") from exc

    for corpus, summary in result.per_corpus.items():
        print(
            f"{corpus:12s} n_selected={summary.n_selected} "
            f"source_bytes={summary.source_bytes}"
        )
    print(
        f"TOTAL n_selected={result.n_selected} source_bytes={result.source_bytes} "
        f"archive_bytes={result.archive_bytes}"
    )
    print(f"FILE_LIST {result.file_list_path}")
    print(f"ARCHIVE_SHA256 {result.archive_sha256}")
    print(f"ARCHIVE_SHA256_FILE {result.archive_sha256_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
