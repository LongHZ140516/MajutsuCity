#!/usr/bin/env python3
"""Scan the repository for common accidental open-source release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_TOP_LEVEL = {
    ".git",
    ".idea",
    ".vscode",
    "envs",
    "external",
    "models",
    "output",
    "outputs",
}
EXCLUDED_FILES = {Path("configs/paths.local.yaml")}
INCLUDED_FILES_IN_EXCLUDED_DIRS: set[Path] = set()
PUBLIC_MARKDOWN_PATHS = {
    Path("README.md"),
    Path("licenses/THIRD_PARTY_NOTICES.md"),
    Path("layout_gen/README.md"),
}
CACHE_NAMES = {".DS_Store", ".coverage"}
CACHE_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp", ".bak", ".swp"}
WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".npy",
    ".npz",
    ".onnx",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
}
PRIVATE_PATH_PATTERNS = (
    re.compile(r"/data\d+/"),
    re.compile(r"/home/[A-Za-z0-9._-]+/"),
    re.compile(r"/mnt/[A-Za-z0-9._-]+/"),
)
# These byte-identical upstream files contain their original authors' example
# filesystem paths. They are preserved for provenance and are not local machine
# configuration. Their content hash is audited separately against the official
# MajutsuCity snapshot; other secret/path checks still apply here.
UPSTREAM_PRIVATE_PATH_EXEMPTIONS = {
    Path("layout_gen/annotator/uniformer/mmcv/utils/config.py"):
        "17149353aefac6d737c67bb2f35a3a6cd2147b0a",
    Path("layout_gen/annotator/zoe/zoedepth/data/vkitti.py"):
        "72a2e5a8346f6e630ede0e28d6959725af8d7e72",
    Path("layout_gen/annotator/zoe/zoedepth/data/vkitti2.py"):
        "9bcfb0414b7f3f21859f30ae34bd71689516a3e7",
    Path("layout_gen/sample/height_sample.txt"):
        "7dd1c3b3486ebcef3fe2ca7ba1639352479abed1",
}
SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
)
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


@dataclass(frozen=True)
class Finding:
    category: str
    path: Path
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check source files without using Git for private absolute paths, "
            "token-shaped secrets, caches, model weights, large files and "
            "unexpected symbolic links."
        )
    )
    parser.add_argument(
        "--max-file-mb",
        type=float,
        default=50.0,
        help="Fail on source-tree files larger than this size (default: 50).",
    )
    parser.add_argument(
        "--forbidden-text",
        action="append",
        default=[],
        help="Additional literal private text to reject; may be repeated.",
    )
    return parser.parse_args()


def candidate_paths() -> tuple[list[Path], list[Path], list[Path]]:
    files: list[Path] = []
    symlinks: list[Path] = []
    cache_directories: list[Path] = []
    for current, directories, names in os.walk(PROJECT_ROOT, followlinks=False):
        current_path = Path(current)
        relative_current = current_path.relative_to(PROJECT_ROOT)
        if relative_current == Path("."):
            directories[:] = [
                name for name in directories if name not in EXCLUDED_TOP_LEVEL
            ]
        for name in directories:
            path = current_path / name
            if path.is_symlink():
                symlinks.append(path)
            elif name == "__pycache__":
                cache_directories.append(path)
        for name in names:
            path = current_path / name
            relative = path.relative_to(PROJECT_ROOT)
            if relative in EXCLUDED_FILES:
                continue
            if path.is_symlink():
                symlinks.append(path)
            elif path.is_file():
                files.append(path)
    for relative in INCLUDED_FILES_IN_EXCLUDED_DIRS:
        path = PROJECT_ROOT / relative
        if path.is_file():
            files.append(path)
    return sorted(set(files)), sorted(symlinks), sorted(cache_directories)


def git_blob_sha(path: Path) -> str:
    """Return Git's content-addressed SHA-1 without invoking Git."""
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def scan_text(path: Path, forbidden_text: list[str]) -> list[Finding]:
    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    findings = []
    relative = path.relative_to(PROJECT_ROOT)
    if relative not in UPSTREAM_PRIVATE_PATH_EXEMPTIONS:
        for pattern in PRIVATE_PATH_PATTERNS:
            if match := pattern.search(content):
                findings.append(
                    Finding("private path", path, f"matched {match.group(0)!r}")
                )
                break
    for pattern in SECRET_PATTERNS:
        if pattern.search(content):
            findings.append(
                Finding("possible secret", path, f"matched pattern {pattern.pattern!r}")
            )
            break
    for literal in forbidden_text:
        if literal and literal in content:
            findings.append(Finding("forbidden text", path, repr(literal)))
    return findings


def scan_markdown_links(path: Path) -> list[Finding]:
    if path.suffix.lower() != ".md":
        return []
    relative = path.relative_to(PROJECT_ROOT)
    if relative not in PUBLIC_MARKDOWN_PATHS and relative.parts[0] not in {
        "docs",
        "requirements",
    }:
        return []
    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    findings = []
    for raw_target in MARKDOWN_LINK.findall(content):
        target = raw_target.strip().strip("<>")
        if (
            not target
            or target.startswith("#")
            or "://" in target
            or target.startswith("mailto:")
        ):
            continue
        local_target = target.split("#", maxsplit=1)[0]
        if not local_target:
            continue
        resolved = (path.parent / local_target).resolve()
        if not resolved.exists():
            findings.append(Finding("broken Markdown link", path, target))
    return findings


def audit(args: argparse.Namespace) -> tuple[list[Finding], int]:
    files, symlinks, cache_directories = candidate_paths()
    findings: list[Finding] = []
    max_bytes = int(args.max_file_mb * 1024 * 1024)
    for path in files:
        relative = path.relative_to(PROJECT_ROOT)
        if expected_sha := UPSTREAM_PRIVATE_PATH_EXEMPTIONS.get(relative):
            actual_sha = git_blob_sha(path)
            if actual_sha != expected_sha:
                findings.append(
                    Finding(
                        "modified upstream path exemption",
                        path,
                        f"expected blob {expected_sha}, found {actual_sha}",
                    )
                )
        if (
            path.name in CACHE_NAMES
            or path.suffix.lower() in CACHE_SUFFIXES
            or path.name.endswith("~")
        ):
            findings.append(Finding("cache artifact", path, path.name))
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            findings.append(
                Finding("weight-like file", path, path.suffix.lower())
            )
        size = path.stat().st_size
        if size > max_bytes:
            findings.append(
                Finding(
                    "large file",
                    path,
                    f"{size / (1024 * 1024):.1f} MiB exceeds {args.max_file_mb:g} MiB",
                )
            )
        findings.extend(scan_text(path, args.forbidden_text))
        findings.extend(scan_markdown_links(path))
    for path in cache_directories:
        findings.append(Finding("cache directory", path, "remove before release"))
    for path in symlinks:
        findings.append(
            Finding("unexpected symlink", path, os.readlink(path))
        )
    return findings, len(files)


def main() -> None:
    args = parse_args()
    if args.max_file_mb <= 0:
        raise ValueError("--max-file-mb must be positive")
    findings, file_count = audit(args)
    if findings:
        for finding in findings:
            relative = finding.path.relative_to(PROJECT_ROOT)
            print(
                f"[FAIL] {finding.category}: {relative} ({finding.detail})",
                flush=True,
            )
        print(
            f"\nRelease audit failed: {len(findings)} finding(s) in "
            f"{file_count} scanned files.",
            flush=True,
        )
        raise SystemExit(1)
    print(
        f"Release audit passed: {file_count} source files scanned; no private "
        "paths, token-shaped secrets, caches, weights, oversized files, broken "
        "Markdown links or unexpected symlinks found.",
        flush=True,
    )


if __name__ == "__main__":
    main()
