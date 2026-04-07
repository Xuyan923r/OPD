#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys
from urllib.parse import unquote, urlparse
from pathlib import Path


def _module_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "-"


def _record_module(import_name: str, package_name: str | None = None) -> tuple[bool, str, str]:
    package_name = package_name or import_name
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        return False, "ModuleSpec not found", "-"
    if spec.origin is not None:
        path = spec.origin
    elif spec.submodule_search_locations:
        path = str(next(iter(spec.submodule_search_locations)))
    else:
        path = "-"
    return True, _module_version(package_name), path


def _editable_location(package_name: str) -> Path | None:
    try:
        dist = importlib.metadata.distribution(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    text = dist.read_text("direct_url.json")
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    url = payload.get("url")
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path)).resolve()


def _samefile(left: Path, right: Path) -> bool:
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def _check_path(path_str: str | None, label: str, must_be_dir: bool) -> str | None:
    if path_str is None:
        return None
    path = Path(path_str).resolve()
    if not path.exists():
        return f"{label} not found: {path}"
    if must_be_dir and not path.is_dir():
        return f"{label} is not a directory: {path}"
    if not must_be_dir and not path.is_file():
        return f"{label} is not a file: {path}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether the local slime/OPD runtime is ready.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--sglang-src")
    parser.add_argument("--megatron-src")
    parser.add_argument("--teacher-model")
    parser.add_argument("--student-model")
    parser.add_argument("--raw-data")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    failures: list[str] = []

    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")

    checks = [
        ("torch", "torch"),
        ("ray", "ray"),
        ("sglang", "sglang"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("omegaconf", "omegaconf"),
        ("flash_attn", "flash_attn"),
        ("ring_flash_attn", "ring_flash_attn"),
        ("megatron.core", "megatron-core"),
        ("slime", "slime"),
    ]
    for import_name, package_name in checks:
        ok, version, path = _record_module(import_name, package_name)
        status = "OK" if ok else "FAIL"
        print(f"{status:4} {import_name:18} version={version:12} path={path}")
        if not ok:
            failures.append(f"import {import_name} failed: {version}")

    slime_spec = importlib.util.find_spec("slime")
    if slime_spec is None or slime_spec.origin is None:
        failures.append("unable to resolve slime module path from import spec")
    else:
        slime_root = Path(slime_spec.origin).resolve().parents[1]
        if not _samefile(slime_root, repo_root):
            failures.append(f"slime package is not pointing at repo root: expected {repo_root}, got {slime_root}")

    if args.sglang_src:
        actual = _editable_location("sglang")
        expected = (Path(args.sglang_src).resolve() / "python").resolve()
        if actual is None:
            failures.append("unable to read editable location for sglang from distribution metadata")
        elif not _samefile(actual, expected):
            failures.append(f"sglang is not pointing at expected source tree: expected {expected}, got {actual}")

    if args.megatron_src:
        actual = _editable_location("megatron-core")
        expected = Path(args.megatron_src).resolve()
        if actual is None:
            failures.append("unable to read editable location for megatron-core from distribution metadata")
        elif not _samefile(actual, expected):
            failures.append(f"megatron-core is not pointing at expected source tree: expected {expected}, got {actual}")

    for path_str, label, must_be_dir in [
        (str(repo_root), "Repository root", True),
        (args.teacher_model, "Teacher model", True),
        (args.student_model, "Student model", True),
        (args.raw_data, "Raw data", False),
    ]:
        error = _check_path(path_str, label, must_be_dir)
        if error:
            failures.append(error)

    if failures:
        print("\nEnvironment check failed:")
        for item in failures:
            print(f"- {item}")
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
