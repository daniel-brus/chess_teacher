#!/usr/bin/env python3
"""Read-only raw object storage inspection for chess_teacher (ObjectStorage only)."""

from __future__ import annotations

import os

os.environ["ENVIRONMENT"] = "AGENT"

import argparse
import json
import re
from collections.abc import Iterable
from fnmatch import fnmatch
from typing import Any

from chess_teacher.utils.env_utils import get_env_variable
from chess_teacher.utils.exception_utils import FileError
from chess_teacher.utils.object_storage.base import ObjectStorage
from chess_teacher.utils.object_storage.factory import get_raw_storage
from chess_teacher.utils.object_storage.health import check_raw_storage_health


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _normalize_prefix(prefix: str) -> str:
    return prefix.strip("/")


def _relative_under(key: str, prefix: str) -> str:
    normalized = _normalize_prefix(prefix)
    if not normalized:
        return key
    if key == normalized:
        return ""
    head = f"{normalized}/"
    if key.startswith(head):
        return key[len(head) :]
    return key


def _children(prefix: str, keys: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return (folder_names, file_names) immediately under ``prefix``."""
    folders: set[str] = set()
    files: set[str] = set()
    for key in keys:
        relative = _relative_under(key, prefix)
        if not relative:
            continue
        if "/" in relative:
            folders.add(relative.split("/", 1)[0])
        else:
            files.add(relative)
    return sorted(folders), sorted(files)


def _cmd_info(_: argparse.Namespace) -> dict[str, Any]:
    backend = get_env_variable("STORAGE_BACKEND")
    root = get_env_variable("STORAGE_ROOT")
    info: dict[str, Any] = {
        "backend": backend,
        "storage_root": root,
        "note": (
            "Keys are POSIX-style paths relative to storage_root. "
            "Object storage has no real folders; prefixes group keys."
        ),
    }
    if backend == "s3":
        info["bucket"] = get_env_variable("S3_BUCKET")
        info["endpoint_url"] = get_env_variable("S3_ENDPOINT_URL")
    return info


def _cmd_list(args: argparse.Namespace, storage: ObjectStorage) -> dict[str, Any]:
    prefix = _normalize_prefix(args.prefix or "")
    keys = storage.list_keys(
        prefix,
        recursive=not args.no_recursive,
        suffix=args.suffix,
        glob_pattern=args.glob,
    )
    if args.limit is not None:
        keys = keys[: args.limit]
    return {
        "prefix": prefix or "/",
        "recursive": not args.no_recursive,
        "suffix": args.suffix,
        "glob": args.glob,
        "count": len(keys),
        "keys": keys,
    }


def _cmd_children(args: argparse.Namespace, storage: ObjectStorage) -> dict[str, Any]:
    prefix = _normalize_prefix(args.prefix or "")
    keys = storage.list_keys(
        prefix,
        recursive=True,
        suffix=args.suffix,
        glob_pattern=args.glob,
    )
    folders, files = _children(prefix, keys)
    return {
        "prefix": prefix or "/",
        "suffix": args.suffix,
        "glob": args.glob,
        "folder_count": len(folders),
        "file_count": len(files),
        "folders": folders,
        "files": files,
    }


def _cmd_exists(args: argparse.Namespace, storage: ObjectStorage) -> dict[str, Any]:
    key = _normalize_prefix(args.key)
    data = storage.read_bytes(key)
    return {"key": key, "exists": data is not None}


def _cmd_any_under(args: argparse.Namespace, storage: ObjectStorage) -> dict[str, Any]:
    prefix = _normalize_prefix(args.prefix)
    keys = storage.list_keys(
        prefix,
        recursive=True,
        suffix=args.suffix,
        glob_pattern=args.glob,
    )
    if args.limit is not None:
        sample = keys[: args.limit]
    else:
        sample = keys[:10]
    return {
        "prefix": prefix,
        "suffix": args.suffix,
        "glob": args.glob,
        "any": bool(keys),
        "count": len(keys),
        "sample_keys": sample,
    }


def _cmd_count(args: argparse.Namespace, storage: ObjectStorage) -> dict[str, Any]:
    prefix = _normalize_prefix(args.prefix or "")
    keys = storage.list_keys(
        prefix,
        recursive=not args.no_recursive,
        suffix=args.suffix,
        glob_pattern=args.glob,
    )
    return {
        "prefix": prefix or "/",
        "recursive": not args.no_recursive,
        "suffix": args.suffix,
        "glob": args.glob,
        "count": len(keys),
    }


def _cmd_match(args: argparse.Namespace, storage: ObjectStorage) -> dict[str, Any]:
    prefix = _normalize_prefix(args.prefix or "")
    keys = storage.list_keys(prefix, recursive=True)
    pattern = args.pattern
    matched = [key for key in keys if fnmatch(key, pattern)]
    if args.limit is not None:
        matched = matched[: args.limit]
    return {
        "prefix": prefix or "/",
        "pattern": pattern,
        "count": len(matched),
        "keys": matched,
    }


def _cmd_health(_: argparse.Namespace, storage: ObjectStorage) -> dict[str, Any]:
    check_raw_storage_health(storage)
    return {"ok": True}


def _add_filter_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--suffix",
        default=None,
        help="File suffix filter (e.g. jsonl or .jsonl). Passed to ObjectStorage.list_keys.",
    )
    parser.add_argument(
        "--glob",
        default=None,
        help="Regex filter on relative keys. Passed to ObjectStorage.list_keys as glob_pattern.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON (always used; kept for parity)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Show configured backend and storage root (no secrets).")

    list_p = sub.add_parser("list", help="List object keys under a prefix.")
    list_p.add_argument("prefix", nargs="?", default="", help="Key prefix (default: storage root).")
    list_p.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only direct files under prefix (no keys in sub-prefixes).",
    )
    list_p.add_argument("--limit", type=int, default=None, help="Return at most N keys.")
    _add_filter_flags(list_p)

    children_p = sub.add_parser(
        "children",
        help="List immediate sub-prefixes (folders) and direct files under a prefix.",
    )
    children_p.add_argument(
        "prefix", nargs="?", default="", help="Key prefix (default: storage root)."
    )
    _add_filter_flags(children_p)

    exists_p = sub.add_parser("exists", help="Check whether a single object key exists.")
    exists_p.add_argument("key", help="Object key relative to storage root.")

    any_p = sub.add_parser(
        "any-under",
        help="Whether any objects exist under a prefix (prefix 'exists' check).",
    )
    any_p.add_argument("prefix", help="Key prefix to probe.")
    any_p.add_argument(
        "--limit", type=int, default=10, help="Sample keys in response (default: 10)."
    )
    _add_filter_flags(any_p)

    count_p = sub.add_parser("count", help="Count object keys under a prefix.")
    count_p.add_argument(
        "prefix", nargs="?", default="", help="Key prefix (default: storage root)."
    )
    count_p.add_argument("--no-recursive", action="store_true")
    _add_filter_flags(count_p)

    match_p = sub.add_parser(
        "match",
        help="List keys under prefix matching a shell-style glob (fnmatch on full key).",
    )
    match_p.add_argument("pattern", help="Shell glob, e.g. 'ingested/*/*.jsonl'.")
    match_p.add_argument("prefix", nargs="?", default="", help="Limit search to this prefix first.")
    match_p.add_argument("--limit", type=int, default=None)
    _add_filter_flags(match_p)

    sub.add_parser("health", help="Run read/write/list/delete storage health probe.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    storage = get_raw_storage()

    handlers: dict[str, Any] = {
        "info": lambda: _cmd_info(args),
        "list": lambda: _cmd_list(args, storage),
        "children": lambda: _cmd_children(args, storage),
        "exists": lambda: _cmd_exists(args, storage),
        "any-under": lambda: _cmd_any_under(args, storage),
        "count": lambda: _cmd_count(args, storage),
        "match": lambda: _cmd_match(args, storage),
        "health": lambda: _cmd_health(args, storage),
    }

    try:
        result = handlers[args.command]()
    except FileError as exc:
        _emit({"error": str(exc), "command": args.command})
        raise SystemExit(1) from exc
    except ValueError as exc:
        _emit({"error": str(exc), "command": args.command})
        raise SystemExit(1) from exc
    except re.error as exc:
        _emit({"error": f"Invalid glob regex: {exc}", "command": args.command})
        raise SystemExit(1) from exc

    _emit({"command": args.command, **result})


if __name__ == "__main__":
    main()
