#!/usr/bin/env python3
"""Read-only Redis inspection for chess_teacher (redis-py read commands only)."""

from __future__ import annotations

import os

os.environ["ENVIRONMENT"] = "AGENT"

import argparse
import json
import re
from typing import Any
from urllib.parse import urlparse

_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9:_\-\*\?\[\]]+$")
_SINGLE_KEY = re.compile(r"^[a-zA-Z0-9:_\-]+$")


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _redis_endpoint_label(redis_url: str) -> str:
    parsed = urlparse(redis_url)
    host = parsed.hostname or "unknown-host"
    if parsed.port is not None:
        return f"{host}:{parsed.port}"
    return host


def _get_client() -> tuple[Any, str, int]:
    redis_url = os.getenv("REDIS_URL", "").strip()
    if not redis_url:
        raise SystemExit("REDIS_URL is not set. Run via Doppler (see skill) or set .env.")
    parsed = urlparse(redis_url)
    db = 0
    if parsed.path and parsed.path != "/":
        try:
            db = int(parsed.path.lstrip("/"))
        except ValueError as exc:
            raise SystemExit(f"Invalid Redis DB in REDIS_URL path: {parsed.path}") from exc
    try:
        import redis
    except ImportError as exc:
        raise SystemExit(
            "redis package not installed. pip install -r requirements-dev.txt"
        ) from exc
    client = redis.from_url(redis_url, decode_responses=False)
    endpoint = _redis_endpoint_label(redis_url)
    return client, endpoint, db


def _validate_pattern(pattern: str) -> str:
    text = pattern.strip()
    if not text:
        raise SystemExit("Pattern must not be empty.")
    if not _KEY_PATTERN.fullmatch(text):
        raise SystemExit("Pattern contains disallowed characters.")
    return text


def _validate_key(key: str) -> str:
    text = key.strip()
    if not text:
        raise SystemExit("Key must not be empty.")
    if not _SINGLE_KEY.fullmatch(text):
        raise SystemExit("Key contains disallowed characters (no wildcards).")
    return text


def cmd_info(_: argparse.Namespace) -> None:
    redis_url = os.getenv("REDIS_URL", "").strip()
    parsed = urlparse(redis_url) if redis_url else None
    db = 0
    if parsed and parsed.path and parsed.path != "/":
        db = int(parsed.path.lstrip("/") or "0")
    _emit({
        "environment": os.environ.get("ENVIRONMENT"),
        "redis_url_set": bool(redis_url),
        "endpoint": _redis_endpoint_label(redis_url) if redis_url else None,
        "db": db,
        "scheme": parsed.scheme if parsed else None,
        "cache_key_patterns": [
            "user:{user_id}:games:v1",
            "user:{user_id}:accounts:v1",
        ],
        "note": "Credentials are never printed. Use ping to test connectivity.",
    })


def cmd_ping(_: argparse.Namespace) -> None:
    client, endpoint, db = _get_client()
    ok = bool(client.ping())
    _emit({"endpoint": endpoint, "db": db, "ping": ok})


def cmd_dbsize(_: argparse.Namespace) -> None:
    client, endpoint, db = _get_client()
    count = int(client.dbsize())
    _emit({"endpoint": endpoint, "db": db, "dbsize": count})


def cmd_scan(args: argparse.Namespace) -> None:
    pattern = _validate_pattern(args.pattern)
    client, endpoint, db = _get_client()
    keys: list[str] = []
    for key in client.scan_iter(match=pattern.encode("utf-8"), count=args.count):
        keys.append(key.decode("utf-8", errors="replace"))
        if len(keys) >= args.limit:
            break
    _emit({
        "endpoint": endpoint,
        "db": db,
        "pattern": pattern,
        "count": len(keys),
        "keys": keys,
        "truncated": len(keys) >= args.limit,
    })


def cmd_exists(args: argparse.Namespace) -> None:
    key = _validate_key(args.key)
    client, endpoint, db = _get_client()
    exists = bool(client.exists(key))
    _emit({"endpoint": endpoint, "db": db, "key": key, "exists": exists})


def cmd_key_info(args: argparse.Namespace) -> None:
    key = _validate_key(args.key)
    client, endpoint, db = _get_client()
    exists = bool(client.exists(key))
    if not exists:
        _emit({"endpoint": endpoint, "db": db, "key": key, "exists": False})
        return
    key_type = client.type(key).decode("utf-8", errors="replace")
    ttl = int(client.ttl(key))
    size_bytes: int | None = None
    if key_type == "string":
        size_bytes = int(client.strlen(key))
    payload: dict[str, Any] = {
        "endpoint": endpoint,
        "db": db,
        "key": key,
        "exists": True,
        "type": key_type,
        "ttl_seconds": ttl,
        "size_bytes": size_bytes,
    }
    if args.show_value and key_type == "string" and size_bytes is not None:
        if size_bytes > args.max_value_bytes:
            payload["value_skipped"] = f"size {size_bytes} > max {args.max_value_bytes}"
        else:
            raw = client.get(key)
            if raw is not None and key.endswith(":accounts:v1"):
                try:
                    payload["value_preview"] = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    payload["value_preview"] = "<invalid json>"
            elif raw is not None:
                payload["value_preview"] = f"<binary {len(raw)} bytes; not decoded>"
    _emit(payload)


def cmd_memory(_: argparse.Namespace) -> None:
    client, endpoint, db = _get_client()
    info = client.info("memory")
    _emit({
        "endpoint": endpoint,
        "db": db,
        "used_memory_human": info.get("used_memory_human"),
        "used_memory_peak_human": info.get("used_memory_peak_human"),
        "maxmemory_human": info.get("maxmemory_human"),
        "maxmemory_policy": info.get("maxmemory_policy"),
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON (always used; kept for parity with other skills).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Show endpoint/db and cache key patterns (no network).")
    sub.add_parser("ping", help="PING Redis (read-only connectivity check).")
    sub.add_parser("dbsize", help="Count keys in the selected DB.")
    sub.add_parser("memory", help="Summarize Redis memory INFO (read-only).")

    scan_p = sub.add_parser("scan", help="SCAN keys matching a pattern (read-only).")
    scan_p.add_argument(
        "--pattern",
        default="user:*",
        help="Key pattern (default: user:* — app cache keys).",
    )
    scan_p.add_argument("--limit", type=int, default=50, help="Max keys to return.")
    scan_p.add_argument("--count", type=int, default=100, help="SCAN COUNT hint per iteration.")

    exists_p = sub.add_parser("exists", help="Check whether a single key exists.")
    exists_p.add_argument("key")

    key_p = sub.add_parser("key-info", help="Key metadata: type, TTL, size (no value by default).")
    key_p.add_argument("key")
    key_p.add_argument(
        "--show-value",
        action="store_true",
        help="Include small JSON preview for accounts keys only.",
    )
    key_p.add_argument(
        "--max-value-bytes",
        type=int,
        default=65536,
        help="Skip value preview when string value exceeds this size.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    handlers = {
        "info": cmd_info,
        "ping": cmd_ping,
        "dbsize": cmd_dbsize,
        "scan": cmd_scan,
        "exists": cmd_exists,
        "key-info": cmd_key_info,
        "memory": cmd_memory,
    }
    try:
        handlers[args.command](args)
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
