#!/usr/bin/env python3
"""Read-only log investigation for chess_teacher (local buffer + shipped S3 segments)."""

from __future__ import annotations

import os

os.environ["ENVIRONMENT"] = "AGENT"

import argparse
import json
import re
from collections import deque
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from chess_teacher.utils.logging.buffer import (
    LOG_STORAGE_PREFIX,
    READY_SUFFIX,
    LogBufferWriterLock,
    get_log_buffer_dir,
)
from chess_teacher.utils.logging.shipping import is_log_ship_enabled, log_storage_key_for_segment

_LINE_LIMIT_DEFAULT = 50
_SEARCH_LIMIT_DEFAULT = 30


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _safe_buffer_dir() -> Path | None:
    try:
        return get_log_buffer_dir()
    except Exception:
        return None


def _hostname() -> str | None:
    return os.getenv("HOSTNAME")


def _parse_since(value: str) -> datetime:
    text = value.strip().lower()
    now = datetime.now(UTC)
    relative = re.fullmatch(r"(\d+)(m|h|d)", text)
    if relative:
        amount = int(relative.group(1))
        unit = relative.group(2)
        if unit == "m":
            return now - timedelta(minutes=amount)
        if unit == "h":
            return now - timedelta(hours=amount)
        return now - timedelta(days=amount)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit(
            f"Invalid --since {value!r}. Use ISO timestamp or relative like 30m, 2h, 1d."
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _parse_json_line(line: str, *, source: str, line_no: int) -> dict[str, Any] | None:
    text = line.strip()
    if not text:
        return None
    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        return {"ts": None, "level": "PARSE_ERROR", "logger": source, "msg": text, "log_id": None}
    if not isinstance(record, dict):
        return None
    record["_source"] = source
    record["_line"] = line_no
    return record


def _iter_file_records(path: Path, *, max_lines: int | None = None) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            if max_lines is None:
                for line_no, line in enumerate(handle, start=1):
                    record = _parse_json_line(line, source=str(path), line_no=line_no)
                    if record is not None:
                        yield record
                return
            tail: deque[str] = deque(maxlen=max_lines)
            for line in handle:
                tail.append(line)
            for line_no, line in enumerate(tail, start=1):
                record = _parse_json_line(line, source=str(path), line_no=line_no)
                if record is not None:
                    yield record
    except OSError:
        return


def _record_ts(record: dict[str, Any]) -> datetime | None:
    raw = record.get("ts")
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _matches_filters(
    record: dict[str, Any],
    *,
    level: str | None,
    logger: str | None,
    contains: str | None,
    environment: str | None,
    since: datetime | None,
) -> bool:
    if level and str(record.get("level", "")).upper() != level.upper():
        return False
    if logger and logger not in str(record.get("logger", "")):
        return False
    if environment and str(record.get("environment", "")).upper() != environment.upper():
        return False
    if contains:
        haystack = json.dumps(record, ensure_ascii=False).lower()
        if contains.lower() not in haystack:
            return False
    if since is not None:
        ts = _record_ts(record)
        if ts is None or ts < since:
            return False
    return True


def _local_active_paths(buffer_dir: Path) -> list[Path]:
    active_dir = buffer_dir / "active"
    if not active_dir.is_dir():
        return []
    paths = sorted(active_dir.glob("*.log"))
    return [path for path in paths if path.is_file()]


def _local_ready_segments(buffer_dir: Path) -> list[Path]:
    closed_dir = buffer_dir / "closed"
    if not closed_dir.is_dir():
        return []
    return sorted(closed_dir.rglob(f"*{READY_SUFFIX}"))


def _local_closed_uploaded_paths(buffer_dir: Path) -> list[Path]:
    """Closed segments without .ready suffix (already uploaded or legacy)."""
    closed_dir = buffer_dir / "closed"
    if not closed_dir.is_dir():
        return []
    paths: list[Path] = []
    for path in sorted(closed_dir.rglob("*.log")):
        if path.is_file() and not str(path).endswith(READY_SUFFIX):
            paths.append(path)
    return paths


def _segment_descriptor_local(path: Path, buffer_dir: Path) -> dict[str, Any]:
    stat = path.stat()
    rel = path.relative_to(buffer_dir).as_posix()
    key = None
    if path.name.endswith(READY_SUFFIX):
        key = log_storage_key_for_segment(path, buffer_dir)
    return {
        "location": "local",
        "path": rel,
        "storage_key": key,
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        "pending_upload": path.name.endswith(READY_SUFFIX),
    }


def _get_storage():
    from chess_teacher.utils.object_storage.factory import get_raw_storage

    return get_raw_storage()


def _s3_segment_keys(prefix: str, *, limit: int) -> list[str]:
    storage = _get_storage()
    keys = storage.list_keys(prefix, recursive=True, suffix="log")
    keys = [key for key in keys if key.startswith(LOG_STORAGE_PREFIX)]
    keys.sort()
    if limit:
        keys = keys[-limit:]
    return keys


def cmd_info(_: argparse.Namespace) -> None:
    buffer_dir = _safe_buffer_dir()
    lock_holder = None
    if buffer_dir is not None:
        lock_holder = LogBufferWriterLock(buffer_dir).holder_pid()
    ship_enabled = is_log_ship_enabled()
    _emit({
        "environment": os.environ.get("ENVIRONMENT"),
        "log_buffer_dir": str(buffer_dir) if buffer_dir else None,
        "log_ship_enabled": ship_enabled,
        "hostname": _hostname(),
        "s3_prefix": LOG_STORAGE_PREFIX,
        "local_layout": {
            "active": "active/app.log (primary) or active/worker-{pid}.log",
            "closed_pending": f"closed/{{YYYY}}/{{MM}}/{{DD}}/{{HOSTNAME}}/app-{{HHMMSS}}Z.log{READY_SUFFIX}",
            "shipped_key": f"{LOG_STORAGE_PREFIX}/closed/{{YYYY}}/{{MM}}/{{DD}}/{{HOSTNAME}}/app-{{HHMMSS}}Z.log",
        },
        "json_fields": [
            "ts",
            "level",
            "logger",
            "msg",
            "log_id",
            "environment",
            "exc_type",
            "exc_msg",
            "traceback",
        ],
        "writer_lock_pid": lock_holder,
        "routing_hint": (
            "Recent/local → buffer-status + tail/search --source local. "
            "Historical/shipped → search --source s3. "
            "Live prod container stdout → chess-teacher-vps logs (not this skill)."
        ),
    })


def cmd_buffer_status(_: argparse.Namespace) -> None:
    buffer_dir = _safe_buffer_dir()
    if buffer_dir is None:
        raise SystemExit("LOG_BUFFER_DIR is not set.")
    active_files = _local_active_paths(buffer_dir)
    ready_segments = _local_ready_segments(buffer_dir)
    lock_holder = LogBufferWriterLock(buffer_dir).holder_pid()
    active_info = []
    for path in active_files:
        stat = path.stat()
        active_info.append({
            "path": path.relative_to(buffer_dir).as_posix(),
            "size_bytes": stat.st_size,
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        })
    pending = [_segment_descriptor_local(path, buffer_dir) for path in ready_segments[-20:]]
    _emit({
        "log_buffer_dir": str(buffer_dir),
        "log_ship_enabled": is_log_ship_enabled(),
        "writer_lock_pid": lock_holder,
        "active_files": active_info,
        "pending_upload_count": len(ready_segments),
        "pending_upload_recent": pending,
        "note": (
            "pending_upload_count > 0 with log_ship_enabled=true may indicate upload failures "
            "or a process that has not shipped yet (scan every ~60s)."
        ),
    })


def cmd_tail(args: argparse.Namespace) -> None:
    buffer_dir = _safe_buffer_dir()
    if buffer_dir is None:
        raise SystemExit("LOG_BUFFER_DIR is not set.")
    paths = _local_active_paths(buffer_dir)
    if not paths:
        _emit({"log_buffer_dir": str(buffer_dir), "lines": [], "note": "No active log files."})
        return
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_iter_file_records(path, max_lines=args.lines))
    records = records[-args.lines :]
    _emit({
        "log_buffer_dir": str(buffer_dir),
        "active_files": [path.relative_to(buffer_dir).as_posix() for path in paths],
        "line_count": len(records),
        "lines": records,
    })


def _collect_local_search_paths(
    buffer_dir: Path,
    *,
    include_active: bool,
    include_pending: bool,
    include_uploaded_local: bool,
) -> list[Path]:
    paths: list[Path] = []
    if include_active:
        paths.extend(_local_active_paths(buffer_dir))
    if include_pending:
        paths.extend(_local_ready_segments(buffer_dir))
    if include_uploaded_local:
        paths.extend(_local_closed_uploaded_paths(buffer_dir))
    return paths


def _search_paths(
    paths: list[Path],
    *,
    level: str | None,
    logger: str | None,
    contains: str | None,
    environment: str | None,
    since: datetime | None,
    limit: int,
    reverse: bool,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    iterable: Iterator[Path] = reversed(paths) if reverse else iter(paths)
    for path in iterable:
        for record in _iter_file_records(path):
            if _matches_filters(
                record,
                level=level,
                logger=logger,
                contains=contains,
                environment=environment,
                since=since,
            ):
                matches.append(record)
                if len(matches) >= limit:
                    return matches
    return matches


def _search_s3(
    *,
    date_prefix: str | None,
    hostname: str | None,
    level: str | None,
    logger: str | None,
    contains: str | None,
    environment: str | None,
    since: datetime | None,
    limit: int,
    segment_limit: int,
) -> list[dict[str, Any]]:
    prefix = LOG_STORAGE_PREFIX
    if date_prefix:
        prefix = f"{LOG_STORAGE_PREFIX}/closed/{date_prefix.strip('/')}"
    if hostname:
        prefix = f"{prefix.rstrip('/')}/{hostname}"
    keys = _s3_segment_keys(prefix, limit=segment_limit)
    storage = _get_storage()
    matches: list[dict[str, Any]] = []
    for key in reversed(keys):
        data = storage.read_bytes(key)
        if data is None:
            continue
        text = data.decode("utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            record = _parse_json_line(line, source=key, line_no=line_no)
            if record is None:
                continue
            if _matches_filters(
                record,
                level=level,
                logger=logger,
                contains=contains,
                environment=environment,
                since=since,
            ):
                matches.append(record)
                if len(matches) >= limit:
                    return matches
    return matches


def cmd_search(args: argparse.Namespace) -> None:
    since = _parse_since(args.since) if args.since else None
    source = args.source
    matches: list[dict[str, Any]] = []

    if source in {"local", "both"}:
        buffer_dir = _safe_buffer_dir()
        if buffer_dir is None:
            if source == "local":
                raise SystemExit("LOG_BUFFER_DIR is not set.")
        else:
            paths = _collect_local_search_paths(
                buffer_dir,
                include_active=True,
                include_pending=True,
                include_uploaded_local=True,
            )
            matches.extend(
                _search_paths(
                    paths,
                    level=args.level,
                    logger=args.logger,
                    contains=args.contains,
                    environment=args.environment,
                    since=since,
                    limit=args.limit,
                    reverse=True,
                )
            )

    if source in {"s3", "both"} and len(matches) < args.limit:
        remaining = args.limit - len(matches)
        s3_matches = _search_s3(
            date_prefix=args.date,
            hostname=args.hostname,
            level=args.level,
            logger=args.logger,
            contains=args.contains,
            environment=args.environment,
            since=since,
            limit=remaining,
            segment_limit=args.segment_limit,
        )
        matches.extend(s3_matches)

    _emit({
        "source": source,
        "match_count": len(matches),
        "filters": {
            "level": args.level,
            "logger": args.logger,
            "contains": args.contains,
            "environment": args.environment,
            "since": since.isoformat() if since else None,
            "date": args.date,
            "hostname": args.hostname,
        },
        "matches": matches[: args.limit],
    })


def cmd_segments(args: argparse.Namespace) -> None:
    segments: list[dict[str, Any]] = []
    buffer_dir = _safe_buffer_dir()
    if buffer_dir is not None and args.source in {"local", "both"}:
        for path in _local_ready_segments(buffer_dir):
            rel = path.relative_to(buffer_dir)
            if args.date and args.date not in rel.as_posix():
                continue
            if args.hostname and args.hostname not in rel.as_posix():
                continue
            segments.append(_segment_descriptor_local(path, buffer_dir))
        for path in _local_closed_uploaded_paths(buffer_dir):
            rel = path.relative_to(buffer_dir)
            if args.date and args.date not in rel.as_posix():
                continue
            if args.hostname and args.hostname not in rel.as_posix():
                continue
            segments.append(_segment_descriptor_local(path, buffer_dir))

    if args.source in {"s3", "both"}:
        prefix = f"{LOG_STORAGE_PREFIX}/closed"
        if args.date:
            prefix = f"{prefix}/{args.date.strip('/')}"
        if args.hostname:
            prefix = f"{prefix.rstrip('/')}/{args.hostname}"
        for key in _s3_segment_keys(prefix, limit=args.limit):
            segments.append({
                "location": "s3",
                "storage_key": key,
                "pending_upload": False,
            })

    _emit({
        "source": args.source,
        "segment_count": len(segments),
        "segments": segments[: args.limit],
    })


def cmd_read_segment(args: argparse.Namespace) -> None:
    target = args.target.strip()
    buffer_dir = _safe_buffer_dir()
    records: list[dict[str, Any]] = []
    source_label = target

    if target.startswith(LOG_STORAGE_PREFIX):
        data = _get_storage().read_bytes(target)
        if data is None:
            raise SystemExit(f"S3 segment not found: {target}")
        text = data.decode("utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            record = _parse_json_line(line, source=target, line_no=line_no)
            if record and _matches_filters(
                record,
                level=args.level,
                logger=args.logger,
                contains=args.contains,
                environment=args.environment,
                since=_parse_since(args.since) if args.since else None,
            ):
                records.append(record)
    elif buffer_dir is not None:
        path = Path(target)
        if not path.is_absolute():
            path = buffer_dir / target
        if not path.is_file():
            raise SystemExit(f"Local segment not found: {path}")
        source_label = str(path.relative_to(buffer_dir))
        for record in _iter_file_records(path):
            if _matches_filters(
                record,
                level=args.level,
                logger=args.logger,
                contains=args.contains,
                environment=args.environment,
                since=_parse_since(args.since) if args.since else None,
            ):
                records.append(record)
    else:
        raise SystemExit("Provide an S3 key or set LOG_BUFFER_DIR for local paths.")

    if args.tail:
        records = records[-args.tail :]
    _emit({"target": source_label, "match_count": len(records), "lines": records})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON (always used).")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Buffer layout, env flags, routing hints.")
    sub.add_parser("buffer-status", help="Active files, pending .ready upload count.")

    tail_p = sub.add_parser("tail", help="Last N JSON log lines from local active file(s).")
    tail_p.add_argument("--lines", type=int, default=_LINE_LIMIT_DEFAULT)

    search_p = sub.add_parser("search", help="Filter JSON log lines across local buffer and/or S3.")
    search_p.add_argument(
        "--source",
        choices=["local", "s3", "both"],
        default="local",
        help="local=LOG_BUFFER_DIR; s3=shipped segments; both=merge (default local).",
    )
    search_p.add_argument("--level", help="Exact level, e.g. ERROR.")
    search_p.add_argument("--logger", help="Substring match on logger name.")
    search_p.add_argument("--contains", help="Substring match anywhere in the record JSON.")
    search_p.add_argument("--environment", help="Exact ENVIRONMENT field value, e.g. DEV, PROD.")
    search_p.add_argument("--since", help="ISO timestamp or relative: 30m, 2h, 1d.")
    search_p.add_argument("--date", help="Limit S3/local path to closed/YYYY/MM/DD segment date.")
    search_p.add_argument(
        "--hostname", help="Filter segments by HOSTNAME folder (pod name in prod)."
    )
    search_p.add_argument("--limit", type=int, default=_SEARCH_LIMIT_DEFAULT)
    search_p.add_argument(
        "--segment-limit",
        type=int,
        default=20,
        help="Max S3 segments to scan (most recent).",
    )

    seg_p = sub.add_parser("segments", help="List log segments (local pending + S3 shipped).")
    seg_p.add_argument("--source", choices=["local", "s3", "both"], default="both")
    seg_p.add_argument("--date", help="Filter by closed/YYYY/MM/DD.")
    seg_p.add_argument("--hostname", help="Filter by hostname/pod folder.")
    seg_p.add_argument("--limit", type=int, default=50)

    read_p = sub.add_parser("read-segment", help="Read one local relative path or S3 storage key.")
    read_p.add_argument(
        "target", help="Relative path under LOG_BUFFER_DIR or S3 key under logs/python/buffer/..."
    )
    read_p.add_argument("--level")
    read_p.add_argument("--logger")
    read_p.add_argument("--contains")
    read_p.add_argument("--environment")
    read_p.add_argument("--since")
    read_p.add_argument("--tail", type=int, default=None, help="Return only last N matching lines.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    handlers = {
        "info": cmd_info,
        "buffer-status": cmd_buffer_status,
        "tail": cmd_tail,
        "search": cmd_search,
        "segments": cmd_segments,
        "read-segment": cmd_read_segment,
    }
    try:
        handlers[args.command](args)
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
