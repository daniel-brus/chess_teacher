#!/usr/bin/env python3
"""Read-only Hetzner VPS inspection via SSH (whitelisted remote commands only)."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

os_environ_agent = __import__("os")
os_environ_agent.environ["ENVIRONMENT"] = "AGENT"

DOPPLER_PROJECT = "chess-teacher"
DOPPLER_CONFIG = "ci"
NAMESPACE = "chess-teacher"
STREAMLIT_DEPLOYMENT = "streamlit"
DB_QUERY_SCRIPT = "scripts/agent_db_query.py"
K8S_NAME = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
DOMAIN_ID_RE = re.compile(r"^[a-z0-9]([a-z0-9/_-]*[a-z0-9])?$")
TABLE_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
COLUMN_LIST_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(,[a-zA-Z_][a-zA-Z0-9_]*)*$")
_WRITE_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|truncate|alter|create|grant|revoke|merge|copy)\b",
    re.IGNORECASE,
)


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _require_doppler() -> None:
    if shutil.which("doppler") is None:
        raise SystemExit(
            "Doppler CLI not found. Install and run `doppler login` "
            "(needs access to chess-teacher / ci config)."
        )


def _doppler_get(key: str) -> str:
    _require_doppler()
    result = subprocess.run(
        [
            "doppler",
            "secrets",
            "get",
            key,
            "--project",
            DOPPLER_PROJECT,
            "--config",
            DOPPLER_CONFIG,
            "--plain",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise SystemExit(
            f"Missing or unreadable Doppler secret '{key}' in config '{DOPPLER_CONFIG}'."
        )
    return result.stdout.strip()


def _require_ssh() -> None:
    if shutil.which("ssh") is None:
        raise SystemExit("OpenSSH client (`ssh`) not found on PATH.")


def _validate_k8s_name(name: str, *, label: str) -> str:
    text = name.strip()
    if not text or len(text) > 253 or not K8S_NAME.fullmatch(text):
        raise SystemExit(f"Invalid {label}: {name!r}")
    return text


def _write_ssh_key(path: Path, key_material: str) -> None:
    """Write an OpenSSH private key with LF newlines (CRLF breaks Windows ssh)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    path.write_text(key_material.strip() + "\n", encoding="utf-8", newline="\n")
    path.chmod(0o600)


def _lock_down_windows_key_acl(path: Path) -> None:
    """OpenSSH on Windows rejects keys readable by other principals (e.g. sandbox groups)."""
    username = os.environ.get("USERNAME", "").strip()
    if not username:
        return
    subprocess.run(
        ["icacls", str(path), "/inheritance:r", "/grant:r", f"{username}:(R)"],
        capture_output=True,
        text=True,
        check=False,
    )


def _ssh_key_path() -> Path:
    if os.name == "nt":
        # Avoid %TEMP%: Cursor/sandbox agents add extra ACLs that OpenSSH rejects.
        base = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "chess-teacher-agent"
        return base / f"vps-deploy-{os.getpid()}.key"
    fd, name = tempfile.mkstemp(suffix=".key")
    os.close(fd)
    return Path(name)


def _ssh_run(remote_command: str, *, timeout: int = 120) -> dict[str, Any]:
    """Run a single whitelisted remote command. Never pass user-supplied shell."""
    _require_ssh()
    host = _doppler_get("DEPLOY_HOST")
    user = _doppler_get("DEPLOY_USER")
    key_material = _doppler_get("DEPLOY_SSH_KEY")

    key_path = _ssh_key_path()
    try:
        _write_ssh_key(key_path, key_material)
        if os.name == "nt":
            _lock_down_windows_key_acl(key_path)
        target = f"{user}@{host}"
        proc = subprocess.run(
            [
                "ssh",
                "-i",
                str(key_path),
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "ConnectTimeout=15",
                target,
                remote_command,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "host": host,
            "user": user,
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    finally:
        key_path.unlink(missing_ok=True)


def cmd_info(_: argparse.Namespace) -> None:
    _require_doppler()
    host = _doppler_get("DEPLOY_HOST")
    user = _doppler_get("DEPLOY_USER")
    _emit({
        "environment": os_environ_agent.environ.get("ENVIRONMENT"),
        "doppler_project": DOPPLER_PROJECT,
        "doppler_config": DOPPLER_CONFIG,
        "host": host,
        "user": user,
        "namespace": NAMESPACE,
        "manifests_path": "/opt/chess_teacher/k8s/",
        "warning": (
            "PRODUCTION Hetzner VPS. This skill runs READ-ONLY commands only. "
            "Never kubectl apply/delete, systemctl restart, or ad-hoc SSH shell."
        ),
    })


def cmd_ping(_: argparse.Namespace) -> None:
    result = _ssh_run("echo ok")
    _emit({**result, "ok": result["exit_code"] == 0 and result["stdout"].strip() == "ok"})


def cmd_cluster_info(_: argparse.Namespace) -> None:
    _emit(_ssh_run("kubectl cluster-info"))


def cmd_nodes(_: argparse.Namespace) -> None:
    _emit(_ssh_run("kubectl get nodes -o wide"))


def cmd_pods(_: argparse.Namespace) -> None:
    _emit(_ssh_run(f"kubectl get pods -n {NAMESPACE} -o wide"))


def cmd_deployments(_: argparse.Namespace) -> None:
    _emit(_ssh_run(f"kubectl get deployments -n {NAMESPACE} -o wide"))


def cmd_services(_: argparse.Namespace) -> None:
    _emit(_ssh_run(f"kubectl get svc -n {NAMESPACE} -o wide"))


def cmd_ingress(_: argparse.Namespace) -> None:
    _emit(_ssh_run(f"kubectl get ingress -n {NAMESPACE} -o wide"))


def cmd_events(args: argparse.Namespace) -> None:
    tail = max(1, min(args.tail, 200))
    _emit(_ssh_run(f"kubectl get events -n {NAMESPACE} --sort-by=.lastTimestamp | tail -n {tail}"))


def cmd_describe_pod(args: argparse.Namespace) -> None:
    pod = _validate_k8s_name(args.pod, label="pod name")
    _emit(_ssh_run(f"kubectl describe pod {pod} -n {NAMESPACE}"))


def cmd_logs(args: argparse.Namespace) -> None:
    pod = _validate_k8s_name(args.pod, label="pod name")
    tail = max(1, min(args.tail, 500))
    remote = f"kubectl logs {pod} -n {NAMESPACE} --tail={tail}"
    if args.container:
        container = _validate_k8s_name(args.container, label="container name")
        remote = f"kubectl logs {pod} -n {NAMESPACE} -c {container} --tail={tail}"
    _emit(_ssh_run(remote, timeout=180))


def cmd_rollout_status(_: argparse.Namespace) -> None:
    _emit(
        _ssh_run(
            f"kubectl rollout status deployment/streamlit -n {NAMESPACE} --timeout=10s",
            timeout=30,
        )
    )


def cmd_disk(_: argparse.Namespace) -> None:
    _emit(_ssh_run("df -h"))


def cmd_memory(_: argparse.Namespace) -> None:
    _emit(_ssh_run("free -m"))


def cmd_uptime(_: argparse.Namespace) -> None:
    _emit(_ssh_run("uptime"))


def _validate_domain_id(domain: str) -> str:
    text = domain.strip()
    if not text or len(text) > 128 or not DOMAIN_ID_RE.fullmatch(text):
        raise SystemExit(f"Invalid domain id: {domain!r}")
    return text


def _validate_table_key(table: str) -> str:
    text = table.strip()
    if not text or len(text) > 128 or not TABLE_KEY_RE.fullmatch(text):
        raise SystemExit(f"Invalid table key: {table!r}")
    return text


def _validate_sql_fragment(fragment: str, *, label: str) -> str:
    text = fragment.strip()
    if not text:
        raise SystemExit(f"{label} must not be empty.")
    if len(text) > 2000:
        raise SystemExit(f"{label} is too long.")
    if ";" in text:
        raise SystemExit(f"{label} must not contain ';'.")
    if _WRITE_KEYWORDS.search(text):
        raise SystemExit(f"{label} contains disallowed SQL keyword.")
    return text


def _validate_column_list(columns: str) -> str:
    text = columns.strip().replace(" ", "")
    if not text or not COLUMN_LIST_RE.fullmatch(text):
        raise SystemExit(f"Invalid --columns value: {columns!r}")
    return text


def _remote_db_argv(*db_args: str) -> str:
    """Build a fixed kubectl-exec remote command (POSIX-quoted for remote shell)."""
    argv = [
        "kubectl",
        "exec",
        "-n",
        NAMESPACE,
        f"deploy/{STREAMLIT_DEPLOYMENT}",
        "--",
        "python",
        DB_QUERY_SCRIPT,
        "--json",
        *db_args,
    ]
    return " ".join(shlex.quote(part) for part in argv)


def _emit_db_ssh(result: dict[str, Any], *, command: str) -> None:
    payload: dict[str, Any] = {
        "command": command,
        "via": "kubectl-exec-streamlit",
        "host": result.get("host"),
        "user": result.get("user"),
        "exit_code": result.get("exit_code"),
        "stderr": result.get("stderr"),
    }
    stdout = (result.get("stdout") or "").strip()
    if result.get("exit_code") == 0 and stdout:
        try:
            payload["result"] = json.loads(stdout)
        except json.JSONDecodeError:
            payload["stdout"] = result.get("stdout")
            payload["parse_error"] = "Remote stdout was not JSON"
    else:
        payload["stdout"] = result.get("stdout")
    _emit(payload)


def cmd_db_list_domains(_: argparse.Namespace) -> None:
    _emit_db_ssh(
        _ssh_run(_remote_db_argv("list-domains"), timeout=180),
        command="db-list-domains",
    )


def cmd_db_list_tables(args: argparse.Namespace) -> None:
    domain = _validate_domain_id(args.domain)
    _emit_db_ssh(
        _ssh_run(_remote_db_argv("list-tables", domain), timeout=180),
        command="db-list-tables",
    )


def cmd_db_count(args: argparse.Namespace) -> None:
    domain = _validate_domain_id(args.domain)
    table = _validate_table_key(args.table)
    db_args = ["count", domain, table]
    if args.where:
        db_args.extend(["--where", _validate_sql_fragment(args.where, label="WHERE")])
    _emit_db_ssh(
        _ssh_run(_remote_db_argv(*db_args), timeout=300),
        command="db-count",
    )


def cmd_db_read(args: argparse.Namespace) -> None:
    domain = _validate_domain_id(args.domain)
    table = _validate_table_key(args.table)
    limit = max(1, min(int(args.limit), 100))
    db_args = ["read", domain, table, "--limit", str(limit)]
    if args.where:
        db_args.extend(["--where", _validate_sql_fragment(args.where, label="WHERE")])
    if args.columns:
        db_args.extend(["--columns", _validate_column_list(args.columns)])
    if args.order_by:
        db_args.extend(["--order-by", _validate_sql_fragment(args.order_by, label="ORDER BY")])
    _emit_db_ssh(
        _ssh_run(_remote_db_argv(*db_args), timeout=300),
        command="db-read",
    )


def cmd_db_exists(args: argparse.Namespace) -> None:
    domain = _validate_domain_id(args.domain)
    table = _validate_table_key(args.table)
    where = _validate_sql_fragment(args.where, label="WHERE")
    _emit_db_ssh(
        _ssh_run(_remote_db_argv("exists", domain, table, "--where", where), timeout=300),
        command="db-exists",
    )


def cmd_db_schema(args: argparse.Namespace) -> None:
    domain = _validate_domain_id(args.domain)
    table = _validate_table_key(args.table)
    _emit_db_ssh(
        _ssh_run(_remote_db_argv("schema", domain, table), timeout=300),
        command="db-schema",
    )


def cmd_db_all_match(args: argparse.Namespace) -> None:
    domain = _validate_domain_id(args.domain)
    table = _validate_table_key(args.table)
    condition = _validate_sql_fragment(args.condition, label="CONDITION")
    sample_limit = max(0, min(int(args.sample_limit), 50))
    _emit_db_ssh(
        _ssh_run(
            _remote_db_argv(
                "all-match",
                domain,
                table,
                "--condition",
                condition,
                "--sample-limit",
                str(sample_limit),
            ),
            timeout=300,
        ),
        command="db-all-match",
    )


def cmd_db_unique(args: argparse.Namespace) -> None:
    domain = _validate_domain_id(args.domain)
    table = _validate_table_key(args.table)
    columns = _validate_column_list(args.columns)
    limit = max(1, min(int(args.limit), 50))
    _emit_db_ssh(
        _ssh_run(
            _remote_db_argv(
                "unique",
                domain,
                table,
                "--columns",
                columns,
                "--limit",
                str(limit),
            ),
            timeout=300,
        ),
        command="db-unique",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="PRODUCTION VPS — read-only whitelisted commands only.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON (always used).")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="SSH target + warnings (no SSH).")
    sub.add_parser("ping", help="SSH connectivity echo ok.")
    sub.add_parser("cluster-info", help="kubectl cluster-info.")
    sub.add_parser("nodes", help="kubectl get nodes -o wide.")
    sub.add_parser("pods", help="kubectl get pods in chess-teacher namespace.")
    sub.add_parser("deployments", help="kubectl get deployments.")
    sub.add_parser("services", help="kubectl get services.")
    sub.add_parser("ingress", help="kubectl get ingress.")
    sub.add_parser("rollout-status", help="Read-only rollout status (no restart).")

    events_p = sub.add_parser("events", help="Recent namespace events.")
    events_p.add_argument("--tail", type=int, default=30)

    desc_p = sub.add_parser("describe-pod", help="kubectl describe pod (read-only).")
    desc_p.add_argument("pod")

    logs_p = sub.add_parser("logs", help="kubectl logs --tail (read-only).")
    logs_p.add_argument("pod")
    logs_p.add_argument("--container", default=None)
    logs_p.add_argument("--tail", type=int, default=100)

    sub.add_parser("disk", help="df -h on VPS.")
    sub.add_parser("memory", help="free -m on VPS.")
    sub.add_parser("uptime", help="uptime on VPS.")

    sub.add_parser(
        "db-list-domains",
        help="Prod DB: list metadata domains via kubectl exec into streamlit.",
    )
    db_tables = sub.add_parser("db-list-tables", help="Prod DB: list tables in a domain.")
    db_tables.add_argument("domain")

    db_count = sub.add_parser("db-count", help="Prod DB: COUNT rows.")
    db_count.add_argument("domain")
    db_count.add_argument("table")
    db_count.add_argument("--where", default=None)

    db_read = sub.add_parser("db-read", help="Prod DB: sample rows (limit capped at 100).")
    db_read.add_argument("domain")
    db_read.add_argument("table")
    db_read.add_argument("--where", default=None)
    db_read.add_argument("--columns", default=None)
    db_read.add_argument("--order-by", default=None)
    db_read.add_argument("--limit", type=int, default=20)

    db_exists = sub.add_parser("db-exists", help="Prod DB: EXISTS check.")
    db_exists.add_argument("domain")
    db_exists.add_argument("table")
    db_exists.add_argument("--where", required=True)

    db_schema = sub.add_parser("db-schema", help="Prod DB: schema introspection.")
    db_schema.add_argument("domain")
    db_schema.add_argument("table")

    db_all = sub.add_parser("db-all-match", help="Prod DB: every row matches CONDITION?")
    db_all.add_argument("domain")
    db_all.add_argument("table")
    db_all.add_argument("--condition", required=True)
    db_all.add_argument("--sample-limit", type=int, default=10)

    db_unique = sub.add_parser("db-unique", help="Prod DB: uniqueness check.")
    db_unique.add_argument("domain")
    db_unique.add_argument("table")
    db_unique.add_argument("--columns", required=True)
    db_unique.add_argument("--limit", type=int, default=20)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    handlers = {
        "info": cmd_info,
        "ping": cmd_ping,
        "cluster-info": cmd_cluster_info,
        "nodes": cmd_nodes,
        "pods": cmd_pods,
        "deployments": cmd_deployments,
        "services": cmd_services,
        "ingress": cmd_ingress,
        "events": cmd_events,
        "describe-pod": cmd_describe_pod,
        "logs": cmd_logs,
        "rollout-status": cmd_rollout_status,
        "disk": cmd_disk,
        "memory": cmd_memory,
        "uptime": cmd_uptime,
        "db-list-domains": cmd_db_list_domains,
        "db-list-tables": cmd_db_list_tables,
        "db-count": cmd_db_count,
        "db-read": cmd_db_read,
        "db-exists": cmd_db_exists,
        "db-schema": cmd_db_schema,
        "db-all-match": cmd_db_all_match,
        "db-unique": cmd_db_unique,
    }
    try:
        handlers[args.command](args)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except subprocess.TimeoutExpired:
        _emit({"command": args.command, "error": "SSH command timed out"})
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
