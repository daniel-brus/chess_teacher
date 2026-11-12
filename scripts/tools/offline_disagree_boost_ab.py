"""Thin CLI: offline style-disagree boost A/B (read-only).

On hosts where ``127.0.0.1:5432`` is shadowed, set ``CT_POSTGRES_HOST_OVERRIDE``
to a non-loopback IPv4 that reaches Docker-published Postgres.

Mid decision::

    doppler run --project chess-teacher --config dev_local -- `
      .venv\\Scripts\\python.exe scripts\\tools\\offline_disagree_boost_ab.py `
      --train-limit 30000 --val-limit 8000 --epochs 3 `
      --output storage\\tmp\\disagree_boost_ab_mid.json
"""

from __future__ import annotations

import os
import socket

os.environ.setdefault("ENVIRONMENT", "AGENT")


def _maybe_override_postgres_host() -> None:
    if os.environ.get("CT_POSTGRES_HOST_OVERRIDE"):
        os.environ["POSTGRES_HOST"] = os.environ["CT_POSTGRES_HOST_OVERRIDE"]
        return
    # Prefer a non-loopback IPv4 when localhost:5432 is stolen by wslrelay.
    if os.environ.get("POSTGRES_HOST", "localhost") not in {"localhost", "127.0.0.1"}:
        return
    for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
        if not ip.startswith("127."):
            os.environ["POSTGRES_HOST"] = ip
            return


_maybe_override_postgres_host()

from chess_teacher.pipelines.neural_network.offline_disagree_boost_ab import (  # noqa: E402
    main,
)
from chess_teacher.pipelines.neural_network.tf_runtime import (  # noqa: E402
    ensure_tensorflow_logging,
)
from chess_teacher.utils.logging import get_logger  # noqa: E402
from chess_teacher.utils.process_utils import (  # noqa: E402
    log_script_runtime_context,
    run_script_main,
)

ensure_tensorflow_logging()
logger = get_logger()


if __name__ == "__main__":
    log_script_runtime_context(logger, script="offline_disagree_boost_ab")
    run_script_main(main)
