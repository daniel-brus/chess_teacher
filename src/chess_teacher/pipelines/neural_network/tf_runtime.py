"""Wire TensorFlow Python logs into app logging; quiet native C++ STDERR.

Python ``tensorflow`` / ``absl`` loggers propagate to the root handlers
(console + JSON segments) at INFO+. Native oneDNN / cpu_feature_guard lines
bypass Python logging — those stay suppressed via ``TF_CPP_MIN_LOG_LEVEL``.
"""

from __future__ import annotations

import logging
import os

_env_configured = False


def ensure_tensorflow_logging() -> None:
    """Idempotent env floor + (re)wire Python TF/absl loggers into app root.

    Call before first ``import tensorflow``, and again right after import
    (TF sometimes attaches its own STDERR handlers).
    """
    global _env_configured
    if not _env_configured:
        # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR — C++ only.
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        _env_configured = True

    # Root handlers (console formatter + segment files).
    from chess_teacher.utils.logging import configure_logging

    configure_logging()
    _wire_python_loggers()


def _wire_python_loggers() -> None:
    """Route TF/absl through root; drop their duplicate STDERR handlers."""
    for name in ("tensorflow", "absl"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.propagate = True
        lg.handlers.clear()


# Back-compat alias used by early call sites.
ensure_tensorflow_quiet = ensure_tensorflow_logging
