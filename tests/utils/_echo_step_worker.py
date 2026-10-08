"""Echo worker for StepProcess tests. Not used by the training pipeline."""

from __future__ import annotations

import sys


def main() -> None:
    protocol_out = sys.stdout.buffer
    sys.stdout = sys.stderr
    from chess_teacher.utils.pipeline_utils.step_process import serve

    def echo(*, value: str) -> str:
        print(f"echo sees {value}")
        return value

    def fail() -> None:
        raise RuntimeError("nope")

    serve({"echo": echo, "fail": fail}, protocol_out)


if __name__ == "__main__":
    main()
