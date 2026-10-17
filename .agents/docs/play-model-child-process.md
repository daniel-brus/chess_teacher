# Play model in a child process

**Status:** implementing. Review changed three rules: send the real board, use SIGKILL for parent death, and do not spawn a child to draw a finished game.

**Last updated:** 2026-10-17

## Problem

The Streamlit pod is the always-on process, and a neural game loads Keras and Stockfish inside it. On 17 Oct that process sat at 560–640MB RSS while choosing a move. The host has 3814MB. Training will not start unless MemAvailable is at least 1000MB, so those games keep the trainers in the 4 hour wait.

Deleting the model in the Streamlit process does not fix that. `tf_worker` already records why: the allocator belongs to the process, and exiting the process is what returns the memory to the host.

## Decision

Copy the training child process. A neural game runs `NeuralBaselineBot` in a fresh interpreter for the length of that game. Streamlit keeps a handle, the same way it already keeps the bot in `st.session_state`. When the game ends, the child exits and the OS takes the RSS back.

Stockfish-only and random opponents stay in the Streamlit process. They are not the 600MB.

No new deployment, port, or HTTP service.

## Why this shape

An always-on inference server is the usual split at a bigger size. On this one 4GB node it stays resident, so the training gate is still closed when nobody is playing. A child that exits is the pattern this repo already uses for the same library (`StepProcess` and `tf_worker`).

## What changes

| Piece | Where | Role |
|---|---|---|
| Worker | `src/chess_teacher/bots/play_worker.py` | `python -m chess_teacher.bots.play_worker`. Ops are `load` and `choose_move`. Speaks the existing `step_process.serve` pickle protocol. |
| Handle | `src/chess_teacher/bots/process_bot.py` | `ProcessNeuralBot(ChessBot)`. Parent-side. `choose_move` sends a FEN. `close` shuts the child down. |
| Factory | `bots/presets.py` `_neural_factory` | Returns `ProcessNeuralBot` instead of constructing `NeuralBaselineBot` in the parent. |

`NeuralBaselineBot` stays the in-process class. Tests and the worker call it. The Play page does not.

The parent already resolves `model_uri` from Postgres before it builds a bot. It passes that string, plus version, display name, and temperature, into `load`. It does not import `neural_baseline_bot` or TensorFlow.

`choose_move` receives the parent `chess.Board`, not a FEN. `NeuralBaselineBot` reads `board.move_stack` for capture and recapture flags. A board built from a FEN has an empty stack, so the child would play a different move. The reply is the UCI plus that bot’s `last_move_analysis`. The parent stores the dataclass and returns `chess.Move.from_uci`.

Progress lines from inside the child ("Starting Stockfish…", "Loading weights…") stay in the child log. The Play page already shows one status until `create_bot` returns, then "Opponent ready". No second channel.

## Lifetime

The Play page already calls `close_bot` when the opponent changes and when the user resets. `close` on the handle kills the child.

`_render_active_game` calls `_ensure_bot` on every run, including a finished game. Close and drop the handle when `game_needs_live_bot` is false, and do not start another child to draw that board. Analysis is already copied into session state before the rerun. A dead child during a live game does not match `_bot_matches_state`, so the next run starts a new one.

Logout goes through `force_logout`, which clears session state and does not call `close`. `close_stored_play_bot` runs first, in a `finally`, so a close error still logs the user out. That helper does not import the Play page.

The child watches its parent process pid and SIGKILLs itself if that pid changes. `prctl(PR_SET_PDEATHSIG)` is the wrong tool here: Linux delivers it when the creating thread exits, and Streamlit's script thread exits after every rerun, so the worker was killed between moves. SIGTERM is also wrong: importing the package installs a handler that only flushes logs. Do not use `Popen` `preexec_fn`. Streamlit is PID 1 in the pod, so a parent pid of 1 is still alive. A crash can leave the Stockfish grandchild; the normal `finally: bot.close()` path is what quits Stockfish.

An in-progress game whose child is gone (server restart, crashed worker) is the same as a missing bot today: `_ensure_bot` starts one. Model load is paid once per game per server process.

The play-session snapshot does not store the bot. Do not start putting the handle in it.

## What this does to memory

During a neural game the child is about the size Streamlit is today (around 600MB) and training still waits. After `close`, that RSS is MemAvailable again. Two neural games at once are two children. This box can run one of those and still be short of the training gate until they exit. There is no queue in this plan.

Streamlit itself no longer imports TensorFlow for play, so its RSS stays the UI process.

## Tests

- `ProcessNeuralBot` against a fake `StepProcess`: `choose_move` returns the UCI, `close` sends shutdown, `temperature` and `name` stay on the parent so `_bot_matches_state` keeps working.
- Worker ops with a fake `NeuralBaselineBot`: `load` then `choose_move(fen)` returns UCI plus analysis. No real weights in the unit test.
- Import check: building a `ProcessNeuralBot` with the process stubbed does not import `tensorflow`.
- Existing `NeuralBaselineBot` tests stay in-process.

## Out of scope

- Moving Stockfish-only bots out of Streamlit.
- A play inference Deployment, TF Serving, or a new port.
- Changing the training worker.
- Lowering `PIPELINE_MIN_MEM_AVAILABLE_MB` or resizing the node.
- Unloading Keras inside the Streamlit process. That does not return the memory.
