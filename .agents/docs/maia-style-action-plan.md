# Maia-style action plan

Goal: a model that mimics the move a person played. Step 0 is an offline
architecture comparison. Later steps stay documented here and are not in this
branch.

The comparison reads registry **train** and **val** only. It does not write
`ml.baseline_models`, does not reset the training queue, and does not read the
test bucket. Do not launch it through `scripts/entrypoints/baseline_training.py`
or the catch-up ops.

## Step 0 — hybrid vs small spatial policy

Two cold starts, same games, same epochs, same batch size, same Adam learning
rate (`1e-3`), same style-disagree weights (boost 2, scale 2 pawns).

| Arm | What it is |
| --- | --- |
| `hybrid` | `HybridBoardTrainer`. Board conv, state tower, Stockfish candidate features, 128-way head. This is the control. |
| `spatial` | `SpatialPolicyTrainer`. 17 board planes only, six residual blocks at 64 filters, 8x8x73 policy, masked cross-entropy on the played move. No engine input. |

The spatial head is indexed in the same orientation as the board tensor
(`oriented_policy.py`). Black's board is mirrored, so a black pawn push is a
+rank step in that frame. Val scores are gathered back onto the same 128
candidate slots, then scored with the existing top-1, agree, disagree, and
pawn-gap metrics. `val_log_loss` is that 128-way cross-entropy, so the two
arms share a loss number. The spatial training loss is 4672-way and is not
the comparison.

Both arms are scored at the **last** epoch. Keeping the best epoch is a later
training-scheme change. A last-epoch score keeps this run to one change: the
head.

Smoke (a few thousand moves, discard the scores, keep the time):

```bash
.venv/bin/python scripts/tools/offline_maia_style_ab.py \
  --train-limit 2000 --val-limit 500 --epochs 2 \
  --output storage/tmp/maia_style_ab_smoke.json
```

On the machine with the full split (PowerShell):

```powershell
doppler run --project chess-teacher --config dev_local -- `
  .venv\Scripts\python.exe scripts/tools/offline_maia_style_ab.py `
  --full-buckets --epochs 2 --output storage/tmp/maia_style_ab_full.json
```

`--full-buckets` loads every train game and every val game. It still skips test.

A decision needs about **+2 top-1** or a clear drop in `val_log_loss`, with the
mean pawn gap no worse. A sub-point gap on a 2,000-move smoke is noise.

## Later, not this branch

1. **Update rule.** Adam peak near `1e-4`, short warmup (moments reset on
   load), 1:1 replay of old games, 1–3 epochs, keep the best val checkpoint.
   Distill the parent human policy. Do not distill Stockfish.
2. **Disagree boost.** Ablate boost 2 vs boost 1 before raising it. Promote on
   log-loss and top-1. Pawn gap is the safety check.
3. **Personal head.** Freeze the conv trunk and train the scorer. Pin each
   personal model to a named parent. Do not restart a good personal model from
   a new platform parent.
4. **Planes.** Colour plane, then one previous ply, only after the spatial head
   wins step 0. History beyond that is a later ablation.

## What this does not copy from Maia-2

Maia-2's setup is the part to copy: current position, spatial policy, human
move cross-entropy, no engine features, no search. The 12x256 tower, skill
attention, and value head stay out of step 0.
