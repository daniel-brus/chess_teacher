# Agent plan: baseline move-policy head

Handoff for the next agent. Goal: replace the stub MSE `action_label` regression head with a **fixed-vocabulary move policy** so models output scores over moves, train with masked cross-entropy, promote on top-k accuracy, and eventually play on Streamlit.

## Locked decisions (do not reopen unless user asks)

1. **Head type:** fixed move vocabulary (AlphaZero-style). Always emit logits of size `V`. Mask illegal moves to `-inf` before softmax / loss.
2. **Not chosen:** variable-length legal-move output dim; nearest-legal decode on MSE coords (band-aid only if needed for demos — prefer real policy).
3. **Live eval for play (Phase 3):** compute Stockfish evaluation live when encoding a board; slower bot think is OK.
4. **Play picker (Phase 3):** list **all** baseline model statuses (production, candidate, archived).
5. **Existing MSE models:** incompatible with policy head. Do **not** resume MSE weights as parent. Fresh candidate chain (new cold-start `v1` or new MLflow experiment / `head_type` metadata). Archive or ignore old MSE rows.
6. **Swappable classes:** mirror promotion pattern (`EvalSetProvider` / `ModelScorer` / `PromotionPolicy`). Introduce clear ABCs so trainer/encoder/scorer can be swapped later without rewriting pipeline steps.
7. **Python env:** never use `uv`. Use project `.venv`. Do **not** run pytest/mypy — ask user to run those.
8. **Commits:** only when user asks.
9. **Communication:** workspace caveman skill unless user says stop/normal mode.

## Current state (what exists today)

| Piece | Location | Notes |
|-------|----------|--------|
| State features `X` | `create_training_set.py` → `TrainingDatum.state_vector()` | Keep. Keys in `_STATE_FEATURE_KEYS`. |
| Stub target `Y` | `TrainingDatum.action_label()` | 4 coords + piece one-hot; **replace as train target**. |
| Legal UCIs on datum | `TrainingDatum.legal_move_ucis`, `move_uci` | Already populated via `TrainingDatumBuilder.derive_move_identity`. |
| Trainer | `train.py` → `BaselineTrainer` | MLP, linear out, MSE/MAE. Stub. |
| Train pipeline | `pipeline_steps.py`, `main.py` → `run_baseline_training_pipeline` | Incremental, batch cap `MAX_MOVES_PER_BASELINE_BATCH=10_000`, gate `MIN_NEW_MOVES_BASELINE=1000`. |
| Promotion | `promotion.py`, `promotion_steps.py` | Default `ActionMaeScorer` (lower MAE better). Swap default to top-k. |
| MLflow / S3 | `mlflow_utils.py` | Postgres tracking + same-bucket artifacts under `{STORAGE_ROOT}/mlflow`. |
| Schema | `models.py`, `metadata.yml` | `ml.baseline_models`, `ml.training_state`. |
| Play UI | `streamlit_pages/play.py`, `presets.py` | Static Random/Stockfish only. No neural bot yet. |
| Notebook loop | `testnotebook.ipynb` | Train→promote until no promote; useful for smoke after Phase 1–2. |

**User mental model (correct product goal):** net ≈ probability distribution over legal moves. Current MSE head does **not** do that — it was infra scaffolding only.

## Architecture target

```
fen / TrainingDatum
    → state_vector (unchanged)
    → BaselineTrainer MLP → logits[V]
    → legal_mask[V] (from board / legal_move_ucis)
    → masked softmax / masked sparse CE on played move index
    → at inference: argmax or sample among legal
```

Suggested new modules / types:

- `move_encoding.py` — `MoveEncoder` (UCI ↔ index, vocab build, `legal_mask(board)`).
- Policy target helpers on `TrainingDatum` / `TrainingBatch` — `(class_index, legal_mask)`.
- Keep `BaselineTrainer` but change build/fit/loss; or introduce `PolicyBaselineTrainer` and switch the step to use it (prefer one trainer class with clear docstring that MSE path is gone).
- `MovePolicyHead` or bot helper — logits + board → chosen `chess.Move`.
- `TopKMoveAccuracyScorer(ModelScorer)` — replace default in `PromotionStrategies`.

## Phase 1 — Policy training (no Streamlit yet)

**Done when:** `run_baseline_training_pipeline()` cold-starts a policy model, logs to MLflow, writes `BaselineModel` candidate with policy weights; parent load only works for same-arch policy weights (skip/refuse MSE parents).

### Tasks

1. **Implement fixed vocab encoder** (`move_encoding.py`)
   - Stable index space covering all practical UCIs (from-to + underpromotions). Document size `V`.
   - `encode(uci) -> int`, `decode(i) -> chess.Move` / UCI string.
   - `mask_from_board(board) -> np.bool_[V]` and/or `mask_from_ucis(legal_ucis)`.
   - Unit-testable pure functions; ask user to run tests.

2. **Targets on training data**
   - Add `policy_target()` / batch matrices: `y_index` shape `(N,)`, `legal_mask` shape `(N, V)`.
   - Drop using `action_matrix()` in the trainer path (may leave `action_label` for now as unused legacy).

3. **Rewrite `BaselineTrainer`**
   - Output dim = `V`, activation linear (logits).
   - Loss: sparse categorical CE with illegal logits masked (custom loss or equivalent).
   - Metric: masked top-1 accuracy (and optionally top-5).
   - `load_or_build`: if loading old MSE artifact (wrong output dim / missing metadata), cold-start instead of crash — log clearly.
   - Persist enough metadata (e.g. MLflow param `head=policy`, `vocab_size=V`) for later gating.

4. **Pipeline parent continuity**
   - Only resume weights when parent is policy-compatible; otherwise cold-start and log.
   - Consider storing `head_type` on `BaselineModel` if schema change is cheap (optional; else infer from MLflow params / output shape).

5. **Smoke**
   - Notebook or script: one train batch succeeds; metrics include accuracy-like values, not only MAE.

### Out of scope for Phase 1

- Streamlit picker / live play
- Changing promotion default (can soft-land in Phase 2)
- User finetune pipeline

## Phase 2 — Promotion metrics

**Done when:** `run_baseline_promotion_pipeline()` compares candidate vs production by **top-k move accuracy** on eval set (higher better); auto-promote still if no production.

### Tasks

1. Implement `TopKMoveAccuracyScorer(ModelScorer)` using same mask+argmax (or top-k hit) on `TrainingDatum` eval batch.
2. Change `PromotionStrategies` default `scorer` from `ActionMaeScorer` to top-k scorer; keep `ActionMaeScorer` available for legacy.
3. Ensure `BetterOrEqualPromotionPolicy` gets `higher_is_better=True` scores.
4. Smoke with train→promote notebook loop; expect meaningful promote/reject on accuracy.

## Phase 3 — Playable bots on Streamlit

**Done when:** Play page opponent dropdown includes baseline models (all statuses); selected model loads weights and moves via masked policy.

### Tasks

1. **Live state encoder** — `chess.Board` (+ optional last opponent move) → same `state_vector` as training.
   - Recompute features via `fen_metrics` where possible.
   - **Live Stockfish eval** for `evaluation_before_user_pov` (user decision).
   - Side-to-move = “user” POV for remap conventions used in training.
2. **`NeuralBaselineBot(ChessBot)`** — download via `MLflowTracker.download_keras_weights(model_uri)`, predict, mask legal, argmax/sample; `close()` releases resources.
3. **Presets / play wiring** — dynamic options from `BaselineModel.fetch_*`; keys like `baseline:{version}`; `create_bot` / `play.py` setup form merge with existing `BOT_PRESETS`.
4. Do not block Play on MSE models; only list policy-compatible rows if distinguishable.

## Non-goals / pitfalls

- Do not use `uv`.
- Do not expand scope to user-finetune or held-out rotation unless asked (held-out still `RandomEvalSetProvider` shortcut).
- Do not force-push / commit unless asked.
- Illegal moves must never be chosen at inference (mask is mandatory).
- Batching: fixed `V` keeps Keras `fit` simple; avoid ragged legal-index labels.
- First policy train after deploy will not continue from MSE parent — expect cold start; cutoff/`training_state` behavior should remain incremental on **data**, even if weights reset.

## Suggested implementation order

1. `move_encoding.py` + small sanity checks
2. Policy targets on `TrainingDatum` / `TrainingBatch`
3. `BaselineTrainer` policy loss/metrics
4. Parent-load gating in `pipeline_steps.py`
5. `TopKMoveAccuracyScorer` + default promotion swap
6. Live encoder + `NeuralBaselineBot` + Streamlit picker

## Acceptance checklist

- [ ] Train produces candidate with output shape `(V,)` logits, not 11-d action vector
- [ ] Illegal moves never win argmax after mask
- [ ] Promotion uses accuracy-style metric (higher better)
- [ ] Old MSE weights not silently loaded as parent
- [ ] (Phase 3) Play can select a baseline and complete a game vs it

## Key file paths

```
src/chess_teacher/pipelines/neural_network/
  create_training_set.py   # state_vector, legal_move_ucis, action_label (legacy)
  train.py                 # BaselineTrainer stub → policy
  pipeline_steps.py        # train pipeline
  promotion.py             # scorers / strategies
  promotion_steps.py
  main.py
  mlflow_utils.py
  models.py
  metadata.yml
  move_encoding.py         # NEW

src/chess_teacher/utils/chess_bots/
  base.py, presets.py      # Phase 3 bot + picker

streamlit_pages/play.py
streamlit_utils/play_game.py
scripts/baseline_training.py
scripts/baseline_promotion.py
testnotebook.ipynb
```

## Open optional niceties (ask user before large extra work)

- Persist `head_type` / `vocab_version` on `ml.baseline_models`
- Temperature sampling vs greedy at play
- Top-5 logging in MLflow
- Reset `training_state` cutoff vs keep cutoff when weight chain resets
