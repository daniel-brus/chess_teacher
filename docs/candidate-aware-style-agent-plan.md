# Agent plan: candidate-aware style policy (eval-delta conditioned)

Handoff for the next agent. Goal: move beyond pure state→move behavioral cloning toward a **candidate-aware** scorer that sees **Stockfish eval delta (or relative quality) per legal move**, so middlegame play can mimic the user’s typical “suboptimal but not blundering” quality band (e.g. often ~−1 vs best) instead of only opening-book patterns.

Related prior work (done): fixed-vocab policy head, promotion top-k, live encoder, Streamlit category picker. See `docs/baseline-policy-head-agent-plan.md` for that stack. **Do not reopen** policy-vocab / mask / caveman / no-uv / no-pytest-unless-asked / commit-only-on-request rules unless user asks.

## Problem statement (user intent)

- Openings look good: imitation ≈ opening book; thick repeated data.
- Later game goes weird: sparse unique positions; pure `P(move | state)` invents legal-but-odd moves.
- Hypothesis: **eval quality of the chosen move** (delta vs best / vs alternatives) should dominate style. If the user often plays ~−1 eval delta, the bot should prefer candidates in that band over “looks like a piece pattern I play in openings.”
- Pure policy with only **position** eval in the state does **not** see per-candidate quality → cannot directly learn that preference.

## Locked decisions (do not reopen unless user asks)

1. **Primary architecture target:** candidate-aware scoring — for each (or top‑K) legal move, model sees shared state embedding **plus move features that include SF-derived quality** (at least eval after move and/or delta vs best in position). Train so the **user’s played move ranks best** among candidates (listwise CE or pairwise ranking).
2. **Stockfish is required for new labels:** DB today only has `evaluation_before/after/delta` for the **played** move. Per-legal-move (or top‑K) scores are **not** stored → need a **new SF pass** (offline backfill + live at play). Do **not** pretend MultiPV-only equals full candidate set (MultiPV = SF’s best lines, not “all moves I might play at −1”).
3. **Recommend Phase 0 first** (inference-only filter) to validate the hypothesis cheaply before burning CPU on full backfill. If user says skip Phase 0, go straight to Phase 1.
4. **Keep fixed vocab / legal mask** as infrastructure: either (a) score only candidates then map to UCI, or (b) keep logits[V] but **gate/reweight** by candidate features. Prefer (a) clear scorer over hacking the existing MLP blindly.
5. **Compatibility:** new head/arch ≠ current policy MLP. Treat as new model family (`head=candidate_style` or similar). Do **not** silently resume old policy weights. Cold-start new candidate chain; archive/ignore incompatible parents.
6. **Python env:** never `uv`. Use `.venv`. Do **not** run pytest/mypy — ask user.
7. **Commits:** only when user asks.
8. **Communication:** workspace caveman skill unless user says stop/normal mode.
9. **Play UI:** keep category selector (Stockfish / Baseline / Personal / Other). Baseline list stays **production + archived** only (no never-promoted candidates).

## Current state (relevant facts)

| Piece | Location | Notes |
|-------|----------|--------|
| Policy MLP | `train.py` `BaselineTrainer` | `state → logits[V]`, masked sparse CE. No per-move SF features. |
| State vector | `create_training_set.py`, `live_state.py` | Includes `evaluation_before_user_pov` (position only). Shared `assemble_state_vector`. |
| Played-move SF | `move_characteristics` / metadata | `evaluation_before`, `evaluation_after`, `evaluation_delta` for **chosen** move only. |
| Legal UCIs | `TrainingDatum.legal_move_ucis` | From FEN at datum build; **no scores**. |
| Live bot | `neural_baseline_bot.py` | Live SF for **position** eval via `LiveStateEncoder`; then policy argmax. |
| SF depth (preprocess) | `StockfishEvaluationTransformation` | Typically depth 20 on FENs for played-move path. |
| Promotion | `TopKMoveAccuracyScorer` | User-move top-k imitation; still valid as a metric, may need a second style-aware metric later. |

## Architecture target

```
Position (FEN / TrainingDatum)
    → state_encoder(state_vector) → h_state
    → for each candidate move m in C(position):
          feats_m = [move_geometry?, sf_eval_after, sf_delta_vs_best, rank_vs_best, …]
          score_m = scorer(h_state, feats_m)   # or MLP on concat(h_state, feats_m)
    → train: softmax CE over candidates with target = user move
             (or pairwise: user > other)
    → play: build C via SF (all legal shallow, or top‑K hybrid), score, argmax/sample
```

**Candidate set `C` (locked default for v1):**

- **Train:** all legal moves, each evaluated at a **configurable shallow–medium depth** (start `depth=8` or `10`; make constant/env-tunable). Store results.
- **Play:** same policy as train (all legal @ same depth), unless latency forces top‑K; if top‑K, document mismatch risk and keep train/play aligned.

**Minimum move features for v1:**

- `delta_vs_best` (user POV pawns): `eval_after(m) - eval_after(best)` or `eval_after(m) - eval_before` consistent with existing delta convention — **pick one convention, document, match train/live**.
- `eval_after_user_pov` (optional but useful).
- Optional cheap geometry: from/to/piece one-hots already available via UCI + board (no SF).

**Out of v1:** full MultiPV-only training set; learning SF from scratch without SF features.

## Phase 0 — Inference filter (optional, validate hypothesis)

**Done when:** Play vs baseline can optionally restrict policy picks to moves whose live SF delta lies in a configurable band (e.g. `[d_min, d_max]` or percentile of a hard-coded / user-derived band), without retraining.

### Tasks

1. In `NeuralBaselineBot` (or wrapper), after policy logits: for each legal move, shallow SF eval → delta vs best; mask out moves outside band; argmax among survivors (fallback: relax band if empty).
2. Expose toggle/params (constructor kwargs / Streamlit advanced controls optional — ask user before Play UI clutter).
3. Manual smoke: middlegame positions that previously looked random should look “weaker but coherent.”

### Out of scope for Phase 0

- DB backfill, new trainer, promotion changes.

**If Phase 0 fails to improve feel:** still proceed to Phase 1–3 only if user insists; report that quality-band alone may not be enough without retraining.

## Phase 1 — Candidate SF data pipeline

**Done when:** for training positions, system can load a list of `(uci, eval_after, delta_vs_best)` for legal moves (or fail soft with empty and skip datum); backfill job can process a batch and persist results.

### Tasks

1. **Storage design** (prefer one; ask user only if schema is ambiguous):
   - **Option A (preferred):** new table / JSON column on characteristics, e.g. `candidate_evaluations` JSONB `{uci: {eval_after, delta_vs_best}, …}` keyed by `move_id` (position = before user move).
   - **Option B:** side table `ml.move_candidate_evals (move_id, uci, eval_after, delta_vs_best, depth, engine_id)`.
   - Record `depth` + engine version/path hash in row or job metadata for reproducibility.
2. **Computation:** given `fen_before` + legal UCIs: evaluate each after-move FEN (push/pop or fen after move); compute best; store deltas. Reuse `StockfishEngine` patterns from preprocess.
3. **Job / transformation:** batch worker with progress logs, resumable (skip move_ids already filled), configurable depth + worker count. Mirror existing fen-characteristic pooling if practical.
4. **TrainingDatum hook:** `candidate_evals: dict[str, …] | None` or fetch-on-demand in store; training skips / warns when missing.
5. **Cost controls:** start with depth low; optional `LIMIT` moves for first backfill; log ETA.

### Stockfish answers (explicit)

- **Must re-run SF** for candidates — existing played-move evals are insufficient.
- **Do not** re-derive historical `evaluation_*` of the played move unless validating consistency.
- Live play will run SF again for candidates (no requirement that live depth == train depth, but **document** and prefer match for v1).

## Phase 2 — Model + training

**Done when:** trainer fits a candidate-aware model on datums that have candidate evals; MLflow logs `head=candidate_style` (or similar); `BaselineModel` candidate row written; incompatible parents cold-start.

### Tasks

1. **Batch construction:** for each datum, tensor of shape `(N_cand, F)` move feats + shared state vector; label = index of user UCI within candidates (assert user move ∈ C; if missing SF for user move, recompute or skip).
2. **Model:** shared state tower + scorer head (keep small MLP; no need for transformer in v1).
3. **Loss:** listwise softmax CE over candidates (preferred) or pairwise logistic. Mask nothing if C is exactly legal set; if top‑K and user move missing, **must include user move in C** (forced add).
4. **Metrics:** candidate top-1 / top-3 accuracy; optional calibration of predicted rank vs SF rank.
5. **Wire** `BaselineTrainer` variant or new `CandidateStyleTrainer`; switch train step behind clear flag / head type so old policy path can remain for comparison until user drops it.
6. **Parent load gating:** only resume same head + feature schema version.

### Out of scope for Phase 2

- User finetune; held-out set redesign (keep `RandomEvalSetProvider` unless asked).

## Phase 3 — Promotion + play

**Done when:** promotion can score candidate-style models; Play can load and move with candidate SF + scorer.

### Tasks

1. **Scorer:** either extend `TopKMoveAccuracyScorer` to load candidate-style models **or** add `CandidateStyleTopKScorer`. Eval path must build candidate feats with SF (slow but OK for promotion batch) **or** use precomputed candidate evals on eval datums (prefer precomputed for speed).
2. **Promotion policy:** keep better-or-equal on primary metric; log SF cost.
3. **`NeuralBaselineBot` (or subclass):** live candidate SF + model scores; release engines in `close()`.
4. **Presets:** only list models that `looks_like_candidate_style` / metadata head flag (mirror policy `looks_like_policy` pattern).
5. Smoke: one game on Play; resign/new game still cleans bots.

## Phase 4 — Soften / productize (optional, ask first)

- Fit user’s delta distribution from DB (histogram) → adaptive band or feature “delta_zscore_vs_user_prior”.
- Distill SF deltas into a cheap net to avoid live SF (later).
- Middlegame upweight in loss.
- Align preprocess depth 20 played-move evals with shallower candidate depth (document bias).

## Non-goals / pitfalls

- Do not use `uv`; do not run pytest/mypy yourself.
- Do not commit unless asked.
- Do not show never-promoted baseline candidates in Play.
- Do not train candidate-aware without user move in the candidate list.
- Do not assume MultiPV top‑3 captures user style.
- Watch latency: all-legal depth 20 at play is likely unusable — default shallow depth.
- Illegal moves must never be chosen.
- Huge backfill: implement resume + progress; do not block Play UI on unfinished backfill.

## Suggested implementation order

1. Phase 0 inference filter (unless user skips) — fastest learning signal
2. Schema + SF candidate eval job (Phase 1) on a **small** move sample
3. Datum plumbing + one offline train script/notebook smoke
4. Trainer + pipeline head switch (Phase 2)
5. Bot + promotion + Play listing (Phase 3)
6. Full backfill + retune depth

## Acceptance checklist

- [ ] (Phase 0) Optional band filter changes middlegame move choices vs pure policy
- [ ] (Phase 1) Persisted candidate evals for sample `move_id`s; resumable job
- [ ] (Phase 2) Train run logs candidate top-1; MLflow `head` distinguishes from plain policy
- [ ] User move always in training candidate set
- [ ] Old policy/MSE weights not silently loaded as parent
- [ ] (Phase 3) Play completes a game vs candidate-style bot; `close()` frees SF
- [ ] Train/play delta convention documented in module docstring

## Key file paths

```
docs/
  baseline-policy-head-agent-plan.md          # prior (policy) plan
  candidate-aware-style-agent-plan.md         # THIS file

src/chess_teacher/pipelines/neural_network/
  create_training_set.py   # TrainingDatum; extend with candidate evals
  live_state.py            # position state only today
  train.py                 # add CandidateStyleTrainer or head switch
  move_encoding.py         # keep for UCI index / masks if still used
  promotion.py / promotion_steps.py
  pipeline_steps.py / main.py / models.py / metadata.yml
  mlflow_utils.py

src/chess_teacher/pipelines/preprocessing/
  move_characteristics/stockfish_evaluation.py   # patterns to reuse
  fen_characteristic.py

src/chess_teacher/utils/chess_bots/
  neural_baseline_bot.py   # Phase 0 filter + Phase 3 candidate scorer
  presets.py

streamlit_pages/play.py
scripts/                   # optional: backfill_candidate_evals.py
testnotebook.ipynb         # smoke train/promote loop
```

## Open questions (ask user before assuming)

1. Skip Phase 0 and go straight to backfill?
2. Storage Option A (JSONB) vs B (side table)?
3. Train/play candidate depth (suggest 8–10 v1)?
4. Replace plain policy training entirely vs dual-head flag for a while?
5. Full-history backfill now vs sample (e.g. last N games / non-opening plies only)?

## One-sentence success

Bot middlegame choices look like **the user’s quality band** (similar SF deltas), not like a broken opening book — measured by candidate top-1 on held data and by feel in Play, with SF candidate features available end-to-end.
