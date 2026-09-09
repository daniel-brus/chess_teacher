# ML train queue

**Last updated:** 2026-09-05 (rev: personal queue increment)

**Audience:** humans and coding agents working on `src/chess_teacher/pipelines/neural_network/`

Daily platform train is **one round**. Catch-up is only a manual/ops **loop of that same round** (optional promote). Do not design a second algorithm called catch-up.

Work queue on the split registry. Not watermarks. Not ingest snapshots. Not hash-scan cursors.

**Production** (baseline) = the latest **promoted** platform model. One official trunk.

Baseline and personal are the **same pipeline object** (two schedules). They do not read each other's processed flags. The only coupling: a baseline promote **resets personal training schemes**.

---

## Intent

- Split registry (`ml.game_split_assignments`) already has train/val/test per `game_id`.
- Add two soft flags on that table. Never hard-delete split rows.
- A round takes the next unprocessed **train** games, in `game_id` order, up to the move cap, trains, then marks those `game_id`s.
- Never mark or consume val/test. Val stays the same games every eval (val bucket is never processed).
- Personal bot is **user-level** (all linked accounts). `game_id` is unique per account. One personal flag is enough: one current personal train chain per user.
- Different users do not share rows.
- Order is `game_id ASC`. Not random. Not oldest `end_time`.

---

## Columns

`ml.game_split_assignments` keeps `split_version`, `game_id`, `bucket`, `assigned_at`.

Add:

| Column | Type | Meaning |
|--------|------|---------|
| `already_processed_baseline` | `timestamptz` NULL | NULL = not processed for the platform baseline queue. Set to `now()` after a successful baseline train that used this game. |
| `already_processed_personal` | `timestamptz` NULL | NULL = not processed for this user's current personal train chain. Set to `now()` after a successful personal train that used this game. |

Soft flags. Bool also fine (`false` / NULL = not processed). Prefer timestamptz so a mark records when it happened. Neither column is a watermark or a fetch bound.

`ensure_metadata` **ADD COLUMN** on existing DBs. Default NULL = fresh queue (every train game is unprocessed).

Useful index:

```text
(split_version, bucket, already_processed_baseline)
```

Personal fetch scopes by this user's linked-account `game_id`s; the baseline index is the hot one.

---

## One round

Same function for daily train and for catch-up.

1. Count unprocessed **train** moves (`bucket = 'train'` and flag NULL). If count `< MIN_NEW_MOVES_BASELINE` (1000), skip.
2. Fetch next games: `bucket = 'train'` AND flag NULL, `ORDER BY game_id ASC`, take complete games until moves reach `MAX_MOVES_PER_BASELINE_BATCH` (~10000). Do not split a game across rounds.
3. Train (finetune parent, or cold-start if no parent).
4. **Only after successful fit:** set the flag on those `game_id`s. Failed fit must not eat the queue. Skip must not mark.

Mark only `game_id`s that actually produced training datums. A selected game that hydrates to 0 datums stays unprocessed.

Offline catch-up also waits until the keras checkpoint is saved and val is scored, then marks. Crash before that leaves the queue unconsumed.

Never select val or test. Never set either flag on val/test rows.

Baseline uses `already_processed_baseline` and all users' train rows.

Personal uses `already_processed_personal` and only that user's linked-account train rows. One current personal train chain per user.

**Reset (baseline promote):** clear `already_processed_personal` for that user's `game_id`s and replay the train pile from the newly promoted baseline. The last personal **production** row stays the serving bot; it is no longer a train parent. One current personal **training** chain per user.

---

## Same pipeline object

Ingest / Stockfish / characteristics stay a different domain. Split assignment is training-scheme, not game processing: **step 1** of this object (ensure buckets for the games this instance cares about). If the queue is too small, still assign; skip fit / mark / promote.

Steps (baseline and personal):

1. Ensure split assignments for this instance's `game_id`s (buckets only; never processed flags).
2. Skip if unprocessed **train** moves `< MIN_NEW_MOVES_BASELINE`.
3. Load parent keras.
4. Fetch next complete train games (~10k moves, `game_id` order, this instance's flag NULL).
5. Fit.
6. Log a **candidate of this instance** (MLflow / model table).
7. Mark this instance's processed flag only after save/log succeeded.
8. Maybe promote this instance's candidate over this instance's production.

Plugs that differ:

| Plug | Baseline | Personal |
|------|----------|----------|
| Scope / filter | all registry train | this user's linked-account train |
| Flag | `already_processed_baseline` | `already_processed_personal` |
| Parent keras | last baseline candidate, else cold-start | last personal keras of **this** scheme, else the promoted baseline that started the scheme |
| Promote table | `ml.baseline_models` | personal model rows |
| Val | registry val bucket | union of that user's val games |
| On successful promote | reset **personal training** schemes (all users) | none (does not touch baseline) |

Orchestrate as two schedules of this object. Do not put personal rounds inside the baseline job. Baseline `ApplyPromotion` fans out the personal-training reset.

Personal **parent keras** looks at baseline only when this scheme has no personal keras yet (first-time user, or just reset). After personal round 1, resume is last personal candidate.

### `baseline_parent_id`

Every personal candidate / production row stores `baseline_parent_id` (the promoted baseline that **started this scheme**). Use it for baseline-disagree **weights** and **eval**. Do not re-query live production each personal round.

With promote-and-reset as one event, that id equals "latest promoted baseline" **at scheme start**, and stays fixed until the next reset. Still store it: serving `U_B` can lag while training is already on `B′`; eval/compare should group by parent.

The live personal production bot is not compared as a **training** peer to the new scheme. The id is still useful for "this serving bot was built on `B`".

---

## Promotion

Score on **frozen registry val** (personal: that user's val). Never test. Never a random sample.

Do not promote on a tie or a tiny bump (today's `BetterOrEqual` / margin 0 is too eager). Personal lineages hang off a promoted baseline, so churn is expensive.

Two numbers on the same val:

| Role | Metric | Rule |
|------|--------|------|
| Primary | `top1_sf_disagree` | Candidate must beat production by a **fixed margin** (not `>=`) |
| Guardrail | `top1_sf_agree` (and/or overall top1) | Must not drop more than ε |

First production may auto-promote. Pick δ / ε from a real frozen-val curve, not before. Optional: do not run promote every catch-up round (cooldown). Rollback = previous production row still in the model table.

A "no" on promote must not un-mark the queue. A failed fit must not create a candidate.

**Val pack vs a val feature store:** the ~30 min you see after each *experiment* round is Python walking FEN geometry for ~45k val moves × ~35 candidates. Train does not need that. Production promote does, **once per promote job** (pack in-process, score candidate and production on the same tensors). That is already enough if promote is infrequent.

A durable `prepare_val_store` step is **optional later**, not a gate. It pays off only if many processes repeatedly score the same frozen val (daily promote, offline curves, A/B). The write is the same 30 min walk; you save it on later reads. Risk: store packing must match live `pack_candidate_tensors` or promotion scores lie. Feat bump (`CANDIDATE_MOVE_FEAT_VERSION`) rebuilds the store.

If built: one row per candidate, PK `(move_id, uci)` plus `feat_version`. Store only real legals, not 128 padded slots. Pad/sort at read time like today. Column for the vector: Postgres `real[]` (one array of 55 four-byte floats) — not BYTEA, not 55 named columns. The 55 dims are a versioned dense vector, not 55 stable facts; named columns mean `metadata.yml` + ALTER on every feat bump. `real[]` = “list of floats in one cell”; check `len == MOVE_FEAT_DIM` in code. SF evals stay in `candidate_evaluations`.

---

## What goes away vs what stays

### Removed as train loaders (code gone)

- Hash-scan cursor (`hash_scan.py`, `HashScanCursor`, `fetch_hash_scan_batch`, `seen_late`, epoch roll, `hash_scan_cursor.json`)
- `TrainingDataStore.count_since` and module wrappers `count_new_moves_since` / `fetch_training_data_since`
- Account hash-scan walk (`load_account_holdout`, `fetch_account_game_ingested_at`)
- `TrainingState.with_cutoff` / `ml.training_state.last_trained_data_cutoff` (dropped; progress is `already_processed_*`)

`LoadNewDataStep` / `CheckSufficientNewDataStep` use the registry queue (`count_unprocessed_train` / `fetch_unprocessed_train_batch`). Oldest-first `--limit` is not the production incremental story.

`load_registry_val_datums(..., full=True)` loads all registry val games. A val **subset** is the lowest-``game_id`` complete-game prefix (`--limit`), not oldest ``end_time``. Experiment train samples use the same prefix on the train bucket (`load_registry_prefix_split`).

### Stay

- Hash buckets in the registry (train/val/test)
- `MAX_MOVES_PER_BASELINE_BATCH` / `MIN_NEW_MOVES_BASELINE` as round size / skip floor
- Offline ops looping the **same** one-round function
- Assignment writes buckets, never processed flags. Target home: step 1 of the shared train+promote object (each instance ensures *its* `game_id`s). `backfill_game_splits.py` stays for empty envs.
- Promote is step 8 of that object (optional after a round). Offline catch-up still does **not** write promote DB if that is existing offline policy - but the **fetch must match production**

---

## Wire

`CheckSufficientNewDataStep`: count unprocessed train moves (join eligible moves + registry `bucket = 'train'` + baseline flag NULL). Compare to `MIN_NEW_MOVES_BASELINE`. Do not use `count_since(cutoff)`.

`LoadNewDataStep`: fetch next N by `game_id` as above. Do not use `fetch_since`.

After successful fit (today that is `UpdateTrainingStateStep`, or a replacement mark step): set `already_processed_baseline` on the batch `game_id`s.

`scripts/ops/baseline_train_until_caught_up.py` / `catch_up.py`: loop `run_baseline_training_pipeline` (or the shared one-round library). Stop when unprocessed train moves `< MIN_NEW_MOVES_BASELINE`. Optional promote stays a loop extra, not a different fetch.

`offline_baseline_catch_up.py`: same one-round fetch + train + mark. No `HashScanCursor`. Still no promote DB write. Fetch must match production.

Personal: same object, scoped to the user's linked accounts, mark `already_processed_personal`. Parent = last personal keras of this scheme, else `baseline_parent_id`. Offline user catch-up loops that same function.

---

## This increment (personal queue + reset flags)

**Goal:** personal train uses the same registry queue as baseline (no HashScanCursor). Baseline reset actually clears the baseline processed flag. No Keras in this increment. Do not start a second train job.

**Library already there:** `count_unprocessed_train` / `fetch_unprocessed_train_batch` take `flag_column` + `extra_where`. `mark_processed` / `clear_processed` take `flag_column`. `PROCESSED_FLAG_PERSONAL` exists. Hydrate-drop games are omitted from the mark list.

**Add / wire:**

1. `extra_where` for linked accounts: `g.account_id IN (...)` (quote literals). Personal is **user-level**: from CLI `account_id`, load that account's `user_id`, then all `User.get_linked_accounts`. Empty account list → empty extra_where that matches nothing (count 0), not an unbounded platform scan.
2. Personal count/fetch: `flag_column=already_processed_personal` + that `extra_where`. Same `game_id` order, complete last game, mark list from hydrated datums.
3. `offline_user.py` catch-up **train** loop: drop `HashScanCursor` / epoch roll / `take_pending_game_ids`. Count → fetch batch → fit → save → eval → `mark_processed(..., flag_column=PERSONAL)`. Skip/fail must not mark. Still must not import production `catch_up` / `run_baseline_training_pipeline`. Recency weights may still use `end_time` from the batch. `--min-new-moves` / `--batch-limit` stay.
4. Personal **val** for that catch-up: registry val games for those linked accounts (bucket `val`, same `split_version`), not a hash-scan cursor and not platform-wide val. One load, reuse every round. If val is too small, fail like today.
5. `reset_baseline_training`: after cutoff clear, `SplitRegistry(...).clear_processed()` for **baseline** flag (all rows of that `split_version`). Dry-run must not write flags. Archive models behaviour unchanged.
6. Helper to reset **personal** flags for one user (linked-account `game_id`s, `flag_column=PERSONAL`). Callers: tests now; baseline-promote hook later. Do not call it from baseline reset (that would wipe every user's personal queue).

**Tests:** extra_where IN-list; personal fetch does not mark/select val; skip does not mark; `test_baseline_reset` asserts `clear_processed`; offline_user source has no HashScanCursor / `fetch_since`. pytest + ruff on touched paths. ASCII hyphens. `pipelines` may import `platform` (DAG). Never `uv`.

**Out of scope here:** Keras runs, `baseline_parent_id` column, promotion margin, `RandomEvalSetProvider`, `prepare_val_store`, merging train+promote into one `Pipeline` class, starting user catch-up against local DB.

---

## Gaps (not this increment)

| Gap | Why it waits |
|-----|----------------|
| Promote δ / agree ε | Need this frozen-val curve to stop bouncing |
| `RandomEvalSetProvider` → registry val | Phase 4 promote wiring |
| Shared `Pipeline` with assign as step 1 | Orchestration; fetch is already the queue |
| `baseline_parent_id` on personal model rows | Needs a personal model table / column |
| Baseline promote → fan-out personal `clear_processed` | Needs promote success hook |
| `prepare_val_store` | Optional; promote can pack once per job |
| Production train skip-if-small already wired; catch-up still uses old promote policy | After δ/ε |
| One-shot `offline_user_finetune_eval` still hash-splits per account | Catch-up is the queue; one-shot can follow |

---

## Experiments

Fresh queue = all baseline flags NULL (the ADD COLUMN default). Then many one-round loops. Log val metrics each round (same val games every time). Cold-start keras (no parent) unless noted.

That is the incremental story: replay the train pile in `game_id` order, ~10k moves per round, frozen val, flags advance only on success.
