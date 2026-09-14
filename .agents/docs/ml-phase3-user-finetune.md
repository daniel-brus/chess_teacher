# Phase 3 — User finetune (kickoff)

**Branch:** `feature/ml-phase3-user-finetune` (from `develop` after Phase 2c merge).
**Roadmap:** `.agents/docs/ml-training-roadmap.md` § Phase 3.
**Prior phase:** `.agents/docs/ml-phase2c-board-encoder.md` (closed).

## Locked before you start

| Topic | Choice |
|-------|--------|
| Encoder | Hybrid preferred (`HybridBoardTrainer`) |
| Loss | `sf_mix` @ `sf_mix_alpha=0` (user-only CE). Soft rejected. Do not raise alpha on user finetune. |
| **Splits** | Persistent **hash registry** only (`baseline-v1` / `ml.game_split_assignments`) |
| Prod wiring | Phase 4 — offline tools/ops/notebook only here |

### User data = same registry, filtered by account

| | Train | Val |
|--|-------|-----|
| One `account_id` | user moves ∩ registry **train** | user moves ∩ registry **val** |

### Forbidden

- Time-ordered user train/val (last N% games, sort by `end_time`)
- Per-account alternate salts / parallel “user registry”
- Cutoff / `fetch_since` as the **train/val partition**
- Reopening time-based splits in docs or code

### Catch-up / incremental

- Progress = **unprocessed registry-train** queue (processed flags), not “games since cutoff” as a split.
- Optional recency = sample **weights** inside train only — never chooses val.

## Phase 3a (do this first)

1. Account×registry loaders (extend `TrainingDataStore` / `offline_eval` — **no** time-based `user_splits`).
2. `scripts/tools/offline_user_finetune_eval.py`: parent baseline → train user∩train → eval user∩val.
3. Stratified metrics; primary `top1_sf_disagree`; guardrail agree.
4. Exit: 2–3 accounts; ~300+ registry-train moves; beat parent on user registry-val disagree.

## Phase 3b (after 3a)

- `offline_user_promotion.py` — user model vs baseline on user∩registry-val.
- `offline_user_catch_up.py` — unprocessed **registry-train** for that account; re-eval user∩val each round.
- Notebook section mirroring tools.

## Agent rules

- Ask before large ambiguous changes.
- One change family per experiment.
- Doppler `dev_local` + `.venv` — never `uv`.
- Do not commit unless asked.
