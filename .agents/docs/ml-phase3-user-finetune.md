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

1. Account×registry loaders (extend `TrainingDataStore` / `offline_eval` — **no** time-based `user_splits`). ✅ `load_account_registry_*` in `offline_eval.py`
2. `scripts/tools/offline_user_finetune_eval.py`: parent baseline → train user∩train → eval user∩val. ✅ (+ `--train-parent` cold hybrid)
3. Stratified metrics; primary `top1_sf_disagree`; guardrail agree (report-only). ✅
4. Exit: 2–3 accounts; ~300+ registry-train moves; beat parent on user registry-val disagree. ✅ smoke E13 + scaled local ladder (see below)

### Smoke E13 (local, 2026-09-15) — ikbendaniel only

Cold hybrid parent `--parent-train-limit 10000` ep5 → finetune train5k/val3k ep5 style boost 4.0.

| | disagree_t1 | agree_t1 | top1 |
|--|------------:|---------:|-----:|
| parent on user val | 0.1942 | 0.7489 | 0.3987 |
| user finetune | **0.2058** | 0.6965 | 0.3867 |
| delta | **+0.0116** | −0.0524 | −0.0120 |

Primary met (disagree↑). Agree drop reported (report-only). Artifacts under `storage/tmp/phase3a/` (gitignored).

### Scale ladder (local, 2026-09-21 → 24) — accept limitations

- Cold hybrid **parent** `100k+100k` (ikbendaniel + RebeccaHarris, balanced) → Play `phase3a_hybrid_parent_100k100k`.
- Per-account FT rounds from that parent (train caps 25k→200k, ep10, style boost 4.0); each round registered archived Play.
- **RebeccaHarris r1–r8** complete + registered.
- **ikbendaniel r1–r5** complete; **r6–r8** resumed after float64 OOM fix (`ply_weights` / disagree-frac sample).
- Known Play feel: side imbalance (e.g. White vs Black strength) — **not** a Phase 3a blocker; Phase 4 / later.

**Closed for 3a code path:** loaders, FT tool, balanced parent, chunked fit, Play register scripts, OOM-safe disagree path. **Not in 3a:** promotion/catch-up (→ **3b**), Personal multi-account UX (→ **4**), color/side calibration.

Tools: `scripts/tools/phase3a_scale_100k_chain.py`, `register_phase3a_playables.py`, `phase3a_resume_daniel_after_rh.ps1`.

## Phase 3b (after 3a)

- `offline_user_promotion.py` — user model vs baseline on user∩registry-val.
- `offline_user_catch_up.py` — unprocessed **registry-train** for that account; re-eval user∩val each round.
- Notebook section mirroring tools.

## Agent rules

- Ask before large ambiguous changes.
- One change family per experiment.
- Doppler `dev_local` + `.venv` — never `uv`.
- Do not commit unless asked.
