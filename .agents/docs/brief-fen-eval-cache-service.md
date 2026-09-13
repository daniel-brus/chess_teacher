# Agent brief: plan a FEN eval cache service

**Mode:** planning only — produce a design/plan doc (tradeoffs, phases, interfaces). Do **not** implement yet.

**Audience:** chess_teacher owner will review the plan before any code.

---

## Problem

Prod preprocessing OOMs forced **batched** `EnrichCheap` / `EnrichExpensive` (`TransformStep.batch_size=2000`). FEN dedup is **per page only**, so Stockfish depth-12 evals and MultiPV candidate evals re-run for the same FEN across pages. One heavy account: ~737k incomplete moves → ~18d wall-clock estimate, partly duplicate SF work.

Goal: a **shared FEN → eval / candidates cache**, like online chess platforms (position hash → engine result), reusable across pipeline batches, runs, accounts, and eventually play/bots.

## Product intent

- Brand-new capability on the platform (not a quick hack inside one transform).
- Strongly consider a **separate service / microservice** (own deploy, API, storage) vs in-process/Postgres table in the monolith — plan must recommend one with rationale for this repo’s size (single Hetzner VPS k3s today).
- Cache keys must encode **engine identity + budget** (depth / `num_nodes` / SF version / payload version), not bare FEN alone.
- Consumers (priority order to discuss): preprocessing expensive enrich; optional later: `bots` / play live MultiPV; offline backfill scripts.

## Current system (read these first)

| Area | Path |
|------|------|
| Import DAG / layering | `.agents/skills/import-dag/SKILL.md` |
| Expensive enrich step (batched join loader) | `src/chess_teacher/pipelines/preprocessing/pipeline_steps.py` (`EnrichExpensiveMoveCharacteristicsStep`) |
| Played-move SF + FEN dedup in-frame | `src/chess_teacher/pipelines/preprocessing/fen_characteristic.py` |
| MultiPV candidates → `candidate_evaluations` JSONB | `src/chess_teacher/pipelines/preprocessing/move_characteristics/candidate_evaluations.py` |
| Payload shape / node budgets | `src/chess_teacher/pipelines/neural_network/candidate_eval.py` (`CANDIDATE_STOCKFISH_NODES=50_000`, depth 12, `build_candidate_payload`) |
| Mid-run FEN checkpoints (existing pattern) | `src/chess_teacher/pipelines/preprocessing/fen_checkpoint.py` |
| Prod ops / k8s jobs | `.agents/skills/chess-teacher-vps/SKILL.md`, `scripts/utils/run_script_job.py` |
| DB metadata style | `**/metadata.yml` + `scripts/tools/agent_db_query.py` |

Today results land on **`games.move_characteristics`** per `move_id`, not per FEN. Cache should be FEN-keyed; mc rows remain the account-facing store.

## Constraints

- Small VPS (≈4GB RAM, no swap); pipeline jobs BestEffort — cache must not recreate OOM.
- Python package stays an **import DAG** (utils → platform → pipelines → bots). A microservice may live outside that DAG (HTTP/gRPC client only from pipelines).
- Doppler secrets; k3s manifests under `orchestration/k8s/`.
- Prefer resumable writes; cold-start empty cache must not block pipelines (compute-on-miss).
- Do not invent write paths that skip existing Stockfish budget contracts used by train/backfill/play.

## Plan deliverable (required sections)

1. **Problem / goals / non-goals**
2. **Cache key & value schema** (FEN normalization, white/STM POV, eval vs candidates, versioning, invalidation)
3. **Architecture options** ranked: (A) Postgres table in monolith, (B) sidecar/library + shared DB, (C) dedicated microservice + API + storage — pick a recommendation for *this* platform now vs later
4. **API / client contract** (get/put/batch get; sync vs async populate; failure modes)
5. **Write path** (who fills cache: enrich miss, background warmer, one-off backfill job)
6. **Read path** into `EnrichExpensive*` (and optionally cheap fen metrics) without breaking `TransformStep` batching
7. **Ops**: deploy on k3s, resources, retention/eviction (LRU / popularity), observability
8. **Rollout phases** with exit criteria (e.g. phase 0 in-process page cache → phase 1 durable store → phase 2 service split if warranted)
9. **Risks**: stale budgets, hash collisions, legal-move set drift, multi-tenant pollution, VPS cost
10. **Open questions** for the owner

## Success criteria for the plan

- Clear recommendation: microservice **now** vs **later**, with trigger conditions to split.
- Concrete schema + 1–2 sequence diagrams (miss → compute → store → hit across batches).
- Honest estimate of ETA impact on the current ~737k-move backlog (order-of-magnitude only).
- Fits chess_teacher conventions (metadata.yml, pipeline steps, k8s jobs) or explicitly proposes new package boundaries.

## Out of scope for the plan’s first build

- Replacing Stockfish with a NN evaluator.
- Cloud-only managed cache (Redis Cloud etc.) unless justified vs local Postgres/volume.
- Changing train/label semantics of `candidate_evaluations` beyond shared compute.
