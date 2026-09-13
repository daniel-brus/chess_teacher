# FEN eval cache — plan

**Status:** planning only. No implementation here.

**Last updated:** 2026-09-13 (rev: EPD PK + lookup/LRU indexes)

---

## Idea

Every Stockfish **eval** goes through one class. That class looks up the position, runs SF only if needed, and remembers the result in a small LRU table.

A position is an **EPD** (the board: pieces, side to move, castling, en passant — not the move clocks in a FEN).

Ingested games and live games are the same: a request is a request. Hot means this EPD was asked for again.

---

## Table

`engine.position_evals`. One row per board. **PK = `epd`.**

| Column | Role |
|---|---|
| `epd` | primary key (the board) |
| `engine` | which engine produced the row (attribute, not key) |
| `eval_white_pov` | scalar, white POV |
| `eval_depth` | how strong that scalar is |
| `candidates` | MultiPV JSON (existing payload shape) |
| `candidate_nodes` | how strong that MultiPV is |
| `last_used` | touched on every hit / write |

Switching engine: overwrite or `TRUNCATE`. One position, one result.

Capacity **10_000**. On insert past the cap, delete the oldest `last_used`.

### Indexes (match the queries)

| Query | Index |
|---|---|
| Point / batch get and upsert: `WHERE epd = $1` / `epd = ANY($1)` | **PK on `epd`** (btree). This is the hot path. |
| LRU evict: `SELECT epd ORDER BY last_used ASC LIMIT …` then delete | **btree on `last_used`** |

No other indexes. `last_used` is updated on every hit, so extra indexes would only add write cost. At 10k rows a seq scan is already cheap; these two keep the plan stable if the cap grows.

Batch enrich should use one `WHERE epd = ANY(...)` (or `IN`), not a per-FEN round trip. Touch and evict in the same transaction as the write.

---

## Class

`PositionEvalService` in `pipelines/fen_eval_cache`.

```text
evaluate(fen, ply, eval_depth, candidate_nodes) → {eval_white_pov, candidates}
```

Plus a batch variant for enrich pages. Same rules.

Callers: expensive enrich, backfill, live play, NN bot. Nothing else talks to `StockfishEngine` for evals. (`choose_move` for the Stockfish opponent is not an eval; leave it.)

`StockfishEngine` stays the dumb binary wrapper in `utils`. The service sits above it so `utils` does not import the cache.

---

## Rules

1. Normalize `fen` → `epd`. Look up `epd` (PK).
2. **Hit** if the row exists, `eval_depth` ≥ requested depth, and `candidate_nodes` ≥ requested nodes. Return it. Touch `last_used`.
3. **Too weak:** rerun only the weak search(es) at the requested budget. Upsert those columns.
4. **No row:** run both searches. Insert if **ply ≤ 32**; otherwise return without storing.
5. Ply unknown or ply > 32: still compute for the caller. No new row; still upgrade if the row already exists.

Depth and nodes are independent budgets, same rule as cheap vs expensive MultiPV: a stronger stored value satisfies a weaker ask; a weaker store is overwritten. Play might ask depth 12 / 1k nodes; enrich depth 12 / 50k. If we later want depth 20 scalars, that is just a higher `eval_depth` ask.

The class always returns both payloads. Scalar and MultiPV stay two searches (do not fake one from the other).

---

## What we are not doing

- Redis / a new microservice. The cost that matters is Stockfish, not a ~1 ms read.
- Ply as a *depth*. Ply only gates **insert**.
- Separate “play cache” vs “pipeline cache.”
- Seeding the table from all of `move_characteristics`.
- Faking the scalar from MultiPV (different search).
- Engine (or version) in the PK. Engine is a column; EPD is the key.

`move_characteristics` stays the per-move store. This table is only a position cache.

---

## Fit

Enrich keeps paging 2000 rows and writing mc. It should call the service once per page (`evaluate_many`) instead of two SF transforms.

---

## Open

- Ply 32 vs 24 vs 40; cap 10k vs higher. Knobs, not design.
