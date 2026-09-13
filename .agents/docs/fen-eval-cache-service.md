# FEN eval cache — plan

**Status:** planning only. No implementation here.

**Last updated:** 2026-09-13 (rev: engine+epd key; depth is a budget)

---

## Idea

Every Stockfish **eval** goes through one class. That class looks up the position, runs SF only if needed, and remembers the result in a small LRU table.

A position is an **EPD** (the board: pieces, side to move, castling, en passant — not the move clocks in a FEN).

Ingested games and live games are the same: a request is a request. Hot means this EPD was asked for again.

---

## Table

`engine.position_evals`. One row per **engine + board**.

| Column | Role |
|---|---|
| `engine` | e.g. `stockfish` — part of the PK |
| `epd` | the board — part of the PK |
| `eval_white_pov` | scalar, white POV |
| `eval_depth` | how strong that scalar is |
| `candidates` | MultiPV JSON (existing payload shape) |
| `candidate_nodes` | how strong that MultiPV is |
| `last_used` | touched on every hit / write |

Primary key: `(engine, epd)`. EPD alone is enough only if a single engine will ever own the table. A later engine switch would otherwise collide or force a wipe. With `engine` in the key, Stockfish rows stay put; a new engine fills its own rows. LRU cap **10_000 per engine**.

`engine` is the family name, not a patch version. A Stockfish apt bump can keep using the same rows (small drift) or we delete `WHERE engine = 'stockfish'`. Do not put version in the key unless we need two Stockfish builds side by side.

No hash, no payload version, no kind lane.

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

1. Normalize `fen` → `epd`. Look up `(engine, epd)`.
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
- Engine **version** in the PK (only the engine name).

`move_characteristics` stays the per-move store. This table is only a position cache.

---

## Fit

Enrich keeps paging 2000 rows and writing mc. It should call the service once per page (`evaluate_many`) instead of two SF transforms.

---

## Open

- Ply 32 vs 24 vs 40; cap 10k vs higher. Knobs, not design.
