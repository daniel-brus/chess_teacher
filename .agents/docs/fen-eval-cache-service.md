# FEN eval cache — plan

**Status:** planning only. No implementation here.

**Last updated:** 2026-09-13 (rev: stripped)

---

## Idea

Every Stockfish **eval** goes through one class. That class looks up the position, runs SF only if needed, and remembers the result in a small LRU table.

A position is an **EPD** (the board: pieces, side to move, castling, en passant — not the move clocks in a FEN).

Ingested games and live games are the same: a request is a request. Hot means this EPD was asked for again.

---

## Table

`engine.position_evals`. One row per board.

| Column | Role |
|---|---|
| `epd` | primary key |
| `eval_white_pov` | scalar, white POV, depth 12 |
| `candidates` | MultiPV JSON (existing payload shape) |
| `candidate_nodes` | how strong that MultiPV is |
| `last_used` | touched on every hit / write |

That is the whole key. No hash, no engine version, no payload version, no “kind” lane.

If we change Stockfish or the JSON shape in a way that matters: `TRUNCATE` the table. It is a cache.

Capacity **10_000** rows. On insert past the cap, delete the row with the oldest `last_used`.

---

## Class

`PositionEvalService` in `pipelines/fen_eval_cache`.

```text
evaluate(fen, ply, candidate_nodes) → {eval_white_pov, candidates}
```

Plus a batch variant for enrich pages. Same rules.

Callers: expensive enrich, backfill, live play, NN bot. Nothing else talks to `StockfishEngine` for evals. (`choose_move` for the Stockfish opponent is not an eval; leave it.)

`StockfishEngine` stays the dumb binary wrapper in `utils`. The service sits above it so `utils` does not import the cache.

---

## Rules

1. Normalize `fen` → `epd`. Look up `epd`.
2. **Hit** if the row exists and `candidate_nodes` ≥ what the caller asked for. Return it. Touch `last_used`.
3. **Miss / too weak:** run Stockfish (scalar depth 12 + MultiPV at the requested nodes). Return that.
4. **Remember** the new result if:
   - there is already a row (always **upgrade** in place), or
   - there is no row and **ply ≤ 32**.
5. Deep positions (ply > 32, or ply unknown) are still computed for the caller. They just do not get a new row.

Play asks for 1k nodes; enrich asks for 50k. Same row. A 50k store satisfies a later 1k ask. A 1k store does not satisfy 50k: recompute and overwrite.

Scalar and MultiPV are two SF searches. The class always returns both. Callers do not request one or the other.

---

## What we are not doing

- Redis / a new microservice. The cost that matters is Stockfish, not a ~1 ms read.
- Ply as a *depth*. Ply only gates **insert**.
- Separate “play cache” vs “pipeline cache.”
- Seeding the table from all of `move_characteristics`.
- Faking the scalar from MultiPV (different search).

`move_characteristics` stays the per-move store. This table is only a position cache.

---

## Fit

Enrich keeps paging 2000 rows and writing mc. It should call the service once per page (`evaluate_many`) instead of two SF transforms.

---

## Open

- Ply 32 vs 24 vs 40; cap 10k vs higher. Knobs, not design.
