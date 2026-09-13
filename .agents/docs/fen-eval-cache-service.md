# FEN eval cache — design plan

**Status:** planning only. No implementation on this branch beyond the brief + this doc.

**Audience:** chess_teacher owner review before any code.

**Source brief:** `.agents/docs/brief-fen-eval-cache-service.md` on `feature/sf_lookup_service`.

**Last updated:** 2026-09-13 (rev: one row per EPD+kind; upgrade-in-place; no ply gate)

---

## 1. Problem / goals / non-goals

### Problem

Preprocess pages at 2000 rows and only de-dupes FENs inside one page. Play pays Stockfish again at a **smaller** live budget (`LIVE_CANDIDATE_STOCKFISH_NODES=1000`) so a move is not minutes. Same boards keep showing up (openings, the current game, a line you play every week).

The product is a **hot registry**, not “never compute twice.” Skip Stockfish when we already have a result that is **at least as strong** as the request. One-off middlegames may fall off the LRU and be recomputed.

### Goals

- Fixed-capacity **LRU** of position → engine result.
- **One row per position per result kind** (not one row per node budget). Stored `depth` / `num_nodes` are the **quality** of that row.
- Request at budget D: hit if stored ≥ D; else compute at D and **upsert** (replace the weaker row).
- **Play is the insert path** (that is the speed win). Preprocess **reads and upgrades** rows that already exist; it does not flood the LRU with the 737k tail.
- Compute-on-miss. Empty cache never blocks play or pipeline.
- Fit import DAG, `metadata.yml`, `DatabaseClient`.

### Non-goals

- NN instead of Stockfish.
- Cloud Redis. Local Redis as the *registry* (see §3).
- Changing `candidate_evaluations` train/label JSON.
- Caching cheap board metrics.
- Rewriting stored `games.moves.fen_*`.
- Ply-based admission (rejected).
- Treating a scalar played-move eval and a MultiPV candidate map as the same row.

---

## 2. Cache key and value schema

### Position key

`position_key = chess.Board(fen).epd()` (placement, STM, castling, EP). Clocks stay on the move row / live board only.

### Identity vs quality

**Identity (unique row):** `kind`, `engine`, `engine_version`, `payload_version`, `position_key`.

**Quality (columns on the row, overwritten on upgrade):** `depth`, `num_nodes`.

| `kind` | What we store | “Higher” means |
|---|---|---|
| `eval` | `eval_white_pov` (played-move white-POV pawns) | larger `depth` |
| `candidates` | `build_candidate_payload` JSONB + `legal_uci_count` | larger `num_nodes` |

`eval` and `candidates` stay **two rows** for the same EPD. A depth-12 scalar is not a 50k MultiPV map. Do not compare depth to nodes.

`payload_version` still bumps if mate/cp mapping or JSON shape changes.

`cache_key = sha256(kind|engine|engine_version|payload_version|position_key)`.

### Lookup / upgrade (the simple rule)

Caller asks for EPD + kind + requested budget `R` (`depth` for `eval`, `num_nodes` for `candidates`).

| Stored row | Action |
|---|---|
| None | Compute at `R`. **Insert only if the caller may admit** (§2.1). |
| Present, quality ≥ `R` | **Hit.** Use it. Move to LRU top. Do not run SF. |
| Present, quality < `R` | Compute at `R`. **Upsert** the same row (new payload + new quality). Move to top. |

That is not too simple **inside one kind**, if “higher” is a single number. It *is* too simple if we pretend every Stockfish call is one totally ordered “depth.”

Complement of “found lower → compute higher”: **found higher → serve it.** A play 1k request against a 50k row is a hit. That is faster *and* closer to train labels (train is 50k). A play miss still computes 1k so the user is not stuck, then sits at 1k until preprocess (or a later 50k caller) upgrades the row.

NULL `num_nodes` (depth-only MultiPV fallback) is weaker than any positive node budget.

### 2.1 Who may insert a *new* row (no ply)

Ply gate is out. Without some other gate, preprocess `put` of every miss still rotates a 10k LRU through unique middlegames and evicts play openings.

**Rule:**

| Caller | No row | Weaker row | Strong enough row |
|---|---|---|---|
| **Play** | Compute `R` (1k), **insert** | Upgrade to `R` (rare; play is the low budget) | Hit |
| **Preprocess / backfill** | Compute `R` for **mc only**. **Do not insert.** | Compute `R`, **upsert** | Hit |

The registry is “positions play (or a future explicit warmer) cared about,” optionally **strengthened** when enrich sees them again. Middlegames still land on `move_characteristics`. They do not get a registry slot unless play also hit that EPD.

Optional later: a one-off warmer that inserts a small opening book. Not a dump of all complete mc.

### LRU

Same as before: move-to-front on get/upsert; fixed cap per **identity lane** `(kind, engine, engine_version, payload_version)` (default 10_000); evict least recently accessed when over cap. `last_access_seq`, not wall-clock TTL, not a linked list.

Play and preprocess now **share** the `candidates` lane (one row per EPD). That is OK because preprocess does not insert new keys. Eviction is “cold play positions,” not “last 2000 backlog FENs.”

### Table sketch

`engine.position_evals`: identity columns + `depth` + `num_nodes` + payload/eval + `legal_uci_count` + `last_access_seq` + `created_at`.

Unique `(kind, engine, engine_version, payload_version, position_key)`.

---

## 3. Architecture and speed (Postgres vs Redis)

**The speed win is skipping Stockfish**, not the cache product.

Order of magnitude on this VPS:

| Step | Typical cost |
|---|---|
| Postgres `SELECT` by PK / unique EPD | ~1 ms |
| Redis `GET` | sub-ms |
| Live MultiPV @ 1k nodes | tens–hundreds of ms (often more with many legals) |
| Pipeline MultiPV @ 50k nodes | seconds per FEN |

A hit is “instant” next to a miss on either store. Redis is the **faster cache**. It is not the better **registry** here:

- We need “find this EPD, compare quality, upsert in place.” That is a row update, not a blob GET/SET.
- LRU + unique key + inspect `num_nodes` is normal SQL.
- Redis on the same 4GB box is RAM-only (unless we add persistence) and already holds the user-games TTL cache.
- 10k × 2KB would *fit* in Redis; the model would not (upgrade-in-place, shared play/preprocess row).

**Recommendation stays Postgres + library client.** Revisit Redis if play traces show cache IO (not SF) on the hit path. A Redis *front* that only stores “already strong enough” blobs is optional later, not v1.

No microservice now.

```
utils → pipelines/fen_eval_cache → preprocessing
                                 → bots / play
```

---

## 4. API / client contract

```python
class FenEvalCacheClient:
    def get_many(self, keys: Sequence[CacheLookup]) -> dict[str, CacheHit]:
        """Hit = stored quality >= requested. Touches LRU. Else omit (miss / too weak)."""

    def put_many(self, rows: Sequence[CacheRow], *, admit_new: bool) -> int:
        """Upsert if key exists (always, when new quality >= stored).
        Insert if missing only when admit_new.
        Evict lane bottom past capacity."""
```

Play: `admit_new=True`. Enrich: `admit_new=False`.

Degrade if Postgres is down. Flag `FEN_EVAL_CACHE=off|readwrite|readonly`.

---

## 5. Write / read path

```mermaid
sequenceDiagram
  participant C as Play or Enrich
  participant R as Registry
  participant SF as Stockfish
  participant Out as UI or mc

  C->>R: lookup EPD + kind + requested R
  alt row quality >= R
    R-->>C: hit (bump LRU)
  else row quality < R
    C->>SF: compute at R
    SF-->>C: payload
    C->>R: upsert same row, bump LRU
  else no row
    C->>SF: compute at R
    SF-->>C: payload
    alt play (admit_new)
      C->>R: insert, evict if over cap
    else enrich
      Note over R: mc only; no new registry row
    end
  end
  C->>Out: move / checkpoint
```

Enrich still pages 2000. Load FENs, EPD-normalize, `get_many` at 50k (or depth 12 for `eval`). Hits skip SF. Weak rows (1k leftovers from play) upgrade. Unknown EPDs run SF into **mc only**.

Play: one EPD per think at requested 1k. Stronger cached row → no SF, better numbers. Else 1k miss path.

---

## 6. Ops

No new Deployment. Cap 10k × 2 kinds is tens of MB. Inline evict on insert. Logs: hits, weak-miss (upgrade), hard-miss, admitted, skipped-admit, evicted.

`STOCKFISH_VERSION` in the image: new version = new identity lane (old rows unused).

---

## 7. Rollout

| Phase | What | Exit |
|---|---|---|
| **1** | Table + client + play insert + enrich upgrade-only | Play repeat/book can skip SF. Enrich of an unknown middlegame does not add a row. Enrich of a play-1k opening upgrades to 50k and play then hits. LRU evicts the coldest after cap. |
| **2** | Backfill uses same client (`admit_new=False`) | Backfill cannot grow the registry. |
| **3** | Optional tiny opening warmer (`admit_new=True`, curated) | Only if play cold-start feels empty. |
| **4** | Service / Redis front | Only if hit-path IO matters. |

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Mixed 1k / 50k on the live bot | Accept: miss = 1k (playable); later upgrade = 50k (closer to train). Do not mix `eval` vs `candidates`. |
| Preprocess insert-all | Forbidden (`admit_new=False`). |
| “Higher” across depth vs nodes | Separate kinds; candidates compare nodes only. |
| Train/live purity | Document the mix. If we must match 1k exactly, add a flag “exact budget only” later. Default is serve-if-stronger. |
| Legal-move drift | `legal_uci_count` mismatch → miss + upsert. |
| Cap too small | Env bump. |

---

## 9. Open questions

1. Capacity 10_000 per kind-lane — OK?
2. Schema `engine.position_evals` vs `games.position_evals`?
3. Reprocess: still read/upgrade, or bypass?
4. Exact-budget play flag now, or accept stronger-is-fine?
5. Cold-start warmer for openings, or wait for play to fill?

Resolved: no ply gate; one row per EPD+kind; weaker → compute requested and upsert; play inserts; enrich upgrades only.

---

## Recommendation summary

| Topic | Decision |
|---|---|
| Speed | From skipping SF. Postgres hit ≈ Redis hit for this purpose. Redis is a faster cache, worse upgrade/upsert store. |
| Row shape | One row per EPD + kind. Quality columns, not key. |
| Weaker stored | Compute higher requested, upsert in place. Works if we stay within kind. |
| Stronger stored | Serve it (do not recompute). |
| Admission | Play inserts. Preprocess does not. No ply. |
| Eviction | LRU cap ~10k, move-to-front. |

---

## ETA

**Play:** first miss in a line pays 1k MultiPV; repeats / takebacks / next game in book are hits. After enrich has seen that book EPD, hits become 50k “for free.” That is the point.

**737k backlog:** almost no pipeline speedup (enrich does not insert, so it only hits EPDs play already stored). Mc still gets every compute. Fine: the registry is for play.

---

## Implementation sketch (not this PR)

1. `pipelines/fen_eval_cache` with quality compare + upsert + `admit_new`.
2. Play: lookup → maybe SF → put admit.
3. Enrich: lookup → SF misses/weak → put upgrade-only → existing mc checkpoint.
4. Tests: weaker upsert, stronger hit, enrich does not insert, LRU cap, degrade.

Do not change candidate JSON shape or train/play default node constants. Do not add a k8s Service.
