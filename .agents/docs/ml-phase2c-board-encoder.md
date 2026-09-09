# Phase 2c — Board representation + training signal (E19–E23)

**Status:** research + offline hybrid path landed on `feature/ml-phase2c-board-encoder`. Production train/promote untouched.

**Primary offline metrics:** stratified **top1 / top3** overall, SF-agree, SF-disagree. Style primary for ranking runs: `top1_sf_disagree` (report top3_disagree alongside).

### Design assumption — greenfield baseline

**No backwards compatibility** with POC `baseline_models` / old Keras heads / old flat-state layouts. Phase 2c+ builds an **improved baseline from scratch** (cold-start). Reuse what is useful:

| Reuse | Drop freely |
|-------|-------------|
| Move characteristics + candidate SF evals in DB | Parent-weight resume from v50/v51 / old feat dims |
| Listwise masked CE over SF candidates | Flat `state` MLP as long-term trunk |
| Registry val + stratified metrics | Silent layout compat shims for production artifacts |

**Package POC:** practically every class under `pipelines/neural_network` is disposable (trainers, promotion, offline CLIs, hybrid encoder, `BaselineModel` rows). Prefer delete/rewrite over API keep. `BaselineTrainer` = A/B control + **TO-BE-SUNSET** if hybrid (or later design) wins; `HybridBoardTrainer` = successor *candidate*, still POC / not production-wired.

Version ints (`BOARD_TENSOR_VERSION`, feat version) stay as **experiment bookkeeping** (log what you trained), not as a parent-resume gate. Feel free to reshape planes, move-feat packs, or the candidate head when A/B says so — still **one change family per A/B**.

---

## E19 — How published nets encode boards (research)

### AlphaZero / Leela (Lc0)

- Input: stacked **spatial planes** `8×8×C`, not a flat hand-crafted vector alone.
- Core planes: **6 piece types × 2 colors** (binary occupancy), plus rules channels (castling rights, side-to-move, en passant, no-progress / move clocks). Strong engines also stack **history** frames and sometimes **attack / mobility** maps.
- Trunk: deep **residual conv** tower → policy over the full move space + value head. Scale: 10⁶–10⁸+ params. Learns geometry (pins, batteries, pawn chains) via weight sharing across squares.

### Maia / human-style nets

- Same family as AlphaZero-style planes + residual towers, trained to predict **human** moves (often by rating band), not perfect play.
- Still a **full policy** over legal moves (or large action encoding), not a listwise ranker over an SF shortlist.

### What transfers to *our* product (rank ≤128 SF candidates)

| Keep / borrow | Drop / defer |
|---------------|--------------|
| Spatial piece planes + castling / EP | Full-move policy from scratch (no SF mask) |
| Shallow **conv trunk → embedding** | Deep residual tower / history stack (v1) |
| Listwise CE over **masked candidates** | Replacing SF + hand move feats on day one |
| Side-to-move orientation (“us” at bottom) | White-fixed board without flip |

Hand-crafted **move feats** (Δeval, geometry, recapture, …) already encode much of what a policy head would rediscover slowly. Bottleneck after Phase 2b / queue catch-up is **position structure** the flat ~state MLP underuses — especially openings — not candidate-slot capacity.

### Chosen plane set (v1) — `BOARD_TENSOR_VERSION = 1`

`C = 17`, float32, shape `(8, 8, 17)`, always oriented so **side to move (“us”) sits on ranks 1–2** (board flipped when Black to move).

| Idx | Plane |
|-----|--------|
| 0–5 | Our P,N,B,R,Q,K |
| 6–11 | Their P,N,B,R,Q,K |
| 12–13 | Our kingside / queenside castling rights (fill 1.0 if legal) |
| 14–15 | Their kingside / queenside castling rights |
| 16 | En-passant target square (1.0 on that square if set) |

**Version note:** bump `BOARD_TENSOR_VERSION` when channel layout changes so logs/MLflow stay auditable. No parent-resume requirement — cold-start after layout change is fine.

**Deferred (ablate later):** history frames; attack maps; per-candidate from/to planes (coords already in move feats); fen_after / delta boards.

### Chosen hybrid architecture (v1 → v1b fuse)

Board conv is **additive** to the flat state tower (not a replacement).

```
board (8×8×17)
  → Conv2D(64, 3, same, relu) × 2
  → GlobalAveragePooling2D
  → Dense(hidden=128, relu)          # board_emb

state (D≈20,)
  → Dense(128, relu) × 2             # same as BaselineTrainer state tower
  → state_emb

concat(board_emb, state_emb) → Dense(128, relu)   # fused_emb
  → RepeatVector(MAX)
  → concat move_feats (MAX, 55)
  → TimeDistributed Dense(score_hidden=64) → Dense(1)
  → logits (MAX,)  + masked sparse CE
```

- **Control:** flat `state` MLP + move feats only (`BaselineTrainer`).
- **Treatment:** board conv **+** state tower + same candidate head.
- First A/B (board **replaced** state, conv=32): negative on full val — see below.
- One change family per A/B: encoder **or** weights/loss — not both.

**Reject criterion:** if full-registry-val A/B shows no clear gain on `disagree_t1` / agreed weighted product vs MLP control under same epochs/batch/weights, document negative and park deeper towers / planes — do not wire production.

### E20 A/B result (2026-09-08) — negative on *replace*-state hybrid (conv=32)

Cold-start both arms: train `--limit 10000` (8532 train moves), `--epochs 20`, style boost 2.0, **full registry val** `n=45027`. Hybrid **replaced** flat state (no fuse); `conv_filters=32`.

| Arm | params | top1 | top3 | agree_t1 | disagree_t1 |
|-----|--------|------|------|----------|-------------|
| MLP (control) | 31041 | 0.3757 | 0.6354 | 0.6366 | **0.2149** |
| Hybrid replace-state | 30241 | 0.3711 | 0.6350 | 0.6332 | **0.2096** |

Primary delta `disagree_t1` (hybrid − mlp) = **−0.0052**. Follow-up design: **fuse** board + state + bump conv to 64 (v1b) — re-run before parking conv.

### E20 A/B result (2026-09-09) — fuse board+state hybrid (conv=64), catch-up 3×10ep

Protocol: catch-up R1 cold → R2–R3 resume; `--epochs 10` / round; `--max-rounds 3`; `--batch-limit 10000` (~10k train/round); style boost 2.0; **full registry val** packed once `n=45027`. Hybrid = board fuse + state (v1b), `conv_filters=64`. Wall ~5.1h. Weights under `storage/tmp/encoder_ab_fuse64_r3_ep10/`.

| Round | mlp disagree_t1 | hybrid disagree_t1 | Δ (h−m) | mlp top1 | hybrid top1 |
|------:|----------------:|-------------------:|--------:|---------:|------------:|
| 1 | 0.2105 | 0.2145 | **+0.0040** | 0.3854 | 0.3829 |
| 2 | 0.2153 | 0.2184 | **+0.0031** | 0.4098 | 0.4088 |
| 3 | 0.2161 | 0.2221 | **+0.0060** | 0.4172 | 0.4102 |

Hybrid beats MLP on primary `disagree_t1` every round (small +3–6‰) but loses overall/agree top1 by R3 (`top1` −0.0071, `agree_t1` −0.0283). Params 119k vs 31k. **Not a clear promote** — style edge tiny; agree regresses under catch-up. Next: park deeper towers, or try E21 forced-weight / E22 loss with **frozen** encoder winner (MLP still control).

---

## E20 — Offline path (implementation map)

| Piece | Path |
|-------|------|
| Tensor pack | `board_tensor.py` |
| Hybrid trainer | `board_encoder.py` (`HybridBoardTrainer`) |
| Eval predict | `eval_metrics.evaluate_datums` (detects `board` vs `state` inputs) |
| A/B CLI | `scripts/tools/offline_baseline_encoder_ab.py` |

Serious decisions: **full registry val** (or `--limit` large enough for ≥100 val games). Small limits = smoke only.

---

## E21 — Sample-weight proposals (forced + disagree)

**Keep:** ply reweight + continuous SF-disagree boost (`ply_weights.py`).

**Add (library helpers; ablate separately from encoder):**

1. **Forced-move downweight (continuous)** — let
   `delta = eval(second_best) - eval(best)` (user POV among masked; ≤ 0).
   Weight factor `exp(delta / scale)` with `scale` in pawns (default `1.5`).
   Equal top-two → factor 1; more negative delta (clearly forced) → factor → 0.
   No binary cliff.
2. **Recapture flag** — already in move feats (`is_recapture`); optional extra term later.
3. **Disagree schedule** — optional epoch-dependent boost. Not in v1 train loop.

Helpers: `second_vs_best_delta_pawns`, `best_vs_second_gap_pawns`,
`forced_move_downweight_factor` (continuous). Wire via `forced_scale_pawns=` on
`candidate_style_sample_weights` (default off). No forced-move **metric slice** —
forcedness is only a soft training weight; style signal lives in SF-disagree eval.

**Ablate:** encoder fixed → forced scale on/off / tune scale; report agree_t1/t3 + disagree_t1/t3.

---

## E22 — Target / loss proposals

Still categorical over masked candidates.

| Variant | Idea | When |
|---------|------|------|
| **A (baseline)** | One-hot sparse CE (current) | Control |
| **B Soft labels** | Mix one-hot with temperature-softmax of SF evals among candidates | Style vs engine prior |
| **C SF-policy mix** | `loss = (1−α)·CE_user + α·CE_sf_best` | Regularize toward engine |
| **D Sliced CE** | Separate agree/disagree heads or weighted CE terms | If single CE underfits disagree |

Pick by registry-val disagree (guardrail: agree must not collapse). **Do not** change loss in the same run as the first encoder A/B.

---

## E23 — Metrics pack + Phase 4 gate proposal

**Report every offline eval (now):**

- Overall / SF-agree / SF-disagree: **top1 and top3**
- Counts: `n_eval`, `n_sf_agree`, `n_sf_disagree`, `sf_disagree_frac`
- Optional slices: game phase (opening / middle / endgame).
  Forcedness stays in the **loss weight only** — no forced/free eval buckets
  (free-as-not-forced is the same empty signal flipped).

**top-k definition (glossary):** played move ranks in the model’s top *k* among masked candidates — not “engine top-k”.

**Phase 4 promotion proposal (not wired):**

- Primary: `top1_sf_disagree` (or weighted product `0.5·disagree_t1 + 0.5·agree_t1` if agree volatility high)
- Guardrail: `top1_sf_agree` must not collapse vs the **MLP control** on the same val (Phase 4: vs prior promoted greenfield baseline, not POC v50)
- Log top3_* for monitoring only until gates proven

---

## Experiment protocol (encoder A/B)

1. Same `split_version`, same train/val datums, same epochs/batch, same style weight knobs.
2. Cold-start both arms (`weights_path=None`).
3. Primary: `top1_sf_disagree`; also print stratified top3.
4. Log param counts + `BOARD_TENSOR_VERSION` / `CANDIDATE_MOVE_FEAT_VERSION`.
5. Full registry val for go/no-go; `--limit 2000` only for plumbing smoke.
