# Phase 5 — Hybrid update rule (kickoff)

**Branch:** `feature/ml-phase5-hybrid-update-rule` (from `develop` after Maia closed).
**Roadmap:** `.agents/docs/ml-training-roadmap.md` § Phase 5 + Current workflow.
**Prior:** Maia board path CLOSED — `.agents/docs/maia-style-action-plan.md`.
Keep **shallow hybrid** as the product encoder.

## Why this phase

Phases 1–4 + play selection are largely on `develop`. Maia residual / spatial /
dual-head A/Bs all lost. Next quality lever is **how** we warm-start and update
the shallow hybrid — not a deeper board tower.

Open Phase 4 leftovers (E16 / E18) stay on the Phase 5 backlog; they are
orchestration parity, not this kickoff’s first experiment.

## Slice A — Update rule (do this first)

**Goal:** Finetune from a parent hybrid without wrecking the human policy.
Distill the **parent human** distribution. Do **not** distill Stockfish.

| Knob | Working default (ablate) |
|------|--------------------------|
| Peak Adam LR | ~`1e-4` (cold start stays ~`1e-3`) |
| Warmup | Short; optimizer moments reset on parent load |
| Replay | ~1:1 old registry-train games mixed with new batch |
| Epochs | 1–3 warm; keep **best registry-val** checkpoint |
| Loss | Library default `sf_mix` @ `α=0` |

### ABCD offline experiment

Shared cold parent on ~50% of registry-train (by game). Four warm arms on the
other half; same registry val. Matched warm epoch budget.

| Arm | Change vs today |
|-----|-----------------|
| `control` | Adam `1e-3`, last epoch, new batch only |
| `lr_only` | Adam `1e-4`, last epoch |
| `lr_best_val` | `1e-4` + warmup + best val epoch |
| `lr_best_val_replay` | same as `lr_best_val` + ~1:1 old/new mix |

CLI: `scripts/tools/offline_update_rule_ab.py`
Trainer knobs: `HybridBoardTrainer(learning_rate=…, warm_learning_rate=…, warm_warmup_epochs=…, select_best_val=…)`.

### Offline shape

- Same registry train/val hygiene as Phase 2 siblings
- Prefer mid-size decision runs (≥10k; ~30k/8k fine) before full buckets
- Primary: val `top1` / `val_log_loss` / `top1_sf_disagree`; pawn gap safety
- Gate vs `control`: clear LL drop or ≥+1–2 top-1 without pawn-gap regress

### Deliverables (Slice A)

1. Trainer knobs + offline ABCD harness ✅
2. Mid-size ABCD result (below / `storage/tmp/update_rule_ab_mid.json`) ✅
3. If a clear winner: port into baseline train path — **not yet** (no winner)

### Mid-size result (decision) — REJECT soft update-rule (2026-11-06)

`storage/tmp/update_rule_ab_mid.json` — parent 15 343 / finetune 14 686 / val 8 006,
warm epochs 3, matched epoch budget. Gate vs `control`: clear LL drop or ~+1–2
top-1; pawn gap no worse.

| Arm | top-1 | disagree | val_log_loss | pawn gap | vs control |
| --- | ---: | ---: | ---: | ---: | --- |
| **control** (`1e-3`, last ep) | **0.421** | **0.196** | **2.032** | **-0.16** | — |
| `lr_only` (`1e-4`) | 0.401 | 0.182 | 2.092 | -0.16 | −2.0 top1, worse LL |
| `lr_best_val` | 0.401 | 0.182 | 2.092 | -0.16 | same as `lr_only` (best=last) |
| `lr_best_val_replay` | 0.407 | 0.186 | 2.070 | -0.16 | −1.3 top1, worse LL |

**Verdict:** keep today’s warm-start (`Adam 1e-3`, last epoch). Soft LR hurts;
best-val tied last epoch here; 1:1 replay only partially recovers, still loses
gate. Do **not** promote Slice A knobs as new defaults. Trainer flags stay for
future ablations.

## Slice B — Disagree boost — DONE (prefer boost 1.0 platform)

Ablate style disagree boost **2 vs 1** before raising it. Promote on log-loss
and top-1 / disagree; pawn gap is the safety check. User bots already use
higher boost (Phase 3); this slice is **platform baseline** weights only.

| Arm | Boost |
| --- | ---: |
| `boost_2` | 2.0 (current default / control) |
| `boost_1` | 1.0 (no SF-disagree upweight) |

Cold start both on the same train sample; same epochs/batch/LR. CLI:
`scripts/tools/offline_disagree_boost_ab.py`.

### Mid-size result (decision) — 2026-11-08

`storage/tmp/disagree_boost_ab_mid.json` — train 30 029, val 8 006, epochs 3,
scale=2.0 pawns. Cold start both arms.

| Arm | top-1 | agree | disagree | val_log_loss | pawn gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| boost_2 (control) | 0.415 | 0.748 | **0.196** | 2.047 | -0.16 |
| **boost_1** | **0.424** | **0.793** | 0.182 | **1.992** | **-0.11** |

Deltas (`boost_1` − `boost_2`): top1 **+0.96**, disagree **−1.4**, LL **−0.055**,
pawn gap better.

**Verdict:** do **not** raise boost above 2. For **platform** baseline, prefer
**boost 1.0** — clearer overall / LL / pawn-gap win. Boost 2 still buys some
disagree top-1 (style), so keep high boost on **user** finetune (Phase 3 already
uses ~4). Pending: change `DEFAULT_STYLE_DISAGREE_BOOST` for platform path only
if we accept trading a bit of disagree for better overall.

### Open product question (was Slice B note → now Slice D)

Baseline → personal child may be the wrong long-term shape if the **only**
goal is strong personalized play. Shared trunk helps cold-start / multi-user
ops, but can cap how “weird” a single human the child becomes. Slice C
rejected freeze trunk and resurfaced the serve-parent vs FT trade. That debate
is now **Slice D** (draft below) — not a full architecture redesign.

## Slice C — Personal head — REJECT freeze trunk (2026-11-11)

Freeze shallow conv trunk; train state tower + candidate scorer per user.
Pin each personal model to a **named parent** baseline. Do not restart a good
personal model from a brand-new platform parent without an explicit migrate.

| Arm | Change |
| --- | --- |
| `full_ft` | Control — all layers trainable (today’s user FT) |
| `freeze_trunk` | Freeze `board_conv1/2`, `board_gap`, `board_emb` |

Shared cold parent on platform registry-train; both arms warm on
account ∩ registry-train (default: ikbendaniel); score account ∩ registry-val.
User FT boost stays ~4. Warm path pinned to product control (`Adam 1e-3`, no
warmup). Trainer flag: `HybridBoardTrainer(freeze_board_trunk=…)`.
CLI: `scripts/tools/offline_personal_head_ab.py`.

### Mid-size result (decision) — 2026-11-11

`storage/tmp/personal_head_ab_mid.json` — parent 30 029 / user train 15 028 /
user val 4 024, parent 3ep, warm 3ep, user boost 4.0.

| Arm | top-1 | agree | disagree | val_log_loss | pawn gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| parent (on user val) | **0.413** | — | 0.200 | **2.096** | **-0.18** |
| **full_ft** (control) | **0.399** | **0.695** | **0.228** | **2.102** | **-0.59** |
| freeze_trunk | 0.395 | 0.692 | 0.223 | 2.102 | -0.62 |

Deltas (`freeze_trunk` − `full_ft`): top1 **−0.45**, disagree **−0.55**, LL
**+0.000**, pawn gap worse.

**Verdict:** reject freeze trunk. Misses gate vs `full_ft`. Keep full-layer
user FT. Flag stays for future ablations. Side note: both FT arms lose overall
top-1 / pawn gap vs parent on this user val (disagree rises) — that trade is
**Slice D**, not a freeze win.

## Slice D — Serve parent vs personal FT — DONE (2026-11-12)
**Goal:** Decide **when** a personal child is worth **serving** instead of the
platform parent on that user’s registry-val. Not “can we train a child,” but
“does this child beat serving the parent under an explicit promote/serve rule.”

### Why this slice

Slice C mid (`personal_head_ab_mid.json`) on ikbendaniel:

| Model | overall top-1 | SF-disagree top-1 | val LL | pawn gap |
| --- | ---: | ---: | ---: | ---: |
| parent @ user val | **0.413** | 0.200 | **2.096** | **-0.18** |
| full_ft @ boost 4 | 0.399 | **0.228** | 2.102 | -0.59 |

FT bought style (disagree **+2.8 pts**) and paid strength (overall **−1.4**,
pawn gap much worse). Today’s product path still trains/serves personal FT at
boost ~4. Slice D measures that trade across boosts and drafts a **serve gate**
so we do not promote a child that only looks “more personal” on disagree.

Out of scope here: freeze trunk (rejected), Maia, train-from-scratch personal
models, inference logit blend (stays backlog — related, different surface).

### Design (one change family: user FT intensity)

Shared cold parent on platform registry-train (same recipe as Slice C).
Warm arms on one account ∩ registry-train; score the same account ∩
registry-val. Warm path = product control (`Adam 1e-3`, last epoch, no
freeze, no soft warmup). Only **style disagree boost** (and serve-parent)
differs.

| Arm | What |
| --- | --- |
| `serve_parent` | No FT — score parent weights on user val (reference) |
| `ft_boost_1` | Full-layer user FT, style boost **1.0** |
| `ft_boost_2` | Full-layer user FT, style boost **2.0** |
| `ft_boost_4` | Full-layer user FT, style boost **4.0** (today / control child) |

Matched warm epoch budget (default mid: 3). Optional later **only if** boost
ablate is inconclusive or the trade looks epoch-starved:

| Follow-up arm | What |
| --- | --- |
| `ft_boost_4_2round` | Second warm round on same user train (catch-up-ish) |

Do **not** start with multi-round; mid boost grid first.

Default subject: ikbendaniel. If mid is decisive, optional cheap confirm on a
second account (e.g. RebeccaHarris) at same limits — still not full-bucket.

### Metrics and proposed serve gate

Report every arm vs `serve_parent` on user val:

- Personalization: `Δ top1_sf_disagree`, also agree (report-only)
- Strength / safety: `Δ top1_overall`, `Δ val_log_loss`, `Δ sf_delta_mean_pawns`

**Draft serve rule** (tune after mid; do not hard-code into prod until approved):

Serve / promote child only if **all** hold vs parent on user val:

1. Disagree top-1 lift ≥ **+1.0** pt (personalization floor)
2. Overall top-1 drop ≤ **1.0** pt (strength ceiling)
3. Pawn-gap not worse by more than **0.20** pawns (safety)
4. Log-loss not clearly worse (no hard number until mid; flag if ΔLL > ~0.03)

If no FT arm passes, verdict = **prefer serve parent** for that user/budget
and keep training optional for research only.

### Offline shape

- Registry train/val only; no test; no `ml.baseline_models` writes
- Mid decision target: parent ~30k, user train ~15k, user val ~4k, 3 warm ep
  (reuse Slice C sizes for comparability)
- Smoke: parent ~2k, user ~1.5k, val ~500, 2 ep
- CLI: `scripts/tools/offline_serve_gate_ab.py`
- Reuse `HybridBoardTrainer` + account∩registry loaders; no new encoder
- Gate helper: `evaluate_serve_gate` in `offline_serve_gate_ab.py`

### Deliverables

1. Plan approval (this section) ✅
2. Harness + unit tests ✅
3. Mid JSON ✅ `storage/tmp/serve_gate_ab_mid.json`

### Mid-size result (decision) — 2026-11-12

`storage/tmp/serve_gate_ab_mid.json` — parent 30 029 / user train 15 028 /
user val 4 024, parent 3ep, warm 3ep.

| Arm | top-1 | disagree | val_log_loss | pawn gap | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| serve_parent | 0.385 | 0.177 | 2.212 | **-0.20** | ref |
| ft_boost_1 | **0.390** | 0.180 | **2.174** | -0.24 | FAIL (disagree +0.35 pt) |
| **ft_boost_2** | 0.377 | 0.189 | 2.227 | -0.38 | **PASS** |
| ft_boost_4 (today) | 0.357 | **0.198** | 2.306 | -0.61 | FAIL (overall −2.8, pawn 0.41, ll_warn) |

**Verdict:** today’s user boost **4 is too aggressive** under the draft serve
gate. **Boost 2** is the only FT arm that passes (barely on pawn regress
0.185 ≤ 0.20). Boost 1 is the healthiest strength/LL model but under-delivers
personalization vs the +1 pt disagree floor. Prefer moving personal FT toward
**boost ~2** + serve gate before promoting; do **not** keep serving boost-4
children by default. Gate numbers stay draft until product accept.

### Explicit non-goals (Slice D)

- Do not reopen freeze trunk or Maia
- Do not change platform `DEFAULT_STYLE_DISAGREE_BOOST` here (Slice B pending)
- Do not commit unless asked

## Critical review — Phase 5 A–D (this thread)

| Slice | Knob | Outcome | What it taught |
| --- | --- | --- | --- |
| A | Soft warm LR / best-val / replay | **REJECT** — control `1e-3` wins | Do not slow warm updates; soft rule hurts human policy |
| B | Platform disagree boost 2 vs 1 | Prefer **boost 1.0** platform | Disagree upweight trades overall/LL/pawn; keep high boost only on user path |
| C | Freeze board trunk vs full FT | **REJECT** freeze | Trunk still wants gradient; freeze ≠ personalization |
| D | Serve parent vs FT boost 1/2/4 | Boost **2** passes gate; **4** fails hard | Serving pure boost-4 child is a style-for-strength tax; gate is real |

Cross-cutting: every “more personal / more disagree weight” lever so far
buys SF-disagree top-1 and spends overall accuracy + pawn gap. Architecture
(Maia / freeze) did not fix that. Intensity + **serve decision** did.

## Slice E — Inference blend — DONE (2026-11-12)

**Goal:** Soften the Slice D trade with logit blend
`(1-α)·parent + α·child` at inference. Child = today’s aggressive FT
(**boost 4**), because that is the path blend must redeem if we keep training
it; pure α=1 failed the serve gate.

| Arm | α (child weight) |
| --- | ---: |
| `serve_parent` | 0.0 |
| `blend_0.25` | 0.25 |
| `blend_0.50` | 0.50 |
| `blend_0.75` | 0.75 |
| `serve_child` | 1.0 |

CLI: `scripts/tools/offline_inference_blend_ab.py`. Scorer:
`score_hybrid_logit_blend`. Same mid sizes / serve gate as D.

### Mid-size result (decision) — 2026-11-12

`storage/tmp/inference_blend_ab_mid.json` — parent 30 029 / user 15 028 /
val 4 024, child boost 4.0, 3+3 ep.

| Arm | top-1 | disagree | val_log_loss | pawn gap | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| serve_parent | **0.384** | 0.175 | **2.212** | **-0.19** | ref |
| blend_0.25 | 0.382 | 0.179 | 2.223 | -0.24 | FAIL (disagree +0.31 pt) |
| **blend_0.50** | 0.379 | 0.188 | 2.241 | -0.31 | **PASS** |
| blend_0.75 | 0.370 | 0.192 | 2.269 | -0.43 | FAIL (overall −1.5, pawn 0.24) |
| serve_child | 0.358 | **0.198** | 2.305 | -0.60 | FAIL (overall −2.6, pawn 0.41) |

**Verdict:** logit blend **α=0.5** redeems a boost-4 child under the draft
serve gate (disagree +1.3 pt, overall drop 0.55 pt, pawn regress 0.12). Pure
child and α≥0.75 still fail. vs Slice D’s pure **FT@boost2** (also PASS):
blend@0.5 looks similar disagree with a milder pawn hit on this run — not a
matched A/B (different parent seed). **Product chose Option B** (FT@boost2 +
serve gate); blend parked.

## Product choice — Option B (2026-11-12)

**Locked:** train personal FT at style disagree boost **2.0**; promote/serve
child only when **serve gate** passes vs parent on user registry-val; else keep
parent. Do not serve pure boost-4. Inference blend (Slice E) parked.

Shipped:
- `DEFAULT_USER_STYLE_DISAGREE_BOOST = 2.0` (`ply_weights` / user FT CLIs)
- `ModelTraining(user_id).fit` uses that boost
- `ModelTraining(user_id).decide_promotion` uses `serve_gate.py`
- Offline `scripts/ops/offline_user_promotion.py` prints gate + recommend

## Still later (Phase 5 backlog)

- Platform DEFAULT boost → 1.0 (Slice B accept)
- Optional: game-count blend α if we revisit Slice E
- “Recent opening” copy when recency weights apply
- E16 / E18 orchestration parity

## Locked

| Topic | Choice |
|-------|--------|
| Encoder | Shallow hybrid (`board_trunk=shallow`). Maia residual / dual / spatial parked |
| Splits | Hash registry `baseline-v1` only |
| Loss default | `sf_mix` @ `α=0` |
| Distill target | Parent **human** policy, never Stockfish |

## Agent rules

- Ask before large ambiguous changes.
- One change family per experiment (Slice A alone first).
- Doppler `dev_local` + `.venv` — never `uv`.
- Do not commit unless asked.
- Do not reopen Maia trunk / dual-head without new evidence + explicit ask.
