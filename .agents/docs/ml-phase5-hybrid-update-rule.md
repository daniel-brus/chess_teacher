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

### Offline shape

- Same registry train/val hygiene as Phase 2 siblings
- Prefer mid-size decision runs (≥10k; ~30k/8k fine) before full buckets
- Primary: val `top1` / `val_log_loss` / `top1_sf_disagree`; pawn gap safety
- Gate vs current warm-start: clear LL drop or ≥+1–2 top-1 without pawn-gap regress

### Deliverables (Slice A)

1. Documented warm-start LR / warmup / checkpoint policy in trainer or config
2. Offline A/B: current warm-start vs update-rule on registry val
3. If win: port into baseline train path (Phase 4 entrypoints already exist)

## Slice B — Disagree boost (after A)

Ablate style disagree boost **2 vs 1** before raising it. Promote on log-loss
and top-1; pawn gap is the safety check. User bots already use higher boost
(Phase 3); this slice is **platform baseline** weights.

## Slice C — Personal head (after B)

Freeze the shallow conv trunk; train the scorer (and/or state tower) per user.
Pin each personal model to a **named parent** baseline. Do not restart a good
personal model from a brand-new platform parent without an explicit migrate.

## Still later (Phase 5 backlog)

- Inference blend by game count: `(1-α)·baseline + α·user`
- “Recent opening” copy when recency weights apply
- E16 — orchestrated train never leaks val/test into batches
- E18 — platform catch-up matches offline sibling round-by-round

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
