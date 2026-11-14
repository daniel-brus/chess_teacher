# Maia-style action plan — CLOSED

**Status:** CLOSED (2026-11-03). Last Maia board attempt done. Keep **shallow hybrid**.
**Outcome:** Steps 0 / 0b / 0c all rejected. Overnight trunk sweep confirms.
**Next:** Phase 5 — hybrid update rule. See `.agents/docs/ml-phase5-hybrid-update-rule.md`
and `.agents/docs/ml-training-roadmap.md` § Current workflow.

Goal was a model that mimics the move a person played. Offline A/Bs read registry
**train** and **val** only (no `ml.baseline_models` writes, no test bucket).

Decision gate (all steps): about **+2 top-1** or a clear drop in `val_log_loss`,
with mean pawn gap no worse. Last-epoch scores. Cold start.

## Step 0 — hybrid vs small spatial policy — REJECTED (2026-10-27)

| Arm | What |
| --- | --- |
| `hybrid` | Shallow board conv + state + SF candidates. Control. |
| `spatial` | Board-only 8x8x73 policy, six residual blocks @ 64f. No SF. |

### Full-bucket result

`storage/tmp/maia_style_ab_full.json` — train 1 023 831, val 119 651, epochs 2.

| Arm | top-1 | agree | disagree | val_log_loss | pawn gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| **hybrid** | **0.461** | **0.790** | 0.246 | **1.780** | **-0.25** |
| spatial | 0.398 | 0.528 | 0.313 | 2.019 | -1.30 |

Spatial loses every gate. Do not ship standalone spatial. Do not re-run.

## Step 0b — Maia trunk inside hybrid (Option A) — REJECTED (2026-11-02)

Swap board encoder only. Keep state + SF + 128-way head. Residual → GAP → dense
(same as shallow), not Maia 8x8x73 head.

Default research size: **64 filters × 2 residual blocks** (~1.9× params; 4GB VPS).

### Mid-size result (decision)

`storage/tmp/maia_trunk_ab_mid.json` — 30k/8k, epochs 2, 64×2.

| Arm | top-1 | agree | disagree | val_log_loss | pawn gap | params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **hybrid (shallow)** | **0.409** | **0.749** | 0.186 | **2.066** | **-0.16** | 119 041 |
| hybrid_maia_trunk | 0.387 | 0.709 | 0.176 | 2.161 | -0.19 | 230 785 |

Residual loses every gate. Keep shallow.

### Overnight trunk sweep — also reject (2026-11-03)

`storage/tmp/maia_trunk_sweep/summary.json` — mid 30k/8k, shallow@10ep reused.

| ID | Residual arm | top-1 | log_loss | vs shallow@10 (0.426 / 1.945) |
| --- | --- | ---: | ---: | --- |
| A | 64×2, 10ep, lr=1e-3 | 0.420 | 1.990 | loses |
| B | 64×2, 10ep, lr=1e-4 | 0.387 | 2.246 | loses hard |
| C | 32×3, 10ep, lr=1e-3 | 0.417 | 1.996 | loses |
| D | 64×2, 20ep, lr=1e-3 | 0.417 | 2.035 | loses |
| E | 32×3, 10ep, lr=1e-4 | 0.401 | 2.122 | loses |

No residual config beats shallow. Longer train / lower LR / size-match do not save it.

## Step 0c — dual-head (spatial + candidate mixture) — REJECTED (2026-11-03)

Hypothesis: residual flopped because trunk never saw Maia spatial CE. Shared
trunk + `α L_spatial + (1-α) L_candidate` (α=0.5). Eval = candidate head only.

### Mid-size result (decision)

`storage/tmp/maia_dual_ab_mid.json` — 30k/8k, epochs 2, 64×2, α=0.5.

| Arm | top-1 | agree | disagree | val_log_loss | pawn gap | params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **hybrid (shallow)** | **0.412** | **0.747** | 0.192 | 2.065 | -0.17 | 119 041 |
| hybrid_dual | 0.411 | 0.743 | 0.193 | **2.061** | **-0.16** | 235 530 |

Flat vs shallow (−0.09 top-1, tiny LL). Misses gate. ~2× params. No α sweep.
No full-bucket follow-up.

## Parked forever (unless new evidence)

- Colour / previous-ply planes for spatial capacity
- State MLP on pure spatial (Option B)
- Maia-2 extras (12x256, skill attention, value head)
- Promoting residual / dual-head into production

## Code left on research branch

Branch `cursor/maia-style-ab-d687` (PR #115) holds offline A/B tooling
(`offline_maia_style_ab.py`, `residual_board_trunk.py`, `dual_head_hybrid.py`,
sweep script). **Do not merge to promote Maia.** Optional later: keep mid-size
offline A/B harness only if useful for non-Maia ablations; otherwise close PR.

## Shared offline rules (still true)

- Registry train/val only; never test for these A/Bs
- Mid-size (~30k/8k) enough for trunk/head A/B; full buckets mostly packing cost
- 4GB VPS: prefer shallow; do not promote fat trunks without clear win + RAM
