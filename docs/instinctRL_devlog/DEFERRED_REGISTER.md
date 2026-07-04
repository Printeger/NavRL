# instinctRL Deferred Item Register

> **Created**: 2026-07-04 (instinctRL-A)  
> **Last Updated**: 2026-07-05 (instinctRL-C complete)
> **Purpose**: Track all items intentionally deferred from current or past stages.  
> **Rule**: Before starting any future stage, read this register and handle all items assigned to that stage. Each item must be implemented, explicitly carried forward, marked blocked, or marked obsolete. Do not silently ignore open items.

---

## Before-D Validation Blockers

None. instinctRL-C acceptance blockers are cleared; instinctRL-D may start.

Completed C validation evidence is recorded in `TEST_PLAN.md`, `tickets/instinctRL-C_measurement_space_anchor.md`, and `tests/instinctRL-C_test_report.md`.

---

## Active Deferred Items

### D-001: Trainable Governor Head (α, v_corr)

| Field | Value |
|-------|-------|
| **Deferred from** | instinctRL-A |
| **Reason** | B0 minimal (α=1, v_corr=0) is sufficient for direct velocity pass-through baseline. Full trainable governor requires observation buffer (instinctRL-B) and reward integration (instinctRL-F) to be meaningful. |
| **Target stage** | instinctRL-A2 or before instinctRL-F |
| **Trigger condition** | Observation buffer (r, m, w, IMU, history) available; actor input schema stable |
| **Acceptance test** | Governor outputs bounded α∈[0,1] and bounded v_corr; α+v_corr form deterministic v_gov; no critic-to-actor leakage; export-compatible; deterministic mean action |
| **Status** | ⬜ Open |
| **Module ref** | Handbook M3 (`instinctRL/governor.py` — replace `MinimalGovernor` with trainable version) |

### D-002: Full MID360 Raw Range / Mask / Weight / History Preprocessing

| Field | Value |
|-------|-------|
| **Deferred from** | instinctRL-A (basic range only), instinctRL-0 (mask/weights/timestamps never implemented) |
| **Reason** | instinctRL-A only needs basic MID360 attachment and raw range tensor. Full preprocessing (valid-return mask m_t, reliability weights w_t, timestamps, history buffer) is a complex subsystem that deserves its own dedicated stage. |
| **Target stage** | instinctRL-B |
| **Trigger condition** | instinctRL-A complete; B0 smoke test passes |
| **Acceptance test** | Stable ray count/ordering; mask derivation from finite in-range returns; dropout handling; reliability bounds [0,1]; timestamp monotonicity; history rollover; stale-frame markers; no actor input leaks |
| **Status** | ✅ Complete |
| **Module ref** | Handbook M1 (`instinctRL/observation.py`) |

### D-003: Measurement-Space Anchor Manager

| Field | Value |
|-------|-------|
| **Deferred from** | instinctRL-A (out of scope) |
| **Reason** | Anchor manager requires MID360 r, m, w from instinctRL-B. No existing anchor lifecycle exists. |
| **Target stage** | instinctRL-C |
| **Trigger condition** | instinctRL-B complete; r, m, w available |
| **Acceptance test** | Null-command hysteresis (ε₀<ε₁); anchor capture on rising edge; masked anchor error e^r=(m⊙m*⊙w)⊙(r−r*); reset on episode/large-command/insufficient-valid-fraction; anchor loss active only under null command; scalar diagnostics in info; actor obs remains clean |
| **Status** | ✅ Complete |
| **Module ref** | Handbook M2 (`instinctRL/anchor.py`) |

**Closeout correction**: C intentionally did not implement anchor reward or B3 ablation execution. Those remain reward/baseline work, not C acceptance blockers for the manager/passive diagnostics ticket.

### D-004: Range-Jacobian Observability Logger

| Field | Value |
|-------|-------|
| **Deferred from** | instinctRL-A (out of scope) |
| **Reason** | Evaluation/analysis module, not a deployed control dependency. Requires RayCaster scene access and surface normal computation (simulation-only, not for deployed path). |
| **Target stage** | instinctRL-D |
| **Trigger condition** | instinctRL-C complete; anchor lifecycle available for drift correlation |
| **Acceptance test** | Produces drift vs σ_min(J) plots; per-scenario drift ranking; weak-direction drift alignment; hardware proxy labeled as proxy (not exact J); no surface normals in deployed code |
| **Status** | ⬜ Open |
| **Module ref** | Handbook M5 (`instinctRL/observability.py`) |

### D-005: ICS-Inspired Command Attenuation (β_t)

| Field | Value |
|-------|-------|
| **Deferred from** | instinctRL-A (out of scope) |
| **Reason** | ICS requires range-rate fitting from MID360 history (D-002) and v_safe(d) braking distance model. B0 does not need command attenuation. |
| **Target stage** | instinctRL-E |
| **Trigger condition** | instinctRL-B complete (history buffer with timestamps available) |
| **Acceptance test** | Monotonic β_t with speed/clearance; empty active set → β_t=1; emergency bypass on min-clearance; no surface-normal imports; no odometry/map access; B5 ablation exists |
| **Status** | ⬜ Open |
| **Module ref** | Handbook M4 (`instinctRL/ics.py`) |

### D-006: Aggressive / Adversarial Command Generator

| Field | Value |
|-------|-------|
| **Deferred from** | instinctRL-A (out of scope; fixed + simple random only) |
| **Reason** | Aggressive/adversarial commands test ICS attenuation and safety boundaries. They belong to later evaluation stages, not the baseline smoke test. |
| **Target stage** | instinctRL-E or instinctRL-G |
| **Trigger condition** | ICS attenuation validated (D-005) |
| **Acceptance test** | 5 modes (Normal Nav, Aggressive Step, Adversarial Suicide, Oscillation, Recovery Hover) produce expected velocity profiles; log command mode per episode |
| **Status** | ⬜ Open |
| **Module ref** | Existing `command_generator.py` (`AdversarialCommandGenerator`) — reuse, do not rewrite |

### D-007: Reward Integration (Tracking, Anchor, Safety, Intervention, Smoothness, Collision)

| Field | Value |
|-------|-------|
| **Deferred from** | instinctRL-A (out of scope) |
| **Reason** | Reward redesign requires governor head output (α, v_corr), anchor manager, and ICS β_t to compute meaningful terms. B0 does not learn. |
| **Target stage** | instinctRL-F |
| **Trigger condition** | Governor head (D-001), anchor (D-003), and ICS (D-005) available |
| **Acceptance test** | Each reward term activates under intended condition; privileged quantities stay reward/critic/eval only; no actor leakage; first stable training run |
| **Status** | ⬜ Open |
| **Module ref** | Handbook M6 (`instinctRL/rewards.py` or gated path in `env.py`) |

### D-008: Full Audit Hooks (Rollout, Evaluation, Checkpoint Export, ROS Inference)

| Field | Value |
|-------|-------|
| **Deferred from** | instinctRL-A (staged: env-construction-only checks implemented) |
| **Reason** | Full audit requires training pipeline to be stable and checkpoint/ROS paths to exist. Current minimal audit is sufficient for B0 smoke test. |
| **Target stage** | instinctRL-F or instinctRL-G |
| **Trigger condition** | Training pipeline stable; checkpoint export path defined; ROS inference path available |
| **Acceptance test** | Audit runs at: env construction ✓, policy init, rollout collection, evaluation, checkpoint export, ROS inference; fails on forbidden key patterns in all contexts |
| **Status** | ⬜ Partially open (env construction check done in instinctRL-A) |
| **Module ref** | Handbook M0 + M7 (`instinctRL/audit.py` — extend with additional hooks) |

---

## Completed / Corrected Items

### D-002: Full MID360 Raw Range / Mask / Weight / History Preprocessing
- **Closeout correction**: Fully accepted as of 2026-07-04 after B-fix implementation, NavRL pytest, and user-side GPU smoke.
- **Current fact**: `instinctRL/observation.py` provides `MID360ObservationBuilder`; active `env.py` uses the MID360 helper wrapper; `prev_action` is wired from the issued command; tests cover pattern/order, masks, weights, timestamps, history, reset, actor schema, PPO hybrid forward, and adapter frame convention.
- **Runtime evidence**: User-side smoke completed 500/500 steps with MID360 raw range `[4, 1, 360, 59]`, valid returns `28.62%`, and `Observation smoke path PASSED`.
- **Status**: ✅ Complete.

---

### D-009: Noise / Dropout Training Curriculum

| Field | Value |
|-------|-------|
| **Deferred from** | instinctRL-B |
| **Reason** | Config switches exist (`enable_noise`, `enable_dropout`) but default OFF. Noise/dropout curriculum requires training convergence first. |
| **Target stage** | instinctRL-H or later robustness stage |
| **Trigger condition** | Training pipeline stable; sim-to-real validation needed |
| **Acceptance test** | Noise+dropout enabled training matches real MID360 statistics; policy transfers without degradation |
| **Status** | ⬜ Open |

### D-010: Neighbor-Consistency Reliability Weights

| Field | Value |
|-------|-------|
| **Deferred from** | instinctRL-B |
| **Reason** | Staleness-weighted binary mask is sufficient for initial implementation. Neighbor-consistency adds complexity without proven benefit at this stage. |
| **Target stage** | instinctRL-C or later |
| **Trigger condition** | Anchor manager requires high-quality weights |
| **Acceptance test** | Neighbor-consistency weights improve anchor error vs binary mask baseline; compute 4-neighbor median deviation |
| **Status** | ⬜ Open, carried forward beyond C |

**C closeout note**: instinctRL-C stores capture-time `w_star` and uses current `w_t`, but it does not implement neighbor-consistency weighting. This is not a C blocker because C acceptance uses the existing B reliability weights.

### D-011: Longer History Ablations (8/16 frames)

| Field | Value |
|-------|-------|
| **Deferred from** | instinctRL-B |
| **Reason** | Config supports arbitrary history_len via Hydra; 4-frame default is the minimal viable. 8/16-frame ablations are evaluation experiments. |
| **Target stage** | instinctRL-G (Baselines) |
| **Trigger condition** | instinctRL-F training complete |
| **Acceptance test** | B2 ablation runs with history_len=1,4,8,16; metrics logged per config |
| **Status** | ⬜ Open |

---

## Obsolete / Cancelled Items

*(None.)*
