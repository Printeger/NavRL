# instinctRL Decision Log

> **Created**: 2026-07-04  
> **Purpose**: Record architectural decisions made during grilling sessions.

---

## D-2026-07-05-001: instinctRL-C Anchor Manager Boundary and Semantics

**Decision**: Mark instinctRL-C complete as an actor-clean measurement-space anchor manager with passive env diagnostics only. Do not implement reward integration, B3 ablation, observability logging, ICS, or training convergence in C.

**Public env boundary**:

- Add only scalar anchor diagnostics to `info`: `anchor_active`, `anchor_loss`, `anchor_valid_fraction`, `anchor_error_mean`, `anchor_error_max`, `anchor_hold_steps`, `anchor_activation_count`, `anchor_reset_reason`.
- Keep dense `anchor_error`, `usable_anchor_mask`, `r_star`, `m_star`, and `w_star` in internal runtime cache only.
- Actor observation remains `lidar_grid` + `state_vec`; no anchor tensors enter actor `state_vec` or `lidar_grid`.

**Locked semantics**:

- Reset enum: `0 none`, `1 episode`, `2 explicit`, `3 command`, `4 invalid`.
- Reset priority: `episode > explicit > command > invalid > none`.
- Capture when inactive and `||v_cmd|| <= eps_enter`; command reset when active and `||v_cmd|| >= eps_exit`; require `eps_enter < eps_exit`.
- `anchor_activation_count` is per-episode cumulative and resets only on episode reset.
- `anchor_hold_steps` is an integer step counter: `0` inactive, `1` on capture step, increments while active, resets to `0` on any reset.
- Store `r_star`, bool `m_star`, and `w_star` at capture. `w_star` gates usability diagnostics but does not multiply `anchor_error`.
- `anchor_valid_fraction = sum(structural_mask & m_t & m_star & (w_t > 0) & (w_star > 0)) / sum(structural_mask)`.
- `anchor_error = m_t_float * m_star_float * w_t * (r_t - r_star)`.
- `anchor_loss` uses per-beam standard Huber reduced over the fixed structural denominator, not over `sum(usable_anchor_mask)`.
- Public metrics describe post-transition state. A reset step reports inactive public metrics and the selected reset reason.

**Validation policy**: `MeasurementSpaceAnchorManager.step()` fails fast on bad shapes/devices/dtypes/non-finite tensors, except it accepts `v_cmd` as `[N,3]` or `[N,1,3]`. Finite weights are clamped to `[0,1]`.

**Config policy**: canonical key is `instinctRL.anchor.min_valid_anchor_fraction`; `min_valid_fraction` is rejected. Threshold must satisfy `0.0 < min_valid_anchor_fraction <= 1.0`.

**Rationale**: C establishes the stable anchor lifecycle and diagnostics required by later reward, ablation, and evaluation work while preserving the actor-clean contract accepted in instinctRL-B.

**Consequence**: instinctRL-D may start next. instinctRL-E/F remain no-go until their own stage scopes are opened.

---

## D-2026-07-04-016: instinctRL-B Complete, instinctRL-C May Start

**Decision**: Mark instinctRL-B complete and allow instinctRL-C to start.

**Rationale**: The B-fix implementation is complete, NavRL pytest/PPO validation passes (`14 passed`), active RayCaster uses the MID360 helper rather than `BpearlPatternCfg`, actor input is restricted to `lidar_grid` and `state_vec`, previous issued action is fed into history, and user-side GPU smoke completed 500/500 steps with actor/schema/action audits, PPO hybrid forward, MID360 raw range `[4, 1, 360, 59]`, valid returns `28.62%`, and `Observation smoke path PASSED`. Smoke mode exits before `SimulationApp.close()` after success to avoid Isaac Kit teardown segfaults after validation.

**Consequence**: instinctRL-C is `GO`. C work must remain scoped to the handbook measurement-space anchor manager and must not re-open unrelated B architecture unless a new C-blocking defect is found.

---

## D-2026-07-04-015: NavRL Unit/PPO Validation Passed, Superseded

**Decision**: Superseded by D-2026-07-04-016. This earlier decision accepted NavRL pytest/PPO validation but still required GPU runtime smoke before C.

**Rationale**: Running through `conda activate NavRL` resolves the earlier apparent dependency problems: Click, Hydra, TorchRL, TensorDict, and `ForkingPickler` are available on the activated Isaac Sim Python path. The PPO hybrid test then exposed a real code bug: critic-only privileged fields had shape `[N,1,D]` and were concatenated with `_actor_feature` `[N,256]` without flattening. After flattening those critic fields, the B unit/PPO test set passes (`13 passed`). The remaining failed command reaches `train.py` CUDA preflight and stops because no CUDA-capable device is visible; `nvidia-smi` also cannot communicate with the NVIDIA driver.

**Consequence**: Superseded. Later user-side GPU smoke exercised the real RayCaster/runtime path and passed the B checks.

---

## D-2026-07-04-014: B-Fix Implementation Does Not Yet Authorize instinctRL-C

**Decision**: Superseded by D-2026-07-04-016. Keep instinctRL-C blocked after the B-fix implementation pass until runtime validation succeeds.

**Rationale**: Superseded in part by D-2026-07-04-015. The code blockers found during closeout have been addressed: active instinctRL RayCaster wiring no longer uses `BpearlPatternCfg`, body-to-world adapter semantics are corrected and unit-tested, previous issued action is fed into the observation builder, actor schema audit exists, and `instinctRL.mode` separates smoke from train. Later NavRL validation proved pytest/PPO now pass; only Isaac runtime smoke remains blocked by GPU visibility.

**Consequence**: Superseded. B runtime smoke passed; C is now GO.

---

## D-2026-07-04-013: A/B Closeout Blocks instinctRL-C

**Decision**: Do not start instinctRL-C until the B-fix checklist passes. instinctRL-A is accepted only as B0 smoke-test / infrastructure baseline, not learning success. instinctRL-B is partial only, not fully accepted.

**Rationale**: The handbook requires Observation / History Buffer acceptance to include MID360 range/mask/weights, timestamps, previous output, history, stable ray ordering, and tests. Current code has a real observation builder and hybrid PPO input, but active `env.py` still uses `patterns.BpearlPatternCfg`, the training path returns after B0 smoke when `instinctRL.enabled=true`, `prev_action` is not wired from the issued governor/controller output, and actor audit scans key names rather than `state_vec` provenance.

**Consequence**: The current stage is `B-closeout / B-fix before instinctRL-C`. Any older devlog entry saying instinctRL-B is complete or "proceed to instinctRL-C" is superseded by this decision.

---

## D-2026-07-04-001: B0 Minimal Governor (α=1, v_corr=0) in instinctRL-A

**Decision**: Implement only the minimal B0 governor (α=1, v_corr=0, v_gov=v_cmd) in instinctRL-A. Defer trainable governor head (α, v_corr) to instinctRL-A2 or instinctRL-F.

**Alternatives considered**:
- A. Minimal B0 (chosen) — simplest path, validates command infrastructure
- B. Full trainable governor in instinctRL-A — premature; requires observation buffer and reward integration

**Rationale**: The goal of instinctRL-A is to establish the clean body-frame velocity command path and baseline. The trainable governor is a learning component that needs the observation space (instinctRL-B) and reward terms (instinctRL-F) to be stable first.

**Registered as**: D-001 in DEFERRED_REGISTER.md

---

## D-2026-07-04-002: Fixed + Simple Random v_cmd for B0

**Decision**: Use fixed low-speed body-frame command + simple bounded random generator for B0 smoke test. Do not use adversarial command generator in instinctRL-A.

**Alternatives considered**:
- A. Fixed + simple random (chosen) — validates command path without complexity
- B. AdversarialCommandGenerator — inappropriate for baseline; belongs to ICS/evaluation stages
- C. Fixed only — too limited to verify multi-axis behavior

**Rationale**: Adversarial/aggressive commands test safety boundaries that don't exist yet (ICS deferred to instinctRL-E). Simple random commands exercise the full 3-DOF body-frame velocity interface.

**Registered as**: D-006 in DEFERRED_REGISTER.md

---

## D-2026-07-04-003: BodyToWorldVelocityAdapter Created and Wired Immediately

**Decision**: Create and wire `BodyToWorldVelocityAdapter` in instinctRL-A. It must be used immediately in the B0 smoke test, not left as dead code.

**Alternatives considered**:
- A. Create + wire now (chosen)
- B. Defer again — would violate the decision from instinctRL-0 grilling

**Rationale**: The adapter was deferred once (instinctRL-0). instinctRL-A's B0 path needs body→world transform to feed VelController. The adapter reads privileged drone quaternion from `info["drone_state"]` — this quaternion never enters actor input.

---

## D-2026-07-04-004: Basic MID360 Attachment in instinctRL-A

**Decision**: instinctRL-A runs on TASLAB_UAV + MID360 configuration (not Bpearl/generic). A provides only basic MID360 attachment and raw range tensor. Full preprocessing deferred to instinctRL-B.

**Alternatives considered**:
- A. Basic attachment only (chosen) — satisfies B0 "runs on MID360" acceptance criterion
- B. Full preprocessing now — too large for A; would duplicate instinctRL-B scope

**Rationale**: "B0 runs on TASLAB/MID360" means the sensor is attached and producing range data, not that all preprocessing is complete. Raw range tensor is sufficient for B0 smoke test.

**Registered as**: D-002 in DEFERRED_REGISTER.md

---

## D-2026-07-04-005: Staged Audit (Env-Construction Only) in instinctRL-A

**Decision**: Implement only minimal audit checks in instinctRL-A: actor input scan, platform lock, action type. Full rollout/eval/checkpoint/ROS hooks deferred.

**Alternatives considered**:
- A. Staged (chosen) — validates the critical checks at the right checkpoint
- B. Full audit now — premature; training pipeline not stable

**Rationale**: The env-construction audit catches the most critical violation: forbidden fields in actor observation. Full hooks require the training and deployment pipeline to exist.

**Registered as**: D-008 in DEFERRED_REGISTER.md

---

## D-2026-07-04-007: Full MID360 Pattern Wiring in instinctRL-B

**Decision**: Use full LivoxMid360Pattern / MID360 ray ordering. Do NOT use BpearlPatternCfg as a substitute.

**Rationale**: B0 in instinctRL-A only needed basic MID360 availability. instinctRL-B must make the observation pipeline platform-correct.

---

## D-2026-07-04-008: Config-Gated Noise and Dropout

**Decision**: Add `enable_noise` and `enable_dropout` config switches, default OFF. Defer noise/dropout training curriculum to later stage.

**Rationale**: Deterministic mode needed for unit tests and initial evaluation. Noise adds realism but complicates debugging.

**Registered as**: D-009

---

## D-2026-07-04-009: Staleness-Weighted Reliability

**Decision**: Default reliability weights as $w_t = m_t \cdot \exp(-\text{age}/\tau)$. Fall back to binary $w_t=m_t$ if age unavailable. Defer neighbor-consistency weighting.

**Rationale**: Staleness captures the most important reliability signal (fresh data is more trustworthy) without the complexity of neighbor comparison.

**Registered as**: D-010

---

## D-2026-07-04-010: Configurable History Buffer (L=4 default)

**Decision**: Default history_len=4, configurable via Hydra for 8/16-frame ablations.

**Rationale**: 4 frames is minimal viable for short-term motion inference. Config allows ablation experiments without code changes.

**Registered as**: D-011

---

## D-2026-07-04-011: IMU Cues from Drone State

**Decision**: Derive allowed IMU cues (body angular velocity + gravity direction) from privileged drone_state. Defer real ROS IMU / simulated IMU sensor integration.

**Rationale**: Drone state provides the same physical quantities an IMU would measure. Real sensor integration is a deployment concern.

**Registered as**: D-012 in DEFERRED_REGISTER.md

---

## D-2026-07-04-012: Hybrid Observation Format

**Decision**: Use hybrid format: `lidar_grid` [N, C, H, V] for spatial data + `state_vec` [N, D] for low-dimensional cues. Do NOT flatten grid into a single vector.

**Rationale**: CNN processes spatial structure efficiently. State vector avoids tiling fake image channels.

---

## D-2026-07-04-006: B0 Smoke Test Definition

**Decision**: "B0 runs" means: env resets, TASLAB_UAV spawns, MID360 attaches, fixed v_cmd generated, governor + adapter produce valid world-frame velocity, VelController executes, simulator advances N steps without crash/NaN, all audits pass. It does NOT require training convergence, WandB logging, or performance metrics.

**Alternatives considered**:
- A. Smoke test (chosen) — validates infrastructure without requiring learning
- B. Full training run — inappropriate for a baseline that doesn't learn
- C. Single eval episode — too narrow; doesn't test env reset or multi-step stability

**Rationale**: B0 is a baseline, not a learned policy. The smoke test proves the command path works end-to-end. Learning comes later.
