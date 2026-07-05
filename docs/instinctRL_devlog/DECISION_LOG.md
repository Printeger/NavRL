# instinctRL Decision Log

> **Created**: 2026-07-04  
> **Purpose**: Record architectural decisions made during grilling sessions.

---

## D-2026-07-05-004: instinctRL-F Reward Integration Boundary and Semantics

**Decision**: Mark instinctRL-F complete for reward integration/readiness only. Do not claim trainable-governor readiness, stable training, or learned-policy success from this stage.

**Public env boundary**:

- Add reward component accumulators to `stats`, not actor observation.
- Preserve actor observation as `lidar_grid` + `state_vec`.
- Keep privileged simulator quantities reward/critic/eval/logging only.
- Preserve the old NavRL reward path when `instinctRL.reward.enabled=false`.

**Locked semantics**:

- Default tracking uses an actor-clean command-consistency proxy between `v_cmd_b` and the issued/final body command proxy.
- Optional actual velocity is reward-only and disabled by default with `use_privileged_velocity_for_reward=false`.
- Anchor reward is active only when anchor is active and `anchor_valid_fraction >= min_anchor_valid_fraction`.
- Safety reward uses latest MID360 clearance, not map, odometry, SLAM, surface normals, or dynamic-obstacle privileged state.
- ICS compliance offsets tracking penalty when beta/emergency says attenuation was necessary.
- Intervention penalty discourages reliance on low beta.
- Reward components are proportionally scaled when total clipping is active so logged components sum to `reward_total`.

**Validation**: F reward unit tests and A/B/C/D/E/F regression tests pass in the activated NavRL conda environment. Runtime Isaac smoke with `instinctRL.reward.enabled=true` was skipped locally because CUDA/NVML is not visible here.

**Consequence**: instinctRL-G may start for baseline/evaluation harness work. Training convergence and learned-governor success remain not proven.

---

## D-2026-07-05-003: instinctRL-E Attenuation Boundary and Semantics

**Decision**: Mark instinctRL-E complete as a deployed-safe, actor-clean command attenuation layer. E attenuates the body-frame governor command and does not add rewards, training changes, actor observation fields, or offline observability plotting.

**Public env boundary**:

- Add only scalar `ics_*` diagnostics to `info` when `instinctRL.ics.enabled=true`.
- Keep dense active masks, per-beam speeds, range-rate estimates, margins, and effective clearances in cache/debug only.
- Actor observation remains `lidar_grid` + `state_vec`.

**Locked semantics**:

- Formula is `v_final_b = beta * v_gov_b + (1 - beta) * v_brake_b`.
- E first pass accepts only `brake_mode="zero"`, so `v_brake_b=0` and unsupported brake modes fail config validation.
- Diagnostics and beta use the unclipped `v_gov_b`; the final body command is norm-clipped to `velocity_limit`.
- Inputs are limited to MID360 range/mask/weight history, body-frame ray directions, body-frame governor command, optional history dt, and config parameters.
- Active beams require latest valid mask, reliability above threshold, closing evidence, and clearance within the active horizon.
- Default closing evidence uses the governor-command approach component only.
- Optional range-rate filter can activate/use negative range rate, but range-rate remains cache-only unless enabled.
- Empty active set uses `empty_active_set_beta`, default 1.0.
- Reliable emergency clearance below threshold forces beta 0.

**Validation**: E unit tests and A/B/C/D/E regression tests pass in the activated NavRL conda environment. Runtime Isaac smoke with `instinctRL.ics.enabled=true` was skipped locally because CUDA/NVML is not visible here.

**Consequence**: instinctRL-F may start for reward-design work. Training convergence remains not complete and must not be claimed from E acceptance.

---

## D-2026-07-05-002: instinctRL-D Observability Logger Boundary and Semantics

**Decision**: Mark instinctRL-D complete as an evaluation-only range-Jacobian observability logger. The logger must not become a deployed control dependency and must not add observability features to actor input.

**Public env boundary**:

- Add only scalar `observability_*` diagnostics to `info` when `instinctRL.observability.enabled=true`.
- Keep dense `jacobian_rows`, `weighted_jacobian_rows`, singular values, weak direction, normals, and finite-difference internals in cache/debug only.
- Actor observation remains `lidar_grid` + `state_vec`.

**Locked semantics**:

- Canonical API uses flat ray layout: ray directions `[R,3]` or `[N,R,3]`, masks/weights `[N,R]`.
- Mode codes: `0 proxy`, `1 normal`, `2 finite_difference`.
- Proxy mode uses `J_i=-u_i^T` from normalized body-frame ray directions and sets `observability_is_proxy=1`.
- Normal mode uses `J_i=-n_i^T` from normalized body-frame surface normals, with `sqrt(w_i)` row scaling.
- Finite-difference mode solves `DeltaP @ j_i ~= Delta r_i` using `pinv(DeltaP) @ Delta r_i`, with no extra sign.
- Offline mode priority is finite-difference, then normals, then proxy fallback. Malformed supplied FD/normals fail fast.
- SVD uses one `torch.linalg.svd` per env on weighted effective rows; insufficient rows return rank 0, zero sigmas, capped condition, zero score, and zero weak direction.
- Drift projection is absolute projection onto normalized weak direction; weak direction is cache-only.

**Validation**: D unit tests and A/B/C/D regression tests pass in the activated NavRL conda environment.

**Consequence**: instinctRL-E may start. instinctRL-F remains no-go until E and reward prerequisites are complete.

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
