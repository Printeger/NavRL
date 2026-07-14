# instinctRL Decision Log

> **Created**: 2026-07-04  
> **Purpose**: Record architectural decisions made during grilling sessions.

---

## D-2026-07-11-001: A2-R3 Station Correction Repair

**Decision**: Do not promote any `20260711_111713` A2-R2 sweep candidate to 1M or formal training. Replace the hard-zero null-command interpretation with a soft, measurement-anchored station-correction path and rerun a fresh 128k A2-R3 sweep.

**Evidence**:

- All six A2-R2 candidates had `passed=false` and `safety_passed=false`.
- Best candidate was `r2_balanced` with `7/15` gates passed, but it still had `station_keeping_drift_mean=1.544`, `station_keeping_drift_p95=3.003`, `anchor_error_mean=3.243`, `null_command_speed_mean=0.196`, `command_amplification_rate=0.318`, and `termination_below_bound=0.25`.
- The hard null decoder did produce `null_command_output_speed_mean=0.0`, so the remaining station failure is not solved by making `v_corr` even harder zero.

**Locked semantics**:

- Split null-command behavior into `null-command bias` and `station correction`.
- Null-command bias remains penalized when the anchor is inactive or anchor loss is low.
- Station correction is allowed when the anchor is active, valid, and anchor loss is high.
- Keep the null-command actual-speed penalty; actual vehicle motion under `v_cmd=0` remains a station failure.
- Use soft decoder defaults: `null_vcorr_gate_enabled=true`, `null_vcorr_gate_eps=0.25`, `null_vcorr_gate_min=0.25`.
- Use A2-R3 reward defaults: `anchor_weight=4.0`, `null_command_output_weight=0.1`.
- Remove `null_command_output_speed_mean` from hard pass/fail gates; keep it as a diagnostic metric.

**A2-R3 short-sweep variants**:

- `r3_soft_null_min025`
- `r3_soft_null_min04`
- `r3_anchor_strong`
- `r3_anchor_strong_safety`
- `r3_balanced_soft`
- `r3_no_decoder_gate_reward_only`

**Screening gate before 1M**:

- `safety_collision_rate == 0`
- `termination_collision == 0`
- `termination_below_bound == 0`
- `safety_min_clearance_p05 >= 1.0`
- `station_keeping_drift_mean < 1.3`
- `station_keeping_drift_p95 < 2.6`
- `anchor_error_mean < 2.0`
- `tracking_rmse_actual_body_vs_v_cmd <= 0.45`
- `command_amplification_rate <= 0.15`

**Consequence**: `r2_balanced` is reference evidence only, not a warm start. Formal learned-governor training remains HOLD until A2-R3 short sweep selects candidates, top 1M runs pass hard gates, and multi-seed stability is checked.

---

## D-2026-07-10-001: Objective Hardening and Automated Short Sweep Gate

**Decision**: Do not continue direct 1M/2M or formal long training after the latest diagnostic failures. Harden the command-governor objective first, then use small automated 128k/256k sweeps and hard eval gates to select candidates.

**Locked semantics**:

- Safe nonzero commands must preserve command magnitude inside a band, default `0.75 <= ||v_final||/||v_cmd|| <= 1.05`.
- Preservation penalties are disabled when ICS is actively attenuating or emergency handling is active; safety intervention is allowed to reduce speed.
- Null-command behavior has a decoder-level prior: as `||v_cmd|| -> 0`, learned `v_corr -> 0` before command synthesis.
- Safety readiness is a hard gate, not a collision-only summary. Clearance p05, ICS violation rate, termination/collision, station drift, null-command motion, preservation, and amplification must be checked together.
- Sweep tooling must default to dry-run; `--execute` is required before launching train/eval jobs.

**Go/no-go gate**:

- `eval/station/handbook.station_keeping_drift_mean <= 1.0 m`
- `eval/station/handbook.station_keeping_drift_p95 <= 2.0 m`
- `eval/station/handbook.null_command_speed_mean <= 0.08 m/s`
- `eval/station/handbook.null_command_output_speed_mean <= 0.08 m/s`
- `eval/station/handbook.anchor_error_mean <= 1.0`
- `eval/tracking/handbook.tracking_rmse_actual_body_vs_v_cmd <= 0.45 m/s`
- `0.75 <= eval/tracking/handbook.command_preservation_ratio <= 1.05`
- `eval/tracking/handbook.command_amplification_mean <= 0.05`
- `eval/tracking/handbook.command_amplification_rate <= 0.10`
- `eval/handbook.safety_collision_rate == 0.0`
- `eval/handbook.safety_min_clearance_p05 >= 1.0 m`
- `eval/handbook.ics_violation_rate <= 0.005`
- below/above/collision terminations remain zero.

**Consequence**: The next experiment is an automated short corrective sweep, not formal training. Only top 2-3 candidates by hard-gate score should move to 1M, and only 1M candidates that pass across multiple seeds should move to formal longer runs.

---

## D-2026-07-09-002: Handbook Diagnostic Eval Boundary

**Decision**: Make formal short diagnostic eval a two-pass suite under static MID360-visible geometry: zero-command station keeping plus command-curriculum tracking. Enable proxy observability by default for this diagnostic suite, label it as proxy, and fail fast if dynamic obstacles are requested before they are MID360 RayCaster-visible.

**Locked semantics**:

- `env_dyn.num_obstacles=0` is required for `short_diagnostic` eval until dynamic obstacles are sensor-visible geometry.
- Station metrics use eval-only simulator pose drift and measurement-space anchor/observability diagnostics; none of these enter actor observation or reward.
- Tracking metrics use the command-governor path with curriculum probabilities initialized at the 600k-frame diagnostic stage.
- Observability proxy metrics are valid diagnostic evidence but not exact finite-difference or surface-normal range-Jacobian evidence.
- Eval success is reported through handbook metrics, not `legacy_reach_goal`.

**Consequence**: Corrected short retrain checkpoints can be compared with a consistent diagnostic eval JSON. Paper-level claims still require the full scenario and B0-B8 baseline matrix.

---

## D-2026-07-09-003: Station-First Objective Repair Gate

**Decision**: Treat the 1M static MID360 short diagnostic as a station-objective failure and repair reward/curriculum before any formal long training. Formal training now uses a station-first command curriculum, stronger measurement-space anchor weight, explicit null-command speed/output penalties, and command-amplification diagnostics.

**Locked semantics**:

- `v_corr` remains available under null command so the policy can correct observable drift, but biased null-command motion is penalized.
- Null-command training uses actual body velocity and final issued command as reward-only signals; these do not enter actor observation.
- Nonzero command training keeps actual-velocity tracking as the main reward and adds light proxy command-chain tracking plus command-amplification penalty when ICS is not attenuating.
- Short diagnostic tracking uses `diagnostic_mixed` rather than the formal training curriculum, so retrain checkpoints remain comparable.
- Long training remains HOLD until a new 1M static MID360 short diagnostic passes the A2-R gate.

**Go/no-go gate**:

- `station_keeping_drift_mean <= 1.0 m`
- `station_keeping_drift_p95 <= 2.0 m`
- `anchor_error_mean <= 1.0`
- `tracking_rmse_actual_body_vs_v_cmd <= 0.45 m/s`
- `0.75 <= command_preservation_ratio <= 1.10`
- `command_amplification_rate <= 0.10`
- `safety_collision_rate == 0.0`
- `safety_min_clearance_p05 >= 1.0 m`
- `ics_violation_rate <= 0.005`

**Consequence**: The failed 1M checkpoint is retained as diagnostic evidence only. The next training run should be a new 1M station-first diagnostic retrain, not an 8M formal run.

---

## D-2026-07-09-001: Command-Governor Train/Eval Semantic Repair

**Decision**: Classify the 8M run as a wrong-objective failed run for handbook purposes and repair train/eval semantics before any new long training. Formal instinctRL training now uses `instinctRL.task=command_governor`, actual body velocity as reward-only tracking signal, staged command curriculum, and ICS enabled by default.

**Locked semantics**:

- `reach_goal` is legacy NavRL target-navigation evidence only and is reported as `legacy_reach_goal`.
- Primary tracking reward uses privileged actual body-frame velocity versus `v_cmd_b`; the actor still receives only `lidar_grid + state_vec`.
- `v_final_b` versus `v_cmd_b` remains a command-chain diagnostic, not the primary reward objective.
- `AdversarialCommandGenerator` is wired through a conservative curriculum before aggressive/adversarial commands dominate.
- Formal method configs enable ICS attenuation by default; `no_ics` remains an ablation/debug setting.
- Streaming eval must emit handbook metrics for actual tracking, command proxy tracking, command preservation, anchor, clearance, collision, ICS intervention, and termination reasons.

**Consequence**: The next run should be a short diagnostic retrain, not another blind 8M run. Long training starts only after metrics are complete, termination reasons are interpretable, and actual tracking/anchor/safety signals are non-degenerate.

---

## D-2026-07-05-007: Learned-Governor PPO Stability Boundary

**Decision**: Treat the observed non-finite learned-governor action around 563k frames as an upstream PPO numerical-stability blocker, not a governor decoder bug. Keep the governor decoder fail-fast behavior and harden PPO rather than replacing invalid actions with zeros.

**Locked semantics**:

- Beta policy concentration parameters are bounded by config using sigmoid-mapped raw outputs.
- `("agents", "action_normalized")` must be finite before it reaches the governor decoder.
- PPO finite-audits observations, distribution parameters, actions, log-probs, entropy, value predictions, returns, advantages, losses, PPO ratio, gradients, and parameters.
- All PPO module groups are gradient-clipped with `algo.max_grad_norm`.
- Advantage normalization uses `std.clamp_min(1e-6)`.
- Target-KL early stop may skip remaining minibatches for the current update.
- Non-finite failures save compact `.pt` diagnostic snapshots and then raise.
- No training path may silently replace NaN actions with zero.

**Validation**:

- A2-S source/unit validation passes: py_compile, PPO stability tests, and A/B/C/D/E/F/A2 regression suite (`73 passed`).
- Runtime 16-frame smoke was blocked before env import by local Omniverse/Nucleus assets-root configuration, so the 1M-frame acceptance run remains pending.

**Consequence**: A2 remains complete for trainable-governor implementation. Formal long learned-governor training is on hold until the 1M-frame stability acceptance run passes. Training convergence remains unproven.

---

## D-2026-07-05-006: instinctRL-A2 Learned Governor Training Readiness

**Decision**: Make `instinctRL.mode=train` default to the learned governor actor path. Keep B0 `MinimalGovernor` and fixed/direct PPO behavior available for smoke and explicit baseline use. Treat this as formal training readiness, not convergence evidence.

**Locked semantics**:

- Actor policy action is 4D normalized: `[alpha, v_corr_x, v_corr_y, v_corr_z]`.
- `alpha` is bounded in `[0,1]`.
- `v_corr = (2 * action[1:4] - 1) * v_corr_limit`, with default `v_corr_limit=0.5 m/s`.
- `v_gov_b = alpha * v_cmd_b + v_corr`, then norm-clipped to `velocity_limit`.
- `v_cmd_b` and previous issued body command are read from actor-clean `state_vec` latest frame.
- PPO log-prob/update uses the 4D normalized governor action.
- The controller action remains 3D world-frame velocity produced by a train wrapper at the controller boundary.

**Boundary**:

- Actor observation remains exactly `lidar_grid + state_vec`.
- Actor/governor code does not read pose, odom, map, SLAM, explicit velocity, dynamic-obstacle privileged state, or `info["v_cmd"]`.
- Critic may still use privileged `info` fields, but actor/learned-governor output must not change when critic-only fields are perturbed.
- ICS remains disabled by default for the first formal learned-governor training; when enabled, it applies between `v_gov_b` and body-to-world controller adaptation.

**Deferred classification**:

- D-001 is resolved by A2.
- D-008 is resolved for first formal training requirements; ROS/H deployment audit remains deferred.
- D-006 is not a first formal training blocker and remains G/evaluation curriculum work.
- D-009/D-010/D-011 remain deferred robustness/ablation work.

**Validation**:

- A2 target tests pass: `13 passed, 5 warnings`.
- A/B/C/D/E/F+A2 regression suite passes: `64 passed, 5 warnings`.
- GPU learned-governor train smoke passes with rollout and checkpoint audits, `env_frames=16`, and final checkpoint `wandb/offline-run-20260705_203852-35lr9uce/files/checkpoint_final.pt`.

**Consequence**: A2 trainable-governor implementation is complete. This consequence is superseded for long training by D-2026-07-05-007: formal long learned-governor training is on hold until the 1M-frame stability acceptance passes. Training convergence and learned-policy success remain unproven until supported by actual training/evaluation logs.

---

## D-2026-07-05-005: instinctRL-F Minimal Training Smoke Readiness

**Decision**: Accept the minimal `instinctRL.mode=train` smoke as training-readiness evidence after fixing PPO update and train-loop smoke controls. Do not treat it as convergence evidence or learned-governor success.

**Locked smoke command**:

`python training/scripts/train.py instinctRL.mode=train instinctRL.reward.enabled=true env.num_envs=4 env_dyn.num_obstacles=0 algo.training_frame_num=4 algo.num_minibatches=1 algo.training_epoch_num=1 max_frame_num=16 eval_interval=0 save_interval=0 wandb.mode=offline headless=true`

**Rationale**:

- `algo.training_frame_num=2` with default minibatch count can create empty minibatches; tiny smoke runs must keep `env.num_envs * algo.training_frame_num >= algo.num_minibatches`.
- PPO minibatch updates must recompute `_critic_feature`; cached rollout internals are not a valid dependency for update-time critic calls.
- `eval_interval=0` is the accepted smoke setting because `i=0` otherwise triggers evaluation immediately and can allocate a long video rollout that is unrelated to the minimal training-step check.
- `save_interval=0` disables periodic saves only; the final checkpoint still verifies checkpoint write.
- Successful instinctRL train completion exits before `SimulationApp.close()` to avoid the known Isaac Kit shutdown segfault after a passed run.

**Validation**:

- Reward + PPO hybrid targeted tests pass: `12 passed, 3 warnings`.
- GPU runtime smoke passed with exit code 0, `env_frames=16`, wandb offline summary, actor/schema audits, reward component stats, and final checkpoint at `wandb/offline-run-20260705_191435-pyfkk0z2/files/checkpoint_final.pt`.

**Consequence**: The repo is ready for short training-scale-up experiments and instinctRL-G baseline/evaluation harness work. Training convergence, robust policy performance, and trainable-governor success remain open evidence items.

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

---

## D-2026-07-14-001: R5F Default-Off Mechanism Hooks and Privileged Height Boundary

**Decision**: R5F may prepare default-off/default-equivalent governor, ICS, safety-filter, and reward hooks for mechanism screening, but privileged root-height filtering is sim/eval-only and cannot be claimed as a Paper-1 deployable actor method.

**Rationale**: R5E evidence splits the remaining failures into station/null XY drift and anchor error, tracking preservation, and near-floor clearance/ICS safety. Actor-clean hooks may be screened later without changing actor inputs. A root-height floor filter can diagnose controller-boundary safety behavior in simulation/evaluation, but root height is privileged and must never enter the deployed actor observation or deployable-method claim.

**Guardrails**: Actor observation remains exactly `lidar_grid + state_vec`; learned-governor action remains `[alpha, v_corr_x, v_corr_y, v_corr_z]`; hard gates remain unchanged; no training, sweep, 1M, warm-start, or promotion is authorized by this decision.
