# instinctRL Current Code vs Handbook Conformance Audit

Date: 2026-07-09  
Handbook: `docs/instinctRL_Development_Handbook_v1_1_platform_locked.tex`  
Scope: current workspace code under `isaac-training/training`, plus ROS deployment residue where it affects handbook H/deployment claims.

## Executive Verdict

The current simulation train/eval path is substantially more handbook-aligned than the earlier 8M run path. The previous P0 semantic drift has mostly been repaired: formal command-governor training now uses TASLAB_UAV + MID360, learned governor action semantics, staged adversarial command generation, actual body-velocity tracking reward, ICS attenuation, and handbook-style eval metrics.

Remaining deviations are mostly P1/P2, not the old P0 wrong-objective failure. The biggest risks are:

1. Dynamic obstacles are still largely NavRL privileged-state logic and are not RayCaster-visible MID360 geometry.
2. Formal eval does not enable observability metrics by default and still lacks station-keeping drift/scenario matrix completeness.
3. Actor audit is schema/key based; it does not prove `state_vec` provenance at runtime.
4. `train.yaml` still defaults to `instinctRL.mode: "smoke"`, so formal training requires command-line override.
5. ROS/deployment path remains old NavRL: odometry, map raycast, dynamic obstacle services, and ORCA-style safeAction remain in the policy/safety path.

## Handbook Requirements Checked

Core handbook requirements used for this audit:

- Locked platform/sensor: TASLAB_UAV and Livox MID360 only for normal instinctRL runs.
- Actor input contract: no pose, odometry, explicit translational velocity, map, SLAM, dynamic-obstacle privileged state, or root state in deployed actor input.
- MID360 observation: true range, valid mask, reliability weights, timestamps/frame age, IMU cues, command, previous command, and history.
- Action interface: learned actor outputs `alpha` and bounded `v_corr`; final command remains velocity-controller based.
- Command regimes: command comes from operator or `AdversarialCommandGenerator`; only `v_cmd` reaches actor.
- Reward: command tracking, anchor, safety, ICS compliance, intervention, smoothness, collision; privileged state reward/critic/eval only.
- Eval/logging: tracking RMSE, range-anchor error, minimum clearance, collision, ICS violation/intervention, command preservation, observability, termination reasons, platform/audit logs.
- Baselines/deployment: B0-B8 config isolation; ROS wrapper must be MID360/IMU/command based, not odom/map based.

## Conformance Matrix

| Area | Current code fact | Verdict |
|---|---|---|
| TASLAB platform lock | `training/cfg/drone.yaml` sets `drone.model_name: "TaslabUAV"`; env resolves the spawned model/base link dynamically. | Pass |
| MID360 active sensor | `env.py` uses `instinctRL.mid360_pattern.create_mid360_pattern_cfg`, logs `LivoxMid360Pattern`, stores ray-order hash, and RayCaster uses that pattern. | Pass with caveat |
| MID360 range/mask/weight/history | `MID360ObservationBuilder` produces true range, mask, staleness-weighted reliability, IMU cues, `v_cmd`, previous command, frame age, sim time, and history. | Pass |
| Actor input contract in sim | Actor observation is only `lidar_grid` and `state_vec`; PPO actor reads only those keys. | Pass with audit caveat |
| Learned governor policy | PPO learned mode outputs 4D normalized action and decodes to `alpha`, `v_corr`, `v_gov_b`. | Pass |
| Controller path | train/eval wrappers apply optional ICS, store previous issued command, then convert body-frame command to world-frame velocity for `VelController`. | Pass |
| Crazy/adversarial command generator | `curriculum_generator` source wires `AdversarialCommandGenerator`; modes include normal/aggressive/adversarial/oscillation/recovery. | Pass |
| Reward semantics | `InstinctRLRewardComputer` implements tracking, anchor, safety, ICS compliance, intervention, smoothness, collision, and component logging. Formal config enables actual velocity reward tracking. | Pass |
| Old NavRL reward objective | New reward path is used when `instinctRL.reward.enabled=true`; old NavRL goal reward remains as fallback. | Partial |
| Eval metrics | Streaming eval emits actual/proxy tracking RMSE, command preservation, anchor, safety, ICS, reward, and termination metrics. | Partial |
| Observability | Logger exists and is actor-clean, but train/eval config disables it by default. | Partial |
| Dynamic obstacles | Old privileged dynamic-obstacle state path still computes obstacle positions/velocities/collision; RayCaster mesh list is only `/World/ground`. | Non-conforming for dynamic-obstacle claims |
| Baselines | B0/direct and learned governor switches exist; B1-B8 isolation and matrix execution are not implemented. | Incomplete |
| ROS deployment | Existing ROS navigation still uses odometry, map raycast, goal-relative state, dynamic obstacle services, and ORCA-style safeAction. | Non-conforming |

## Findings

### P1: Dynamic Obstacles Are Not MID360-Observable in the Active Training Sensor

Evidence:

- RayCaster is created with `mesh_prim_paths=["/World/ground"]` only in `training/scripts/env.py`.
- Dynamic obstacle state is still computed from simulator arrays: positions, velocities, sizes, nearest-obstacle selection, and collision logic.
- Dynamic obstacle prims are spawned with `collision_enabled=False`; their positions/velocities are then used as privileged labels.

Impact:

This means `env_dyn.num_obstacles=80` does not create a fully MID360-observable dynamic-obstacle training problem. The actor does not receive dynamic-obstacle privileged state, which is good, but ICS and the adversarial nearest-obstacle vector operate on MID360 range returns that do not include those dynamic obstacles. Any claim about dynamic-obstacle robustness would be invalid until dynamic obstacles are included in RayCaster-visible geometry or explicitly labeled as privileged/eval-only.

Recommendation:

- For corrected short diagnostic retrain, either set `env_dyn.num_obstacles=0` and label it static-obstacle command-governor training, or wire dynamic obstacles into RayCaster-visible/collidable geometry before claiming dynamic-obstacle performance.
- Add a runtime audit that checks RayCaster mesh sources include all obstacle classes intended to be sensor-visible.

### P1: Observability Metrics Are Implemented but Disabled in Formal Config

Evidence:

- `instinctRL.observability.enabled: false` in `training/cfg/train.yaml` and `training/cfg/eval.yaml`.
- `RangeJacobianObservabilityLogger` exists and can compute proxy/offline metrics, but env only calls it when enabled.
- Streaming eval emits many `eval/handbook.*` metrics, but not observability summaries unless the env supplies them.

Impact:

The handbook requires observability metrics as part of the eval/logging set. Current eval can prove tracking/safety/ICS/termination, but it cannot prove observability-related claims with default configs.

Recommendation:

- Enable observability in eval configs or add an explicit `eval_observability=true` override for diagnostic eval.
- Add `eval/handbook.observability_*` summaries, not only raw optional diagnostics.

### P1: Actor Audit Does Not Prove `state_vec` Provenance at Runtime

Evidence:

- Actor audit checks forbidden key names and expected schema: `lidar_grid + state_vec`.
- `state_vec` is produced by the observation builder from IMU cues, `v_cmd`, previous command, and frame age, which is compliant today.
- The audit does not inspect producer metadata or value provenance inside `state_vec`.

Impact:

The current implementation is compliant by source inspection, but the runtime audit would not catch a future change that silently packs position, translational velocity, target direction, or dynamic obstacle state into `state_vec` while keeping the same key.

Recommendation:

- Add an observation-provenance manifest or runtime schema contract for `state_vec` slices: `[imu6, v_cmd3, prev_action3, frame_age1]`.
- Make the audit verify slice names/dimensions from metadata, not just tensor keys and shape.

### P1: Formal Eval Still Lacks Full Handbook Experiment Pipeline

Evidence:

- Streaming eval reports actual/proxy tracking RMSE, command preservation, anchor, safety, ICS, termination, and reward summaries.
- The handbook also requires station-keeping drift, observability metrics, scenario IDs/groups, and baseline matrix coverage.
- Current terrain remains the NavRL generated obstacle terrain; dedicated scenario groups such as corner, flat wall, corridor/tunnel, open field, unsafe command toward wall, gap/doorway, and real-robot observable holding are not isolated in config.

Impact:

Corrected short retrain can answer whether the repaired objective is non-degenerate. It still cannot support paper-level claims or full handbook success.

Recommendation:

- Add scenario config IDs and log them in eval summaries.
- Add station-keeping drift as eval-only metric using pose/mocap/sim state, explicitly outside actor input.
- Keep long training blocked from paper-level claims until the scenario/baseline harness exists.

### P1: Old NavRL Reward Path Remains as a Fallback

Evidence:

- `env.py` still computes target-relative position, distance, goal-frame velocity, old velocity reward, old safety logs, and height penalty.
- The instinctRL reward path is used only when `_reward_computer` and `_obs_builder` exist.
- If `instinctRL.reward.enabled=false` or the observation builder is absent, training falls back to the old NavRL navigation reward.

Impact:

Formal config is currently correct (`reward.enabled=true` and actual-velocity tracking), so this is not the old 8M failure anymore. The risk is accidental command-line/config drift: a user can still run `instinctRL.mode=train instinctRL.reward.enabled=false` and silently train a NavRL-like objective.

Recommendation:

- In `instinctRL.task=command_governor` mode, fail fast if `instinctRL.reward.enabled` is false unless `baseline.id` explicitly names a legacy/debug baseline.
- Move old NavRL reward computation behind a non-instinctRL branch to reduce accidental objective drift.

### P1: Train Config Defaults to Smoke Mode, Not Formal Training

Evidence:

- `training/cfg/train.yaml` has `instinctRL.enabled: true` but `instinctRL.mode: "smoke"`.
- Formal training works only when command-line overrides set `instinctRL.mode=train`.

Impact:

This is easy to misuse. A user launching `python training/scripts/train.py` will run the B0/B smoke path, not corrected learned-governor training.

Recommendation:

- Split configs: `train.yaml` for formal training and `smoke.yaml` or explicit override for smoke.
- Or require `instinctRL.mode` to be set explicitly when `instinctRL.enabled=true`.

### P2: MID360 Fidelity Is Static/Deterministic; Noise/Dropout Are Deferred

Evidence:

- `mid360_pattern.py` reuses `livox_mid360.py`, but creates a static RayCaster pattern with `enable_dynamic_scan=False`, `enable_occlusion_mask=False`, and `enable_noise=False`.
- `training/cfg/train.yaml` sets observation `enable_noise=false`, `enable_dropout=false`.

Impact:

The active sensor is MID360-shaped and uses the MID360 ray generator, so this is not a generic sensor violation. However, it does not yet exercise MID360 noise/dropout robustness.

Recommendation:

- Label current runs as deterministic MID360-ray simulation.
- Add robustness configs with noise/dropout enabled after corrected objective health is confirmed.

### P2: Baseline/Ablation Harness Is Mostly Not Implemented

Evidence:

- Config contains `baseline.id: direct_velocity`, learned/fixed governor mode, and comments for `no_history`, `no_anchor`, `no_imu`, `no_ics`.
- There is no complete B0-B8 config isolation or evaluator matrix.

Impact:

No full handbook contribution claims can be made yet. Current code supports mainline training/eval readiness and B0-ish smoke, but not the paper comparison matrix.

Recommendation:

- Add explicit config groups for B0-B8.
- Make evaluator output baseline ID, actor input schema hash, and required metric set for each run.

### P2: ROS Deployment Path Is Still Old NavRL

Evidence:

- `ros1/navigation_runner/scripts/navigation.py` subscribes to odometry, constructs actor `state`, `direction`, and `dynamic_obstacle`, uses map raycast points, and sends old policy requests.
- `safeAction.cpp` consumes map-frame obstacle positions/velocities and ORCA-style constraints.

Impact:

Simulation training/eval can be handbook-aligned while deployment remains non-conforming. The current ROS path must not be represented as instinctRL deployment.

Recommendation:

- Add a new instinctRL ROS wrapper with MID360 range/mask/weight/history + IMU + command input only.
- Keep old NavRL ROS files labeled legacy.
- Add deployment actor-input audit before H/real-robot validation.

## Positive Conformance Notes

- Platform config uses TASLAB_UAV and MID360 parameters.
- Active env now resolves TASLAB base link dynamically instead of hardcoding Hummingbird.
- MID360 helper is wired through `instinctRL.mid360_pattern`, not `BpearlPatternCfg`.
- Actor observation is `lidar_grid + state_vec` only in the sim training path.
- PPO actor uses only actor-clean observation; privileged fields are critic-only.
- Learned governor action is 4D normalized `[alpha, v_corr_x, v_corr_y, v_corr_z]`.
- Command generator is wired through a staged curriculum with normal/aggressive/adversarial/oscillation/recovery modes.
- Reward no longer uses NavRL reach-goal as formal success; `legacy_reach_goal` is diagnostic only.
- Streaming eval now exposes the key corrected metrics needed for short diagnostic retrain.

## Go / No-Go

Go for corrected short diagnostic retrain under static-obstacle, MID360-shaped, command-governor conditions.

No-go for:

- paper-level success claims;
- dynamic-obstacle robustness claims;
- ROS/deployment claims;
- full handbook baseline/ablation claims;
- any training run launched without explicitly confirming `instinctRL.mode=train`, `reward.enabled=true`, `ics.enabled=true`, and `command.source=curriculum_generator`.

## Immediate Next Fixes Before Long Training

1. Decide whether the short diagnostic retrain should disable dynamic obstacles or first make them MID360-visible.
2. Enable observability during diagnostic eval, or explicitly label observability as pending.
3. Add fail-fast protection against `instinctRL.task=command_governor` with disabled instinctRL reward.
4. Add `state_vec` provenance metadata/audit.
5. Split smoke and formal train configs to prevent accidental smoke-mode launches.
