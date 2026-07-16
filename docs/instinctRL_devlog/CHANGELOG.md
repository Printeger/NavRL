# instinctRL Development Changelog

> Format: [YYYY-MM-DD] — Ticket ID — Summary

---

## 2026-07-16 (A2-R5J next-step decision)

### instinctRL-A2-R5J: Default-Off Implementation Validation Authorized

**Status**: R5J mechanism plan complete; tests, default-off implementation, and disabled default-equivalence replay are the only authorized next work.

- Reviewed the R5G/R5H/R5I and Evidence-1/2/3 chain against the platform-locked handbook.
- Confirmed that the latest evidence does not support another parameter sweep or any 1M/formal learned-governor training.
- Selected the primary next mechanism as an actor-clean ICS residual-margin pre-emption guard.
- Defined a test-first gate covering default-disabled equivalence, a non-redundant synthetic trigger, positive-residual/no-closing/invalid-beam cases, invalid config, actor-audit isolation, regression tests, and disabled checkpoint replay equivalence.
- Added `NEXT_PROMPT.md` as the copy/paste execution prompt and updated current status, decision, test-plan, and README navigation records.

**Scope boundary**:

- No runtime behavior code or experiment was executed in this documentation decision.
- No enabled R5J replay, sweep, 1M, warm-start, promotion, gate change, actor-observation change, platform/sensor change, governor/controller method change, reward-default change, or privileged safety-filter default is authorized.
- Passing the bounded R5J step permits only the design of one enabled single-variable dry-run in a later turn.

---

## 2026-07-11 (A2-R3 station correction repair)

### instinctRL-A2-R3: Soft Null Station Correction Implemented

**Status**: source/unit implemented; fresh 128k A2-R3 sweep pending.

- Recorded the `20260711_111713` A2-R2 sweep as failed evidence: all six candidates had `passed=false` and `safety_passed=false`; `r2_balanced` was best at `7/15` gates but still failed station drift, anchor error, actual null-command speed, command amplification, and below-bound termination.
- Replaced hard-zero null correction semantics with a soft measurement-anchored station-correction path:
  - `null_vcorr_gate_enabled=true`
  - `null_vcorr_gate_eps=0.25`
  - `null_vcorr_gate_min=0.25`
- Split null-command behavior in reward semantics:
  - actual null-command speed is still penalized;
  - null-command output bias is penalized when the anchor is inactive or low-loss;
  - high anchor loss under an active/valid anchor relaxes the output-bias penalty so the governor can correct observable station drift.
- Updated formal reward defaults to `anchor_weight=4.0` and `null_command_output_weight=0.1`.
- Removed `null_command_output_speed_mean` from hard gate pass/fail while keeping the metric for diagnosis.
- Replaced R2 sweep variants with A2-R3 variants:
  `r3_soft_null_min025`, `r3_soft_null_min04`, `r3_anchor_strong`, `r3_anchor_strong_safety`, `r3_balanced_soft`, and `r3_no_decoder_gate_reward_only`.
- Added the A2-R3 ticket and updated context, decision log, test plan, and status docs.

**Validation**:

- Targeted tests passed: `27 passed`.
- instinctRL regression suite passed: `101 passed, 11 warnings`.
- A2-R3 sweep dry-run passed and emitted six `a2r3_sweep` jobs without launching training.

**Scope boundary**:

- Do not warm-start from `r2_balanced`.
- Do not run 1M or formal training until a fresh A2-R3 128k sweep is executed and candidates pass the screening gate.

---

## 2026-07-10 (A2-R2 objective hardening + sweep gate)

### instinctRL-A2-R2: Safety-Preservation Objective Hardening Implemented

**Status**: source/unit implemented; short 128k/256k corrective sweep pending.

- Added a learned-governor null-command deadband prior: when `||v_cmd||` is near zero, `v_corr` is ramped down to zero before command synthesis.
- Added reward preservation band terms:
  - penalize safe-command slowdown below `preservation_lower=0.75`;
  - penalize safe-command amplification above `preservation_upper=1.05`;
  - gate these penalties off when ICS is actively attenuating or emergency handling is active.
- Updated train/eval configs so preservation band and null-command gate are defaults, not ad-hoc CLI overrides.
- Made reward stats spec derive from `REWARD_COMPONENT_KEYS`, reducing future reward/stat drift.
- Added hard diagnostic gate tooling at `training/scripts/instinctRL/gates.py`.
- Added dry-run-first sweep tooling at `training/scripts/instinctRL/sweep.py`; real train/eval requires `--execute`.
- Added unit tests for governor deadband, preservation band, hard gates, and sweep command generation.

**Validation**:

- `py_compile` passed for governor, rewards, gates, sweep, PPO, and env modules.
- Targeted tests passed: `25 passed`.
- instinctRL regression suite passed: `99 passed, 11 warnings`.

**Scope boundary**:

- Do not start formal 1M/2M or 8M training directly.
- First run `sweep.py` dry-run, then execute short 128k/256k candidates and rank by hard gate.
- Only top 2-3 passing/improving candidates may enter 1M; only stable 1M candidates may enter formal training.

---

## 2026-07-09 (A2-R station objective repair)

### instinctRL-A2-R: Short Diagnostic Root-Cause Repair

**Status**: source/unit implemented; runtime 1M retrain pending.

- Classified `short_diag_static_mid360_1m_eval.json` as a diagnostic failure for station-keeping.
- Added root-cause audit at `audits/short_diag_static_mid360_1m_root_cause_2026-07-09.md`.
- Added null-command reward terms for actual body velocity and final issued command.
- Added command-chain proxy tracking and command-amplification penalties for safe nonzero commands.
- Increased formal `anchor_weight` to `2.0`.
- Added `station_first` command curriculum for formal training.
- Added `diagnostic_mixed` command curriculum for short diagnostic tracking eval comparability.
- Added handbook eval metrics for null-command speed/output and command amplification.
- Added source/unit tests for reward, curriculum, metrics, and eval diagnostic defaults.

**Validation**:

- Reward/task metric tests passed: `20 passed`.
- Reward/task/eval diagnostic tests passed: `24 passed`.
- instinctRL regression suite passed: `92 passed, 11 warnings`.
- 16-frame GPU smoke passed with `station_first` curriculum and new reward/metric specs.

**Scope boundary**:

- Do not claim convergence or formal learned-policy success.
- Do not start long training until a new 1M station-first short diagnostic passes the A2-R gate.

---

## 2026-07-09 (command-governor train/eval semantic repair)

### P0/P1: Corrected Formal Training and Evaluation Semantics

**Status**: source/unit implementation complete; corrected 16-frame GPU smoke passed; short diagnostic retrain pending.

- Reclassified the 8M run as wrong-objective evidence for handbook success.
- Added `instinctRL.task=command_governor` and demoted old `reach_goal` to `legacy_reach_goal`.
- Wired `AdversarialCommandGenerator` as the default staged command curriculum.
- Enabled ICS by default in formal train/eval configs.
- Switched formal tracking reward to reward-only actual body velocity tracking.
- Kept command-proxy tracking as eval/diagnostic signal.
- Replaced legacy target-relative critic inputs with task-aligned critic fields.
- Added explicit below-bound, above-bound, collision, timeout, and reason-code stats.
- Added streaming `eval/handbook.*` metrics for actual tracking, proxy tracking, command preservation, anchor, safety, ICS, and termination.
- Added `training/scripts/instinctRL/task_metrics.py` and `training/unit_test/test_instinctrl_task_metrics.py`.

**Validation**:

- Semantic/reward tests passed: `16 passed`.
- PPO/audit/governor tests passed: `27 passed, 11 warnings`.
- Source py_compile passed for touched train/eval/reward/env/PPO files.
- instinctRL regression suite passed: `84 passed, 11 warnings`.
- Corrected 16-frame GPU smoke passed with ICS and command curriculum enabled; final checkpoint written to `wandb/offline-run-20260709_155509-jqrryl8z/files/checkpoint_final.pt`.

**Scope boundary**:

- Do not start another long run until the corrected short diagnostic retrain passes.
- This is not a convergence or paper-level baseline-matrix claim.

---

## 2026-07-05 (A2 stability hardening)

### instinctRL-A2-S: PPO Numerical-Stability Hardening Implemented

**Status**: source/unit readiness complete; runtime 1M-frame acceptance pending because the local Isaac runtime failed before env import with missing Omniverse/Nucleus assets root.

- Hardened PPO against the learned-governor non-finite action failure observed around 563k frames.
- `BetaActor` now maps raw concentration outputs through bounded sigmoid ranges instead of unbounded Softplus.
- PPO validates finite observations, action distribution parameters, normalized actions, log-probs, entropy, values, returns, advantages, losses, ratios, gradients, and parameters.
- PPO normalizes advantages with `std.clamp_min(1e-6)`.
- PPO clips actor, critic, actor-feature, and critic-feature gradients with `algo.max_grad_norm=0.5`.
- PPO supports target-KL early stop with `algo.target_kl=0.02`.
- PPO writes compact `.pt` diagnostic snapshots on finite-audit failures.
- Added stability config defaults in `training/cfg/ppo.yaml`.
- Added `training/scripts/instinctRL/ppo_stability.py`.
- Added `training/unit_test/test_instinctrl_ppo_stability.py`.

**Validation**:

- `python -m py_compile training/scripts/ppo.py training/scripts/utils.py training/scripts/instinctRL/ppo_stability.py training/scripts/instinctRL/governor.py` passed.
- PPO/A2 targeted tests passed: `22 passed, 12 warnings`.
- A/B/C/D/E/F/A2/stability regression suite passed: `73 passed, 12 warnings`.
- Runtime train smoke command failed before `NavigationEnv` import: `RuntimeError: Unable to perform Nucleus login on Omniverse. Assets root path is not set.` This is an Isaac/Nucleus environment blocker, not a PPO finite-audit failure.

**Scope boundary**:

- Do not claim formal long-run training ready until the 1M-frame acceptance run completes.
- Do not claim convergence or learned-policy success.
- Do not silence NaNs by replacing actions with zeros; invalid tensors remain fail-fast diagnostics.

---

## 2026-07-05 (instinctRL-A2 trainable governor readiness)

### instinctRL-A2: Trainable Governor Head Complete; Short Smoke Ready

**Status**: trainable governor implementation and training-readiness audit hooks are complete. This is not a convergence or learned-policy success claim.

- Added actor-clean learned governor decoding:
  - PPO learned mode now samples a 4D normalized action: `[alpha, v_corr_x, v_corr_y, v_corr_z]`.
  - `alpha` is bounded in `[0,1]`; `v_corr` is bounded by `v_corr_limit=0.5 m/s`.
  - `v_gov_b = alpha * v_cmd_b + v_corr` is norm-clipped to `velocity_limit`.
  - `v_cmd_b` and previous body action are read from actor-allowed `state_vec`, not privileged `info`.
- Preserved B0/direct baseline support:
  - `MinimalGovernor` remains the smoke/direct baseline path.
  - `alpha_mode=fixed` still produces a 3D direct velocity policy action.
- Integrated learned-governor rollout path:
  - `InstinctRLTrainPolicy` wraps PPO during collection/eval.
  - The wrapper applies optional ICS, records `v_final_b`, then converts body-frame command to world-frame velocity at the controller boundary.
  - PPO update still trains against the 4D normalized governor action/log-prob.
- Added training-readiness audit hooks:
  - Policy init actor/governor audit.
  - Rollout batch audit.
  - Checkpoint file load sanity.
  - Forbidden-key actor input scan remains hard-fail.
- Made tiny PPO smoke failures clearer:
  - `make_batch()` now raises a direct error if `num_minibatches` exceeds the collected PPO batch.

**Validation**:

- `python -m py_compile training/scripts/ppo.py training/scripts/train.py training/scripts/env.py training/scripts/instinctRL/governor.py training/scripts/instinctRL/audit.py training/scripts/utils.py` passed.
- `python -m pytest -q training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_actor_audit.py` passed: `13 passed, 5 warnings`.
- A/B/C/D/E/F+A2 regression suite passed: `64 passed, 5 warnings`.
- GPU learned-governor train smoke passed with exit code 0:
  - `python training/scripts/train.py instinctRL.mode=train instinctRL.reward.enabled=true env.num_envs=4 env_dyn.num_obstacles=0 algo.training_frame_num=4 algo.num_minibatches=1 algo.training_epoch_num=1 max_frame_num=16 eval_interval=0 save_interval=0 wandb.mode=offline headless=true`
  - Logged learned-governor wrapper enabled, rollout batch audit pass, checkpoint audit pass, `env_frames=16`, and final checkpoint at `wandb/offline-run-20260705_203852-35lr9uce/files/checkpoint_final.pt`.

**Scope boundary**:

- A2 trainable-governor implementation is complete, but formal long learned-governor training is now gated by A2-S numerical-stability runtime acceptance.
- Training convergence is not proven.
- G baseline/evaluation matrix is not implemented.
- H real-robot deployment is not implemented.

---

## 2026-07-05 (instinctRL-F train-smoke readiness follow-up)

### instinctRL-F: Minimal Training Smoke Path Now Passes

**Status**: reward integration remains complete; the minimal PPO training smoke now runs to completion. This is not a training-convergence claim.

- Fixed PPO minibatch update path:
  - `PPO._update()` now recomputes critic features from minibatch actor features and critic-only `info` fields before calling the critic.
  - Added regression coverage for minibatches that do not carry cached `_critic_feature`.
- Fixed instinctRL train-mode logging:
  - `instinctRL.mode=train` now initializes wandb instead of leaving `run=None`.
  - `instinctRL.mode=smoke` still skips wandb.
- Added smoke-friendly train loop controls:
  - `eval_interval=0` disables periodic evaluation.
  - `save_interval=0` disables periodic checkpoint saves.
  - Final checkpoint save remains active.
- Added an instinctRL train completion exit path before `SimulationApp.close()` to avoid the known Isaac Kit shutdown segfault after successful validation.

**Validation**:

- `python -m py_compile training/scripts/train.py training/scripts/ppo.py && python -m pytest -q training/unit_test/test_instinctrl_rewards.py training/unit_test/test_instinctrl_ppo_hybrid.py` passed: `12 passed, 3 warnings`.
- GPU runtime training smoke passed with exit code 0:
  - `python training/scripts/train.py instinctRL.mode=train instinctRL.reward.enabled=true env.num_envs=4 env_dyn.num_obstacles=0 algo.training_frame_num=4 algo.num_minibatches=1 algo.training_epoch_num=1 max_frame_num=16 eval_interval=0 save_interval=0 wandb.mode=offline headless=true`
  - Logged `env_frames=16`, PPO loss scalars, actor/schema audits, reward component stats, and final checkpoint.
  - Final checkpoint path: `wandb/offline-run-20260705_191435-pyfkk0z2/files/checkpoint_final.pt`.

**Scope boundary**:

- This validates the minimal reward/PPO training path, not policy convergence.
- The PPO path still uses the current direct-velocity policy path; the trainable governor head remains open.
- Actor observation remains `lidar_grid` + `state_vec`.

---

## 2026-07-05 (instinctRL-F acceptance)

### instinctRL-F: Reward Integration Complete; Training Convergence Not Proven

**Status**: instinctRL-F reward integration/readiness is complete. This is not a learned-policy success or convergence claim.

- Added `training/scripts/instinctRL/rewards.py`:
  - `RewardConfig`, `RewardTerms`, and `InstinctRLRewardComputer`.
  - Command-consistency tracking proxy, anchor term, MID360 clearance safety term, ICS compliance offset, intervention penalty, smoothness penalty, and collision penalty.
  - Default reward path uses actor-clean/deployed-safe command and MID360 signals.
  - Optional privileged actual velocity is supported only behind `use_privileged_velocity_for_reward=false` default.
  - Reward totals and components remain finite and are clipped/scaled by `max_reward_abs`.
- Integrated F in `env.py`:
  - `instinctRL.reward.enabled=true` uses the new reward path.
  - `instinctRL.reward.enabled=false` preserves the old NavRL reward path.
  - Reward components are accumulated in `stats` for `EpisodeStats` logging.
  - Actor observation remains `lidar_grid` + `state_vec`.
  - Anchor/ICS disabled paths degrade gracefully.
- Added `instinctRL.reward.*` config in `training/cfg/train.yaml`, enabled by default for instinctRL.
- Added `training/unit_test/test_instinctrl_rewards.py`.
- Updated `CONTEXT.md` with F glossary terms.

**Validation**:

- `python -m pytest -q training/unit_test/test_instinctrl_rewards.py` passed: `10 passed, 1 warning`.
- A/B/C/D/E/F regression suite passed: `54 passed, 2 warnings`.
- `py_compile` passed for changed F code/tests.
- TorchRL spec probe passed for reward component stats insertion.
- Runtime Isaac smoke with `instinctRL.reward.enabled=true` was not run in this environment because CUDA/NVML is not visible here.

**Scope boundary**:

- No trainable governor head was implemented.
- No actor observation schema change was made.
- No baseline matrix or real-robot deployment work was implemented.
- Privileged simulator quantities remain reward/critic/eval/logging only.

**Final conclusion**:

- `instinctRL-F`: COMPLETE for reward integration/readiness
- Training convergence: NOT PROVEN
- `instinctRL-G`: GO for baseline/evaluation harness only, not learned-policy success claims

---

## 2026-07-05 (instinctRL-E acceptance)

### instinctRL-E: ICS-Inspired Attenuation Complete

**Status**: instinctRL-E unit and A/B/C/D/E regression acceptance complete. instinctRL-F may start for reward-design work; training convergence remains unproven.

- Added `training/scripts/instinctRL/ics.py`:
  - `ICSConfig`, `ICSOutput`, and `RangeHistoryICSAttenuator`.
  - Deployed-safe attenuation: `v_final_b = beta * v_gov_b` with `brake_mode="zero"` only.
  - MID360 range/mask/weight history inputs, body-frame ray directions, body-frame governor command, and config parameters only.
  - Empty active set returns beta 1.0 by default.
  - Emergency reliable-min-clearance bypass forces beta 0.
  - Optional finite-difference range-rate filter is cached by default and only affects beta when enabled.
  - Final body command is norm-clipped after beta computation.
- Integrated E without actor-schema changes:
  - `MID360ObservationBuilder.get_history(copy=True)` exposes range/mask/weight history in oldest-to-latest order.
  - `NavigationEnv.get_instinctrl_range_history()` wraps the builder accessor.
  - `env.py` adds scalar `ics_*` info specs when enabled and stores dense cache in `self.ics_outputs`.
  - `train.py` applies ICS before `BodyToWorldVelocityAdapter` and stores `v_final_body` as previous issued action.
- Added `instinctRL.ics.*` config in `training/cfg/train.yaml`, disabled by default.
- Added `training/unit_test/test_instinctrl_ics.py`.

**Validation**:

- `python -m pytest -q training/unit_test/test_instinctrl_ics.py` passed: `10 passed, 1 warning`.
- A/B/C/D/E regression suite passed: `44 passed, 2 warnings`.
- `py_compile` passed for changed E code/tests.
- Runtime Isaac smoke with `instinctRL.ics.enabled=true` was not run in this environment because CUDA/NVML is not visible here; earlier B smoke coverage remains the last live Isaac smoke evidence.

**Scope boundary**:

- No reward/training implementation was added.
- Actor observation remains `lidar_grid` + `state_vec`.
- E deployed path does not use surface normals, map, odometry, SLAM, pose, or dynamic-obstacle privileged state.

**Final conclusion**:

- `instinctRL-E`: COMPLETE
- `instinctRL-F`: GO for reward-design work only; training convergence remains not complete

---

## 2026-07-05 (instinctRL-D acceptance)

### instinctRL-D: Observability Logger Complete

**Status**: instinctRL-D full acceptance complete. instinctRL-E may start; instinctRL-F remains deferred.

- Added `training/scripts/instinctRL/observability.py`:
  - `ObservabilityConfig`, `ObservabilityOutput`, and `RangeJacobianObservabilityLogger`.
  - Offline finite-difference Jacobian estimator.
  - Offline surface-normal geometric approximation.
  - Deployed-safe scan-geometry proxy mode, labeled with `observability_is_proxy`.
  - SVD rank, singular values, condition number cap, weak direction cache, and drift projection diagnostics.
- Integrated observability passively in `env.py`:
  - Default config keeps `instinctRL.observability.enabled=false`.
  - When enabled, env writes only scalar `observability_*` metrics to `info`.
  - Dense Jacobian/SVD internals remain in `self.observability_outputs`.
  - Actor observation remains `lidar_grid` + `state_vec`.
- Added `training/unit_test/test_instinctrl_observability.py`.

**Validation**:

- `python -m pytest -q training/unit_test/test_instinctrl_observability.py` passed: `9 passed, 1 warning`.
- A/B/C/D regression suite passed: `34 passed, 2 warnings`.
- `py_compile` passed for changed D code/tests.
- TorchRL int64 spec probe passed for `observability_mode_code`.

**Final conclusion**:

- `instinctRL-D`: COMPLETE
- `instinctRL-E`: GO
- `instinctRL-F`: NO-GO until E and reward prerequisites are complete

---

## 2026-07-05 (instinctRL-C acceptance)

### instinctRL-C: Measurement-Space Anchor Manager Complete

**Status**: instinctRL-C full acceptance complete. instinctRL-D may start; instinctRL-E/F remain deferred.

- Added `training/scripts/instinctRL/anchor.py`:
  - `AnchorConfig`, `AnchorState`, `AnchorStepOutput`, `MeasurementSpaceAnchorManager`, and pure `huber_loss()`.
  - Null-command hysteresis with `||v_cmd|| <= eps_enter` capture and `||v_cmd|| >= eps_exit` command reset.
  - Fixed reset enum: `0 none`, `1 episode`, `2 explicit`, `3 command`, `4 invalid`.
  - Reset priority: `episode > explicit > command > invalid > none`.
  - Capture-time `r_star`, bool `m_star`, and reliability `w_star`.
  - Handbook-aligned dense error: `(m_t * m_star * w_t) * (r_t - r_star)`.
  - Fixed-structural-denominator Huber `anchor_loss`.
  - Public scalar metrics separated from dense internal cache.
- Integrated anchor manager passively in `env.py`:
  - Scalar diagnostics are written to `info`: `anchor_active`, `anchor_loss`, `anchor_valid_fraction`, `anchor_error_mean`, `anchor_error_max`, `anchor_hold_steps`, `anchor_activation_count`, `anchor_reset_reason`.
  - Dense `anchor_error`, masks, and anchor references remain internal through `self.anchor_outputs`.
  - Actor observation remains `lidar_grid` + `state_vec`; no anchor tensors are added to actor input.
- Added `instinctRL.anchor.*` config in `training/cfg/train.yaml`.
- Added `training/unit_test/test_instinctrl_anchor.py`.

**Validation**:

- `python -m pytest -q training/unit_test/test_instinctrl_anchor.py` passed: `11 passed, 1 warning`.
- B+C pytest suite passed: `25 passed, 2 warnings`.
- `py_compile` passed for changed C code/tests.
- TorchRL int64 spec probe passed for `anchor_reset_reason`.

**Final conclusion**:

- `instinctRL-C`: COMPLETE
- `instinctRL-D`: GO
- `instinctRL-E/F`: NO-GO until their stage prerequisites are opened

---

## 2026-07-04 (instinctRL-B acceptance)

### instinctRL-B: Complete; instinctRL-C GO

**Status**: instinctRL-B full acceptance complete. instinctRL-C may start.

- Final user-side smoke passed after the shutdown workaround:
  - PPO hybrid forward smoke passed.
  - Actor input audit passed.
  - Actor schema audit passed.
  - Action type audit passed.
  - 500/500 smoke steps completed.
  - MID360 raw range shape `[4, 1, 360, 59]`, valid returns `28.62%`.
  - `B0 Smoke Test PASSED` and `Observation smoke path PASSED`.
  - Smoke success path exited before `SimulationApp.close()` to avoid Isaac Kit teardown segfault after pass.
- NavRL pytest remains green: `14 passed, 2 warnings`.
- Active MID360 path is `LivoxMid360Pattern`, not `BpearlPatternCfg`.
- Hybrid actor input is `lidar_grid` + `state_vec`; privileged fields remain critic/reward/info only.

**Final conclusion**:

- `instinctRL-A`: PASS
- `instinctRL-B`: COMPLETE
- `instinctRL-C`: GO

---

## 2026-07-04 (Smoke shutdown handling)

### instinctRL-B: Avoid Isaac Kit shutdown segfault after passed smoke

**Status**: Smoke success path updated.

- User-side smoke reached 500/500 steps and printed:
  - `B0 Smoke Test PASSED`
  - `Observation smoke path PASSED`
  - MID360 raw range shape `[4, 1, 360, 59]`, valid returns `33.04%`
- The subsequent failure was not an observation/control failure. It occurred after validation, inside `SimulationApp.close()`, with a native segmentation fault during Isaac Kit shutdown.
- Smoke mode now exits with code 0 immediately after all B0/B validation checks pass, before calling `SimulationApp.close()`. Failure paths still close the app and raise the real exception.

**Acceptance note**: The smoke output before the segfault already satisfied the B runtime checks. The code change prevents Isaac shutdown from turning a passed smoke into a failed shell command.

---

## 2026-07-04 (MID360 RayCaster runtime fix)

### instinctRL-B: Fixed RayCaster in-place offset failure

**Status**: Code fix and regression test complete. Superseded by the later smoke shutdown handling entry: user-side runtime smoke reached and passed all B checks.

- User-side smoke reached `NavigationEnv` and active MID360 RayCaster initialization, proving the prior Python dependency blockers were resolved.
- Runtime failed in Orbit `RayCaster._initialize_rays_impl()` at `self.ray_starts += offset_pos`.
- Root cause: the Livox MID360 helper returns ray origins via `expand()`, which creates an overlapping-memory view. Orbit mutates ray starts in-place when applying the sensor offset, and PyTorch rejects in-place writes to overlapping expanded views.
- Fix: `instinctRL.mid360_pattern._mid360_pattern()` now returns cloned contiguous `ray_starts` and `ray_directions`.
- Added regression coverage: `test_mid360_ray_starts_support_orbit_inplace_offset()` simulates Orbit's in-place offset operation.
- NavRL B pytest now passes: `14 passed, 2 warnings`.

**Command note**: The smoke command must pass `env_dyn.num_obstacles=0` as a Hydra override on the same command, or on a continued line with `\`. If entered as a separate shell line, it becomes `env_dyn.num_obstacles=0: command not found` and is not applied.

---

## 2026-07-04 (NavRL environment validation)

### instinctRL-B: NavRL pytest/PPO validation passed; runtime smoke GPU-blocked

**Status**: Unit/PPO validation passed in the activated `NavRL` conda environment. Superseded by later user-side GPU smoke evidence.

- Corrected the validation command path: tests must run after `conda activate NavRL`, not by directly invoking `/home/mint/miniconda3/envs/NavRL/bin/python`.
- Confirmed activated `NavRL` resolves:
  - `torch 2.0.1+cu118` from the Isaac Sim prebundle, with `ForkingPickler` available.
  - `tensordict 0.4.0+3725bcc` and `torchrl 0.4.0+3725bcc` from repo third-party paths.
  - `click 8.1.3`, `wandb 0.23.1`, `hydra 1.3.2`.
- Fixed PPO critic privileged-field concatenation by flattening `info.drone_state`, `info.target_rpos`, and `info.target_distance` before concatenating them with `_actor_feature`.
- Updated the PPO hybrid test to match the current PPO action shape `[N, 3]`.
- `python -m pytest training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_ppo_hybrid.py -q` now passes: `13 passed`.
- Minimal Isaac smoke now reaches the CUDA preflight but cannot run here because no CUDA-capable device is visible; `nvidia-smi` cannot communicate with the NVIDIA driver and `torch.cuda.is_available()` is `False`.

**Acceptance conclusion**: Superseded by later user-side smoke evidence. B runtime checks passed; final acceptance waits only on a post-workaround smoke rerun returning shell exit status 0.

---

## 2026-07-04 (B-fix implementation)

### instinctRL-B: Observation / History Buffer Fix Pass

**Status**: Implementation complete. Superseded by the later NavRL validation entry above.

- Replaced the active instinctRL RayCaster pattern path with a Livox MID360 helper wrapper; `env.py` no longer uses `patterns.BpearlPatternCfg` for instinctRL.
- Fixed `BodyToWorldVelocityAdapter` to use body-to-world quaternion rotation semantics and added identity, yaw, and roll/pitch frame tests.
- Wired `prev_action` from the previously issued governor/controller command into `MID360ObservationBuilder`; reset now clears selected env history and previous issued action.
- Hardened Observation / History Buffer behavior: raw range remains true distance, invalid beams have zero reliability, weights clamp to `[0, 1]`, stale/repeated frames are flagged, and history rollover/reset are unit-tested.
- Added actor schema audit for hybrid actor input and tests proving actor obs only exposes `lidar_grid` and `state_vec`.
- Added explicit `instinctRL.mode` separation: `smoke` runs B0/B observation smoke; `train` initializes PPO and runs the hybrid actor audit/forward path before continuing training.
- Added pure Python/PyTorch unit tests for MID360 pattern shape/order, observation semantics, previous-action feedback, actor schema, PPO hybrid forward smoke, and command-adapter frame convention.

**Initial validation note**:

- This entry originally recorded failures from direct Python invocation paths. The later `NavRL environment validation` entry corrected that: after `conda activate NavRL`, pytest/PPO validation passes.
- Superseded by later user-side smoke evidence: B runtime checks passed; final acceptance waits only on a post-workaround smoke rerun returning shell exit status 0.

**Acceptance conclusion**: Superseded by later smoke entries.

---

## 2026-07-04 (Closeout Acceptance Review)

### instinctRL-A / instinctRL-B Closeout Before C

**Status**: Historical closeout. Superseded in part by the later B-fix implementation entry above.

- **instinctRL-A**: Accepted as B0 smoke-test / infrastructure baseline, not learning success.
- **instinctRL-A verification update**: Adapter frame direction has now been corrected and unit-tested. Runtime integration remains covered by the before-C smoke validation.
- **instinctRL-B**: Historical closeout finding superseded. Later B-fix implementation, NavRL pytest/PPO validation, and user-side GPU smoke completed B acceptance.
- **instinctRL-C**: Current status is GO.

This entry supersedes earlier 2026-07-04 entries that described instinctRL-B as complete or recommended proceeding directly to instinctRL-C.

---

## 2026-07-04 (PM3)

### instinctRL-B: Observation / History Buffer

**Status**: Superseded by closeout review. Implementation exists, but B is only partially accepted.

- **New**: `instinctRL/observation.py` — `MID360ObservationBuilder` (220 lines)
  - Raw MID360 range $r_t$ (true distance, not danger-coded)
  - Valid-return mask $m_t$ (finite + in-range)
  - Staleness-weighted reliability $w_t = m_t \cdot \exp(-age/\tau)$
  - IMU cues: body ang_vel(3) + gravity_dir_body(3)
  - v_cmd + prev_action tracking
  - Fixed-size history buffer (L=4, configurable)
- **Config**: `instinctRL.observation.*` (history_len, enable_noise, enable_dropout, tau_staleness)
- **Env**: replaced danger-coded LiDAR with ObservationBuilder; hybrid obs spec (lidar_grid + state_vec)
- **PPO**: multi-channel CNN + state vector encoder with CatTensors merge
- **Deferred**: D-009 (noise/dropout curriculum), D-010 (neighbor-consistency weights), D-011 (longer history ablations)
- **Superseded claim**: This entry originally marked D-002 resolved. The 2026-07-04 closeout review corrected D-002 to partial only.

---

## 2026-07-04 (PM4)

### B0 Smoke Test — PASSED ✅ (7/7 criteria)

**Command**: `python3 training/scripts/train.py env.num_envs=4 env_dyn.num_obstacles=0`  
**Runtime**: 141.9s, Exit Code 0, GPU: RTX 4070 Ti SUPER (16GB)  

- A.1 Platform Lock ✅ `TaslabUAV` + MID360 FOV [-7°,52°] + 40m range
- A.2 Actor Input ✅ no forbidden fields
- A.3 Action Type ✅ 3-dim velocity
- A.9 Smoke Rollout ✅ 500 steps, no NaN, no crash
- A.6/A.10 LiDAR ✅ `[4,1,360,59]`, 18.97% valid returns
- A.7 Governor ✅ B0 α=1.0, direct_velocity baseline
- instinctRL-B hybrid obs ✅ `lidar_grid=[4,12,360,59]`, `state_vec=[4,52]`

**Note**: Previous "Blocked" diagnosis (PM2) was a 512-env scaling issue. With 4 envs everything works.

---

### instinctRL-A: B0 Smoke Test Runtime (OBSOLETE — superseded by PM4)

~~Blocked by 512-env PhysX fabric issue~~ — resolved by using fewer envs.

---

## 2026-07-04 (PM)

### instinctRL-A: Direct Velocity-Governor Baseline (B0)

**Status**: ✅ Complete

**New modules**:
- `instinctRL/audit.py` — Staged audit: platform lock, actor input, action type (140 lines)
- `instinctRL/command_adapter.py` — `BodyToWorldVelocityAdapter` using `quat_rotate_inverse` (55 lines)
- `instinctRL/governor.py` — `MinimalGovernor` (B0: α=1, v_corr=0) + `GovernorOutput` dataclass (75 lines)

**Config changes**:
- `cfg/train.yaml`: Added `instinctRL.enabled` + `instinctRL.baseline.id`
- `cfg/ppo.yaml`: Added `instinctRL.governor.*` (alpha_mode, alpha_fixed, v_corr_limit, velocity_limit, smoothing_tau)

**Code wiring**:
- `env.py`: v_cmd production (fixed + random body-frame), MID360 raw range, v_cmd in info spec
- `train.py`: B0 smoke test path (audit → governor → adapter → VelController → 500-step loop → exit)

**Documentation**:
- `DEFERRED_REGISTER.md` — 8 items (D-001 to D-008)
- `DECISION_LOG.md` — 6 architectural decisions
- `TEST_PLAN.md` — 10 B0 smoke tests + future registries
- `tickets/instinctRL-A_direct_velocity_governor_baseline.md` — Full ticket report

**Method consistency**: All checks pass (velocity action, actor input clean, platform/sensor locked).

---

## 2026-07-04 (AM)

### instinctRL-0: Blocker Fixes (All 5 Resolved)

**Status**: ✅ Complete

**Blocker 1 — Prim path**:
- Replaced hardcoded `Hummingbird_0/base_link` with dynamic `{model_name}_0/{base_link}` resolution
- Added `_resolve_base_link()` helper using robust search from MID360 integration test
- Set `attach_yaw_only=False` for solid-state MID360
- Logs resolved prim path at init

**Blockers 2–4 — Actor input sanitization**:
- Removed `state[8]` (vel_g, rpos_clipped_g, distance_2d/z) from actor observation
- Removed `direction[3]` from actor observation
- Removed `dynamic_obstacle[N,10]` from actor observation
- Actor now receives only `lidar` (raw sensor input)
- Forbidden fields retained for reward/collision/evaluation use only (not in actor TensorDict)

**Blocker 5 — Critic privileged branch (Option B)**:
- Added asymmetric actor-critic architecture in `ppo.py`:
  - `actor_feature_extractor`: LiDAR only → `_actor_feature` (256d)
  - `critic_feature_extractor`: `_actor_feature` + `info["drone_state"]` + `info["target_rpos"]` + `info["target_distance"]` → `_critic_feature` (256d)
- Actor head uses `_actor_feature`; Critic head uses `_critic_feature`
- Added `info["target_rpos"]` and `info["target_distance"]` to info spec
- Added `verify_actor_critic_separation()` test proving critic-field perturbation does not affect actor output
- Removed dead `dynamic_obstacle_network` and unused `vec_to_world` import

**Blocker 5 — Command adapter**:
- Deferred to instinctRL-A (no unused infrastructure in runtime code)
- Interface documented in ticket report

**CONTEXT.md**:
- Added active method-lock banner (Paper-1 velocity-governor route)
- Marked CTBR, CMDP/PPO-Lagrangian, GRU-required actor as ⚠️ LEGACY
- Added Paper-1 vs Paper-2 distinction table

**Files changed**: `env.py` (+55/-40 lines), `ppo.py` (+80/-65 lines), `CONTEXT.md` (+35/-15 lines)

---

## 2026-07-03

### instinctRL-0: Platform and Sensor Infrastructure Audit

**Status**: ✅ Complete

- Created devlog structure (`docs/instinctRL_devlog/`)
- Produced comprehensive platform and sensor audit (`docs/instinctRL_0_platform_sensor_audit.md`)
- Confirmed MID360 simulation infrastructure (`training/envs/livox_mid360.py`, integration helpers, unit tests)
- Confirmed TASLAB_UAV model registration, physical parameters, and controller gains
- Identified 5 blockers preventing instinctRL-A:
  1. LiDAR prim path hardcoded to Hummingbird (not TASLAB_UAV)
  2. Actor receives ground-truth velocity `vel_g`
  3. Actor receives goal-relative position (`rpos_clipped_g`, `distance_2d`, `distance_z`)
  4. Actor receives privileged dynamic obstacle state
  5. No body-frame velocity command interface
- Catalogued 100% actor input non-compliance with instinctRL contract
- Documented frame convention risks (body vs world frame, attach_yaw_only, mount rotation order)
- Produced reuse vs. new-implementation assessment for all instinctRL components
- Issued conditional go/no-go: proceed after blockers 1–5 resolved
- Updated `DEV_STATUS.md`

---

## Template

```markdown
## YYYY-MM-DD

### Ticket ID: Title

**Status**: [Not Started | In Progress | Complete | Blocked]

- Change 1
- Change 2
```

## 2026-07-16

### A2-R5J default-off residual pre-emption — HOLD recorded

- Added default-off R5J per-beam residual guard, config validation/defaults, neutral disabled diagnostics, internal evidence cache, and scalar info specs.
- Added full six-case `6f6dee3` legacy golden coverage, R5J eval summaries, pure comparator tests, and an artifact-local replay wrapper under `tests/artifacts/r5j_default_equivalence/20260714_234801/`.
- The wrapper recorded `nvidia-smi` exit 9 (driver communication failure) and `torch.cuda.is_available() == false`; it did not launch eval or produce replay JSON. `comparison.json` records that actual wrapper failure as `HOLD`. No enabled replay, sweep, training, or main merge occurred.
