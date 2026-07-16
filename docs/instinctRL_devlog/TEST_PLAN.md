# instinctRL Test Plan

> **Created**: 2026-07-04 (instinctRL-A)  
> **Last Updated**: 2026-07-16 (A2-R5J provenance repair verification)
> **Purpose**: Define verification procedures for each instinctRL stage.

---

## instinctRL-A2-R5J: Braking-Residual Pre-emption

**Current verdict**: Default-off implementation is complete and the provenance repair is under verification. The historical disabled artifact is `HOLD` because it was created from a dirty worktree and then recorded NVIDIA-driver communication failure with `torch.cuda.is_available() == false`; eval was not started. Enabled behavior evaluation, sweeps, 1M, warm-starting, promotion, and formal training remain forbidden.

### Required A2-R5J Tests

| Test | Required evidence | Current status |
|------|-------------------|----------------|
| R5J.1 Default-off config | New residual pre-emption enable flag is false in code, train config, and eval config | Passed |
| R5J.2 Disabled equivalence | Disabled guard preserves current `v_final_b`, existing public metrics, and existing cache behavior | Passed by literal six-scenario golden matrix |
| R5J.3 Non-redundant trigger | Synthetic eligible-beam case triggers residual pre-emption before current ICS would fully stop | Passed |
| R5J.4 Unchanged safe cases | Positive residual, no closing evidence, unreliable/invalid beams, and empty active set do not pre-empt | Passed |
| R5J.5 Config validation | Non-finite/negative residual margin or collision threshold is rejected | Passed |
| R5J.6 Actor-clean boundary | New R5J keys are rejected as actor input; actor remains exactly `lidar_grid + state_vec` | Passed |
| R5J.7 Source/regression | `py_compile`, targeted tests, and full `test_instinctrl_*.py` suite pass | Passed: py_compile; targeted `19 passed, 1 warning`; full Isaac Sim Conda suite `163 passed, 13 warnings` (one NVML and twelve LazyModule) |
| R5J.8 Default-equivalence replay | Existing `r5g_downatten_z010` checkpoint is replayed with R5J explicitly disabled and compared with stored baseline evidence | HOLD: the historical CUDA artifact was dirty-worktree/preflight-only and cannot satisfy the repaired provenance contract |

### Completed Order and Future Exit Gate

1. The test-first default-off implementation and actor audit are complete.
2. Py-compile, targeted regression, and the full active-environment regression are complete.
3. Commit and push the provenance repair, then require empty `git status --porcelain`.
4. Run raw CUDA checks. If CUDA is unavailable, stop without invoking the wrapper or creating a runtime attempt.
5. Only if CUDA is available, replay the stored `r5g_downatten_z010` checkpoint exactly once with the guard explicitly disabled. Change only `result_path` and the explicit false enable override.
6. Record one exit decision:
   - `GO (design only)` only if the new replay, exact JSON comparison, disabled diagnostics, gate report, and provenance all pass. It permits later design, never execution, of an enabled single-variable R5J dry-run.
   - `HOLD` for any failed or missing condition. Do not retry a failed attempt by reusing a result path.

The exact execution prompt is stored in [`NEXT_PROMPT.md`](./NEXT_PROMPT.md).

---

## instinctRL-A2-R3: Station Correction Repair

**Historical verdict**: SOURCE/UNIT COMPLETE; the `20260711_144051` 128k sweep ran, all six candidates failed, and the path was superseded by later R4/R5 investigation. No A2-R3 candidate is eligible for 1M.

### Required A2-R3 Tests

| Test | Required evidence | Current status |
|------|-------------------|----------------|
| A2-R3.1 Soft null decoder gate | `v_cmd=0` scales `v_corr` by `null_vcorr_gate_min` instead of forcing zero | Passed |
| A2-R3.2 Anchor-aware null output reward | High anchor loss under an active/valid anchor relaxes null output bias penalty | Passed |
| A2-R3.3 Low-loss/null-anchor bias penalty | Low anchor loss or inactive anchor still penalizes null output bias | Passed |
| A2-R3.4 Hard gate update | `null_command_output_speed_mean` is diagnostic-only and cannot fail the gate by itself | Passed |
| A2-R3.5 Sweep variants | A2-R3 dry-run emits the six `r3_*` variants and no R2 warm-start assumption | Passed |
| A2-R3.6 Runtime sweep | 128k A2-R3 candidates are trained, evaled, ranked, and screened before 1M | Passed as execution evidence; all candidates failed the promotion gate |

### Actual Commands and Results

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_rewards.py training/unit_test/test_instinctrl_gates.py` | Passed: `27 passed`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_*.py` | Passed: `101 passed, 11 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python training/scripts/instinctRL/sweep.py --frames 131072 --seeds 0 --limit 6` | Passed dry-run: emitted six `a2r3_sweep` jobs for the `r3_*` variants without launching training. |

### A2-R3 Runtime Procedure

1. Dry-run the A2-R3 sweep:

```bash
python training/scripts/instinctRL/sweep.py \
  --frames 131072 \
  --seeds 0 \
  --limit 6
```

2. Execute only after dry-run review:

```bash
python training/scripts/instinctRL/sweep.py \
  --execute \
  --frames 131072 \
  --seeds 0 \
  --limit 6
```

3. Promote to 1M only if the candidate satisfies the A2-R3 screening gate:

- `safety_collision_rate == 0`
- `termination_collision == 0`
- `termination_below_bound == 0`
- `safety_min_clearance_p05 >= 1.0`
- `station_keeping_drift_mean < 1.3`
- `station_keeping_drift_p95 < 2.6`
- `anchor_error_mean < 2.0`
- `tracking_rmse_actual_body_vs_v_cmd <= 0.45`
- `command_amplification_rate <= 0.15`

---

## instinctRL-A2-R2: Objective Hardening + Sweep Gate

**Current verdict**: SOURCE/UNIT COMPLETE; superseded by A2-R3 after the `20260711_111713` sweep failed all six candidates.

### Required A2-R2 Tests

| Test | Required evidence | Current status |
|------|-------------------|----------------|
| A2-R2.1 Null-command decoder prior | `v_corr` is forced/ramped toward zero when `||v_cmd||` is in the deadband | Passed |
| A2-R2.2 Preservation low band | Safe nonzero command with `preservation < 0.75` is penalized | Passed |
| A2-R2.3 Preservation high band | Safe nonzero command with `preservation > 1.05` is penalized | Passed |
| A2-R2.4 ICS allowance | Preservation slowdown penalty is disabled when ICS attenuates | Passed |
| A2-R2.5 Reward/spec sync | Env reward stats derive from `REWARD_COMPONENT_KEYS` | Passed by source/unit coverage |
| A2-R2.6 Hard gate scorer | Eval JSON is classified by station/tracking/safety gates, not collision alone | Passed |
| A2-R2.7 Sweep dry-run | Short-sweep commands are generated without launching training by default | Passed |
| A2-R2.8 Runtime sweep | 128k/256k candidates are trained, evaled, and ranked automatically | Pending |

### Actual Commands and Results

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/governor.py training/scripts/instinctRL/rewards.py training/scripts/instinctRL/gates.py training/scripts/instinctRL/sweep.py training/scripts/ppo.py training/scripts/env.py && python -m pytest -q training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_rewards.py training/unit_test/test_instinctrl_gates.py` | Passed: `25 passed`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_*.py` | Passed: `99 passed, 11 warnings`. |

### Runtime Gate Source of Truth

- Gate implementation: `training/scripts/instinctRL/gates.py`
- Sweep implementation: `training/scripts/instinctRL/sweep.py`
- A2-R2 gate included station drift, null-command speed/output, station anchor error, tracking RMSE, preservation band, amplification, clearance p05, collision, ICS violation, and termination reasons. A2-R3 removes null-command output speed from hard pass/fail because bounded anchor-aware correction is now allowed.
- Dry-run command: `python training/scripts/instinctRL/sweep.py --frames 131072 --seeds 0 --limit 6`
- Execution command: `python training/scripts/instinctRL/sweep.py --execute --frames 131072 --seeds 0 --limit 6`

---

## instinctRL-A2-R: Station Objective Repair

**Current verdict**: SOURCE/UNIT COMPLETE. Runtime 1M station-first short diagnostic retrain remains pending.

### Required A2-R Tests

| Test | Required evidence | Current status |
|------|-------------------|----------------|
| A2-R.1 Null-command speed reward | `v_cmd≈0` penalizes actual body velocity without actor leakage | Passed |
| A2-R.2 Null-command output reward | `v_cmd≈0` penalizes nonzero final issued command | Passed |
| A2-R.3 Command preservation | Safe nonzero commands penalize proxy tracking error and command amplification | Passed |
| A2-R.4 Curriculum profile | `station_first` starts recovery-heavy; `diagnostic_mixed` remains fixed for eval comparability | Passed |
| A2-R.5 Eval metrics | Eval exposes null-command and amplification handbook metrics | Passed |
| A2-R.6 Runtime gate | New 1M static MID360 retrain passes A2-R go/no-go thresholds | Pending |

### Actual Commands and Results

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_rewards.py training/unit_test/test_instinctrl_task_metrics.py` | Passed: `20 passed`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_rewards.py training/unit_test/test_instinctrl_task_metrics.py training/unit_test/test_instinctrl_eval_diagnostic.py` | Passed: `24 passed`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/rewards.py training/scripts/instinctRL/task_metrics.py training/scripts/env.py training/scripts/utils.py training/scripts/eval.py` | Passed. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_*.py` | Passed: `92 passed, 11 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python training/scripts/train.py instinctRL.mode=train instinctRL.task=command_governor instinctRL.reward.enabled=true instinctRL.reward.use_privileged_velocity_for_reward=true instinctRL.ics.enabled=true instinctRL.command.source=curriculum_generator instinctRL.command.curriculum_profile=station_first env.num_envs=4 env.num_obstacles=20 env_dyn.num_obstacles=0 algo.training_frame_num=4 algo.num_minibatches=1 algo.training_epoch_num=1 max_frame_num=16 eval_interval=0 save_interval=0 wandb.mode=offline wandb.name=instinctrl_a2r_smoke headless=true` | Passed with exit code 0; rollout and checkpoint audits passed. |

### A2-R Go/No-Go Runtime Gate

- `station_keeping_drift_mean <= 1.0 m`
- `station_keeping_drift_p95 <= 2.0 m`
- `anchor_error_mean <= 1.0`
- `tracking_rmse_actual_body_vs_v_cmd <= 0.45 m/s`
- `0.75 <= command_preservation_ratio <= 1.10`
- `command_amplification_rate <= 0.10`
- `safety_collision_rate == 0.0`
- `safety_min_clearance_p05 >= 1.0 m`
- `ics_violation_rate <= 0.005`

---

## Command-Governor Train/Eval Semantic Repair

**Current verdict**: SOURCE/UNIT COMPLETE and corrected 16-frame GPU smoke PASSED. Short diagnostic retrain remains pending.

### Required Semantic Repair Tests

| Test | Required evidence | Current status |
|------|-------------------|----------------|
| SR.1 Task metrics | Actual body velocity conversion, termination reason codes, curriculum schedule, and handbook step metrics are pure-tested | Passed |
| SR.2 Reward target | Formal config uses actual body velocity reward-only tracking; env passes `actual_velocity_b` to reward computer | Passed |
| SR.3 Command source | Default command source is staged `curriculum_generator`, not simple random | Passed |
| SR.4 ICS default | Formal train/eval configs enable ICS by default | Passed |
| SR.5 Eval metrics | Streaming eval exposes `eval/handbook.*` actual tracking, proxy tracking, command preservation, anchor, safety, ICS, and termination metrics | Passed |
| SR.6 Critic semantics | PPO critic no longer consumes legacy `target_rpos` / `target_distance`; actor/governor remains independent of critic-only fields | Passed |

### Actual Commands and Results

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_task_metrics.py training/unit_test/test_instinctrl_rewards.py` | Passed: `16 passed`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_ppo_stability.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_governor.py` | Passed: `27 passed, 11 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/task_metrics.py training/scripts/instinctRL/rewards.py training/scripts/env.py training/scripts/ppo.py training/scripts/utils.py training/scripts/train.py training/scripts/eval.py` | Passed. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_ppo_stability.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py training/unit_test/test_instinctrl_ics.py training/unit_test/test_instinctrl_rewards.py training/unit_test/test_instinctrl_task_metrics.py` | Passed: `84 passed, 11 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python training/scripts/train.py instinctRL.mode=train instinctRL.task=command_governor instinctRL.reward.enabled=true instinctRL.reward.use_privileged_velocity_for_reward=true instinctRL.ics.enabled=true instinctRL.command.source=curriculum_generator env.num_envs=4 env_dyn.num_obstacles=0 algo.training_frame_num=4 algo.num_minibatches=1 algo.training_epoch_num=1 max_frame_num=16 eval_interval=0 save_interval=0 wandb.mode=offline headless=true` | Passed with exit code 0; wrote final checkpoint to `wandb/offline-run-20260709_155509-jqrryl8z/files/checkpoint_final.pt`; rollout and checkpoint audits passed. |

### Scope Boundary

- Complete: train/eval semantic repair source and unit coverage.
- Pending: short diagnostic retrain.
- Not complete: long 8M retrain, full G baseline matrix, paper-level acceptance thresholds.

---

## instinctRL-A2-S: PPO Numerical Stability

**Current verdict**: SOURCE/UNIT READY. Runtime 1M-frame acceptance is pending because local Isaac failed before env import with missing Omniverse/Nucleus assets root.

### Required A2-S Tests

| Test | Required evidence | Current status |
|------|-------------------|----------------|
| A2-S.1 Bounded Beta params | Alpha/beta concentration params finite and within configured min/max for extreme raw outputs | Pure stability test passed |
| A2-S.2 Finite actor action | Random finite observations produce finite normalized actions within action epsilon bounds | Pure stability test passed |
| A2-S.3 NaN actor output fail-fast | NaN raw actor output is caught before governor decoder and writes diagnostic snapshot | Pure stability test passed |
| A2-S.4 Gradient audit | Non-finite gradients fail before optimizer step and write diagnostic snapshot | Pure stability test passed |
| A2-S.5 Parameter audit | Non-finite parameters fail after optimizer step and write diagnostic snapshot | Pure stability test passed |
| A2-S.6 Advantage normalization | Zero-std advantages normalize without NaN using `clamp_min(1e-6)` | Pure stability test passed |
| A2-S.7 Grad clipping coverage | Actor, critic, actor feature extractor, and critic feature extractor are all clipped | Pure stability test passed |
| A2-S.8 Target KL stop | Approximate KL above threshold stops remaining minibatches for the update | Pure stability test passed |
| A2-S.9 No NaN sanitization | Training code does not replace NaN actions with zero | Source-level stability test passed |
| A2-S.10 Runtime acceptance | Conservative learned-governor config completes at least 1,048,576 frames without non-finite action/distribution/loss/gradient/parameter | Pending; blocked locally by Isaac/Nucleus assets root |

### Added / Updated Test Files

- `training/unit_test/test_instinctrl_ppo_stability.py`
- `training/scripts/instinctRL/ppo_stability.py`
- `training/scripts/ppo.py`
- `training/scripts/utils.py`
- `training/cfg/ppo.yaml`

### Actual Commands and Results

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/ppo.py training/scripts/utils.py training/scripts/instinctRL/ppo_stability.py training/scripts/instinctRL/governor.py` | Passed. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_ppo_stability.py` | Passed: `9 passed, 8 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_ppo_stability.py training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_actor_audit.py` | Passed: `22 passed, 12 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_ppo_stability.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py training/unit_test/test_instinctrl_ics.py training/unit_test/test_instinctrl_rewards.py` | Passed: `73 passed, 12 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python training/scripts/train.py instinctRL.mode=train instinctRL.reward.enabled=true env.num_envs=4 env_dyn.num_obstacles=0 algo.training_frame_num=4 algo.num_minibatches=1 algo.training_epoch_num=1 max_frame_num=16 eval_interval=0 save_interval=0 wandb.mode=offline headless=true` | Failed before env import: `Unable to perform Nucleus login on Omniverse. Assets root path is not set.` |

### A2-S Scope Boundary

- Complete in A2-S source/unit: bounded Beta, finite audits, grad clipping, safe advantage normalization, target-KL stop, diagnostic snapshots, tests.
- Pending in A2-S runtime: 1M-frame acceptance on a working Isaac/Nucleus setup.
- Not complete in A2-S: convergence proof, G baseline matrix, H deployment.

---

## instinctRL-A2: Trainable Governor Head and Training Readiness

**Current verdict**: COMPLETE for trainable-governor implementation. Formal long learned-governor training is on hold until A2-S runtime acceptance passes. Training convergence is not proven.

### Required A2 Tests

| Test | Required evidence | Current status |
|------|-------------------|----------------|
| A2.1 Governor bounds | `alpha in [0,1]`, `v_corr` bounded by config, `v_gov_b` norm-clipped | Pure governor test passed |
| A2.2 Governor formula | `v_gov_b = alpha * v_cmd_b + v_corr`, with actor-clean `state_vec` source | Pure governor + source tests passed |
| A2.3 PPO learned forward | 4D `action_normalized`; scalar/vector governor outputs finite and bounded | PPO hybrid test passed |
| A2.4 PPO update | Log-prob/update works on 4D normalized governor action | PPO update smoke passed |
| A2.5 Actor/critic leakage | Perturbing critic-only privileged fields does not alter actor/governor output | PPO separation test passed |
| A2.6 Deterministic action | Mean/deterministic policy output is stable | PPO deterministic test passed |
| A2.7 Checkpoint sanity | Save/load preserves deterministic governor output | PPO checkpoint test and runtime checkpoint audit passed |
| A2.8 Train smoke | Small `instinctRL.mode=train` completes rollout + update + reward stats + checkpoint | GPU smoke passed |
| A2.9 Actor contract | Actor observation remains `lidar_grid + state_vec`; governor does not read privileged `info["v_cmd"]` | Actor audit/source tests passed |

### Added / Updated Test Files

- `training/unit_test/test_instinctrl_governor.py`
- `training/unit_test/test_instinctrl_ppo_hybrid.py`
- `training/unit_test/test_instinctrl_actor_audit.py`

### Actual Commands and Results

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/ppo.py training/scripts/train.py training/scripts/env.py training/scripts/instinctRL/governor.py training/scripts/instinctRL/audit.py training/scripts/utils.py` | Passed. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_actor_audit.py` | Passed: `13 passed, 5 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py training/unit_test/test_instinctrl_ics.py training/unit_test/test_instinctrl_rewards.py` | Passed: `64 passed, 5 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python training/scripts/train.py instinctRL.mode=train instinctRL.reward.enabled=true env.num_envs=4 env_dyn.num_obstacles=0 algo.training_frame_num=4 algo.num_minibatches=1 algo.training_epoch_num=1 max_frame_num=16 eval_interval=0 save_interval=0 wandb.mode=offline headless=true` | Passed with exit code 0. Logged learned-governor wrapper enabled, rollout batch audit pass, checkpoint audit pass, `env_frames=16`, and final checkpoint at `wandb/offline-run-20260705_203852-35lr9uce/files/checkpoint_final.pt`. |

### A2 Scope Boundary

- Complete in A2: trainable governor head, PPO 4D governor action path, train wrapper, training-readiness audit hooks, checkpoint sanity, smoke readiness.
- Not complete in A2: convergence proof, full G baseline/evaluation matrix, adversarial curriculum integration, H real-robot deployment.

---

## instinctRL-A: B0 Smoke Test

**Closeout verdict**: PASS with open verification item(s). Accepted as B0 smoke-test / infrastructure baseline, not learning success.

### Test A.1: Platform Lock Audit
- **Runtime result**: ✅ PASSED (2026-07-04)
- **Evidence**: `PLATFORM AUDIT PASS: drone.model_name='TaslabUAV' | sensor matches Livox MID360 FOV [-7°, 52°] | lidar_range=40m (MID360)`

### Test A.2: Actor Input Audit (Runtime ✅ 2026-07-04)
- **Evidence**: `ACTOR INPUT AUDIT PASS: no forbidden fields in actor observation`

### Test A.3: Action Type Audit (Runtime ✅ 2026-07-04)
- **Evidence**: `ACTION TYPE AUDIT PASS: 3-dim velocity command`

### Test A.4: Environment Reset
- **What**: `env.reset()` returns valid TensorDict with required keys
- **Where**: `train.py` B0 smoke test
- **Pass**: No exception; tensordict contains "info" and "agents" namespaces
- **Fail**: Exception or missing keys

### Test A.5: TASLAB_UAV Spawn
- **What**: Drone spawns via `REGISTRY["TaslabUAV"]` and prim path resolves
- **Where**: Verified by `_resolve_base_link()` in `env.py`
- **Pass**: Base link found and logged; no fallback to root
- **Fail**: "No base_link found" warning

### Test A.6: MID360 Basic Attachment
- **What**: LiDAR raw range tensor available and non-empty
- **Where**: `env.lidar_raw_range` after env step
- **Pass**: Shape > 0, valid return fraction > 0%
- **Fail**: All-zero or missing tensor

### Test A.7: Body→World Velocity Adapter
- **What**: Body-frame v_cmd correctly transformed to world-frame
- **Where**: `BodyToWorldVelocityAdapter.forward()` during smoke test
- **Current status**: Unit test passed; B0/B smoke path passed.
- **Evidence**: `training/unit_test/test_instinctrl_command_adapter.py` covers identity, 90 deg yaw, and roll/pitch cases.
- **Pass**: Known quaternion cases prove body -> world direction; integration smoke shows body X/Y/Z commands map to expected world motion.
- **Fail**: Any yaw/roll/pitch case maps through inverse direction, or only shape/NaN checks are performed.

### Test A.8: VelController Execution
- **What**: World-frame velocity → motor commands via `VelController(LeePositionController)`
- **Where**: `transformed_env.step()` in smoke test
- **Pass**: Motor commands in [-1, 1] range, drone moves
- **Fail**: NaN in motor commands, drone doesn't move

### Test A.9: Smoke Rollout Stability (Runtime ✅ 2026-07-04)
- **Evidence**: `Completed 50/500 ... 500/500 steps.`, no NaN errors

### Test A.10: LiDAR Range Statistics (Runtime ✅ 2026-07-04)
- **Evidence**: `shape=torch.Size([4, 1, 360, 59]), valid=18.97%`

---

## instinctRL-B: Observation / History Buffer

**Current verdict**: COMPLETE. Code fixes, NavRL pytest/PPO validation, and user-side GPU smoke all pass. Smoke mode exits before `SimulationApp.close()` after successful validation to avoid Isaac Kit teardown segfaults after pass.

### Required Before-C Tests

| Test | Required evidence | Current status |
|------|-------------------|----------------|
| B.1 Active MID360 pattern and ray ordering | Active `NavigationEnv` uses Livox MID360 ray ordering or a documented equivalent; no `BpearlPatternCfg` substitute in instinctRL training path | User-side runtime smoke passed MID360 shape/valid-return checks |
| B.2 Ray count and shape stability | Repeated reset/step preserves `[N, H, V]` ray layout and expected ray count | User-side runtime smoke passed with `[4, 1, 360, 59]` raw range |
| B.3 Raw range correctness | `r_i = ||ray_hit_i - lidar_pos||`, not danger-coded inverse range, with max-range handling | Pure observation test passed |
| B.4 Valid-return mask | Mask derives from finite in-range returns and handles max-range/dropout explicitly | Pure observation test passed |
| B.5 Reliability bounds | `w_t` stays in `[0, 1]`; stale/dropout returns are represented correctly | Pure observation test passed |
| B.6 Timestamp monotonicity and frame age | Sim time is monotonic; repeated/stale frames are detectable | Pure observation test passed |
| B.7 History rollover | Fixed window rolls exactly one frame per policy step and resets per env reset | Pure observation test passed |
| B.8 Previous issued action feedback | `prev_action` slots equal prior governor/controller output, not default zeros | Code fixed; pure observation test passed; user-side runtime smoke completed 500 steps |
| B.9 Actor input provenance | Audit proves `lidar_grid` and `state_vec` contain only allowed fields | Runtime actor/schema audit passed |
| B.10 PPO/training-path smoke | `instinctRL.enabled=true` can run a PPO hybrid initialization/forward path, or smoke-only mode is explicitly separated | Mode split implemented; NavRL PPO hybrid forward test passes |

### Added Test Files

- `training/unit_test/test_instinctrl_command_adapter.py`
- `training/unit_test/test_instinctrl_mid360_pattern.py`
- `training/unit_test/test_instinctrl_observation.py`
- `training/unit_test/test_instinctrl_actor_audit.py`
- `training/unit_test/test_instinctrl_ppo_hybrid.py`

### Actual Commands and Results

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd isaac-training && python -m pytest training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_ppo_hybrid.py -q` | Passed: `14 passed, 2 warnings`. Includes regression test for Orbit RayCaster in-place offset on MID360 ray starts. |
| `python3 -m py_compile isaac-training/training/scripts/train.py isaac-training/training/scripts/env.py isaac-training/training/scripts/ppo.py isaac-training/training/scripts/instinctRL/audit.py isaac-training/training/scripts/instinctRL/command_adapter.py isaac-training/training/scripts/instinctRL/observation.py isaac-training/training/scripts/instinctRL/mid360_pattern.py isaac-training/training/unit_test/test_instinctrl_*.py` | Passed. |
| `rg -n "BpearlPatternCfg|patterns\\." isaac-training/training/scripts/env.py isaac-training/training/scripts/instinctRL isaac-training/training/cfg -S` | No matches. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && python - <<'PY' ... dependency probe ... PY` | Passed: activated NavRL resolves Isaac torch 2.0.1, TorchRL/TensorDict, Hydra, WandB, and Click; `ForkingPickler=True`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd isaac-training && python training/scripts/train.py instinctRL.mode=smoke env.num_envs=4 env_dyn.num_obstacles=0` | Reaches CUDA preflight, then fails: no CUDA-capable device visible. |
| `nvidia-smi` | Failed: could not communicate with NVIDIA driver. |
| `conda activate NavRL && python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` | `False`, `0`. |
| User-side GPU smoke after MID360 RayCaster fix | Passed all B0/B checks: 500/500 steps, PPO hybrid forward, actor/schema/action audits, MID360 raw range `[4, 1, 360, 59]`, valid returns `33.04%`; then segfaulted inside `SimulationApp.close()` during Isaac Kit shutdown. |
| User-side post-workaround GPU smoke | Passed: PPO hybrid forward, actor/schema/action audits, 500/500 steps, MID360 raw range `[4, 1, 360, 59]`, valid returns `28.62%`, `B0 Smoke Test PASSED`, `Observation smoke path PASSED`, and success path exited before `SimulationApp.close()`. |

### Before-C Validation

- No remaining B blocker before C.
- When running the smoke command manually, keep `env_dyn.num_obstacles=0` on the same command line, or use shell line continuations (`\`) so Hydra receives it as an override.
- Smoke mode exits before `SimulationApp.close()` after successful validation because Isaac Kit can segfault during shutdown after an otherwise-passed smoke.

## instinctRL-C: Measurement-Space Anchor

**Current verdict**: COMPLETE. Anchor manager unit tests and B+C regression suite pass in the activated NavRL conda environment. Env integration is passive and preserves the actor-clean `lidar_grid` + `state_vec` contract.

### Required C Tests

| Test | Required evidence | Current status |
|------|-------------------|----------------|
| C.1 Config validation | `eps_enter < eps_exit`; `0.0 < min_valid_anchor_fraction <= 1.0`; canonical key rejects `min_valid_fraction`; `huber_delta > 0` | Pure anchor test passed |
| C.2 Null-command hysteresis | Capture at `||v_cmd|| <= eps_enter`; no capture between enter/exit; command reset at `||v_cmd|| >= eps_exit` | Pure anchor test passed |
| C.3 Reset priority | `episode > explicit > command > invalid > none` with fixed enum `0..4` | Pure anchor test passed |
| C.4 Anchor capture | Rising edge freezes `r_star`, bool `m_star`, and `w_star`; later frames do not mutate them | Pure anchor test passed |
| C.5 Reset state rules | Episode reset clears activation count; explicit/command/invalid resets preserve it; all resets clear active anchor and hold steps | Pure anchor test passed |
| C.6 Hold duration | `anchor_hold_steps` is an integer step counter, not seconds | Pure anchor test passed |
| C.7 Mask/weight semantics | `m_t/m_star` are boolean validity; `w_t/w_star` are reliability weights; `w_star` gates usability but not anchor error | Pure anchor test passed |
| C.8 Valid anchor fraction | Fixed structural denominator; inactive reports zero; active below threshold resets invalid; no `sum(m_star)` denominator | Pure anchor test passed |
| C.9 Masked anchor error | `anchor_error = m_t_float * m_star_float * w_t * (r_t - r_star)` | Pure anchor test passed |
| C.10 Huber helper/loss | Pure per-element Huber helper; anchor loss reduced over fixed structural denominator; zero usable beams produce zero/no NaN | Pure anchor test passed |
| C.11 Diagnostics | Public `anchor_error_mean/max` are weighted residual diagnostics over usable beams; reset steps report post-transition inactive metrics | Pure anchor test passed |
| C.12 Structural mask | Optional `[H,V]` structural mask; all-ones default; reject per-env masks; denominator uses structural mask sum | Pure anchor test passed |
| C.13 Fail-fast validation | Bad shapes/devices/dtypes/non-finite inputs fail fast; only `v_cmd [N,1,3] -> [N,3]` is normalized | Pure anchor test passed |
| C.14 Return boundary | `AnchorStepOutput.metrics` contains `[N,1]` public scalar diagnostics; dense tensors are only in `cache` | Pure anchor test passed |
| C.15 Env actor contract | `env.py` writes scalar metrics to `info`; dense cache remains internal; actor obs does not contain `anchor_*` keys | Source-level env integration test passed |

### Added Test File

- `training/unit_test/test_instinctrl_anchor.py`

### Actual Commands and Results

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_anchor.py` | Passed: `11 passed, 1 warning`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py` | Passed: `25 passed, 2 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/anchor.py training/scripts/env.py training/unit_test/test_instinctrl_anchor.py` | Passed. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python - <<'PY' ... TorchRL int64 spec probe ... PY` | Passed: int64 `UnboundedContinuousTensorSpec` is supported for `anchor_reset_reason`. |

### C Scope Boundary

- Complete in C: anchor lifecycle, masked error, robust loss helper, scalar diagnostics, passive env integration, tests.
- Deferred beyond C: anchor reward integration, B3 ablation execution, observability logger, ICS attenuation, reward redesign, training convergence.

## instinctRL-D: Observability Logger

**Current verdict**: COMPLETE. Observability logger unit tests and A/B/C/D regression tests pass in the activated NavRL conda environment. Env integration is passive and disabled by default. Actor observation remains `lidar_grid` + `state_vec`.

### Required D Tests

| Test | Required evidence | Current status |
|------|-------------------|----------------|
| D.1 Config validation | Positive `rank_tol`, finite positive `condition_number_cap`, valid mode only | Pure observability test passed |
| D.2 Proxy mode | `J_i=-normalized_ray_direction_i`; proxy labeled with `is_proxy=1`, `mode_code=0` | Pure observability test passed |
| D.3 Normal mode | `J_i=-n_i`; normals normalized; invalid/near-zero normals excluded; `sqrt(w)` row scaling | Pure observability test passed |
| D.4 Finite-difference mode | `pinv(DeltaP) @ Delta r_i`; K/rank validation; exact and overdetermined synthetic recovery | Pure observability test passed |
| D.5 Mode precedence | Proxy always proxy; offline chooses FD, then normals, then proxy fallback; malformed supplied inputs fail fast | Pure observability test passed |
| D.6 SVD/rank metrics | Full-rank, rank-2, rank-1, insufficient rows, finite capped condition number | Pure observability test passed |
| D.7 Weak direction | Cache-only weak direction from `Vh[-1]`; zero for insufficient/rank-0 cases | Pure observability test passed |
| D.8 Drift correlation helper | Missing drift zero; finite drift norm; absolute projection onto weak direction | Pure observability test passed |
| D.9 Public metrics boundary | Scalar `[N,1]` metrics; dense J/SVD internals in cache only | Pure observability test passed |
| D.10 Env actor contract | `env.py` actor obs block remains only `lidar_grid` and `state_vec`; no observability/J/normal/map/odom actor fields | Source-level env integration test passed |

### Added Test File

- `training/unit_test/test_instinctrl_observability.py`

### Actual Commands and Results

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observability.py` | Passed: `9 passed, 1 warning`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py` | Passed: `34 passed, 2 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/observability.py training/scripts/env.py training/unit_test/test_instinctrl_observability.py` | Passed. |
| TorchRL int64 spec probe for `UnboundedContinuousTensorSpec((1,), dtype=torch.long)` | Passed for `observability_mode_code`. |

### D Scope Boundary

- Complete in D: range-Jacobian/proxy logger, scalar metrics, dense cache, drift projection primitive, passive env integration, tests.
- Deferred beyond D: plot generation, full evaluation report matrix, ICS attenuation, reward integration, training convergence.

## instinctRL-E: ICS Attenuation

**Current verdict**: COMPLETE. ICS attenuator unit tests and A/B/C/D/E regression tests pass in the activated NavRL conda environment. Env/train integration is disabled by default and preserves the actor-clean `lidar_grid` + `state_vec` contract.

### Required E Tests

| Test | Required evidence | Current status |
|------|-------------------|----------------|
| E.1 Config validation | Positive `a_max` and `velocity_limit`; valid clearances; `0 < min_reliability <= 1`; `brake_mode="zero"` only | Pure ICS test passed |
| E.2 Shape/device validation | Accept `[N,L,H,V]` and `[N,L,R]` histories; rays `[R,3]`/`[N,R,3]`; commands `[N,3]`/`[N,1,3]`; malformed inputs fail | Pure ICS test passed |
| E.3 Empty active set | No valid/reliable/closing beams gives beta 1 and preserves command unless clipped | Pure ICS test passed |
| E.4 Emergency bypass | Reliable latest clearance below threshold forces beta 0 and zero final command | Pure ICS test passed |
| E.5 Monotonic beta | Lower clearance or higher speed does not increase beta | Pure ICS test passed |
| E.6 Active set rules | Invalid, low-reliability, non-closing, outside-horizon beams inactive; inside-horizon beams active; ratios clamp to beta 1 | Pure ICS test passed |
| E.7 Range-rate behavior | Finite-difference estimate cached; default flag does not affect beta; enabled flag can activate on negative rate | Pure ICS test passed |
| E.8 Command clipping | Beta computed from unclipped command; final norm clipped; direction preserved; scalar speeds/clip ratio shaped `[N,1]` | Pure ICS test passed |
| E.9 History accessors | Builder and env expose range/mask/weight history; copy protects internals; latest/previous ordering correct | Builder unit + env source-level test passed |
| E.10 Source-level safety | `ics.py` has no privileged deployed dependencies; actor block remains `lidar_grid` + `state_vec`; `train.py` applies ICS before body-to-world adapter and stores `v_final_b` | Source-level integration test passed |

### Added Test File

- `training/unit_test/test_instinctrl_ics.py`

### Actual Commands and Results

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_ics.py` | Passed: `10 passed, 1 warning`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py training/unit_test/test_instinctrl_ics.py` | Passed: `44 passed, 2 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/ics.py training/scripts/instinctRL/observation.py training/scripts/env.py training/scripts/train.py training/unit_test/test_instinctrl_ics.py` | Passed. |

### Runtime Smoke

No Isaac GPU runtime smoke was run for instinctRL-E in this environment. CUDA/NVML is not visible locally, so the optional command `python training/scripts/train.py instinctRL.mode=smoke instinctRL.ics.enabled=true env.num_envs=4 env_dyn.num_obstacles=0` is recorded as skipped here. A later GPU-side smoke should verify live `ics_*` info metrics and attenuated action execution.

### E Scope Boundary

- Complete in E: command attenuation, scalar diagnostics, cache-only dense internals, history accessors, smoke-path integration, and tests.
- Not implemented in E: reward/training changes, actor observation changes, surface-normal/map/odom/SLAM/pose/dynamic-obstacle deployed dependencies, D plotting, training convergence.

### instinctRL-F: Reward Integration

**Current verdict**: COMPLETE for reward integration/readiness. Training convergence is not proven. The trainable governor head remains pending, so F acceptance here is reward path integration and auditability, not learned-governor success.

### Required F Tests

| Test | Required evidence | Current status |
|------|-------------------|----------------|
| F.1 Config validation | Finite non-negative weights, positive `max_reward_abs`, valid clearance thresholds and anchor valid fraction | Pure reward test passed |
| F.2 Tracking reward | `v_final_b == v_cmd_b` is better than mismatch under command-consistency proxy | Pure reward test passed |
| F.3 Beta/emergency gating | Low beta or emergency removes/reduces unsafe tracking penalty and emits ICS compliance offset | Pure reward test passed |
| F.4 Anchor reward | Inactive anchor gives zero; active anchor penalizes anchor loss; low valid fraction masks term | Pure reward test passed |
| F.5 Safety | Lower MID360 clearance gives worse reward; missing/invalid clearance remains finite | Pure reward test passed |
| F.6 Intervention | Lower beta gives larger intervention penalty | Pure reward test passed |
| F.7 Smoothness | Larger final-command jump is penalized | Pure reward test passed |
| F.8 Collision | Collision flag adds large negative term | Pure reward test passed |
| F.9 Total reward | Total equals sum of logged components after clipping/scaling and stays finite | Pure reward test passed |
| F.10 Disabled modules | Anchor/ICS disabled paths use zero/default terms | Pure reward test passed |
| F.11 Actor contract | Reward inputs are not added to actor obs; actor obs remains `lidar_grid` + `state_vec` | Source-level test passed |
| F.12 Privileged boundary | Default config does not require actual velocity; optional actual velocity is labeled reward-only | Pure reward/source test passed |
| F.13 Env integration | Reward components are accumulated in `stats`; old reward path remains when disabled | Source-level test passed |

### Added Test File

- `training/unit_test/test_instinctrl_rewards.py`

### Actual Commands and Results

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_rewards.py` | Passed: `10 passed, 1 warning`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py training/unit_test/test_instinctrl_ics.py training/unit_test/test_instinctrl_rewards.py` | Passed: `54 passed, 2 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/rewards.py training/scripts/env.py training/scripts/instinctRL/__init__.py training/unit_test/test_instinctrl_rewards.py` | Passed. |
| TorchRL spec probe for reward component stats construction before spec expansion | Passed. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/train.py training/scripts/ppo.py && python -m pytest -q training/unit_test/test_instinctrl_rewards.py training/unit_test/test_instinctrl_ppo_hybrid.py` | Passed: `12 passed, 3 warnings`. Covers reward path and PPO minibatch critic-feature recomputation. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python training/scripts/train.py instinctRL.mode=train instinctRL.reward.enabled=true env.num_envs=4 env_dyn.num_obstacles=0 algo.training_frame_num=4 algo.num_minibatches=1 algo.training_epoch_num=1 max_frame_num=16 eval_interval=0 save_interval=0 wandb.mode=offline headless=true` | Passed with exit code 0. Logged `env_frames=16`, PPO loss scalars, actor/schema audits, reward component stats, and final checkpoint at `wandb/offline-run-20260705_191435-pyfkk0z2/files/checkpoint_final.pt`. |

### Runtime Smoke Notes

- The accepted minimal train smoke disables periodic evaluation with `eval_interval=0`; otherwise `i=0` evaluation runs immediately and can consume much more memory than the tiny training rollout.
- The accepted minimal train smoke disables periodic checkpoint saves with `save_interval=0`; the final checkpoint is still saved.
- The known Isaac Kit `SimulationApp.close()` shutdown segfault is avoided for successful instinctRL train completion by exiting after `wandb.finish()` and checkpoint write.
- This smoke validates reward/PPO training readiness, not training convergence or learned-policy success.

### F Scope Boundary

- Complete in F: reward computer, config, env reward switch, component stats logging, actor/privileged-boundary tests.
- Not complete in F: trainable governor head, first stable learned-governor training run, G baseline matrix, H real-robot deployment.

### instinctRL-G: Baselines
- B0–B8 config isolation
- Explicit input-schema logs per baseline
- Required metrics per baseline

### instinctRL-H: Real-Robot Deployment
- No odom/map in actor input
- Latency logs
- Safety logs
- Bag replay audit

## A2-R5J default-off residual pre-emption (2026-07-16)

Contract: `command_closing_i=max(dot(v_gov_b, ray_i),0)`. A range-rate is available only when adjacent range/mask/weight frames are valid and reliable with finite positive `dt`; missing rate evidence is separately cached, never encoded as zero. `closing_i=max(command_closing_i, range_closing_i)`, `required_stop_i=closing_i*latency_sec+closing_i^2/(2*a_max)`, and `residual_i=latest_range_i-collision_clearance_threshold-required_stop_i`. A latest reliable beam is eligible only when `closing_i > approach_eps`; any eligible `residual_i <= residual_margin` pre-empts beta to zero. Evidence-3's global-clearance/worst-speed residual was conservative mixed evidence; R5J uses genuine per-beam residuals.

Coverage includes disabled neutral diagnostics/cache, namespace/config defaults and invalid finite/nonnegative values, non-redundant R5G-like trigger, zero-command rate trigger, opening/single-frame/invalid masks or weights, command-only evidence, empty active set, and explicit rejection of every public/cache R5J key from actor input. `ics_emergency` remains legacy emergency-only.

The exact R5G argv, CUDA preflight, wrapper record, zero-tolerance comparator, and attempt artifacts are under `tests/artifacts/r5j_default_equivalence/20260714_234801/`. Before an attempt directory is created, the wrapper requires empty `git status --porcelain=v1 --untracked-files=all`, resolved `source_commit`, and `source_commit == commit == current HEAD`; it writes a provenance-only `HOLD` best-effort if this fails. The comparator requires that provenance in addition to checkpoint SHA-256, seed, argv, cwd, CUDA, exit, freshness, all eight flattened/per-pass station/tracking R5J summaries as finite exact zero, exact remaining JSON, and exact recomputed gates. The old `attempts/20260716T074648884514Z-0a6a2be/` record contains a non-empty worktree status, so it is a historical dirty/preflight-only `HOLD`; CUDA reported driver failure and torch CUDA false, eval did not start, and no replay JSON exists. Repair verification used the activated NavRL/Isaac Sim Conda environment: py_compile passed; targeted replay coverage passed with `19 passed, 1 warning`; the full suite passed with `163 passed, 13 warnings` (one NVML warning and twelve LazyModule warnings); `git diff --check` passed. No first-party lint/type configuration applies under `training/`; py_compile and pytest are the applicable checks. Evidence-3 remains limited by missing contact-body identity, surface normal, measured deceleration, and final safety-fix proof.
