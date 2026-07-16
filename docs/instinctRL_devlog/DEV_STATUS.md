# instinctRL Development Status

> **Last Updated**: 2026-07-16
> **Current Stage**: A2-R5J CUDA HOLD; disabled replay was not invoked
> **Authority order**: code facts > handbook acceptance criteria > devlog records.

---

## Stage Summary

| Field | Value |
|-------|-------|
| **Current stage** | `8298a7d` provenance repair and `c2e8367` synchronization are pushed; fresh CUDA was not ready, so this turn is `HOLD` without a replay attempt |
| **Active ticket** | A2-R5J braking-residual mechanism |
| **Next ticket** | No runtime action in this turn. A future authorized turn must start with a fresh clean-worktree CUDA check before considering the single disabled wrapper |
| **Final go/no-go** | Enabled R5J behavior evaluation and all training remain HOLD until source/unit/audit/default-equivalence gates pass |
| **instinctRL-A** | PASS |
| **instinctRL-B** | COMPLETE |
| **instinctRL-C** | COMPLETE |
| **instinctRL-D** | COMPLETE |
| **instinctRL-E** | COMPLETE |
| **instinctRL-F** | COMPLETE for reward integration/readiness |
| **instinctRL-A2** | COMPLETE for trainable governor and training-readiness smoke |
| **instinctRL-A2-S** | SOURCE/UNIT READY; runtime acceptance pending |
| **instinctRL-A2-R** | SOURCE/UNIT COMPLETE; runtime diagnostic retrain pending |
| **instinctRL-A2-R2** | SOURCE/UNIT COMPLETE; superseded by later failed sweep evidence |
| **instinctRL-A2-R3** | SOURCE/UNIT COMPLETE; superseded by the R4/R5 investigation path |
| **instinctRL-A2-R5** | SWEEPS STOPPED; no R5 candidate passed all 14 gates; Evidence-1/2/3 completed |
| **instinctRL-A2-R5J** | HOLD; default-off source/tests complete, disabled replay blocked by NVIDIA-driver failure |
| **train/eval semantic repair** | SOURCE/UNIT COMPLETE; corrected 16-frame GPU smoke passed |

---

## Acceptance Status

| Ticket | Verdict | Notes |
|--------|---------|-------|
| instinctRL-0 | Accepted as prior platform audit baseline | Earlier audit remains useful context, but current acceptance is judged against active code. |
| instinctRL-A | PASS | Accepted as B0 smoke-test / infrastructure baseline, not learning success. |
| instinctRL-B | COMPLETE | MID360 observation/history, actor schema, previous-action feedback, PPO hybrid path, and B runtime smoke evidence are complete. |
| instinctRL-C | COMPLETE | `MeasurementSpaceAnchorManager` is implemented with actor-clean inputs, null-command hysteresis, anchor capture/reset, masked error, fixed-denominator Huber loss, scalar info diagnostics, internal dense cache, and NavRL pytest coverage. |
| instinctRL-D | COMPLETE | Evaluation-only observability logger exists with offline finite-difference, offline normal-mode, proxy mode, scalar metrics, cache-only dense internals, and NavRL pytest coverage. |
| instinctRL-E | COMPLETE | ICS-inspired attenuation exists, unit/regression tests pass, actor contract remains clean, no reward/training implementation added. |
| instinctRL-F | COMPLETE for reward integration/readiness; minimal train smoke passed | Reward path, component stats, actor contract tests, regression tests, and a 16-frame GPU training smoke pass. Training convergence remains not proven. |
| instinctRL-A2 | COMPLETE for trainable governor/readiness; learned-governor train smoke passed | PPO actor now outputs 4D normalized governor action `(alpha, v_corr)`; train wrapper applies body-frame governor output before controller; rollout/checkpoint audits pass. Training convergence remains not proven. |
| instinctRL-A2-S | SOURCE/UNIT READY; runtime acceptance pending | PPO now has bounded Beta concentrations, finite tensor/gradient/parameter audits, grad clipping for all PPO modules, safe advantage normalization, target-KL early stop, and diagnostic snapshots. Runtime smoke was blocked before env import by missing Omniverse/Nucleus assets root. |
| instinctRL-A2-R | SOURCE/UNIT COMPLETE; runtime diagnostic retrain pending | Reward/curriculum now explicitly targets null-command station-keeping and command-amplification control. New 1M static MID360 retrain required before long training. |
| instinctRL-A2-R2 | SOURCE/UNIT COMPLETE; superseded by A2-R3 | Decoder-level hard null prior, preservation band rewards, hard gate scorer, and dry-run-first sweep runner were implemented. The `20260711_111713` sweep failed all six candidates, so R2 is diagnostic evidence only. |
| instinctRL-A2-R3 | SOURCE/UNIT COMPLETE; runtime sweep failed and was superseded | Soft null-command station correction was implemented, but the `20260711_144051` sweep produced no passing candidate; best was `r3_soft_null_min025` at `7/14`, `passed=false`, `safety_passed=false`. Later R4/R5 work supersedes it. |
| instinctRL-A2-R5 | BLOCKED / STOP SWEEPS | R5G best was `r5g_downatten_z010` at `6/14`, `passed=false`, `safety_passed=false`. R5H/R5I and Evidence-1/2/3 closed parameter tuning and identified a braking-residual mechanism gap for bounded R5J planning only. |
| instinctRL-A2-R5J | HOLD; CUDA unavailable | Provenance repair `8298a7d` and synchronization `c2e8367` are pushed. Fresh `nvidia-smi` failed (exit `9`) and Torch reported CUDA unavailable, so no wrapper, attempt directory, or replay JSON was created. No enabled behavior run, sweep, 1M, warm-start, promotion, or formal training. |
| instinctRL-G | GO for baseline/evaluation harness only | Reward integration passes. Do not claim learned-policy success without training logs. |

---

## Current Code Facts

| Component | Current fact | Status |
|-----------|--------------|--------|
| Config namespace | `instinctRL.enabled=true`, `instinctRL.mode=smoke`, `baseline.id=direct_velocity`, observation and anchor config blocks exist | Present |
| B0 governor | `MinimalGovernor` implements alpha=1, v_corr=0 pass-through and remains the smoke/direct baseline support path | Present |
| Trainable governor | `TrainableGovernorDecoder` maps 4D normalized actor action to bounded `alpha`, bounded `v_corr`, and norm-clipped body-frame `v_gov_b` using actor-clean `state_vec` only | Present |
| Command adapter | Body-to-world rotation is covered by identity/yaw/roll-pitch unit tests | Present |
| Observation builder | Builds range/mask/weight/IMU/v_cmd/prev_action/history tensors; requires real `prev_action` | Present |
| Active sensor pattern | Active instinctRL env path uses MID360 helper wrapper, not `BpearlPatternCfg` | Present |
| PPO hybrid input | `ppo.py` consumes `lidar_grid` and `state_vec`; critic privileged fields stay in critic branch | Present |
| Actor audit | Key scan plus hybrid schema audit | Present |
| Anchor manager | `instinctRL/anchor.py` implements vectorized state, reset priority, `w_star`, structural mask, Huber helper, and public metrics/cache separation | Present |
| Env anchor integration | `env.py` writes only scalar `anchor_*` diagnostics into `info`; dense cache is stored in `self.anchor_outputs`; actor obs remains `lidar_grid` + `state_vec` | Present |
| Observability logger | `instinctRL/observability.py` computes proxy/normal/finite-difference observability metrics and keeps dense Jacobian/SVD internals in cache | Present |
| Env observability integration | `env.py` writes only scalar `observability_*` diagnostics into `info` when enabled; dense cache is stored in `self.observability_outputs`; actor obs remains `lidar_grid` + `state_vec` | Present |
| ICS attenuator | `instinctRL/ics.py` implements range-history command attenuation with zero brake mode, emergency bypass, range-rate cache, scalar metrics, and cache-only dense per-beam internals | Present |
| Env/train ICS integration | `env.py` exposes MID360 history and scalar `ics_*` info specs; `train.py` applies ICS before body-to-world adaptation and stores `v_final_b` as previous issued action when enabled | Present |
| Reward computer | `instinctRL/rewards.py` implements tracking, anchor, safety, ICS-compliance, intervention, smoothness, collision, and clipped total reward terms | Present |
| Station objective repair | Reward includes null-command actual-speed/output penalties, safe-command proxy tracking, and command-amplification penalty; formal config uses `station_first` curriculum | Present |
| Objective hardening | Reward includes preservation-low/high band penalties gated off during ICS intervention; trainable governor now uses a soft null-command correction prior with `null_vcorr_gate_min=0.25` | Present |
| Station correction repair | Reward distinguishes null-command output bias from anchor-active station correction; high anchor loss relaxes output-bias penalty while actual null-command speed remains penalized | Present |
| Automated hard gate | `training/scripts/instinctRL/gates.py` scores short diagnostic eval JSON against station, tracking, preservation, safety, ICS, and termination thresholds; null-command output speed is diagnostic-only in A2-R3 | Present |
| Corrective sweep runner | `training/scripts/instinctRL/sweep.py` generates A2-R3 `r3_*` short corrective sweep train/eval commands and defaults to dry-run unless `--execute` is passed | Present |
| Env reward integration | `env.py` uses F reward path when `instinctRL.reward.enabled=true`, preserves old NavRL reward when disabled, and writes reward components to `stats` | Present |
| Command-governor task semantics | `instinctRL.task=command_governor`; `legacy_reach_goal` is diagnostic only; explicit below/above/collision/timeout termination stats exist | Present |
| Actual-velocity tracking reward | Formal config sets `use_privileged_velocity_for_reward=true`; env passes actual body velocity to reward-only tracking while actor input remains clean | Present |
| Command curriculum | Env wires `AdversarialCommandGenerator` through staged command probabilities; `scripted_eval` and `basic_random` remain explicit alternatives | Present |
| Handbook eval metrics | Streaming eval emits `eval/handbook.*` actual tracking, proxy tracking, command preservation, anchor, safety, ICS, and termination metrics | Present |
| PPO train update smoke fix | `PPO._update()` recomputes critic features inside minibatch update instead of relying on cached `_critic_feature`; regression test covers missing-cache minibatches | Present |
| instinctRL train smoke controls | `train.py` initializes wandb for `instinctRL.mode=train`, supports `eval_interval=0` and `save_interval=0`, and exits before `SimulationApp.close()` after successful instinctRL train completion to avoid Isaac Kit shutdown segfault | Present |
| Train policy wrapper | `InstinctRLTrainPolicy` wraps PPO during collection/eval, converts `v_gov_b` through optional ICS and body-to-world adapter, stores `v_final_b` for previous-action/reward feedback, and leaves PPO update on 4D normalized governor actions | Present |
| Training-readiness audit hooks | `audit_policy_init`, `audit_rollout_batch`, and `audit_checkpoint_file` fail hard on actor/governor/rollout/checkpoint violations | Present |
| PPO numerical stability hardening | `BetaActor` concentration parameters are bounded; PPO finite-audits observations/actions/log-probs/entropy/value/returns/advantages/losses/ratio/gradients/parameters; all PPO module groups are gradient-clipped; target KL early-stop and diagnostic snapshots exist | Present |

---

## Actual Test Evidence

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_rewards.py training/unit_test/test_instinctrl_gates.py` | Passed: `27 passed`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_*.py` | Passed: `101 passed, 11 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python training/scripts/instinctRL/sweep.py --frames 131072 --seeds 0 --limit 6` | Passed dry-run: emitted six `a2r3_sweep` jobs for the `r3_*` variants without launching training. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/governor.py training/scripts/instinctRL/rewards.py training/scripts/instinctRL/gates.py training/scripts/instinctRL/sweep.py training/scripts/ppo.py training/scripts/env.py && python -m pytest -q training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_rewards.py training/unit_test/test_instinctrl_gates.py` | Passed: `25 passed`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_*.py` | Passed: `99 passed, 11 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_rewards.py training/unit_test/test_instinctrl_task_metrics.py training/unit_test/test_instinctrl_eval_diagnostic.py` | Passed: `24 passed`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_*.py` | Passed: `92 passed, 11 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python training/scripts/train.py instinctRL.mode=train instinctRL.task=command_governor instinctRL.reward.enabled=true instinctRL.reward.use_privileged_velocity_for_reward=true instinctRL.ics.enabled=true instinctRL.command.source=curriculum_generator instinctRL.command.curriculum_profile=station_first env.num_envs=4 env.num_obstacles=20 env_dyn.num_obstacles=0 algo.training_frame_num=4 algo.num_minibatches=1 algo.training_epoch_num=1 max_frame_num=16 eval_interval=0 save_interval=0 wandb.mode=offline wandb.name=instinctrl_a2r_smoke headless=true` | Passed with exit code 0; rollout/checkpoint audits passed. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_anchor.py` | Passed: `11 passed, 1 warning`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py` | Passed: `25 passed, 2 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/anchor.py training/scripts/env.py training/unit_test/test_instinctrl_anchor.py` | Passed. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python - <<'PY' ... TorchRL int64 spec probe ... PY` | Passed: `anchor_reset_reason` can use int64 `UnboundedContinuousTensorSpec`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observability.py` | Passed: `9 passed, 1 warning`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py` | Passed: `34 passed, 2 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/observability.py training/scripts/env.py training/unit_test/test_instinctrl_observability.py` | Passed. |
| TorchRL int64 spec probe for `observability_mode_code` | Passed. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_ics.py` | Passed: `10 passed, 1 warning`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py training/unit_test/test_instinctrl_ics.py` | Passed: `44 passed, 2 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/ics.py training/scripts/instinctRL/observation.py training/scripts/env.py training/scripts/train.py training/unit_test/test_instinctrl_ics.py` | Passed. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_rewards.py` | Passed: `10 passed, 1 warning`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py training/unit_test/test_instinctrl_ics.py training/unit_test/test_instinctrl_rewards.py` | Passed: `54 passed, 2 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/rewards.py training/scripts/env.py training/scripts/instinctRL/__init__.py training/unit_test/test_instinctrl_rewards.py` | Passed. |
| TorchRL spec probe for reward component stats insertion | Passed. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/train.py training/scripts/ppo.py && python -m pytest -q training/unit_test/test_instinctrl_rewards.py training/unit_test/test_instinctrl_ppo_hybrid.py` | Passed: `12 passed, 3 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python training/scripts/train.py instinctRL.mode=train instinctRL.reward.enabled=true env.num_envs=4 env_dyn.num_obstacles=0 algo.training_frame_num=4 algo.num_minibatches=1 algo.training_epoch_num=1 max_frame_num=16 eval_interval=0 save_interval=0 wandb.mode=offline headless=true` | Passed with exit code 0. Logged `env_frames=16`, PPO loss scalars, actor/schema audits, reward component stats, and final checkpoint at `wandb/offline-run-20260705_191435-pyfkk0z2/files/checkpoint_final.pt`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/ppo.py training/scripts/train.py training/scripts/env.py training/scripts/instinctRL/governor.py training/scripts/instinctRL/audit.py training/scripts/utils.py` | Passed. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_actor_audit.py` | Passed: `13 passed, 5 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py training/unit_test/test_instinctrl_ics.py training/unit_test/test_instinctrl_rewards.py` | Passed: `64 passed, 5 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python training/scripts/train.py instinctRL.mode=train instinctRL.reward.enabled=true env.num_envs=4 env_dyn.num_obstacles=0 algo.training_frame_num=4 algo.num_minibatches=1 algo.training_epoch_num=1 max_frame_num=16 eval_interval=0 save_interval=0 wandb.mode=offline headless=true` | Passed with exit code 0 after A2. Logged learned-governor wrapper enabled, rollout batch audit pass, checkpoint audit pass, `env_frames=16`, and final checkpoint at `wandb/offline-run-20260705_203852-35lr9uce/files/checkpoint_final.pt`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/ppo.py training/scripts/utils.py training/scripts/instinctRL/ppo_stability.py training/scripts/instinctRL/governor.py` | Passed after A2-S stability hardening. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_ppo_stability.py training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_actor_audit.py` | Passed: `22 passed, 12 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_ppo_stability.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py training/unit_test/test_instinctrl_ics.py training/unit_test/test_instinctrl_rewards.py` | Passed: `73 passed, 12 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python training/scripts/train.py instinctRL.mode=train instinctRL.reward.enabled=true env.num_envs=4 env_dyn.num_obstacles=0 algo.training_frame_num=4 algo.num_minibatches=1 algo.training_epoch_num=1 max_frame_num=16 eval_interval=0 save_interval=0 wandb.mode=offline headless=true` | Failed before env import due local Isaac/Nucleus setup: `Unable to perform Nucleus login on Omniverse. Assets root path is not set.` No PPO finite-audit failure was reached. |

---

## Final Current Conclusion

- `instinctRL-A`: PASS
- `instinctRL-B`: COMPLETE
- `instinctRL-C`: COMPLETE
- `instinctRL-D`: COMPLETE
- `instinctRL-E`: COMPLETE
- `instinctRL-F`: COMPLETE for reward integration/readiness; minimal 16-frame training smoke passed
- `instinctRL-A2`: COMPLETE for trainable governor and training-readiness audit hooks; learned-governor 16-frame training smoke passed
- `instinctRL-A2-S`: SOURCE/UNIT READY for PPO numerical-stability hardening; runtime acceptance pending
- `instinctRL-A2-R2`: SOURCE/UNIT COMPLETE but superseded by failed `20260711_111713` sweep evidence
- `instinctRL-A2-R3`: SOURCE/UNIT COMPLETE; the `20260711_144051` runtime sweep failed all six candidates and was superseded by R4/R5
- `instinctRL-A2-R5`: all R5 sweeps stopped; best R5G result was `6/14`, and no candidate is promotable
- `instinctRL-A2-R5J`: HOLD; default-off source/tests are complete, provenance repair `8298a7d` and synchronization `c2e8367` are pushed to `origin/a2-r5j-default-off-residual`, but fresh CUDA was unavailable and no replay was invoked
- Enabled R5J behavior evaluation, 128k/1M/formal training, warm-start, and promotion: HOLD until R5J source/unit/audit/default-equivalence gates pass
- Training convergence: NOT PROVEN
- `instinctRL-G`: GO for baseline/evaluation harness only

### A2-R5J implementation update (2026-07-16)

- Implemented the actor-clean, default-off per-beam residual pre-emption guard in `training/scripts/instinctRL/ics.py`; config defaults are off in both train/eval YAMLs and new scalar diagnostics are info-only.
- Provenance repair commit `8298a7d256bec6a82dee49d9af41a87628135ed6` (`Close R5J replay provenance gaps`) was confirmed at the pre-documentation-sync `HEAD` and `origin/a2-r5j-default-off-residual`; that worktree was clean.
- Fresh repair verification in the Isaac Sim Conda environment passed: `python -m py_compile training/scripts/instinctRL/ics.py training/unit_test/test_instinctrl_r5j_replay.py ../docs/instinctRL_devlog/tests/artifacts/r5j_default_equivalence/20260714_234801/replay_wrapper.py ../docs/instinctRL_devlog/tests/artifacts/r5j_default_equivalence/20260714_234801/compare_disabled_replay.py` exited `0`; `test_instinctrl_r5j_replay.py` passed `19` tests with `1` NVML warning; `test_instinctrl_*.py` passed `163` tests with `13` warnings (one NVML and twelve LazyModule); `git diff --check` exited `0`.
- Fresh clean-worktree CUDA decision: `nvidia-smi` exited `9` with NVIDIA-driver communication failure; activated NavRL Python exited `0` and printed `torch.cuda.is_available() = False` and `torch.cuda.device_count() = 0` (plus the expected NVML initialization warning). The wrapper was not invoked; no new attempts directory or replay JSON exists. Final verdict: `HOLD`.
- The historical attempt `tests/artifacts/r5j_default_equivalence/20260714_234801/attempts/20260716T074648884514Z-0a6a2be/` is a dirty-worktree/preflight-only `HOLD`, not an eligible replay. Its CUDA check recorded `nvidia-smi` exit 9 and `torch.cuda.is_available() == false`; eval did not start and no replay JSON exists. The repaired wrapper now rejects that condition before a runtime attempt is made.
- `training/scripts/instinctRL/sweep.py` contains the R5G variants, but R5 sweeps are stopped. R5J enabled replay, dry-run, training, sweep, 1M, promotion, and main integration remain blocked until the disabled equivalence replay can run and pass exactly.
