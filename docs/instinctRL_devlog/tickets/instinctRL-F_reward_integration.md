# instinctRL-F: Reward Integration and Training Readiness

> **Ticket ID**: instinctRL-F
> **Status**: COMPLETE for reward integration/readiness
> **Date**: 2026-07-05
> **Dependencies**: instinctRL-B, instinctRL-C, instinctRL-E
> **Next stage**: instinctRL-G baseline/evaluation harness
> **Handbook**: `instinctRL_Development_Handbook_v1_1_platform_locked.tex` Reward and Training Plan

---

## Scope

Implement the instinctRL reward path so B/C/E signals can enter reward and logging without changing the actor observation contract.

This ticket includes:

- Tracking reward via command-consistency proxy.
- Anchor reward from scalar anchor diagnostics.
- MID360 clearance safety penalty.
- ICS command-compliance offset.
- Intervention usage penalty.
- Smoothness penalty over final body command proxy.
- Collision penalty.
- Reward component logging through `stats`.
- Actor-contract and privileged-boundary tests.

This ticket does not implement:

- Trainable governor head `(alpha, v_corr)`.
- Stable learned-governor training run.
- instinctRL-G full baseline matrix.
- instinctRL-H real-robot deployment.
- Any actor observation schema change.

---

## Files Changed

| File | Change |
|------|--------|
| `training/scripts/instinctRL/rewards.py` | New reward config/output/computer module. |
| `training/scripts/env.py` | Gated F reward path, reward component stats, old reward fallback, command proxy state. |
| `training/scripts/instinctRL/__init__.py` | Marks rewards as active F module. |
| `training/cfg/train.yaml` | Adds `instinctRL.reward.*`, enabled by default. |
| `training/unit_test/test_instinctrl_rewards.py` | Adds reward math and source-boundary tests. |
| `CONTEXT.md` | Adds F glossary terms. |

---

## Reward Terms Implemented

- `reward_tracking`: command-consistency proxy between `v_cmd_b` and final/issued body command.
- `reward_anchor`: active anchor loss penalty, masked by minimum valid fraction.
- `reward_safety`: low-clearance penalty from latest MID360 range.
- `reward_ics_compliance`: offsets tracking penalty when ICS attenuation/emergency marks command unsafe.
- `reward_intervention`: penalty for low beta.
- `reward_smoothness`: penalty for final-command jumps.
- `reward_collision`: terminal collision penalty.
- `reward_total`: clipped/scaled sum of logged components.

---

## Actual Validation

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_rewards.py` | Passed: `10 passed, 1 warning`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py training/unit_test/test_instinctrl_ics.py training/unit_test/test_instinctrl_rewards.py` | Passed: `54 passed, 2 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/rewards.py training/scripts/env.py training/scripts/instinctRL/__init__.py training/unit_test/test_instinctrl_rewards.py` | Passed. |
| TorchRL spec probe for reward component stats insertion | Passed. |
| CUDA availability probe | `torch.cuda.is_available() == False`, `torch.cuda.device_count() == 0`. |

---

## Runtime Smoke

No Isaac GPU runtime smoke was run for instinctRL-F in this environment. CUDA/NVML is not visible locally. The optional command is:

`python training/scripts/train.py instinctRL.mode=smoke instinctRL.reward.enabled=true env.num_envs=4 env_dyn.num_obstacles=0`

A later GPU-side smoke should verify live reward component stats and controller execution.

---

## Actor and Privileged-State Boundary

- Actor observation remains `lidar_grid` + `state_vec`.
- Reward components are accumulated in `stats`.
- Privileged actual velocity is optional reward-only input and disabled by default.
- Pose, odometry, map, SLAM, surface normals, and dynamic-obstacle privileged state were not added to actor input.

---

## Final Verdict

- `instinctRL-F`: COMPLETE for reward integration/readiness.
- Training convergence: NOT PROVEN.
- `instinctRL-G`: GO for baseline/evaluation harness only.
