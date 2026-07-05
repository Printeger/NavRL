# instinctRL Development Status

> **Last Updated**: 2026-07-05
> **Current Stage**: instinctRL-F reward integration complete
> **Authority order**: code facts > handbook acceptance criteria > devlog records.

---

## Stage Summary

| Field | Value |
|-------|-------|
| **Current stage** | instinctRL-F reward integration complete |
| **Active ticket** | instinctRL-F complete for reward integration/readiness |
| **Next ticket** | instinctRL-G baseline/evaluation harness |
| **Final go/no-go** | instinctRL-F: COMPLETE for reward integration; training convergence NOT PROVEN; instinctRL-G: GO for baseline/evaluation harness only |
| **instinctRL-A** | PASS |
| **instinctRL-B** | COMPLETE |
| **instinctRL-C** | COMPLETE |
| **instinctRL-D** | COMPLETE |
| **instinctRL-E** | COMPLETE |
| **instinctRL-F** | COMPLETE for reward integration/readiness |

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
| instinctRL-F | COMPLETE for reward integration/readiness | Reward path, component stats, actor contract tests, and regression tests pass. Training convergence remains not proven. |
| instinctRL-G | GO for baseline/evaluation harness only | Reward integration passes. Do not claim learned-policy success without training logs. |

---

## Current Code Facts

| Component | Current fact | Status |
|-----------|--------------|--------|
| Config namespace | `instinctRL.enabled=true`, `instinctRL.mode=smoke`, `baseline.id=direct_velocity`, observation and anchor config blocks exist | Present |
| B0 governor | `MinimalGovernor` implements alpha=1, v_corr=0 pass-through | Present |
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
| Env reward integration | `env.py` uses F reward path when `instinctRL.reward.enabled=true`, preserves old NavRL reward when disabled, and writes reward components to `stats` | Present |
| Trainable governor | `MinimalGovernor` remains the only governor; PPO train collector still does not implement learned `(alpha, v_corr)` governor action path | Open gap |

---

## Actual Test Evidence

| Command | Result |
|---------|--------|
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

---

## Final Current Conclusion

- `instinctRL-A`: PASS
- `instinctRL-B`: COMPLETE
- `instinctRL-C`: COMPLETE
- `instinctRL-D`: COMPLETE
- `instinctRL-E`: COMPLETE
- `instinctRL-F`: COMPLETE for reward integration/readiness
- Training convergence: NOT PROVEN
- `instinctRL-G`: GO for baseline/evaluation harness only
