# instinctRL-E: ICS-Inspired Attenuation

> **Ticket ID**: instinctRL-E
> **Status**: COMPLETE
> **Date**: 2026-07-05
> **Dependencies**: instinctRL-B, instinctRL-D
> **Next stage**: instinctRL-F reward-design work
> **Handbook**: `instinctRL_Development_Handbook_v1_1_platform_locked.tex` ICS attenuation requirements

---

## Scope

Implement a deployed-safe, actor-clean command attenuation layer:

`v_final_b = beta * v_gov_b + (1 - beta) * v_brake_b`

E first pass locks `brake_mode="zero"`, so `v_final_b = beta * v_gov_b` followed by norm clipping.

This ticket includes:

- Range-history attenuation from MID360 range/mask/weight history.
- Body-frame ray-direction and body-frame governor-command inputs.
- Active-beam filtering, braking-distance beta, emergency bypass, and optional range-rate filter.
- Scalar public `ics_*` metrics and cache-only dense per-beam diagnostics.
- History accessors for builder/env.
- Smoke-path integration before body-to-world adaptation.
- Unit tests and A/B/C/D/E regression tests.

This ticket does not implement:

- Reward/training integration.
- Actor observation schema changes.
- Surface normals, map, odometry, SLAM, pose, or dynamic-obstacle privileged state in the E deployed path.
- D observability plotting.
- Training convergence.

---

## Files Changed

| File | Change |
|------|--------|
| `training/scripts/instinctRL/ics.py` | New config/output/attenuator module. |
| `training/scripts/instinctRL/observation.py` | Added `MID360ObservationBuilder.get_history(copy=True)`. |
| `training/scripts/env.py` | Added `ics_*` scalar info specs, history wrapper, and dense cache storage helper. |
| `training/scripts/train.py` | Applies ICS before `BodyToWorldVelocityAdapter` when enabled and stores `v_final_body` as previous issued action. |
| `training/scripts/instinctRL/__init__.py` | Marks ICS as active E module. |
| `training/cfg/train.yaml` | Adds disabled-by-default `instinctRL.ics.*` config. |
| `training/unit_test/test_instinctrl_ics.py` | Adds pure PyTorch E behavior and source-boundary tests. |

---

## Acceptance Criteria

- `ICSConfig` validates braking/clearance/velocity/reliability parameters and rejects unsupported brake modes.
- Inputs support `[N,L,H,V]` and `[N,L,R]` histories, `[R,3]` or `[N,R,3]` rays, and `[N,3]` or `[N,1,3]` commands.
- Empty active set returns beta 1.0 by default.
- Reliable emergency clearance below threshold forces beta 0.
- Beta is monotonic with lower clearance and higher speed.
- Invalid, low-reliability, non-closing, and outside-horizon beams stay inactive.
- Range-rate finite difference is cached; it affects beta only when `use_range_rate_filter=true`.
- Beta is computed from the unclipped command; final body command is norm-clipped.
- Public metrics are scalar `[N,1]`; dense tensors are cache-only.
- Actor observation remains only `lidar_grid` and `state_vec`.
- `train.py` applies ICS before body-to-world adaptation and stores `v_final_b` as previous issued action.

---

## Actual Validation

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_ics.py` | Passed: `10 passed, 1 warning`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py training/unit_test/test_instinctrl_ics.py` | Passed: `44 passed, 2 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/ics.py training/scripts/instinctRL/observation.py training/scripts/env.py training/scripts/train.py training/unit_test/test_instinctrl_ics.py` | Passed. |

---

## Runtime Smoke

No Isaac GPU runtime smoke was run for instinctRL-E in this environment. CUDA/NVML is not visible locally. The optional command is:

`python training/scripts/train.py instinctRL.mode=smoke instinctRL.ics.enabled=true env.num_envs=4 env_dyn.num_obstacles=0`

A later GPU-side smoke should verify live attenuation and `ics_*` info metrics.

---

## Final Verdict

- `instinctRL-E`: COMPLETE
- `instinctRL-F`: GO for reward-design work only
- Training convergence: not complete
