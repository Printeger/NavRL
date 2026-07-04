# instinctRL-B: Observation / History Buffer

> **Ticket ID**: instinctRL-B
> **Status**: COMPLETE
> **Date**: 2026-07-04
> **Dependencies**: instinctRL-A
> **Blocks**: instinctRL-E, instinctRL-F
> **Risk**: High
> **Handbook**: `instinctRL_Development_Handbook_v1_1_platform_locked.tex` Observation / History Buffer and ticket acceptance criteria
> **Current stage**: instinctRL-C ready

---

## Scope

Finish instinctRL-B without starting instinctRL-C. The work covers the actor-clean MID360 observation path, history buffer, previous issued action feedback, actor schema audit, command-adapter frame test coverage, smoke/train mode separation, and B-stage tests.

This ticket does not implement the instinctRL-C anchor manager.

---

## Acceptance Criteria

Handbook-aligned B acceptance requires:

- Active MID360 ray pattern/ray ordering in the training path.
- Raw range `r_t` as true distance, not danger-coded inverse range.
- Valid-return mask `m_t` from finite in-range returns.
- Reliability weights `w_t` bounded in `[0, 1]`, with invalid beams weighted `0`.
- Timestamp/frame-age handling able to identify stale or repeated frames.
- Allowed IMU cues only: body angular velocity and gravity direction.
- Body-frame `v_cmd` and previous issued governor/controller output in history.
- Fixed-size history rollover and selected-env reset clearing.
- Actor input limited to allowed fields such as `lidar_grid` and `state_vec`.
- PPO hybrid input path initializes and forwards through `lidar_grid` + `state_vec`.
- B runtime smoke passes reset, step, actor audit, no NaN, and MID360 valid returns.

---

## Code Changes

| File | Change Summary |
|------|----------------|
| `training/scripts/instinctRL/mid360_pattern.py` | Added Livox MID360 RayCaster pattern wrapper, mount helpers, and deterministic ray-order hash helper. |
| `training/scripts/env.py` | Replaced instinctRL `BpearlPatternCfg` path with MID360 helper wrapper; added previous issued action storage; clears selected env history/action on reset; passes `prev_action` into the builder. |
| `training/scripts/instinctRL/observation.py` | Requires externally supplied `prev_action`; clamps weights to `[0, 1]`; keeps invalid-beam weights at `0`; tracks frame age and stale frames; supports selected-env history reset. |
| `training/scripts/instinctRL/command_adapter.py` | Corrected body-to-world quaternion rotation semantics; removed incorrect inverse rotation use. |
| `training/scripts/instinctRL/audit.py` | Added hybrid actor schema audit for `lidar_grid` and `state_vec`. |
| `training/scripts/ppo.py` | Flattened critic-only privileged fields before concatenation with `_actor_feature`, fixing the NavRL PPO hybrid forward path. |
| `training/scripts/instinctRL/__init__.py` | Moved observation and MID360 pattern helpers into the active instinctRL module list. |
| `training/scripts/train.py` | Added `instinctRL.mode` split; smoke mode runs B0/B observation checks; train mode runs actor/schema audit and PPO hybrid forward smoke before normal training. |
| `training/cfg/train.yaml` | Added `instinctRL.mode: "smoke"`. |

---

## Test Coverage Added

| Test file | Coverage |
|-----------|----------|
| `training/unit_test/test_instinctrl_command_adapter.py` | Identity, yaw 90 deg, and roll/pitch body-to-world transform cases. |
| `training/unit_test/test_instinctrl_mid360_pattern.py` | MID360 ray count/shape, deterministic order/hash, and sensor-configured count. |
| `training/unit_test/test_instinctrl_observation.py` | True raw range, valid mask, reliability bounds, invalid weights, timestamp/frame age, stale frames, history rollover, selected-env reset, and previous action feedback. |
| `training/unit_test/test_instinctrl_actor_audit.py` | Actor absence/schema tests for forbidden fields and hybrid actor obs keys. |
| `training/unit_test/test_instinctrl_ppo_hybrid.py` | PPO hybrid forward smoke plus actor/critic separation in the activated NavRL environment. |

---

## Code Evidence

- `env.py` no longer imports `patterns` from Orbit sensors and no longer configures `patterns.BpearlPatternCfg` in the inspected active instinctRL path.
- `mid360_pattern.py` returns cloned contiguous ray starts/directions so Orbit RayCaster can apply sensor offsets in-place without hitting overlapping-memory writes from Livox helper `expand()` outputs.
- `MID360ObservationBuilder.build()` raises if `prev_action` is not supplied, preventing silent zero/default history.
- `env.set_prev_issued_action_body()` stores the governor/controller command that was issued before the current environment step.
- `reset_history(env_ids)` clears only selected env history rows while preserving global frame-time continuity.
- `state_vec` remains `history_len * 13`: IMU6 + `v_cmd`3 + `prev_action`3 + frame_age1 per frame.
- `audit.check_actor_schema()` requires exactly `lidar_grid` and `state_vec` in actor observation and validates expected dimensions.
- `ppo.py` now flattens `info.drone_state`, `info.target_rpos`, and `info.target_distance` before critic concatenation. Actor feature extraction still reads only `lidar_grid` and `state_vec`.
- `train.py` separates `instinctRL.mode=smoke` from `instinctRL.mode=train` instead of treating every instinctRL run as a B0-only early return.

---

## Actual Validation Run

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd isaac-training && python -m pytest training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_ppo_hybrid.py -q` | Passed: `14 passed, 2 warnings`. |
| `conda activate NavRL` dependency probe | Passed: `torch 2.0.1+cu118`, `tensordict 0.4.0+3725bcc`, `torchrl 0.4.0+3725bcc`, `click 8.1.3`, `wandb 0.23.1`, `hydra 1.3.2`; `ForkingPickler=True`. |
| `python3 -m py_compile ...` for changed code/tests | Passed. |
| `rg -n "BpearlPatternCfg|patterns\\." isaac-training/training/scripts/env.py isaac-training/training/scripts/instinctRL isaac-training/training/cfg -S` | No matches. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd isaac-training && python training/scripts/train.py instinctRL.mode=smoke env.num_envs=4 env_dyn.num_obstacles=0` | Reaches CUDA preflight, then fails: no CUDA-capable device visible. |
| `nvidia-smi` | Failed: could not communicate with NVIDIA driver. |
| `conda activate NavRL && python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` | `False`, `0`. |
| User-side GPU smoke after MID360 RayCaster fix | Passed all B0/B runtime checks: 500/500 steps, PPO hybrid forward, actor/schema/action audits, MID360 raw range `[4, 1, 360, 59]`, valid returns `33.04%`; then segfaulted inside `SimulationApp.close()` during Isaac Kit shutdown. |
| User-side post-workaround GPU smoke | Passed: PPO hybrid forward, actor/schema/action audits, 500/500 steps, MID360 raw range `[4, 1, 360, 59]`, valid returns `28.62%`, `B0 Smoke Test PASSED`, `Observation smoke path PASSED`, and success path exited before `SimulationApp.close()`. |

---

## Remaining Blockers Before C

None. instinctRL-B acceptance blockers are cleared.

Runtime smoke note:

- A user-side GPU smoke exposed and this branch fixed the Orbit RayCaster in-place offset failure on expanded MID360 ray starts.
- A later user-side GPU smoke passed the B runtime checks and then segfaulted during Isaac Kit shutdown. `instinctRL.mode=smoke` now exits before `SimulationApp.close()` after successful validation so the shell exit status reflects the smoke result.
- The post-workaround smoke output confirms the success path reaches the intentional pre-shutdown exit after the B pass messages.
- If the smoke command is split across shell lines, `env_dyn.num_obstacles=0` must be continued with `\`; otherwise the shell reports `env_dyn.num_obstacles=0: command not found` and Hydra never receives that override.

---

## Verdict

- `instinctRL-A`: PASS with open runtime verification item(s)
- `instinctRL-B`: COMPLETE
- `instinctRL-C`: GO
