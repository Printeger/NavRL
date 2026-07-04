# instinctRL Development Status

> **Last Updated**: 2026-07-04
> **Current Stage**: instinctRL-C ready
> **Authority order**: code facts > handbook acceptance criteria > devlog records.

---

## Stage Summary

| Field | Value |
|-------|-------|
| **Current stage** | instinctRL-C ready |
| **Active ticket** | instinctRL-C may start |
| **Next ticket** | instinctRL-C measurement-space anchor manager |
| **Final go/no-go** | instinctRL-C: GO |
| **instinctRL-A** | PASS |
| **instinctRL-B** | COMPLETE |

---

## Acceptance Status

| Ticket | Verdict | Notes |
|--------|---------|-------|
| instinctRL-0 | Accepted as prior platform audit baseline | Earlier audit remains useful context, but current B acceptance is judged against active code. |
| instinctRL-A | PASS | Accepted as B0 smoke-test / infrastructure baseline, not learning success. Adapter frame direction is unit-tested and covered by the B0 smoke command path. |
| instinctRL-B | COMPLETE | Code blockers are addressed. NavRL pytest/PPO hybrid validation passes. User-side GPU smoke completed 500/500 B0/B steps, actor/schema/action audits, PPO hybrid forward, MID360 `[4, 1, 360, 59]` returns, and shutdown workaround exits before Isaac Kit teardown segfault. |
| instinctRL-C | GO | B acceptance blockers are cleared. C may start, but C implementation must stay within the handbook anchor-manager scope. |

---

## Current B-Fix State

| ID | Former blocker | Current code fact | Remaining acceptance requirement |
|----|----------------|-------------------|----------------------------------|
| B-FIX-001 | Active env RayCaster used `patterns.BpearlPatternCfg` | `env.py` now uses `instinctRL.mid360_pattern.create_mid360_pattern_cfg()`; active smoke prints `LivoxMid360Pattern rays=21240 shape=(360, 59)`. | Complete |
| B-FIX-002 | Body-to-world adapter frame direction was wrong/unverified | `BodyToWorldVelocityAdapter` uses body-to-world quaternion rotation; identity, yaw 90 deg, and roll/pitch unit tests pass; adapter path is exercised by B0 smoke action audit. | Complete |
| B-FIX-003 | `instinctRL.enabled=true` returned after B0 smoke only | `instinctRL.mode` separates `smoke` and `train`; PPO hybrid forward is covered by NavRL pytest and smoke. | Complete |
| B-FIX-004 | `prev_action` was not fed from issued governor output | `env.set_prev_issued_action_body()` stores the previous issued body-frame command; observation tests verify feedback; smoke completed 500 steps. | Complete |
| B-FIX-005 | Actor audit scanned key names only | `audit.py` includes `check_actor_schema()`; runtime actor/schema audit passed on real `TensorDict`. | Complete |
| B-FIX-006 | B tests were missing | NavRL pytest passes all added B tests, including PPO hybrid forward, actor/critic separation, and RayCaster in-place offset regression coverage. | Complete |

---

## Actual Test Evidence

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd isaac-training && python -m pytest training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_ppo_hybrid.py -q` | Passed: `14 passed, 2 warnings`. |
| `conda activate NavRL` dependency probe | Passed: `torch 2.0.1+cu118`, `tensordict 0.4.0+3725bcc`, `torchrl 0.4.0+3725bcc`, `click 8.1.3`, `wandb 0.23.1`, `hydra 1.3.2`; `ForkingPickler=True`. |
| `python3 -m py_compile ...` for changed code/tests | Passed. |
| `rg -n "BpearlPatternCfg|patterns\\." ...` | No matches in the inspected active instinctRL env/cfg paths. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd isaac-training && python training/scripts/train.py instinctRL.mode=smoke env.num_envs=4 env_dyn.num_obstacles=0` | Reaches CUDA preflight, then fails: no CUDA-capable device visible. |
| `nvidia-smi` | Failed: could not communicate with NVIDIA driver. |
| `conda activate NavRL && python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` | `False`, `0`. |
| User-side post-workaround GPU smoke | Passed: PPO hybrid forward, actor/schema/action audits, 500/500 steps, MID360 raw range `[4, 1, 360, 59]`, valid returns `28.62%`, `B0 Smoke Test PASSED`, `Observation smoke path PASSED`, and smoke success path exited before `SimulationApp.close()`. |

---

## Clean Status Table

| Component | Current fact | Status |
|-----------|--------------|--------|
| Config namespace | `instinctRL.enabled=true`, `instinctRL.mode=smoke`, `baseline.id=direct_velocity`, observation history config exists | Present |
| B0 governor | `MinimalGovernor` implements alpha=1, v_corr=0 pass-through | Present |
| Command adapter | Corrected body-to-world rotation and covered by pure unit tests | Present, user-side runtime checks passed |
| Observation builder | Builds range/mask/weight/IMU/v_cmd/prev_action/history tensors; requires real `prev_action` | Present, pure tests pass |
| Active sensor pattern | Active instinctRL env path uses MID360 helper wrapper, not `BpearlPatternCfg` | Present, user-side runtime checks passed |
| PPO hybrid input | `ppo.py` consumes `lidar_grid` and `state_vec`; critic privileged fields are flattened before concatenation | Present, NavRL pytest passes |
| Actor audit | Key scan plus hybrid schema audit | Present, provenance remains by schema/code review rather than full taint proof |
| Training path | Explicit smoke/train mode split; smoke success exits before Isaac Kit shutdown | Present, user-side runtime checks passed |

---

## Superseded Records

Earlier 2026-07-04 devlog entries that marked instinctRL-B as partial or kept instinctRL-C blocked are superseded. The current truthful conclusion is:

- `instinctRL-A`: PASS
- `instinctRL-B`: COMPLETE
- `instinctRL-C`: GO
