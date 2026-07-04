# instinctRL Development Status

> **Last Updated**: 2026-07-04
> **Current Stage**: B-closeout / Isaac runtime validation before instinctRL-C
> **Authority order**: code facts > handbook acceptance criteria > devlog records.

---

## Stage Summary

| Field | Value |
|-------|-------|
| **Current stage** | B-closeout / Isaac runtime validation before instinctRL-C |
| **Active ticket** | instinctRL-B runtime validation |
| **Next ticket** | instinctRL-C only after Isaac runtime smoke passes |
| **Final go/no-go** | instinctRL-C: NO-GO |
| **instinctRL-A** | PASS with open runtime verification item(s) |
| **instinctRL-B** | PARTIAL / NOT FULLY ACCEPTED |

---

## Acceptance Status

| Ticket | Verdict | Notes |
|--------|---------|-------|
| instinctRL-0 | Accepted as prior platform audit baseline | Earlier audit remains useful context, but current B acceptance is judged against active code. |
| instinctRL-A | PASS with open runtime verification item(s) | Accepted as B0 smoke-test / infrastructure baseline, not learning success. Adapter frame direction has a pure unit test; full runtime smoke must still run in a provisioned Isaac environment. |
| instinctRL-B | PARTIAL / NOT FULLY ACCEPTED | Code blockers are addressed. NavRL pytest/PPO hybrid validation passes. Full handbook acceptance is blocked only by local inability to run Isaac runtime smoke because no CUDA-capable GPU/driver is visible. |
| instinctRL-C | NO-GO | Do not start until B validation checklist passes. |

---

## Current B-Fix State

| ID | Former blocker | Current code fact | Remaining acceptance requirement |
|----|----------------|-------------------|----------------------------------|
| B-FIX-001 | Active env RayCaster used `patterns.BpearlPatternCfg` | `env.py` now uses `instinctRL.mid360_pattern.create_mid360_pattern_cfg()` for the active instinctRL RayCaster path; local search found no `BpearlPatternCfg` use in `training/scripts/env.py`, `training/scripts/instinctRL`, or `training/cfg`. | Run Isaac runtime smoke proving active MID360 returns and stable ray layout in the real env. |
| B-FIX-002 | Body-to-world adapter frame direction was wrong/unverified | `BodyToWorldVelocityAdapter` now uses body-to-world quaternion rotation; identity, yaw 90 deg, and roll/pitch unit tests pass. | Include adapter path in runtime smoke. |
| B-FIX-003 | `instinctRL.enabled=true` returned after B0 smoke only | `instinctRL.mode` now separates `smoke` and `train`; PPO hybrid forward is covered by NavRL pytest. | Run `instinctRL.mode=smoke` runtime smoke in a GPU-visible Isaac environment. |
| B-FIX-004 | `prev_action` was not fed from issued governor output | `env.set_prev_issued_action_body()` stores the previous issued body-frame command; `observation.py` now requires `prev_action` and unit tests verify feedback. | Confirm in runtime smoke across reset/step boundaries. |
| B-FIX-005 | Actor audit scanned key names only | `audit.py` now includes `check_actor_schema()` for the hybrid actor input shape/key contract; pure actor schema tests pass. | Runtime actor audit must pass on real `TensorDict` from `NavigationEnv`. |
| B-FIX-006 | B tests were missing | NavRL pytest now passes all added B tests, including PPO hybrid forward and actor/critic separation. | Run the Isaac runtime smoke once GPU/driver visibility is restored. |

---

## Actual Test Evidence

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd isaac-training && python -m pytest training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_ppo_hybrid.py -q` | Passed: `13 passed, 2 warnings`. |
| `conda activate NavRL` dependency probe | Passed: `torch 2.0.1+cu118`, `tensordict 0.4.0+3725bcc`, `torchrl 0.4.0+3725bcc`, `click 8.1.3`, `wandb 0.23.1`, `hydra 1.3.2`; `ForkingPickler=True`. |
| `python3 -m py_compile ...` for changed code/tests | Passed. |
| `rg -n "BpearlPatternCfg|patterns\\." ...` | No matches in the inspected active instinctRL env/cfg paths. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd isaac-training && python training/scripts/train.py instinctRL.mode=smoke env.num_envs=4 env_dyn.num_obstacles=0` | Reaches CUDA preflight, then fails: no CUDA-capable device visible. |
| `nvidia-smi` | Failed: could not communicate with NVIDIA driver. |
| `conda activate NavRL && python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` | `False`, `0`. |

---

## Clean Status Table

| Component | Current fact | Status |
|-----------|--------------|--------|
| Config namespace | `instinctRL.enabled=true`, `instinctRL.mode=smoke`, `baseline.id=direct_velocity`, observation history config exists | Present |
| B0 governor | `MinimalGovernor` implements alpha=1, v_corr=0 pass-through | Present |
| Command adapter | Corrected body-to-world rotation and covered by pure unit tests | Present, runtime smoke pending |
| Observation builder | Builds range/mask/weight/IMU/v_cmd/prev_action/history tensors; requires real `prev_action` | Present, pure tests pass |
| Active sensor pattern | Active instinctRL env path uses MID360 helper wrapper, not `BpearlPatternCfg` | Present, runtime smoke pending |
| PPO hybrid input | `ppo.py` consumes `lidar_grid` and `state_vec`; critic privileged fields are flattened before concatenation | Present, NavRL pytest passes |
| Actor audit | Key scan plus hybrid schema audit | Present, provenance remains by schema/code review rather than full taint proof |
| Training path | Explicit smoke/train mode split | Present, runtime validation pending |

---

## Superseded Records

Earlier 2026-07-04 devlog entries that mark instinctRL-B as `Complete` or say to proceed to instinctRL-C remain superseded. The current truthful conclusion is:

- `instinctRL-A`: PASS with open runtime verification item(s)
- `instinctRL-B`: PARTIAL / NOT FULLY ACCEPTED
- `instinctRL-C`: NO-GO until Isaac runtime smoke passes
