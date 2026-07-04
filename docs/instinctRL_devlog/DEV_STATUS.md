# instinctRL Development Status

> **Last Updated**: 2026-07-04
> **Current Stage**: B-closeout / B-runtime validation before instinctRL-C
> **Authority order**: code facts > handbook acceptance criteria > devlog records.

---

## Stage Summary

| Field | Value |
|-------|-------|
| **Current stage** | B-closeout / B-runtime validation before instinctRL-C |
| **Active ticket** | instinctRL-B validation |
| **Next ticket** | instinctRL-C only after B runtime/PPO validation passes |
| **Final go/no-go** | instinctRL-C: NO-GO |
| **instinctRL-A** | PASS with open runtime verification item(s) |
| **instinctRL-B** | PARTIAL / NOT FULLY ACCEPTED |

---

## Acceptance Status

| Ticket | Verdict | Notes |
|--------|---------|-------|
| instinctRL-0 | Accepted as prior platform audit baseline | Earlier audit remains useful context, but current B acceptance is judged against active code. |
| instinctRL-A | PASS with open runtime verification item(s) | Accepted as B0 smoke-test / infrastructure baseline, not learning success. Adapter frame direction has a pure unit test; full runtime smoke must still run in a provisioned Isaac environment. |
| instinctRL-B | PARTIAL / NOT FULLY ACCEPTED | Code blockers have been addressed, and pure tests pass. Full handbook acceptance is blocked by local inability to run Isaac runtime smoke and TorchRL PPO hybrid forward validation. |
| instinctRL-C | NO-GO | Do not start until B validation checklist passes. |

---

## Current B-Fix State

| ID | Former blocker | Current code fact | Remaining acceptance requirement |
|----|----------------|-------------------|----------------------------------|
| B-FIX-001 | Active env RayCaster used `patterns.BpearlPatternCfg` | `env.py` now uses `instinctRL.mid360_pattern.create_mid360_pattern_cfg()` for the active instinctRL RayCaster path; local search found no `BpearlPatternCfg` use in `training/scripts/env.py`, `training/scripts/instinctRL`, or `training/cfg`. | Run Isaac runtime smoke proving active MID360 returns and stable ray layout in the real env. |
| B-FIX-002 | Body-to-world adapter frame direction was wrong/unverified | `BodyToWorldVelocityAdapter` now uses body-to-world quaternion rotation; identity, yaw 90 deg, and roll/pitch unit tests pass. | Include adapter path in runtime smoke. |
| B-FIX-003 | `instinctRL.enabled=true` returned after B0 smoke only | `instinctRL.mode` now separates `smoke` and `train`; train mode runs actor audit, schema audit, and PPO hybrid forward smoke before normal training. | Run `instinctRL.mode=smoke` runtime smoke and a minimal `instinctRL.mode=train` PPO initialization/forward smoke in a working environment. |
| B-FIX-004 | `prev_action` was not fed from issued governor output | `env.set_prev_issued_action_body()` stores the previous issued body-frame command; `observation.py` now requires `prev_action` and unit tests verify feedback. | Confirm in runtime smoke across reset/step boundaries. |
| B-FIX-005 | Actor audit scanned key names only | `audit.py` now includes `check_actor_schema()` for the hybrid actor input shape/key contract; pure actor schema tests pass. | Runtime actor audit must pass on real `TensorDict` from `NavigationEnv`. |
| B-FIX-006 | B tests were missing | Pure tests for MID360 pattern, observation semantics, history/reset, previous action, actor schema, PPO hybrid smoke, and adapter frame convention were added. | Install/repair local test dependencies so `pytest`, TorchRL, and Isaac smoke can run normally. |

---

## Actual Test Evidence

| Command | Result |
|---------|--------|
| `python3 -m pytest isaac-training/training/unit_test/test_instinctrl_*.py` | Not runnable in base Python: `No module named pytest`. |
| Manual pure unit-test runner under base Python | Passed all runnable tests; PPO hybrid test skipped because TorchRL/TensorDict unavailable. |
| Manual pure unit-test runner under `/home/mint/miniconda3/envs/NavRL/bin/python` | Passed all runnable tests; PPO hybrid test skipped because `tensordict/torchrl` import fails with `ImportError: cannot import name 'ForkingPickler' from torch.multiprocessing.reductions`. |
| `python3 -m py_compile ...` for changed code/tests | Passed. |
| `rg -n "BpearlPatternCfg|patterns\\." ...` | No matches in the inspected active instinctRL env/cfg paths. |
| `python3 isaac-training/training/scripts/train.py instinctRL.mode=smoke env.num_envs=4 env_dyn.num_obstacles=0` | Not runnable in base Python: `No module named hydra`. |
| `/home/mint/miniconda3/envs/NavRL/bin/python isaac-training/training/scripts/train.py instinctRL.mode=smoke env.num_envs=4 env_dyn.num_obstacles=0` | Not runnable locally: `No module named click` from `wandb`. |

---

## Clean Status Table

| Component | Current fact | Status |
|-----------|--------------|--------|
| Config namespace | `instinctRL.enabled=true`, `instinctRL.mode=smoke`, `baseline.id=direct_velocity`, observation history config exists | Present |
| B0 governor | `MinimalGovernor` implements alpha=1, v_corr=0 pass-through | Present |
| Command adapter | Corrected body-to-world rotation and covered by pure unit tests | Present, runtime smoke pending |
| Observation builder | Builds range/mask/weight/IMU/v_cmd/prev_action/history tensors; requires real `prev_action` | Present, pure tests pass |
| Active sensor pattern | Active instinctRL env path uses MID360 helper wrapper, not `BpearlPatternCfg` | Present, runtime smoke pending |
| PPO hybrid input | `ppo.py` consumes `lidar_grid` and `state_vec` | Present, local TorchRL validation blocked |
| Actor audit | Key scan plus hybrid schema audit | Present, provenance remains by schema/code review rather than full taint proof |
| Training path | Explicit smoke/train mode split | Present, runtime validation pending |

---

## Superseded Records

Earlier 2026-07-04 devlog entries that mark instinctRL-B as `Complete` or say to proceed to instinctRL-C remain superseded. The current truthful conclusion is:

- `instinctRL-A`: PASS with open runtime verification item(s)
- `instinctRL-B`: PARTIAL / NOT FULLY ACCEPTED
- `instinctRL-C`: NO-GO until runtime/PPO validation passes
