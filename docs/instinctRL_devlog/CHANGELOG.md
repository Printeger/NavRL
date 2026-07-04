# instinctRL Development Changelog

> Format: [YYYY-MM-DD] — Ticket ID — Summary

---

## 2026-07-04 (B-fix implementation)

### instinctRL-B: Observation / History Buffer Fix Pass

**Status**: Implementation complete, runtime acceptance pending. Current stage remains `B-closeout / B-runtime validation before instinctRL-C`.

- Replaced the active instinctRL RayCaster pattern path with a Livox MID360 helper wrapper; `env.py` no longer uses `patterns.BpearlPatternCfg` for instinctRL.
- Fixed `BodyToWorldVelocityAdapter` to use body-to-world quaternion rotation semantics and added identity, yaw, and roll/pitch frame tests.
- Wired `prev_action` from the previously issued governor/controller command into `MID360ObservationBuilder`; reset now clears selected env history and previous issued action.
- Hardened Observation / History Buffer behavior: raw range remains true distance, invalid beams have zero reliability, weights clamp to `[0, 1]`, stale/repeated frames are flagged, and history rollover/reset are unit-tested.
- Added actor schema audit for hybrid actor input and tests proving actor obs only exposes `lidar_grid` and `state_vec`.
- Added explicit `instinctRL.mode` separation: `smoke` runs B0/B observation smoke; `train` initializes PPO and runs the hybrid actor audit/forward path before continuing training.
- Added pure Python/PyTorch unit tests for MID360 pattern shape/order, observation semantics, previous-action feedback, actor schema, PPO hybrid forward smoke, and command-adapter frame convention.

**Actual validation run**:

- `python3 -m pytest ...` could not run because the base Python environment has no `pytest`.
- Manual pure unit-test runner passed all runnable tests under both base Python and the `NavRL` conda Python.
- `test_instinctrl_ppo_hybrid.py` skipped in the local environment because `tensordict/torchrl` import fails with `ImportError: cannot import name 'ForkingPickler' from torch.multiprocessing.reductions`.
- Minimal Isaac runtime smoke did not run to completion locally:
  - base Python: `ModuleNotFoundError: No module named 'hydra'`
  - `NavRL` conda Python: `ModuleNotFoundError: No module named 'click'` via `wandb`

**Acceptance conclusion**: instinctRL-B is **PARTIAL / NOT FULLY ACCEPTED** until the runtime smoke and PPO hybrid forward validation pass in a correctly provisioned Isaac/TorchRL environment. instinctRL-C remains **NO-GO**.

---

## 2026-07-04 (Closeout Acceptance Review)

### instinctRL-A / instinctRL-B Closeout Before C

**Status**: Historical closeout. Superseded in part by the later B-fix implementation entry above.

- **instinctRL-A**: Accepted as B0 smoke-test / infrastructure baseline, not learning success.
- **instinctRL-A verification update**: Adapter frame direction has now been corrected and unit-tested. Runtime integration remains covered by the before-C smoke validation.
- **instinctRL-B**: Partial acceptance only. The later implementation fixed the known code blockers, but full acceptance remains blocked by unrun Isaac runtime smoke and unrun local TorchRL PPO hybrid forward validation.
- **instinctRL-C**: NO-GO until runtime validation in `TEST_PLAN.md` passes.

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
