# instinctRL Development Changelog

> Format: [YYYY-MM-DD] — Ticket ID — Summary

---

## 2026-07-04

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
