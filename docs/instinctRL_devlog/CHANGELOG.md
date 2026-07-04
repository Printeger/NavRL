# instinctRL Development Changelog

> Format: [YYYY-MM-DD] — Ticket ID — Summary

---

## 2026-07-04 (PM3)

### instinctRL-B: Observation / History Buffer

**Status**: ✅ Complete

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
- **Resolved**: D-002 (Full MID360 preprocessing)

---

### instinctRL-A: B0 Smoke Test Runtime

**Status**: ⚠️ Blocked — pre-existing Isaac Sim PhysX fabric issue

- All code imports/config/specs verified correct at Python level
- `NavigationEnv.__init__` hangs at `drone.initialize()` — PhysX fabric GPU stall
- Root cause: `IsaacEnv` init flow vs `test_flight.py` flow differ in `sim.reset()` timing
- Ruled out: headless mode, terrain, env count, import order
- Fix: moved OmniDrones imports inside `main()` after `SimulationApp`

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
