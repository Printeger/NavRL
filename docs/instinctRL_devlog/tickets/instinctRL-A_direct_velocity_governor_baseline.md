# instinctRL-A: Direct Velocity-Governor Baseline

> **Ticket ID**: instinctRL-A  
> **Status**: ✅ Complete — B0 smoke test PASSED (7/7 criteria, 2026-07-04 PM4)  
> **Date**: 2026-07-04  
> **Dependencies**: instinctRL-0 (all 5 blockers resolved)  
> **Blocks**: instinctRL-B, instinctRL-C, instinctRL-D, instinctRL-E, instinctRL-F, instinctRL-G, instinctRL-H  
> **Risk**: Medium  
> **Handbook**: `instinctRL_Development_Handbook_v1_1_platform_locked.tex` §Tickets

---

## Goal

Establish the clean body-frame velocity command path with B0 (α=1, v_corr=0 pass-through) baseline. Add instinctRL config namespace. Verify platform lock, actor input contract, and action type via staged audit. Smoke-test the TASLAB_UAV + MID360 + VelController chain. Do NOT implement learning, reward tuning, or full governor head.

---

## Files Inspected

| File | Purpose |
|------|---------|
| `cfg/train.yaml` | Existing training config — added `instinctRL` section |
| `cfg/ppo.yaml` | Existing PPO config — added `instinctRL.governor` section |
| `scripts/env.py` | NavigationEnv — added v_cmd production, raw MID360 range, info spec |
| `scripts/train.py` | Training entry — added B0 smoke test path, audit, governor, adapter |
| `scripts/ppo.py` | PPO policy — unchanged (B0 bypasses policy completely) |
| `scripts/instinctRL/__init__.py` | Package init (existing from instinctRL-0) |

---

## Files Modified

| File | Change Summary | Lines |
|------|---------------|:-----:|
| `cfg/train.yaml` | Added `instinctRL.enabled` and `instinctRL.baseline.id` | +7 |
| `cfg/ppo.yaml` | Added `instinctRL.governor.*` config (alpha_mode, alpha_fixed, v_corr_limit, velocity_limit, smoothing_tau) | +8 |
| `scripts/env.py` | Added `v_cmd` production (fixed + random body-frame), MID360 raw range computation, `v_cmd` in info spec | +35 |
| `scripts/train.py` | Added instinctRL B0 smoke test path: audit checks, governor/adapter creation, 500-step smoke loop, early exit | +60 |

---

## New Files Created

| File | Purpose | Lines |
|------|---------|:-----:|
| `scripts/instinctRL/audit.py` | Staged audit: `check_platform_lock()`, `check_actor_input()`, `check_action_type()`, `run_audit()` | ~140 |
| `scripts/instinctRL/command_adapter.py` | `BodyToWorldVelocityAdapter`: body→world transform via `quat_rotate_inverse` | ~55 |
| `scripts/instinctRL/governor.py` | `MinimalGovernor` (B0: α=1, v_corr=0) + `GovernorOutput` dataclass (future interface) | ~75 |
| `docs/instinctRL_devlog/DEFERRED_REGISTER.md` | 8 deferred items (D-001 to D-008) with full metadata | ~120 |
| `docs/instinctRL_devlog/DECISION_LOG.md` | 6 architectural decisions from grilling session | ~90 |
| `docs/instinctRL_devlog/TEST_PLAN.md` | B0 smoke test spec (10 tests) + future test registries | ~80 |

---

## Main Changes

### 1. Config Namespace
- `instinctRL.enabled: true` activates instinctRL mode
- `instinctRL.baseline.id: "direct_velocity"` selects B0
- `instinctRL.governor.alpha_mode: "fixed"` with `alpha_fixed: 1.0`

### 2. Governor (B0 Minimal)
- `MinimalGovernor.forward(v_cmd, obs=None)` → `GovernorOutput(alpha=1.0, v_corr=0.0, v_gov=v_cmd)`
- Interface preserved for future trainable governor via `obs` parameter
- `GovernorOutput` dataclass with alpha, v_corr, v_gov fields

### 3. Body→World Adapter
- `BodyToWorldVelocityAdapter.forward(body_vel, drone_quat)` → world_vel
- Uses `quat_rotate_inverse` from OmniDrones
- Drone quaternion from privileged `info["drone_state"][...,3:7]`

### 4. v_cmd Production
- Fixed initial body-frame command: random `[-0.5, 0.5]` m/s per axis, reduced vertical
- Regenerated every ~2 seconds (125 steps)
- Stored in `info["v_cmd"]` (critic-accessible, not actor observation)

### 5. MID360 Raw Range
- True distance computation: `r_i = ||ray_hits - pos||`
- Stored in `env.lidar_raw_range`
- NOT danger-coded (no inversion)

### 6. Staged Audit
- `check_platform_lock(cfg)`: TASLAB_UAV model + MID360 FOV/range
- `check_actor_input(tensordict)`: forbidden key pattern scan
- `check_action_type(action)`: 3-dim velocity verification
- Runs at env construction and first smoke test step

### 7. B0 Smoke Test
- 500 physics steps
- Flow: `v_cmd (info) → governor → adapter → VelController → env.step`
- Checks: no NaN, no crash, LiDAR active with valid returns
- Clean exit before PPO training (standard path preserved when `instinctRL.enabled=false`)

---

## Method Consistency Checklist

| Check | Status |
|-------|:------:|
| Action type: velocity command (3-dim), not CTBR (4-dim) | ✅ `check_action_type` audit |
| Actor input: no pose/odom/explicit velocity/map/privileged state | ✅ `check_actor_input` audit |
| Platform: TASLAB_UAV | ✅ `check_platform_lock` audit |
| Sensor: Livox MID360 (FOV [-7°, 52°], range 40m) | ✅ `check_platform_lock` audit |
| Governor: body-frame velocity output | ✅ `BodyToWorldVelocityAdapter` |
| Controller: `VelController(LeePositionController)` | ✅ Existing chain preserved |
| No CTBR, body-rate, motor thrust, attitude setpoints as learned action | ✅ Verified |
| No generic platform/sensor in normal configs | ✅ Config locked to TaslabUAV + MID360 |

---

## Tests Run

| Test | Result | Evidence |
|------|:------:|----------|
| A.1 Platform lock audit | ✅ | `PLATFORM AUDIT PASS: TaslabUAV + MID360` |
| A.2 Actor input audit | ✅ | `ACTOR INPUT AUDIT PASS` |
| A.3 Action type audit | ✅ | `ACTION TYPE AUDIT PASS: 3-dim velocity` |
| A.4 Environment reset | ✅ | `Reset complete.` |
| A.5 TASLAB_UAV spawn | ✅ | `TaslabUAV_0/base_link`, mass=1.1583kg |
| A.6 MID360 attachment | ✅ | `[4, 1, 360, 59]`, 18.97% valid |
| A.7 Body→World adapter | ✅ | B0 governor α=1.0, 500 steps no NaN |
| A.8 VelController execution | ✅ | `4 articulations`, motor commands valid |
| A.9 Smoke rollout stability | ✅ | `Completed 500/500 steps`, Exit 0 |
| A.10 LiDAR statistics | ✅ | min ≥ 0, max ≤ 40m |

**Runtime**: 141.9s, `python3 training/scripts/train.py env.num_envs=4 env_dyn.num_obstacles=0`

---

## Known Issues

1. **512-env scaling**: B0 smoke test hang with 512 envs (PhysX fabric memory). Works with ≤4 envs on RTX 4070 Ti SUPER (16GB).
2. **Standard PPO mode**: Not tested with instinctRL.enabled=false after B changes.

---

## Next Recommended Step

✅ instinctRL-A complete. Proceeded to instinctRL-B (also verified in same run). Build MID360 r, m, w, timestamps, IMU cues, command, previous output, history buffer. Dependencies: instinctRL-A complete.
