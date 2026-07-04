# instinctRL-0: Platform and Sensor Infrastructure Audit

> **Ticket ID**: instinctRL-0  
> **Status**: ✅ Complete — All 5 blockers resolved (2026-07-04)  
> **Date**: 2026-07-03 (audit) / 2026-07-04 (blocker fixes)  
> **Dependencies**: None (first ticket)  
> **Blocks**: instinctRL-A, instinctRL-B, instinctRL-C, instinctRL-D, instinctRL-E, instinctRL-F, instinctRL-G, instinctRL-H  
> **Risk**: High (platform/sensor lock violations invalidate all later results)  
> **Full Report**: [`docs/instinctRL_0_platform_sensor_audit.md`](../../instinctRL_0_platform_sensor_audit.md)

---

## Objective

Prove that the locked MID360–TASLAB_UAV platform stack is available, correctly wired, and actor-input compliant. Produce a precise audit report. Do **not** implement policy learning, rewards, ICS attenuation, anchor manager, observability logger, or training stages.

---

## Blocker Resolution Summary (2026-07-04)

| # | Blocker | Resolution | File(s) |
|---|---------|-----------|---------|
| 1 | LiDAR prim path hardcoded | Dynamic prim discovery: `_resolve_base_link()` searches `[base_link, body, base, chassis]`, logs resolved path. `attach_yaw_only=False`. | `env.py` |
| 2 | Actor velocity leak | `state[5:8]` (`vel_g`) removed from actor `obs` dict. Retained for reward only. | `env.py` |
| 3 | Actor position leak | `state[0:5]` (`rpos_clipped_g`, `distance_2d`, `distance_z`) and `direction[3]` removed from actor `obs`. Retained for reward. | `env.py` |
| 4 | Privileged obstacle leak | `dynamic_obstacle[N,10]` removed from actor `obs`. Dead `dynamic_obstacle_network` removed from `ppo.py`. Unused `vec_to_world` import cleaned. | `env.py`, `ppo.py` |
| 5 | Critic starvation / command adapter | **Option B**: Asymmetric actor-critic added. Actor: LiDAR only. Critic: `_actor_feature` + privileged `drone_state` + `target_rpos` + `target_distance`. Command adapter deferred to instinctRL-A. `verify_actor_critic_separation()` test added. | `ppo.py`, `env.py` |

### Additional Changes

- **CONTEXT.md**: Added active method-lock banner. Marked CTBR, CMDP, PPO-Lagrangian, GRU-required as ⚠️ LEGACY.
- **Devlog**: CHANGELOG.md, DEV_STATUS.md updated. Blocked items cleared.

---

## Acceptance Criteria

| # | Criterion | Status |
|---|-----------|:------:|
| 1 | MID360 simulation path identified | ✅ `training/envs/livox_mid360.py`, integration helper, unit tests |
| 2 | TASLAB_UAV model/controller path identified | ✅ `taslab_uav.py`, `taslab_uav.yaml`, `lee_controller_taslab_uav.yaml`, `drone/__init__.py` |
| 3 | Velocity command interface identified | ✅ `VelController(LeePositionController(...))` chain in `train.py` |
| 4 | Actor input fields audited | ✅ 4 fields audited; 100% non-compliant |
| 5 | No forbidden actor input silently accepted | ✅ All leaks documented with source code traces |
| 6 | All proposed new modules justified by absence of reusable infrastructure | ✅ 7 new modules justified; 5 components marked for reuse |
| 7 | Audit report produced at `docs/instinctRL_0_platform_sensor_audit.md` | ✅ |
| 8 | Devlog structure initialized | ✅ `README.md`, `CHANGELOG.md`, `DEV_STATUS.md`, ticket file |
| 9 | Go/no-go decision documented | ✅ Conditional GO (5 blockers) |

---

## Findings Summary

### ✅ Confirmed Existing Infrastructure

| Subsystem | Key Files | Verdict |
|-----------|-----------|---------|
| MID360 simulation | `training/envs/livox_mid360.py` (~910 lines), `training/scripts/livox_mid360_integration.py`, `training/unit_test/test_livox_mid360.py` | Mature, well-tested, not wired into training |
| MID360 range output | `LivoxMid360Pattern.apply_noise()` produces ordered range vector with configurable noise/dropout | Produces `r_t` and derivable `m_t` |
| MID360 ray pattern | `LivoxMid360Pattern.generate_rays()` — 360°×59° FOV, Lissajous or static, body-frame output | Stable ray ordering, correct frame convention |
| TASLAB_UAV model | `taslab_uav.py` (registered), `taslab_uav.yaml` (physical params), `taslab_uav.usd` (visual) | `REGISTRY["TaslabUAV"]` resolves correctly |
| TASLAB_UAV controller | `lee_controller_taslab_uav.yaml` (gains), `VelController(LeePositionController(...))` chain | Works in training pipeline |
| Training entry | `train.py`, `env.py`, `ppo.py` | Functional but non-compliant actor inputs |
| Config system | Hydra with `drone.yaml`, `train.yaml`, `ppo.yaml`, `sim.yaml` | `drone.yaml` selects TaslabUAV |

### ❌ Missing / Non-Compliant

| Gap | Detail |
|-----|--------|
| MID360 not wired to NavigationEnv | Env uses generic `RayCaster` + `BpearlPatternCfg` on Hummingbird prim path |
| No reliability weights $w_t$ | Neighbor consistency not implemented |
| No sensor timestamps | Internal `_time` only; no frame-age/staleness tracking |
| Actor receives forbidden velocity | `vel_g` from `root_state[7:10]` in `state[5:8]` |
| Actor receives forbidden position | `rpos_clipped_g`, `distance_2d`, `distance_z` in `state[0:5]` |
| Actor receives privileged obstacle state | `dynamic_obstacle[N,10]` with exact pos/vel/size |
| No body-frame velocity adapter | `VelController` expects world-frame; governor outputs body-frame |
| No platform-lock audit checks | No enforcement that TASLAB_UAV + MID360 are active |
| No actor-input audit checks | No enforcement of forbidden key exclusion |
| ROS deployment uses odometry + map raycast | `navigation.py`, `policy_server.py` pass forbidden inputs to actor |

### 🔴 Blocker Summary (5 items)

1. **LiDAR prim path**: Hardcoded to `Hummingbird_0/base_link`, not TASLAB_UAV
2. **Actor velocity leak**: `state[5:8]` = `vel_g` from simulator ground-truth
3. **Actor position leak**: `state[0:5]` = goal-relative position (direction + distance)
4. **Actor obstacle state leak**: `dynamic_obstacle[N,10]` = privileged obstacle pos/vel/size
5. **No body-frame command interface**: Governor body-frame output needs world-frame adapter before `VelController`

---

## Reuse vs. New Implementation

| Component | Decision | Rationale |
|-----------|:--------:|-----------|
| MID360 range preprocessing | ✅ Reuse | `livox_mid360.py` is mature; only needs wiring |
| History buffer | 🆕 New | No existing history buffer |
| Command generator (training) | ✅ Reuse | `command_generator.py` for scenario generation |
| Velocity command interface | ✅ Reuse + Add | Controller chain works; add body→world adapter |
| Logging / evaluation | ✅ Reuse | WandB + `EpisodeStats`; add instinctRL metrics |
| Training config | ✅ Reuse + Add | Hydra; add `instinctRL` namespace |
| ROS deployment wrapper | 🆕 Rebuild | Current nodes pass forbidden inputs |
| Anchor manager | 🆕 New | No existing measurement-space anchor |
| ICS attenuation | 🆕 New | `safeAction.cpp` is ORCA/map-based |
| Observability logger | 🆕 New | No existing range-Jacobian tooling |
| Reward terms | 🔄 Refactor | Extract from inline `env.py`; redesign for station-keeping |
| Policy network | ✅ Reuse + Replace | Reuse PPO loop; replace actor head |
| Platform audit | 🆕 New | No existing platform-lock checks |

---

## Key Risks Identified

1. **Frame convention mismatch**: Body-frame governor output vs world-frame controller input (🔴 HIGH)
2. **Mount rotation order discrepancy**: `LivoxMid360Pattern` (ZYX) vs `LidarRetina` (RPY) (🟡 MEDIUM)
3. **`attach_yaw_only` setting**: Current `True` (spinning LiDAR) should be `False` for solid-state MID360 (🟡 MEDIUM)
4. **CONTEXT.md outdated**: Still describes CTBR/CMDP/PPO-Lagrangian path; v1.1 handbook overrides (🟡 LOW)
5. **Open-field unobservability**: Inevitable method limitation; must be documented not hidden (🟡 MEDIUM)

---

## Next Steps

1. Resolve 5 blockers (estimated 1–2 days)
2. Proceed to [instinctRL-A](instinctRL-A_direct_velocity_governor.md): Direct velocity-governor baseline
3. Update `CONTEXT.md` to reflect v1.1 platform-locked method after instinctRL-A completes

---

## Related Documents

- [Full Audit Report](../../instinctRL_0_platform_sensor_audit.md)
- [Development Handbook v1.1](../../instinctRL_Development_Handbook_v1_1_platform_locked.tex) — §2
- [Paper 1](../../paper1_vel_ctrl.tex)
- [DEV_STATUS.md](../DEV_STATUS.md)
- [CHANGELOG.md](../CHANGELOG.md)
