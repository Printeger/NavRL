# instinctRL Development Status

> **Last Updated**: 2026-07-04  
> **Current Stage**: instinctRL-B (Observation / History Buffer)

---

## Stage Summary

| Field | Value |
|-------|-------|
| **Current stage** | instinctRL-B — Observation / History Buffer |
| **Stage status** | ✅ Complete — Hybrid observation pipeline built + runtime verified |
| **Active ticket** | instinctRL-B |
| **Next ticket** | instinctRL-C — Measurement-Space Anchor |
| **Method consistency** | ✅ Velocity-controller-based. Paper 1 route preserved. |
| **Platform lock** | ✅ TASLAB_UAV runtime verified (4 envs, 500-step smoke test PASSED) |
| **Sensor lock** | ✅ MID360 runtime verified: LiDAR [4,1,360,59], 18.97% valid returns |
| **Actor input compliance** | ✅ Both `check_actor_input` and `check_action_type` PASSED at runtime |

---

## Completed Tickets

| Ticket | Title | Date | Notes |
|--------|-------|------|-------|
| instinctRL-0 | Platform and Sensor Infrastructure Audit | 2026-07-03 | Audit report produced. 5 blockers identified. |
| instinctRL-0 | Blocker Fixes (all 5 resolved) | 2026-07-04 | env.py, ppo.py, CONTEXT.md updated. Asymmetric actor-critic. |
| instinctRL-A | Direct Velocity-Governor Baseline (B0) | 2026-07-04 | Config namespace, governor, adapter, audit. B0 smoke test PASSED (7/7). |
| instinctRL-B | Observation / History Buffer | 2026-07-04 | Hybrid observation: raw range, mask, weight, IMU, history. Runtime verified. |

---

## Active Tickets

| Ticket | Title | Status | Assigned |
|--------|-------|--------|----------|
| instinctRL-0 | Platform and Sensor Infrastructure Audit | ✅ Complete | — |

---

## Blocked Items

**None.** All 5 instinctRL-0 blockers resolved.

### Blocker Resolution Details

| # | Blocker | Resolution | File(s) |
|---|---------|-----------|---------|
| 1 | LiDAR prim path hardcoded | Dynamic prim discovery via `_resolve_base_link()` | `env.py` |
| 2 | Actor velocity leak (`vel_g`) | Removed `state[5:8]` from actor obs | `env.py` |
| 3 | Actor position leak (`rpos_clipped_g`, `distance`) | Removed `state[0:5]`, `direction` from actor obs | `env.py` |
| 4 | Actor privileged obstacle state leak | Removed `dynamic_obstacle` from actor obs + dead code cleanup | `env.py`, `ppo.py` |
| 5 | No body-frame command interface | Critic privileged branch added (asymmetric). Command adapter deferred to instinctRL-A. | `ppo.py`, `env.py` |

---

## Upcoming Tickets

| Ticket | Title | Dependencies | Risk |
|--------|-------|-------------|:----:|
| instinctRL-A | Direct Velocity-Governor Baseline | instinctRL-0 blockers resolved | Medium |
| instinctRL-B | Observation / History Buffer | instinctRL-A | High |
| instinctRL-C | Measurement-Space Anchor | instinctRL-B | High |
| instinctRL-D | Observability Logger | instinctRL-C | Medium |
| instinctRL-E | ICS-Inspired Attenuation | instinctRL-B | High |
| instinctRL-F | Reward Integration and Training | instinctRL-B, C, E | High |
| instinctRL-G | Baselines and Ablations | instinctRL-F | Medium |
| instinctRL-H | Real-Robot Deployment Validation | instinctRL-F | High |

---

## Known Risks

| Risk | Probability | Impact | Status |
|------|:----------:|:------:|--------|
| Active RayCaster remains attached to Hummingbird path | Confirmed | High | 🔴 Must fix (blocker 1) |
| MID360 helper exists but not wired into training | Confirmed | High | 🔴 Must fix (instinctRL-0) |
| Low-level controller uses state internally | Acceptable | Low | ✅ Per handbook: controller internals are conventional stabilization |
| Open-field unobservability | Inevitable | Medium | 🟡 Report as method limitation; use observability logger |
| Range dropout and latency | Expected | Medium | 🟡 Use masks, weights, timestamps; defer to instinctRL-B |
| Frame convention mismatch (body vs world) | High | High | 🔴 Must fix (blocker 5); add body→world adapter + tests |
| Policy overuses ICS | Potential | Medium | 🟡 Penalize intervention usage; compare against B5 no-ICS baseline |
| Reward hacking | Potential | Low | 🟡 Log component rewards; validate with baselines |
| Sim-to-real gap | Expected | High | 🟡 Defer to instinctRL-H; keep TASLAB params and MID360 noise model |
| Surface normals leak into deployed safety | Preventable | Medium | 🟡 Separate observability logger from ICS; add no-normal audit |
| CONTEXT.md describes CTBR/CMDP (outdated) | Confirmed | Low | 🟡 v1.1 handbook overrides; update CONTEXT.md after instinctRL-A |

---

## Method Consistency Status

| Aspect | Handbook (v1.1) | Current Code | Status |
|--------|:---------------:|:------------:|:------:|
| Action type | Body-frame velocity command | World-frame velocity via CNN+BetaActor | ⚠️ Needs change |
| Actor input | MID360 $r_t, m_t, w_t$, IMU, $\vcmd$, history | `lidar` (danger-coded grid), `state` (vel+pos), `direction`, `dynamic_obstacle` | ❌ Non-compliant |
| Controller | `VelController(LeePositionController)` with TASLAB gains | Same | ✅ Compliant |
| Platform | TASLAB_UAV | `cfg.drone.model_name = "TaslabUAV"` | ⚠️ Config matches but not enforced |
| Sensor | Livox MID360 | `BpearlPatternCfg` in env, MID360 helpers unused | ❌ Not wired |
| Frame convention | Body/governor frame for commands | Goal-frame and world-frame used | ❌ Incorrect |
| Actor architecture | Governor head $(\alpha_t, \vcorr)$ | BetaActor outputting raw velocity | ❌ Needs replacement |
| Critic architecture | Critic-only privileged branch | Single critic, no privilege separation | ❌ Needs redesign |
| Reward terms | Tracking, anchor, intervention, safety | Goal-navigation reward (vel·dir + log-clearance) | ❌ Needs redesign |

---

## Environment Summary

| Component | Path | Status |
|-----------|------|--------|
| Simulator | Isaac Sim via OmniDrones + Orbit | ✅ Working |
| Training env | `isaac-training/training/scripts/env.py` | ⚠️ Needs MID360 + actor input fixes |
| Training entry | `isaac-training/training/scripts/train.py` | ✅ Working |
| PPO policy | `isaac-training/training/scripts/ppo.py` | ⚠️ Needs governor head |
| MID360 sim | `isaac-training/training/envs/livox_mid360.py` | ✅ Mature |
| TASLAB model | `isaac-training/third_party/OmniDrones/.../taslab_uav.py` | ✅ Working |
| Velocity ctrl | `VelController` + `LeePositionController` | ✅ Working |
| ROS1 deploy | `ros1/navigation_runner/` | ❌ Non-compliant actor inputs |
| ROS2 deploy | `ros2/navigation_runner/` | ❌ Non-compliant actor inputs |
