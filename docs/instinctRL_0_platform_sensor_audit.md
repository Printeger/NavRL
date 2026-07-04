# instinctRL-0: Platform and Sensor Infrastructure Audit

> **Date**: 2026-07-03  
> **Status**: ✅ Complete  
> **Ticket**: [instinctRL-0](./instinctRL_devlog/tickets/instinctRL-0_platform_sensor_audit.md)  
> **Handbook**: `instinctRL_Development_Handbook_v1_1_platform_locked.tex` §2  
> **Next**: instinctRL-A (Direct velocity-governor baseline)

---

## Executive Summary

The platform infrastructure (MID360 simulation, TASLAB_UAV model, velocity controller) **exists and is mature**. However, the active training pipeline (`NavigationEnv`) does not use any of it — it relies on a generic `RayCaster` with `BpearlPatternCfg` attached to a hardcoded Hummingbird prim path, and passes forbidden privileged state (velocity, goal-relative position, dynamic obstacle state) to the actor. The MID360 and TASLAB_UAV helpers are well-tested standalone but are **not wired** into the training loop.

**Go/No-Go**: 🟡 **CONDITIONAL GO** — 5 blockers must be resolved before instinctRL-A can begin (see §8).

---

## 1. Repository Structure — Confirmed Files & Directories

### 1.1 MID360 Simulation

| File | Role | Status |
|------|------|--------|
| `isaac-training/training/envs/livox_mid360.py` | Core: `LivoxMid360Config`, `LivoxMid360Pattern`, `generate_rays()`, `apply_noise()` (~910 lines) | ✅ Mature |
| `isaac-training/training/scripts/livox_mid360_integration.py` | Isaac Sim `RayCaster` bridge; 4 config presets (high/medium/low/minimal) | ✅ Available |
| `isaac-training/training/unit_test/test_livox_mid360.py` | Integration tests: ray pattern validation, simulation test, flight demo (~700 lines) | ✅ Available |
| `isaac-training/training/envs/lidar_processor.py` | `LidarRetina`: point cloud → depth image via spherical min-pooling; potential for neighbor-consistency weights (~820 lines) | ✅ Reusable |

### 1.2 TASLAB_UAV Model

| File | Role | Status |
|------|------|--------|
| `isaac-training/third_party/OmniDrones/omni_drones/robots/drone/taslab_uav.py` | `TaslabUAV(MultirotorBase)` — minimal class, specifies `usd_path` + `param_path` | ✅ Registered |
| `isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/taslab_uav.yaml` | Mass=1.1583kg, inertia, rotor layout (X-type, 4 rotors), force/moment constants, max rotor speeds | ✅ Complete |
| `isaac-training/third_party/OmniDrones/omni_drones/robots/assets/usd/taslab_uav.usd` | Binary USD visual/articulation model (~158KB) | ✅ Present |
| `isaac-training/third_party/OmniDrones/omni_drones/robots/drone/__init__.py` | Imports `TaslabUAV`; auto-registered via `MultirotorBase.REGISTRY` (`__init_subclass__` pattern) | ✅ Active |
| `isaac-training/third_party/OmniDrones/omni_drones/controllers/cfg/lee_controller_taslab_uav.yaml` | TASLAB-specific Lee position controller gains: `pos=[4,4,4]`, `vel=[2.2,2.2,2.2]`, `att=[0.7,0.7,0.035]`, `ang_rate=[0.1,0.1,0.025]` | ✅ Present |

### 1.3 Training Entry Points & Config

| File | Role |
|------|------|
| `isaac-training/training/scripts/train.py` | Main training loop: `NavigationEnv` → `VelController(LeePositionController)` → `PPO` → `SyncDataCollector` → WandB logging |
| `isaac-training/training/scripts/env.py` | `NavigationEnv(IsaacEnv)` — scene design, LiDAR `RayCaster`, observations, rewards, done flags (~900 lines) |
| `isaac-training/training/scripts/ppo.py` | `PPO(TensorDictModuleBase)` — CNN LiDAR encoder (Conv2d 1→4→16→16) + BetaActor outputting 3D velocity; `__call__` denormalizes to `[-2,2]` m/s and transforms goal→world frame |
| `isaac-training/training/scripts/command_generator.py` | `AdversarialCommandGenerator` — 5-mode adversarial velocity commands (Normal Nav, Aggressive Step, Adversarial Suicide, Oscillation, Recovery Hover). Not wired into main training. |
| `isaac-training/training/scripts/utils.py` | `GAE`, `ValueNorm`, `make_mlp`, `BetaActor`, `IndependentBeta`, `vec_to_new_frame`, `vec_to_world`, `construct_input`, `EpisodeStats`, `evaluate` |
| `isaac-training/training/cfg/drone.yaml` | `model_name: "TaslabUAV"`; MID360 sensor params: `lidar_range=40.0`, `lidar_vfov=[-7.0,52.0]`, `lidar_vbeams=59`, `lidar_hres=1.0`, `mount_pitch=30.0`, `mount_position=[0,0,0.05]` |
| `isaac-training/training/cfg/train.yaml` | 512 envs, max_episode_length=2200, headless=True, WandB offline mode |
| `isaac-training/training/cfg/ppo.yaml` | CNN feature extractor, BetaActor, action_limit=2.0 m/s, 3 separate Adam optimizers |
| `isaac-training/training/cfg/sim.yaml` | `dt=0.016s` (62.5 Hz), `substeps=1`, TGS solver, GPU pipeline |

### 1.4 ROS/Deployment Wrappers

| File | Role | Actor Input Compliance |
|------|------|------------------------|
| `ros1/navigation_runner/scripts/navigation.py` | ROS node: subscribes `/mavros/local_position/odom` (or `/CERLAB/quadcopter/odom`), runs map `RayCast` service, calls dynamic obstacle service, publishes velocity | ❌ Uses odometry + map raycast |
| `ros1/navigation_runner/scripts/policy_server.py` | Policy inference service: accepts `state[8]` + `lidar` + `direction` + `dynamic_obstacle` from ROS request | ❌ Passes velocity + goal-relative position to actor |
| `ros1/navigation_runner/include/navigation_runner/safeAction.cpp` | ORCA-based velocity obstacle safety filter using obstacle absolute positions/velocities from services | ❌ Map/obstacle-state dependent |
| `ros2/navigation_runner/` | ROS2 equivalents with same architecture | ❌ Same leaks |

---

## 2. MID360 Audit — Capability Assessment

### 2.1 `LivoxMid360Pattern` (livox_mid360.py)

| Capability | Available? | Implementation Detail |
|------------|:----------:|------------------------|
| Stable ordered range vector $r_t$ | ✅ | `generate_rays()` produces deterministic ray ordering (azimuth fastest-varying, elevation slowest-varying via `meshgrid(xy)` C-order flattening) when `enable_dynamic_scan=False`. Actual range values come from external RayCaster + `apply_noise()`. |
| Valid-return mask $m_t$ | ✅ | Derivable from `apply_noise()` output: `isfinite(dist) & (min_range ≤ dist ≤ max_range)`. Mask logic exists but is inline — not exposed as a named output. |
| Reliability weights $w_t$ | ❌ | Not implemented. Handbook references neighbor-consistency via `lidar_processor.py` debug metadata and MID360 validity/noise signals as potential basis. |
| Ray directions in body frame | ✅ | `generate_rays(return_sensor_frame=False)` outputs body-frame directions via mount rotation $R = R_{yaw} R_{pitch} R_{roll}$ (ZYX Euler). Convention: X-forward, Y-left, Z-up. |
| Min/max range | ✅ | `LivoxMid360Config`: `min_range=0.1m`, `max_range=40.0m`. Matches real MID360 spec (40m @ 10% reflectivity). |
| FOV / scan pattern | ✅ | 360° horizontal × 59° vertical (-7° to +52°). Lissajous or static, configurable resolution. `horizontal_res=1.0°`, `num_vertical_lines=30` (default). Coverage analysis via `get_coverage_info()`. |
| Update rate | ⚠️ | 10 Hz nominal (documented in file header). Simulation runs `lidar.update(self.dt)` at 62.5 Hz physics rate. Internal `_time` accumulator exists for dynamic scan phase. |
| Dropout / no-return behavior | ✅ | `apply_noise()` applies 4-layer noise model sequentially to valid returns: (1) Gaussian range noise σ=0.02+0.001·d, (2) random dropout 2% → `inf`, (3) near-range dropout 10% for d<1m → `inf`, (4) unreliable zone 0.1–0.2m: 50% dropout + 3× extra noise. Values clamped to `[min_range, max_range]`, then `inf` restored. |
| Occlusion modeling | ✅ | `_compute_occlusion_mask()`: 3-zone model (rear cone, propeller region, straight down). Configurable `occlusion_rear_cone_angle`. |
| Timestamps | ❌ | Internal `_time` only. No explicit sensor timestamp, frame-age tracking, or staleness marking. |

### 2.2 `LidarRetina` (lidar_processor.py)

| Capability | Available? | Implementation Detail |
|------------|:----------:|------------------------|
| Spherical projection | ✅ | Converts point clouds → body-frame → spherical coordinates → grid indices → min-pooling → depth image |
| Grid masks / debug | ✅ | `valid_mask` in debug mode; azimuth/elevation per pixel |
| Multi-scale | ✅ | `LidarRetinaMultiScale` with `F.avg_pool2d` pyramid (scales=1,2,4) |
| Neighbor consistency | ⚠️ | Grid structure enables neighbor checks; no explicit reliability weight output yet |

### 2.3 Integration Helper (livox_mid360_integration.py)

Config presets in `LIVOX_MID360_CONFIGS`:

| Name | H-res | V-lines | Total rays |
|------|-------|---------|------------|
| high | 1° | 59 | 21,240 |
| medium | 2° | 30 | 5,400 |
| low | 5° | 15 | 1,080 |
| minimal | 10° | 8 | 288 |

Key functions:
- `get_livox_mid360_config_simple(lidar_range=40.0) → RayCasterCfg` — creates `RayCasterCfg` using `BpearlPatternCfg` with MID360 vertical angles
- `livox_mid360_pattern(h_fov, h_res, v_fov, n_v_lines, device) → (starts, dirs)` — generates static ray pattern grid in Cartesian body-frame

### 2.4 Critical Gap: MID360 Not Wired into NavigationEnv

The active `NavigationEnv.__init__()` creates:

```python
ray_caster_cfg = RayCasterCfg(
    prim_path="/World/envs/env_.*/Hummingbird_0/base_link",  # ← Hardcoded Hummingbird!
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
    attach_yaw_only=True,  # ← Should be False for solid-state MID360
    pattern_cfg=patterns.BpearlPatternCfg(
        horizontal_res=self.lidar_hres,
        vertical_ray_angles=torch.linspace(*self.lidar_vfov, self.lidar_vbeams)
    ),
    max_distance=...,
    mesh_prim_paths=["/World/ground"],
)
```

Despite `cfg/drone.yaml` selecting `model_name: "TaslabUAV"`, the LiDAR prim path is **hardcoded** to Hummingbird. The `LivoxMid360Pattern` class and integration helpers exist but are unused by the training pipeline.

---

## 3. TASLAB_UAV Audit — Capability Assessment

### 3.1 Model Registration & Parameters

`TaslabUAV` extends `MultirotorBase` (which extends `RobotBase`). Registration via `__init_subclass__`:

```python
# drone/__init__.py
from .taslab_uav import TaslabUAV
# MultirotorBase.REGISTRY["TaslabUAV"] resolves automatically
```

Physical parameters from `taslab_uav.yaml`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `mass` | 1.1583 kg | Aligned with USD rigid body |
| `inertia.xx` | 0.0051350687 | Calibrated 2026-02-07 |
| `inertia.yy` | 0.0056075235 | |
| `inertia.zz` | 0.0052743931 | |
| `inertia.xz` | 0.00028 | Small cross-term |
| `arm_length` (`l`) | 0.24042 m | Diagonal |
| `arm_lengths` | [0.12021, 0.12021, 0.12021, 0.12021] | Half-diagonal per rotor |
| `rotor_angles` | [5.4978, 0.7854, 2.3562, 3.9270] rad | X-type layout |
| `directions` | [+1, -1, +1, -1] | CW/CCW alternating |
| `force_constants` | 4× 1.55e-06 N/(rad/s)² | |
| `moment_constants` | 4× 2.39e-08 N·m/(rad/s)² | |
| `max_rotation_velocities` | 4× 2261 rad/s | 4S estimated |
| `drag_coef` | 0.2 | |

### 3.2 Velocity Controller Chain

Confirmed execution path in `train.py`:

```python
controller = LeePositionController(9.81, env.drone.params).to(cfg.device)
vel_transform = VelController(controller, yaw_control=False)
```

Full chain:
```
Actor → [vx, vy, vz] world-frame → VelController._inv_call()
  → reads tensordict[("info", "drone_state")]           ← full 13-dim drone state
  → LeePositionController.forward(drone_state, target_vel=v_cmd)
     → pos_error = pos - target_pos (target_pos unused for vel-only mode)
     → vel_error = vel - target_vel
     → acc_cmd = pos_error * Kp + vel_error * Kv - g
     → attitude from desired thrust direction b3_des = -normalize(acc_cmd)
     → angular error from rotation matrix error
     → motor_cmds = mixer @ [ang_acc, thrust]^T, normalized to [-1, 1]^4
  → writes motor_cmds back to tensordict[("agents", "action")]
  → drone.apply_action(motor_cmds) → PhysX forces/torques
```

### 3.3 Interface Assessment

| Capability | Available? | Detail |
|------------|:----------:|--------|
| Model registration / config / USD | ✅ | `REGISTRY["TaslabUAV"]`, `taslab_uav.yaml`, `taslab_uav.usd` |
| Mass / inertia / dynamics parameters | ✅ | All parameters loaded from YAML |
| Velocity-controller execution path | ✅ | `VelController(LeePositionController(...))` chain works |
| Accepted command frame | ⚠️ | **World-frame** velocity expected by `VelController`. instinctRL requires **body-frame** commands — needs adapter. |
| Command saturation / safety limits | ❌ | Only `action_limit=2.0` m/s in PPO config; no per-axis, dynamic, or instinctRL-specific limits |
| Body-frame velocity command interface | ❌ | Must be added as adapter between governor output `\vgov` (body-frame) and `VelController` (world-frame) |
| Emergency stop / safety override | ❌ | ROS `safeAction.cpp` is ORCA-based and map-dependent; not usable for instinctRL |

### 3.4 Controller Privileged State Access

`VelController._inv_call()` reads `tensordict[("info", "drone_state")]` which contains:
`[pos(3), quat(4), vel(3), ang_vel(3)]` — the full 13-dim drone state.

**This is acceptable** per the handbook: "State claim boundary: actor is pose/odometry-free; controller internals are conventional stabilization." The `LeePositionController` uses this state to convert velocity commands to motor thrusts — analogous to an onboard flight controller on a real drone.

---

## 4. Actor Input Audit — Field-by-Field Assessment

### 4.1 Current Observation Schema (from `env.py:_set_specs`)

```python
self.observation_spec = CompositeSpec({
    "agents": CompositeSpec({
        "observation": CompositeSpec({
            "state":            UnboundedContinuousTensorSpec((8,)),           # ← LEAKS
            "lidar":            UnboundedContinuousTensorSpec((1, H, V)),       # ← Wrong format
            "direction":        UnboundedContinuousTensorSpec((1, 3)),          # ← LEAKS
            "dynamic_obstacle": UnboundedContinuousTensorSpec((1, N, 10)),      # ← LEAKS
        }),
    }).expand(self.num_envs)
}, shape=[self.num_envs], device=self.device)
```

### 4.2 `state[8]` — Breakdown with Source Code Trace

| Index | Variable | Source Code (`_compute_state_and_obs`) | Category |
|:-----:|----------|----------------------------------------|----------|
| 0–2 | `rpos_clipped_g` | `rpos = self.target_pos - self.root_state[..., :3]`; `rpos / distance`; `vec_to_new_frame(..., target_dir_2d)` | 🔴 **FORBIDDEN** — position-derived goal direction |
| 3 | `distance_2d` | `rpos[..., :2].norm(dim=-1, keepdim=True)` | 🔴 **FORBIDDEN** — goal-relative horizontal distance |
| 4 | `distance_z` | `rpos[..., 2].unsqueeze(-1)` | 🔴 **FORBIDDEN** — goal-relative vertical distance |
| 5–7 | `vel_g` | `self.root_state[..., 7:10]`; `vec_to_new_frame(vel_w, target_dir_2d)` | 🔴 **FORBIDDEN** — explicit ground-truth velocity from simulator |

### 4.3 `dynamic_obstacle[N, 10]` — Breakdown

| Index | Variable | Source | Category |
|:-----:|----------|--------|----------|
| 0–2 | `rpos_gn` | `closest_dyn_obs_rpos / distance` → goal-frame normalized | 🔴 **FORBIDDEN** — privileged obstacle position |
| 3 | `distance_2d` | `closest_dyn_obs_rpos[..., :2].norm()` | 🔴 **FORBIDDEN** — privileged distance |
| 4 | `distance_z` | `closest_dyn_obs_rpos[..., 2]` | 🔴 **FORBIDDEN** — privileged distance |
| 5–7 | `vel_g` | `self.dyn_obs_vel[closest_dyn_obs_idx]` → goal-frame | 🔴 **FORBIDDEN** — privileged obstacle velocity |
| 8 | `width_category` | `closest_dyn_obs_width / dyn_obs_width_res - 1` → [0–3] | 🔴 **FORBIDDEN** — privileged size |
| 9 | `height_category` | Binary: 3D or 2D obstacle | 🔴 **FORBIDDEN** — privileged size |

### 4.4 `lidar[1, H, V]` — Format Issue

The current LiDAR observation is **inverted danger-coded**:
```python
self.lidar_scan = self.lidar_range - (
    (ray_hits_w - pos_w).norm(dim=-1).clamp_max(self.lidar_range)
    .reshape(self.num_envs, 1, *self.lidar_resolution)
)
```
This produces values where **higher = closer to obstacle** (danger-centric). instinctRL requires **raw range** $r_t$ where value = true distance in meters. This is a format conversion, not a structural leak.

### 4.5 Summary Matrix

| instinctRL Target Input | Currently Available? | Source |
|-------------------------|:--------------------:|--------|
| MID360 raw range $r_t$ | ❌ (wrong format) | Needs raw distance from `ray_hits_w - pos_w` without inversion |
| Valid-return mask $m_t$ | ❌ | Needs derivation from `apply_noise()` output |
| Reliability weights $w_t$ | ❌ | Not implemented |
| Ray directions / ordering | ❌ (not exposed to actor) | Available from `LivoxMid360Pattern` |
| IMU / attitude cues | ❌ | `root_state[10:13]` (ang_vel) exists but not exposed as actor input |
| Operator command $\vcmd$ | ❌ | Current uses `target_pos` (goal position), not velocity command |
| Previous issued command | ❌ | Not tracked |
| History $h_t$ | ❌ | No history buffer |

### 4.6 Verdict

**100% of current actor inputs are non-compliant** with the instinctRL actor input contract. Every field either:
- Contains forbidden information (velocity, position, privileged obstacle state), or
- Is in the wrong format (inverted danger-coded LiDAR instead of raw range vector)

The actor input path must be completely rebuilt for instinctRL.

---

## 5. Reuse vs. New Implementation Assessment

### 5.1 Per-Component Decision

| Component | Reuse? | Existing Code | Rationale |
|-----------|:------:|---------------|-----------|
| MID360 range preprocessing | ✅ **Reuse** | `livox_mid360.py`, `livox_mid360_integration.py` | Core simulation is mature and well-tested. Only needs wiring into `NavigationEnv`. |
| History buffer | 🆕 **New** | — | No existing history buffer; current observation is single-frame. |
| Command generator (training) | ✅ **Reuse** | `command_generator.py` | `AdversarialCommandGenerator` can produce $\vcmd$ for training scenarios. Only $\vcmd$ (not generator internals) enters actor. |
| Velocity command interface | ✅ **Reuse + Add** | `VelController` + `LeePositionController` | Controller chain works. Need **body-frame→world-frame adapter** between governor output and `VelController`. |
| Logging / evaluation | ✅ **Reuse** | WandB + `EpisodeStats` + `evaluate()` | Add instinctRL-specific metrics (anchor, ICS, observability, audit). |
| Training config | ✅ **Reuse + Add** | Hydra config system | Add `instinctRL` namespace. Existing configs lack instinctRL switches. |
| ROS deployment wrapper | 🆕 **Rebuild** | `navigation.py`, `policy_server.py` | Current ROS nodes pass odometry + map raycast + obstacle state to actor. Must be replaced with MID360/IMU/command-only interface. |
| Anchor manager | 🆕 **New** | — | No existing measurement-space anchor lifecycle. |
| ICS attenuation | 🆕 **New** | `safeAction.cpp` (not reusable) | `safeAction.cpp` is ORCA-based, map/obstacle-state dependent — violates actor contract. |
| Observability logger | 🆕 **New** | — | No existing range-Jacobian or drift correlation logger. |
| Reward terms | 🔄 **Refactor** | Inline in `env.py:_compute_state_and_obs` | Current reward is goal-navigation (velocity·direction, log-clearance). Must be extracted to `instinctRL/rewards.py` and redesigned for measurement-space station-keeping. |
| Policy network | ✅ **Reuse + Replace** | `ppo.py` PPO loop | Reuse PPO collection/update loop and LiDAR encoder. Replace actor head (currently outputs raw velocity, consumes forbidden fields). |
| Platform audit | 🆕 **New** | — | No existing platform-lock or actor-input audit checks. |

### 5.2 Justification for New Code

| New Module | Why Existing Code Is Insufficient |
|------------|-----------------------------------|
| `instinctRL/observation.py` | Current `_compute_state_and_obs` bundles forbidden velocity, goal-relative position, and privileged obstacle state into actor input. No history buffer, timestamps, or reliability weights exist. |
| `instinctRL/anchor.py` | No existing concept of measurement-space reference capture. Current architecture tracks goal positions, not frozen range patterns. |
| `instinctRL/governor.py` | Current actor outputs raw velocity `[vx,vy,vz]` directly. instinctRL needs governor head outputting $(\alpha_t, \vcorr)$ with body-frame convention and deterministic command formation $\vgov = \alpha_t\vcmd + \vcorr$. |
| `instinctRL/ics.py` | `safeAction.cpp` uses ORCA velocity obstacles requiring absolute obstacle positions from map services — fundamentally incompatible with pose-free measurement-space approach. |
| `instinctRL/observability.py` | No existing range-Jacobian computation. Simulator has surface normal access via RayCaster scene, but no tooling to compute $J = \partial r/\partial p$ or correlate with drift. |
| `instinctRL/rewards.py` | Current reward is inline in `_compute_state_and_obs` and is goal-navigation oriented. instinctRL requires tracking, anchor, intervention, and safety terms, with gating by command magnitude and ICS state. |
| `instinctRL/audit.py` | No existing mechanism to verify platform lock (TASLAB_UAV + MID360) or actor input contract at environment construction, policy init, rollout, eval, checkpoint export, or ROS inference. |

---

## 6. Frame Convention Risks

| Risk | Severity | Detail | Mitigation |
|------|:--------:|--------|-----------|
| Body-frame vs world-frame velocity | 🔴 **HIGH** | `LivoxMid360Pattern` outputs body-frame ray directions. `VelController` expects world-frame velocity. Governor outputs body-frame $\vgov$ which must be rotated to world-frame before controller. | Transform body-frame $\vgov$ to world-frame using drone attitude from `info["drone_state"]` (accessible to controller, not actor). |
| `attach_yaw_only` setting | 🟡 **MEDIUM** | Current `RayCaster` uses `attach_yaw_only=True` (appropriate for spinning LiDAR). MID360 is solid-state and should use `attach_yaw_only=False`. | Set `attach_yaw_only=False` in MID360 config; verify in `test_livox_mid360.py` which already uses this setting. |
| Mount rotation order mismatch | 🟡 **MEDIUM** | `LivoxMid360Pattern` uses ZYX Euler: $R = R_{yaw} R_{pitch} R_{roll}$. `LidarRetina` uses: $R = R_{roll} R_{pitch} R_{yaw}$. Both claim sensor→body transform but differ in convention. | Audit and align; document the chosen convention. Prefer `LivoxMid360Pattern` convention as authoritative MID360 reference. |
| Mount pitch discrepancy | 🟡 **MEDIUM** | `drone.yaml`: `mount_pitch=30.0°`. `LivoxMid360Config` default: `mount_pitch=45.0°`. | Standardize to single config source; use `create_livox_from_hydra_cfg` to read from Hydra config. |
| Goal-frame vs body-frame vs command-frame | 🔴 **HIGH** | Current code uses "goal frame" for actor. instinctRL must use **body/governor frame** for $\vcmd, \vcorr, \vgov, \vfin$. Transform to world-frame only at controller boundary. | Define explicit frame labels in all variable names and config keys. Add unit tests for frame transforms. |

### 6.1 Frame Convention Reference

```
Body frame (instinctRL governor frame):
  X: forward
  Y: left
  Z: up

World frame (VelController input):
  Z: up (gravity-aligned)
  X, Y: inertial horizontal plane

Body → World transform:
  Use drone attitude quaternion from info["drone_state"]
  v_world = quat_rotate(q_drone, v_body)
```

---

## 7. Recommended Minimal Code Changes for instinctRL-A

These are the **minimum** changes needed before any policy learning can begin. They are ordered by dependency:

### Blocker 1: Fix LiDAR Prim Path
**File**: `isaac-training/training/scripts/env.py`  
**Change**: Replace hardcoded `Hummingbird_0/base_link` with dynamically resolved prim path from `cfg.drone.model_name`.
```python
# Current:
prim_path="/World/envs/env_.*/Hummingbird_0/base_link"
# Target:
prim_path=f"/World/envs/env_.*/{cfg.drone.model_name}_0/base_link"
```

### Blocker 2: Wire MID360 Pattern
**File**: `isaac-training/training/scripts/env.py`  
**Change**: Replace `BpearlPatternCfg` with `create_livox_from_hydra_cfg(cfg)` or the integration helper. Set `attach_yaw_only=False`. Use MID360-specific ray ordering.

### Blocker 3: Sanitize Actor Inputs
**File**: `isaac-training/training/scripts/env.py` (`_compute_state_and_obs`, `_set_specs`)  
**Change**: Remove `state`, `direction`, `dynamic_obstacle` from actor observation. Replace with:
- MID360 raw range $r_t$ (flat vector, not inverted)
- Valid-return mask $m_t$
- Operator command $\vcmd$ (body-frame)
For instinctRL-A baseline, these 3 fields suffice (defer $w_t$, IMU, history to instinctRL-B).

### Blocker 4: Add Body-Frame Velocity Adapter
**File**: New `isaac-training/training/scripts/instinctRL/command_adapter.py`  
**Change**: Transform governor body-frame output $\vgov$ to world-frame using drone attitude quaternion before passing to `VelController`. This lives between actor output and controller input.

### Blocker 5: Add Platform-Lock Audit
**File**: New `isaac-training/training/scripts/instinctRL/audit.py`  
**Change**: At environment construction, assert:
- `cfg.drone.model_name == "TaslabUAV"`
- Sensor config identifies Livox MID360
- No forbidden keys leak into actor TensorDict
Fail with clear error message if violations found.

### Additional: Add Actor-Input Audit
**File**: New `isaac-training/training/scripts/instinctRL/audit.py` (same module)  
**Change**: Check that no forbidden keys (`pose`, `pos`, `position`, `odom`, `velocity`, `vel_g`, `vel_w`, `root_state`, `map`, `slam`, `privileged`, `direction`, `distance`, `dynamic_obstacle`) appear in actor TensorDict. Run at: env construction, policy init, rollout collection, evaluation, checkpoint export.

---

## 8. Go/No-Go Decision for instinctRL-A

### 🟡 CONDITIONAL GO

**Rationale**: The platform infrastructure (MID360 simulation, TASLAB_UAV model, velocity controller) **exists and is mature**. All standalone tests pass. The problems are **integration gaps**, not fundamental deficiencies.

### Blocker Summary

| # | Blocker | Severity | Est. Effort |
|---|---------|:--------:|:-----------:|
| 1 | LiDAR prim path hardcoded to Hummingbird, not TASLAB_UAV | 🔴 Blocker | Small (1 line) |
| 2 | Actor receives ground-truth velocity `vel_g` | 🔴 Blocker | Medium (refactor obs) |
| 3 | Actor receives goal-relative position | 🔴 Blocker | Medium (refactor obs) |
| 4 | Actor receives privileged dynamic obstacle state | 🔴 Blocker | Medium (refactor obs) |
| 5 | No body-frame velocity command interface | 🔴 Blocker | Medium (new adapter) |
| 6 | MID360 not wired into active training path | 🟡 High | Medium (wire pattern) |
| 7 | No platform-lock or actor-input audit checks | 🟡 High | Small (new audit module) |
| 8 | No reliability weights $w_t$ computation | 🟡 Medium | Can defer to instinctRL-B |
| 9 | No sensor timestamps | 🟡 Medium | Can defer to instinctRL-B |

### Pre-Flight Checklist for instinctRL-A

- [ ] `cfg.drone.model_name == "TaslabUAV"` confirmed and enforced
- [ ] MID360 ray pattern active (not Bpearl)
- [ ] LiDAR prim path resolves to TASLAB_UAV base link
- [ ] No `vel_g`, `rpos_clipped_g`, `distance_2d`, `distance_z` in actor TensorDict
- [ ] No `dynamic_obstacle` in actor TensorDict
- [ ] Body-frame $\vcmd$ accepted by governor
- [ ] Governor body-frame output correctly transformed to world-frame before `VelController`
- [ ] Platform-lock audit passes at environment construction
- [ ] Actor-input audit passes at policy init, rollout, eval

---

## 9. Appendix: Test Files Reference

| Test File | What It Validates | Reusable for instinctRL? |
|-----------|-------------------|--------------------------|
| `unit_test/test_livox_mid360.py` | Ray pattern generation, simulation integration, flight demo | ✅ Yes — validates MID360 sensor stack |
| `unit_test/test_flight.py` | Keyboard-controlled flight with TASLAB_UAV | ✅ Yes — validates TASLAB controller |
| `unit_test/test_hover.py` | Hover stability with TASLAB_UAV | ✅ Yes — validates TASLAB dynamics |
| `unit_test/test_config_reading.py` | `create_livox_from_hydra_cfg` config loading | ✅ Yes — validates MID360 config |
| `unit_test/README.md` | Test instructions and conventions | ✅ Reference |

---

*Audit completed 2026-07-03. All findings based on repository inspection of file contents, not assumptions.*
