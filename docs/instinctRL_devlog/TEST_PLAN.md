# instinctRL Test Plan

> **Created**: 2026-07-04 (instinctRL-A)  
> **Purpose**: Define verification procedures for each instinctRL stage.

---

## instinctRL-A: B0 Smoke Test

### Test A.1: Platform Lock Audit
- **What**: Assert `cfg.drone.model_name == "TaslabUAV"` and MID360 sensor params
- **Where**: `instinctRL/audit.py:check_platform_lock()`
- **Pass**: Both checks return True
- **Fail**: RuntimeError with descriptive message

### Test A.2: Actor Input Audit
- **What**: Scan actor observation keys for forbidden substrings (pose, pos, position, odom, velocity, vel_g, vel_w, root_state, map, slam, privileged, direction, distance, dynamic_obstacle)
- **Where**: `instinctRL/audit.py:check_actor_input()`
- **Pass**: No forbidden substrings found in actor observation keys
- **Fail**: RuntimeError listing violated patterns

### Test A.3: Action Type Audit
- **What**: Verify action is 3-dim velocity, not 4-dim CTBR or motor thrust
- **Where**: `instinctRL/audit.py:check_action_type()`
- **Pass**: action.shape[-1] == 3
- **Fail**: RuntimeError if 4-dim (CTBR) or other unexpected dimension

### Test A.4: Environment Reset
- **What**: `env.reset()` returns valid TensorDict with required keys
- **Where**: `train.py` B0 smoke test
- **Pass**: No exception; tensordict contains "info" and "agents" namespaces
- **Fail**: Exception or missing keys

### Test A.5: TASLAB_UAV Spawn
- **What**: Drone spawns via `REGISTRY["TaslabUAV"]` and prim path resolves
- **Where**: Verified by `_resolve_base_link()` in `env.py`
- **Pass**: Base link found and logged; no fallback to root
- **Fail**: "No base_link found" warning

### Test A.6: MID360 Basic Attachment
- **What**: LiDAR raw range tensor available and non-empty
- **Where**: `env.lidar_raw_range` after env step
- **Pass**: Shape > 0, valid return fraction > 0%
- **Fail**: All-zero or missing tensor

### Test A.7: Body→World Velocity Adapter
- **What**: Body-frame v_cmd correctly transformed to world-frame
- **Where**: `BodyToWorldVelocityAdapter.forward()` during smoke test
- **Pass**: Output shape matches input, no NaN, non-zero when input non-zero
- **Fail**: NaN in output, zero output for non-zero input, shape mismatch

### Test A.8: VelController Execution
- **What**: World-frame velocity → motor commands via `VelController(LeePositionController)`
- **Where**: `transformed_env.step()` in smoke test
- **Pass**: Motor commands in [-1, 1] range, drone moves
- **Fail**: NaN in motor commands, drone doesn't move

### Test A.9: Smoke Rollout Stability
- **What**: 500 consecutive physics steps without crash or NaN
- **Where**: `train.py` B0 smoke test loop
- **Pass**: Loop completes; no RuntimeError
- **Fail**: NaN in action/reward, crash, or exception

### Test A.10: LiDAR Range Statistics
- **What**: LiDAR returns physically plausible ranges
- **Where**: End of B0 smoke test
- **Pass**: min ≥ 0, max ≤ lidar_range (40m), valid fraction > 0
- **Fail**: Negative ranges, all ranges at max, valid fraction = 0

---

## Future Test Registries

### instinctRL-B: Observation / History Buffer
- MID360 ray count stability
- Ray ordering determinism
- Valid-return mask correctness
- Reliability weight bounds [0,1]
- Timestamp monotonicity
- History buffer rollover
- Stale-frame detection
- Actor input absence (re-check)

### instinctRL-C: Measurement-Space Anchor
- Null-command hysteresis (ε₀ < ε₁)
- Anchor capture on rising edge
- Masked anchor error computation
- Anchor reset (episode, large cmd, insufficient valid)
- Anchor loss active only under null cmd

### instinctRL-D: Observability Logger
- Range-Jacobian rank estimation
- σ_min(J) vs drift correlation plot
- Per-scenario drift ranking
- Hardware proxy labeled as proxy

### instinctRL-E: ICS Attenuation
- β_t monotonic with speed/clearance
- Empty active set → β_t=1
- Emergency bypass on min-clearance
- No surface-normal imports in deployed ICS
- No odometry/map access in ICS

### instinctRL-F: Reward Integration
- Each term activates under intended condition
- Privileged quantities stay reward/critic/eval only
- No actor leakage through reward path
- First stable training run

### instinctRL-G: Baselines
- B0–B8 config isolation
- Explicit input-schema logs per baseline
- Required metrics per baseline

### instinctRL-H: Real-Robot Deployment
- No odom/map in actor input
- Latency logs
- Safety logs
- Bag replay audit
