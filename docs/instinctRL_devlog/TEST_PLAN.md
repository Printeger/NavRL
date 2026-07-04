# instinctRL Test Plan

> **Created**: 2026-07-04 (instinctRL-A)  
> **Purpose**: Define verification procedures for each instinctRL stage.

---

## instinctRL-A: B0 Smoke Test

### Test A.1: Platform Lock Audit
- **Runtime result**: ✅ PASSED (2026-07-04)
- **Evidence**: `PLATFORM AUDIT PASS: drone.model_name='TaslabUAV' | sensor matches Livox MID360 FOV [-7°, 52°] | lidar_range=40m (MID360)`

### Test A.2: Actor Input Audit (Runtime ✅ 2026-07-04)
- **Evidence**: `ACTOR INPUT AUDIT PASS: no forbidden fields in actor observation`

### Test A.3: Action Type Audit (Runtime ✅ 2026-07-04)
- **Evidence**: `ACTION TYPE AUDIT PASS: 3-dim velocity command`

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

### Test A.9: Smoke Rollout Stability (Runtime ✅ 2026-07-04)
- **Evidence**: `Completed 50/500 ... 500/500 steps.`, no NaN errors

### Test A.10: LiDAR Range Statistics (Runtime ✅ 2026-07-04)
- **Evidence**: `shape=torch.Size([4, 1, 360, 59]), valid=18.97%`

---

## instinctRL-B: Observation / History Buffer

### Test B.1: Raw Range Computation

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
