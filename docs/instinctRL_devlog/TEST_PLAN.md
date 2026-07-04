# instinctRL Test Plan

> **Created**: 2026-07-04 (instinctRL-A)  
> **Last Updated**: 2026-07-04 (NavRL validation)
> **Purpose**: Define verification procedures for each instinctRL stage.

---

## instinctRL-A: B0 Smoke Test

**Closeout verdict**: PASS with open verification item(s). Accepted as B0 smoke-test / infrastructure baseline, not learning success.

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
- **Current status**: Pure unit test passed; runtime smoke still pending.
- **Evidence**: `training/unit_test/test_instinctrl_command_adapter.py` covers identity, 90 deg yaw, and roll/pitch cases.
- **Pass**: Known quaternion cases prove body -> world direction; integration smoke shows body X/Y/Z commands map to expected world motion.
- **Fail**: Any yaw/roll/pitch case maps through inverse direction, or only shape/NaN checks are performed.

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

**Current verdict**: RUNTIME CHECKS PASSED / FINAL EXIT-STATUS RERUN PENDING. Code fixes and NavRL pytest/PPO validation pass. User-side GPU smoke completed the B checks, then hit an Isaac Kit shutdown segfault after success; smoke mode now exits before `SimulationApp.close()` and needs one post-workaround rerun to confirm shell exit code 0.

### Required Before-C Tests

| Test | Required evidence | Current status |
|------|-------------------|----------------|
| B.1 Active MID360 pattern and ray ordering | Active `NavigationEnv` uses Livox MID360 ray ordering or a documented equivalent; no `BpearlPatternCfg` substitute in instinctRL training path | User-side runtime smoke passed MID360 shape/valid-return checks |
| B.2 Ray count and shape stability | Repeated reset/step preserves `[N, H, V]` ray layout and expected ray count | User-side runtime smoke passed with `[4, 1, 360, 59]` raw range |
| B.3 Raw range correctness | `r_i = ||ray_hit_i - lidar_pos||`, not danger-coded inverse range, with max-range handling | Pure observation test passed |
| B.4 Valid-return mask | Mask derives from finite in-range returns and handles max-range/dropout explicitly | Pure observation test passed |
| B.5 Reliability bounds | `w_t` stays in `[0, 1]`; stale/dropout returns are represented correctly | Pure observation test passed |
| B.6 Timestamp monotonicity and frame age | Sim time is monotonic; repeated/stale frames are detectable | Pure observation test passed |
| B.7 History rollover | Fixed window rolls exactly one frame per policy step and resets per env reset | Pure observation test passed |
| B.8 Previous issued action feedback | `prev_action` slots equal prior governor/controller output, not default zeros | Code fixed; pure observation test passed; user-side runtime smoke completed 500 steps |
| B.9 Actor input provenance | Audit proves `lidar_grid` and `state_vec` contain only allowed fields | Runtime actor/schema audit passed |
| B.10 PPO/training-path smoke | `instinctRL.enabled=true` can run a PPO hybrid initialization/forward path, or smoke-only mode is explicitly separated | Mode split implemented; NavRL PPO hybrid forward test passes |

### Added Test Files

- `training/unit_test/test_instinctrl_command_adapter.py`
- `training/unit_test/test_instinctrl_mid360_pattern.py`
- `training/unit_test/test_instinctrl_observation.py`
- `training/unit_test/test_instinctrl_actor_audit.py`
- `training/unit_test/test_instinctrl_ppo_hybrid.py`

### Actual Commands and Results

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd isaac-training && python -m pytest training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_ppo_hybrid.py -q` | Passed: `14 passed, 2 warnings`. Includes regression test for Orbit RayCaster in-place offset on MID360 ray starts. |
| `python3 -m py_compile isaac-training/training/scripts/train.py isaac-training/training/scripts/env.py isaac-training/training/scripts/ppo.py isaac-training/training/scripts/instinctRL/audit.py isaac-training/training/scripts/instinctRL/command_adapter.py isaac-training/training/scripts/instinctRL/observation.py isaac-training/training/scripts/instinctRL/mid360_pattern.py isaac-training/training/unit_test/test_instinctrl_*.py` | Passed. |
| `rg -n "BpearlPatternCfg|patterns\\." isaac-training/training/scripts/env.py isaac-training/training/scripts/instinctRL isaac-training/training/cfg -S` | No matches. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && python - <<'PY' ... dependency probe ... PY` | Passed: activated NavRL resolves Isaac torch 2.0.1, TorchRL/TensorDict, Hydra, WandB, and Click; `ForkingPickler=True`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd isaac-training && python training/scripts/train.py instinctRL.mode=smoke env.num_envs=4 env_dyn.num_obstacles=0` | Reaches CUDA preflight, then fails: no CUDA-capable device visible. |
| `nvidia-smi` | Failed: could not communicate with NVIDIA driver. |
| `conda activate NavRL && python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` | `False`, `0`. |
| User-side GPU smoke after MID360 RayCaster fix | Passed all B0/B checks: 500/500 steps, PPO hybrid forward, actor/schema/action audits, MID360 raw range `[4, 1, 360, 59]`, valid returns `33.04%`; then segfaulted inside `SimulationApp.close()` during Isaac Kit shutdown. |

### Remaining Required Validation Before C

- Re-run `instinctRL.mode=smoke env.num_envs=4 env_dyn.num_obstacles=0` after the shutdown workaround and confirm shell exit status is 0.
- When running the smoke command manually, keep `env_dyn.num_obstacles=0` on the same command line, or use shell line continuations (`\`) so Hydra receives it as an override.
- Smoke mode exits before `SimulationApp.close()` after successful validation because Isaac Kit can segfault during shutdown after an otherwise-passed smoke.

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
