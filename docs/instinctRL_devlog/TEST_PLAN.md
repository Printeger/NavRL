# instinctRL Test Plan

> **Created**: 2026-07-04 (instinctRL-A)  
> **Last Updated**: 2026-07-05 (instinctRL-E complete)
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
- **Current status**: Unit test passed; B0/B smoke path passed.
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

**Current verdict**: COMPLETE. Code fixes, NavRL pytest/PPO validation, and user-side GPU smoke all pass. Smoke mode exits before `SimulationApp.close()` after successful validation to avoid Isaac Kit teardown segfaults after pass.

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
| User-side post-workaround GPU smoke | Passed: PPO hybrid forward, actor/schema/action audits, 500/500 steps, MID360 raw range `[4, 1, 360, 59]`, valid returns `28.62%`, `B0 Smoke Test PASSED`, `Observation smoke path PASSED`, and success path exited before `SimulationApp.close()`. |

### Before-C Validation

- No remaining B blocker before C.
- When running the smoke command manually, keep `env_dyn.num_obstacles=0` on the same command line, or use shell line continuations (`\`) so Hydra receives it as an override.
- Smoke mode exits before `SimulationApp.close()` after successful validation because Isaac Kit can segfault during shutdown after an otherwise-passed smoke.

## instinctRL-C: Measurement-Space Anchor

**Current verdict**: COMPLETE. Anchor manager unit tests and B+C regression suite pass in the activated NavRL conda environment. Env integration is passive and preserves the actor-clean `lidar_grid` + `state_vec` contract.

### Required C Tests

| Test | Required evidence | Current status |
|------|-------------------|----------------|
| C.1 Config validation | `eps_enter < eps_exit`; `0.0 < min_valid_anchor_fraction <= 1.0`; canonical key rejects `min_valid_fraction`; `huber_delta > 0` | Pure anchor test passed |
| C.2 Null-command hysteresis | Capture at `||v_cmd|| <= eps_enter`; no capture between enter/exit; command reset at `||v_cmd|| >= eps_exit` | Pure anchor test passed |
| C.3 Reset priority | `episode > explicit > command > invalid > none` with fixed enum `0..4` | Pure anchor test passed |
| C.4 Anchor capture | Rising edge freezes `r_star`, bool `m_star`, and `w_star`; later frames do not mutate them | Pure anchor test passed |
| C.5 Reset state rules | Episode reset clears activation count; explicit/command/invalid resets preserve it; all resets clear active anchor and hold steps | Pure anchor test passed |
| C.6 Hold duration | `anchor_hold_steps` is an integer step counter, not seconds | Pure anchor test passed |
| C.7 Mask/weight semantics | `m_t/m_star` are boolean validity; `w_t/w_star` are reliability weights; `w_star` gates usability but not anchor error | Pure anchor test passed |
| C.8 Valid anchor fraction | Fixed structural denominator; inactive reports zero; active below threshold resets invalid; no `sum(m_star)` denominator | Pure anchor test passed |
| C.9 Masked anchor error | `anchor_error = m_t_float * m_star_float * w_t * (r_t - r_star)` | Pure anchor test passed |
| C.10 Huber helper/loss | Pure per-element Huber helper; anchor loss reduced over fixed structural denominator; zero usable beams produce zero/no NaN | Pure anchor test passed |
| C.11 Diagnostics | Public `anchor_error_mean/max` are weighted residual diagnostics over usable beams; reset steps report post-transition inactive metrics | Pure anchor test passed |
| C.12 Structural mask | Optional `[H,V]` structural mask; all-ones default; reject per-env masks; denominator uses structural mask sum | Pure anchor test passed |
| C.13 Fail-fast validation | Bad shapes/devices/dtypes/non-finite inputs fail fast; only `v_cmd [N,1,3] -> [N,3]` is normalized | Pure anchor test passed |
| C.14 Return boundary | `AnchorStepOutput.metrics` contains `[N,1]` public scalar diagnostics; dense tensors are only in `cache` | Pure anchor test passed |
| C.15 Env actor contract | `env.py` writes scalar metrics to `info`; dense cache remains internal; actor obs does not contain `anchor_*` keys | Source-level env integration test passed |

### Added Test File

- `training/unit_test/test_instinctrl_anchor.py`

### Actual Commands and Results

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_anchor.py` | Passed: `11 passed, 1 warning`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py` | Passed: `25 passed, 2 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/anchor.py training/scripts/env.py training/unit_test/test_instinctrl_anchor.py` | Passed. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python - <<'PY' ... TorchRL int64 spec probe ... PY` | Passed: int64 `UnboundedContinuousTensorSpec` is supported for `anchor_reset_reason`. |

### C Scope Boundary

- Complete in C: anchor lifecycle, masked error, robust loss helper, scalar diagnostics, passive env integration, tests.
- Deferred beyond C: anchor reward integration, B3 ablation execution, observability logger, ICS attenuation, reward redesign, training convergence.

## instinctRL-D: Observability Logger

**Current verdict**: COMPLETE. Observability logger unit tests and A/B/C/D regression tests pass in the activated NavRL conda environment. Env integration is passive and disabled by default. Actor observation remains `lidar_grid` + `state_vec`.

### Required D Tests

| Test | Required evidence | Current status |
|------|-------------------|----------------|
| D.1 Config validation | Positive `rank_tol`, finite positive `condition_number_cap`, valid mode only | Pure observability test passed |
| D.2 Proxy mode | `J_i=-normalized_ray_direction_i`; proxy labeled with `is_proxy=1`, `mode_code=0` | Pure observability test passed |
| D.3 Normal mode | `J_i=-n_i`; normals normalized; invalid/near-zero normals excluded; `sqrt(w)` row scaling | Pure observability test passed |
| D.4 Finite-difference mode | `pinv(DeltaP) @ Delta r_i`; K/rank validation; exact and overdetermined synthetic recovery | Pure observability test passed |
| D.5 Mode precedence | Proxy always proxy; offline chooses FD, then normals, then proxy fallback; malformed supplied inputs fail fast | Pure observability test passed |
| D.6 SVD/rank metrics | Full-rank, rank-2, rank-1, insufficient rows, finite capped condition number | Pure observability test passed |
| D.7 Weak direction | Cache-only weak direction from `Vh[-1]`; zero for insufficient/rank-0 cases | Pure observability test passed |
| D.8 Drift correlation helper | Missing drift zero; finite drift norm; absolute projection onto weak direction | Pure observability test passed |
| D.9 Public metrics boundary | Scalar `[N,1]` metrics; dense J/SVD internals in cache only | Pure observability test passed |
| D.10 Env actor contract | `env.py` actor obs block remains only `lidar_grid` and `state_vec`; no observability/J/normal/map/odom actor fields | Source-level env integration test passed |

### Added Test File

- `training/unit_test/test_instinctrl_observability.py`

### Actual Commands and Results

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observability.py` | Passed: `9 passed, 1 warning`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py` | Passed: `34 passed, 2 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/observability.py training/scripts/env.py training/unit_test/test_instinctrl_observability.py` | Passed. |
| TorchRL int64 spec probe for `UnboundedContinuousTensorSpec((1,), dtype=torch.long)` | Passed for `observability_mode_code`. |

### D Scope Boundary

- Complete in D: range-Jacobian/proxy logger, scalar metrics, dense cache, drift projection primitive, passive env integration, tests.
- Deferred beyond D: plot generation, full evaluation report matrix, ICS attenuation, reward integration, training convergence.

## instinctRL-E: ICS Attenuation

**Current verdict**: COMPLETE. ICS attenuator unit tests and A/B/C/D/E regression tests pass in the activated NavRL conda environment. Env/train integration is disabled by default and preserves the actor-clean `lidar_grid` + `state_vec` contract.

### Required E Tests

| Test | Required evidence | Current status |
|------|-------------------|----------------|
| E.1 Config validation | Positive `a_max` and `velocity_limit`; valid clearances; `0 < min_reliability <= 1`; `brake_mode="zero"` only | Pure ICS test passed |
| E.2 Shape/device validation | Accept `[N,L,H,V]` and `[N,L,R]` histories; rays `[R,3]`/`[N,R,3]`; commands `[N,3]`/`[N,1,3]`; malformed inputs fail | Pure ICS test passed |
| E.3 Empty active set | No valid/reliable/closing beams gives beta 1 and preserves command unless clipped | Pure ICS test passed |
| E.4 Emergency bypass | Reliable latest clearance below threshold forces beta 0 and zero final command | Pure ICS test passed |
| E.5 Monotonic beta | Lower clearance or higher speed does not increase beta | Pure ICS test passed |
| E.6 Active set rules | Invalid, low-reliability, non-closing, outside-horizon beams inactive; inside-horizon beams active; ratios clamp to beta 1 | Pure ICS test passed |
| E.7 Range-rate behavior | Finite-difference estimate cached; default flag does not affect beta; enabled flag can activate on negative rate | Pure ICS test passed |
| E.8 Command clipping | Beta computed from unclipped command; final norm clipped; direction preserved; scalar speeds/clip ratio shaped `[N,1]` | Pure ICS test passed |
| E.9 History accessors | Builder and env expose range/mask/weight history; copy protects internals; latest/previous ordering correct | Builder unit + env source-level test passed |
| E.10 Source-level safety | `ics.py` has no privileged deployed dependencies; actor block remains `lidar_grid` + `state_vec`; `train.py` applies ICS before body-to-world adapter and stores `v_final_b` | Source-level integration test passed |

### Added Test File

- `training/unit_test/test_instinctrl_ics.py`

### Actual Commands and Results

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_ics.py` | Passed: `10 passed, 1 warning`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py training/unit_test/test_instinctrl_ics.py` | Passed: `44 passed, 2 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/ics.py training/scripts/instinctRL/observation.py training/scripts/env.py training/scripts/train.py training/unit_test/test_instinctrl_ics.py` | Passed. |

### Runtime Smoke

No Isaac GPU runtime smoke was run for instinctRL-E in this environment. CUDA/NVML is not visible locally, so the optional command `python training/scripts/train.py instinctRL.mode=smoke instinctRL.ics.enabled=true env.num_envs=4 env_dyn.num_obstacles=0` is recorded as skipped here. A later GPU-side smoke should verify live `ics_*` info metrics and attenuated action execution.

### E Scope Boundary

- Complete in E: command attenuation, scalar diagnostics, cache-only dense internals, history accessors, smoke-path integration, and tests.
- Not implemented in E: reward/training changes, actor observation changes, surface-normal/map/odom/SLAM/pose/dynamic-obstacle deployed dependencies, D plotting, training convergence.

### instinctRL-F: Reward Integration

**Current verdict**: COMPLETE for reward integration/readiness. Training convergence is not proven. The trainable governor head remains pending, so F acceptance here is reward path integration and auditability, not learned-governor success.

### Required F Tests

| Test | Required evidence | Current status |
|------|-------------------|----------------|
| F.1 Config validation | Finite non-negative weights, positive `max_reward_abs`, valid clearance thresholds and anchor valid fraction | Pure reward test passed |
| F.2 Tracking reward | `v_final_b == v_cmd_b` is better than mismatch under command-consistency proxy | Pure reward test passed |
| F.3 Beta/emergency gating | Low beta or emergency removes/reduces unsafe tracking penalty and emits ICS compliance offset | Pure reward test passed |
| F.4 Anchor reward | Inactive anchor gives zero; active anchor penalizes anchor loss; low valid fraction masks term | Pure reward test passed |
| F.5 Safety | Lower MID360 clearance gives worse reward; missing/invalid clearance remains finite | Pure reward test passed |
| F.6 Intervention | Lower beta gives larger intervention penalty | Pure reward test passed |
| F.7 Smoothness | Larger final-command jump is penalized | Pure reward test passed |
| F.8 Collision | Collision flag adds large negative term | Pure reward test passed |
| F.9 Total reward | Total equals sum of logged components after clipping/scaling and stays finite | Pure reward test passed |
| F.10 Disabled modules | Anchor/ICS disabled paths use zero/default terms | Pure reward test passed |
| F.11 Actor contract | Reward inputs are not added to actor obs; actor obs remains `lidar_grid` + `state_vec` | Source-level test passed |
| F.12 Privileged boundary | Default config does not require actual velocity; optional actual velocity is labeled reward-only | Pure reward/source test passed |
| F.13 Env integration | Reward components are accumulated in `stats`; old reward path remains when disabled | Source-level test passed |

### Added Test File

- `training/unit_test/test_instinctrl_rewards.py`

### Actual Commands and Results

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_rewards.py` | Passed: `10 passed, 1 warning`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py training/unit_test/test_instinctrl_ics.py training/unit_test/test_instinctrl_rewards.py` | Passed: `54 passed, 2 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/rewards.py training/scripts/env.py training/scripts/instinctRL/__init__.py training/unit_test/test_instinctrl_rewards.py` | Passed. |
| TorchRL spec probe for reward component stats construction before spec expansion | Passed. |

### Runtime Smoke

No Isaac GPU runtime smoke was run for instinctRL-F in this environment. CUDA/NVML is not visible locally, so the optional command `python training/scripts/train.py instinctRL.mode=smoke instinctRL.reward.enabled=true env.num_envs=4 env_dyn.num_obstacles=0` is recorded as skipped here. A GPU-side smoke should verify live reward component stats and controller execution.

### F Scope Boundary

- Complete in F: reward computer, config, env reward switch, component stats logging, actor/privileged-boundary tests.
- Not complete in F: trainable governor head, first stable learned-governor training run, G baseline matrix, H real-robot deployment.

### instinctRL-G: Baselines
- B0–B8 config isolation
- Explicit input-schema logs per baseline
- Required metrics per baseline

### instinctRL-H: Real-Robot Deployment
- No odom/map in actor input
- Latency logs
- Safety logs
- Bag replay audit
