# instinctRL-A/B Closeout Acceptance Review

> **Date**: 2026-07-04
> **Stage**: Historical B-closeout before the B-fix implementation pass
> **Final conclusion**: instinctRL-C was NO-GO at this checkpoint. See `instinctRL-B_observation_history_buffer.md` for the later B-fix implementation and remaining runtime validation blockers.

> **Supersession note**: This report records the pre-fix closeout facts. Later on 2026-07-04, code fixes addressed the listed B-FIX implementation blockers, but instinctRL-B remained partial because runtime/PPO validation could not be executed locally.

## Scope

This review closes out instinctRL-A and instinctRL-B before instinctRL-C. It does not start instinctRL-C and does not modify functional code.

Authority order:

1. Code contents are facts.
2. `docs/instinctRL_Development_Handbook_v1_1_platform_locked.tex` is the acceptance standard.
3. `docs/instinctRL_devlog/*` is reference history only and cannot override code facts.

## Files Inspected

- `isaac-training/training/scripts/train.py`
- `isaac-training/training/scripts/env.py`
- `isaac-training/training/scripts/ppo.py`
- `isaac-training/training/scripts/instinctRL/audit.py`
- `isaac-training/training/scripts/instinctRL/governor.py`
- `isaac-training/training/scripts/instinctRL/command_adapter.py`
- `isaac-training/training/scripts/instinctRL/observation.py`
- `isaac-training/training/cfg/train.yaml`
- `isaac-training/training/cfg/ppo.yaml`
- `isaac-training/training/cfg/drone.yaml`
- `isaac-training/third_party/OmniDrones/omni_drones/utils/torch.py`
- `isaac-training/third_party/OmniDrones/omni_drones/robots/drone/multirotor.py`
- `isaac-training/third_party/OmniDrones/omni_drones/controllers/lee_position_controller.py`
- `docs/instinctRL_Development_Handbook_v1_1_platform_locked.tex`
- `docs/instinctRL_devlog/*`

## Acceptance Criteria From Handbook

instinctRL-A acceptance:

- B0 direct velocity baseline runs on TASLAB_UAV / MID360.
- Actor audit passes.
- Action remains velocity-controller based through the existing `VelController(LeePositionController)` path.
- B0 is a smoke-test / infrastructure baseline, not a learned-policy success criterion.

instinctRL-B acceptance:

- Build MID360 range `r_t`, valid-return mask `m_t`, reliability weights `w_t`, timestamps, IMU cues, command, previous output, and fixed history.
- Use stable MID360 ray ordering from `LivoxMid360Pattern` or an active RayCaster configured with MID360 angles/order.
- Tests cover MID360 shape/ray count, ray-order stability, timestamp monotonicity, mask/dropout behavior, reliability bounds, history rollover, stale-frame handling, and actor input contract.

## Code Evidence

- `env.py` creates the active RayCaster with `pattern_cfg=patterns.BpearlPatternCfg(...)`. This is not full MID360 ray-ordering success under the handbook or decision log.
- `env.py` creates `MID360ObservationBuilder` when `cfg.instinctRL.enabled` is true and passes `lidar_hbeams`, `lidar_vbeams`, and `lidar_range`.
- `observation.py` builds range, mask, staleness-weighted reliability, IMU cues, `v_cmd`, `prev_action`, and history tensors.
- `env.py` calls `MID360ObservationBuilder.build(...)` without a `prev_action` argument, so the history does not receive the actual issued governor/controller output.
- `train.py` runs a 500-step B0 smoke path when `instinctRL.enabled=true`, then closes Isaac Sim and returns before PPO construction, collection, or training.
- `ppo.py` has a hybrid actor input path consuming `("agents", "observation", "lidar_grid")` and `("agents", "observation", "state_vec")`.
- `audit.py` `check_actor_input()` scans actor observation key names for forbidden substrings. It does not verify the internal provenance of `state_vec`.
- `command_adapter.py` uses `quat_rotate_inverse` for body -> world. Local OmniDrones uses `quat_rotate_inverse(self.rot, vel_w)` to compute body velocity from world velocity, so adapter direction requires verification and likely correction.
- `train.yaml` sets `instinctRL.enabled: true` and `baseline.id: "direct_velocity"`.
- `ppo.yaml` configures a fixed B0 governor with `alpha_fixed: 1.0`, `v_corr_limit: 0.0`, and velocity limit 2.0.

## A Verdict

`instinctRL-A`: PASS with open verification item(s).

A is accepted as B0 smoke-test / infrastructure baseline, not learning success. The code establishes the B0 minimal governor, adapter boundary, config namespace, audit calls, and a 500-step smoke route. This does not prove policy learning, reward quality, or full training.

Open A verification item before C:

- Verify and fix `BodyToWorldVelocityAdapter` frame direction. Shape/no-NaN smoke is not sufficient.

## B Verdict

`instinctRL-B`: PARTIAL / NOT FULLY ACCEPTED.

Partial acceptance:

- Observation builder exists.
- Hybrid actor input (`lidar_grid`, `state_vec`) exists.
- PPO has a path that can consume the hybrid observation format.

Not fully accepted:

- Active env ray source still uses `patterns.BpearlPatternCfg`.
- Full MID360 ray ordering is not proven in the training path.
- Previous issued governor/controller output is not fed back into observation history.
- Actor audit does not verify `state_vec` provenance.
- Required Observation / History Buffer tests are not recorded as passing.
- `instinctRL.enabled=true` does not validate PPO/training because `train.py` returns after B0 smoke.

## Blocking Issues Before C

| ID | Issue | Required fix |
|----|-------|--------------|
| B-FIX-001 | Active RayCaster uses `BpearlPatternCfg` | Wire or prove MID360 ray ordering in active instinctRL env path. |
| B-FIX-002 | Adapter frame direction unverified | Add known-quaternion tests and fix body -> world transform if needed. |
| B-FIX-003 | PPO path not validated under instinctRL mode | Add explicit smoke-only mode or allow short PPO collect/update validation. |
| B-FIX-004 | `prev_action` not actual previous issued command | Feed previous governor/controller output into builder history. |
| B-FIX-005 | Actor audit only scans keys | Add schema/provenance or perturbation audit for `state_vec`. |
| B-FIX-006 | B tests missing | Add/run required B tests and record evidence. |

## Required Tests Before C

- Adapter frame unit tests with identity, yaw 90 deg, roll/pitch cases.
- Active sensor integration test proving MID360 ray count and deterministic ordering.
- Observation tests for raw range, valid mask, reliability bounds, timestamp monotonicity, stale-frame markers, and history rollover.
- Previous-action feedback test proving history contains the last issued governor/controller command.
- Actor provenance test proving no pose, linear velocity, odometry, map, or privileged simulator state enters actor tensors.
- Short instinctRL PPO collect/update smoke or explicit config-level separation of B0 smoke from training.

## Final Go/No-Go

- `instinctRL-A`: PASS with open verification item(s)
- `instinctRL-B`: PARTIAL / NOT FULLY ACCEPTED
- `instinctRL-C`: NO-GO until B-fix checklist passes
