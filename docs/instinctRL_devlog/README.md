# instinctRL Development Log

> **Project**: instinctRL — Measurement-space velocity governor for TASLAB_UAV + Livox MID360  
> **Handbook**: `instinctRL_Development_Handbook_v1_1_platform_locked.tex`  
> **Scientific Target**: Paper 1 (`paper1_vel_ctrl.tex`)  
> **Created**: 2026-07-03

---

## Purpose

This devlog tracks the design, implementation, testing, and deployment of instinctRL. It serves as the single source of truth for development status, decisions, and known issues.

## Structure

```
docs/instinctRL_devlog/
├── README.md              ← This file
├── NEXT_PROMPT.md         ← Current bounded next step and copy/paste agent prompt
├── CHANGELOG.md           ← Chronological change log
├── DEV_STATUS.md          ← Current development status dashboard
├── DECISION_LOG.md        ← Go/no-go and scope-boundary decisions
├── TEST_PLAN.md           ← Required tests and runtime acceptance gates
└── tickets/               ← Individual ticket reports
    ├── instinctRL-0_platform_sensor_audit.md
    ├── instinctRL-A_direct_velocity_governor.md
    ├── instinctRL-B_observation_history_buffer.md
    ├── instinctRL-C_measurement_space_anchor.md
    ├── instinctRL-D_observability_logger.md
    ├── instinctRL-E_ics_inspired_attenuation.md
    ├── instinctRL-F_reward_integration_training.md
    ├── instinctRL-G_baselines_ablations.md
    └── instinctRL-H_real_robot_deployment.md
```

## Conventions

1. **Naming**: All code, config, and log artifacts use `instinctRL` (not INSTINCT, NavRL, or Paper1).
2. **Platform lock**: Normal mode = TASLAB_UAV + Livox MID360. Any deviation must be labeled as debugging-only.
3. **Method lock**: Velocity-governor with body-frame velocity commands. No CTBR, body-rate, thrust, or motor-level learned actions.
4. **Actor input contract**: No pose, odometry, explicit velocity, map, SLAM state, or privileged simulator state in deployed actor.
5. **Ticket workflow**: Each ticket → independent branch → PR with audit checks → merge to main.
6. **Dependencies**: instinctRL-0 must complete before any other ticket.

## Quick Links

- [Development Status](./DEV_STATUS.md)
- [Current Next-Step Prompt](./NEXT_PROMPT.md)
- [Test Plan](./TEST_PLAN.md)
- [Decision Log](./DECISION_LOG.md)
- [Change Log](./CHANGELOG.md)
- [Platform & Sensor Audit (instinctRL-0)](./tickets/instinctRL-0_platform_sensor_audit.md)
- [Full Audit Report](../instinctRL_0_platform_sensor_audit.md)
- [Development Handbook](../instinctRL_Development_Handbook_v1_1_platform_locked.tex)
- [Scientific Paper](../paper1_vel_ctrl.tex)

## A2-R5J Current Execution Boundary

`8298a7d256bec6a82dee49d9af41a87628135ed6` repaired disabled-replay
provenance, and `927e166` is the current pushed baseline on
`origin/a2-r5j-default-off-residual`. The historical
`20260716T074648884514Z-0a6a2be` item remains a dirty-worktree,
preflight-only `HOLD`: eval did not run and it created no replay JSON, so it
did not consume the single clean disabled replay.

CUDA is reported ready in the current environment (`nvidia-smi` exit `0`, and
NavRL Torch reports CUDA available with one device), but that observation is
not replay authorization. The current decision is pending the four ordered
steps in `NEXT_PROMPT.md`: commit/push this documentation synchronization,
run fresh raw CUDA gates from that clean commit, run exactly one stored
disabled replay only if both gates pass, then record `GO (design only)` or
`HOLD`. The prior `19 passed, 1 warning`, `163 passed, 13 warnings`, and
Evidence-3 limitations (no contact-body identity, surface normals, measured
deceleration, or final safety-fix proof) remain unchanged.
