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

After synchronization commit `5c9ab7a`, raw `nvidia-smi` exited `0` and the
specified NavRL Torch command printed `True` with exit `0`. Exactly one stored
disabled replay then ran at
`tests/artifacts/r5j_default_equivalence/20260714_234801/attempts/20260716T161710730878Z-5c9ab7a/`.
Its source commit, checkpoint SHA-256, seed `0`, argv, freshness, CUDA
preflight, eval exit `0`, legacy JSON, gates, and eight exact-zero diagnostics
all passed; the recorded result is `GO (design only)`. The captured stderr
also contains the Isaac/W&B shutdown segfault trace; it is retained verbatim
in the artifact, while the subprocess exit and strict comparator result remain
`0` and `GO (design only)`, respectively. This authorizes design only, not an
enabled execution. The prior `19 passed, 1 warning`, `163 passed, 13 warnings`,
and Evidence-3 limitations (no contact-body identity, surface normals,
measured deceleration, or final safety-fix proof) remain unchanged.
