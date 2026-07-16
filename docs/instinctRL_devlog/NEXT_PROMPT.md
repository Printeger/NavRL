# instinctRL Next-Step Prompt

> **Updated**: 2026-07-16
>
> **Current step**: A2-R5J test-first, default-off ICS residual-margin pre-emption
>
> **Authority**: implementation/equivalence validation only; no enabled behavior experiment or training

## Current Decision

The latest Evidence-3 results are sufficient to proceed to a bounded R5J implementation-validation step. They are not sufficient to resume parameter sweeps or learned-governor training.

The next experiment is a default-equivalence replay of an existing R5G checkpoint after a default-off R5J implementation passes source, unit, config, and actor-audit tests. The replay must leave the new guard disabled and must change only the output `result_path` plus an explicit disabled override if needed.

## Copy/Paste Prompt

```text
Continue instinctRL from the current repository state. Start by reading:

- docs/instinctRL_Development_Handbook_v1_1_platform_locked.tex
- docs/instinctRL_devlog/DEV_STATUS.md
- docs/instinctRL_devlog/TEST_PLAN.md
- docs/instinctRL_devlog/DECISION_LOG.md
- docs/instinctRL_devlog/tickets/instinctRL-A2-R5_hypothesis_sweep_plan.md, especially Evidence-3 and A2-R5J
- docs/instinctRL_devlog/tests/artifacts/r5e3_braking_residual/20260714_234801/
- isaac-training/training/scripts/instinctRL/ics.py
- isaac-training/training/unit_test/test_instinctrl_ics.py

Task: execute only the A2-R5J test-first, default-off implementation and default-equivalence validation step.

Evidence boundary:

- In downatten collision windows, v_final_b is already zero while realized body speed remains about 0.193-0.197 m/s.
- Conservative residual-to-collision p05 is negative in every downatten collision window.
- Worst-ICS-beam residual p05 is negative in the strongest collision windows.
- Contact-body telemetry, exact surface normals, and measured deceleration are missing. Do not claim an exact physics/contact proof or a final safety fix.

Required implementation scope:

1. Add tests first for a deployable, actor-clean ICS residual-margin pre-emption guard.
2. Add an explicit config enable flag that defaults to false in code, train.yaml, and eval.yaml. Add finite, nonnegative configuration for the residual margin and collision-clearance threshold. Keep current ICS outputs exactly unchanged while the flag is false.
3. When enabled, compute per-beam residual only from existing actor-clean ICS inputs: reliable MID360 range/mask/weight history, body-frame ray directions, governed body command, range-rate history, and existing braking constants. Do not use actual/root velocity, pose, odometry, root height, map/SLAM, contact labels, simulator state, surface normals, or eval-only r5e3 diagnostics.
4. Define and document the residual consistently with the Evidence-3 concept: available clearance above the configured collision threshold minus latency distance and stopping distance. Use eligible/reliable closing beams only. Preserve the existing emergency bypass and final velocity bounds. Keep all new dense per-beam values in the ICS internal cache; expose at most scalar info/eval diagnostics.
5. Before settling the behavior, demonstrate with a synthetic test that the guard has a non-redundant trigger case: current ICS would not yet fully stop, but the configured residual margin makes the guard pre-empt. Also test positive-residual/no-closing/invalid-beam cases remain unchanged.

Required tests and acceptance:

- Disabled/default equivalence: same v_final_b, existing public metrics, and existing cache values as pre-R5J behavior across empty-active, normal, emergency, downward-attenuation, and clipping cases.
- Enabled synthetic exhausted-margin case pre-empts as specified.
- Positive residual, no closing evidence, unreliable/invalid beams, and empty active set do not spuriously pre-empt.
- Invalid threshold/margin values fail configuration validation.
- Actor-audit tests reject every new R5J diagnostic/cache key as actor input; actor observation remains exactly lidar_grid + state_vec.
- TASLAB_UAV + Livox MID360, the learned action [alpha, v_corr_x, v_corr_y, v_corr_z], body-frame velocity-governor method, hard gates, reward defaults, controller, and privileged height-filter default remain unchanged.
- Run py_compile, targeted ICS/config/actor-audit tests, then the full training/unit_test/test_instinctrl_*.py suite.

Default-equivalence replay:

- Only after all source/unit tests pass, take the stored eval_command for r5g_downatten_z010 from docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_234801/summary.json.
- Replay that existing checkpoint with the R5J guard explicitly disabled. Change only result_path and, if the stored command needs it for audit clarity, add instinctRL.ics.<new_enable_flag>=false.
- Store the artifact under docs/instinctRL_devlog/tests/artifacts/r5j_default_equivalence/<source-run-id>/.
- Compare the replay with the existing R5G/R5E3 baseline and report command-chain, handbook, termination, gate, and new-disabled-diagnostic equivalence. Do not reinterpret stochastic drift as improvement; document the comparison method and tolerances.

Stop conditions:

- Do not run an enabled R5J replay, dry-run variant, 128k sweep, 1M run, formal training, warm-start, or promotion.
- Do not change hard gates, actor observations, platform/sensor settings, governor/controller method, reward defaults, or privileged safety-filter defaults.
- If the disabled path is not equivalent, or the new guard is behaviorally redundant with existing ICS in all synthetic cases, stop, diagnose, and record the failure instead of widening scope.

Documentation and Git:

- Update DEV_STATUS.md, TEST_PLAN.md, CHANGELOG.md, DECISION_LOG.md, and the A2-R5 ticket with exact files changed, commands run, results, artifact paths, equivalence outcome, caveats, and the next go/no-go decision.
- Preserve unrelated worktree changes.
- Commit the completed bounded step and push the current branch only after tests and documentation are complete.

Exit decision:

- If every required test passes and the disabled replay is equivalent, authorize only the design of one enabled, single-variable R5J dry-run experiment for the next turn; do not execute it in this turn.
- Otherwise keep R5J blocked, record the exact failing gate, and do not proceed to enabled behavior evaluation or training.
```

## Expected Output of This Prompt

The prompt should end with one of two recorded outcomes:

- `GO (design only)`: default-off implementation and equivalence replay passed; one enabled single-variable dry-run may be designed next, but not executed.
- `HOLD`: a source/unit/audit/equivalence condition failed or the mechanism was redundant; no enabled experiment or training is allowed.
