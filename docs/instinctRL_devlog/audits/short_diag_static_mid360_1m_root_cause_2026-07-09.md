# Short Diagnostic 1M Root-Cause Audit

**Date**: 2026-07-09  
**Artifact**: `docs/instinctRL_devlog/tests/artifacts/short_diag_static_mid360_1m_eval.json`  
**Run**: `wandb/offline-run-20260709_163610-391zkphm/files/checkpoint_final.pt`

## Verdict

The 1M corrected static MID360 run is a useful diagnostic failure, not a formal-training go signal.

The model is safe in the static diagnostic scene, but it fails the handbook station-keeping objective. The failure is rooted in objective/curriculum semantics, not in platform lock, MID360 attachment, or ICS.

## Key Metrics

| Metric | Value | Interpretation |
|---|---:|---|
| `station_keeping_drift_mean` | `3.9156 m` | Fails station-keeping. |
| `station_keeping_drift_p95` | `7.7390 m` | Tail drift is severe. |
| `station/anchor_error_mean` | `3.1215` | Measurement-space anchor is not constraining motion. |
| `tracking_rmse_actual_body_vs_v_cmd` | `0.6035 m/s` | Tracking is not yet tight. |
| `command_preservation_ratio` | `1.3801` | Learned governor often amplifies commands. |
| `safety_collision_rate` | `0.0` | Static collision safety is good. |
| `safety_min_clearance_p05` | `1.7672 m` | Clearance is acceptable for this diagnostic. |
| `ics_intervention_frequency` | `0.0` | Policy is not relying on ICS in this run. |
| `ics_violation_rate` | `0.00159` | Low violation rate. |
| `observability_rank_mean` | `3.0` | Proxy observability is not degenerate. |

## Root Cause

1. Null-command station-keeping was underconstrained.
   - The existing tracking reward was gated off when `||v_cmd||` was near zero.
   - Under null command, reward pressure came mainly from anchor loss.
   - `anchor_weight=0.5` was too weak relative to learned-governor freedom.

2. The learned governor could create motion at zero command.
   - `v_gov = alpha * v_cmd + v_corr`.
   - When `v_cmd=0`, `v_corr` can still produce nonzero velocity.
   - There was no explicit null-command penalty on actual velocity or final issued command.

3. The command curriculum was not station-first.
   - Early training mixed nonzero command tracking too soon.
   - The 1M run entered mixed command modes before proving zero-command station behavior.

4. Command amplification was not penalized directly.
   - Actual-velocity reward alone can make a policy track by issuing oversized commands through the velocity controller.
   - Eval exposed this as `command_preservation_ratio=1.3801`.

## Required Repair

- Add null-command speed/output penalties.
- Increase anchor reward pressure for station-keeping.
- Add proxy command-chain tracking and command-amplification penalties for nonzero safe commands.
- Switch formal training to a station-first command curriculum.
- Keep short diagnostic eval fixed to `diagnostic_mixed` tracking so results remain comparable.

## Go/No-Go Gate For Next 1M Diagnostic

The next 1M static MID360 diagnostic may unlock formal longer training only if all hold:

- `station_keeping_drift_mean <= 1.0 m`
- `station_keeping_drift_p95 <= 2.0 m`
- `anchor_error_mean <= 1.0`
- `tracking_rmse_actual_body_vs_v_cmd <= 0.45 m/s`
- `0.75 <= command_preservation_ratio <= 1.10`
- `command_amplification_rate <= 0.10`
- `safety_collision_rate == 0.0`
- `safety_min_clearance_p05 >= 1.0 m`
- `ics_violation_rate <= 0.005`

