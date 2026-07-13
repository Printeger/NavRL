# instinctRL-A2-R: Station Objective Repair

**Status**: source/unit implemented  
**Date**: 2026-07-09  
**Trigger**: 1M short diagnostic failed station-keeping despite static collision safety.

## Problem

The learned governor passed runtime stability and static safety checks, but the 1M diagnostic showed large zero-command drift. The existing reward/curriculum allowed `v_corr` to create motion when `v_cmd=0`, and did not directly penalize command amplification when `v_cmd!=0`.

## Implemented Repair

- Added null-command reward terms:
  - `reward_null_command_speed`: penalizes actual body velocity when `||v_cmd|| <= command_eps`.
  - `reward_null_command_output`: penalizes final issued body command when `||v_cmd|| <= command_eps`.
- Added nonzero-command preservation terms:
  - `reward_proxy_tracking`: penalizes `||v_final_b - v_cmd_b||` only when ICS is not attenuating.
  - `reward_command_amplification`: penalizes `max(||v_final_b|| / ||v_cmd_b|| - 1, 0)` only when ICS is not attenuating.
- Changed formal reward defaults:
  - `anchor_weight=2.0`
  - `null_command_speed_weight=2.0`
  - `null_command_output_weight=0.5`
  - `proxy_tracking_weight=0.25`
  - `command_amplification_weight=0.5`
- Added command curriculum profiles:
  - `station_first` for formal training.
  - `diagnostic_mixed` for fixed short diagnostic tracking pass.
- Added eval diagnostics:
  - `eval/handbook.null_command_speed_mean`
  - `eval/handbook.null_command_output_speed_mean`
  - `eval/handbook.command_amplification_mean`
  - `eval/handbook.command_amplification_rate`

## Boundary

- No privileged actor input was added.
- No global-position drift reward was added.
- Station drift remains an eval-only metric.
- Dynamic-obstacle claims remain out of scope until dynamic obstacles are MID360 RayCaster-visible.

## Acceptance

Source/unit acceptance is the instinctRL reward/task/eval diagnostic test set passing. Runtime acceptance requires a new 1M static MID360 retrain and short diagnostic eval passing the gate in the root-cause audit.

