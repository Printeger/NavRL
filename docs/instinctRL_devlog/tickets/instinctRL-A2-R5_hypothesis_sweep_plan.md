# instinctRL-A2-R5 Hypothesis-Driven Sweep Plan

**Status**: R5F dry-run sweep design and validation completed; no `--execute`, 1M, promotion, warm-start, parameter change, or hard-gate change
**Created**: 2026-07-12  
**Owner for next Codex turn**: update this document after every code change, dry-run, sweep, eval, and decision  
**Artifact root**: `docs/instinctRL_devlog/tests/artifacts/sweeps/`

## Purpose

A2-R5 is not a broad tuning pass. It is a bounded experiment loop for finding a 128k-screening configuration that can be promoted to 1M without drifting away from the instinctRL command-governor objective.

The active target is:

- keep the actor observation contract clean;
- keep the A2-R3 soft-null station-correction semantics;
- pass the current hard gates at 128k before any 1M run;
- record every failed branch so later Codex sessions do not repeat it.

Formal long training remains on HOLD until a candidate passes the 128k hard gate, then passes a 1M confirmation, then shows multi-seed stability.

## Non-Negotiable Rules

- Do not run 1M before a 128k candidate passes all hard gates.
- Do not relax hard gates to promote a candidate.
- Do not warm-start from failed A2-R2, A2-R3, or A2-R4 checkpoints.
- Do not let height, actual velocity, anchor internals, ICS dense state, map state, or simulator pose enter actor observation.
- Treat `r3_soft_null_min025` and `r4_vcorr035_amp_safety` as evidence, not final parameters.
- Prefer small hypothesis-driven sweeps over manual one-off command overrides.
- After every run, update this file before starting the next run.

## Current Hard Gates

The hard gates are the gates implemented by `training/scripts/instinctRL/gates.py` at the time of this plan:

| Gate | Requirement |
|---|---:|
| station drift mean | `<= 1.3` |
| station drift p95 | `<= 2.6` |
| station null actual speed mean | `<= 0.08` |
| station anchor error mean | `<= 2.0` |
| tracking RMSE actual | `<= 0.45` |
| tracking preservation ratio | `0.75 <= value <= 1.05` |
| tracking command amplification mean | `<= 0.05` |
| tracking command amplification rate | `<= 0.15` |
| safety collision rate | `== 0` |
| safety min clearance p05 | `>= 1.0` |
| ICS violation rate | `<= 0.005` |
| termination collision | `== 0` |
| termination below bound | `== 0` |
| termination above bound | `== 0` |

Promotion rule: a candidate must pass all gates, not merely rank highest by score.

## Previous Evidence

### A2-R2 Sweep: `20260711_111713`

Artifact: `docs/instinctRL_devlog/tests/artifacts/sweeps/20260711_111713/summary.json`

| Variant | Gates | Score | Main failures | Decision |
|---|---:|---:|---|---|
| `r2_balanced` | 7/15 | 2.995 | drift, null actual speed, anchor, amplification, ICS, below-bound | Reference only; do not promote |
| `r2_null_strong` | 5/15 | 1.101 | drift, null actual speed, anchor, preservation, collision, clearance, ICS, below-bound | Superseded |
| `r2_preserve_strong` | 5/15 | 0.155 | drift, null actual speed, anchor, preservation, collision, clearance, ICS, below-bound | Superseded |
| `r2_base` | 5/15 | 0.154 | drift, null actual speed, anchor, preservation, amplification, clearance, ICS, below-bound | Superseded |
| `r2_safety_strong` | 3/15 | -1.648 | drift, tracking RMSE, preservation, collision, clearance, ICS, termination | Superseded |
| `r2_low_corr` | 3/15 | -1.709 | drift, tracking RMSE, amplification, collision, clearance, ICS, termination | Superseded |

R2 diagnosis:

- Hard-zero null behavior removed useful station correction.
- `r2_balanced` had zero null-command output bias but actual null speed and drift remained high.
- R2 led to A2-R3 soft null semantics.

### A2-R3 Sweep: `20260711_144051`

Artifact: `docs/instinctRL_devlog/tests/artifacts/sweeps/20260711_144051/summary.json`

| Variant | Gates | Score | Main failures | Decision |
|---|---:|---:|---|---|
| `r3_soft_null_min025` | 7/14 | 5.338 | null actual speed, anchor, amplification, clearance, ICS, below-bound | Best R3 direction, not promotable |
| `r3_anchor_strong_safety` | 6/14 | 4.703 | null actual speed, anchor, preservation high, amplification, clearance, ICS, below-bound | Strong anchor hurts tracking/amp |
| `r3_anchor_strong` | 6/14 | 4.216 | null actual speed, anchor, preservation high, amplification, clearance, ICS, below-bound | Do not continue as primary |
| `r3_no_decoder_gate_reward_only` | 5/14 | 3.738 | null actual speed, anchor, amplification, collision, clearance, ICS, termination | Decoder gate still needed |
| `r3_balanced_soft` | 3/14 | 0.424 | drift, null speed, anchor, tracking RMSE, amplification, safety, below-bound | Reject |
| `r3_soft_null_min04` | 2/14 | -1.464 | broad regression | Reject `null_vcorr_gate_min=0.4` as primary |

R3 diagnosis:

- `null_vcorr_gate_min=0.25` is the right soft-null direction.
- `null_vcorr_gate_min=0.4` regresses.
- Stronger anchor lowers some station error but increases command amplification and preservation-high violations.
- Below-bound terminations are height failures, not obstacle-clearance failures.

### A2-R4 Sweep: `20260711_163023`

Artifact: `docs/instinctRL_devlog/tests/artifacts/sweeps/20260711_163023/summary.json`

Implemented R4 source changes:

- Added `reward_height_floor`.
- Passed world/root height `z` as reward-only `height_w`.
- Kept A2-R3 soft-null semantics.
- Set default sweep tag to `a2r4_sweep`.
- Tested six R4 variants.

| Variant | Gates | Score | Main failures | Decision |
|---|---:|---:|---|---|
| `r4_vcorr035_amp_safety` | 7/14 | 6.288 | null actual speed, anchor, amplification mean/rate, clearance p05, ICS, below-bound | Best R4 baseline for R5 |
| `r4_vcorr035_amp` | 7/14 | 6.175 | drift mean, null actual speed, anchor, amplification, ICS, above-bound | Useful evidence: height can flip from below to above |
| `r4_min025_amp_safety` | 6/14 | 4.480 | drift, null speed, anchor, preservation low, clearance, ICS, below-bound | Amp fixed, preservation/height not fixed |
| `r4_anchor5_vcorr035_amp_safety` | 6/14 | 3.841 | null speed, anchor, preservation high, amplification, clearance, ICS, below-bound | Anchor 5.0 is not a primary path |
| `r4_min025_safety_guard` | 5/14 | 2.394 | drift, null speed, anchor, amplification, clearance, ICS, below-bound | Safety alone is insufficient |
| `r4_min025_amp_guard` | 3/14 | 0.036 | drift, null speed, anchor, amplification, collision, clearance, ICS, termination | Reject |

R4 diagnosis:

- Height floor helps but does not make height safe by itself.
- `v_corr_limit=0.35` is useful: it improved drift, clearance, and below-bound relative to many R3/R4 branches.
- `v_corr_limit=0.35` without enough safety/height shaping can produce above-bound termination.
- Strong anchor is not the next lever; it worsens amplification and preservation.
- The best R4 candidate is close on clearance but still far on ICS and null actual speed.

## A2-R5 Hypotheses

R5 should test four specific hypotheses:

1. **Null-speed penalty hypothesis**: actual station motion under zero command is under-penalized. Increasing `null_command_speed_weight` may reduce null actual speed and anchor error without stronger anchor.
2. **Amp pressure hypothesis**: R4's best candidate is only slightly over amp gates, so moderate `command_amplification_weight` increase may pass amp without destroying preservation.
3. **Height floor hypothesis**: R4 below-bound failures persist because `height_floor_weight=8.0` is too weak for short training; increasing it may zero below-bound without adding actor-observation leakage.
4. **Safety margin hypothesis**: R4 safety guard nearly passed clearance but not ICS. Increasing reward and ICS margins together may reduce `ics_violation_rate` and lift `clearance_p05`.

If R5A config-only variants still produce zero pass candidates, stop and move to R5B code changes:

- add an explicit height ceiling or height-band reward;
- consider asymmetric vertical correction limits or a governor-side vertical safety clamp;
- add diagnostics to split horizontal command amplification from vertical correction amplification;
- only then run another 128k sweep.

## A2-R5A Config-Only Sweep Variants

Base for all R5A variants:

```text
instinctRL.mode=train
instinctRL.task=command_governor
instinctRL.command.source=curriculum_generator
instinctRL.command.curriculum_profile=station_first
instinctRL.reward.enabled=true
instinctRL.reward.use_privileged_velocity_for_reward=true
instinctRL.reward.anchor_weight=4.0
instinctRL.reward.null_command_output_weight=0.1
instinctRL.ics.enabled=true
algo.instinctRL.governor.null_vcorr_gate_enabled=true
algo.instinctRL.governor.null_vcorr_gate_eps=0.25
algo.instinctRL.governor.null_vcorr_gate_min=0.25
algo.instinctRL.governor.v_corr_limit=0.35
instinctRL.reward.preservation_high_weight=2.0
instinctRL.reward.command_amplification_weight=2.0
instinctRL.reward.proxy_tracking_weight=0.5
instinctRL.reward.safety_weight=1.2
instinctRL.reward.clearance_margin=0.4
instinctRL.ics.active_horizon_margin=1.0
instinctRL.ics.clearance_margin=0.15
```

Update `training/scripts/instinctRL/sweep.py` default tag to `a2r5_sweep` and replace the default variants with exactly these six variants:

| Variant | Additional overrides | Hypothesis tested |
|---|---|---|
| `r5_null_speed4` | `instinctRL.reward.null_command_speed_weight=4.0` | Can actual station speed drop without stronger anchor? |
| `r5_amp3` | `instinctRL.reward.command_amplification_weight=3.0` | Can the near-miss amp gates pass without breaking preservation? |
| `r5_height16` | `instinctRL.reward.height_floor_weight=16.0` | Can below-bound termination become zero with stronger reward-only height floor? |
| `r5_safety_margin` | `instinctRL.reward.safety_weight=1.5`, `instinctRL.reward.clearance_margin=0.5`, `instinctRL.ics.active_horizon_margin=1.2`, `instinctRL.ics.clearance_margin=0.2` | Can clearance p05 and ICS pass together? |
| `r5_null_amp_height` | `instinctRL.reward.null_command_speed_weight=4.0`, `instinctRL.reward.command_amplification_weight=3.0`, `instinctRL.reward.height_floor_weight=16.0` | Combined station/amp/height correction without extra safety margin |
| `r5_null_amp_height_safety` | all `r5_null_amp_height` overrides plus all `r5_safety_margin` overrides | Candidate bundle if individual levers move in the right direction |

Do not include `anchor_weight=5.0` in R5A unless a later documented result proves it is needed. R4 already showed anchor 5.0 worsens amplification and preservation.

## R5A Implementation Checklist

Use this checklist in the next Codex turn.

- [x] Update `training/scripts/instinctRL/sweep.py` default tag to `a2r5_sweep`.
- [x] Replace the default variant list with the six R5A variants above.
- [x] Keep the dry-run default behavior; `--execute` must still be explicit.
- [x] Add/update sweep unit tests to assert the six `r5_*` variants and default tag.
- [x] Run targeted validation:

```bash
cd /home/mint/rl_dev/NavRL/isaac-training
python -m py_compile training/scripts/instinctRL/sweep.py
python -m pytest -q training/unit_test/test_instinctrl_gates.py
python -m pytest -q training/unit_test/test_instinctrl_*.py
```

- [x] Dry-run:

```bash
cd /home/mint/rl_dev/NavRL/isaac-training
python training/scripts/instinctRL/sweep.py --frames 131072 --seeds 0 --limit 6
```

- [x] Execute only after dry-run review:

```bash
cd /home/mint/rl_dev/NavRL/isaac-training
python training/scripts/instinctRL/sweep.py --execute --frames 131072 --seeds 0 --limit 6
```

- [x] Record results in this document before deciding on 1M, R5B, or another bounded R5A tweak.

## R5A Implementation Backfill - 2026-07-12

Code changes:

- Updated `training/scripts/instinctRL/sweep.py` default `--tag` from `a2r4_sweep` to `a2r5_sweep`.
- Replaced default sweep variants with exactly the six R5A variants: `r5_null_speed4`, `r5_amp3`, `r5_height16`, `r5_safety_margin`, `r5_null_amp_height`, `r5_null_amp_height_safety`.
- Kept dry-run as the default behavior; `--execute` remains explicit and was not run.
- Updated `training/unit_test/test_instinctrl_gates.py` to assert the R5 variant names, R5 base overrides, per-variant additional overrides, default tag, and absence of `anchor_weight=5.0`.

Validation results:

- `python -m py_compile training/scripts/instinctRL/sweep.py`: passed.
- `python -m pytest -q training/unit_test/test_instinctrl_gates.py`: passed, `5 passed in 1.14s`.
- `python -m pytest -q training/unit_test/test_instinctrl_*.py`: passed, `102 passed, 11 warnings in 3.08s`.

Dry-run review:

- Command run: `python training/scripts/instinctRL/sweep.py --frames 131072 --seeds 0 --limit 6`.
- Result: passed as dry-run only with `execute=false`, `frames=131072`, `seed=0`, and six generated jobs.
- Generated artifact paths pointed under `docs/instinctRL_devlog/tests/artifacts/sweeps/20260712_215415/`, but no artifacts were written because `--execute` was not used.
- Generated variants, in order: `r5_null_speed4`, `r5_amp3`, `r5_height16`, `r5_safety_margin`, `r5_null_amp_height`, `r5_null_amp_height_safety`.
- All run names used the `instinctrl_a2r5_sweep_*_131072_seed0` prefix.

Decision:

- R5A was ready for execution after dry-run review. No 128k training sweep, 1M run, or hard-gate change was performed in this implementation pass.

## R5A Execution Backfill - 2026-07-12

Execution:

- Preflight found only A2-R2 `20260711_111713`, A2-R3 `20260711_144051`, and A2-R4 `20260711_163023` summaries; none contained the six `r5_*` variants with `instinctrl_a2r5_sweep_*` run names.
- Command run exactly once: `python training/scripts/instinctRL/sweep.py --execute --frames 131072 --seeds 0 --limit 6`.
- Artifact: `docs/instinctRL_devlog/tests/artifacts/sweeps/20260712_223240/summary.json`.
- Gate truth source: embedded `gate_report` produced by `training/scripts/instinctRL/gates.py`.
- Hard gates were unchanged; `training/scripts/instinctRL/gates.py` was not modified.
- No 1M, formal long training, warm-start, or extra sweep command was executed.

Execution decision:

- No candidate passed all 14 hard gates.
- Best candidate was `r5_null_speed4`, seed `0`, with `9/14` gates and score `8.337`.
- `r5_null_speed4` is not promotable because it failed station null speed, station anchor error, command amplification mean/rate, and `termination_above_bound=0.1875`.
- R5A micro-sweep is not allowed from this result because the best candidate is below `10/14` and has a height termination.
- Stop config-only tuning and move to R5B height-band control / vertical-correction diagnostics.
- Stop reward-only safety tuning before another sweep: safety-margin and combined variants kept ICS more than 2x above `0.005` (`0.0358125` to `0.13725`) and introduced or retained terminations.

## Result Recording

Rows below are one per executed sweep job.

| Date | Sweep artifact | Variant | Seed | Gates | Score | Passed | Failed gates | Decision |
|---|---|---|---:|---:|---:|---|---|---|
| 2026-07-12 | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260712_223240/summary.json` | `r5_null_speed4` | 0 | 9/14 | 8.337 | false | `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_command_amplification_mean`, `tracking_command_amplification_rate`, `termination_above_bound` | Best failed R5A; not promotable; no micro-sweep because 9/14 and height termination |
| 2026-07-12 | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260712_223240/summary.json` | `r5_safety_margin` | 0 | 5/14 | 4.047 | false | `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_command_amplification_mean`, `tracking_command_amplification_rate`, `safety_collision_rate`, `ics_violation_rate`, `termination_collision`, `termination_below_bound`, `termination_above_bound` | Reject; safety margins did not fix ICS and introduced terminations |
| 2026-07-12 | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260712_223240/summary.json` | `r5_null_amp_height` | 0 | 6/14 | 3.539 | false | `station_drift_mean`, `station_drift_p95`, `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_preservation_ratio`, `safety_min_clearance_p05`, `ics_violation_rate`, `termination_below_bound` | Reject; combined station/amp/height worsened height and ICS |
| 2026-07-12 | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260712_223240/summary.json` | `r5_amp3` | 0 | 2/14 | 0.678 | false | `station_drift_mean`, `station_drift_p95`, `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_command_amplification_mean`, `tracking_command_amplification_rate`, `safety_collision_rate`, `safety_min_clearance_p05`, `ics_violation_rate`, `termination_collision`, `termination_below_bound`, `termination_above_bound` | Reject; amp-only branch broadly regressed |
| 2026-07-12 | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260712_223240/summary.json` | `r5_height16` | 0 | 2/14 | -0.520 | false | `station_drift_mean`, `station_drift_p95`, `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_rmse_actual`, `tracking_preservation_ratio`, `safety_collision_rate`, `safety_min_clearance_p05`, `ics_violation_rate`, `termination_collision`, `termination_below_bound`, `termination_above_bound` | Reject; height reward increase alone did not control height |
| 2026-07-12 | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260712_223240/summary.json` | `r5_null_amp_height_safety` | 0 | 1/14 | -2.776 | false | `station_drift_mean`, `station_drift_p95`, `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_rmse_actual`, `tracking_preservation_ratio`, `tracking_command_amplification_mean`, `tracking_command_amplification_rate`, `safety_collision_rate`, `safety_min_clearance_p05`, `ics_violation_rate`, `termination_collision`, `termination_below_bound` | Reject; bundle collapsed height/safety and station performance |

For any candidate with `passed=true`, add a promotion note:

```text
Candidate:
Artifact:
Checkpoint:
Reason for promotion:
Remaining risks:
1M command:
```

For any failed sweep, add a decision note:

```text
Failed sweep:
Best candidate:
Main remaining blockers:
Interpretation:
Next adjustment:
Stop condition:
```

Failed sweep:
`docs/instinctRL_devlog/tests/artifacts/sweeps/20260712_223240/summary.json`

Best candidate:
`r5_null_speed4`, seed `0`, checkpoint `/home/mint/rl_dev/NavRL/isaac-training/wandb/offline-run-20260712_223245-bspwwx22/files/checkpoint_final.pt`, `9/14`, score `8.337`, `passed=false`.

Main remaining blockers:
`station_null_speed_mean=0.15179473582375794`, `station_anchor_error_mean=2.581868985198438`, `tracking_command_amplification_mean=0.08557469345256687`, `tracking_command_amplification_rate=0.2273125`, and `termination_above_bound=0.1875`.

Interpretation:
`r5_null_speed4` cleared collision, clearance, ICS, termination collision, and termination below-bound, but it still failed station/amp gates and above-bound height. The safety-margin and combined variants did not provide a useful safety path: ICS remained more than 2x above the `0.005` gate and height/collision terminations persisted.

Next adjustment:
Do not promote, do not run 1M, and do not run another config-only R5A sweep. Start R5B code work on height-band control, vertical-correction diagnostics, and ICS/effective-clearance diagnostics before another 128k sweep.

Stop condition:
Hard gates unchanged; `training/scripts/instinctRL/gates.py` unchanged; no 1M/formal/warm-start run executed.

## A2-R5B Height-Band / Vertical-Correction Diagnostics - 2026-07-13

R5B entry decision:

- R5A config-only tuning stopped because the best executed candidate, `r5_null_speed4`, reached only `9/14` gates and still had `termination_above_bound=0.1875`.
- The R5A result is not promotable and does not qualify for a config-only micro-sweep because it is below `10/14` and still has a height termination.
- R5A reward-only safety tuning is also stopped before another sweep because safety-margin variants kept ICS more than 2x above the `0.005` hard gate.

Height interpretation:

- Existing reward logic had `height_floor` and `height_floor_weight`, which only penalized low altitude.
- Above-bound termination is currently triggered at world/root `z > 4.0`, but there was no explicit reward-side ceiling or height-band diagnostic to explain proximity to that termination.
- R5B therefore starts with diagnostics and dormant reward config, not another sweep.

Implemented R5B diagnostics/code readiness:

- Added eval/logging-only height diagnostics for `height_world_z`, `height_floor_violation`, `height_ceiling_violation`, and `height_ceiling_margin`.
- Added eval/logging-only vertical signal diagnostics for `v_cmd_z`, `v_final_b_z`, `governor_v_corr_z`, `governor_v_cmd_b_z`, `governor_v_gov_b_z`, and `governor_v_final_b_z`.
- Split command amplification into horizontal and vertical diagnostics while keeping the original norm-level amplification gates unchanged.
- Added dormant reward-only height-band config draft: `height_ceiling=4.0`, `height_ceiling_weight=0.0`, and `reward_height_ceiling`.
- Confirmed these fields are `info`, stats, eval, or top-level diagnostic tensors only; actor observation remains `lidar_grid + state_vec`.

Validation results:

- `python -m py_compile training/scripts/env.py training/scripts/train.py training/scripts/eval.py training/scripts/ppo.py training/scripts/utils.py training/scripts/instinctRL/rewards.py training/scripts/instinctRL/task_metrics.py training/scripts/instinctRL/governor.py`: passed.
- `python -m pytest -q training/unit_test/test_instinctrl_rewards.py training/unit_test/test_instinctrl_task_metrics.py training/unit_test/test_instinctrl_eval_diagnostic.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py`: passed, `39 passed, 4 warnings in 2.09s`.
- `python -m pytest -q training/unit_test/test_instinctrl_*.py`: passed, `105 passed, 11 warnings in 3.17s`.

Next adjustment:

- R5B sweep variants may be designed only after the diagnostics patch passes the full `training/unit_test/test_instinctrl_*.py` set and the first R5B diagnostic eval artifact is inspected.
- Still do not run 1M, do not run a new sweep, do not warm-start a failed checkpoint, and do not modify hard gates.

## A2-R5B 128k Sweep Plan - 2026-07-13

R5B sweep-only scope:

- Use the existing R5B height-band reward and vertical diagnostics. Do not modify actor observation, learned action semantics, or hard gates.
- Start from the best R5A direction, `r5_null_speed4`, with `algo.instinctRL.governor.v_corr_limit=0.35`, `instinctRL.reward.null_command_speed_weight=4.0`, `instinctRL.reward.height_floor=0.5`, `instinctRL.reward.height_floor_weight=8.0`, and `instinctRL.reward.height_ceiling=4.0`.
- Retain the R5A common base: anchor 4.0, null output 0.1, amp 2.0, proxy tracking 0.5, safety 1.2, clearance margin 0.4, ICS active horizon margin 1.0, and ICS clearance margin 0.15.
- Do not add `anchor_weight=5.0` and do not reintroduce the failed R5A safety-margin bundle.
- This pass is dry-run only. Do not pass `--execute`, do not run 1M, and do not warm-start any checkpoint.

Default sweep tag: `a2r5b_sweep`.

| Variant | Additional overrides | Hypothesis |
|---|---|---|
| `r5b_ceiling4` | `instinctRL.reward.height_ceiling_weight=4.0` | A light ceiling penalty may remove above-bound termination without destabilizing the R5A best station/safety direction. |
| `r5b_ceiling8` | `instinctRL.reward.height_ceiling_weight=8.0` | A ceiling penalty matching the floor weight may enforce the height band more reliably than the light ceiling branch. |
| `r5b_band_floor12_ceiling8` | `instinctRL.reward.height_floor_weight=12.0`, `instinctRL.reward.height_ceiling_weight=8.0` | A stronger lower band plus ceiling may reduce both below-bound and above-bound failures. |
| `r5b_ceiling8_amp25` | `instinctRL.reward.height_ceiling_weight=8.0`, `instinctRL.reward.command_amplification_weight=2.5` | Moderate amp pressure may reduce the remaining amplification misses without the R5A amp3 collapse. |
| `r5b_ceiling8_vcorr030` | `instinctRL.reward.height_ceiling_weight=8.0`, `algo.instinctRL.governor.v_corr_limit=0.30` | A slightly smaller correction envelope may reduce vertical/height overshoot while preserving the R5A `v_corr_limit=0.35` direction as a reference. |
| `r5b_band_amp25_vcorr030` | `instinctRL.reward.height_floor_weight=12.0`, `instinctRL.reward.height_ceiling_weight=8.0`, `instinctRL.reward.command_amplification_weight=2.5`, `algo.instinctRL.governor.v_corr_limit=0.30` | Combined height band, moderate amp, and lower correction envelope may be the best balanced branch if single levers move in the right direction. |

## R5B Sweep Dry-Run Backfill - 2026-07-13

Code changes:

- Updated `training/scripts/instinctRL/sweep.py` default `--tag` from `a2r5_sweep` to `a2r5b_sweep`.
- Replaced the default R5A variant list with exactly the six R5B variants in the plan above.
- Kept dry-run as the default behavior; `--execute` remains explicit and was not run.
- Kept `training/scripts/instinctRL/gates.py` unchanged.
- Updated `training/unit_test/test_instinctrl_gates.py` to assert the R5B variant names/order, R5B base overrides, effective ceiling/floor/amp/vcorr overrides, default tag, short diagnostic eval suite, absence of `anchor_weight=5.0`, and absence of the failed R5A safety-margin bundle.

Validation results:

- `python -m py_compile training/scripts/instinctRL/sweep.py`: passed.
- `python -m pytest -q training/unit_test/test_instinctrl_gates.py`: passed, `5 passed in 1.11s`.
- `python -m pytest -q training/unit_test/test_instinctrl_*.py`: passed, `105 passed, 11 warnings in 3.79s`.

Dry-run review:

- Command run: `python training/scripts/instinctRL/sweep.py --frames 131072 --seeds 0 --limit 6`.
- Result: passed as dry-run only with `execute=false`, `frames=131072`, `seed=0`, and six generated jobs.
- Generated artifact paths pointed under `docs/instinctRL_devlog/tests/artifacts/sweeps/20260713_115654/`, but that directory was absent after the run because `--execute` was not used.
- Generated variants, in order: `r5b_ceiling4`, `r5b_ceiling8`, `r5b_band_floor12_ceiling8`, `r5b_ceiling8_amp25`, `r5b_ceiling8_vcorr030`, `r5b_band_amp25_vcorr030`.
- All run names used the `instinctrl_a2r5b_sweep_*_131072_seed0` prefix.
- Eval commands retained `instinctRL.eval.suite=short_diagnostic`.

Decision:

- R5B dry-run passed and is ready for explicit execution review.
- No R5B sweep was executed, no 1M/formal run was executed, no warm-start checkpoint was used, and no hard gates were modified.
- 1M remains forbidden until a 128k R5B candidate passes all hard gates.

## R5B Execution Backfill - 2026-07-13

Execution:

- Preflight found no complete or incomplete prior `a2r5b_sweep` execution summary.
- Command run exactly once: `python training/scripts/instinctRL/sweep.py --execute --frames 131072 --seeds 0 --limit 6`.
- Artifact: `docs/instinctRL_devlog/tests/artifacts/sweeps/20260713_125240/summary.json`.
- All six jobs completed with `error=null`, non-null checkpoints, eval artifacts, and embedded `gate_report`.
- Gate truth source: embedded `gate_report` produced by `training/scripts/instinctRL/gates.py`.
- Hard gates were unchanged; `training/scripts/instinctRL/gates.py` was not modified.
- No 1M, formal long training, warm-start, code change, parameter change, hard-gate change, or extra sweep command was executed.

Execution decision:

- No candidate passed all 14 hard gates.
- Best candidate by hard-gate ranking was `r5b_ceiling8_amp25`, seed `0`, with `8/14` gates and score `7.301`.
- `r5b_ceiling8_amp25` is not promotable because it failed station drift mean/p95, station null speed, station anchor error, tracking preservation ratio, and ICS violation rate.
- R5B micro-sweep is not allowed under the default R5B branch rule: best candidate is below `10/14`, even though it has zero terminations.
- Stop reward/config tuning. Next work should diagnose mechanism-level control: vertical correction/action constraints, whether height control belongs in a governor-side safety clamp, and horizontal/vertical amplification governance.

Rows below are ranked as written in `summary.json`.

| Date | Sweep artifact | Variant | Seed | Gates | Score | Passed | Failed gates | Decision |
|---|---|---|---:|---:|---:|---|---|---|
| 2026-07-13 | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260713_125240/summary.json` | `r5b_ceiling8_amp25` | 0 | 8/14 | 7.301 | false | `station_drift_mean`, `station_drift_p95`, `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_preservation_ratio`, `ics_violation_rate` | Best failed R5B; zero terminations but below 10/14, so no micro-sweep |
| 2026-07-13 | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260713_125240/summary.json` | `r5b_ceiling4` | 0 | 6/14 | 5.125 | false | `station_drift_mean`, `station_drift_p95`, `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_command_amplification_mean`, `tracking_command_amplification_rate`, `ics_violation_rate`, `termination_above_bound` | Reject; ceiling 4 did not control station/amp and still had above-bound termination |
| 2026-07-13 | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260713_125240/summary.json` | `r5b_ceiling8_vcorr030` | 0 | 5/14 | 4.191 | false | `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_command_amplification_mean`, `tracking_command_amplification_rate`, `safety_collision_rate`, `safety_min_clearance_p05`, `ics_violation_rate`, `termination_collision`, `termination_above_bound` | Reject; lower correction limit did not fix safety/ICS and retained terminations |
| 2026-07-13 | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260713_125240/summary.json` | `r5b_band_amp25_vcorr030` | 0 | 4/14 | 2.408 | false | `station_drift_mean`, `station_drift_p95`, `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_preservation_ratio`, `safety_collision_rate`, `safety_min_clearance_p05`, `ics_violation_rate`, `termination_collision`, `termination_below_bound` | Reject; combined lower band/amp/lower correction regressed station, safety, and below-bound |
| 2026-07-13 | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260713_125240/summary.json` | `r5b_band_floor12_ceiling8` | 0 | 4/14 | 1.671 | false | `station_drift_mean`, `station_drift_p95`, `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_preservation_ratio`, `safety_collision_rate`, `safety_min_clearance_p05`, `ics_violation_rate`, `termination_collision`, `termination_below_bound` | Reject; stronger floor shifted failure toward low altitude/clearance/ICS |
| 2026-07-13 | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260713_125240/summary.json` | `r5b_ceiling8` | 0 | 2/14 | -0.132 | false | `station_drift_mean`, `station_drift_p95`, `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_rmse_actual`, `tracking_command_amplification_mean`, `tracking_command_amplification_rate`, `safety_collision_rate`, `safety_min_clearance_p05`, `ics_violation_rate`, `termination_collision`, `termination_above_bound` | Reject; ceiling 8 alone broadly regressed and had above-bound/collision terminations |

Key diagnostics:

| Variant | Station drift mean/p95 | Null speed | Anchor error | Tracking RMSE | Preservation | Amp mean/rate | H amp mean/rate | V amp mean/rate | Height z p05/p95 | Floor max | Ceiling margin min | Clearance p05 | ICS | Term below/above/collision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `r5b_ceiling8_amp25` | 1.357 / 2.629 | 0.172 | 2.861 | 0.417 | 0.631 | 0.024 / 0.038 | 0.019 / 0.035 | 0.191 / 0.311 | 1.289 / 3.087 | 0.000 | 0.283 | 1.438 | 0.00759 | 0 / 0 / 0 |
| `r5b_ceiling4` | 1.362 / 2.646 | 0.173 | 2.799 | 0.351 | 0.940 | 0.105 / 0.280 | 0.103 / 0.257 | 0.299 / 0.433 | 1.125 / 3.496 | 0.217 | 0.000276 | 1.161 | 0.01291 | 0 / 0.125 / 0 |
| `r5b_ceiling8_vcorr030` | 1.299 / 2.522 | 0.165 | 2.781 | 0.374 | 0.821 | 0.052 / 0.175 | 0.065 / 0.199 | 0.195 / 0.318 | 1.044 / 3.333 | 0.197 | 0.000563 | 0.846 | 0.03313 | 0 / 0.03125 / 0.09375 |
| `r5b_band_amp25_vcorr030` | 1.586 / 3.110 | 0.204 | 3.136 | 0.444 | 0.668 | 0.037 / 0.071 | 0.050 / 0.100 | 0.141 / 0.237 | 0.793 / 2.771 | 0.300 | 0.772 | 0.800 | 0.05009 | 0.125 / 0 / 0.03125 |
| `r5b_band_floor12_ceiling8` | 1.590 / 3.095 | 0.202 | 3.324 | 0.396 | 0.641 | 0.019 / 0.025 | 0.042 / 0.073 | 0.043 / 0.077 | 0.360 / 2.149 | 0.300 | 1.527 | 0.592 | 0.08784 | 0.5 / 0 / 0.03125 |
| `r5b_ceiling8` | 1.547 / 3.009 | 0.197 | 2.825 | 0.466 | 0.937 | 0.158 / 0.372 | 0.063 / 0.131 | 0.434 / 0.562 | 1.964 / 3.820 | 0.000 | 0.000016 | 0.739 | 0.06059 | 0 / 0.46875 / 0.125 |

Interpretation:

`r5b_ceiling8_amp25` is the only R5B branch that removed height terminations, collision termination, and hard amp failures while keeping clearance above the hard gate. It still failed all station gates, preservation low, and ICS. Its vertical amplification diagnostics remained high (`vertical_mean=0.1906`, `vertical_rate=0.3109`) even though norm-level amplification gates passed. This supports a mechanism diagnosis rather than another reward-weight sweep.

Stop condition:

Hard gates unchanged; `training/scripts/instinctRL/gates.py` unchanged; no actor observation, learned action method, platform/sensor, or body-frame velocity-governor method was modified; no 1M/formal/warm-start run executed.

## A2-R5C Mechanism Diagnosis Plan - 2026-07-13

R5C entry evidence:

- R5B best candidate was `r5b_ceiling8_amp25`, seed `0`, checkpoint `/home/mint/rl_dev/NavRL/isaac-training/wandb/offline-run-20260713_133443-ym5ojvas/files/checkpoint_final.pt`.
- It reached only `8/14` hard gates and is not promotable. Failed gates were station drift mean/p95, station null speed, station anchor error, tracking preservation ratio, and ICS violation rate.
- R5B station/null/anchor remained weak: station drift `1.357 / 2.629`, null speed `0.172`, anchor error `2.861`.
- R5B preservation and ICS remained weak: preservation `0.631`, ICS `0.00759`.
- R5B vertical amplification remained high even though norm-level amp gates passed: vertical amp mean/rate `0.191 / 0.311`.

R5C decision:

- Stop reward/config tuning for this pass.
- Do not run another sweep, do not run 1M/formal training, do not promote, do not warm-start, and do not modify hard gates.
- Do not alter `PPO.decode_action`, governor action semantics, actor observation, platform/sensor, or the body-frame velocity-governor method.
- Implement diagnostics only so the future mechanism fix can be selected from measured vertical-channel behavior.

Implemented R5C diagnostics:

- Added pure torch `compute_vertical_channel_step_metrics` in `training/scripts/instinctRL/task_metrics.py`.
- Added streaming eval aggregation for aligned `governor_v_cmd_b_z`, `governor_v_corr_z`, `governor_v_gov_b_z`, `governor_v_final_b_z`, station drift, preservation, vertical amplification, `ics_beta`, and `ics_emergency`.
- Emitted `eval/handbook.vertical_*` diagnostics for correction sign/magnitude/saturation, governor/final/ICS deltas, null-command correction, and tracking-command correction conditionals.
- Kept all new metrics in eval/logging summaries only. `training/scripts/instinctRL/gates.py` was not modified.
- Confirmed actor observation remains exactly `lidar_grid + state_vec`; vertical, height/root state, velocity, map, SLAM, and privileged simulator state stay out of actor input.

Future mechanism candidates, classified:

- Governor-side vertical correction clamp: governor-action constraint; actor-clean; behavior-changing if enabled.
- Asymmetric z correction limit: governor-action constraint; actor-clean; behavior-changing if enabled.
- Height safety clamp after governor before body-to-world command: safety filter; actor-clean if kept outside actor observation; behavior-changing if enabled.
- Horizontal/vertical amplification split: diagnostic/reward-only if only logged or rewarded; governor-action constraint if it clips commands; behavior-changing if enabled.

Validation results:

- `python -m py_compile training/scripts/instinctRL/task_metrics.py training/scripts/utils.py training/scripts/eval.py training/scripts/ppo.py training/scripts/train.py training/scripts/env.py`: passed.
- `python -m pytest -q training/unit_test/test_instinctrl_task_metrics.py training/unit_test/test_instinctrl_eval_diagnostic.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py`: passed, `23 passed, 4 warnings in 1.86s`.
- `python -m pytest -q training/unit_test/test_instinctrl_*.py`: passed, `107 passed, 11 warnings in 2.07s`.

Diagnostic eval:

- Command run exactly once, eval-only: `python training/scripts/eval.py checkpoint_path=/home/mint/rl_dev/NavRL/isaac-training/wandb/offline-run-20260713_133443-ym5ojvas/files/checkpoint_final.pt result_path=../docs/instinctRL_devlog/tests/artifacts/r5c_diagnostics/20260713_r5b_ceiling8_amp25_vertical_eval.json env.num_envs=32 env.max_episode_length=1000 env.num_obstacles=350 env_dyn.num_obstacles=0 instinctRL.eval.suite=short_diagnostic instinctRL.observability.enabled=true instinctRL.observability.mode=proxy wandb.mode=offline headless=true`.
- Artifact: `docs/instinctRL_devlog/tests/artifacts/r5c_diagnostics/20260713_r5b_ceiling8_amp25_vertical_eval.json`.
- Saturation is measured against the eval-time governor limit used by the replayed eval config: `eval/handbook.vertical_v_corr_limit=0.5`.

Key R5C vertical readings:

| Scope | corr z mean | corr z abs | positive / negative | saturation | gov-cmd abs | final-cmd abs | ICS delta abs | null active | null abs | null drift when active | tracking active | tracking amp when active | tracking preservation when active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Top-level diagnostic mix | 0.1165 | 0.1165 | 1.000 / 0.000 | 0.000 | 0.1582 | 0.1599 | 0.00171 | 1.000 | 0.0343 | 1.3566 | 1.000 | 0.2264 | 0.5641 |
| Station pass | 0.0343 | 0.0343 | 1.000 / 0.000 | 0.000 | 0.0343 | 0.0343 | 0.00000 | 1.000 | 0.0343 | 1.3566 | n/a | n/a | n/a |
| Tracking pass | 0.1165 | 0.1165 | 1.000 / 0.000 | 0.000 | 0.1582 | 0.1599 | 0.00171 | 1.000 | 0.0486 | 1.4250 | 1.000 | 0.2264 | 0.5641 |

Interpretation:

- Vertical correction is always positive in this replay and never saturates against `v_corr_limit=0.5`, so the immediate failure is not a hard correction-limit saturation.
- The station pass applies a small persistent upward null-command correction (`0.0343`) on every null step while station drift remains above gate (`1.3566` mean when correction is active).
- The tracking pass applies vertical correction on every vertical-active step, with high vertical amplification when active (`0.2264`) and low preservation when active (`0.5641`).
- ICS changes the vertical channel only slightly on average (`vertical_ics_delta_z_abs_mean=0.00171`), so the R5B vertical preservation/amplification issue appears mostly pre-ICS in this replay.

Stop condition:

R5C implemented diagnostics only. No mechanism candidate was enabled, no R5C dry-run or sweep was run, no 1M/formal/warm-start run was executed, and hard gates remain unchanged.

## A2-R5D Mechanism-Fix Readiness - 2026-07-13

R5D scope:

- Implement only default-off/default-equivalent governor-side vertical mechanisms.
- Do not run training, sweep, 1M, promotion, hard-gate changes, warm-start, reward changes, or actor-observation changes.
- Primary mechanism hooks are separate vertical correction authority and actor-clean tracking z-correction attenuation using `abs(v_cmd_b[..., 2]) > eps`.
- Height safety clamp and horizontal/vertical amplification split governance are documented/classified only in this pass.

Code changes:

- Added `TrainableGovernorDecoder(v_corr_z_limit=None)`. `None` inherits `v_corr_limit`, so existing configs and old overrides preserve current behavior.
- Added default-off tracking z-correction attenuation: `tracking_vcorr_z_gate_enabled=false`, `tracking_vcorr_z_gate_eps=1e-3`, and `tracking_vcorr_z_gain=1.0`.
- Kept learned governor action schema at 4D `[alpha, v_corr_x, v_corr_y, v_corr_z]`.
- Kept `PPO.decode_action` semantics and output tensor names unchanged.
- Wired the new config through PPO construction and `training/cfg/ppo.yaml`.
- Did not implement a height safety clamp or any amplification clipping/gating code.

Candidate classification:

- Separate `v_corr_z_limit`: actor-clean; governor-action constraint; default unchanged; deployable under the handbook.
- Tracking z-correction attenuation: actor-clean using only `v_cmd_b[..., 2]`; governor-action constraint; default unchanged; deployable under the handbook.
- Governor-side height safety clamp: actor-clean only if outside actor observation; safety filter; default-off if ever implemented; not deployable under the handbook when using privileged/root height.
- Horizontal/vertical amplification split governance: reward-only/diagnostic if logged or rewarded; governor-action constraint if clipping commands; default unchanged in R5D; deployability depends on no privileged inputs.

Validation results:

- `python -m py_compile training/scripts/instinctRL/governor.py training/scripts/ppo.py training/scripts/train.py training/scripts/eval.py training/scripts/env.py`: passed.
- `python -m pytest -q training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_actor_audit.py`: passed, `24 passed, 5 warnings in 1.83s`.
- `python -m pytest -q training/unit_test/test_instinctrl_*.py`: passed, `114 passed, 12 warnings in 3.69s`.

Readiness decision:

R5D is ready for mechanism-screening design. No training, sweep, 1M, promotion, hard-gate change, warm-start, reward change, actor-observation change, or behavior-changing default was executed.

## A2-R5D 128k Sweep Plan - 2026-07-13

R5D sweep-only scope:

- Use `r5b_ceiling8_amp25` as the fixed base because it had zero terminations and passed norm-level amplification gates.
- Vary only `algo.instinctRL.governor.v_corr_z_limit`, `algo.instinctRL.governor.tracking_vcorr_z_gate_enabled`, `algo.instinctRL.governor.tracking_vcorr_z_gate_eps`, and `algo.instinctRL.governor.tracking_vcorr_z_gain`.
- Keep `algo.instinctRL.governor.v_corr_limit=0.35`, `instinctRL.reward.height_ceiling_weight=8.0`, and `instinctRL.reward.command_amplification_weight=2.5`.
- Retain the R5B best station/safety/height/ICS base: anchor 4.0, null output 0.1, preservation high 2.0, proxy tracking 0.5, safety 1.2, clearance margin 0.4, ICS active horizon margin 1.0, ICS clearance margin 0.15, null speed 4.0, height floor 0.5, height floor weight 8.0, and height ceiling 4.0.
- Do not change hard gates, actor observation, platform/sensor, rewards beyond the fixed R5B best base, height clamp behavior, or the velocity-governor/body-frame command method.
- This pass is dry-run only. Do not pass `--execute`, do not run 1M, and do not warm-start any checkpoint.

Default sweep tag: `a2r5d_sweep`.

Implementation note:

- Keep `default_safety_preservation_variants()` as the sweep entry point.
- Propagate variant overrides to both train and eval commands so eval replays the same R5D governor settings.

| Variant | Additional overrides | Hypothesis |
|---|---|---|
| `r5d_zlimit020` | `algo.instinctRL.governor.v_corr_z_limit=0.20` | A smaller vertical correction envelope may reduce vertical amplification while preserving the R5B best horizontal authority. |
| `r5d_zlimit012` | `algo.instinctRL.governor.v_corr_z_limit=0.12` | A tighter vertical envelope tests whether the persistent upward z correction is over-authorized. |
| `r5d_trackzgain050` | `algo.instinctRL.governor.tracking_vcorr_z_gate_enabled=true`, `algo.instinctRL.governor.tracking_vcorr_z_gate_eps=0.001`, `algo.instinctRL.governor.tracking_vcorr_z_gain=0.50` | Half-strength z correction only on vertical tracking commands may reduce tracking vertical amplification without changing station null-command correction. |
| `r5d_trackzgain000` | `algo.instinctRL.governor.tracking_vcorr_z_gate_enabled=true`, `algo.instinctRL.governor.tracking_vcorr_z_gate_eps=0.001`, `algo.instinctRL.governor.tracking_vcorr_z_gain=0.0` | Fully removing z correction during vertical tracking tests whether tracking z correction is the main preservation/ICS driver. |
| `r5d_zlimit020_trackzgain050` | `algo.instinctRL.governor.v_corr_z_limit=0.20`, `algo.instinctRL.governor.tracking_vcorr_z_gate_enabled=true`, `algo.instinctRL.governor.tracking_vcorr_z_gate_eps=0.001`, `algo.instinctRL.governor.tracking_vcorr_z_gain=0.50` | Combined moderate z limit and half tracking attenuation tests the least disruptive mechanism fix. |
| `r5d_zlimit012_trackzgain000` | `algo.instinctRL.governor.v_corr_z_limit=0.12`, `algo.instinctRL.governor.tracking_vcorr_z_gate_enabled=true`, `algo.instinctRL.governor.tracking_vcorr_z_gate_eps=0.001`, `algo.instinctRL.governor.tracking_vcorr_z_gain=0.0` | Combined tight z limit and no tracking z correction is the aggressive mechanism screen. |

## R5D Sweep Dry-Run Backfill - 2026-07-13

Code changes:

- Updated `training/scripts/instinctRL/sweep.py` default `--tag` from `a2r5b_sweep` to `a2r5d_sweep`.
- Replaced the default R5B variant list with exactly the six R5D variants in the plan above.
- Kept `default_safety_preservation_variants()` as the sweep entry point.
- Added `eval_overrides` to `SweepJob`, added `extra_overrides=()` to `build_eval_command()`, propagated variant overrides into eval commands during dry-run planning, and preserved them when rebuilding eval commands after checkpoint discovery.
- Kept dry-run as the default behavior; `--execute` remains explicit and was not run.
- Kept `training/scripts/instinctRL/gates.py` unchanged.
- Updated `training/unit_test/test_instinctrl_gates.py` to assert the R5D variant names/order, default tag, `instinctrl_a2r5d_sweep_*` run-name prefix, R5D base overrides, train/eval override propagation, and absence of forbidden R5A/R5B/height-clamp overrides.

Validation results:

- `python -m py_compile training/scripts/instinctRL/sweep.py`: passed.
- `python -m pytest -q training/unit_test/test_instinctrl_gates.py`: passed, `5 passed in 1.11s`.
- `python -m pytest -q training/unit_test/test_instinctrl_*.py`: passed, `114 passed, 12 warnings in 5.41s`.

Dry-run review:

- Command run: `python training/scripts/instinctRL/sweep.py --frames 131072 --seeds 0 --limit 6`.
- Result: passed as dry-run only with `execute=false`, `frames=131072`, `seed=0`, and six generated jobs.
- Generated artifact paths pointed under `docs/instinctRL_devlog/tests/artifacts/sweeps/20260713_201326/`, but that directory was absent after the run because `--execute` was not used.
- Dry-run printed planned jobs only and did not create an execution `summary.json`.
- Generated variants, in order: `r5d_zlimit020`, `r5d_zlimit012`, `r5d_trackzgain050`, `r5d_trackzgain000`, `r5d_zlimit020_trackzgain050`, `r5d_zlimit012_trackzgain000`.
- All run names used the `instinctrl_a2r5d_sweep_*_131072_seed0` prefix.
- Train and eval commands both included the R5D base overrides and each variant's z-limit/tracking-z overrides.

Stop condition:

- No `--execute`, no 1M, no warm-start, no hard-gate change, and no actor-observation change.
- No platform/sensor, reward, height-clamp, or velocity-governor/body-frame command method change was made in this sweep dry-run pass.
- Next step after this clean dry-run is a controlled 128k R5D execute sweep.
- Only a `14/14` candidate with `passed=true` can be considered for a 1M confirmation.

## R5D Execution Backfill - 2026-07-13

Execution:

- Preflight found a clean worktree at baseline commit `f1f9fef01495c5c7952600f7653023823768767a`.
- Re-read the handbook, decision log, R5 ticket, `training/scripts/instinctRL/gates.py`, and `training/scripts/instinctRL/sweep.py` before execution.
- Verified `training/scripts/instinctRL/sweep.py` default tag was `a2r5d_sweep`.
- Verified the R5D variant set was exactly `r5d_zlimit020`, `r5d_zlimit012`, `r5d_trackzgain050`, `r5d_trackzgain000`, `r5d_zlimit020_trackzgain050`, `r5d_zlimit012_trackzgain000`.
- Command run exactly once: `python training/scripts/instinctRL/sweep.py --execute --frames 131072 --seeds 0 --limit 6`.
- Artifact: `docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_000507/summary.json`.
- The artifact timestamp is `20260714_000507` because this controlled execution ran in local Asia/Shanghai time after midnight.
- All six jobs completed with `error=null`, non-null checkpoints, eval JSON artifacts, and embedded `gate_report`.
- Gate truth source: only embedded `job.gate_report` in the new `summary.json`, produced by `training/scripts/instinctRL/gates.py`.
- Hard gates were unchanged; `training/scripts/instinctRL/gates.py` was not modified.
- No 1M, formal long training, warm-start, code change, parameter change, hard-gate change, actor-observation change, platform/sensor change, reward change, height-clamp change, or velocity-governor/body-frame command method change was executed.

Execution decision:

- No candidate passed all 14 hard gates.
- Best candidate by embedded hard-gate ranking was `r5d_zlimit012`, seed `0`, with `8/14` gates and score `6.841`.
- `r5d_zlimit012` is not promotable because it failed station null speed, station anchor error, tracking preservation, clearance, ICS, and below-bound termination.
- No 1M confirmation command is written because there is no `passed=true`, `14/14` candidate.
- R5D micro-sweep is not allowed: the best candidate is below `10/14`, has `termination_below_bound=0.3125`, fails the clearance gate, and has `safety_passed=false`.
- Stop R5D sweeping and continue mechanism diagnosis. The remaining blockers are station/null-anchor behavior, low preservation, clearance/ICS, and below-bound height/safety behavior.

Rows below are ranked as written in `summary.json`.

| Variant | Seed | Gates | Score | Passed | Safety passed | Failed gates |
|---|---:|---:|---:|---|---|---|
| `r5d_zlimit012` | 0 | 8/14 | 6.841 | false | false | `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_preservation_ratio`, `safety_min_clearance_p05`, `ics_violation_rate`, `termination_below_bound` |
| `r5d_trackzgain050` | 0 | 6/14 | 4.847 | false | false | `station_drift_mean`, `station_drift_p95`, `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_preservation_ratio`, `safety_min_clearance_p05`, `ics_violation_rate`, `termination_below_bound` |
| `r5d_trackzgain000` | 0 | 6/14 | 4.604 | false | false | `station_drift_mean`, `station_drift_p95`, `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_preservation_ratio`, `safety_min_clearance_p05`, `ics_violation_rate`, `termination_below_bound` |
| `r5d_zlimit012_trackzgain000` | 0 | 6/14 | 4.357 | false | false | `station_drift_mean`, `station_drift_p95`, `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_preservation_ratio`, `safety_min_clearance_p05`, `ics_violation_rate`, `termination_below_bound` |
| `r5d_zlimit020_trackzgain050` | 0 | 4/14 | 2.069 | false | false | `station_drift_mean`, `station_drift_p95`, `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_preservation_ratio`, `safety_collision_rate`, `safety_min_clearance_p05`, `ics_violation_rate`, `termination_collision`, `termination_below_bound` |
| `r5d_zlimit020` | 0 | 4/14 | 1.161 | false | false | `station_drift_mean`, `station_drift_p95`, `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_preservation_ratio`, `safety_collision_rate`, `safety_min_clearance_p05`, `ics_violation_rate`, `termination_collision`, `termination_below_bound` |

Key station, tracking, and safety diagnostics:

| Variant | Station drift mean/p95 | Null speed | Anchor error | Tracking RMSE | Preservation | Amp mean/rate | H amp mean/rate | V amp mean/rate | Clearance p05 | Collision | ICS | Term below/above/collision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `r5d_zlimit012` | 1.288 / 2.506 | 0.164 | 2.925 | 0.378 | 0.701 | 0.039 / 0.119 | 0.084 / 0.218 | 0.056 / 0.089 | 0.792 | 0.000 | 0.049 | 0.312 / 0.000 / 0.000 |
| `r5d_trackzgain050` | 1.353 / 2.631 | 0.172 | 2.882 | 0.403 | 0.628 | 0.022 / 0.035 | 0.035 / 0.091 | 0.075 / 0.127 | 0.782 | 0.000 | 0.046 | 0.188 / 0.000 / 0.000 |
| `r5d_trackzgain000` | 1.432 / 2.779 | 0.182 | 3.116 | 0.381 | 0.683 | 0.024 / 0.062 | 0.056 / 0.163 | 0.003 / 0.005 | 0.820 | 0.000 | 0.043 | 0.281 / 0.000 / 0.000 |
| `r5d_zlimit012_trackzgain000` | 1.516 / 2.951 | 0.193 | 3.226 | 0.384 | 0.684 | 0.032 / 0.052 | 0.051 / 0.107 | 0.003 / 0.005 | 0.723 | 0.000 | 0.060 | 0.219 / 0.000 / 0.000 |
| `r5d_zlimit020_trackzgain050` | 1.416 / 2.754 | 0.180 | 3.192 | 0.394 | 0.665 | 0.030 / 0.091 | 0.047 / 0.128 | 0.035 / 0.059 | 0.563 | 0.000063 | 0.100 | 0.438 / 0.000 / 0.031 |
| `r5d_zlimit020` | 1.779 / 3.465 | 0.226 | 3.451 | 0.428 | 0.615 | 0.033 / 0.048 | 0.055 / 0.104 | 0.030 / 0.054 | 0.559 | 0.000219 | 0.101 | 0.469 / 0.000 / 0.125 |

Key vertical mechanism diagnostics:

| Variant | Corr z mean | Corr z abs | Saturation | Tracking corr active | Tracking amp when active | Tracking preservation when active | Null corr abs | Null station drift when corr active | ICS delta z abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `r5d_zlimit012` | 0.037 | 0.037 | 0.000 | 1.000 | 0.065 | 0.658 | 0.011 | 1.288 | 0.004 |
| `r5d_trackzgain050` | 0.048 | 0.048 | 0.000 | 1.000 | 0.088 | 0.568 | 0.028 | 1.353 | 0.008 |
| `r5d_trackzgain000` | 0.000438 | 0.001 | 0.000 | 0.000 | n/a | n/a | 0.002 | 1.403 | 0.008 |
| `r5d_zlimit012_trackzgain000` | -0.000307 | 0.000312 | 0.000 | 0.000 | n/a | n/a | 0.001 | 1.521 | 0.006 |
| `r5d_zlimit020_trackzgain050` | -0.012 | 0.012 | 0.000 | 1.000 | 0.041 | 0.603 | 0.009 | 1.416 | 0.010 |
| `r5d_zlimit020` | -0.013 | 0.013 | 0.000 | 1.000 | 0.035 | 0.537 | 0.005 | 1.779 | 0.015 |

Interpretation:

- R5D did reduce vertical amplification for some variants, especially `tracking_vcorr_z_gain=0.0`, but those variants still failed station/null-anchor, preservation, clearance, ICS, and below-bound behavior.
- `r5d_zlimit012` was the only branch that kept station drift mean/p95 within the hard gates, but its null speed, anchor error, low preservation, clearance, ICS, and below-bound failures make it a failed screen.
- The zero-tracking-z variants lowered vertical amp but did not recover preservation enough and did not solve safety/height failures.
- The `v_corr_z_limit=0.20` branches regressed safety/collision and below-bound behavior, so they are not useful follow-up anchors.

Stop condition:

Hard gates unchanged; `training/scripts/instinctRL/gates.py` unchanged; no actor observation, learned action method, platform/sensor, reward, height clamp, or body-frame velocity-governor method was modified; no 1M/formal/warm-start run executed.

## A2-R5E Mechanism Diagnosis Plan - 2026-07-14

Scope:

- R5E is diagnostics code plus documentation backfill only.
- Do not change hard gates, actor observation, PPO actor feature extraction, reward behavior, env action semantics, or governor behavior.
- Do not train, do not sweep, do not run 1M, do not warm-start, and do not promote `r5d_zlimit012`.
- Do not start R5E dry-run sweep variants from current evidence. First run exactly one best-only eval replay for `r5d_zlimit012` with the new diagnostics.

R5D failure decomposition:

- `r5d_zlimit012` was best-ranked but still failed the screen at `8/14`, `passed=false`, `safety_passed=false`.
- `v_corr_z_limit=0.12` made station drift mean/p95 pass, but it did not solve station null actual speed or anchor error.
- Tracking still failed command preservation. R5E must separate pre-ICS preservation, post-ICS preservation, horizontal preservation, vertical preservation, and ICS preservation loss before any mechanism patch is considered.
- Safety still failed clearance and ICS violation, with below-bound termination still present. R5E must isolate near-floor command/gov/final z behavior, near-floor ICS beta, near-floor clearance tail, and whether ICS violations cluster near the floor.
- The likely source split remains unresolved: null station correction, tracking preservation, ICS attenuation, and source-of-z-control near the floor are not yet separable from existing R5D summaries.

R5E eval-only diagnostics to emit:

- `eval/handbook.r5e_null_actual_speed_xy_mean`
- `eval/handbook.r5e_null_actual_speed_z_abs_mean`
- `eval/handbook.r5e_null_output_speed_xy_mean`
- `eval/handbook.r5e_null_output_speed_z_abs_mean`
- `eval/handbook.r5e_command_preservation_pre_ics_ratio`
- `eval/handbook.r5e_command_preservation_post_ics_ratio`
- `eval/handbook.r5e_command_preservation_ics_loss_ratio`
- `eval/handbook.r5e_command_preservation_horizontal_ratio`
- `eval/handbook.r5e_command_preservation_vertical_abs_ratio`
- `eval/handbook.r5e_near_floor_rate`
- `eval/handbook.r5e_near_floor_v_cmd_z_mean`
- `eval/handbook.r5e_near_floor_v_gov_z_mean`
- `eval/handbook.r5e_near_floor_v_final_z_mean`
- `eval/handbook.r5e_near_floor_ics_beta_mean`
- `eval/handbook.r5e_near_floor_clearance_p05`
- `eval/handbook.r5e_ics_violation_near_floor_rate`

Mechanism candidates to document but not enable:

- Station/null actual-speed constraint: allowed as diagnostic or reward-only if it uses privileged actual velocity; not deployable as actor input.
- Actor-clean z lower-bound/floor-safety: deployable only if driven by command, history, or range-derived signals; not by root height.
- Governor-side height floor clamp using root height: privileged safety filter; diagnostic/sim-only; not deployable as the Paper-1 method.
- Horizontal/vertical preservation governance: diagnostic or reward-only first; clipping is behavior-changing and must stay default-off.
- Null station correction vs tracking preservation split: candidate decoder/reward separation only after diagnostics isolate the source; default-off until then.

Replay target:

- Checkpoint: `/home/mint/rl_dev/NavRL/isaac-training/wandb/offline-run-20260714_001959-2fhm2u4z/files/checkpoint_final.pt`.
- Result path: `docs/instinctRL_devlog/tests/artifacts/r5e_diagnostics/20260714_r5d_zlimit012_mechanism_eval.json`.
- Use the same R5D `r5d_zlimit012` eval overrides from `docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_000507/summary.json`.
- This replay is eval-only. It is not a 1M confirmation, not a promotion run, and not evidence to start R5E variants unless it identifies a single default-off or default-equivalent mechanism worth screening.

R5E replay result:

- Command executed: `training/scripts/eval.py` only, with the checkpoint and R5D `r5d_zlimit012` eval overrides above.
- Artifact written: `docs/instinctRL_devlog/tests/artifacts/r5e_diagnostics/20260714_r5d_zlimit012_mechanism_eval.json`.
- The JSON contains all R5E keys at top level, under `passes.station_static_mid360`, and under `passes.tracking_static_mid360`. Station tracking/near-floor conditionals are `null` when their active mask count is zero.
- Top-level station/null split: actual xy `0.1482`, actual |z| `0.0697`, output xy `0.0373`, output |z| `0.0114`.
- Top-level tracking preservation split: pre-ICS `0.6869`, post-ICS `0.6620`, ICS loss `0.0249`, horizontal `0.7525`, vertical |z| `0.5581`.
- Top-level near-floor split: near-floor rate `0.0724`, `v_cmd_z` `-0.1611`, `v_gov_z` `-0.0499`, `v_final_z` `-0.0328`, beta `0.6486`, clearance p05 `0.4143`, near-floor ICS violation rate `0.5838`.
- R5D failed gates remain failed: null speed `0.1638`, anchor error `2.9254`, preservation `0.7015`, clearance p05 `0.7922`, ICS violation `0.0486`, below-bound termination `0.3125`.
- Interpretation: preservation loss is mostly pre-ICS/governor-side rather than ICS-only, while near-floor samples show strong downward commands, partial governor/ICS attenuation, poor clearance tail, and high near-floor ICS violation rate. This supports mechanism diagnosis only; it does not authorize promotion, 1M, a sweep, or a behavior-changing patch.

## A2-R5F Mechanism Design Plan - 2026-07-14

R5F scope:

- R5F is mechanism design, documentation backfill, default-off/default-equivalent code readiness, and unit tests only.
- Do not train, do not sweep, do not run 1M, do not warm-start, do not promote, and do not modify `training/scripts/instinctRL/gates.py`.
- Preserve the Paper-1 method boundary: TASLAB_UAV, Livox MID360, body-frame velocity-governor command method, actor observation `lidar_grid + state_vec`, and learned-governor action `[alpha, v_corr_x, v_corr_y, v_corr_z]`.

R5E evidence decomposition:

- Station/null behavior is not solved by the R5D best branch. R5E top-level null split showed actual XY speed `0.1482`, actual |z| `0.0697`, output XY `0.0373`, and output |z| `0.0114`; station drift and anchor error remained failed.
- Tracking preservation loss is mostly pre-ICS/governor-side. R5E top-level preservation split showed pre-ICS `0.6869`, post-ICS `0.6620`, ICS loss `0.0249`, horizontal `0.7525`, and vertical |z| `0.5581`.
- Near-floor safety is a separate mechanism line. R5E top-level near-floor split showed near-floor rate `0.0724`, `v_cmd_z=-0.1611`, `v_gov_z=-0.0499`, `v_final_z=-0.0328`, beta `0.6486`, clearance p05 `0.4143`, and near-floor ICS violation rate `0.5838`.

R5F mechanism lines:

1. Station/null XY drift plus anchor error:
   - Add a default-off axis-split null correction authority hook in the decoder.
   - Keep the existing scalar null gate as the default behavior when the new hook is off.
   - Use only actor-clean `||v_cmd_b||`; do not read actual velocity, root state, anchor internals, map, SLAM, or privileged simulator state.

2. Tracking preservation, especially vertical preservation:
   - Add a default-off sign-aware vertical correction hook in the decoder.
   - Keep the existing 4D action and current R5D tracking-z gain behavior unchanged unless explicitly enabled.
   - Add reward-only horizontal/vertical preservation terms with zero default weights and ICS-safe command gating.

3. Near-floor clearance and ICS safety:
   - Add a default-off MID360-derived downward attenuator inside ICS.
   - Add a privileged root-height floor filter helper only as a sim/eval-only safety filter at the controller/eval boundary.
   - The root-height filter is not deployable under the handbook and must not be described as Paper-1 actor evidence.

Candidate classification:

| Candidate | Class | Deployable under handbook | Default behavior | Targets | Main risk |
|---|---|---:|---:|---|---|
| Null XY axis-split correction authority | actor-clean governor-action constraint | Yes | unchanged/off | null XY actual drift, anchor error | more null correction may create biased motion |
| Null-mode damping | actor-clean governor-action constraint | Yes | unchanged/off | null output bias | R5E says output is already small; may worsen drift |
| Actual-speed station penalty split | reward-only privileged actual velocity | Reward-only yes, actor no | off/zero | XY drift source | reward conflict with tracking/safety |
| Sign-aware vertical correction gate | actor-clean governor-action constraint | Yes | unchanged/off | vertical preservation, pre-ICS loss | may remove useful stabilization or worsen near-floor descent |
| Axis preservation reward terms | reward-only | Yes | zero weight | horizontal/vertical preservation | can fight safety attenuation if ungated |
| Range-derived downward attenuator | actor-clean ICS/safety filter | Yes | unchanged/off | near-floor clearance, ICS violation | MID360 floor visibility / false positives |
| Root-height floor clamp | privileged safety-filter | No; sim/eval-only | unchanged/off | below-bound, near-floor diagnosis | invalid deployable claim if misused |

Implemented code readiness:

- Governor decoder:
  - Added `null_vcorr_axis_split_enabled=false`, `null_vcorr_xy_gate_min=0.25`, and `null_vcorr_z_gate_min=0.25`.
  - Added `tracking_vcorr_z_sign_gate_enabled=false`, `tracking_vcorr_z_opposing_gain=1.0`, and `tracking_vcorr_z_reinforcing_gain=1.0`.
  - `PPO` passes the new config through without changing actor inputs or action dimension.
- ICS:
  - Added `downward_attenuation_enabled=false`, `downward_ray_min_z=0.25`, and `downward_clearance_margin=0.0`.
  - The enabled path uses only MID360 range, mask, weight, ray direction, and body-frame governor command z.
- Safety filter:
  - Added `training/scripts/instinctRL/safety_filter.py` with `PrivilegedHeightFloorSafetyFilter`.
  - It is wired at the train/eval controller boundary only when `instinctRL.safety_filter.privileged_height_floor_enabled=true`.
  - It reads root height only from controller/eval-boundary info and is marked sim/eval-only.
- Rewards and audits:
  - Added zero-weight `reward_horizontal_preservation` and `reward_vertical_preservation`.
  - Actor audit explicitly rejects `r5f_*` and `safety_filter*` actor-observation keys.

Validation commands for R5F:

```bash
cd /home/mint/rl_dev/NavRL/isaac-training
python -m py_compile training/scripts/instinctRL/governor.py training/scripts/instinctRL/ics.py training/scripts/instinctRL/rewards.py training/scripts/instinctRL/safety_filter.py training/scripts/ppo.py training/scripts/train.py training/scripts/eval.py training/scripts/env.py training/scripts/utils.py
python -m pytest -q training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_ics.py training/unit_test/test_instinctrl_rewards.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_eval_diagnostic.py training/unit_test/test_instinctrl_ppo_hybrid.py
```

Validation backfill:

- `python -m py_compile training/scripts/instinctRL/governor.py training/scripts/instinctRL/ics.py training/scripts/instinctRL/rewards.py training/scripts/instinctRL/safety_filter.py training/scripts/ppo.py training/scripts/train.py training/scripts/eval.py training/scripts/env.py training/scripts/utils.py`: passed.
- Focused pre-doc test: `python -m pytest -q training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_ics.py training/unit_test/test_instinctrl_rewards.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py`: passed, `60 passed, 5 warnings`.
- Final R5F validation: `python -m pytest -q training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_ics.py training/unit_test/test_instinctrl_rewards.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_eval_diagnostic.py training/unit_test/test_instinctrl_ppo_hybrid.py`: passed, `64 passed, 5 warnings`.
- Broader instinctRL suite: `python -m pytest -q training/unit_test/test_instinctrl_*.py`: passed, `125 passed, 12 warnings`.

Next-step decision:

- R5F code readiness is not evidence for promotion or 1M.
- The next round may design R5F dry-run sweep variants only after validation remains clean.
- Any privileged height-filter variant must be labeled sim/eval-only and excluded from deployable Paper-1 claims.

## A2-R5F Dry-Run Sweep Plan - 2026-07-14

R5F dry-run scope:

- Design and validate only the R5F default sweep variants.
- Do not pass `--execute`, do not run 1M, do not warm-start, do not promote, and do not modify `training/scripts/instinctRL/gates.py`.
- Preserve TASLAB_UAV + Livox MID360, actor observation `lidar_grid + state_vec`, learned-governor action `[alpha, v_corr_x, v_corr_y, v_corr_z]`, and the body-frame velocity-governor method.
- Exclude `instinctRL.safety_filter.privileged_height_floor_enabled=true` from the default R5F sweep set. The privileged root-height filter remains deferred sim/eval-only diagnostic tooling, not deployable Paper-1 actor evidence.

Evidence entering R5F dry-run design:

- R5D execution failed the 128k screen. Best candidate `r5d_zlimit012` reached only `8/14`, `passed=false`, and `safety_passed=false`, with station null speed, anchor error, preservation, clearance, ICS, and below-bound failures.
- R5E replay showed station/null XY drift and anchor error remain separate from vertical output bias: actual XY speed `0.1482`, actual |z| `0.0697`, output XY `0.0373`, and output |z| `0.0114`.
- R5E preservation split showed mostly pre-ICS/governor-side preservation loss: pre-ICS `0.6869`, post-ICS `0.6620`, ICS loss `0.0249`, horizontal `0.7525`, and vertical |z| `0.5581`.
- R5E near-floor split showed near-floor safety remains unresolved: near-floor rate `0.0724`, `v_cmd_z=-0.1611`, `v_gov_z=-0.0499`, `v_final_z=-0.0328`, beta `0.6486`, clearance p05 `0.4143`, and near-floor ICS violation rate `0.5838`.

Fixed R5D-best base for every R5F variant:

```text
algo.instinctRL.governor.v_corr_limit=0.35
instinctRL.reward.preservation_high_weight=2.0
instinctRL.reward.command_amplification_weight=2.5
instinctRL.reward.proxy_tracking_weight=0.5
instinctRL.reward.safety_weight=1.2
instinctRL.reward.clearance_margin=0.4
instinctRL.ics.active_horizon_margin=1.0
instinctRL.ics.clearance_margin=0.15
instinctRL.reward.null_command_speed_weight=4.0
instinctRL.reward.height_floor=0.5
instinctRL.reward.height_floor_weight=8.0
instinctRL.reward.height_ceiling=4.0
instinctRL.reward.height_ceiling_weight=8.0
algo.instinctRL.governor.v_corr_z_limit=0.12
```

Default sweep tag: `a2r5f_sweep`.

Variants, in order:

| Variant | Additional overrides | Hypothesis |
|---|---|---|
| `r5f_null_axis_xy050_z000` | `algo.instinctRL.governor.null_vcorr_axis_split_enabled=true`, `algo.instinctRL.governor.null_vcorr_xy_gate_min=0.50`, `algo.instinctRL.governor.null_vcorr_z_gate_min=0.0` | Increase null XY correction authority while suppressing null z correction. |
| `r5f_null_axis_xy075_z000` | `algo.instinctRL.governor.null_vcorr_axis_split_enabled=true`, `algo.instinctRL.governor.null_vcorr_xy_gate_min=0.75`, `algo.instinctRL.governor.null_vcorr_z_gate_min=0.0` | More aggressive XY null correction tests station/anchor recovery. |
| `r5f_zsign_opp050_reinf100` | `algo.instinctRL.governor.tracking_vcorr_z_sign_gate_enabled=true`, `algo.instinctRL.governor.tracking_vcorr_z_gate_eps=0.001`, `algo.instinctRL.governor.tracking_vcorr_z_opposing_gain=0.50`, `algo.instinctRL.governor.tracking_vcorr_z_reinforcing_gain=1.0` | Reduce z correction that opposes vertical tracking while preserving reinforcing correction. |
| `r5f_zsign_opp100_reinf050` | `algo.instinctRL.governor.tracking_vcorr_z_sign_gate_enabled=true`, `algo.instinctRL.governor.tracking_vcorr_z_gate_eps=0.001`, `algo.instinctRL.governor.tracking_vcorr_z_opposing_gain=1.0`, `algo.instinctRL.governor.tracking_vcorr_z_reinforcing_gain=0.50` | Preserve opposing correction but reduce reinforcing z correction that may amplify descent/ascent. |
| `r5f_downatten` | `instinctRL.ics.downward_attenuation_enabled=true`, `instinctRL.ics.downward_ray_min_z=0.25`, `instinctRL.ics.downward_clearance_margin=0.0` | Test MID360 range-derived downward attenuation for near-floor clearance and ICS violations. |
| `r5f_preserve_h050_v100` | `instinctRL.reward.horizontal_preservation_weight=0.5`, `instinctRL.reward.vertical_preservation_weight=1.0` | Add reward-only axis preservation pressure without changing actor inputs or command method. |

Validation commands:

```bash
cd /home/mint/rl_dev/NavRL/isaac-training
python -m py_compile training/scripts/instinctRL/sweep.py
python -m pytest -q training/unit_test/test_instinctrl_gates.py
python -m pytest -q training/unit_test/test_instinctrl_*.py
python training/scripts/instinctRL/sweep.py --frames 131072 --seeds 0 --limit 6
```

Execute eligibility:

- A clean dry-run only authorizes review of the planned commands.
- A 128k R5F execute sweep is allowed only in a later turn, still requires explicit `--execute`, and must use the same six default variants unless this ticket is updated first.
- No 1M confirmation is allowed unless a later executed 128k R5F candidate passes all hard gates.

## R5F Sweep Dry-Run Backfill - 2026-07-14

Changed files:

- `docs/instinctRL_devlog/DECISION_LOG.md`
- `docs/instinctRL_devlog/tickets/instinctRL-A2-R5_hypothesis_sweep_plan.md`
- `isaac-training/training/scripts/instinctRL/sweep.py`
- `isaac-training/training/unit_test/test_instinctrl_gates.py`
- `isaac-training/training/unit_test/test_instinctrl_actor_audit.py`

Code and test changes:

- Added `default_r5f_mechanism_variants()` and kept `default_safety_preservation_variants()` as a compatibility wrapper.
- Updated the CLI default tag to `a2r5f_sweep`.
- Replaced R5D defaults with the six R5F variants above, all sharing the fixed R5D-best base plus `algo.instinctRL.governor.v_corr_z_limit=0.12`.
- Preserved train/eval override propagation: every variant's full override tuple appears in `train_command`, `eval_command`, and `job.eval_overrides`.
- Added focused tests for exact R5F order, default tag, common base, per-variant train/eval propagation, absence of privileged height filtering, hard-gate snapshot stability, actor observation contract, forbidden actor keys, and sim/eval-only labeling of the privileged height filter.

Validation results:

- `python -m py_compile training/scripts/instinctRL/sweep.py`: passed.
- `python -m pytest -q training/unit_test/test_instinctrl_gates.py`: passed, `6 passed in 1.10s`.
- `python -m pytest -q training/unit_test/test_instinctrl_*.py`: passed, `127 passed, 12 warnings in 2.37s`.

Dry-run JSON summary:

- Command run: `python training/scripts/instinctRL/sweep.py --frames 131072 --seeds 0 --limit 6`.
- Result: passed as dry-run only with `execute=false`, `frames=131072`, `seed=0`, six generated jobs, `checkpoint_path=null`, `gate_report=null`, and `error=null`.
- Generated artifact paths pointed under `docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_185019/`, but no execution artifact is expected because `--execute` was not used.
- Generated variants, in order: `r5f_null_axis_xy050_z000`, `r5f_null_axis_xy075_z000`, `r5f_zsign_opp050_reinf100`, `r5f_zsign_opp100_reinf050`, `r5f_downatten`, `r5f_preserve_h050_v100`.
- All run names used the `instinctrl_a2r5f_sweep_*_131072_seed0` prefix.
- Train and eval commands both included the fixed R5D-best base, `algo.instinctRL.governor.v_corr_z_limit=0.12`, and each variant's R5F-specific overrides.
- No default variant included `instinctRL.safety_filter.privileged_height_floor_enabled=true`.

Next step:

- R5F 128k execute sweep is allowed only after this dry-run is reviewed and only with explicit `--execute` in a later turn.
- No 1M, promotion, warm-start, hard-gate change, actor-observation change, platform/sensor change, reward-default change, or body-frame velocity-governor method change is authorized by this dry-run backfill.

## R5F Execution Backfill - 2026-07-14

Execution:

- Preflight found a clean tracked worktree at baseline commit `1fb251bb9da6f9000b04b9b358e4988d92397bea`.
- Re-read the handbook, decision log, R5 ticket, `training/scripts/instinctRL/gates.py`, `training/scripts/instinctRL/sweep.py`, R5D summary, and R5E diagnostic artifact before execution.
- Verified `training/scripts/instinctRL/sweep.py` default tag was `a2r5f_sweep`.
- Verified the R5F variant set was exactly `r5f_null_axis_xy050_z000`, `r5f_null_axis_xy075_z000`, `r5f_zsign_opp050_reinf100`, `r5f_zsign_opp100_reinf050`, `r5f_downatten`, and `r5f_preserve_h050_v100`.
- Verified no default variant contained `instinctRL.safety_filter.privileged_height_floor_enabled=true`.
- Command run exactly once: `python training/scripts/instinctRL/sweep.py --execute --frames 131072 --seeds 0 --limit 6`.
- Artifact directory: `docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_195313/`.
- Summary: `docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_195313/summary.json`.
- Summary validation: `execute=true`, `frames=131072`, six jobs, all with `error=null`, non-null checkpoints, non-null eval JSON artifacts, and embedded `gate_report`.
- Gate truth source: only embedded `job.gate_report` in the new `summary.json`, produced by `training/scripts/instinctRL/gates.py`.
- Train and eval commands retained the TASLAB_UAV + Livox MID360, command-governor, station-first, short-diagnostic, proxy-observability assumptions and propagated each R5F override to eval.
- No 1M, formal long training, warm-start, code change, hard-gate change, actor-observation change, platform/sensor change, method change, reward-default change, or privileged height-filter sweep was executed.

Execution decision:

- No candidate passed all 14 hard gates.
- No job has `gate_report.passed=true`, `passed_count=14`, and `total_count=14`.
- Best ranked job by embedded hard-gate report was `r5f_zsign_opp100_reinf050`, seed `0`, with `6/14`, score `4.842`, `passed=false`, and `safety_passed=false`.
- No 1M confirmation command is written because there is no `passed=true`, `14/14` candidate.
- R5F micro-sweep is not allowed: the best candidate is below `10/14`, fails station gates, has collision and below-bound terminations, fails clearance, and has `ics_violation_rate=0.05428125`, more than 2x the `0.005` gate.
- Continue mechanism diagnosis. The remaining blockers are station/null-anchor behavior, clearance/ICS, collision termination, and below-bound height/safety behavior.

Rows below are ranked as written in `summary.json`.

| Variant | Seed | Gates | Score | Passed | Safety passed | Failed gates |
|---|---:|---:|---:|---|---|---|
| `r5f_zsign_opp100_reinf050` | 0 | 6/14 | 4.842 | false | false | `station_drift_mean`, `station_null_speed_mean`, `station_anchor_error_mean`, `safety_collision_rate`, `safety_min_clearance_p05`, `ics_violation_rate`, `termination_collision`, `termination_below_bound` |
| `r5f_zsign_opp050_reinf100` | 0 | 5/14 | 2.918 | false | false | `station_drift_mean`, `station_drift_p95`, `station_null_speed_mean`, `station_anchor_error_mean`, `safety_collision_rate`, `safety_min_clearance_p05`, `ics_violation_rate`, `termination_collision`, `termination_below_bound` |
| `r5f_preserve_h050_v100` | 0 | 4/14 | 2.027 | false | false | `station_drift_mean`, `station_drift_p95`, `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_preservation_ratio`, `safety_collision_rate`, `safety_min_clearance_p05`, `ics_violation_rate`, `termination_collision`, `termination_below_bound` |
| `r5f_downatten` | 0 | 4/14 | 1.754 | false | false | `station_drift_mean`, `station_drift_p95`, `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_preservation_ratio`, `safety_collision_rate`, `safety_min_clearance_p05`, `ics_violation_rate`, `termination_collision`, `termination_below_bound` |
| `r5f_null_axis_xy075_z000` | 0 | 3/14 | 0.197 | false | false | `station_drift_mean`, `station_drift_p95`, `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_preservation_ratio`, `tracking_command_amplification_mean`, `safety_collision_rate`, `safety_min_clearance_p05`, `ics_violation_rate`, `termination_collision`, `termination_below_bound` |
| `r5f_null_axis_xy050_z000` | 0 | 2/14 | -1.401 | false | false | `station_drift_mean`, `station_drift_p95`, `station_null_speed_mean`, `station_anchor_error_mean`, `tracking_rmse_actual`, `tracking_preservation_ratio`, `tracking_command_amplification_mean`, `safety_collision_rate`, `safety_min_clearance_p05`, `ics_violation_rate`, `termination_collision`, `termination_below_bound` |

Checkpoint and eval artifacts:

| Variant | Checkpoint | Eval artifact |
|---|---|---|
| `r5f_zsign_opp100_reinf050` | `/home/mint/rl_dev/NavRL/isaac-training/wandb/offline-run-20260714_203127-dar23kry/files/checkpoint_final.pt` | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_195313/a2r5f_sweep_r5f_zsign_opp100_reinf050_131072_seed0_eval.json` |
| `r5f_zsign_opp050_reinf100` | `/home/mint/rl_dev/NavRL/isaac-training/wandb/offline-run-20260714_201707-vrsci8pr/files/checkpoint_final.pt` | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_195313/a2r5f_sweep_r5f_zsign_opp050_reinf100_131072_seed0_eval.json` |
| `r5f_preserve_h050_v100` | `/home/mint/rl_dev/NavRL/isaac-training/wandb/offline-run-20260714_210024-egnker17/files/checkpoint_final.pt` | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_195313/a2r5f_sweep_r5f_preserve_h050_v100_131072_seed0_eval.json` |
| `r5f_downatten` | `/home/mint/rl_dev/NavRL/isaac-training/wandb/offline-run-20260714_204344-6gata2y4/files/checkpoint_final.pt` | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_195313/a2r5f_sweep_r5f_downatten_131072_seed0_eval.json` |
| `r5f_null_axis_xy075_z000` | `/home/mint/rl_dev/NavRL/isaac-training/wandb/offline-run-20260714_200613-ybgfftj7/files/checkpoint_final.pt` | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_195313/a2r5f_sweep_r5f_null_axis_xy075_z000_131072_seed0_eval.json` |
| `r5f_null_axis_xy050_z000` | `/home/mint/rl_dev/NavRL/isaac-training/wandb/offline-run-20260714_195318-t7v41msm/files/checkpoint_final.pt` | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_195313/a2r5f_sweep_r5f_null_axis_xy050_z000_131072_seed0_eval.json` |

Station/null/anchor diagnostics:

| Variant | Station drift mean/p95 | Null actual | Null output | Anchor error |
|---|---:|---:|---:|---:|
| `r5f_zsign_opp100_reinf050` | 1.334 / 2.596 | 0.1697 | 0.0304 | 2.978 |
| `r5f_zsign_opp050_reinf100` | 1.591 / 3.100 | 0.2026 | 0.0382 | 3.195 |
| `r5f_preserve_h050_v100` | 1.513 / 2.941 | 0.1922 | 0.00910 | 3.190 |
| `r5f_downatten` | 1.520 / 2.957 | 0.1933 | 0.00619 | 3.223 |
| `r5f_null_axis_xy075_z000` | 1.807 / 3.550 | 0.2341 | 0.0867 | 3.445 |
| `r5f_null_axis_xy050_z000` | 2.109 / 4.044 | 0.2629 | 0.0792 | 3.682 |

R5E null split diagnostics:

| Variant | Null actual xy | Null actual z abs | Null output xy | Null output z abs |
|---|---:|---:|---:|---:|
| `r5f_zsign_opp100_reinf050` | 0.1548 | 0.0694 | 0.0281 | 0.0117 |
| `r5f_zsign_opp050_reinf100` | 0.1894 | 0.0717 | 0.0371 | 0.00909 |
| `r5f_preserve_h050_v100` | 0.1764 | 0.0762 | 0.00811 | 0.00411 |
| `r5f_downatten` | 0.1760 | 0.0796 | 0.00619 | 0.000287 |
| `r5f_null_axis_xy075_z000` | 0.2199 | 0.0799 | 0.0867 | 0.0000 |
| `r5f_null_axis_xy050_z000` | 0.2503 | 0.0800 | 0.0793 | 0.0000 |

Tracking and command-preservation diagnostics:

| Variant | RMSE | Preservation | Pre-ICS | Post-ICS | ICS loss | Horizontal | Vertical abs | Amp mean/rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `r5f_zsign_opp100_reinf050` | 0.3212 | 0.7893 | 0.7971 | 0.7576 | 0.0395 | 0.7959 | 0.6826 | 0.0406 / 0.1186 |
| `r5f_zsign_opp050_reinf100` | 0.4004 | 0.7547 | 0.8316 | 0.7184 | 0.1132 | 0.7375 | 0.6990 | 0.0437 / 0.1494 |
| `r5f_preserve_h050_v100` | 0.4172 | 0.5425 | 0.5001 | 0.4693 | 0.0308 | 0.4757 | 0.4798 | 0.00978 / 0.0162 |
| `r5f_downatten` | 0.4238 | 0.5193 | 0.5037 | 0.4612 | 0.0425 | 0.4772 | 0.4491 | 0.00358 / 0.00706 |
| `r5f_null_axis_xy075_z000` | 0.4272 | 0.6481 | 0.6970 | 0.5915 | 0.1055 | 0.6228 | 0.4686 | 0.0625 / 0.0948 |
| `r5f_null_axis_xy050_z000` | 0.4598 | 0.7100 | 0.8135 | 0.6651 | 0.1484 | 0.7060 | 0.5612 | 0.0694 / 0.1481 |

Safety and termination diagnostics:

| Variant | Clearance p05 | Collision | ICS | Term below | Term above | Term collision |
|---|---:|---:|---:|---:|---:|---:|
| `r5f_zsign_opp100_reinf050` | 0.7466 | 0.000125 | 0.0543 | 0.1875 | 0.0000 | 0.0625 |
| `r5f_zsign_opp050_reinf100` | 0.5570 | 0.000188 | 0.1013 | 0.3125 | 0.0000 | 0.0938 |
| `r5f_preserve_h050_v100` | 0.6749 | 0.000125 | 0.0680 | 0.3125 | 0.0000 | 0.0625 |
| `r5f_downatten` | 0.5976 | 0.000219 | 0.0863 | 0.3750 | 0.0000 | 0.1250 |
| `r5f_null_axis_xy075_z000` | 0.5696 | 0.000250 | 0.0998 | 0.4062 | 0.0000 | 0.1250 |
| `r5f_null_axis_xy050_z000` | 0.4938 | 0.000563 | 0.1279 | 0.2188 | 0.0000 | 0.2812 |

Vertical diagnostics:

| Variant | Corr z mean/abs | Pos/neg frac | Sat | Null abs | Null active | Null drift | Tracking active | Tracking amp | Tracking preserve | ICS dz abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `r5f_zsign_opp100_reinf050` | 0.0314 / 0.0314 | 1.000 / 0.000 | 0.000 | 0.0117 | 1.000 | 1.334 | 1.000 | 0.0558 | 0.7499 | 0.0107 |
| `r5f_zsign_opp050_reinf100` | 0.0222 / 0.0222 | 1.000 / 0.000031 | 0.000 | 0.00909 | 1.000 | 1.591 | 1.000 | 0.0604 | 0.7114 | 0.0224 |
| `r5f_preserve_h050_v100` | 0.0130 / 0.0130 | 1.000 / 0.000031 | 0.000 | 0.00411 | 0.9999 | 1.513 | 0.9999 | 0.0358 | 0.4727 | 0.00797 |
| `r5f_downatten` | 0.00329 / 0.00352 | 0.717 / 0.283 | 0.000 | 0.000287 | 0.00231 | 0.00130 | 0.482 | 0.0249 | 0.4495 | 0.0101 |
| `r5f_null_axis_xy075_z000` | -0.0181 / 0.0181 | 0.000 / 0.866 | 0.000 | 0.0000 | 0.0000 | n/a | 0.988 | 0.0343 | 0.5670 | 0.0194 |
| `r5f_null_axis_xy050_z000` | 0.0265 / 0.0267 | 0.868 / 0.00263 | 0.000 | 0.0000 | 0.0000 | n/a | 0.998 | 0.0489 | 0.6542 | 0.0221 |

Near-floor diagnostics:

| Variant | Near-floor rate | v_cmd_z | v_gov_z | v_final_z | Beta | Clearance p05 | Near-floor ICS |
|---|---:|---:|---:|---:|---:|---:|---:|
| `r5f_zsign_opp100_reinf050` | 0.0498 | -0.0848 | -0.0255 | -0.00718 | 0.4001 | 0.4175 | 0.5881 |
| `r5f_zsign_opp050_reinf100` | 0.1114 | -0.1360 | -0.0763 | -0.00362 | 0.1873 | 0.4215 | 0.7246 |
| `r5f_preserve_h050_v100` | 0.0961 | -0.0427 | -0.00582 | 0.00302 | 0.4927 | 0.4193 | 0.5982 |
| `r5f_downatten` | 0.1184 | -0.0932 | -0.0415 | -0.000768 | 0.3684 | 0.4042 | 0.6269 |
| `r5f_null_axis_xy075_z000` | 0.0935 | -0.2383 | -0.1418 | -0.0175 | 0.0982 | 0.4115 | 0.7371 |
| `r5f_null_axis_xy050_z000` | 0.0704 | -0.1414 | -0.0588 | 0.00205 | 0.1214 | 0.3695 | 0.6701 |

Interpretation:

- The sign-aware vertical branch `r5f_zsign_opp100_reinf050` was best ranked and recovered tracking preservation (`0.7893`) and norm-level amplification gates, but it still failed station drift mean, null speed, anchor error, collision, clearance, ICS, collision termination, and below-bound termination.
- The null-axis variants regressed station drift, null actual speed, anchor error, safety, and terminations, so increased null XY authority is not a useful follow-up anchor from this seed.
- `r5f_downatten` did not solve near-floor safety: near-floor rate was `0.1184`, near-floor clearance p05 was `0.4042`, near-floor ICS was `0.6269`, and total ICS remained `0.0863`.
- Axis preservation reward reduced amplification but drove preservation below the gate (`0.5425`) and did not solve station or safety failures.

Stop condition:

Hard gates unchanged; `training/scripts/instinctRL/gates.py` unchanged; no actor observation, learned action method, platform/sensor, reward default, height clamp, privileged height filter, or body-frame velocity-governor method was modified; no 1M/formal/warm-start run executed.

## A2-R5G Mechanism Diagnosis Backfill - 2026-07-14

Scope:

- Added eval/logging-only R5G diagnostics for station null command-to-motion mismatch, anchor active/valid/loss conditionals, near-floor pre-termination windows, and MID360 downward attenuation activation/effectiveness.
- Diagnostics are emitted only through `info`, `eval/diagnostics.*`, and `eval/handbook.r5g_*`; actor observation remains `lidar_grid + state_vec`.
- No hard gates, training behavior, actor observation, platform/sensor, method, reward defaults, warm-start behavior, promotion rule, or privileged root-height deployable sweep path changed.

Validation:

- `python -m py_compile training/scripts/instinctRL/ics.py training/scripts/instinctRL/task_metrics.py training/scripts/instinctRL/audit.py training/scripts/env.py training/scripts/utils.py training/scripts/eval.py`: passed.
- `python -m pytest -q training/unit_test/test_instinctrl_task_metrics.py training/unit_test/test_instinctrl_ics.py training/unit_test/test_instinctrl_eval_diagnostic.py training/unit_test/test_instinctrl_actor_audit.py`: passed, `39 passed`.
- Eval-only replay artifacts:
  - `docs/instinctRL_devlog/tests/artifacts/r5g_diagnostics/20260714_195313/r5g_r5f_zsign_opp100_reinf050_eval.json`
  - `docs/instinctRL_devlog/tests/artifacts/r5g_diagnostics/20260714_195313/r5g_r5f_downatten_eval.json`
- Eval-only replay commands reused the corresponding R5F checkpoint and eval overrides from `docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_195313/summary.json`, changing only `result_path`.
- No training, sweep, 1M, warm-start, or promotion command was run.

R5F micro-sweep remains disallowed:

- Best R5F candidate `r5f_zsign_opp100_reinf050` is still only `6/14`.
- It fails station drift, null speed, anchor error, clearance, ICS, collision termination, and below-bound termination.
- Its `ics_violation_rate=0.05428125`, more than 2x the `0.005` gate.
- Failures are not limited to small amplification/clearance misses, so the bounded micro-sweep branch is closed.

Station/null command-to-motion audit:

| Variant | Null actual XY | Null output XY | Mismatch XY | Mismatch Z abs | Actual/output XY ratio | XY alignment |
|---|---:|---:|---:|---:|---:|---:|
| `r5f_zsign_opp100_reinf050` | 0.1548 | 0.0281 | 0.1824 | 0.0811 | 5.50 | -0.974 |
| `r5f_downatten` | 0.1760 | 0.00619 | 0.1797 | 0.0799 | 28.58 | -0.574 |

Interpretation:

- The station failure is not caused by large issued null commands. Issued output is low, but actual motion stays high.
- Negative/poor XY alignment means actual motion is often not following the small issued output direction.
- The next station diagnosis should inspect velocity-controller response, command latency/hold behavior, state reset/episode timing, and command-to-motion dynamics before tuning station reward or null-axis gains.

Anchor audit:

| Variant | Anchor valid rate | High-loss rate | Drift when valid | Drift when high loss | Error when valid | Error when high loss | Obs-poor rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `r5f_zsign_opp100_reinf050` | 0.999 | 0.898 | 1.336 | 1.476 | 2.981 | 3.267 | 0.000 |
| `r5f_downatten` | 0.999 | 0.909 | 1.521 | 1.663 | 3.226 | 3.501 | 0.000 |

Interpretation:

- High anchor error is not explained by inactive anchors, invalid anchor masks, or poor observability in this replay.
- Anchor error rises under high anchor loss and tracks station drift, so the failure is consistent with active-valid anchor loss under actual vehicle drift rather than anchor lifecycle/observability absence.

Near-floor and termination audit:

| Variant | Near-floor rate | Near-floor ICS | Below-bound near-floor rate in last 25 steps | Below-bound v_cmd/gov/final z | Below-bound beta | Below-bound clearance p05 | Collision near-floor rate in last 25 steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `r5f_zsign_opp100_reinf050` | 0.0498 | 0.588 | 0.96 | -0.040 / 0.002 / 0.001 | 0.104 | 0.388 | 0.00 |
| `r5f_downatten` | 0.1184 | 0.627 | 0.96 | -0.199 / -0.092 / -0.013 | 0.066 | 0.387 | 0.48 |

Interpretation:

- Below-bound failures cluster strongly near the floor: 96% of the final 25-step windows before below-bound termination are near-floor for both variants.
- For best, below-bound occurs even when `v_gov_z` and `v_final_z` are near zero/slightly positive, so it is not only a downward issued-command problem.
- For downatten, below-bound has stronger downward command/gov/final z and lower beta, while collision windows include zero final z, beta `0.0`, and lower clearance p05 (`0.303`), indicating severe late safety intervention without recovery.

Downatten audit:

| Variant | Downward has-ray rate | Downward active rate | Active beta | Active attenuation |
|---|---:|---:|---:|---:|
| `r5f_zsign_opp100_reinf050` | 0.000 | 0.000 | n/a | n/a |
| `r5f_downatten` | 0.000 | 0.000 | n/a | n/a |

Interpretation:

- The R5F `r5f_downatten` mechanism did not improve near-floor safety because it never became eligible: `downward_ray_min_z=0.25` found no valid MID360 downward rays in the eval path.
- This is consistent with the locked Livox MID360 vertical field-of-view and means the previous downatten result was effectively not a tested downward-attenuation mechanism.
- Any future downward attenuation design must first be reconciled with actual MID360 ray geometry and remain actor-clean/default-off until explicitly reviewed.

R5G conclusion:

- No promotion, no 1M, no R5F micro-sweep.
- R5G dry-run variant design is allowed next, but execution is not authorized in this ticket state.
- The next design should target only documented mechanisms: command-to-motion/null-station mismatch, active-valid anchor loss under drift, and MID360-compatible near-floor/downward safety diagnostics.
- If those mechanisms cannot be made actor-clean under TASLAB_UAV + Livox MID360 with body-frame velocity-governor commands, stop and re-review task/environment/handbook assumptions before any more sweep design.

## A2-R5G Dry-Run Variant Design - 2026-07-14

R5G dry-run scope:

- This round implements dry-run readiness only: docs, default sweep variants, unit tests, and dry-run validation.
- Do not pass `--execute`, do not run 1M, do not warm-start any checkpoint, and do not promote any candidate.
- Keep hard gates unchanged, keep TASLAB_UAV + Livox MID360, keep actor observation exactly `lidar_grid + state_vec`, keep the body-frame velocity-governor method, and exclude privileged height safety filtering from defaults.

Shared R5G base for every variant, derived from the best R5F candidate:

```text
algo.instinctRL.governor.v_corr_limit=0.35
instinctRL.reward.preservation_high_weight=2.0
instinctRL.reward.command_amplification_weight=2.5
instinctRL.reward.proxy_tracking_weight=0.5
instinctRL.reward.safety_weight=1.2
instinctRL.reward.clearance_margin=0.4
instinctRL.ics.active_horizon_margin=1.0
instinctRL.ics.clearance_margin=0.15
instinctRL.reward.null_command_speed_weight=4.0
instinctRL.reward.height_floor=0.5
instinctRL.reward.height_floor_weight=8.0
instinctRL.reward.height_ceiling=4.0
instinctRL.reward.height_ceiling_weight=8.0
algo.instinctRL.governor.v_corr_z_limit=0.12
algo.instinctRL.governor.tracking_vcorr_z_sign_gate_enabled=true
algo.instinctRL.governor.tracking_vcorr_z_gate_eps=0.001
algo.instinctRL.governor.tracking_vcorr_z_opposing_gain=1.0
algo.instinctRL.governor.tracking_vcorr_z_reinforcing_gain=0.50
```

Default sweep tag: `a2r5g_sweep`.

R5G variants, in order:

| Variant | Additional overrides | Hypothesis |
|---|---|---|
| `r5g_smooth025` | `algo.instinctRL.governor.smoothing_tau=0.25` | Command-to-motion/null-station mismatch may come from command/controller lag or sign chatter; mild actor-clean command smoothing should reduce actual/output mismatch. |
| `r5g_smooth040` | `algo.instinctRL.governor.smoothing_tau=0.40` | Dose test for the same command-to-motion hypothesis without changing observation or method. |
| `r5g_anchor_huber050` | `instinctRL.anchor.huber_delta=0.50` | Active-valid anchor loss under drift may be under-sensitive to large MID360 residuals; increase Huber slope without repeating `anchor_weight=5.0`. |
| `r5g_smooth025_anchor_huber050` | `algo.instinctRL.governor.smoothing_tau=0.25`, `instinctRL.anchor.huber_delta=0.50` | Combined station mechanism: smoother issued commands plus stronger active-valid anchor-loss sensitivity. |
| `r5g_downatten_z010` | `instinctRL.ics.downward_attenuation_enabled=true`, `instinctRL.ics.downward_ray_min_z=0.10`, `instinctRL.ics.downward_clearance_margin=0.0` | MID360 lower FOV is shallow; `0.25` never triggered, while `0.10` should catch physically downward lower beams. |
| `r5g_downatten_z005` | `instinctRL.ics.downward_attenuation_enabled=true`, `instinctRL.ics.downward_ray_min_z=0.05`, `instinctRL.ics.downward_clearance_margin=0.0` | More permissive but still nonzero MID360-compatible downward eligibility; do not use `0.0`. |

Excluded from R5G defaults:

- No null-axis XY variants.
- No `tracking_vcorr_z_gain=0` or `tracking_vcorr_z_gain=0.50`.
- No `v_corr_z_limit=0.20`.
- No axis-preservation reward-only variant.
- No `downward_ray_min_z=0.25`.
- No `anchor_weight=5.0`.
- No `instinctRL.safety_filter.*` override and no privileged height filter.

Implementation notes:

- Replaced `training/scripts/instinctRL/sweep.py` defaults with `default_r5g_dry_run_variants()` and kept `default_safety_preservation_variants()` as the sweep entry point.
- Updated the CLI default tag to `a2r5g_sweep`.
- Kept `variant.overrides` propagated into both train and eval commands.
- Kept `instinctRL.eval.suite=short_diagnostic`.
- Extended gate and actor-audit tests for exact R5G order, full base propagation, train/eval override propagation, dry-run run-name prefix, absence of safety-filter overrides, unchanged hard-gate snapshot, and forbidden R5G/height/downward/safety-filter actor-observation diagnostics.

Validation:

- `python -m py_compile training/scripts/instinctRL/sweep.py`: passed.
- `python -m pytest -q training/unit_test/test_instinctrl_gates.py training/unit_test/test_instinctrl_actor_audit.py`: passed, `15 passed`.
- `python -m pytest -q training/unit_test/test_instinctrl_*.py`: passed, `129 passed, 12 warnings`.
- `python training/scripts/instinctRL/sweep.py --frames 131072 --seeds 0 --limit 6`: passed as dry-run only.
- Dry-run stdout validation: `execute=false`, `frames=131072`, six jobs, only the six `r5g_*` variants above, `checkpoint_path=null`, `gate_report=null`, `error=null`, short diagnostic eval, no `instinctRL.safety_filter.*` override, and `wandb.name=instinctrl_a2r5g_sweep_*`.

Decision:

- R5G dry-run readiness is complete for review.
- This dry-run readiness does not authorize `--execute`, 1M confirmation, warm-start, promotion, hard-gate change, actor-observation change, platform/sensor change, reward-default change, body-frame method change, or privileged height-filter default use.
- The next step may request human approval for a controlled 128k R5G execute sweep, but execute remains unauthorized until explicitly approved later.

## Decision Tree After R5A

1. If any candidate passes all 14 gates:
   - run a 1M confirmation for that exact config;
   - do not change parameters between 128k and 1M;
   - after 1M passes, run 128k multi-seed screening with seeds `0 1 2`.

2. If no candidate passes but one candidate has at least 10/14 gates, zero terminations, and only small amp/clearance misses:
   - do not run 1M;
   - create one more R5A micro-sweep with at most four variants around that candidate;
   - record the micro-sweep plan here before changing `sweep.py`.

3. If no candidate passes and failures still include below-bound or above-bound:
   - stop config-only tuning;
   - start R5B code work on height-band control and vertical correction diagnostics.

4. If no candidate passes and ICS remains above `0.005` by more than 2x:
   - stop reward-only safety tuning;
   - inspect ICS attenuation timing and effective clearance diagnostics before more sweeps.

## Context Reset Protocol For Future Codex Sessions

At the start of every future session, Codex should:

1. Read this file first.
2. Read `training/scripts/instinctRL/sweep.py`.
3. Read the newest sweep `summary.json` mentioned in this file.
4. Use `training/scripts/instinctRL/gates.py` as the source of hard-gate truth.
5. Update this file with what changed, what ran, what passed/failed, and the next adjustment.

Do not rely on chat history for experimental state.

## Current Next Action

R5G dry-run readiness is complete for review with six default `r5g_*` variants under tag `a2r5g_sweep`. Do not promote, run 1M, warm-start, change hard gates, change actor observation, change platform/sensor, or include the privileged root-height safety filter in the default deployable sweep set. The next step may request human approval for a controlled 128k R5G execute sweep, but execute remains unauthorized until explicitly approved later.

## A2-R5G 128k Execute Sweep - 2026-07-14

Execution:

- Command run exactly once: `python training/scripts/instinctRL/sweep.py --execute --frames 131072 --seeds 0 --limit 6`.
- Summary path: `docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_234801/summary.json`.
- Artifact timestamp: `20260714_234801`.
- Validation: `execute=true`, `frames=131072`, six jobs, every job has `error=null`, non-empty `checkpoint_path`, non-empty `result_path`, and an embedded `gate_report`; no train/eval command contains an `instinctRL.safety_filter.*` override.
- Generation order remained the R5G dry-run order: `r5g_smooth025`, `r5g_smooth040`, `r5g_anchor_huber050`, `r5g_smooth025_anchor_huber050`, `r5g_downatten_z010`, `r5g_downatten_z005`.
- The stored summary job order is the post-execute `sweep.py` rank order: `passed`, then `safety_passed`, then `score`.
- No 1M, warm-start, promotion, code/config tuning, hard-gate change, actor-observation change, platform/sensor change, command-method change, or privileged safety-filter default change was run.

Six-variant result table:

| Rank | Variant | Gates | Score | Passed | Safety passed | Checkpoint | Result |
|---:|---|---:|---:|---|---|---|---|
| 1 | `r5g_downatten_z010` | 6/14 | 4.6971 | `false` | `false` | `/home/mint/rl_dev/NavRL/isaac-training/wandb/offline-run-20260715_004807-durd58lw/files/checkpoint_final.pt` | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_234801/a2r5g_sweep_r5g_downatten_z010_131072_seed0_eval.json` |
| 2 | `r5g_downatten_z005` | 6/14 | 4.6533 | `false` | `false` | `/home/mint/rl_dev/NavRL/isaac-training/wandb/offline-run-20260715_005952-j5r0dcp6/files/checkpoint_final.pt` | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_234801/a2r5g_sweep_r5g_downatten_z005_131072_seed0_eval.json` |
| 3 | `r5g_smooth040` | 5/14 | 2.9564 | `false` | `false` | `/home/mint/rl_dev/NavRL/isaac-training/wandb/offline-run-20260715_000711-zpdlyl15/files/checkpoint_final.pt` | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_234801/a2r5g_sweep_r5g_smooth040_131072_seed0_eval.json` |
| 4 | `r5g_anchor_huber050` | 4/14 | 2.1560 | `false` | `false` | `/home/mint/rl_dev/NavRL/isaac-training/wandb/offline-run-20260715_002209-ezp1ynnp/files/checkpoint_final.pt` | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_234801/a2r5g_sweep_r5g_anchor_huber050_131072_seed0_eval.json` |
| 5 | `r5g_smooth025` | 4/14 | 1.6384 | `false` | `false` | `/home/mint/rl_dev/NavRL/isaac-training/wandb/offline-run-20260714_234806-ns92lyb2/files/checkpoint_final.pt` | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_234801/a2r5g_sweep_r5g_smooth025_131072_seed0_eval.json` |
| 6 | `r5g_smooth025_anchor_huber050` | 4/14 | 1.5920 | `false` | `false` | `/home/mint/rl_dev/NavRL/isaac-training/wandb/offline-run-20260715_003448-zold97jo/files/checkpoint_final.pt` | `docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_234801/a2r5g_sweep_r5g_smooth025_anchor_huber050_131072_seed0_eval.json` |

Per-variant gate table (`P` = passed, `F` = failed):

| Gate | `r5g_downatten_z010` | `r5g_downatten_z005` | `r5g_smooth040` | `r5g_anchor_huber050` | `r5g_smooth025` | `r5g_smooth025_anchor_huber050` |
|---|---:|---:|---:|---:|---:|---:|
| `station_drift_mean` | F | F | F | F | F | F |
| `station_drift_p95` | F | F | F | F | F | F |
| `station_null_speed_mean` | F | F | F | F | F | F |
| `station_anchor_error_mean` | F | F | F | F | F | F |
| `tracking_rmse_actual` | P | P | F | P | P | P |
| `tracking_preservation_ratio` | F | F | F | F | F | F |
| `tracking_command_amplification_mean` | P | P | P | P | P | P |
| `tracking_command_amplification_rate` | P | P | P | P | P | P |
| `safety_collision_rate` | F | F | P | F | F | F |
| `safety_min_clearance_p05` | P | P | F | F | F | F |
| `ics_violation_rate` | F | F | F | F | F | F |
| `termination_collision` | F | F | P | F | F | F |
| `termination_below_bound` | P | P | F | F | F | F |
| `termination_above_bound` | P | P | P | P | P | P |

Failed gates:

| Variant | Gate | Key | Value | Min | Max | Reason |
|---|---|---|---:|---:|---:|---|
| `r5g_downatten_z010` | `station_drift_mean` | `eval/station/handbook.station_keeping_drift_mean` | 1.5319 | n/a | 1.3000 | above max by 0.2319 |
| `r5g_downatten_z010` | `station_drift_p95` | `eval/station/handbook.station_keeping_drift_p95` | 2.9870 | n/a | 2.6000 | above max by 0.387048 |
| `r5g_downatten_z010` | `station_null_speed_mean` | `eval/station/handbook.null_command_speed_mean` | 0.195142 | n/a | 0.08 | above max by 0.115142 |
| `r5g_downatten_z010` | `station_anchor_error_mean` | `eval/station/handbook.anchor_error_mean` | 3.2438 | n/a | 2.0000 | above max by 1.2438 |
| `r5g_downatten_z010` | `tracking_preservation_ratio` | `eval/tracking/handbook.command_preservation_ratio` | 0.558618 | 0.75 | 1.0500 | below min by 0.191382 |
| `r5g_downatten_z010` | `safety_collision_rate` | `eval/handbook.safety_collision_rate` | 0.000063 | n/a | 0 | above max by 6.25e-05 |
| `r5g_downatten_z010` | `ics_violation_rate` | `eval/handbook.ics_violation_rate` | 0.020906 | n/a | 0.005 | above max by 0.0159063 |
| `r5g_downatten_z010` | `termination_collision` | `eval/handbook.termination_collision` | 0.03125 | n/a | 0 | above max by 0.03125 |
| `r5g_downatten_z005` | `station_drift_mean` | `eval/station/handbook.station_keeping_drift_mean` | 1.5706 | n/a | 1.3000 | above max by 0.270574 |
| `r5g_downatten_z005` | `station_drift_p95` | `eval/station/handbook.station_keeping_drift_p95` | 3.0563 | n/a | 2.6000 | above max by 0.456273 |
| `r5g_downatten_z005` | `station_null_speed_mean` | `eval/station/handbook.null_command_speed_mean` | 0.199751 | n/a | 0.08 | above max by 0.119751 |
| `r5g_downatten_z005` | `station_anchor_error_mean` | `eval/station/handbook.anchor_error_mean` | 3.2565 | n/a | 2.0000 | above max by 1.25646 |
| `r5g_downatten_z005` | `tracking_preservation_ratio` | `eval/tracking/handbook.command_preservation_ratio` | 0.620261 | 0.75 | 1.0500 | below min by 0.129739 |
| `r5g_downatten_z005` | `safety_collision_rate` | `eval/handbook.safety_collision_rate` | 0.000125 | n/a | 0 | above max by 0.000125 |
| `r5g_downatten_z005` | `ics_violation_rate` | `eval/handbook.ics_violation_rate` | 0.027688 | n/a | 0.005 | above max by 0.0226875 |
| `r5g_downatten_z005` | `termination_collision` | `eval/handbook.termination_collision` | 0.0625 | n/a | 0 | above max by 0.0625 |
| `r5g_smooth040` | `station_drift_mean` | `eval/station/handbook.station_keeping_drift_mean` | 1.4883 | n/a | 1.3000 | above max by 0.188278 |
| `r5g_smooth040` | `station_drift_p95` | `eval/station/handbook.station_keeping_drift_p95` | 2.8967 | n/a | 2.6000 | above max by 0.296748 |
| `r5g_smooth040` | `station_null_speed_mean` | `eval/station/handbook.null_command_speed_mean` | 0.189257 | n/a | 0.08 | above max by 0.109257 |
| `r5g_smooth040` | `station_anchor_error_mean` | `eval/station/handbook.anchor_error_mean` | 3.2372 | n/a | 2.0000 | above max by 1.23724 |
| `r5g_smooth040` | `tracking_rmse_actual` | `eval/tracking/handbook.tracking_rmse_actual_body_vs_v_cmd` | 0.453415 | n/a | 0.45 | above max by 0.00341462 |
| `r5g_smooth040` | `tracking_preservation_ratio` | `eval/tracking/handbook.command_preservation_ratio` | 0.420378 | 0.75 | 1.0500 | below min by 0.329622 |
| `r5g_smooth040` | `safety_min_clearance_p05` | `eval/handbook.safety_min_clearance_p05` | 0.660399 | 1.0000 | n/a | below min by 0.339601 |
| `r5g_smooth040` | `ics_violation_rate` | `eval/handbook.ics_violation_rate` | 0.076656 | n/a | 0.005 | above max by 0.0716562 |
| `r5g_smooth040` | `termination_below_bound` | `eval/handbook.termination_below_bound` | 0.3125 | n/a | 0 | above max by 0.3125 |
| `r5g_anchor_huber050` | `station_drift_mean` | `eval/station/handbook.station_keeping_drift_mean` | 1.5476 | n/a | 1.3000 | above max by 0.247587 |
| `r5g_anchor_huber050` | `station_drift_p95` | `eval/station/handbook.station_keeping_drift_p95` | 3.0087 | n/a | 2.6000 | above max by 0.408676 |
| `r5g_anchor_huber050` | `station_null_speed_mean` | `eval/station/handbook.null_command_speed_mean` | 0.196576 | n/a | 0.08 | above max by 0.116576 |
| `r5g_anchor_huber050` | `station_anchor_error_mean` | `eval/station/handbook.anchor_error_mean` | 3.2779 | n/a | 2.0000 | above max by 1.27789 |
| `r5g_anchor_huber050` | `tracking_preservation_ratio` | `eval/tracking/handbook.command_preservation_ratio` | 0.652314 | 0.75 | 1.0500 | below min by 0.0976861 |
| `r5g_anchor_huber050` | `safety_collision_rate` | `eval/handbook.safety_collision_rate` | 0.000063 | n/a | 0 | above max by 6.25e-05 |
| `r5g_anchor_huber050` | `safety_min_clearance_p05` | `eval/handbook.safety_min_clearance_p05` | 0.748736 | 1.0000 | n/a | below min by 0.251264 |
| `r5g_anchor_huber050` | `ics_violation_rate` | `eval/handbook.ics_violation_rate` | 0.053062 | n/a | 0.005 | above max by 0.0480625 |
| `r5g_anchor_huber050` | `termination_collision` | `eval/handbook.termination_collision` | 0.03125 | n/a | 0 | above max by 0.03125 |
| `r5g_anchor_huber050` | `termination_below_bound` | `eval/handbook.termination_below_bound` | 0.3125 | n/a | 0 | above max by 0.3125 |
| `r5g_smooth025` | `station_drift_mean` | `eval/station/handbook.station_keeping_drift_mean` | 1.6182 | n/a | 1.3000 | above max by 0.318217 |
| `r5g_smooth025` | `station_drift_p95` | `eval/station/handbook.station_keeping_drift_p95` | 3.1496 | n/a | 2.6000 | above max by 0.549575 |
| `r5g_smooth025` | `station_null_speed_mean` | `eval/station/handbook.null_command_speed_mean` | 0.205828 | n/a | 0.08 | above max by 0.125828 |
| `r5g_smooth025` | `station_anchor_error_mean` | `eval/station/handbook.anchor_error_mean` | 3.3028 | n/a | 2.0000 | above max by 1.30281 |
| `r5g_smooth025` | `tracking_preservation_ratio` | `eval/tracking/handbook.command_preservation_ratio` | 0.552503 | 0.75 | 1.0500 | below min by 0.197497 |
| `r5g_smooth025` | `safety_collision_rate` | `eval/handbook.safety_collision_rate` | 0.00025 | n/a | 0 | above max by 0.00025 |
| `r5g_smooth025` | `safety_min_clearance_p05` | `eval/handbook.safety_min_clearance_p05` | 0.586708 | 1.0000 | n/a | below min by 0.413292 |
| `r5g_smooth025` | `ics_violation_rate` | `eval/handbook.ics_violation_rate` | 0.084687 | n/a | 0.005 | above max by 0.0796875 |
| `r5g_smooth025` | `termination_collision` | `eval/handbook.termination_collision` | 0.125 | n/a | 0 | above max by 0.125 |
| `r5g_smooth025` | `termination_below_bound` | `eval/handbook.termination_below_bound` | 0.3125 | n/a | 0 | above max by 0.3125 |
| `r5g_smooth025_anchor_huber050` | `station_drift_mean` | `eval/station/handbook.station_keeping_drift_mean` | 1.6354 | n/a | 1.3000 | above max by 0.335378 |
| `r5g_smooth025_anchor_huber050` | `station_drift_p95` | `eval/station/handbook.station_keeping_drift_p95` | 3.2091 | n/a | 2.6000 | above max by 0.60911 |
| `r5g_smooth025_anchor_huber050` | `station_null_speed_mean` | `eval/station/handbook.null_command_speed_mean` | 0.210073 | n/a | 0.08 | above max by 0.130073 |
| `r5g_smooth025_anchor_huber050` | `station_anchor_error_mean` | `eval/station/handbook.anchor_error_mean` | 3.3282 | n/a | 2.0000 | above max by 1.32815 |
| `r5g_smooth025_anchor_huber050` | `tracking_preservation_ratio` | `eval/tracking/handbook.command_preservation_ratio` | 0.630095 | 0.75 | 1.0500 | below min by 0.119905 |
| `r5g_smooth025_anchor_huber050` | `safety_collision_rate` | `eval/handbook.safety_collision_rate` | 0.000188 | n/a | 0 | above max by 0.0001875 |
| `r5g_smooth025_anchor_huber050` | `safety_min_clearance_p05` | `eval/handbook.safety_min_clearance_p05` | 0.56949 | 1.0000 | n/a | below min by 0.43051 |
| `r5g_smooth025_anchor_huber050` | `ics_violation_rate` | `eval/handbook.ics_violation_rate` | 0.107219 | n/a | 0.005 | above max by 0.102219 |
| `r5g_smooth025_anchor_huber050` | `termination_collision` | `eval/handbook.termination_collision` | 0.09375 | n/a | 0 | above max by 0.09375 |
| `r5g_smooth025_anchor_huber050` | `termination_below_bound` | `eval/handbook.termination_below_bound` | 0.375 | n/a | 0 | above max by 0.375 |

Key diagnostics:

| Variant | Drift mean | Drift p95 | Null speed | Anchor err | Preserve | Amp mean | Amp rate | Collision rate | Clearance p05 | ICS | Term collision | Term below | Term above | Down has-ray | Down active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `r5g_downatten_z010` | 1.5319 | 2.9870 | 0.195142 | 3.2438 | 0.558618 | 0.009351 | 0.011219 | 0.000063 | 1.0047 | 0.020906 | 0.03125 | 0 | 0 | 1.0000 | 0.655469 |
| `r5g_downatten_z005` | 1.5706 | 3.0563 | 0.199751 | 3.2565 | 0.620261 | 0.012644 | 0.019844 | 0.000125 | 1.1229 | 0.027688 | 0.0625 | 0 | 0 | 1.0000 | 0.651531 |
| `r5g_smooth040` | 1.4883 | 2.8967 | 0.189257 | 3.2372 | 0.420378 | 0.005167 | 0.011469 | 0 | 0.660399 | 0.076656 | 0 | 0.3125 | 0 | 0 | 0 |
| `r5g_anchor_huber050` | 1.5476 | 3.0087 | 0.196576 | 3.2779 | 0.652314 | 0.020452 | 0.034031 | 0.000063 | 0.748736 | 0.053062 | 0.03125 | 0.3125 | 0 | 0 | 0 |
| `r5g_smooth025` | 1.6182 | 3.1496 | 0.205828 | 3.3028 | 0.552503 | 0.0066 | 0.013187 | 0.00025 | 0.586708 | 0.084687 | 0.125 | 0.3125 | 0 | 0 | 0 |
| `r5g_smooth025_anchor_huber050` | 1.6354 | 3.2091 | 0.210073 | 3.3282 | 0.630095 | 0.015375 | 0.026187 | 0.000188 | 0.56949 | 0.107219 | 0.09375 | 0.375 | 0 | 0 | 0 |

Best candidate and decision:

- Best candidate by `sweep.py` rank logic: `r5g_downatten_z010` with 6/14 gates, score 4.6971, `passed=false`, `safety_passed=false`.
- Failed gates for best: `station_drift_mean`=1.5319 (above max by 0.2319), `station_drift_p95`=2.9870 (above max by 0.387048), `station_null_speed_mean`=0.195142 (above max by 0.115142), `station_anchor_error_mean`=3.2438 (above max by 1.2438), `tracking_preservation_ratio`=0.558618 (below min by 0.191382), `safety_collision_rate`=0.000063 (above max by 6.25e-05), `ics_violation_rate`=0.020906 (above max by 0.0159063), `termination_collision`=0.03125 (above max by 0.03125).
- Decision rule hit: best is below 10/14, `safety_passed=false`, collision termination is 0.03125, and `ics_violation_rate` is 0.020906 (> 0.01).
- Final decision: stop sweeps, do not run 1M, forbid R5H micro-sweep, and route to mechanism diagnosis.
