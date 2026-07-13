# instinctRL-A2-R5 Hypothesis-Driven Sweep Plan

**Status**: R5D dry-run-only sweep implemented and validated; no training execution, 1M, promotion, warm-start, or hard-gate change
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

Do not promote any R5B candidate and do not run 1M. R5D dry-run passed; the next step is a controlled 128k R5D execute sweep with `--execute`. Only a `14/14` candidate with `passed=true` can be considered for a 1M confirmation.
