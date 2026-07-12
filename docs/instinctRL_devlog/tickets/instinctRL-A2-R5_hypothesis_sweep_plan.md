# instinctRL-A2-R5 Hypothesis-Driven Sweep Plan

**Status**: R5A 128k sweep executed; no promotion; config-only tuning stopped pending R5B height/safety diagnostics  
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

Start R5B planning before any further sweep. The next work should address height-band control and vertical-correction diagnostics, then inspect ICS attenuation timing and effective clearance diagnostics. Do not run 1M, do not change hard gates, and do not run another R5A config-only sweep without a recorded R5B plan.
