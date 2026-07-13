# instinctRL Training/Eval Conformance Audit

Date: 2026-07-07  
Scope: current learned-governor training and evaluation flow, compared against `docs/instinctRL_Development_Handbook_v1_1_platform_locked.tex`  
Authority order: code facts > handbook acceptance criteria > devlog records

## Executive Summary

The current repo has a real learned-governor action path: the PPO actor outputs a 4D governor action, the decoder forms `v_gov_b = alpha * v_cmd_b + v_corr`, and the train/eval wrappers route the body-frame command through the velocity-controller boundary. That part is broadly aligned with the handbook.

The surrounding task and evaluation semantics are not yet aligned. The current environment and eval reports still carry NavRL goal-navigation concepts such as `target_pos`, `reach_goal`, target-relative critic fields, and "success rate" interpreted as reaching a target. The handbook defines instinctRL as a range-inertial measurement-space velocity governor with station-keeping, command tracking, safety, ICS attenuation, and observability metrics. It does not define primary success as flying to a sampled goal position.

The 8M run is therefore useful evidence that the training pipeline can run stably, but it is not evidence of handbook-level learned-governor success. Continuing eval with the current `reach_goal` metric will not answer the right question.

## Source Evidence

### Handbook Requirements

- Locked method: range-inertial measurement-space velocity governor, corrected body-frame velocity command, station-keeping anchor, observability logging, and ICS-inspired attenuation (`docs/instinctRL_Development_Handbook_v1_1_platform_locked.tex:46`).
- Hard actor contract: no pose, odometry, explicit velocity estimate, map, SLAM state, or privileged simulator state in deployed actor input; privileged state only for reward, critic, evaluation, logging, or upper-bound baselines (`docs/instinctRL_Development_Handbook_v1_1_platform_locked.tex:71`).
- Existing NavRL logging/eval is only basic return/reach/collision/video plumbing; instinctRL needs anchor, ICS, observability, audit, and platform-lock logs (`docs/instinctRL_Development_Handbook_v1_1_platform_locked.tex:147`).
- Operator command `v_cmd` should come from an operator or `AdversarialCommandGenerator`; generator internals may use privileged scenario data, but only `v_cmd` enters actor input (`docs/instinctRL_Development_Handbook_v1_1_platform_locked.tex:264`).
- Reward/eval terms are command tracking, anchor, safety, beta-gated command compliance, intervention usage, smoothness, and collision (`docs/instinctRL_Development_Handbook_v1_1_platform_locked.tex:395`).
- Required metrics include station-keeping drift, range-anchor error, tracking RMSE, minimum clearance, collision rate, ICS violation rate, command preservation ratio, intervention frequency, and observability metrics (`docs/instinctRL_Development_Handbook_v1_1_platform_locked.tex:445`).

### Current Code Facts

- `env.py` generates `v_cmd` with simple bounded random sampling every about 2 seconds, not through `command_generator.py` (`isaac-training/training/scripts/env.py:846`).
- `env.py` stores `info["v_cmd"]` for critic/debug after building actor-clean observation history (`isaac-training/training/scripts/env.py:894`).
- `env.py` still computes `target_pos`, `rpos`, `distance`, `distance_2d`, target-frame velocity, and `drone_state` from goal-relative state (`isaac-training/training/scripts/env.py:923`).
- `env.py` exposes `target_rpos` and `target_distance` to `info` as critic-only privileged fields (`isaac-training/training/scripts/env.py:929`).
- instinctRL reward integration uses `v_cmd_b`, `v_final_b`, anchor, ICS, clearance, and collision when `_reward_computer` is enabled (`isaac-training/training/scripts/env.py:1090`).
- `reach_goal` is computed from target distance but is not included in `terminated`; termination is below-bound, above-bound, or collision, while truncation is max episode length (`isaac-training/training/scripts/env.py:1150`).
- Only `return`, `episode_len`, `reach_goal`, `collision`, and `truncated` are logged for termination outcome; below-bound and above-bound are not logged (`isaac-training/training/scripts/env.py:1164`).
- `utils.evaluate()` still documents `eval/stats.reach_goal` as success rate and reports old NavRL-style metrics (`isaac-training/training/scripts/utils.py:699`).
- Streaming eval accumulates optional governor/ICS tensor summaries and reward, but not the full required handbook metric set (`isaac-training/training/scripts/utils.py:566`).
- `ppo.py` critic still consumes `info/drone_state`, `info/target_rpos`, and `info/target_distance` (`isaac-training/training/scripts/ppo.py:147`).
- `command_generator.py` contains a vectorized 5-mode adversarial command generator, but it is not connected to the current env command path (`isaac-training/training/scripts/command_generator.py:6`).

### Current Devlog Facts

- `DEV_STATUS.md` still says 1M runtime acceptance is pending and formal long learned-governor training is on hold (`docs/instinctRL_devlog/DEV_STATUS.md:3`, `docs/instinctRL_devlog/DEV_STATUS.md:115`).
- `DEFERRED_REGISTER.md` says D-006 adversarial command generator is not a first-formal-training blocker and remains G/evaluation curriculum work (`docs/instinctRL_devlog/DEFERRED_REGISTER.md:96`).
- `TEST_PLAN.md` still records A2-S runtime acceptance as pending due to Isaac/Nucleus assets root (`docs/instinctRL_devlog/TEST_PLAN.md:9`).
- The 8M eval report records that eval completed but static `reach_goal` was zero across evaluated checkpoints and dynamic robustness was not demonstrated (`docs/instinctRL_devlog/tests/instinctRL_8M_eval_report.md:141`).

## Conformance Matrix

| ID | Requirement | Current implementation | Verdict | Severity | Impact |
| --- | --- | --- | --- | --- | --- |
| A-001 | Actor/governor action should be body-frame velocity governor `(alpha, v_corr)` | PPO learned mode emits 4D governor action; wrapper routes body-frame command through controller boundary | Conforms | Low | This part is usable and should be preserved. |
| A-002 | Actor input must exclude pose, odom, explicit velocity, map, privileged state | Actor observation is `lidar_grid + state_vec`; target fields are in critic/info only | Mostly conforms | Medium | Actor-clean contract appears protected, but source provenance should keep being audited. |
| T-001 | Primary task should be velocity-governor command tracking / station-keeping, not target-goal navigation | Env still computes target-relative state and eval uses `reach_goal` as success | Non-conforming | P0 | Current eval does not answer whether the learned governor works. |
| T-002 | `v_cmd` source should be operator or command generator regime | Current `v_cmd` is simple random sampling every about 2 seconds | Partial | P1 | Training is a basic random-command run, not a crazy/adversarial command curriculum. |
| T-003 | Null-command station-keeping should be a first-class task case | Anchor exists, but current eval report does not isolate null-command anchor-hold scenarios | Partial | P0 | Station-keeping performance is unknown. |
| R-001 | Reward should cover tracking, anchor, safety, beta compliance, intervention, smoothness, collision | Reward computer implements these terms | Mostly conforms | Medium | Reward terms exist, but eval metrics do not validate their intended behavior. |
| R-002 | Tracking should measure command following in the relevant physical/task sense | Default reward tracks `v_final_b` against `v_cmd_b`, not actual achieved velocity | Risk | P1 | Policy can look good on command-proxy reward without proving physical velocity tracking. |
| E-001 | Eval should log required handbook metrics | Current eval reports return/reach/collision/truncated plus optional governor/ICS summaries | Non-conforming | P0 | 8M eval cannot establish handbook success or diagnose failures. |
| E-002 | Termination reasons must be interpretable | Below-bound/above-bound are termination causes but not logged | Non-conforming | P0 | `reach_goal=0`, `collision=0`, `truncated=0` is ambiguous. |
| C-001 | Critic may use privileged state, but it should support the active task | Critic uses target-relative fields from legacy goal task | Risk | P1 | Value learning may be shaped by stale target semantics unrelated to command-governor reward. |
| D-001 | Devlog should reflect current truth | Status files lag behind later 1.31M/8M runs and eval report | Stale | P2 | Planning from devlog alone can lead to wrong go/no-go decisions. |

## Findings

### F-001: `reach_goal` Is Not a Valid Primary instinctRL Success Metric

Current eval treats `eval/stats.reach_goal` as success rate. In the code, `reach_goal` is derived from `target_pos - root_state` distance, while the handbook defines the task around `v_cmd`, `v_gov`, `v_final`, anchor holding, clearance, ICS attenuation, and observability metrics.

This makes the 8M eval conclusion easy to misread. `reach_goal=0.0` does not prove the learned governor failed at command tracking, because reaching the NavRL target is not the handbook task. It also does not prove the learned governor succeeded, because the eval does not report tracking RMSE, anchor error, command preservation, intervention frequency, or minimum clearance in the required form.

Recommendation: demote `reach_goal` to a legacy NavRL diagnostic unless an explicit goal-navigation baseline is running. For instinctRL, make command-governor metrics primary.

### F-002: The Environment Task Shell Still Mixes Legacy Goal Navigation With instinctRL

Even with instinctRL reward enabled, `env.py` computes target-relative quantities and uses target distance for `reach_goal`. These target fields are not actor inputs, so this is not currently an actor leakage finding. The issue is semantic: the environment still frames episodes and stats around a target-goal task while training a measurement-space velocity governor.

Recommendation: split task semantics explicitly:

- `legacy_navrl_goal`: target-pos navigation and `reach_goal`.
- `instinctrl_command_governor`: `v_cmd` tracking, null-command anchor hold, safety/intervention metrics.

Do not use one task's success metric to judge the other.

### F-003: Termination Reasons Are Not Auditable

`terminated = below_bound | above_bound | collision`, but only `collision` and `truncated` are logged. The 8M report shows `collision=0`, `truncated=0`, and `reach_goal=0`; because below-bound and above-bound are missing, the actual failure mode is hidden.

Recommendation: add explicit stats for:

- `terminated_below_bound`
- `terminated_above_bound`
- `terminated_collision`
- `terminated_success_goal` only for legacy goal mode
- `terminated_timeout` / `truncated`
- `termination_reason_code`

This is a P0 prerequisite before using eval results to make training decisions.

### F-004: Current Eval Does Not Measure Handbook Required Metrics

The handbook-required metrics are station-keeping drift, range-anchor error, tracking RMSE, minimum clearance, collision rate, ICS violation rate, command preservation ratio, intervention frequency, and observability metrics. The current streaming eval collects only old episode stats, optional governor/ICS tensor summaries, and reward summaries.

Recommendation: create instinctRL-specific eval summaries:

- `tracking/rmse_v_actual_body_vs_v_cmd`
- `tracking/rmse_v_final_body_vs_v_cmd`
- `tracking/command_preservation_ratio`
- `anchor/active_fraction`
- `anchor/error_mean`, `anchor/error_max`, `anchor/loss`
- `safety/min_clearance_mean`, `safety/min_clearance_p05`, `safety/collision_rate`
- `ics/beta_mean`, `ics/intervention_frequency`, `ics/emergency_rate`, `ics/violation_rate`
- `termination/*`
- `observability/sigma_min`, `observability/rank`, when logger is enabled

### F-005: The 8M Run Was Not a Crazy/Adversarial Command-Curriculum Run

`command_generator.py` has a 5-mode command generator, including aggressive and adversarial modes. Current training did not use it. `env.py` uses simple bounded random velocity commands with low vertical scale and no command-mode logging.

This does not invalidate the 8M run as a stability artifact. It does invalidate any claim that the run trained against realistic human operator regimes or adversarial/crazy command regimes.

Recommendation: add an explicit command config namespace, for example:

- `instinctRL.command.source=basic_random|adversarial_generator|scripted_eval`
- `instinctRL.command.mode_probabilities=[...]`
- `stats.command_mode_*`
- `stats.command_speed`, `stats.command_vertical_speed`, `stats.command_change_rate`

### F-006: ICS Is Implemented But Not Part of the Completed 8M Method Run

The handbook includes ICS-inspired command attenuation as part of the locked method. Current default config and the 8M command used `instinctRL.ics.enabled=false`. This is acceptable for a no-ICS baseline or early training stability run, but it is not the full instinctRL method.

Recommendation: label the 8M run as `no_ics` or `learned_governor_no_ics_basic_random_command`. Do not compare it as the full method against future ICS-enabled baselines.

### F-007: Reward Proxy Can Hide Physical Tracking Failure

The reward computer defaults to comparing `v_final_b` to `v_cmd_b`. That verifies the command pipeline's chosen output, not necessarily actual achieved body-frame velocity. This is actor-clean and useful, but it is insufficient as the only performance signal.

Recommendation: keep command-proxy reward if needed for actor-clean training, but eval must compute privileged actual velocity tracking metrics. If training remains unstable or reward hacking is suspected, consider a reward-only privileged actual-velocity option as an explicitly labeled variant.

### F-008: Critic Privileged Fields Still Encode Legacy Target Semantics

The critic uses `drone_state`, `target_rpos`, and `target_distance`. Privileged critic inputs are allowed by the handbook, but these particular target fields belong to the legacy goal-navigation task. If target position is unrelated to command-governor objectives, the critic can learn a value function over irrelevant task variables.

Recommendation: replace critic target fields in instinctRL mode with privileged fields aligned to the command-governor task, such as actual body velocity, min clearance, altitude/bounds, collision/termination reason, and optionally command-regime metadata. Keep actor leakage tests.

### F-009: Development Records Are Stale Relative to Runtime Evidence

`DEV_STATUS.md`, `DEFERRED_REGISTER.md`, and `TEST_PLAN.md` still record A2-S 1M runtime acceptance as pending due to Isaac/Nucleus assets. Later user-side runs passed 1.31M and 8M frame counts and produced eval artifacts.

Recommendation: update devlog after this audit is reviewed. Do not rewrite history; add a dated status entry that says:

- numerical stability acceptance passed later by user-side run;
- task/eval semantic audit found P0 blockers;
- training convergence and handbook performance remain unproven;
- next phase is task/eval conformance repair, not more blind training.

### F-010: Tests Cover Components, Not End-to-End Task Semantics

Current tests cover governor bounds, actor audit, PPO stability, anchor, ICS, and reward units. They do not yet assert that instinctRL eval uses handbook metrics, that command regimes are wired, or that legacy goal metrics are not mistaken for task success.

Recommendation: add end-to-end semantic tests after code fixes:

- eval summary contains required instinctRL metric keys;
- below/above-bound termination appears in stats;
- `reach_goal` is absent or marked legacy when `instinctRL.task=command_governor`;
- adversarial command generator mode stats are logged when enabled;
- null-command eval produces anchor active/error metrics;
- deterministic eval does not call PPO update.

## What Remains Valid From the 8M Run

The 8M run is not worthless. It remains useful for:

- proving the current learned-governor/PPO stability hardening can survive a long run under the basic random-command setup;
- providing checkpoints to test new eval diagnostics before retraining;
- validating that no new PPO diagnostic snapshots were emitted during the documented eval;
- serving as a no-ICS, basic-random-command baseline candidate.

It should not be used as evidence of:

- full handbook-defined instinctRL success;
- command-generator/crazy-command robustness;
- dynamic-obstacle robustness;
- station-keeping quality;
- physical velocity tracking quality;
- convergence to a deployable policy.

## Recommended Repair Sequence

### P0: Stop Training Until Task/Eval Semantics Are Fixed

Do not start another long formal run with the current primary metric setup. More frames will not resolve a metric/task mismatch.

Immediate fixes:

1. Add an explicit instinctRL task mode, e.g. `instinctRL.task=command_governor`, separate from legacy goal navigation.
2. Add termination reason stats and JSON reporting.
3. Replace `reach_goal` as the primary instinctRL eval criterion with handbook-aligned metrics.
4. Re-evaluate the 8M checkpoints with the new diagnostics before retraining.

### P1: Build a Minimal Handbook-Aligned Eval Suite

Create deterministic eval scenarios:

1. Null-command anchor hold: `v_cmd=0`, measure anchor error and drift proxy.
2. Safe constant command: nonzero `v_cmd`, measure actual velocity tracking RMSE and command preservation.
3. Unsafe command toward obstacle: measure min clearance, beta/intervention, collision, and emergency behavior.
4. Static clutter: measure safety and tracking under obstacle density.
5. Dynamic OOD only after static command-governor metrics are non-degenerate.

### P1: Wire Command Regimes Deliberately

Use `AdversarialCommandGenerator` only after the metric repair, and label runs clearly:

- `basic_random_command`: current-style random command, useful for stability.
- `adversarial_command_curriculum`: 5-mode command generator with mode probabilities and mode stats.
- `scripted_eval_command`: deterministic commands for eval reproducibility.

### P1: Revisit Critic Privileged Inputs

In instinctRL mode, remove or isolate target-goal critic features unless running a legacy NavRL baseline. Prefer privileged critic fields aligned with the active task.

### P2: Update Devlog and Test Plan

After the P0/P1 fixes are implemented, update:

- `DEV_STATUS.md`
- `DEFERRED_REGISTER.md`
- `TEST_PLAN.md`
- `DECISION_LOG.md`
- a new test report for task/eval conformance

The update should say that numerical stability and task correctness are separate axes.

## Go/No-Go

Current state:

- Numerical long-run stability: likely yes for the basic random-command no-ICS setup, based on the 8M completion.
- Formal handbook-aligned training success: no evidence yet.
- Current eval fit for decision-making: no.
- Next action: repair task/eval semantics and re-evaluate the existing 8M checkpoints before new training.

The only honest conclusion is: pause more long training, fix the audit findings, then run a handbook-aligned eval on existing checkpoints. If those diagnostics are poor or reveal reward hacking, start a new training run under the corrected task/eval setup.
