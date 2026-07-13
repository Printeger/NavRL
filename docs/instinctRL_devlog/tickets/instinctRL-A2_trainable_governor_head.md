# instinctRL-A2: Trainable Governor Head

> **Date**: 2026-07-05  
> **Status**: COMPLETE for trainable-governor implementation and first formal training readiness  
> **Authority order**: code facts > handbook acceptance criteria > devlog records.

---

## Scope

Implement the learned governor actor path required before the first formal instinctRL learned-governor training run.

Implemented:

- PPO actor learned mode outputs 4D normalized governor action: `[alpha, v_corr_x, v_corr_y, v_corr_z]`.
- `alpha` is bounded in `[0,1]`.
- `v_corr` is bounded by `v_corr_limit`, default `0.5 m/s`.
- `v_gov_b = alpha * v_cmd_b + v_corr`, then norm-clipped to `velocity_limit`.
- `v_cmd_b` and previous issued body command are parsed from actor-clean `state_vec`.
- Train collector/eval wrapper applies optional ICS and body-to-world velocity adaptation at the controller boundary.
- PPO update remains on the 4D normalized governor action/log-prob.
- Policy init, rollout, checkpoint, and actor-forbidden-key audits hard-fail on violations.

Non-scope:

- No convergence claim.
- No full G baseline/evaluation matrix.
- No H real-robot deployment.
- No adversarial command curriculum requirement for the first formal training run.

---

## Files Changed

- `training/scripts/instinctRL/governor.py`
- `training/scripts/ppo.py`
- `training/scripts/train.py`
- `training/scripts/instinctRL/audit.py`
- `training/scripts/utils.py`
- `training/cfg/ppo.yaml`
- `training/unit_test/test_instinctrl_governor.py`
- `training/unit_test/test_instinctrl_ppo_hybrid.py`
- `training/unit_test/test_instinctrl_actor_audit.py`

---

## Actor Contract

Actor observation remains exactly:

- `lidar_grid`
- `state_vec`

The learned governor does not read `info["v_cmd"]`, pose, odom, map, SLAM, explicit velocity, dynamic-obstacle privileged state, or deployment-unsafe simulator state. Critic-only privileged fields remain critic-only; tests perturb them and verify actor/governor output does not change.

---

## Deferred Classification

| Item | Verdict |
|------|---------|
| D-001 Trainable Governor Head | Resolved by A2 |
| D-008 Full Audit Hooks | Resolved for first formal training requirements; ROS/H audit remains deferred |
| D-006 Aggressive / Adversarial Command Generator | Still deferred to G; not a first formal training blocker |
| D-009 Noise / Dropout Curriculum | Still deferred |
| D-010 Neighbor-Consistency Weights | Still deferred |
| D-011 Longer History Ablations | Still deferred |

---

## Validation

- A2 unit tests passed: `13 passed, 5 warnings`.
- A/B/C/D/E/F+A2 regression tests passed: `64 passed, 5 warnings`.
- GPU learned-governor train smoke passed with exit code 0:
  - rollout batch audit pass
  - checkpoint audit pass
  - `env_frames=16`
  - final checkpoint: `wandb/offline-run-20260705_203852-35lr9uce/files/checkpoint_final.pt`

---

## Verdict

- instinctRL-A2: COMPLETE
- Formal learned-governor training: superseded by A2-S stability gate; HOLD until 1M-frame numerical-stability acceptance passes
- Training convergence: NOT PROVEN
- instinctRL-G: GO for baseline/evaluation harness only
