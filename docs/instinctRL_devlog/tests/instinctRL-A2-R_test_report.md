# instinctRL-A2-R Test Report

**Date**: 2026-07-09  
**Scope**: station objective repair source/unit validation.

## Results

| Command | Result |
|---|---|
| `python -m pytest -q training/unit_test/test_instinctrl_rewards.py training/unit_test/test_instinctrl_task_metrics.py` | Passed: `20 passed`. |
| `python -m pytest -q training/unit_test/test_instinctrl_rewards.py training/unit_test/test_instinctrl_task_metrics.py training/unit_test/test_instinctrl_eval_diagnostic.py` | Passed: `24 passed`. |
| `python -m py_compile training/scripts/instinctRL/rewards.py training/scripts/instinctRL/task_metrics.py training/scripts/env.py training/scripts/utils.py training/scripts/eval.py` | Passed. |
| `python -m pytest -q training/unit_test/test_instinctrl_rewards.py training/unit_test/test_instinctrl_task_metrics.py training/unit_test/test_instinctrl_eval_diagnostic.py training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_actor_audit.py` | Passed: `33 passed`. |
| `python -m pytest -q training/unit_test/test_instinctrl_*.py` | Passed: `92 passed, 11 warnings`. |
| `python training/scripts/train.py instinctRL.mode=train instinctRL.task=command_governor instinctRL.reward.enabled=true instinctRL.reward.use_privileged_velocity_for_reward=true instinctRL.ics.enabled=true instinctRL.command.source=curriculum_generator instinctRL.command.curriculum_profile=station_first env.num_envs=4 env.num_obstacles=20 env_dyn.num_obstacles=0 algo.training_frame_num=4 algo.num_minibatches=1 algo.training_epoch_num=1 max_frame_num=16 eval_interval=0 save_interval=0 wandb.mode=offline wandb.name=instinctrl_a2r_smoke headless=true` | Passed with exit code 0; rollout/checkpoint audits passed; final smoke checkpoint at `wandb/offline-run-20260709_185417-f0e3rq1j/files/checkpoint_final.pt`. |

## Pending Runtime Evidence

- Run a new 1M static MID360 diagnostic retrain with `instinctRL.command.curriculum_profile=station_first`.
- Run short diagnostic eval and compare against the A2-R go/no-go gate.
