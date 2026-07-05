# instinctRL-F Test Report

> **Date**: 2026-07-05
> **Ticket**: instinctRL-F Reward Integration and Training Readiness
> **Verdict**: PASS for reward integration/readiness; minimal training smoke passed

---

## Commands Run

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_rewards.py` | `10 passed, 1 warning` |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py training/unit_test/test_instinctrl_ics.py training/unit_test/test_instinctrl_rewards.py` | `54 passed, 2 warnings` |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/rewards.py training/scripts/env.py training/scripts/instinctRL/__init__.py training/unit_test/test_instinctrl_rewards.py` | Passed |
| TorchRL spec probe for reward component stats insertion | Passed |
| Sandbox CUDA availability probe | `False 0` |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/train.py training/scripts/ppo.py && python -m pytest -q training/unit_test/test_instinctrl_rewards.py training/unit_test/test_instinctrl_ppo_hybrid.py` | `12 passed, 3 warnings` |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python training/scripts/train.py instinctRL.mode=train instinctRL.reward.enabled=true env.num_envs=4 env_dyn.num_obstacles=0 algo.training_frame_num=4 algo.num_minibatches=1 algo.training_epoch_num=1 max_frame_num=16 eval_interval=0 save_interval=0 wandb.mode=offline headless=true` | Passed with exit code 0; logged `env_frames=16`; wrote final checkpoint to `wandb/offline-run-20260705_191435-pyfkk0z2/files/checkpoint_final.pt` |

Warnings observed:

- `Can't initialize NVML` in this local environment.
- Existing Torch lazy-module warning from the PPO hybrid test.

Neither warning indicates an instinctRL-F reward failure.

---

## Coverage Summary

- Config validation.
- Tracking reward command-match vs mismatch.
- Beta/emergency tracking gate and ICS compliance offset.
- Anchor active/inactive/loss/valid-fraction behavior.
- MID360 clearance safety penalty and invalid-clearance handling.
- Intervention, smoothness, and collision terms.
- Clipped total reward with component scaling.
- Anchor/ICS disabled defaults.
- Actor observation source contract.
- Privileged actual velocity boundary.
- Env source integration for stats logging and old reward fallback.

---

## Runtime Smoke Note

A minimal GPU training smoke was run and passed after train-path follow-up fixes. The accepted smoke disables periodic evaluation and periodic checkpointing with `eval_interval=0 save_interval=0`; the final checkpoint remains written. This avoids step-0 evaluation video memory pressure while validating rollout collection, PPO update, reward stats, wandb offline logging, and final checkpoint write.

This smoke does not prove policy convergence and does not implement a learned governor head.

---

## Final Result

- `instinctRL-F`: PASS / COMPLETE for reward integration and minimal smoke readiness.
- Training convergence: NOT PROVEN.
