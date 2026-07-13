# instinctRL-A2 Test Report

> **Date**: 2026-07-05  
> **Verdict**: PASS for trainable-governor implementation and first formal training readiness.

---

## Commands Run

```bash
source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL
cd /home/mint/rl_dev/NavRL/isaac-training
python -m py_compile training/scripts/ppo.py training/scripts/train.py training/scripts/env.py training/scripts/instinctRL/governor.py training/scripts/instinctRL/audit.py training/scripts/utils.py
```

Result: passed.

```bash
source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL
cd /home/mint/rl_dev/NavRL/isaac-training
python -m pytest -q training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_actor_audit.py
```

Result: `13 passed, 5 warnings`.

```bash
source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL
cd /home/mint/rl_dev/NavRL/isaac-training
python -m pytest -q training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py training/unit_test/test_instinctrl_ics.py training/unit_test/test_instinctrl_rewards.py
```

Result: `64 passed, 5 warnings`.

```bash
source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL
cd /home/mint/rl_dev/NavRL/isaac-training
python training/scripts/train.py instinctRL.mode=train instinctRL.reward.enabled=true env.num_envs=4 env_dyn.num_obstacles=0 algo.training_frame_num=4 algo.num_minibatches=1 algo.training_epoch_num=1 max_frame_num=16 eval_interval=0 save_interval=0 wandb.mode=offline headless=true
```

Result: passed with exit code 0.

Runtime evidence:

- learned-governor collector wrapper enabled
- actor/schema audit passed
- rollout batch audit passed
- checkpoint audit passed
- `env_frames=16`
- final checkpoint: `wandb/offline-run-20260705_203852-35lr9uce/files/checkpoint_final.pt`

---

## Boundaries Verified

- Actor observation remains `lidar_grid + state_vec`.
- PPO learned governor action is 4D normalized.
- Critic-only privileged field perturbation does not alter actor/governor output.
- `info["v_cmd"]` is not used by PPO/governor actor path.
- Training smoke proves rollout + PPO update + reward stats + checkpoint sanity only.

---

## Final Status

- instinctRL-A2: COMPLETE
- Formal learned-governor training: superseded by A2-S stability gate; HOLD until 1M-frame numerical-stability acceptance passes
- First stable convergence run: NOT COMPLETE
- instinctRL-G baseline/evaluation: NOT COMPLETE
