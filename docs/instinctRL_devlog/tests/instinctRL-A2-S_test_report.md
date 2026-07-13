# instinctRL-A2-S Test Report

> Date: 2026-07-05  
> Scope: PPO numerical-stability hardening for learned-governor training.

## Commands Run

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && python -m pytest -q training/unit_test/test_instinctrl_ppo_hybrid.py` | Passed: `4 passed, 5 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && python -m pytest -q training/unit_test/test_instinctrl_ppo_stability.py` | Passed: `9 passed, 8 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && python -m py_compile training/scripts/ppo.py training/scripts/utils.py training/scripts/instinctRL/ppo_stability.py training/scripts/instinctRL/governor.py` | Passed. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && python -m pytest -q training/unit_test/test_instinctrl_ppo_stability.py training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_actor_audit.py` | Passed: `22 passed, 12 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && python -m pytest -q training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_ppo_stability.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py training/unit_test/test_instinctrl_ics.py training/unit_test/test_instinctrl_rewards.py` | Passed: `73 passed, 12 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && python training/scripts/train.py instinctRL.mode=train instinctRL.reward.enabled=true env.num_envs=4 env_dyn.num_obstacles=0 algo.training_frame_num=4 algo.num_minibatches=1 algo.training_epoch_num=1 max_frame_num=16 eval_interval=0 save_interval=0 wandb.mode=offline headless=true` | Failed before env import: missing Omniverse/Nucleus assets root. |

## Coverage

- Bounded Beta params remain finite and within configured bounds.
- Finite observations produce finite normalized actions.
- NaN actor raw output is caught before the governor decoder and writes a diagnostic snapshot.
- Non-finite gradients and parameters are caught and write diagnostic snapshots.
- Zero-std advantage normalization stays finite.
- Gradient clipping covers actor, critic, actor feature extractor, and critic feature extractor.
- Target-KL early stop triggers.
- Source-level test confirms no NaN-to-zero action replacement.

## Runtime Blocker

The runtime smoke did not reach PPO or the environment. It failed during `env.py` import through Orbit asset resolution:

`RuntimeError: Unable to perform Nucleus login on Omniverse. Assets root path is not set.`

This blocks the 1M-frame acceptance run. It is not evidence of a PPO stability failure.

## Verdict

- Source/unit test readiness: PASS.
- Runtime 1M-frame stability acceptance: PENDING.
- Formal long learned-governor training: HOLD.

