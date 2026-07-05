# instinctRL-F Test Report

> **Date**: 2026-07-05
> **Ticket**: instinctRL-F Reward Integration and Training Readiness
> **Verdict**: PASS for reward integration/readiness

---

## Commands Run

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_rewards.py` | `10 passed, 1 warning` |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py training/unit_test/test_instinctrl_ics.py training/unit_test/test_instinctrl_rewards.py` | `54 passed, 2 warnings` |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/rewards.py training/scripts/env.py training/scripts/instinctRL/__init__.py training/unit_test/test_instinctrl_rewards.py` | Passed |
| TorchRL spec probe for reward component stats insertion | Passed |
| CUDA availability probe | `False 0` |

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

No Isaac GPU runtime smoke was run for instinctRL-F here. CUDA/NVML is not visible locally, so the optional command with `instinctRL.reward.enabled=true` is skipped in this report. A GPU-side smoke should be run later for live RayCaster/controller/reward-stat execution.

---

## Final Result

- `instinctRL-F`: PASS / COMPLETE for reward integration.
- Training convergence: NOT PROVEN.
