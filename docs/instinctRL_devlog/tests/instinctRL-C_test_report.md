# instinctRL-C Test Report

> **Date**: 2026-07-05
> **Ticket**: instinctRL-C Measurement-Space Anchor Manager
> **Verdict**: PASS

---

## Commands Run

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_anchor.py` | `11 passed, 1 warning` |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py` | `25 passed, 2 warnings` |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/anchor.py training/scripts/env.py training/unit_test/test_instinctrl_anchor.py` | Passed |
| TorchRL int64 spec probe for `anchor_reset_reason` | Passed |

Warnings observed:

- `Can't initialize NVML` in this local environment.
- Torch lazy-module warning from the existing PPO hybrid test.

Neither warning indicates an instinctRL-C anchor test failure.

---

## Coverage Summary

- Anchor config validation, canonical key, and rejected alias.
- Reset enum and priority.
- Null-command hysteresis and boundary equality.
- Anchor capture and frozen `r_star/m_star/w_star`.
- `m_star` bool validity and separate `w_star` reliability.
- `anchor_valid_fraction` fixed structural denominator.
- Masked `anchor_error` and fixed-denominator Huber `anchor_loss`.
- Weighted residual diagnostics for `anchor_error_mean/max`.
- Post-transition reset metrics.
- Selected-env and all-env reset behavior.
- Fail-fast validation.
- `AnchorStepOutput.metrics` vs `AnchorStepOutput.cache` separation.
- Passive env integration source boundary: scalar metrics to `info`, dense cache internal, actor obs unchanged.

---

## Runtime Smoke Note

No Isaac GPU runtime smoke was required for instinctRL-C pure anchor acceptance. The env integration is passive and was validated through source-level actor-contract tests plus the existing B runtime evidence. A later Isaac smoke can verify scalar anchor metrics appear in runtime `info`, but C unit acceptance does not depend on a full GPU training run.

---

## Final Result

`instinctRL-C`: PASS / COMPLETE
