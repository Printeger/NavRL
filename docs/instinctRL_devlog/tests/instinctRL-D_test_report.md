# instinctRL-D Test Report

> **Date**: 2026-07-05
> **Ticket**: instinctRL-D Observability Logger
> **Verdict**: PASS

---

## Commands Run

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observability.py` | `9 passed, 1 warning` |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py` | `34 passed, 2 warnings` |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/observability.py training/scripts/env.py training/unit_test/test_instinctrl_observability.py` | Passed |
| TorchRL int64 spec probe for `observability_mode_code` | Passed |

Warnings observed:

- `Can't initialize NVML` in this local environment.
- Torch lazy-module warning from the existing PPO hybrid test.

Neither warning indicates an instinctRL-D observability failure.

---

## Coverage Summary

- Config validation.
- Proxy, normal, and finite-difference Jacobian paths.
- Mode precedence and malformed-input failure.
- Weighted validity and `sqrt(w)` row scaling.
- SVD rank, singular values, condition cap, score, and degenerate safe outputs.
- Cache-only weak direction and dense Jacobian internals.
- Drift projection/norm scalar diagnostics.
- Source-level actor contract for env integration.

---

## Runtime Smoke Note

No Isaac GPU runtime smoke was run for instinctRL-D. The logger is pure PyTorch and passive in env integration. Live runtime validation can be run later with `instinctRL.observability.enabled=true`, but D acceptance does not depend on a full GPU training run.

---

## Final Result

`instinctRL-D`: PASS / COMPLETE
