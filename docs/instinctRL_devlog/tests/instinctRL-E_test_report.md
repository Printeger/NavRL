# instinctRL-E Test Report

> **Date**: 2026-07-05
> **Ticket**: instinctRL-E ICS-Inspired Attenuation
> **Verdict**: PASS

---

## Commands Run

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_ics.py` | `10 passed, 1 warning` |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py training/unit_test/test_instinctrl_ics.py` | `44 passed, 2 warnings` |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/ics.py training/scripts/instinctRL/observation.py training/scripts/env.py training/scripts/train.py training/unit_test/test_instinctrl_ics.py` | Passed |

Warnings observed:

- `Can't initialize NVML` in this local environment.
- Existing Torch lazy-module warning from the PPO hybrid test.

Neither warning indicates an instinctRL-E attenuation failure.

---

## Coverage Summary

- Config validation and `brake_mode="zero"` enforcement.
- History/ray/command shape validation.
- Empty active set behavior.
- Emergency reliable-min-clearance bypass.
- Beta monotonicity with clearance and speed.
- Active-set inclusion/exclusion rules and ratio clamp.
- Finite-difference range-rate cache and optional range-rate beta influence.
- Command clipping after beta computation from the unclipped command.
- Builder/env history accessors and copy semantics.
- Source-level actor contract and train action-path ordering.

---

## Runtime Smoke Note

No Isaac GPU runtime smoke was run for instinctRL-E here. CUDA/NVML is not visible locally, so the optional command with `instinctRL.ics.enabled=true` is skipped in this report. A GPU-side smoke should be run later for live RayCaster/controller execution.

---

## Scope Boundary

- No reward/training implementation was added.
- No actor observation schema change was added.
- The E deployed path does not use surface normals, map, odometry, SLAM, pose, or dynamic-obstacle privileged state.

---

## Final Result

`instinctRL-E`: PASS / COMPLETE
