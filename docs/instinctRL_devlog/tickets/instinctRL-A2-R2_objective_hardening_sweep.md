# instinctRL-A2-R2 Objective Hardening and Short Sweep Gate

**Status**: source/unit complete; superseded by A2-R3 after failed runtime sweep  
**Created**: 2026-07-10  
**Decision authority**: D-2026-07-10-001

> Supersession note (2026-07-11): the `20260711_111713` A2-R2 sweep failed all six candidates. `r2_balanced` is reference evidence only and must not be promoted or used as a warm start. Continue with `instinctRL-A2-R3_station_correction_repair.md`.

## Problem

Recent 1M/2M diagnostic runs produced inconsistent outcomes: collision could be zero while station drift, null-command motion, clearance p05, ICS violation, and command amplification remained outside handbook intent. Continuing manual 1M/2M trials is low signal because it changes training scale before fixing the objective and selection method.

## Implemented Changes

- Added decoder-level null-command prior in `TrainableGovernorDecoder`.
- Added `null_vcorr_gate_enabled`, `null_vcorr_gate_eps`, and `null_vcorr_gate_min` PPO config defaults.
- Added reward terms:
  - `reward_preservation_low`
  - `reward_preservation_high`
- Added reward config:
  - `preservation_low_weight`
  - `preservation_high_weight`
  - `preservation_lower`
  - `preservation_upper`
- Gated preservation penalties by safe command state, so ICS attenuation/emergency can reduce speed without being punished by preservation-low.
- Updated env reward stats to derive from `REWARD_COMPONENT_KEYS`.
- Added hard gate scorer: `training/scripts/instinctRL/gates.py`.
- Added dry-run-first sweep runner: `training/scripts/instinctRL/sweep.py`.

## Validation

```bash
source /home/mint/miniconda3/etc/profile.d/conda.sh
conda activate NavRL
cd /home/mint/rl_dev/NavRL/isaac-training
python -m py_compile \
  training/scripts/instinctRL/governor.py \
  training/scripts/instinctRL/rewards.py \
  training/scripts/instinctRL/gates.py \
  training/scripts/instinctRL/sweep.py \
  training/scripts/ppo.py \
  training/scripts/env.py
python -m pytest -q \
  training/unit_test/test_instinctrl_governor.py \
  training/unit_test/test_instinctrl_rewards.py \
  training/unit_test/test_instinctrl_gates.py
```

Result: `25 passed`.

Full instinctRL regression:

```bash
python -m pytest -q training/unit_test/test_instinctrl_*.py
```

Result: `99 passed, 11 warnings`.

## Next Runtime Procedure

1. Review generated short-sweep commands:

```bash
python training/scripts/instinctRL/sweep.py \
  --frames 131072 \
  --seeds 0 \
  --limit 6
```

2. Execute the short sweep only after dry-run review:

```bash
python training/scripts/instinctRL/sweep.py \
  --execute \
  --frames 131072 \
  --seeds 0 \
  --limit 6
```

3. Promote only top 2-3 candidates by hard-gate score to 1M.

4. Promote to formal training only after 1M candidates pass hard gates across multiple seeds.

## Hard Gate Summary

- Station drift mean/p95 must pass.
- Null-command actual and output speeds had to pass in A2-R2. A2-R3 keeps actual null-command speed as a hard station gate but treats null-command output speed as diagnostic-only so anchor-aware station correction is allowed.
- Station anchor error mean must pass.
- Tracking RMSE must pass.
- Command preservation must stay in `[0.75, 1.05]`.
- Amplification mean/rate must pass.
- Collision alone is insufficient; clearance p05 and ICS violation rate are hard safety gates.
- Below/above/collision terminations must remain zero.

## Scope Boundary

This ticket is not convergence evidence and not permission for formal training. It only makes the objective and candidate-selection process less ad hoc.
