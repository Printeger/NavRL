# instinctRL-A2-R3 Station Correction Repair

**Status**: source/unit complete; fresh 128k sweep pending  
**Created**: 2026-07-11  
**Decision authority**: D-2026-07-11-001

## Problem

The `20260711_111713` A2-R2 sweep cannot be promoted. All six candidates had `passed=false` and `safety_passed=false`; the best candidate, `r2_balanced`, reached only `7/15` gates. Its decoder produced zero null-command output, but station drift and anchor error remained high:

- `null_command_output_speed_mean=0.0`
- `null_command_speed_mean=0.196`
- `station_keeping_drift_mean=1.544`
- `station_keeping_drift_p95=3.003`
- `anchor_error_mean=3.243`
- `command_amplification_rate=0.318`
- `termination_below_bound=0.25`

This shows the R2 hard-zero null correction removed output bias but also prevented measurement-anchored correction. The next repair is not a stronger hard zero; it is a bounded station-correction path.

## Implemented Repair

- Changed learned-governor null-command defaults to soft correction:
  - `null_vcorr_gate_enabled=true`
  - `null_vcorr_gate_eps=0.25`
  - `null_vcorr_gate_min=0.25`
- Updated PPO fallback defaults to the same A2-R3 values.
- Changed formal train/eval reward defaults:
  - `anchor_weight=4.0`
  - `null_command_output_weight=0.1`
  - `null_output_anchor_loss_threshold=0.05`
- Made `reward_null_command_output` anchor-aware:
  - high anchor loss under an active/valid anchor relaxes the output-bias penalty;
  - low anchor loss or inactive/invalid anchor still penalizes output bias;
  - actual null-command speed remains penalized separately.
- Removed `null_command_output_speed_mean` from hard gate pass/fail while preserving the metric as a diagnostic.
- Updated A2-R3 gate thresholds for screening:
  - station drift mean `<= 1.3`
  - station drift p95 `<= 2.6`
  - anchor error mean `<= 2.0`
  - command amplification rate `<= 0.15`
- Replaced A2-R2 sweep candidates with:
  - `r3_soft_null_min025`
  - `r3_soft_null_min04`
  - `r3_anchor_strong`
  - `r3_anchor_strong_safety`
  - `r3_balanced_soft`
  - `r3_no_decoder_gate_reward_only`

## Validation

```bash
source /home/mint/miniconda3/etc/profile.d/conda.sh
conda activate NavRL
cd /home/mint/rl_dev/NavRL/isaac-training
python -m pytest -q \
  training/unit_test/test_instinctrl_governor.py \
  training/unit_test/test_instinctrl_rewards.py \
  training/unit_test/test_instinctrl_gates.py
```

Result: `27 passed`.

Full instinctRL regression:

```bash
python -m pytest -q training/unit_test/test_instinctrl_*.py
```

Result: `101 passed, 11 warnings`.

A2-R3 dry-run:

```bash
python training/scripts/instinctRL/sweep.py --frames 131072 --seeds 0 --limit 6
```

Result: emitted six `a2r3_sweep` jobs for the `r3_*` variants without launching training.

## Next Runtime Procedure

1. Dry-run the A2-R3 sweep:

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

3. Promote to 1M only if the candidate satisfies the screening gate:

- `safety_collision_rate == 0`
- `termination_collision == 0`
- `termination_below_bound == 0`
- `safety_min_clearance_p05 >= 1.0`
- `station_keeping_drift_mean < 1.3`
- `station_keeping_drift_p95 < 2.6`
- `anchor_error_mean < 2.0`
- `tracking_rmse_actual_body_vs_v_cmd <= 0.45`
- `command_amplification_rate <= 0.15`

## Scope Boundary

`r2_balanced` is diagnostic evidence only and must not be used as a 1M warm start. Formal learned-governor training remains HOLD until a fresh A2-R3 sweep selects candidates, top 1M runs pass hard gates, and multi-seed stability is checked.
