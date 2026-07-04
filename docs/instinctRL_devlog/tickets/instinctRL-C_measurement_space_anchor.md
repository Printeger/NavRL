# instinctRL-C: Measurement-Space Anchor Manager

> **Ticket ID**: instinctRL-C
> **Status**: COMPLETE
> **Date**: 2026-07-05
> **Dependencies**: instinctRL-B
> **Next stage**: instinctRL-D observability logger
> **Handbook**: `instinctRL_Development_Handbook_v1_1_platform_locked.tex` Measurement-Space Anchor requirements

---

## Scope

Implement an actor-clean measurement-space anchor manager over MID360 `r_t`, `m_t`, `w_t`, and body-frame `v_cmd`.

This ticket includes:

- Anchor lifecycle: null-command hysteresis, capture, reset, selected-env reset.
- Masked measurement-space error and robust Huber helper.
- Scalar anchor diagnostics in env `info`.
- Dense internal runtime cache for later reward integration.
- Unit tests and B+C regression tests.

This ticket does not implement:

- Anchor reward integration.
- B3 ablation execution.
- Observability logger.
- ICS attenuation.
- Reward redesign.
- Training convergence.

---

## Files Changed

| File | Change |
|------|--------|
| `training/scripts/instinctRL/anchor.py` | New anchor config/state/output, manager, reset enum, Huber helper, validation, metrics/cache separation. |
| `training/scripts/env.py` | Passive C integration: construct anchor manager, reset selected envs, write scalar metrics to `info`, keep dense cache internal. |
| `training/scripts/instinctRL/__init__.py` | Mark anchor as active C module. |
| `training/cfg/train.yaml` | Add `instinctRL.anchor.*` config with canonical `min_valid_anchor_fraction`. |
| `training/unit_test/test_instinctrl_anchor.py` | Add pure PyTorch anchor lifecycle/math/validation/env-boundary tests. |

---

## Acceptance Criteria

Handbook-aligned C acceptance requires:

- Consume only MID360 `r_t`, `m_t`, `w_t`, body-frame `v_cmd`, and reset flags.
- No pose, odometry, explicit linear velocity, map/SLAM, dynamic-obstacle privileged state, or surface normals in anchor public API.
- Activate/capture when `||v_cmd|| <= eps_enter`.
- Reset on command exit when `||v_cmd|| >= eps_exit`.
- Require `eps_enter < eps_exit`.
- Capture frozen `r_star`, `m_star`, and `w_star` on activation edge.
- Reset on episode, explicit reset, command exit, or insufficient valid anchor fraction.
- Compute `anchor_error = m_t_float * m_star_float * w_t * (r_t - r_star)`.
- Provide Huber robust loss active only while anchor is active.
- Report scalar diagnostics: active flag, activation count, hold duration, valid anchor fraction, mean/max anchor error, loss, reset reason.
- Vectorize over `num_envs` and support selected-env reset.
- Preserve actor observation contract.

---

## Code Evidence

- `anchor.py` exposes `AnchorConfig`, `AnchorState`, `AnchorStepOutput`, `MeasurementSpaceAnchorManager`, and `huber_loss`.
- Reset enum is fixed: `0 none`, `1 episode`, `2 explicit`, `3 command`, `4 invalid`.
- Reset priority is fixed: `episode > explicit > command > invalid > none`.
- `anchor_activation_count` is per-episode cumulative and resets only on episode reset.
- `anchor_hold_steps` is an integer step counter and does not depend on `env.dt`.
- `m_t` and `m_star` are boolean validity masks. `w_t` and `w_star` are reliability weights.
- `w_star` gates `usable_anchor_mask` and `anchor_valid_fraction`; it does not multiply `anchor_error`.
- `anchor_loss` uses fixed structural denominator normalization.
- `AnchorStepOutput.metrics` contains only scalar `[N,1]` diagnostics. `AnchorStepOutput.cache` contains dense runtime tensors.
- `env.py` writes only `out.metrics` to `info` and stores `out.cache` in `self.anchor_outputs`.
- Actor observation remains only `lidar_grid` and `state_vec`.

---

## Tests Added

`training/unit_test/test_instinctrl_anchor.py` covers:

- Config validation and canonical config key.
- Pure Huber helper.
- Hysteresis boundary rules.
- Capture immutability.
- Masked error, loss, valid fraction, and `w_star` semantics.
- Invalid reset post-transition metrics.
- Reset priority and activation count rules.
- Structural mask denominator.
- Fail-fast validation and weight clamping.
- `AnchorStepOutput` metrics/cache split.
- Env source-level actor-contract boundary for scalar info and dense cache.

---

## Actual Validation

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_anchor.py` | Passed: `11 passed, 1 warning`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py` | Passed: `25 passed, 2 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/anchor.py training/scripts/env.py training/unit_test/test_instinctrl_anchor.py` | Passed. |
| TorchRL int64 spec probe for `UnboundedContinuousTensorSpec((1,), dtype=torch.long)` | Passed. |

---

## Remaining Deferred Work

- Anchor reward integration remains instinctRL-F/reward work.
- B3 `no_anchor` ablation remains baseline/evaluation work.
- Observability logger remains instinctRL-D.
- ICS attenuation remains instinctRL-E.
- Training convergence remains instinctRL-F or later.

---

## Final Verdict

- `instinctRL-C`: COMPLETE
- `instinctRL-D`: GO
- `instinctRL-E`: NO-GO
- `instinctRL-F`: NO-GO
