# instinctRL-D: Observability Logger

> **Ticket ID**: instinctRL-D
> **Status**: COMPLETE
> **Date**: 2026-07-05
> **Dependencies**: instinctRL-C
> **Next stage**: instinctRL-E ICS-inspired attenuation
> **Handbook**: `instinctRL_Development_Handbook_v1_1_platform_locked.tex` Observability Logger requirements

---

## Scope

Implement an evaluation-only range-Jacobian observability logger for MID360 measurement space.

This ticket includes:

- Offline finite-difference Jacobian estimator.
- Offline surface-normal geometric approximation.
- Deployed-safe scan-geometry proxy mode labeled as proxy.
- SVD rank/sigma/condition metrics.
- Weak-direction cache and scalar drift projection diagnostics.
- Passive env integration for scalar `info` metrics when enabled.
- Unit tests and A/B/C/D regression tests.

This ticket does not implement:

- instinctRL-E ICS attenuation.
- instinctRL-F reward/training.
- Anchor reward or B3 ablation.
- Plot/report generation.
- Actor observability features.

---

## Files Changed

| File | Change |
|------|--------|
| `training/scripts/instinctRL/observability.py` | New config/output/logger module for proxy, normal, and finite-difference observability. |
| `training/scripts/env.py` | Passive D integration: save MID360 ray directions, instantiate logger when enabled, write scalar metrics to `info`, keep dense cache internal. |
| `training/scripts/instinctRL/__init__.py` | Mark observability as active D module. |
| `training/cfg/train.yaml` | Add `instinctRL.observability.*` config, disabled by default. |
| `training/unit_test/test_instinctrl_observability.py` | Add pure PyTorch observability math/API/boundary tests. |

---

## Acceptance Criteria

Handbook-aligned D acceptance requires:

- Observability logger is evaluation/analysis only, not deployed control dependency.
- Surface normals are allowed only in offline/evaluation logger paths.
- Hardware/proxy path requires only deployed-safe quantities and is labeled proxy.
- Compute weighted range-Jacobian observability metrics.
- Use robust SVD, configurable rank tolerance, finite condition-number cap, and safe degenerate outputs.
- Provide scalar metrics for rank, singular values, condition number, score, valid fractions, drift projection/norm, proxy flag, and mode code.
- Keep dense J/SVD/weak-direction internals out of actor observation.
- Preserve actor input contract.

---

## Code Evidence

- `ObservabilityConfig` validates mode, rank tolerance, condition cap, valid fraction, log interval, and epsilon.
- `RangeJacobianObservabilityLogger.compute()` accepts flattened MID360 ray geometry, masks, weights, optional normals, optional finite-difference samples, optional drift, and optional scenario id.
- Proxy mode uses `J_i=-normalized_ray_direction_i` and sets `observability_is_proxy=1`, `mode_code=0`.
- Normal mode uses `J_i=-n_i`, normalizes normals, and applies `sqrt(w)` downstream row scaling.
- Finite-difference mode estimates `J` with `pinv(fd_perturbations_b) @ fd_range_delta` and preserves measured sign.
- SVD is per env and uses safe rank-0 degenerate outputs for insufficient rows.
- `ObservabilityOutput.metrics` contains only scalar `[N,1]` public diagnostics.
- `ObservabilityOutput.cache` contains dense Jacobian rows, weighted rows, singular values, weak direction, and effective row mask.
- `env.py` writes only `out.metrics` to `info` and stores `out.cache` in `self.observability_outputs`.
- Actor observation remains only `lidar_grid` and `state_vec`.

---

## Tests Added

`training/unit_test/test_instinctrl_observability.py` covers:

- Config validation.
- Proxy row construction and proxy labels.
- Normal-mode row construction and `sqrt(w)` scaling.
- Finite-difference exact and overdetermined recovery.
- Mode precedence and malformed-input behavior.
- SVD rank/degeneracy/condition behavior.
- Drift projection and weak-direction cache boundary.
- Public metrics shape/dtype and cache-only dense internals.
- Source-level actor-contract boundary.

---

## Actual Validation

| Command | Result |
|---------|--------|
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observability.py` | Passed: `9 passed, 1 warning`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m pytest -q training/unit_test/test_instinctrl_observation.py training/unit_test/test_instinctrl_command_adapter.py training/unit_test/test_instinctrl_mid360_pattern.py training/unit_test/test_instinctrl_actor_audit.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_anchor.py training/unit_test/test_instinctrl_observability.py` | Passed: `34 passed, 2 warnings`. |
| `source /home/mint/miniconda3/etc/profile.d/conda.sh && conda activate NavRL && cd /home/mint/rl_dev/NavRL/isaac-training && python -m py_compile training/scripts/instinctRL/observability.py training/scripts/env.py training/unit_test/test_instinctrl_observability.py` | Passed. |
| TorchRL int64 spec probe for `UnboundedContinuousTensorSpec((1,), dtype=torch.long)` | Passed. |

---

## Runtime Smoke

No Isaac GPU runtime smoke was run for instinctRL-D in this pass. D acceptance is based on the pure logger tests, passive env source-level actor-contract test, and A/B/C/D regression suite. A later Isaac runtime smoke can verify live `observability_*` `info` metrics when `instinctRL.observability.enabled=true`.

---

## Remaining Deferred Work

- Plot generation for drift versus `sigma_min`, per-scenario drift ranking, and weak-direction drift alignment.
- instinctRL-E ICS attenuation.
- instinctRL-F reward integration and training.
- Anchor reward and B3 ablation.

---

## Final Verdict

- `instinctRL-D`: COMPLETE
- `instinctRL-E`: GO
- `instinctRL-F`: NO-GO
