# instinctRL-A2-S: PPO Numerical-Stability Hardening

> Date: 2026-07-05  
> Status: SOURCE/UNIT READY; runtime 1M-frame acceptance pending  
> Authority order: code facts > handbook acceptance criteria > devlog records.

## Scope

Implement a fail-fast numerical-stability hardening patch for learned-governor PPO after a conservative long run failed around 563k frames with non-finite `("agents", "action_normalized")`.

This ticket addresses PPO actor/distribution/optimizer stability. It does not change actor observation schema, does not add privileged state to the actor, and does not claim convergence.

## Implemented

- Bounded Beta actor concentration parameters:
  - `alpha = alpha_min + (alpha_max - alpha_min) * sigmoid(raw_alpha)`
  - `beta = beta_min + (beta_max - beta_min) * sigmoid(raw_beta)`
- New config defaults:
  - `algo.max_grad_norm=0.5`
  - `algo.target_kl=0.02`
  - `algo.finite_audit=true`
  - `algo.diagnostic_dir="ppo_diagnostics"`
  - `algo.actor.beta_alpha_min=1.0`
  - `algo.actor.beta_alpha_max=30.0`
  - `algo.actor.beta_beta_min=1.0`
  - `algo.actor.beta_beta_max=30.0`
  - `algo.actor.action_eps=1e-6`
- PPO finite checks for observations, action parameters, normalized actions, log-prob, entropy, values, returns, advantages, losses, PPO ratio, gradients, and parameters.
- Advantage normalization uses `std.clamp_min(1e-6)`.
- Gradient clipping covers actor, critic, actor feature extractor, and critic feature extractor.
- Target-KL early stop skips remaining minibatches for the current update.
- Non-finite failures save compact `.pt` diagnostic snapshots and then raise.
- No NaN-to-zero action fallback is implemented.

## Files Changed

- `training/scripts/utils.py`
- `training/scripts/ppo.py`
- `training/scripts/instinctRL/ppo_stability.py`
- `training/cfg/ppo.yaml`
- `training/unit_test/test_instinctrl_ppo_stability.py`
- `CONTEXT.md`
- `docs/instinctRL_devlog/CHANGELOG.md`
- `docs/instinctRL_devlog/DEV_STATUS.md`
- `docs/instinctRL_devlog/DECISION_LOG.md`
- `docs/instinctRL_devlog/DEFERRED_REGISTER.md`
- `docs/instinctRL_devlog/TEST_PLAN.md`

## Validation

Passed:

- `python -m py_compile training/scripts/ppo.py training/scripts/utils.py training/scripts/instinctRL/ppo_stability.py training/scripts/instinctRL/governor.py`
- `python -m pytest -q training/unit_test/test_instinctrl_ppo_stability.py`
- `python -m pytest -q training/unit_test/test_instinctrl_ppo_stability.py training/unit_test/test_instinctrl_governor.py training/unit_test/test_instinctrl_ppo_hybrid.py training/unit_test/test_instinctrl_actor_audit.py`
- A/B/C/D/E/F/A2/stability regression suite: `73 passed, 12 warnings`

Runtime blocked:

- The 16-frame train smoke failed before `NavigationEnv` import because local Isaac/Orbit could not resolve Omniverse/Nucleus assets root:
  - `RuntimeError: Unable to perform Nucleus login on Omniverse. Assets root path is not set.`

## Verdict

- A2-S source/unit readiness: READY.
- Formal long learned-governor training: HOLD until 1M-frame runtime acceptance passes.
- Training convergence: NOT PROVEN.
- G baseline/evaluation harness: still GO, but learned-policy success must not be claimed from this ticket.

