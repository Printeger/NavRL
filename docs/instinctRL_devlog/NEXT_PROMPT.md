# instinctRL Next-Step Prompt

> **Updated**: 2026-07-16
>
> **Current step**: A2-R5J disabled-replay provenance hardening and HOLD closeout

## Task

R5J default-off ICS behavior is already implemented. Do not modify it. Work only
in this order:

1. Repair the artifact-local disabled replay wrapper/comparator so any failed,
   stale, missing, or unproven attempt is fail-closed `HOLD`.
2. Run py_compile, the targeted R5J suite, and the full instinctRL suite in the
   Isaac Sim Conda environment.
3. Only if those pass, run `replay_wrapper.py` once. It must preflight CUDA and
   run exactly one `r5g_downatten_z010` replay with
   `instinctRL.ics.residual_preemption_enabled=false` only if CUDA is ready.
4. Record `GO (design only)` only after strict disabled equivalence passes.
   Never execute an enabled replay, dry-run, sweep, training, warm-start, or
   promotion in this task.

## Required provenance contract

- Verify the stored absolute checkpoint path and expected SHA-256 before eval;
  do not rewrite a worker-specific checkpoint path.
- Verify seed `0`, unchanged stored legacy argv, current branch/commit, unique
  unused attempt result path, CUDA readiness, subprocess exit code, and result
  freshness before a result can be compared.
- A wrapper/preflight/eval/provenance failure is unconditional `HOLD`, even if
  an old exact replay JSON exists. Preserve the actual failure in the attempt
  record and comparison report.
- Require exact legacy JSON and gates plus finite exact-zero disabled R5J
  diagnostics. A pass is only `GO (design only)`.

## Current evidence

The current fail-closed attempt is
`tests/artifacts/r5j_default_equivalence/20260714_234801/attempts/20260716T074648884514Z-0a6a2be/`.
Checkpoint, seed, legacy argv, branch/commit, cwd, and unique result path
validated. CUDA did not: `nvidia-smi` exited 9 because it could not communicate
with the NVIDIA driver, and `torch.cuda.is_available()` was false. Eval was not
started and no replay JSON exists, so the recorded decision is `HOLD`.

The default-off core remains actor-clean. Evidence-3 still lacks exact
contact-body identity, surface normals, and measured deceleration; it is not a
final safety-fix proof.

## Boundaries and Git

Do not change actor/action, reward, hard gates, controller, TASLAB_UAV, Livox
MID360, safety-filter defaults, or the R5J residual algorithm. Commit and push
only `a2-r5j-default-off-residual`; do not create a PR or push/merge `main`.
