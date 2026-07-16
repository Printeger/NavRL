# instinctRL Next-Step Prompt

> **Updated**: 2026-07-16
>
> **Current step**: A2-R5J clean-commit CUDA decision

## Task

R5J default-off ICS behavior and provenance repair are already implemented.
`8298a7d` is pushed to `origin/a2-r5j-default-off-residual`. Do not modify the
implementation, wrapper, or comparator. Work only in this order:

1. Synchronize the status documents and create/push a clean documentation commit.
2. From its empty worktree, run raw `nvidia-smi` and NavRL Python
   `torch.cuda.is_available()` checks and record their exact outputs and exits.
3. Only when CUDA is ready, invoke the stored wrapper exactly once for the
   disabled `r5g_downatten_z010` replay. It alone creates a unique result path
   and appends `instinctRL.ics.residual_preemption_enabled=false`; do not modify
   or reuse any JSON. When CUDA is not ready, do not invoke it or create an
   attempts directory.
4. Record final `GO (design only)` only after strict provenance, CUDA, eval,
   freshness, eight exact-zero diagnostic, legacy-JSON, and gate-equality
   checks pass; otherwise record `HOLD`. Never execute an enabled replay,
   dry-run, sweep, training, warm-start, or promotion in this task.

## Required provenance contract

- Verify the stored absolute checkpoint path and expected SHA-256 before eval;
  do not rewrite a worker-specific checkpoint path.
- Before creating an attempt directory, require
  `git status --porcelain=v1 --untracked-files=all` to be empty and verified
  `HEAD` to resolve consistently as `source_commit` and `commit`. A dirty,
  unresolved, or mismatched source is a provenance-only `HOLD`.
- Verify seed `0`, unchanged stored legacy argv, unique unused attempt result
  path, CUDA readiness, subprocess exit code, and result freshness before a
  result can be compared.
- A wrapper/preflight/eval/provenance failure is unconditional `HOLD`, even if
  an old exact replay JSON exists. Preserve the actual failure in the attempt
  record and comparison report.
- Require exact legacy JSON and gates plus finite exact-zero disabled R5J
  diagnostics. A pass is only `GO (design only)`.

## Current evidence

The historical fail-closed artifact is
`tests/artifacts/r5j_default_equivalence/20260714_234801/attempts/20260716T074648884514Z-0a6a2be/`.
It was made from a dirty worktree, so its provenance cannot satisfy the repair.
CUDA did not run eval: `nvidia-smi` exited 9 because it could not communicate
with the NVIDIA driver, and `torch.cuda.is_available()` was false. No replay
JSON exists, so the record remains `HOLD`.

The default-off core remains actor-clean. Evidence-3 still lacks exact
contact-body identity, surface normals, measured deceleration, and a final
safety-fix proof.

## Boundaries and Git

Do not change actor/action, reward, hard gates, controller, TASLAB_UAV, Livox
MID360, safety-filter defaults, or the R5J residual algorithm. Commit and push
only `a2-r5j-default-off-residual`; do not create a PR or push/merge `main`.
