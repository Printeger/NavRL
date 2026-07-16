# instinctRL Next-Step Prompt

> **Updated**: 2026-07-16
>
> **Current step**: A2-R5J CUDA HOLD; no runtime replay was created

## Task

R5J default-off ICS behavior and provenance repair are already implemented.
`8298a7d` and clean synchronization `c2e8367` are pushed to
`origin/a2-r5j-default-off-residual`. Fresh CUDA is unavailable: `nvidia-smi`
exited `9`; NavRL Python exited `0` with `torch.cuda.is_available() = False`
and device count `0`. The wrapper was not invoked, and no attempts directory
or replay JSON was created. This turn is final `HOLD`.

For a future separately authorized turn: start from a new clean, pushed
documentation state; record raw CUDA output and exits; and only if CUDA is
ready invoke the stored disabled wrapper exactly once. It alone creates a
unique result path and appends `instinctRL.ics.residual_preemption_enabled=false`.
Do not modify/reuse JSON. A non-ready CUDA result must again stop without a
wrapper invocation. `GO (design only)` still requires strict provenance, CUDA,
eval, freshness, eight exact-zero diagnostics, legacy JSON, and gate equality.

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
