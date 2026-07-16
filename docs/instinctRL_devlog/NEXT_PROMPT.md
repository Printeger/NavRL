# instinctRL Next-Step Prompt

> **Updated**: 2026-07-17
>
> **Current step**: A2-R5J disabled default-equivalence replay passed — GO (design only)

## Task

R5J default-off ICS behavior and provenance repair are already implemented.
`8298a7d256bec6a82dee49d9af41a87628135ed6` is the provenance repair;
`927e166` was the incoming pushed baseline, and documentation synchronization
`5c9ab7a71365fb3899d586b09ba6ea8e231aa80e` was pushed before the decision.
From that clean source commit, raw `nvidia-smi` exited `0` and the specified
NavRL Torch command printed `True` with exit `0`.

The one authorized, argument-free stored disabled wrapper ran once and created
`attempts/20260716T161710730878Z-5c9ab7a/`. Its record has matching branch and
source/compatibility commit, empty pre-attempt porcelain, verified checkpoint
SHA-256, seed `0`, unchanged legacy argv, CUDA ready, eval exit `0`, and a
fresh matching result. `comparison.json` records exact legacy JSON and gate
equality plus all eight required disabled R5J diagnostic summaries as finite
exact zero. It returned `GO (design only)`.

Do not rerun, reuse, or modify this replay or JSON. The artifact's stderr
contains the captured Isaac/W&B shutdown segfault trace even though its eval
subprocess returned `0` and the wrapper/comparator returned `GO (design only)`;
preserve that evidence. This conclusion authorizes only later design work. Do
not execute an enabled replay, dry-run, sweep, training, warm-start, or
promotion without separate authorization.

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
At that time CUDA did not run eval: `nvidia-smi` exited 9 because it could not
communicate with the NVIDIA driver, and `torch.cuda.is_available()` was false.
No replay JSON exists, so that historical record remains `HOLD` and the single
clean replay is unconsumed.

The default-off core remains actor-clean. Evidence-3 still lacks exact
contact-body identity, surface normals, measured deceleration, and a final
safety-fix proof.

## Boundaries and Git

Do not change actor/action, reward, hard gates, controller, TASLAB_UAV, Livox
MID360, safety-filter defaults, or the R5J residual algorithm. Commit and push
only `a2-r5j-default-off-residual`; do not create a PR or push/merge `main`.
