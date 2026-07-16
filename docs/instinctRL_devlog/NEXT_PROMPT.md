# instinctRL Next-Step Prompt

> **Updated**: 2026-07-17
>
> **Current step**: A2-R5J CUDA-ready decision pending execution

## Task

R5J default-off ICS behavior and provenance repair are already implemented.
`8298a7d256bec6a82dee49d9af41a87628135ed6` is the provenance repair;
`927e166` is the current pushed baseline on
`origin/a2-r5j-default-off-residual`. A prior environmental observation found
`nvidia-smi` exit `0` and NavRL Torch CUDA available with one device. It is
informative only: replay authorization requires a new raw gate after the
documentation commit below.

Work only in this order:

1. Run the specified py_compile and pytest verification, then commit and push
   only the six R5J status documents. Confirm empty porcelain.
2. From that clean pushed commit, record the literal output and exit code of
   raw `nvidia-smi` and `/home/mint/miniconda3/envs/NavRL/bin/python -c
   "import torch; print(torch.cuda.is_available())"`.
3. Only if both raw CUDA gates pass, reconfirm empty porcelain and invoke the
   stored `replay_wrapper.py` exactly once, without arguments or JSON edits.
   It alone chooses the attempt ID/result path and appends
   `instinctRL.ics.residual_preemption_enabled=false`. A failed CUDA gate stops
   here: do not invoke the wrapper or create/reuse a replay JSON.
4. Inspect the one wrapper record, fresh result, comparison, and stdout/stderr.
   Record `GO (design only)` only after strict provenance, CUDA, eval,
   freshness, eight exact-zero diagnostics, legacy JSON, and gate equality;
   otherwise record `HOLD`. Never execute an enabled replay, dry-run, sweep,
   training, warm-start, or promotion.

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
