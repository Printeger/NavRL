# instinctRL 8M Learned-Governor Eval Report

Date: 2026-07-07  
Repo: `isaac-training`  
Commit: `ecbbe0d`

## Setup

Preflight:

- `python -m py_compile training/scripts/eval.py training/scripts/utils.py`: pass.
- `git status --short`: dirty before eval. Existing modified files included docs, `training/scripts/env.py`, `training/scripts/ppo.py`, `training/scripts/train.py`, `training/scripts/utils.py`, and related unit tests. These pre-existing changes were not reverted.
- `ppo_diagnostics/` before eval contained:
  - `ppo_stability_1783349125631_nonfinite_value_norm_running_mean_sq.pt`
  - `ppo_stability_1783361100343_nonfinite_denormalized_value.pt`
- `ppo_diagnostics/` after eval: no new files.

Eval script changes used for this report:

- `training/cfg/eval.yaml` now requires explicit `checkpoint_path` and supports optional `result_path`.
- `training/scripts/eval.py` fails fast if `checkpoint_path` is empty, prints the resolved checkpoint path, and loads only that path.
- Learned-governor eval uses the train-equivalent path: PPO learned governor output in body frame, optional ICS attenuation, body-to-world adapter, then `VelController`.
- Eval uses deterministic `ExplorationType.MEAN`.
- No PPO update or `policy.train(data)` is called.
- Headless artifact eval uses a streaming eval path so MID360 LiDAR histories are not retained for every timestep.
- JSON summaries are printed to stdout and saved beside the logs.

Common eval settings:

- Static obstacles: `env.num_obstacles=350`
- Static sweep envs: `env.num_envs=64`
- Max episode length: `env.max_episode_length=2200`
- Frame cap argument: `max_frame_num=4096` for sweep/OOD, `1024` for sanity
- Headless: `true`
- WandB: `offline`
- instinctRL mode: `train`
- ICS: disabled, so `ics_beta`, `ics_command_speed`, and `ics_final_speed` are absent by design.

## Commands

Sanity command:

```bash
python training/scripts/eval.py \
  checkpoint_path=wandb/offline-run-20260707_114333-piys3ctl/files/checkpoint_final.pt \
  result_path=../docs/instinctRL_devlog/tests/artifacts/20260707_215806_sanity_checkpoint_final.json \
  env.num_envs=16 env.num_obstacles=350 env_dyn.num_obstacles=0 \
  env.max_episode_length=2200 max_frame_num=1024 \
  headless=true wandb.mode=offline instinctRL.mode=train
```

Static sweep command shape:

```bash
python training/scripts/eval.py \
  checkpoint_path=wandb/offline-run-20260707_114333-piys3ctl/files/<checkpoint>.pt \
  result_path=../docs/instinctRL_devlog/tests/artifacts/<timestamp>_static_<checkpoint>.json \
  env.num_envs=64 env.num_obstacles=350 env_dyn.num_obstacles=0 \
  env.max_episode_length=2200 max_frame_num=4096 \
  headless=true wandb.mode=offline instinctRL.mode=train
```

Dynamic OOD command:

```bash
python training/scripts/eval.py \
  checkpoint_path=wandb/offline-run-20260707_114333-piys3ctl/files/checkpoint_8000.pt \
  result_path=../docs/instinctRL_devlog/tests/artifacts/20260707_220048_dynamic_ood_checkpoint_8000.json \
  env.num_envs=64 env.num_obstacles=350 env_dyn.num_obstacles=20 \
  env.max_episode_length=2200 max_frame_num=4096 \
  headless=true wandb.mode=offline instinctRL.mode=train
```

Note: the dynamic command requested `env_dyn.num_obstacles=20`, but the resolved runtime summary and Isaac log reported `16` dynamic rigid bodies.

## Sanity Eval

The first sanity attempt using TorchRL `rollout()` failed with CUDA OOM because it retained full MID360 LiDAR history tensors for each timestep. The streaming eval path was added after that failure. Failed log:

- `artifacts/20260707_215545_sanity_checkpoint_final.log`

The rerun passed and formal eval continued.

| checkpoint | envs | collision | reach_goal | episode_len | return | reward_total | episodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint_final.pt` | 16 | 0.000 | 0.000 | 573.438 | -369.315 | -369.315 | 16/16 |

Artifacts:

- `artifacts/20260707_215806_sanity_checkpoint_final.log`
- `artifacts/20260707_215806_sanity_checkpoint_final.json`

## Static Sweep

| checkpoint | collision | reach_goal | episode_len | return | reward_total | truncated | episodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint_4000.pt` | 0.000 | 0.000 | 351.250 | -270.986 | -270.986 | 0.000 | 64/64 |
| `checkpoint_6000.pt` | 0.000 | 0.000 | 741.391 | -481.584 | -481.584 | 0.000 | 64/64 |
| `checkpoint_8000.pt` | 0.000 | 0.000 | 636.484 | -358.892 | -358.892 | 0.000 | 64/64 |
| `checkpoint_final.pt` | 0.000 | 0.000 | 644.469 | -415.839 | -415.839 | 0.000 | 64/64 |

Artifacts:

- `artifacts/20260707_215835_static_checkpoint_4000.log`
- `artifacts/20260707_215835_static_checkpoint_4000.json`
- `artifacts/20260707_215852_static_checkpoint_6000.log`
- `artifacts/20260707_215852_static_checkpoint_6000.json`
- `artifacts/20260707_215917_static_checkpoint_8000.log`
- `artifacts/20260707_215917_static_checkpoint_8000.json`
- `artifacts/20260707_215949_static_checkpoint_final.log`
- `artifacts/20260707_215949_static_checkpoint_final.json`

## Best Checkpoint

Selected static-best: `checkpoint_8000.pt`.

Rationale:

- All static checkpoints had equal collision and reach-goal rates: `collision=0.0`, `reach_goal=0.0`.
- `checkpoint_4000.pt` had the highest return, but also the shortest mean episode length and no successes, so the return is likely inflated by earlier non-goal termination rather than better navigation.
- `checkpoint_8000.pt` had a more reasonable episode length than `checkpoint_4000.pt` and better return than `checkpoint_6000.pt` and `checkpoint_final.pt`.

This is a best available checkpoint, not a successful converged policy.

## Dynamic OOD Eval

Dynamic obstacles are out-of-distribution for this run because training/eval used `env_dyn.num_obstacles=0`. The OOD command requested 20 dynamic obstacles; runtime resolved to 16.

| checkpoint | setting | dynamic obstacles | collision | reach_goal | episode_len | return | reward_total | episodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `checkpoint_8000.pt` | static best | 0 | 0.000 | 0.000 | 636.484 | -358.892 | -358.892 | 64/64 |
| `checkpoint_8000.pt` | dynamic OOD | 16 reported | 0.000 | 0.000 | 645.203 | -374.265 | -374.265 | 64/64 |

Artifacts:

- `artifacts/20260707_220048_dynamic_ood_checkpoint_8000.log`
- `artifacts/20260707_220048_dynamic_ood_checkpoint_8000.json`

## Conclusion

- 8M learned-governor training completed: yes, assuming the provided `offline-run-20260707_114333-piys3ctl` checkpoint set is the intended completed run.
- Eval completed: yes, after replacing memory-heavy rollout collection with streaming eval for artifact runs.
- Best checkpoint selected: yes, `checkpoint_8000.pt`, with the caveat that selection is among failing policies.
- Policy convergence/performance success: no. Static reach-goal rate is `0.0` across all evaluated checkpoints.
- Dynamic obstacle robustness: not demonstrated. Dynamic OOD reach-goal rate is also `0.0`; return is slightly worse than static for the selected checkpoint.

## Next Steps

1. Diagnose the non-goal termination mode, since collision and truncation are both zero while reach-goal is zero.
2. Add explicit eval stats for out-of-bounds / below-bound / above-bound termination so return is not used as a proxy for failure type.
3. Revisit reward sign/scale: returns are dominated by negative tracking and safety terms.
4. Keep streaming eval for MID360 runs; full TorchRL rollout is not memory-safe at this observation size.
5. Only evaluate dynamic obstacles as robustness after static reach-goal becomes nonzero.
