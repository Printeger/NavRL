# R5J default-off exact replay

The command below is the stored `r5g_downatten_z010` eval command, with only
`result_path` replaced and `instinctRL.ics.residual_preemption_enabled=false`
appended. The resolved `eval.yaml` seed is `0`.

`replay_wrapper.py` is the executable provenance record. Before it creates an
attempt directory, it requires a clean porcelain-v1 worktree and a consistently
verified `HEAD`; failed pre-attempt provenance produces a separate best-effort
`HOLD` record and never reaches CUDA/eval. Only after this repair is committed
and pushed and raw CUDA checks are available may it run from the repository
root. It reconstructs this argv, invokes eval with `shell=False`, captures
stdout/stderr, and writes a unique `attempts/<attempt-id>/wrapper_record.json`
plus `comparison.json`. It never reuses, overwrites, or deletes an existing
replay result.

```bash
/home/mint/miniconda3/envs/NavRL/bin/python docs/instinctRL_devlog/tests/artifacts/r5j_default_equivalence/20260714_234801/replay_wrapper.py
```

The reconstructed eval argv is identical except that `result_path` is the
unique absolute path under `attempts/<attempt-id>/`:

```bash
/home/mint/miniconda3/envs/NavRL/bin/python training/scripts/eval.py checkpoint_path=/home/mint/rl_dev/NavRL/isaac-training/wandb/offline-run-20260715_004807-durd58lw/files/checkpoint_final.pt result_path=<artifact>/attempts/<attempt-id>/r5j_r5g_downatten_z010_eval.json env.num_envs=32 env.max_episode_length=1000 env.num_obstacles=350 env_dyn.num_obstacles=0 instinctRL.eval.suite=short_diagnostic instinctRL.observability.enabled=true instinctRL.observability.mode=proxy wandb.mode=offline headless=true algo.instinctRL.governor.v_corr_limit=0.35 instinctRL.reward.preservation_high_weight=2.0 instinctRL.reward.command_amplification_weight=2.5 instinctRL.reward.proxy_tracking_weight=0.5 instinctRL.reward.safety_weight=1.2 instinctRL.reward.clearance_margin=0.4 instinctRL.ics.active_horizon_margin=1.0 instinctRL.ics.clearance_margin=0.15 instinctRL.reward.null_command_speed_weight=4.0 instinctRL.reward.height_floor=0.5 instinctRL.reward.height_floor_weight=8.0 instinctRL.reward.height_ceiling=4.0 instinctRL.reward.height_ceiling_weight=8.0 algo.instinctRL.governor.v_corr_z_limit=0.12 algo.instinctRL.governor.tracking_vcorr_z_sign_gate_enabled=true algo.instinctRL.governor.tracking_vcorr_z_gate_eps=0.001 algo.instinctRL.governor.tracking_vcorr_z_opposing_gain=1.0 algo.instinctRL.governor.tracking_vcorr_z_reinforcing_gain=0.50 instinctRL.ics.downward_attenuation_enabled=true instinctRL.ics.downward_ray_min_z=0.10 instinctRL.ics.downward_clearance_margin=0.0 instinctRL.ics.residual_preemption_enabled=false
```

`compare_disabled_replay.py` permits only top-level `result_path` and the two
explicit disabled R5J diagnostics, including flattened station/tracking
diagnostic keys. Each present R5J summary must have positive `count` and
`finite_count` and exact-zero finite statistics; both flattened and per-pass
station/tracking summaries may not be missing. Every remaining JSON field and the recomputed
`gates.py` report must match exactly.
