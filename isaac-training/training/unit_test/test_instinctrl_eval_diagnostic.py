import os


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPTS = os.path.join(ROOT, "training", "scripts")
CFG = os.path.join(ROOT, "training", "cfg")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_eval_yaml_defaults_to_static_proxy_short_diagnostic():
    source = _read(os.path.join(CFG, "eval.yaml"))

    assert 'suite: "short_diagnostic"' in source
    assert "require_static_mid360: true" in source
    assert "scenario_id: \"static_mid360_short_diag\"" in source
    assert "tracking_curriculum_frame: 600000" in source
    assert 'tracking_curriculum_profile: "diagnostic_mixed"' in source
    assert "env_dyn:\n  num_obstacles: 0" in source

    observability_block = source.split("  observability:", 1)[1].split("  ics:", 1)[0]
    assert "enabled: true" in observability_block
    assert 'mode: "proxy"' in observability_block


def test_eval_entrypoint_runs_handbook_audits_and_two_pass_suite():
    source = _read(os.path.join(SCRIPTS, "eval.py"))

    for token in [
        "check_platform_lock(cfg)",
        "require_static_mid360",
        "env_dyn.num_obstacles=0",
        "check_actor_input(td)",
        "check_actor_schema(td, cfg.instinctRL.observation.history_len)",
        "_run_short_diagnostic_eval(",
        "station_static_mid360",
        "tracking_static_mid360",
        "configure_instinctrl_eval_pass(",
        "tracking_curriculum_frame",
        "tracking_curriculum_profile",
    ]:
        assert token in source


def test_env_exposes_eval_diagnostics_without_actor_input_leakage():
    source = _read(os.path.join(SCRIPTS, "env.py"))

    for token in [
        '"station_keeping_drift"',
        '"height_world_z"',
        '"height_ceiling_margin"',
        '"command_amplification_horizontal"',
        '"command_amplification_vertical"',
        '"v_cmd_z"',
        '"v_final_b_z"',
        '"observability_scenario_id"',
        '"ics_downward_active"',
        '"ics_downward_attenuation_ratio"',
        "drift_b=station_drift_b",
        "scenario_id=scenario_id",
        "def configure_instinctrl_eval_pass(",
    ]:
        assert token in source

    actor_block = source.split("# -----------------Network Input Final--------------", 1)[1]
    actor_block = actor_block.split("# ============================================", 1)[0]
    assert '"lidar_grid": obs_hist["lidar_grid"]' in actor_block
    assert '"state_vec": obs_hist["state_vec"]' in actor_block
    for forbidden in [
        "station_keeping_drift",
        "observability_",
        "scenario_id",
        "drift",
        "root_state",
    ]:
        assert forbidden not in actor_block


def test_streaming_eval_summary_reports_missing_handbook_diagnostic_keys():
    source = _read(os.path.join(SCRIPTS, "utils.py"))

    for key in [
        "eval/handbook.station_keeping_drift_mean",
        "eval/handbook.station_keeping_drift_p95",
        "eval/handbook.height_world_z_mean",
        "eval/handbook.height_world_z_p05",
        "eval/handbook.height_world_z_p95",
        "eval/handbook.height_ceiling_violation_mean",
        "eval/handbook.height_ceiling_margin_p05",
        "eval/handbook.command_amplification_horizontal_mean",
        "eval/handbook.command_amplification_vertical_mean",
        "eval/handbook.vertical_corr_z_mean",
        "eval/handbook.vertical_corr_z_abs_mean",
        "eval/handbook.vertical_corr_z_positive_fraction",
        "eval/handbook.vertical_corr_z_negative_fraction",
        "eval/handbook.vertical_corr_z_saturation_rate",
        "eval/handbook.vertical_gov_minus_cmd_z_abs_mean",
        "eval/handbook.vertical_final_minus_cmd_z_abs_mean",
        "eval/handbook.vertical_ics_delta_z_abs_mean",
        "eval/handbook.vertical_null_corr_active_rate",
        "eval/handbook.vertical_null_corr_abs_mean",
        "eval/handbook.vertical_null_station_drift_mean_when_corr_active",
        "eval/handbook.vertical_tracking_corr_active_rate",
        "eval/handbook.vertical_tracking_amplification_mean_when_corr_active",
        "eval/handbook.vertical_tracking_preservation_mean_when_corr_active",
        "eval/handbook.r5e_null_actual_speed_xy_mean",
        "eval/handbook.r5e_null_actual_speed_z_abs_mean",
        "eval/handbook.r5e_null_output_speed_xy_mean",
        "eval/handbook.r5e_null_output_speed_z_abs_mean",
        "eval/handbook.r5e_command_preservation_pre_ics_ratio",
        "eval/handbook.r5e_command_preservation_post_ics_ratio",
        "eval/handbook.r5e_command_preservation_ics_loss_ratio",
        "eval/handbook.r5e_command_preservation_horizontal_ratio",
        "eval/handbook.r5e_command_preservation_vertical_abs_ratio",
        "eval/handbook.r5e_near_floor_rate",
        "eval/handbook.r5e_near_floor_v_cmd_z_mean",
        "eval/handbook.r5e_near_floor_v_gov_z_mean",
        "eval/handbook.r5e_near_floor_v_final_z_mean",
        "eval/handbook.r5e_near_floor_ics_beta_mean",
        "eval/handbook.r5e_near_floor_clearance_p05",
        "eval/handbook.r5e_ics_violation_near_floor_rate",
        "eval/handbook.r5g_station_null_mismatch_xy_mean",
        "eval/handbook.r5g_station_null_actual_output_xy_ratio_mean",
        "eval/handbook.r5g_anchor_{suffix}_when_{condition}",
        "eval/handbook.r5g_downward_active_rate",
        "eval/handbook.r5g_downward_attenuation_ratio_mean_when_active",
        "eval/handbook.r5g_near_floor_rate_before_",
        "eval/handbook.r5g_near_floor_{suffix}_before_",
        "eval/handbook.r5h_{condition}_rate",
        "eval/handbook.r5h_{value_name}_mean_when_{condition}",
        "eval/handbook.r5h_station_null_{value_name}_mean",
        "eval/handbook.r5h_anchor_{value_name}_mean_when_{condition}",
        "eval/handbook.r5h_collision_window{window}_steps",
        "eval/handbook.r5h_collision_window{window}_{field_name}_{suffix}",
        "eval/handbook.observability_is_proxy",
    ]:
        assert key in source
    for key in [
        "compute_vertical_channel_step_metrics(",
        "compute_r5e_mechanism_step_metrics(",
        "compute_r5g_station_anchor_step_metrics(",
        "compute_r5g_downward_step_metrics(",
        "compute_r5h_mechanism_step_metrics(",
        "vertical_diagnostic_accumulators",
        "r5e_diagnostic_accumulators",
        "r5g_station_accumulators",
        "r5h_diagnostic_accumulators",
        "r5g_termination_tracker",
        "_R5HCollisionWindowTracker",
        "r5h_collision_tracker",
        "R5H_COLLISION_WINDOW_STEPS",
        "v_corr_limit = _governor_v_corr_limit(cfg)",
    ]:
        assert key in source

    eval_source = _read(os.path.join(SCRIPTS, "eval.py"))
    for key in [
        '"vertical_null"',
        '"vertical_corr"',
        '"vertical_tracking"',
        "_copy_r5e_top_level_keys(",
        "r5e_null_actual_speed_xy_mean",
        "r5e_command_preservation_pre_ics_ratio",
        "r5e_near_floor_clearance_p05",
        "r5g_station",
        "r5g_anchor",
        "r5g_near_floor",
        "r5g_downward",
        "r5h_station",
        "r5h_anchor",
        "r5h_collision",
        "r5h_low_beta",
        "r5h_tracking",
    ]:
        assert key in eval_source
    for key in [
        "observability_sigma_min_mean",
        "observability_condition_number_mean",
        "observability_drift_projection_mean",
    ]:
        assert key in source
    assert 'f"eval/handbook.{handbook_name}"' in source
    assert 'f"eval/handbook.command_mode_fraction.{label}"' in source
    assert '"adversarial"' in source
