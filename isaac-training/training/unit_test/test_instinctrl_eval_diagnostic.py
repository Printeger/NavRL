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
        "eval/handbook.observability_is_proxy",
    ]:
        assert key in source
    for key in [
        "observability_sigma_min_mean",
        "observability_condition_number_mean",
        "observability_drift_projection_mean",
    ]:
        assert key in source
    assert 'f"eval/handbook.{handbook_name}"' in source
    assert 'f"eval/handbook.command_mode_fraction.{label}"' in source
    assert '"adversarial"' in source
