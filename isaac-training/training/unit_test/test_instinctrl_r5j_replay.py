import importlib.util
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
COMPARATOR_PATH = os.path.join(
    ROOT,
    "docs/instinctRL_devlog/tests/artifacts/r5j_default_equivalence/20260714_234801/compare_disabled_replay.py",
)
WRAPPER_PATH = os.path.join(
    ROOT,
    "docs/instinctRL_devlog/tests/artifacts/r5j_default_equivalence/20260714_234801/replay_wrapper.py",
)


def _load_comparator():
    spec = importlib.util.spec_from_file_location("instinctrl_r5j_comparator_test", COMPARATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_wrapper():
    artifact_dir = os.path.dirname(WRAPPER_PATH)
    if artifact_dir not in sys.path:
        sys.path.insert(0, artifact_dir)
    spec = importlib.util.spec_from_file_location("instinctrl_r5j_wrapper_test", WRAPPER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _summary(value=0.0):
    return {"count": 4, "finite_count": 4, "mean": value, "min": value, "max": value, "std": value}


def _documents():
    return (
        {"checkpoint_path": "checkpoint", "result_path": "baseline.json", "legacy": {"value": 1}},
        {"checkpoint_path": "checkpoint", "result_path": "replay.json", "legacy": {"value": 1}},
    )


def _compare(comparator, baseline, replay, expected, **kwargs):
    return comparator.compare_documents(
        baseline,
        replay,
        checkpoint_sha256=comparator.EXPECTED_CHECKPOINT_SHA256,
        checkpoint_path_matches=True,
        resolved_seed_is_zero=True,
        legacy_overrides_unchanged=True,
        gate_evaluator=lambda document: {"legacy": document["legacy"]},
        expected_diagnostic_keys=expected,
        **kwargs,
    )


def test_comparator_accepts_exact_direct_disabled_diagnostics():
    comparator = _load_comparator()
    baseline, replay = _documents()
    for field in comparator.R5J_DIAGNOSTIC_FIELDS:
        replay[field] = _summary()
    report = _compare(comparator, baseline, replay, comparator.R5J_DIAGNOSTIC_FIELDS)
    assert report["status"] == "GO (design only)"
    assert report["checks"]["existing_json_fields_exact"]
    assert report["checks"]["gate_report_exact"]


def test_comparator_accepts_exact_flattened_disabled_diagnostics():
    comparator = _load_comparator()
    baseline, replay = _documents()
    expected = []
    for field in comparator.R5J_DIAGNOSTIC_FIELDS:
        key = f"eval/station/diagnostics.{field}"
        replay[key] = _summary()
        expected.append(key)
    report = _compare(comparator, baseline, replay, expected)
    assert report["status"] == "GO (design only)"


def test_comparator_rejects_nonzero_disabled_diagnostic():
    comparator = _load_comparator()
    baseline, replay = _documents()
    field = comparator.R5J_DIAGNOSTIC_FIELDS[0]
    replay[field] = _summary(1.0)
    report = _compare(comparator, baseline, replay, [field])
    assert report["status"] == "HOLD"
    assert not report["checks"]["disabled_r5j_diagnostics_exact_zero"]


def test_comparator_rejects_missing_expected_disabled_diagnostic():
    comparator = _load_comparator()
    baseline, replay = _documents()
    report = _compare(comparator, baseline, replay, comparator.R5J_DIAGNOSTIC_FIELDS)
    assert report["status"] == "HOLD"
    assert not report["checks"]["expected_disabled_r5j_diagnostics_present"]


def test_short_diagnostic_requires_flattened_and_per_pass_r5j_summaries():
    comparator = _load_comparator()
    expected = set(comparator.expected_short_diagnostic_keys())
    for field in comparator.R5J_DIAGNOSTIC_FIELDS:
        assert f"eval/station/diagnostics.{field}" in expected
        assert f"eval/tracking/diagnostics.{field}" in expected
        assert f"passes.station_static_mid360.eval/diagnostics.{field}" in expected
        assert f"passes.tracking_static_mid360.eval/diagnostics.{field}" in expected


def test_comparator_rejects_nonzero_baseline_diagnostic_before_filtering():
    comparator = _load_comparator()
    baseline, replay = _documents()
    field = comparator.R5J_DIAGNOSTIC_FIELDS[0]
    baseline[field] = _summary(1.0)
    replay[field] = _summary()
    report = _compare(comparator, baseline, replay, [field])
    assert report["status"] == "HOLD"
    assert not report["checks"]["disabled_r5j_diagnostics_exact_zero"]


def test_comparator_rejects_unlisted_nested_r5j_difference():
    comparator = _load_comparator()
    baseline, replay = _documents()
    field = comparator.R5J_DIAGNOSTIC_FIELDS[0]
    baseline["unlisted"] = {field: _summary()}
    replay["unlisted"] = {field: _summary()}
    replay["unlisted"][field]["mean"] = 0.0
    replay["unlisted"][field]["count"] = 5
    report = _compare(comparator, baseline, replay, [])
    assert report["status"] == "HOLD"
    assert not report["checks"]["existing_json_fields_exact"]


def test_comparator_records_missing_replay_from_wrapper_failure():
    comparator = _load_comparator()
    baseline, _ = _documents()
    report = _compare(
        comparator,
        baseline,
        None,
        (),
        wrapper_failure="nvidia-smi failed (exit 1): driver communication failed",
    )
    assert report["status"] == "HOLD"
    assert report["checks"]["replay_exists"] is False
    assert report["hold_reason"] == "nvidia-smi failed (exit 1): driver communication failed"


def test_comparator_holds_when_failed_wrapper_has_a_stale_exact_replay():
    comparator = _load_comparator()
    baseline, replay = _documents()
    for field in comparator.R5J_DIAGNOSTIC_FIELDS:
        replay[field] = _summary()
    report = _compare(
        comparator,
        baseline,
        replay,
        comparator.R5J_DIAGNOSTIC_FIELDS,
        wrapper_failure="eval subprocess failed with exit 1",
    )
    assert report["status"] == "HOLD"
    assert report["checks"]["replay_exists"] is True
    assert report["hold_reason"] == "eval subprocess failed with exit 1"


def test_comparator_rejects_legacy_json_mismatch():
    comparator = _load_comparator()
    baseline, replay = _documents()
    field = comparator.R5J_DIAGNOSTIC_FIELDS[0]
    replay[field] = _summary()
    replay["legacy"]["value"] = 2
    report = _compare(comparator, baseline, replay, [field])
    assert report["status"] == "HOLD"
    assert not report["checks"]["existing_json_fields_exact"]


def test_comparator_rejects_gate_report_mismatch():
    comparator = _load_comparator()
    baseline, replay = _documents()
    field = comparator.R5J_DIAGNOSTIC_FIELDS[0]
    replay[field] = _summary()
    reports = iter(({"gate": True}, {"gate": False}))
    report = comparator.compare_documents(
        baseline,
        replay,
        checkpoint_sha256=comparator.EXPECTED_CHECKPOINT_SHA256,
        checkpoint_path_matches=True,
        resolved_seed_is_zero=True,
        legacy_overrides_unchanged=True,
        gate_evaluator=lambda _document: next(reports),
        expected_diagnostic_keys=[field],
    )
    assert report["status"] == "HOLD"
    assert not report["checks"]["gate_report_exact"]


def _ready_preflight(_python):
    empty = {"stdout": "", "stderr": "", "returncode": 0}
    return {"ready": True, "failure": None, "nvidia_smi": empty, "torch": empty}


def _patch_wrapper_provenance(monkeypatch, wrapper, checkpoint, *, seed=0, legacy=True):
    (checkpoint.parent / "isaac-training").mkdir(exist_ok=True)
    monkeypatch.setattr(wrapper, "git_value", lambda _root, *args: {
        ("branch", "--show-current"): wrapper.EXPECTED_BRANCH,
        ("rev-parse", "HEAD"): "commit-123",
        ("rev-parse", "--verify", "HEAD"): "commit-123",
        ("rev-parse", "--short", "HEAD"): "commit-123",
        ("status", "--porcelain=v1", "--untracked-files=all"): "",
    }.get(args))
    monkeypatch.setattr(wrapper, "resolved_eval_seed", lambda _root: seed)
    monkeypatch.setattr(
        wrapper,
        "stored_and_replay_argv",
        lambda _root, result: (
            ["python", "eval.py", f"checkpoint_path={checkpoint}", "result_path=old.json"],
            ["python", "eval.py", f"checkpoint_path={checkpoint}", f"result_path={result}", "instinctRL.ics.residual_preemption_enabled=false"],
            legacy,
            str(checkpoint),
        ),
    )


def test_wrapper_does_not_launch_eval_when_checkpoint_is_wrong(tmp_path, monkeypatch):
    wrapper = _load_wrapper()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    _patch_wrapper_provenance(monkeypatch, wrapper, checkpoint)
    monkeypatch.setattr(wrapper, "EXPECTED_CHECKPOINT_SHA256", "wrong")
    calls = []
    record = wrapper.run_attempt(
        root=tmp_path,
        artifact_dir=tmp_path / "artifact",
        attempt_id="wrong-checkpoint",
        preflight_fn=lambda _python: (_ for _ in ()).throw(AssertionError("preflight must not run")),
        eval_runner=lambda *args, **kwargs: calls.append((args, kwargs)),
        compare_fn=lambda *args, **kwargs: {"status": "HOLD"},
    )
    assert calls == []
    assert record["status"] == "HOLD"
    assert "checkpoint_verified" in record["failure"]


def test_wrapper_does_not_launch_eval_when_seed_or_legacy_argv_is_invalid(tmp_path, monkeypatch):
    wrapper = _load_wrapper()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    _patch_wrapper_provenance(monkeypatch, wrapper, checkpoint, seed=1, legacy=False)
    monkeypatch.setattr(wrapper, "EXPECTED_CHECKPOINT_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    calls = []
    record = wrapper.run_attempt(
        root=tmp_path,
        artifact_dir=tmp_path / "artifact",
        attempt_id="bad-seed-argv",
        preflight_fn=lambda _python: (_ for _ in ()).throw(AssertionError("preflight must not run")),
        eval_runner=lambda *args, **kwargs: calls.append((args, kwargs)),
        compare_fn=lambda *args, **kwargs: {"status": "HOLD"},
    )
    assert calls == []
    assert record["status"] == "HOLD"
    assert "seed_verified" in record["failure"]
    assert "legacy_overrides_unchanged" in record["failure"]


def test_wrapper_holds_when_eval_succeeds_without_a_fresh_result(tmp_path, monkeypatch):
    wrapper = _load_wrapper()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    _patch_wrapper_provenance(monkeypatch, wrapper, checkpoint)
    monkeypatch.setattr(wrapper, "EXPECTED_CHECKPOINT_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    record = wrapper.run_attempt(
        root=tmp_path,
        artifact_dir=tmp_path / "artifact",
        attempt_id="missing-result",
        preflight_fn=_ready_preflight,
        eval_runner=lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
        compare_fn=lambda *args, **kwargs: {"status": "HOLD"},
    )
    assert record["status"] == "HOLD"
    assert record["exit_code"] == 0
    assert record["freshness"]["ready"] is False
    assert record["failure"] == "eval exited 0 but did not produce a fresh replay result"


def test_wrapper_holds_nonzero_eval_even_if_it_writes_an_exact_looking_result(tmp_path, monkeypatch):
    wrapper = _load_wrapper()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    _patch_wrapper_provenance(monkeypatch, wrapper, checkpoint)
    monkeypatch.setattr(wrapper, "EXPECTED_CHECKPOINT_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    compare_calls = []

    def failed_eval(argv, **_kwargs):
        result = Path(next(item.split("=", 1)[1] for item in argv if item.startswith("result_path=")))
        result.write_text(json.dumps({"result_path": str(result), "legacy": "stale-looking"}), encoding="utf-8")
        return SimpleNamespace(returncode=1, stdout="", stderr="failed")

    def compare(*args, **kwargs):
        compare_calls.append(kwargs)
        return {"status": "HOLD"}

    record = wrapper.run_attempt(
        root=tmp_path,
        artifact_dir=tmp_path / "artifact",
        attempt_id="failed-eval-with-result",
        preflight_fn=_ready_preflight,
        eval_runner=failed_eval,
        compare_fn=compare,
    )
    assert record["status"] == "HOLD"
    assert record["exit_code"] == 1
    assert record["failure"] == "eval subprocess failed with exit 1"
    assert compare_calls[0]["wrapper_failure"] == "eval subprocess failed with exit 1"


def _gate_metrics():
    return {
        "eval/station/handbook.station_keeping_drift_mean": 0.0,
        "eval/station/handbook.station_keeping_drift_p95": 0.0,
        "eval/station/handbook.null_command_speed_mean": 0.0,
        "eval/station/handbook.anchor_error_mean": 0.0,
        "eval/tracking/handbook.tracking_rmse_actual_body_vs_v_cmd": 0.0,
        "eval/tracking/handbook.command_preservation_ratio": 1.0,
        "eval/tracking/handbook.command_amplification_mean": 0.0,
        "eval/tracking/handbook.command_amplification_rate": 0.0,
        "eval/handbook.safety_collision_rate": 0.0,
        "eval/handbook.safety_min_clearance_p05": 1.0,
        "eval/handbook.ics_violation_rate": 0.0,
        "eval/handbook.termination_collision": 0.0,
        "eval/handbook.termination_below_bound": 0.0,
        "eval/handbook.termination_above_bound": 0.0,
    }


def _r5j_summaries(comparator):
    return {key: _summary() for key in comparator.expected_short_diagnostic_keys()}


def _real_comparator_fixture(tmp_path, monkeypatch):
    """Build a minimal root that exercises compare_files and the real gate evaluator."""
    comparator = _load_comparator()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    baseline = tmp_path / "docs/instinctRL_devlog/tests/artifacts/r5e3_braking_residual/20260714_234801/baseline.json"
    baseline.parent.mkdir(parents=True)
    baseline_document = {
        "checkpoint_path": str(checkpoint),
        "result_path": "baseline.json",
        "legacy": {"value": 1},
        **_gate_metrics(),
    }
    baseline.write_text(json.dumps(baseline_document), encoding="utf-8")
    summary = tmp_path / "docs/instinctRL_devlog/tests/artifacts/sweeps/20260714_234801/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"jobs": [{
        "variant": comparator.SOURCE_VARIANT,
        "checkpoint_path": str(checkpoint),
        "eval_command": ["python", "eval.py", f"checkpoint_path={checkpoint}", "result_path=stored.json"],
    }]}), encoding="utf-8")
    eval_config = tmp_path / "isaac-training/training/cfg/eval.yaml"
    eval_config.parent.mkdir(parents=True)
    eval_config.write_text("seed: 0\n", encoding="utf-8")
    gates = tmp_path / "isaac-training/training/scripts/instinctRL/gates.py"
    gates.parent.mkdir(parents=True)
    shutil.copyfile(Path(ROOT) / "isaac-training/training/scripts/instinctRL/gates.py", gates)
    monkeypatch.setattr(comparator, "EXPECTED_CHECKPOINT_SHA256", checkpoint_sha256)
    monkeypatch.setattr(comparator, "git_value", lambda _root, *args: "commit-123" if args == ("rev-parse", "HEAD") else None)
    return comparator, checkpoint, checkpoint_sha256, baseline, baseline_document


def _real_attempt_record(replay_path, checkpoint, checkpoint_sha256, *, clean=True):
    return {
        "attempt_id": "real-chain",
        "status": "RUNNING",
        "result_path": str(replay_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "resolved_seed": 0,
        "seed_verified": True,
        "legacy_overrides_unchanged": True,
        "cuda_preflight": {"ready": True},
        "exit_code": 0,
        "freshness": {"ready": True},
        "branch": "a2-r5j-default-off-residual",
        "worktree": "temporary-root",
        "pre_attempt_worktree_status": "" if clean else " M dirty.py",
        "worktree_clean": clean,
        "source_commit": "commit-123",
        "commit": "commit-123",
        "failure": None,
    }


def test_real_comparison_chain_requires_all_provenance_diagnostics_and_gates(tmp_path, monkeypatch):
    comparator, checkpoint, checkpoint_sha256, baseline, baseline_document = _real_comparator_fixture(tmp_path, monkeypatch)
    replay = tmp_path / "fresh-replay.json"
    replay_document = {
        **baseline_document,
        "result_path": str(replay),
        **_r5j_summaries(comparator),
    }
    replay.write_text(json.dumps(replay_document), encoding="utf-8")
    record = _real_attempt_record(replay, checkpoint, checkpoint_sha256)

    report = comparator.compare_files(tmp_path, baseline, replay, checkpoint, attempt_record=record)
    assert report["status"] == "GO (design only)"
    assert report["checks"]["attempt.worktree_clean"]
    assert report["checks"]["gate_report_exact"]

    replay_document["legacy"]["value"] = 2
    replay.write_text(json.dumps(replay_document), encoding="utf-8")
    assert comparator.compare_files(tmp_path, baseline, replay, checkpoint, attempt_record=record)["status"] == "HOLD"
    replay_document["legacy"]["value"] = 1
    replay_document.pop(comparator.expected_short_diagnostic_keys()[0])
    replay.write_text(json.dumps(replay_document), encoding="utf-8")
    assert comparator.compare_files(tmp_path, baseline, replay, checkpoint, attempt_record=record)["status"] == "HOLD"
    replay_document.update(_r5j_summaries(comparator))
    replay_document[comparator.expected_short_diagnostic_keys()[0]]["mean"] = 1.0
    replay.write_text(json.dumps(replay_document), encoding="utf-8")
    assert comparator.compare_files(tmp_path, baseline, replay, checkpoint, attempt_record=record)["status"] == "HOLD"
    replay_document.update(_r5j_summaries(comparator))
    replay.write_text(json.dumps(replay_document), encoding="utf-8")
    assert comparator.compare_files(
        tmp_path, baseline, replay, checkpoint,
        attempt_record=_real_attempt_record(replay, checkpoint, checkpoint_sha256, clean=False),
    )["status"] == "HOLD"
    source_mismatch = _real_attempt_record(replay, checkpoint, checkpoint_sha256)
    source_mismatch["source_commit"] = "other-commit"
    assert comparator.compare_files(
        tmp_path, baseline, replay, checkpoint, attempt_record=source_mismatch,
    )["status"] == "HOLD"
    replay_document["eval/handbook.safety_collision_rate"] = 1.0
    replay.write_text(json.dumps(replay_document), encoding="utf-8")
    assert comparator.compare_files(tmp_path, baseline, replay, checkpoint, attempt_record=record)["status"] == "HOLD"


def test_dirty_worktree_writes_provenance_hold_without_cuda_or_eval(tmp_path, monkeypatch):
    wrapper = _load_wrapper()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    _patch_wrapper_provenance(monkeypatch, wrapper, checkpoint)
    monkeypatch.setattr(wrapper, "git_value", lambda _root, *args: {
        ("branch", "--show-current"): wrapper.EXPECTED_BRANCH,
        ("rev-parse", "HEAD"): "commit-123",
        ("rev-parse", "--verify", "HEAD"): "commit-123",
        ("rev-parse", "--short", "HEAD"): "commit-123",
        ("status", "--porcelain=v1", "--untracked-files=all"): " M dirty.py",
    }.get(args))
    record = wrapper.run_attempt(
        root=tmp_path,
        artifact_dir=tmp_path / "artifact",
        attempt_id="dirty",
        preflight_fn=lambda _python: (_ for _ in ()).throw(AssertionError("preflight must not run")),
        eval_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("eval must not run")),
    )
    assert record["status"] == "HOLD"
    assert record["worktree_clean"] is False
    assert "pre-attempt provenance failed" in record["failure"]
    assert not (tmp_path / "artifact/attempts/dirty").exists()
    assert json.loads(Path(record["record_path"]).read_text(encoding="utf-8"))["status"] == "HOLD"
    assert json.loads(Path(record["comparison_path"]).read_text(encoding="utf-8"))["status"] == "HOLD"


def test_wrapper_stage_exceptions_leave_parseable_hold_artifacts(tmp_path, monkeypatch):
    wrapper = _load_wrapper()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    _patch_wrapper_provenance(monkeypatch, wrapper, checkpoint)
    monkeypatch.setattr(wrapper, "EXPECTED_CHECKPOINT_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest())

    def malformed_eval(argv, **_kwargs):
        result = Path(next(item.split("=", 1)[1] for item in argv if item.startswith("result_path=")))
        result.write_text("{", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fresh_eval(argv, **_kwargs):
        result = Path(next(item.split("=", 1)[1] for item in argv if item.startswith("result_path=")))
        result.write_text(json.dumps({"result_path": str(result)}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    for name, preflight, eval_runner, compare_fn in (
        ("preflight", lambda _python: (_ for _ in ()).throw(RuntimeError("no CUDA")), None, None),
        ("eval", _ready_preflight, lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("eval broke")), None),
        ("malformed", _ready_preflight, malformed_eval, None),
        ("compare", _ready_preflight, fresh_eval, lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("comparison broke"))),
    ):
        record = wrapper.run_attempt(
            root=tmp_path,
            artifact_dir=tmp_path / "artifact",
            attempt_id=name,
            preflight_fn=preflight,
            eval_runner=eval_runner or (lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("eval must not run"))),
            compare_fn=compare_fn or (lambda *_args, **_kwargs: {"status": "HOLD"}),
        )
        assert record["status"] == "HOLD"
        assert "exception" in record["failure"]
        assert json.loads(Path(record["record_path"]).read_text(encoding="utf-8"))["status"] == "HOLD"
        assert json.loads(Path(record["comparison_path"]).read_text(encoding="utf-8"))["status"] == "HOLD"


def test_wrapper_records_artifact_record_write_exception_as_hold(tmp_path, monkeypatch):
    wrapper = _load_wrapper()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    _patch_wrapper_provenance(monkeypatch, wrapper, checkpoint)
    monkeypatch.setattr(wrapper, "EXPECTED_CHECKPOINT_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    original_write_json = wrapper._write_json

    def fail_record_write(path, value):
        if path.name == "wrapper_record.json":
            raise OSError("record disk error")
        original_write_json(path, value)

    monkeypatch.setattr(wrapper, "_write_json", fail_record_write)
    record = wrapper.run_attempt(
        root=tmp_path,
        artifact_dir=tmp_path / "artifact",
        attempt_id="artifact-write",
        preflight_fn=_ready_preflight,
        eval_runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr=""),
        compare_fn=lambda *_args, **_kwargs: {"status": "HOLD"},
    )
    assert record["status"] == "HOLD"
    assert "artifact record exception OSError: record disk error" in record["failure"]
    assert json.loads(Path(record["comparison_path"]).read_text(encoding="utf-8"))["status"] == "HOLD"
