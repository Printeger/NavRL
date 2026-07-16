import importlib.util
import hashlib
import json
import os
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
        ("rev-parse", "--short", "HEAD"): "commit-123",
        ("status", "--short"): "",
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


def test_wrapper_records_go_only_for_a_fresh_result_and_go_comparison(tmp_path, monkeypatch):
    wrapper = _load_wrapper()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    _patch_wrapper_provenance(monkeypatch, wrapper, checkpoint)
    monkeypatch.setattr(wrapper, "EXPECTED_CHECKPOINT_SHA256", hashlib.sha256(checkpoint.read_bytes()).hexdigest())

    def fresh_eval(argv, **_kwargs):
        result = Path(next(item.split("=", 1)[1] for item in argv if item.startswith("result_path=")))
        result.write_text(json.dumps({"result_path": str(result)}), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    record = wrapper.run_attempt(
        root=tmp_path,
        artifact_dir=tmp_path / "artifact",
        attempt_id="fresh-result",
        preflight_fn=_ready_preflight,
        eval_runner=fresh_eval,
        compare_fn=lambda *args, **kwargs: {"status": "GO (design only)"},
    )
    assert record["status"] == "GO (design only)"
    assert record["failure"] is None
    assert record["exit_code"] == 0
    assert record["freshness"]["ready"] is True
    comparison = json.loads(Path(record["comparison_path"]).read_text(encoding="utf-8"))
    assert comparison["status"] == "GO (design only)"
