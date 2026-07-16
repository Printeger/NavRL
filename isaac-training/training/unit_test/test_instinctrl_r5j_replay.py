import importlib.util
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
COMPARATOR_PATH = os.path.join(
    ROOT,
    "docs/instinctRL_devlog/tests/artifacts/r5j_default_equivalence/20260714_234801/compare_disabled_replay.py",
)


def _load_comparator():
    spec = importlib.util.spec_from_file_location("instinctrl_r5j_comparator_test", COMPARATOR_PATH)
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


def test_comparator_rejects_legacy_json_mismatch():
    comparator = _load_comparator()
    baseline, replay = _documents()
    field = comparator.R5J_DIAGNOSTIC_FIELDS[0]
    replay[field] = _summary()
    replay["legacy"]["value"] = 2
    report = _compare(comparator, baseline, replay, [field])
    assert report["status"] == "HOLD"
    assert not report["checks"]["existing_json_fields_exact"]
