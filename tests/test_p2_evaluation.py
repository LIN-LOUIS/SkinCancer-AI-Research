from __future__ import annotations

from pathlib import Path

import pytest

from src.p2.evaluation import (
    EVALUATION_MATRIX,
    EvaluationGateError,
    aggregate_prediction_metrics,
    pair_e2_predictions,
    validate_evaluation_contract,
)


def _seal() -> object:
    return type("Seal", (), {"sha256": "seal", "counts": {"A_test": 395, "B_test": 395, "C_all": 2298}})()


def _locks() -> dict[str, dict[str, object]]:
    base = {
        "seed": 42,
        "immutable": True,
        "data_seal_sha256": "seal",
        "config_sha256": "config",
        "weights_sha256": "weights",
        "threshold_origin": "source_val_youden_j",
        "threshold": 0.4,
        "checkpoint": "best.pt",
        "checkpoint_sha256": "digest",
    }
    return {"A": {**base, "source": "A"}, "B": {**base, "source": "B", "threshold": 0.6}}


def test_evaluation_contract_requires_two_immutable_source_locks(tmp_path: Path) -> None:
    locks = _locks()
    validate_evaluation_contract(_seal(), locks, EVALUATION_MATRIX, tmp_path / "consumed.json")

    invalid_cases = [
        {"A": locks["A"]},
        {**locks, "B": {**locks["B"], "immutable": False}},
        {**locks, "B": {**locks["B"], "threshold_origin": "target_youden_j"}},
        {**locks, "B": {**locks["B"], "data_seal_sha256": "other"}},
    ]
    for invalid in invalid_cases:
        with pytest.raises(EvaluationGateError):
            validate_evaluation_contract(_seal(), invalid, EVALUATION_MATRIX, tmp_path / "consumed.json")


def test_evaluation_contract_rejects_wrong_matrix_or_consumed_seal(tmp_path: Path) -> None:
    wrong = dict(EVALUATION_MATRIX)
    wrong["E3"] = ("B", "C_all")
    with pytest.raises(EvaluationGateError, match="EXPERIMENT_MATRIX_INVALID"):
        validate_evaluation_contract(_seal(), _locks(), wrong, tmp_path / "consumed.json")

    consumed = tmp_path / "consumed.json"
    consumed.write_text("{}", encoding="utf-8")
    with pytest.raises(EvaluationGateError, match="EVALUATION_SEAL_ALREADY_CONSUMED"):
        validate_evaluation_contract(_seal(), _locks(), EVALUATION_MATRIX, consumed)


def test_e2_pairing_is_one_to_one_and_reports_error_transitions() -> None:
    a = [
        {"case_index": "1", "target": 1, "score": 0.9, "prediction": 1},
        {"case_index": "2", "target": 0, "score": 0.8, "prediction": 1},
    ]
    b = [
        {"case_index": "1", "target": 1, "score": 0.2, "prediction": 0},
        {"case_index": "2", "target": 0, "score": 0.1, "prediction": 0},
    ]
    result = pair_e2_predictions(a, b)
    assert result["counts"] == {
        "both_correct": 0,
        "A_correct_B_wrong": 1,
        "A_wrong_B_correct": 1,
        "both_wrong": 0,
    }
    assert result["n_pairs"] == 2
    assert result["mean_score_delta_A_minus_B"] == pytest.approx(0.7)

    with pytest.raises(EvaluationGateError, match="DERM_PAIRING_INVALID"):
        pair_e2_predictions(a, b + [b[0]])


def test_prediction_metrics_use_fixed_lock_threshold() -> None:
    predictions = [
        {"target": 1, "score": 0.9},
        {"target": 1, "score": 0.4},
        {"target": 0, "score": 0.6},
        {"target": 0, "score": 0.1},
    ]
    result = aggregate_prediction_metrics(predictions, threshold=0.7)
    assert result["threshold"] == 0.7
    assert result["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 1, "tp": 1}
    assert all("case_index" not in key and "patient" not in key for key in result)
