from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from src.p2.evaluation import EvaluationBundle
from src.p2.reporting import (
    paired_patient_cluster_difference,
    patient_cluster_intervals,
    publish_aggregate_reports,
)


def _pad_records(reverse: bool = False) -> list[dict[str, object]]:
    scores = (0.1, 0.2, 0.8, 0.9)
    if reverse:
        scores = tuple(reversed(scores))
    return [
        {
            "patient_id": f"private-patient-{index}",
            "image_id": f"private-image-{index}",
            "target": int(index >= 2),
            "score": score,
        }
        for index, score in enumerate(scores)
    ]


def test_patient_cluster_bootstrap_has_seeded_counts_and_literal_interval() -> None:
    result = patient_cluster_intervals(
        _pad_records(), metric_names=("auroc",), threshold=0.5,
        replicates=2000, seed=42, minimum_valid_replicates=1000,
    )
    assert result["requested_replicates"] == 2000
    assert result["valid_replicates"] == 1764
    assert result["rejected_one_class"] == 236
    assert result["intervals"]["auroc"] == {"lower": 1.0, "upper": 1.0}


def test_patient_cluster_bootstrap_fails_when_valid_draws_are_inadequate() -> None:
    records = _pad_records()[:1] + _pad_records()[2:3]
    with pytest.raises(ValueError, match="INADEQUATE_VALID_REPLICATES"):
        patient_cluster_intervals(
            records, metric_names=("auroc",), threshold=0.5,
            replicates=2000, seed=42, minimum_valid_replicates=1000,
        )


def test_paired_pad_difference_reuses_patient_draws() -> None:
    result = paired_patient_cluster_difference(
        _pad_records(), _pad_records(reverse=True), metric_names=("auroc",),
        left_threshold=0.5, right_threshold=0.5, replicates=2000, seed=42,
        minimum_valid_replicates=1000,
    )
    assert result["valid_replicates"] == 1764
    assert result["rejected_one_class"] == 236
    assert result["intervals"]["auroc_difference_E3_minus_E5"] == {
        "lower": 1.0,
        "upper": 1.0,
    }


def test_publication_is_allowlisted_and_emits_required_300dpi_figures(tmp_path: Path) -> None:
    predictions: dict[str, list[dict[str, object]]] = {}
    for experiment in ("E1", "E2", "E3", "E4", "E5"):
        base = _pad_records(reverse=experiment in {"E2", "E5"})
        predictions[experiment] = [
            {
                **row,
                "case_index": f"secret-case-{index}",
                "prediction": int(float(row["score"]) >= 0.5),
                "image_path": "/secret/raw/image.jpg",
            }
            for index, row in enumerate(base)
        ]
    metrics = {
        experiment: {
            "n_pos": 2, "n_neg": 2, "prevalence": 0.5,
            "auroc": 1.0 if experiment not in {"E2", "E5"} else 0.0,
            "average_precision": 1.0 if experiment not in {"E2", "E5"} else 0.5,
            "sensitivity": 1.0, "specificity": 1.0,
            "balanced_accuracy": 1.0, "f1": 1.0,
            "brier_score": 0.1, "ece": 0.1, "ece_bins": 10,
            "ece_binning": "[0.0,0.1),...,[0.9,1.0]",
            "confusion_matrix": {"tn": 2, "fp": 0, "fn": 0, "tp": 2},
            "threshold": 0.5,
        }
        for experiment in predictions
    }
    bundle = EvaluationBundle(
        schema_version="p2_evaluation_bundle_v1",
        metrics=metrics,
        predictions=predictions,
        e2_paired={
            "n_pairs": 4,
            "counts": {"both_correct": 0, "A_correct_B_wrong": 4, "A_wrong_B_correct": 0, "both_wrong": 0},
            "mean_score_delta_A_minus_B": 0.0,
        },
        prediction_paths={"E1": "/secret/predictions.csv"},
        seal_sha256="secret-seal",
    )
    histories = {
        "A": [{"epoch": 0, "train_loss": 1.0, "source_val_auroc": 0.7, "source_val_auprc": 0.6}],
        "B": [{"epoch": 0, "train_loss": 1.2, "source_val_auroc": 0.6, "source_val_auprc": 0.5}],
    }
    locks = {
        "A": {"selected_epoch": 0, "source_val_auroc": 0.7, "threshold": 0.5, "threshold_origin": "source_val_youden_j"},
        "B": {"selected_epoch": 0, "source_val_auroc": 0.6, "threshold": 0.5, "threshold_origin": "source_val_youden_j"},
    }

    result = publish_aggregate_reports(
        bundle=bundle,
        histories=histories,
        locks=locks,
        report_root=tmp_path / "reports",
        figure_root=tmp_path / "figures",
        bootstrap_replicates=2000,
        bootstrap_seed=42,
        minimum_valid_replicates=1000,
    )

    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in result["reports"]
        if path.suffix in {".json", ".csv", ".md"}
    )
    for secret in ("private-patient", "private-image", "secret-case", "/secret/", "secret-seal"):
        assert secret not in public_text
    payload = json.loads((tmp_path / "reports/metrics.json").read_text(encoding="utf-8"))
    assert set(payload["experiments"]) == {"E1", "E2", "E3", "E4", "E5"}
    expected = {
        "training_curves.png", "roc_pr_curves.png", "confusion_matrices.png",
        "e2_paired_errors.png", "pad_cluster_ci.png", "calibration.png",
    }
    assert {path.name for path in result["figures"]} == expected
    for figure in result["figures"]:
        with Image.open(figure) as image:
            image.verify()
            assert image.info.get("dpi", (0, 0))[0] >= 299
