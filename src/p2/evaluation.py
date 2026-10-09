from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.metrics.binary import binary_metrics
from src.p2.data import DataSeal, open_evaluation_views
from src.p2.torch_runtime import ManifestImageDataset, build_transforms
from src.p2.training import infer_scores


EVALUATION_MATRIX: dict[str, tuple[str, str]] = {
    "E1": ("A", "A_test"),
    "E2": ("A", "B_test"),
    "E3": ("A", "C_all"),
    "E4": ("B", "B_test"),
    "E5": ("B", "C_all"),
}


class EvaluationGateError(RuntimeError):
    pass


@dataclass
class EvaluationBundle:
    schema_version: str
    metrics: dict[str, dict[str, Any]]
    predictions: dict[str, list[dict[str, Any]]]
    e2_paired: dict[str, Any]
    prediction_paths: dict[str, str]
    seal_sha256: str


def _lock_dict(lock: Any) -> dict[str, Any]:
    if hasattr(lock, "as_dict"):
        return lock.as_dict()
    return dict(lock)


def validate_evaluation_contract(
    seal: Any,
    locks: Mapping[str, Any],
    matrix: Mapping[str, tuple[str, str]],
    consumed_path: Path,
) -> None:
    if consumed_path.exists():
        raise EvaluationGateError("EVALUATION_SEAL_ALREADY_CONSUMED")
    if dict(matrix) != EVALUATION_MATRIX:
        raise EvaluationGateError("EXPERIMENT_MATRIX_INVALID")
    if set(locks) != {"A", "B"}:
        raise EvaluationGateError("BOTH_SOURCE_LOCKS_REQUIRED")
    normalized = {source: _lock_dict(lock) for source, lock in locks.items()}
    for source in ("A", "B"):
        lock = normalized[source]
        if (
            lock.get("source") != source
            or lock.get("seed") != 42
            or lock.get("immutable") is not True
            or lock.get("data_seal_sha256") != seal.sha256
            or lock.get("threshold_origin") != "source_val_youden_j"
            or not isinstance(lock.get("threshold"), (int, float))
        ):
            raise EvaluationGateError("SOURCE_LOCK_INVALID")
    if normalized["A"].get("config_sha256") != normalized["B"].get("config_sha256"):
        raise EvaluationGateError("LOCK_CONFIG_MISMATCH")
    if normalized["A"].get("weights_sha256") != normalized["B"].get("weights_sha256"):
        raise EvaluationGateError("LOCK_WEIGHTS_MISMATCH")


def aggregate_prediction_metrics(
    predictions: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, Any]:
    targets = [int(row["target"]) for row in predictions]
    scores = [float(row["score"]) for row in predictions]
    return binary_metrics(targets, scores, float(threshold))


def pair_e2_predictions(
    e1_predictions: Sequence[Mapping[str, Any]],
    e2_predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    a_by_case = {str(row["case_index"]): row for row in e1_predictions}
    b_by_case = {str(row["case_index"]): row for row in e2_predictions}
    if (
        len(a_by_case) != len(e1_predictions)
        or len(b_by_case) != len(e2_predictions)
        or set(a_by_case) != set(b_by_case)
    ):
        raise EvaluationGateError("DERM_PAIRING_INVALID")
    counts = {
        "both_correct": 0,
        "A_correct_B_wrong": 0,
        "A_wrong_B_correct": 0,
        "both_wrong": 0,
    }
    deltas: list[float] = []
    for case in sorted(a_by_case, key=lambda value: int(value)):
        left, right = a_by_case[case], b_by_case[case]
        if int(left["target"]) != int(right["target"]):
            raise EvaluationGateError("DERM_PAIRING_INVALID")
        left_correct = int(left["prediction"]) == int(left["target"])
        right_correct = int(right["prediction"]) == int(right["target"])
        key = (
            "both_correct" if left_correct and right_correct
            else "A_correct_B_wrong" if left_correct
            else "A_wrong_B_correct" if right_correct
            else "both_wrong"
        )
        counts[key] += 1
        deltas.append(float(left["score"]) - float(right["score"]))
    return {
        "n_pairs": len(deltas),
        "counts": counts,
        "mean_score_delta_A_minus_B": sum(deltas) / len(deltas),
    }


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _load_model(lock: Mapping[str, Any], device: Any) -> Any:
    import torch
    from torch import nn
    from torchvision.models import resnet50

    checkpoint = Path(str(lock["checkpoint"]))
    if not checkpoint.is_file() or _digest(checkpoint) != lock["checkpoint_sha256"]:
        raise EvaluationGateError("CHECKPOINT_PROVENANCE_INVALID")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    return model


def _private_rows(
    records: Sequence[Mapping[str, Any]],
    targets: Sequence[int],
    scores: Sequence[float],
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, target, score in zip(records, targets, scores):
        identifiers = {
            key: source[key]
            for key in ("case_index", "image_id", "patient_id", "lesion_id")
            if key in source
        }
        rows.append(
            {
                **identifiers,
                "target": int(target),
                "score": float(score),
                "prediction": int(float(score) >= threshold),
            }
        )
    return rows


def _write_predictions(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evaluate_locked_matrix(
    *,
    config: Any,
    seal: DataSeal,
    locks: Mapping[str, Any],
    output_root: Path,
) -> EvaluationBundle:
    import torch
    from torch.utils.data import DataLoader

    if not torch.cuda.is_available():
        raise EvaluationGateError("CUDA_REQUIRED")
    consumed_path = output_root / "evaluation_consumed.json"
    validate_evaluation_contract(seal, locks, EVALUATION_MATRIX, consumed_path)
    normalized = {source: _lock_dict(lock) for source, lock in locks.items()}
    evaluation_views = open_evaluation_views(seal, normalized)
    _, transform = build_transforms(config)
    device = torch.device("cuda")
    models = {source: _load_model(normalized[source], device) for source in ("A", "B")}
    predictions: dict[str, list[dict[str, Any]]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    prediction_paths: dict[str, str] = {}
    for experiment, (source, view_name) in EVALUATION_MATRIX.items():
        records = evaluation_views[view_name]
        loader = DataLoader(
            ManifestImageDataset(records, transform),
            batch_size=config.micro_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        targets, scores = infer_scores(models[source], loader, device)
        threshold = float(normalized[source]["threshold"])
        rows = _private_rows(records, targets, scores, threshold)
        path = output_root / "predictions" / f"{experiment}.csv"
        if path.exists():
            raise EvaluationGateError("PRIVATE_PREDICTIONS_ALREADY_EXIST")
        _write_predictions(path, rows)
        predictions[experiment] = rows
        metrics[experiment] = aggregate_prediction_metrics(rows, threshold)
        prediction_paths[experiment] = str(path)
    e2_paired = pair_e2_predictions(predictions["E1"], predictions["E2"])
    bundle = EvaluationBundle(
        schema_version="p2_evaluation_bundle_v1",
        metrics=metrics,
        predictions=predictions,
        e2_paired=e2_paired,
        prediction_paths=prediction_paths,
        seal_sha256=seal.sha256,
    )
    consumed_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = consumed_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": "p2_evaluation_consumed_v1",
                "data_seal_sha256": seal.sha256,
                "experiments": {key: len(value) for key, value in predictions.items()},
                "prediction_sha256": {
                    key: _digest(Path(path)) for key, path in prediction_paths.items()
                },
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    temporary.replace(consumed_path)
    return bundle
