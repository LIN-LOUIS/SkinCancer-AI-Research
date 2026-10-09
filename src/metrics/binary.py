from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence


def _validated(targets: Sequence[int], scores: Sequence[float]) -> tuple[list[int], list[float]]:
    if len(targets) != len(scores) or not targets:
        raise ValueError("targets and scores must be non-empty and equal length")
    labels = [int(value) for value in targets]
    values = [float(value) for value in scores]
    if any(label not in (0, 1) for label in labels):
        raise ValueError("targets must be binary")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("scores must be finite values in [0, 1]")
    return labels, values


def _counts(targets: Sequence[int], scores: Sequence[float], threshold: float) -> dict[str, int]:
    predicted = [int(score >= threshold) for score in scores]
    return {
        "tn": sum(y == 0 and p == 0 for y, p in zip(targets, predicted)),
        "fp": sum(y == 0 and p == 1 for y, p in zip(targets, predicted)),
        "fn": sum(y == 1 and p == 0 for y, p in zip(targets, predicted)),
        "tp": sum(y == 1 and p == 1 for y, p in zip(targets, predicted)),
    }


def _auroc(targets: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for target, score in zip(targets, scores) if target == 1]
    negatives = [score for target, score in zip(targets, scores) if target == 0]
    if not positives or not negatives:
        raise ValueError("AUROC requires both classes")
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def _average_precision(targets: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(targets)
    if positives == 0:
        raise ValueError("average precision requires a positive class")
    grouped: dict[float, list[int]] = defaultdict(list)
    for target, score in zip(targets, scores):
        grouped[score].append(target)
    tp = 0
    seen = 0
    value = 0.0
    for score in sorted(grouped, reverse=True):
        group = grouped[score]
        group_positives = sum(group)
        tp += group_positives
        seen += len(group)
        value += (group_positives / positives) * (tp / seen)
    return value


def youden_threshold(targets: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    labels, values = _validated(targets, scores)
    if not any(labels) or all(labels):
        raise ValueError("Youden threshold requires both classes")
    candidates = []
    for threshold in sorted(set(values)):
        counts = _counts(labels, values, threshold)
        sensitivity = counts["tp"] / (counts["tp"] + counts["fn"])
        specificity = counts["tn"] / (counts["tn"] + counts["fp"])
        candidates.append(
            {
                "threshold": threshold,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "youden_j": sensitivity + specificity - 1.0,
            }
        )
    return max(
        candidates,
        key=lambda row: (row["youden_j"], row["specificity"], row["threshold"]),
    )


def binary_metrics(
    targets: Sequence[int],
    scores: Sequence[float],
    threshold: float,
    ece_bins: int = 10,
) -> dict[str, Any]:
    labels, values = _validated(targets, scores)
    if ece_bins <= 0:
        raise ValueError("ece_bins must be positive")
    counts = _counts(labels, values, threshold)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        raise ValueError("metrics require both classes")
    sensitivity = counts["tp"] / n_pos
    specificity = counts["tn"] / n_neg
    precision_denominator = counts["tp"] + counts["fp"]
    precision = counts["tp"] / precision_denominator if precision_denominator else 0.0
    f1_denominator = precision + sensitivity
    f1 = 2 * precision * sensitivity / f1_denominator if f1_denominator else 0.0
    brier = sum((score - target) ** 2 for target, score in zip(labels, values)) / len(labels)
    ece = 0.0
    for bin_index in range(ece_bins):
        lower = bin_index / ece_bins
        upper = (bin_index + 1) / ece_bins
        members = [
            (target, score)
            for target, score in zip(labels, values)
            if lower <= score < upper or (bin_index == ece_bins - 1 and score == 1.0)
        ]
        if members:
            accuracy = sum(target for target, _ in members) / len(members)
            confidence = sum(score for _, score in members) / len(members)
            ece += len(members) / len(labels) * abs(accuracy - confidence)
    return {
        "auroc": _auroc(labels, values),
        "average_precision": _average_precision(labels, values),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "f1": f1,
        "brier_score": brier,
        "ece": ece,
        "ece_bins": ece_bins,
        "ece_binning": "[0.0,0.1),...,[0.9,1.0]" if ece_bins == 10 else "left-closed; final bin right-closed",
        "confusion_matrix": counts,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": n_pos / len(labels),
        "threshold": threshold,
    }


def select_source_validation_checkpoint(
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not history or any(row.get("split_role") != "source_val" for row in history):
        raise ValueError("checkpoint and threshold selection require source_val only")
    candidates = []
    for row in history:
        targets, scores = _validated(row["targets"], row["scores"])
        candidates.append((float(_auroc(targets, scores)), int(row["epoch"]), row, targets, scores))
    _, epoch, selected, targets, scores = max(candidates, key=lambda item: (item[0], -item[1]))
    threshold = youden_threshold(targets, scores)
    return {
        "epoch": epoch,
        "checkpoint": selected["checkpoint"],
        "validation_auroc": _auroc(targets, scores),
        "threshold": threshold["threshold"],
        "threshold_origin": "source_val_youden_j",
        "validation_metrics": binary_metrics(targets, scores, threshold["threshold"]),
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def patient_cluster_bootstrap(
    records: Sequence[Mapping[str, Any]],
    metric_name: str,
    *,
    threshold: float = 0.5,
    replicates: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    if metric_name not in {"auroc", "average_precision", "sensitivity", "specificity", "balanced_accuracy", "f1"}:
        raise ValueError("unsupported bootstrap metric")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    by_patient: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        patient = str(record.get("patient_id", ""))
        if not patient:
            raise ValueError("patient_id is required for cluster bootstrap")
        by_patient[patient].append(record)
    patients = sorted(by_patient)
    if not patients:
        raise ValueError("bootstrap records must not be empty")
    generator = random.Random(seed)
    estimates: list[float] = []
    rejected = 0
    for _ in range(replicates):
        sampled = [generator.choice(patients) for _ in patients]
        expanded = [record for patient in sampled for record in by_patient[patient]]
        targets = [int(record["target"]) for record in expanded]
        scores = [float(record["score"]) for record in expanded]
        if not any(targets) or all(targets):
            rejected += 1
            continue
        metrics = binary_metrics(targets, scores, threshold)
        estimates.append(float(metrics[metric_name]))
    if not estimates:
        raise ValueError("no valid two-class bootstrap replicates")
    return {
        "metric": metric_name,
        "confidence_level": 0.95,
        "lower": _percentile(estimates, 0.025),
        "upper": _percentile(estimates, 0.975),
        "requested_replicates": replicates,
        "valid_replicates": len(estimates),
        "rejected_one_class": rejected,
        "seed": seed,
    }


def percentile_interval(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("interval values must not be empty")
    return {"lower": _percentile(values, 0.025), "upper": _percentile(values, 0.975)}
