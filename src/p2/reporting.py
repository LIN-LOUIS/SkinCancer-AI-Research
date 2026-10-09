from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import shutil
from typing import Any, Mapping, Sequence

from src.metrics.binary import binary_metrics, percentile_interval


PUBLIC_METRIC_FIELDS = (
    "n_pos", "n_neg", "prevalence", "auroc", "average_precision",
    "sensitivity", "specificity", "balanced_accuracy", "f1",
    "brier_score", "ece", "ece_bins", "ece_binning",
    "confusion_matrix", "threshold",
)


def _clustered(records: Sequence[Mapping[str, Any]]) -> tuple[list[str], dict[str, list[Mapping[str, Any]]]]:
    by_patient: dict[str, list[Mapping[str, Any]]] = {}
    for row in records:
        patient = str(row.get("patient_id", ""))
        if not patient:
            raise ValueError("PATIENT_ID_REQUIRED")
        by_patient.setdefault(patient, []).append(row)
    return sorted(by_patient), by_patient


def patient_cluster_intervals(
    records: Sequence[Mapping[str, Any]],
    *,
    metric_names: Sequence[str],
    threshold: float,
    replicates: int = 2000,
    seed: int = 42,
    minimum_valid_replicates: int = 1000,
) -> dict[str, Any]:
    patients, by_patient = _clustered(records)
    generator = random.Random(seed)
    estimates = {metric: [] for metric in metric_names}
    rejected = 0
    for _ in range(replicates):
        sampled = [generator.choice(patients) for _ in patients]
        expanded = [row for patient in sampled for row in by_patient[patient]]
        targets = [int(row["target"]) for row in expanded]
        if not any(targets) or all(targets):
            rejected += 1
            continue
        scores = [float(row["score"]) for row in expanded]
        metrics = binary_metrics(targets, scores, threshold)
        for metric in metric_names:
            if metric not in metrics or not isinstance(metrics[metric], (int, float)):
                raise ValueError("UNSUPPORTED_BOOTSTRAP_METRIC")
            estimates[metric].append(float(metrics[metric]))
    valid = replicates - rejected
    if valid < minimum_valid_replicates:
        raise ValueError("INADEQUATE_VALID_REPLICATES")
    return {
        "requested_replicates": replicates,
        "valid_replicates": valid,
        "rejected_one_class": rejected,
        "seed": seed,
        "unit": "patient_cluster",
        "intervals": {metric: percentile_interval(values) for metric, values in estimates.items()},
    }


def paired_patient_cluster_difference(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    metric_names: Sequence[str],
    left_threshold: float,
    right_threshold: float,
    replicates: int = 2000,
    seed: int = 42,
    minimum_valid_replicates: int = 1000,
) -> dict[str, Any]:
    def key(row: Mapping[str, Any]) -> str:
        return str(row.get("image_id", ""))

    left_by_image = {key(row): row for row in left}
    right_by_image = {key(row): row for row in right}
    if (
        "" in left_by_image
        or "" in right_by_image
        or len(left_by_image) != len(left)
        or len(right_by_image) != len(right)
        or set(left_by_image) != set(right_by_image)
    ):
        raise ValueError("PAIRED_PAD_ALIGNMENT_INVALID")
    paired: list[dict[str, Any]] = []
    for image in sorted(left_by_image):
        first, second = left_by_image[image], right_by_image[image]
        if (
            str(first.get("patient_id", "")) != str(second.get("patient_id", ""))
            or int(first["target"]) != int(second["target"])
        ):
            raise ValueError("PAIRED_PAD_ALIGNMENT_INVALID")
        paired.append(
            {
                "patient_id": first["patient_id"],
                "target": int(first["target"]),
                "left_score": float(first["score"]),
                "right_score": float(second["score"]),
            }
        )
    patients, by_patient = _clustered(paired)
    generator = random.Random(seed)
    differences = {metric: [] for metric in metric_names}
    rejected = 0
    for _ in range(replicates):
        sampled = [generator.choice(patients) for _ in patients]
        expanded = [row for patient in sampled for row in by_patient[patient]]
        targets = [int(row["target"]) for row in expanded]
        if not any(targets) or all(targets):
            rejected += 1
            continue
        left_metrics = binary_metrics(targets, [float(row["left_score"]) for row in expanded], left_threshold)
        right_metrics = binary_metrics(targets, [float(row["right_score"]) for row in expanded], right_threshold)
        for metric in metric_names:
            differences[metric].append(float(left_metrics[metric]) - float(right_metrics[metric]))
    valid = replicates - rejected
    if valid < minimum_valid_replicates:
        raise ValueError("INADEQUATE_VALID_REPLICATES")
    return {
        "requested_replicates": replicates,
        "valid_replicates": valid,
        "rejected_one_class": rejected,
        "seed": seed,
        "unit": "paired_patient_cluster",
        "difference": "E3_minus_E5",
        "intervals": {
            f"{metric}_difference_E3_minus_E5": percentile_interval(values)
            for metric, values in differences.items()
        },
    }


def _allowlisted_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {field: metrics[field] for field in PUBLIC_METRIC_FIELDS}


def _roc_points(rows: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[float]]:
    targets = [int(row["target"]) for row in rows]
    thresholds = [float("inf"), *sorted({float(row["score"]) for row in rows}, reverse=True), float("-inf")]
    x: list[float] = []
    y: list[float] = []
    for threshold in thresholds:
        tp = sum(target == 1 and float(row["score"]) >= threshold for target, row in zip(targets, rows))
        fp = sum(target == 0 and float(row["score"]) >= threshold for target, row in zip(targets, rows))
        positives, negatives = sum(targets), len(targets) - sum(targets)
        x.append(fp / negatives)
        y.append(tp / positives)
    return x, y


def _pr_points(rows: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[float]]:
    targets = [int(row["target"]) for row in rows]
    points = [(0.0, 1.0)]
    for threshold in sorted({float(row["score"]) for row in rows}, reverse=True):
        predicted = [float(row["score"]) >= threshold for row in rows]
        tp = sum(target == 1 and value for target, value in zip(targets, predicted))
        fp = sum(target == 0 and value for target, value in zip(targets, predicted))
        points.append((tp / sum(targets), tp / (tp + fp) if tp + fp else 1.0))
    return [point[0] for point in points], [point[1] for point in points]


def _save_figure(figure: Any, path: Path, title: str) -> None:
    figure.savefig(path, dpi=300, bbox_inches="tight", metadata={"Title": title})


def _create_figures(
    bundle: Any,
    histories: Mapping[str, Sequence[Mapping[str, Any]]],
    pad_intervals: Mapping[str, Any],
    paired_difference: Mapping[str, Any],
    output: Path,
) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for source, rows in histories.items():
        epochs = [int(row["epoch"]) + 1 for row in rows]
        axes[0].plot(epochs, [row["train_loss"] for row in rows], marker="o", label=source)
        axes[1].plot(epochs, [row["source_val_auroc"] for row in rows], marker="o", label=source)
        axes[2].plot(epochs, [row["source_val_auprc"] for row in rows], marker="o", label=source)
    for axis, label in zip(axes, ("Train loss", "Source-val AUROC", "Source-val AUPRC")):
        axis.set_xlabel("Epoch"); axis.set_ylabel(label); axis.legend()
    path = output / "training_curves.png"; _save_figure(figure, path, "P2 training curves"); plt.close(figure); paths.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for experiment, rows in bundle.predictions.items():
        fpr, tpr = _roc_points(rows); axes[0].plot(fpr, tpr, label=experiment)
        recall, precision = _pr_points(rows); axes[1].plot(recall, precision, label=experiment)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    for axis in axes: axis.legend(fontsize=7)
    path = output / "roc_pr_curves.png"; _save_figure(figure, path, "P2 ROC and PR curves"); plt.close(figure); paths.append(path)

    figure, axes = plt.subplots(1, 5, figsize=(12, 2.5))
    for axis, experiment in zip(axes, ("E1", "E2", "E3", "E4", "E5")):
        cm = bundle.metrics[experiment]["confusion_matrix"]
        matrix = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
        axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2): axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
        axis.set(title=experiment, xticks=(0, 1), yticks=(0, 1), xlabel="Predicted", ylabel="True")
    path = output / "confusion_matrices.png"; _save_figure(figure, path, "P2 confusion matrices"); plt.close(figure); paths.append(path)

    figure, axis = plt.subplots(figsize=(6, 3.5))
    transitions = bundle.e2_paired["counts"]
    axis.bar(range(len(transitions)), list(transitions.values()))
    axis.set_xticks(range(len(transitions)), [key.replace("_", "\n") for key in transitions], fontsize=7)
    axis.set_ylabel("Paired cases"); axis.set_title("E1 dermoscopy vs E2 clinical errors")
    path = output / "e2_paired_errors.png"; _save_figure(figure, path, "E2 paired error transitions"); plt.close(figure); paths.append(path)

    figure, axis = plt.subplots(figsize=(7, 3.5))
    labels: list[str] = []; centers: list[float] = []; lower: list[float] = []; upper: list[float] = []
    for experiment in ("E3", "E5"):
        for metric in ("auroc", "balanced_accuracy"):
            interval = pad_intervals[experiment]["intervals"][metric]
            center = float(bundle.metrics[experiment][metric])
            labels.append(f"{experiment} {metric}"); centers.append(center)
            lower.append(max(0.0, center - interval["lower"]))
            upper.append(max(0.0, interval["upper"] - center))
    axis.errorbar(range(len(labels)), centers, yerr=[lower, upper], fmt="o", capsize=4)
    axis.set_xticks(range(len(labels)), labels, rotation=20, ha="right"); axis.set_ylim(0, 1); axis.set_ylabel("Estimate and 95% CI")
    diff = paired_difference["intervals"]["auroc_difference_E3_minus_E5"]
    axis.set_title(f"PAD patient-cluster CI; AUROC diff CI [{diff['lower']:.3f}, {diff['upper']:.3f}]")
    path = output / "pad_cluster_ci.png"; _save_figure(figure, path, "PAD patient-cluster confidence intervals"); plt.close(figure); paths.append(path)

    figure, axis = plt.subplots(figsize=(5, 4))
    for experiment, rows in bundle.predictions.items():
        xs: list[float] = []; ys: list[float] = []
        for index in range(10):
            members = [row for row in rows if index / 10 <= float(row["score"]) < (index + 1) / 10 or index == 9 and float(row["score"]) == 1.0]
            if members:
                xs.append(sum(float(row["score"]) for row in members) / len(members))
                ys.append(sum(int(row["target"]) for row in members) / len(members))
        axis.plot(xs, ys, marker="o", label=experiment)
    axis.plot((0, 1), (0, 1), linestyle="--", color="black"); axis.set(xlabel="Mean uncalibrated score", ylabel="Observed fraction", title="Calibration (10 fixed bins)"); axis.legend(fontsize=7)
    path = output / "calibration.png"; _save_figure(figure, path, "P2 calibration curves"); plt.close(figure); paths.append(path)
    return paths


def publish_aggregate_reports(
    *,
    bundle: Any,
    histories: Mapping[str, Sequence[Mapping[str, Any]]],
    locks: Mapping[str, Any],
    report_root: Path,
    figure_root: Path,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 42,
    minimum_valid_replicates: int = 1000,
) -> dict[str, list[Path]]:
    if report_root.exists() or figure_root.exists():
        raise ValueError("PUBLIC_OUTPUT_ALREADY_EXISTS")
    metric_names = ("auroc", "average_precision", "sensitivity", "specificity", "balanced_accuracy", "f1")
    pad_intervals = {
        experiment: patient_cluster_intervals(
            bundle.predictions[experiment], metric_names=metric_names,
            threshold=float(bundle.metrics[experiment]["threshold"]),
            replicates=bootstrap_replicates, seed=bootstrap_seed,
            minimum_valid_replicates=minimum_valid_replicates,
        )
        for experiment in ("E3", "E5")
    }
    paired = paired_patient_cluster_difference(
        bundle.predictions["E3"], bundle.predictions["E5"],
        metric_names=metric_names,
        left_threshold=float(bundle.metrics["E3"]["threshold"]),
        right_threshold=float(bundle.metrics["E5"]["threshold"]),
        replicates=bootstrap_replicates, seed=bootstrap_seed,
        minimum_valid_replicates=minimum_valid_replicates,
    )
    normalized_locks = {
        source: (lock.as_dict() if hasattr(lock, "as_dict") else dict(lock))
        for source, lock in locks.items()
    }
    payload = {
        "schema_version": "p2_aggregate_report_v1",
        "score_interpretation": "uncalibrated_probability_score",
        "experiments": {name: _allowlisted_metrics(bundle.metrics[name]) for name in ("E1", "E2", "E3", "E4", "E5")},
        "source_selection": {
            source: {
                "selected_epoch": int(lock["selected_epoch"]),
                "source_val_auroc": float(lock["source_val_auroc"]),
                "threshold": float(lock["threshold"]),
                "threshold_origin": str(lock["threshold_origin"]),
            }
            for source, lock in normalized_locks.items()
        },
        "e2_paired": bundle.e2_paired,
        "pad_patient_cluster_intervals": pad_intervals,
        "pad_paired_difference": paired,
    }
    report_stage = report_root.with_name(report_root.name + ".staging")
    figure_stage = figure_root.with_name(figure_root.name + ".staging")
    for stage in (report_stage, figure_stage):
        if stage.exists(): shutil.rmtree(stage)
        stage.mkdir(parents=True)
    (report_stage / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (report_stage / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["experiment", *[field for field in PUBLIC_METRIC_FIELDS if field not in {"confusion_matrix", "ece_binning"}], "tn", "fp", "fn", "tp"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for experiment, metrics in payload["experiments"].items():
            row = {key: metrics[key] for key in fields if key in metrics}
            row["experiment"] = experiment; row.update(metrics["confusion_matrix"]); writer.writerow(row)
    lines = ["# Paper 1 P2 Seed 42 聚合结果", "", "分数为未校准概率分数；阈值仅来自各自 source validation。", "", "| 实验 | N+ | N- | prevalence | AUROC | AUPRC | sensitivity | specificity | balanced accuracy | F1 | Brier | ECE |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for experiment, metrics in payload["experiments"].items():
        lines.append(f"| {experiment} | {metrics['n_pos']} | {metrics['n_neg']} | {metrics['prevalence']:.6f} | {metrics['auroc']:.6f} | {metrics['average_precision']:.6f} | {metrics['sensitivity']:.6f} | {metrics['specificity']:.6f} | {metrics['balanced_accuracy']:.6f} | {metrics['f1']:.6f} | {metrics['brier_score']:.6f} | {metrics['ece']:.6f} |")
    (report_stage / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figures_stage = _create_figures(bundle, histories, pad_intervals, paired, figure_stage)
    report_stage.replace(report_root); figure_stage.replace(figure_root)
    reports = [report_root / name for name in ("metrics.json", "metrics.csv", "metrics.md")]
    figures = [figure_root / path.name for path in figures_stage]
    return {"reports": reports, "figures": figures}
