#!/usr/bin/env python3
"""Reproducible aggregate audit for the local skin-cancer datasets."""

from __future__ import annotations

import csv
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


MAPPING_FIELDS = (
    "dataset",
    "original_label",
    "target_name",
    "target",
    "include",
    "reason",
)


class AuditError(RuntimeError):
    """Raised when an audit input violates a required data contract."""


@dataclass(frozen=True)
class LabelMapping:
    dataset: str
    original_label: str
    target_name: str
    target: int
    include: bool
    reason: str


@dataclass(frozen=True)
class AuditPaths:
    data_root: Path
    derm7pt_dir: Path
    pad_ufes20_dir: Path
    mapping_path: Path
    report_dir: Path
    figure_dir: Path


def resolve_data_root(env: Mapping[str, str], home: Path) -> Path:
    """Resolve the local dataset root without touching it."""

    configured = env.get("SKINCANCER_DATA_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return home.expanduser() / "Datasets" / "SkinCancer"


def _read_csv_rows(
    path: Path, required_fields: Sequence[str] | None = None
) -> list[dict[str, str]]:
    if not path.is_file():
        raise AuditError(f"Required CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if required_fields is not None:
            missing = [field for field in required_fields if field not in fields]
            if missing:
                raise AuditError(
                    f"Required columns missing from {path.name}: {', '.join(missing)}"
                )
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def load_mapping(path: Path) -> dict[tuple[str, str], LabelMapping]:
    """Load and strictly validate the versioned binary-label mapping."""

    rows = _read_csv_rows(path, MAPPING_FIELDS)
    mapping: dict[tuple[str, str], LabelMapping] = {}
    for row_number, row in enumerate(rows, start=2):
        dataset = row["dataset"]
        original_label = row["original_label"]
        target_text = row["target"]
        include_text = row["include"].lower()
        if not dataset or not original_label:
            raise AuditError(f"Empty mapping key at row {row_number}")
        if target_text not in {"0", "1"}:
            raise AuditError(f"Mapping target must be 0 or 1 at row {row_number}")
        if include_text not in {"true", "false"}:
            raise AuditError(
                f"Mapping include must be true or false at row {row_number}"
            )
        target = int(target_text)
        expected_name = "melanoma" if target == 1 else "non-melanoma"
        if row["target_name"] != expected_name:
            raise AuditError(
                f"Mapping target_name does not match target at row {row_number}"
            )
        key = (dataset, original_label)
        if key in mapping:
            raise AuditError(
                f"Duplicate mapping for dataset={dataset}, label={original_label}"
            )
        mapping[key] = LabelMapping(
            dataset=dataset,
            original_label=original_label,
            target_name=row["target_name"],
            target=target,
            include=include_text == "true",
            reason=row["reason"],
        )
    if not mapping:
        raise AuditError("Mapping file contains no rows")
    return mapping


def binary_distribution(
    dataset: str,
    rows: Sequence[Mapping[str, str]],
    diagnosis_column: str,
    mapping: Mapping[tuple[str, str], LabelMapping],
) -> dict[str, Any]:
    """Aggregate raw and binary diagnoses with explicit mapping coverage."""

    raw_counts = Counter(row.get(diagnosis_column, "").strip() for row in rows)
    if "" in raw_counts:
        raise AuditError(f"Empty diagnosis in {dataset}")

    melanoma_count = 0
    non_melanoma_count = 0
    excluded_count = 0
    distribution: list[dict[str, Any]] = []
    for label in sorted(raw_counts):
        entry = mapping.get((dataset, label))
        if entry is None:
            raise AuditError(f"Unmapped diagnosis for {dataset}: {label}")
        count = raw_counts[label]
        if not entry.include:
            excluded_count += count
        elif entry.target == 1:
            melanoma_count += count
        else:
            non_melanoma_count += count
        distribution.append(
            {
                "diagnosis": label,
                "count": count,
                "fraction": count / len(rows) if rows else 0.0,
                "target_name": entry.target_name,
                "target": entry.target,
                "include": entry.include,
            }
        )

    included_count = melanoma_count + non_melanoma_count
    return {
        "distribution": distribution,
        "melanoma_count": melanoma_count,
        "non_melanoma_count": non_melanoma_count,
        "included_count": included_count,
        "excluded_count": excluded_count,
        "melanoma_prevalence": (
            melanoma_count / included_count if included_count else 0.0
        ),
    }


if __name__ == "__main__":
    raise SystemExit("Audit CLI wiring is not implemented yet")
