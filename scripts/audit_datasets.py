#!/usr/bin/env python3
"""Reproducible aggregate audit for the local skin-cancer datasets."""

from __future__ import annotations

import csv
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MAPPING_FIELDS = (
    "dataset",
    "original_label",
    "target_name",
    "target",
    "include",
    "reason",
)

DERM7PT_FIELDS = (
    "case_num",
    "diagnosis",
    "seven_point_score",
    "pigment_network",
    "streaks",
    "pigmentation",
    "regression_structures",
    "dots_and_globules",
    "blue_whitish_veil",
    "vascular_structures",
    "level_of_diagnostic_difficulty",
    "elevation",
    "location",
    "sex",
    "management",
    "clinic",
    "derm",
    "case_id",
    "notes",
)

DERM7PT_CONCEPTS = (
    "pigment_network",
    "streaks",
    "pigmentation",
    "regression_structures",
    "dots_and_globules",
    "blue_whitish_veil",
    "vascular_structures",
)

PAD_FIELDS = (
    "patient_id",
    "lesion_id",
    "smoke",
    "drink",
    "background_father",
    "background_mother",
    "age",
    "pesticide",
    "gender",
    "skin_cancer_history",
    "cancer_history",
    "has_piped_water",
    "has_sewage_system",
    "fitspatrick",
    "region",
    "diameter_1",
    "diameter_2",
    "diagnostic",
    "itch",
    "grew",
    "hurt",
    "changed",
    "bleed",
    "elevation",
    "img_id",
    "biopsed",
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


def validate_official_splits(meta_dir: Path, metadata_rows: int) -> dict[str, Any]:
    """Read and validate the three official Derm7pt split files."""

    splits: dict[str, list[int]] = {}
    for split in ("train", "valid", "test"):
        rows = _read_csv_rows(meta_dir / f"{split}_indexes.csv", ("indexes",))
        try:
            splits[split] = [int(row["indexes"]) for row in rows]
        except ValueError as exc:
            raise AuditError(f"Non-integer index in {split} split") from exc

    for split, indexes in splits.items():
        if len(indexes) != len(set(indexes)):
            raise AuditError(f"Official split contains duplicate indexes: {split}")
        invalid = [index for index in indexes if not 0 <= index < metadata_rows]
        if invalid:
            raise AuditError(f"Official split contains out-of-range indexes: {split}")

    split_names = tuple(splits)
    for position, left_name in enumerate(split_names):
        for right_name in split_names[position + 1 :]:
            if set(splits[left_name]) & set(splits[right_name]):
                raise AuditError(
                    f"Official splits overlap: {left_name} and {right_name}"
                )

    covered = set().union(*(set(indexes) for indexes in splits.values()))
    if covered != set(range(metadata_rows)):
        raise AuditError("Official splits do not cover all metadata rows exactly")

    return {
        "sizes": {name: len(indexes) for name, indexes in splits.items()},
        "unique_within_splits": True,
        "pairwise_disjoint": True,
        "indexes_in_range": True,
        "complete_coverage": True,
        "valid": True,
    }


def _distribution(values: Sequence[str]) -> list[dict[str, Any]]:
    counts = Counter(values)
    total = len(values)
    return [
        {
            "value": value,
            "count": counts[value],
            "fraction": counts[value] / total if total else 0.0,
        }
        for value in sorted(counts)
    ]


def audit_derm7pt(
    dataset_dir: Path,
    mapping: Mapping[tuple[str, str], LabelMapping],
) -> dict[str, Any]:
    """Audit an extracted Derm7pt release using aggregate statistics only."""

    rows = _read_csv_rows(dataset_dir / "meta/meta.csv", DERM7PT_FIELDS)
    images_dir = dataset_dir / "images"
    if not images_dir.is_dir():
        raise AuditError(f"Required image directory not found: {images_dir}")

    clinical_references = [row["clinic"] for row in rows]
    dermoscopy_references = [row["derm"] for row in rows]
    references = clinical_references + dermoscopy_references
    reference_counts = Counter(references)
    extracted = {
        path.relative_to(images_dir).as_posix()
        for path in images_dir.rglob("*")
        if path.is_file()
    }
    referenced = set(references)
    missing_references = referenced - extracted
    unreferenced_images = extracted - referenced
    if missing_references or unreferenced_images:
        raise AuditError(
            "Image inventory mismatch for Derm7pt: "
            f"missing={len(missing_references)}, "
            f"unreferenced={len(unreferenced_images)}"
        )

    missingness = [
        {
            "column": column,
            "missing_count": sum(not row[column] for row in rows),
            "missing_fraction": (
                sum(not row[column] for row in rows) / len(rows) if rows else 0.0
            ),
        }
        for column in DERM7PT_FIELDS
    ]
    concepts = {
        concept: _distribution([row[concept] for row in rows])
        for concept in DERM7PT_CONCEPTS
    }

    return {
        "dataset": "Derm7pt",
        "metadata_rows": len(rows),
        "binary": binary_distribution(
            "Derm7pt", rows, "diagnosis", mapping
        ),
        "images": {
            "clinical_references": len(clinical_references),
            "dermoscopy_references": len(dermoscopy_references),
            "total_references": len(references),
            "unique_referenced_images": len(referenced),
            "extracted_images": len(extracted),
            "duplicate_reference_excess": len(references) - len(referenced),
            "duplicate_referenced_paths": sum(
                count > 1 for count in reference_counts.values()
            ),
            "missing_references": len(missing_references),
            "unreferenced_images": len(unreferenced_images),
        },
        "missingness": missingness,
        "concepts": concepts,
        "splits": validate_official_splits(dataset_dir / "meta", len(rows)),
    }


def summarize_multiplicity(values: Iterable[str]) -> dict[str, Any]:
    """Summarize repeated entity IDs without returning the identifiers."""

    counts = Counter(values)
    count_distribution = Counter(counts.values())
    return {
        "unique_entities": len(counts),
        "entities_with_multiple_images": sum(count > 1 for count in counts.values()),
        "max_images_per_entity": max(counts.values(), default=0),
        "image_count_distribution": [
            {
                "images_per_entity": images_per_entity,
                "entity_count": count_distribution[images_per_entity],
            }
            for images_per_entity in sorted(count_distribution)
        ],
    }


def summarize_structured_missingness(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Describe missingness signatures without exposing row identifiers."""

    columns = tuple(rows[0]) if rows else PAD_FIELDS
    missing_sets: dict[str, frozenset[int]] = {
        column: frozenset(
            index for index, row in enumerate(rows) if not row.get(column, "")
        )
        for column in columns
    }
    grouped_columns: dict[frozenset[int], list[str]] = {}
    for column, missing_rows in missing_sets.items():
        if missing_rows:
            grouped_columns.setdefault(missing_rows, []).append(column)

    shared_groups = []
    group_by_column: dict[str, str] = {}
    shared = [
        (missing_rows, grouped)
        for missing_rows, grouped in grouped_columns.items()
        if len(grouped) > 1
    ]
    shared.sort(key=lambda item: (-len(item[0]), tuple(item[1])))
    for number, (missing_rows, grouped) in enumerate(shared, start=1):
        group_id = f"missing_set_{number:03d}"
        shared_groups.append(
            {
                "group_id": group_id,
                "columns": grouped,
                "missing_count": len(missing_rows),
            }
        )
        for column in grouped:
            group_by_column[column] = group_id

    pattern_rows: dict[tuple[str, ...], list[Mapping[str, str]]] = {}
    for row in rows:
        signature = tuple(column for column in columns if not row.get(column, ""))
        pattern_rows.setdefault(signature, []).append(row)

    patterns = []
    for signature in sorted(pattern_rows, key=lambda value: (len(value), value)):
        grouped_rows = pattern_rows[signature]
        patterns.append(
            {
                "missing_columns": list(signature),
                "row_count": len(grouped_rows),
                "diagnosis_counts": dict(
                    sorted(Counter(row["diagnostic"] for row in grouped_rows).items())
                ),
                "biopsed_counts": dict(
                    sorted(Counter(row["biopsed"] for row in grouped_rows).items())
                ),
            }
        )
    return {
        "shared_groups": shared_groups,
        "patterns": patterns,
        "group_by_column": group_by_column,
    }


def audit_pad_ufes20(
    dataset_dir: Path,
    mapping: Mapping[tuple[str, str], LabelMapping],
) -> dict[str, Any]:
    """Audit an extracted PAD-UFES-20 copy using aggregate statistics only."""

    rows = _read_csv_rows(dataset_dir / "metadata.csv", PAD_FIELDS)
    images_dir = dataset_dir / "images"
    if not images_dir.is_dir():
        raise AuditError(f"Required image directory not found: {images_dir}")

    image_ids = [row["img_id"] for row in rows]
    extracted_names = [
        path.name for path in images_dir.rglob("*") if path.is_file()
    ]
    if any(not image_id for image_id in image_ids):
        raise AuditError("Empty PAD image ID")
    unique_image_ids = set(image_ids)
    unique_extracted_names = set(extracted_names)
    if len(image_ids) != len(unique_image_ids):
        raise AuditError("Duplicate PAD image IDs")
    if len(extracted_names) != len(unique_extracted_names):
        raise AuditError("Duplicate extracted PAD filenames")
    missing_images = unique_image_ids - unique_extracted_names
    unreferenced_images = unique_extracted_names - unique_image_ids
    if missing_images or unreferenced_images:
        raise AuditError(
            "PAD image inventory mismatch: "
            f"missing={len(missing_images)}, "
            f"unreferenced={len(unreferenced_images)}"
        )
    structured = summarize_structured_missingness(rows)
    missingness = [
        {
            "column": column,
            "missing_count": sum(not row[column] for row in rows),
            "missing_fraction": (
                sum(not row[column] for row in rows) / len(rows) if rows else 0.0
            ),
            "structural_group": structured["group_by_column"].get(column, ""),
        }
        for column in PAD_FIELDS
    ]

    return {
        "dataset": "PAD-UFES-20",
        "metadata_rows": len(rows),
        "unique_patients": len({row["patient_id"] for row in rows}),
        "unique_lesions": len({row["lesion_id"] for row in rows}),
        "patient_multiplicity": summarize_multiplicity(
            row["patient_id"] for row in rows
        ),
        "lesion_multiplicity": summarize_multiplicity(
            row["lesion_id"] for row in rows
        ),
        "binary": binary_distribution(
            "PAD-UFES-20", rows, "diagnostic", mapping
        ),
        "images": {
            "metadata_references": len(image_ids),
            "unique_image_ids": len(unique_image_ids),
            "extracted_images": len(extracted_names),
            "duplicate_image_id_excess": len(image_ids) - len(unique_image_ids),
            "duplicate_extracted_filename_excess": (
                len(extracted_names) - len(unique_extracted_names)
            ),
            "missing_images": len(missing_images),
            "unreferenced_images": len(unreferenced_images),
            "metadata_image_match": unique_image_ids == unique_extracted_names,
        },
        "missingness": missingness,
        "structured_missingness": {
            "shared_groups": structured["shared_groups"],
            "patterns": structured["patterns"],
        },
    }


if __name__ == "__main__":
    raise SystemExit("Audit CLI wiring is not implemented yet")
