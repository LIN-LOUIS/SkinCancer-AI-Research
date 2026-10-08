#!/usr/bin/env python3
"""Reproducible aggregate audit for the local skin-cancer datasets."""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
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

PAD_ENCODED_UNKNOWN_VALUES = frozenset({"UNK"})

REPORT_ARTIFACT_NAMES = frozenset(
    {
        "dataset_summary.csv",
        "derm7pt_diagnosis_distribution.csv",
        "derm7pt_concept_distribution.csv",
        "derm7pt_missingness.csv",
        "pad_diagnosis_distribution.csv",
        "pad_missingness.csv",
        "audit_summary.json",
        "audit_report.md",
    }
)

FIGURE_ARTIFACT_NAMES = frozenset(
    {
        "01_dataset_overview.png",
        "02_diagnosis_distribution.png",
        "03_melanoma_prevalence.png",
        "04_derm7pt_concept_distribution.png",
        "05_pad_metadata_missingness.png",
    }
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
    blank_sets: dict[str, frozenset[int]] = {
        column: frozenset(
            index for index, row in enumerate(rows) if not row.get(column, "")
        )
        for column in columns
    }
    encoded_unknown_sets: dict[str, frozenset[int]] = {
        column: frozenset(
            index
            for index, row in enumerate(rows)
            if row.get(column, "").upper() in PAD_ENCODED_UNKNOWN_VALUES
        )
        for column in columns
    }
    missing_sets = {
        column: blank_sets[column] | encoded_unknown_sets[column]
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

    pattern_rows: dict[
        tuple[tuple[str, ...], tuple[str, ...]], list[Mapping[str, str]]
    ] = {}
    for row in rows:
        blank_signature = tuple(
            column for column in columns if not row.get(column, "")
        )
        encoded_signature = tuple(
            column
            for column in columns
            if row.get(column, "").upper() in PAD_ENCODED_UNKNOWN_VALUES
        )
        pattern_rows.setdefault((blank_signature, encoded_signature), []).append(row)

    patterns = []
    for blank_signature, encoded_signature in sorted(
        pattern_rows,
        key=lambda value: (
            len(value[0]) + len(value[1]),
            value[0] + value[1],
            value,
        ),
    ):
        grouped_rows = pattern_rows[(blank_signature, encoded_signature)]
        unavailable = tuple(
            column
            for column in columns
            if column in blank_signature or column in encoded_signature
        )
        patterns.append(
            {
                "missing_columns": list(unavailable),
                "blank_columns": list(blank_signature),
                "encoded_unknown_columns": list(encoded_signature),
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
            "blank_count": sum(not row[column] for row in rows),
            "encoded_unknown_count": sum(
                row[column].upper() in PAD_ENCODED_UNKNOWN_VALUES for row in rows
            ),
            "missing_count": sum(
                not row[column]
                or row[column].upper() in PAD_ENCODED_UNKNOWN_VALUES
                for row in rows
            ),
            "missing_fraction": (
                sum(
                    not row[column]
                    or row[column].upper() in PAD_ENCODED_UNKNOWN_VALUES
                    for row in rows
                )
                / len(rows)
                if rows
                else 0.0
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


def _write_csv_file(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_csv_reports(
    summary: Mapping[str, Any], report_dir: Path
) -> list[Path]:
    """Write the six deterministic aggregate CSV reports."""

    derm = summary["derm7pt"]
    pad = summary["pad_ufes20"]
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary = [
        {
            "dataset": "Derm7pt",
            "sample_unit": "case",
            "modalities": "clinical;dermoscopy",
            "metadata_rows": derm["metadata_rows"],
            "image_count": derm["images"]["extracted_images"],
            "unique_patients": "",
            "unique_lesions": "",
            "melanoma_count": derm["binary"]["melanoma_count"],
            "non_melanoma_count": derm["binary"]["non_melanoma_count"],
            "melanoma_prevalence": derm["binary"]["melanoma_prevalence"],
        },
        {
            "dataset": "PAD-UFES-20",
            "sample_unit": "image row",
            "modalities": "clinical",
            "metadata_rows": pad["metadata_rows"],
            "image_count": pad["images"]["extracted_images"],
            "unique_patients": pad["unique_patients"],
            "unique_lesions": pad["unique_lesions"],
            "melanoma_count": pad["binary"]["melanoma_count"],
            "non_melanoma_count": pad["binary"]["non_melanoma_count"],
            "melanoma_prevalence": pad["binary"]["melanoma_prevalence"],
        },
    ]
    paths = [
        _write_csv_file(
            report_dir / "dataset_summary.csv",
            (
                "dataset",
                "sample_unit",
                "modalities",
                "metadata_rows",
                "image_count",
                "unique_patients",
                "unique_lesions",
                "melanoma_count",
                "non_melanoma_count",
                "melanoma_prevalence",
            ),
            dataset_summary,
        )
    ]

    diagnosis_fields = (
        "dataset",
        "diagnosis",
        "count",
        "fraction",
        "target_name",
        "target",
        "include",
    )
    paths.append(
        _write_csv_file(
            report_dir / "derm7pt_diagnosis_distribution.csv",
            diagnosis_fields,
            [
                {"dataset": "Derm7pt", **row}
                for row in derm["binary"]["distribution"]
            ],
        )
    )
    concept_rows = [
        {"concept": concept, **row}
        for concept in DERM7PT_CONCEPTS
        for row in derm["concepts"][concept]
    ]
    paths.append(
        _write_csv_file(
            report_dir / "derm7pt_concept_distribution.csv",
            ("concept", "value", "count", "fraction"),
            concept_rows,
        )
    )
    paths.append(
        _write_csv_file(
            report_dir / "derm7pt_missingness.csv",
            ("column", "missing_count", "missing_fraction"),
            derm["missingness"],
        )
    )
    paths.append(
        _write_csv_file(
            report_dir / "pad_diagnosis_distribution.csv",
            diagnosis_fields,
            [
                {"dataset": "PAD-UFES-20", **row}
                for row in pad["binary"]["distribution"]
            ],
        )
    )
    paths.append(
        _write_csv_file(
            report_dir / "pad_missingness.csv",
            (
                "column",
                "blank_count",
                "encoded_unknown_count",
                "missing_count",
                "missing_fraction",
                "structural_group",
            ),
            pad["missingness"],
        )
    )
    return paths


def write_json_summary(summary: Mapping[str, Any], path: Path) -> Path:
    """Write a deterministic machine-readable audit summary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def build_markdown_report(summary: Mapping[str, Any]) -> str:
    """Build a human-readable report entirely from computed aggregates."""

    derm = summary["derm7pt"]
    pad = summary["pad_ufes20"]
    warning_lines = "\n".join(f"- {warning}" for warning in summary["warnings"])
    unresolved = summary.get("unresolved_issues", [])
    unresolved_lines = (
        "\n".join(f"- {issue}" for issue in unresolved)
        if unresolved
        else "- None that block cohort construction or binary target mapping."
    )
    readiness_issues = summary.get("readiness_issues", [])
    readiness_lines = (
        "\n".join(f"- {issue}" for issue in readiness_issues)
        if readiness_issues
        else "- None."
    )
    shared_groups = pad["structured_missingness"]["shared_groups"]
    structured_lines = (
        "\n".join(
            "- "
            f"{group['missing_count']} rows share missingness across: "
            + ", ".join(group["columns"])
            for group in shared_groups
        )
        if shared_groups
        else "- No columns share an identical non-empty missing-row set."
    )
    encoded_unknown_lines = "\n".join(
        f"- {row['column']}: {row['encoded_unknown_count']} encoded `UNK` values"
        for row in pad["missingness"]
        if row["encoded_unknown_count"]
    )
    if not encoded_unknown_lines:
        encoded_unknown_lines = "- No encoded `UNK` values were observed."
    return f"""# Data Audit v0.1

## Scope and methods

This report was generated from the local metadata and extracted image
inventories by `scripts/audit_datasets.py`. It contains aggregate statistics
only. The target is melanoma versus non-melanoma, not malignant versus benign;
PAD-UFES-20 BCC and SCC are therefore non-melanoma.

## Dataset overview

| Dataset | Metadata rows | Images | Melanoma | Non-melanoma | Prevalence |
|---|---:|---:|---:|---:|---:|
| Derm7pt | {derm['metadata_rows']} | {derm['images']['extracted_images']} | {derm['binary']['melanoma_count']} | {derm['binary']['non_melanoma_count']} | {derm['binary']['melanoma_prevalence']:.2%} |
| PAD-UFES-20 | {pad['metadata_rows']} | {pad['images']['extracted_images']} | {pad['binary']['melanoma_count']} | {pad['binary']['non_melanoma_count']} | {pad['binary']['melanoma_prevalence']:.2%} |

## Derm7pt integrity

- Clinical image references: {derm['images']['clinical_references']}
- Dermoscopy image references: {derm['images']['dermoscopy_references']}
- Unique referenced images: {derm['images']['unique_referenced_images']}
- Duplicate-reference excess: {derm['images']['duplicate_reference_excess']}
- Missing references: {derm['images']['missing_references']}
- Unreferenced images: {derm['images']['unreferenced_images']}
- Official split sizes: train={derm['splits']['sizes']['train']}, valid={derm['splits']['sizes']['valid']}, test={derm['splits']['sizes']['test']}
- Official split integrity: {'PASS' if derm['splits']['valid'] else 'FAIL'}

## PAD-UFES-20 integrity

- Unique patients: {pad['unique_patients']}
- Unique lesions: {pad['unique_lesions']}
- Patients with multiple images: {pad['patient_multiplicity']['entities_with_multiple_images']}
- Lesions with multiple images: {pad['lesion_multiplicity']['entities_with_multiple_images']}
- Duplicate image-ID excess: {pad['images']['duplicate_image_id_excess']}
- Missing images: {pad['images']['missing_images']}
- Unreferenced images: {pad['images']['unreferenced_images']}

## Structured missingness

The following patterns are descriptive and do not establish a missingness
mechanism or justify automatic imputation.

{structured_lines}

Blank cells and explicit encoded unknowns are reported separately. Total
missingness used in the CSV and figure is their union.

{encoded_unknown_lines}

## Warnings

{warning_lines}

## Unresolved issues

{unresolved_lines}

## Baseline v0.1 readiness

**{summary['baseline_readiness']}**

Readiness checks:

{readiness_lines}

Readiness covers data inventory, target mapping, and official split integrity.
It does not remove the need for patient-aware evaluation, class-imbalance
handling, calibration analysis, or a frozen experiment protocol.
"""


def _finish_figure(figure: Any, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    return path


def render_figures(summary: Mapping[str, Any], figure_dir: Path) -> list[Path]:
    """Render the five publication-oriented audit figures."""

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "skincancer-audit-mpl"),
    )
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    derm = summary["derm7pt"]
    pad = summary["pad_ufes20"]
    blue = "#0072B2"
    orange = "#E69F00"
    green = "#009E73"
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(9, 6))
    labels = ["Derm7pt", "PAD-UFES-20"]
    x = [0, 1]
    width = 0.34
    rows = [derm["metadata_rows"], pad["metadata_rows"]]
    images = [
        derm["images"]["extracted_images"],
        pad["images"]["extracted_images"],
    ]
    bars_rows = ax.bar(
        [value - width / 2 for value in x], rows, width, label="Metadata rows", color=blue
    )
    bars_images = ax.bar(
        [value + width / 2 for value in x], images, width, label="Images", color=orange
    )
    ax.bar_label(bars_rows, padding=3)
    ax.bar_label(bars_images, padding=3)
    ax.set_xticks(x, labels)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Count")
    ax.set_title("Dataset Overview")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    path = figure_dir / "01_dataset_overview.png"
    paths.append(_finish_figure(fig, path))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 11))
    for ax, dataset_name, dataset, color in (
        (axes[0], "Derm7pt", derm, blue),
        (axes[1], "PAD-UFES-20", pad, orange),
    ):
        ordered = sorted(
            dataset["binary"]["distribution"],
            key=lambda row: (row["count"], row["diagnosis"]),
        )
        bars = ax.barh(
            [row["diagnosis"] for row in ordered],
            [row["count"] for row in ordered],
            color=color,
        )
        ax.bar_label(bars, padding=3, fontsize=8)
        ax.set_xlim(left=0)
        ax.set_xlabel("Cases / images")
        ax.set_title(f"{dataset_name} Diagnosis Distribution")
        ax.grid(axis="x", alpha=0.25)
    path = figure_dir / "02_diagnosis_distribution.png"
    paths.append(_finish_figure(fig, path))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    prevalence = [
        derm["binary"]["melanoma_prevalence"] * 100,
        pad["binary"]["melanoma_prevalence"] * 100,
    ]
    bars = ax.bar(labels, prevalence, color=[blue, orange], width=0.6)
    ax.bar_label(bars, labels=[f"{value:.2f}%" for value in prevalence], padding=4)
    ax.set_ylim(0, max(prevalence) * 1.25 if max(prevalence) else 1)
    ax.set_ylabel("Melanoma prevalence (%)")
    ax.set_title("Melanoma Prevalence by Dataset")
    ax.grid(axis="y", alpha=0.25)
    path = figure_dir / "03_melanoma_prevalence.png"
    paths.append(_finish_figure(fig, path))
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    flat_axes = list(axes.flat)
    for ax, concept in zip(flat_axes, DERM7PT_CONCEPTS):
        values = derm["concepts"][concept]
        bars = ax.barh(
            [row["value"] for row in values],
            [row["fraction"] * 100 for row in values],
            color=green,
        )
        ax.bar_label(
            bars,
            labels=[f"{row['fraction']:.1%}" for row in values],
            padding=3,
            fontsize=8,
        )
        ax.set_xlim(0, 100)
        ax.set_xlabel("Cases (%)")
        ax.set_title(concept.replace("_", " ").title())
        ax.grid(axis="x", alpha=0.25)
    flat_axes[-1].axis("off")
    fig.suptitle("Derm7pt Clinical Concept Distributions", fontsize=15, y=1.01)
    path = figure_dir / "04_derm7pt_concept_distribution.png"
    paths.append(_finish_figure(fig, path))
    plt.close(fig)

    nonzero_missingness = [
        row for row in pad["missingness"] if row["missing_count"] > 0
    ]
    nonzero_missingness.sort(
        key=lambda row: (row["missing_fraction"], row["column"])
    )
    fig, ax = plt.subplots(figsize=(11, max(6, len(nonzero_missingness) * 0.45)))
    bars = ax.barh(
        [row["column"].replace("_", " ") for row in nonzero_missingness],
        [row["missing_fraction"] * 100 for row in nonzero_missingness],
        color=orange,
    )
    ax.bar_label(
        bars,
        labels=[f"{row['missing_fraction']:.1%}" for row in nonzero_missingness],
        padding=3,
        fontsize=8,
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Missing values (%)")
    ax.set_title("PAD-UFES-20 Metadata Missingness")
    ax.grid(axis="x", alpha=0.25)
    path = figure_dir / "05_pad_metadata_missingness.png"
    paths.append(_finish_figure(fig, path))
    plt.close(fig)

    return paths


def _audit_warnings(derm: Mapping[str, Any], pad: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    if derm["images"]["duplicate_reference_excess"]:
        warnings.append(
            "Derm7pt contains repeated image references across its clinical and "
            "dermoscopy reference columns; these are duplicates, not missing files."
        )
    if pad["binary"]["melanoma_prevalence"] < 0.05:
        warnings.append(
            "PAD-UFES-20 melanoma prevalence is below 5%; ordinary accuracy is "
            "not suitable as a primary conclusion."
        )
    if pad["patient_multiplicity"]["entities_with_multiple_images"]:
        warnings.append(
            "PAD-UFES-20 contains patients with multiple images; future splits "
            "must be patient-aware to avoid leakage."
        )
    if pad["lesion_multiplicity"]["entities_with_multiple_images"]:
        warnings.append(
            "PAD-UFES-20 contains lesions with multiple images; lesion-level "
            "grouping must be preserved during evaluation."
        )
    if pad["structured_missingness"]["shared_groups"]:
        warnings.append(
            "PAD-UFES-20 metadata missingness is structured; automatic "
            "complete-case filtering or imputation requires a protocol decision."
        )
    if any(row["encoded_unknown_count"] for row in pad["missingness"]):
        warnings.append(
            "PAD-UFES-20 uses explicit UNK values in some metadata fields; the "
            "audit reports these separately from blank cells and includes both "
            "in total missingness."
        )
    return warnings


def evaluate_baseline_readiness(
    derm: Mapping[str, Any], pad: Mapping[str, Any]
) -> tuple[str, list[str]]:
    """Evaluate whether the audited cohorts can support Baseline v0.1."""

    issues: list[str] = []
    for dataset_name, audit in (("Derm7pt", derm), ("PAD-UFES-20", pad)):
        rows = audit["metadata_rows"]
        binary = audit["binary"]
        if rows <= 0:
            issues.append(f"{dataset_name} cohort is empty.")
        if binary["excluded_count"]:
            issues.append(
                f"{dataset_name} has {binary['excluded_count']} observed rows "
                "excluded by the binary mapping."
            )
        if binary["included_count"] != rows:
            issues.append(
                f"{dataset_name} included binary rows do not cover all metadata rows."
            )
        if rows > 0 and binary["melanoma_count"] <= 0:
            issues.append(f"{dataset_name} has no included melanoma examples.")
        if rows > 0 and binary["non_melanoma_count"] <= 0:
            issues.append(f"{dataset_name} has no included non-melanoma examples.")

    derm_rows = derm["metadata_rows"]
    if derm["images"]["clinical_references"] != derm_rows:
        issues.append("Derm7pt clinical references do not cover every case.")
    if derm["images"]["dermoscopy_references"] != derm_rows:
        issues.append("Derm7pt dermoscopy references do not cover every case.")
    if not derm["splits"]["valid"]:
        issues.append("Derm7pt official splits failed integrity validation.")
    if derm["images"]["missing_references"] or derm["images"]["unreferenced_images"]:
        issues.append("Derm7pt image inventory is not an exact reference match.")

    if not pad["images"]["metadata_image_match"]:
        issues.append("PAD-UFES-20 image inventory is not an exact metadata match.")
    if pad["images"]["unique_image_ids"] != pad["metadata_rows"]:
        issues.append("PAD-UFES-20 does not have one unique image ID per row.")

    return ("READY" if not issues else "BLOCKED", issues)


def _validate_artifact_directory(path: Path, expected: frozenset[str]) -> None:
    actual = {entry.name for entry in path.iterdir() if entry.is_file()}
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise AuditError(
            f"Incomplete staged artifacts in {path.name}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    empty = sorted(entry.name for entry in path.iterdir() if entry.stat().st_size == 0)
    if empty:
        raise AuditError(f"Empty staged artifacts in {path.name}: {empty}")


def _publish_artifact_directories(
    staged_report_dir: Path,
    report_dir: Path,
    staged_figure_dir: Path,
    figure_dir: Path,
    repo_root: Path,
) -> None:
    """Publish complete report and figure directories with rollback."""

    report_dir.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_root = Path(
        tempfile.mkdtemp(prefix=".data-audit-backup-", dir=repo_root)
    )
    pairs = (
        (staged_report_dir, report_dir, backup_root / "report"),
        (staged_figure_dir, figure_dir, backup_root / "figures"),
    )
    published: list[tuple[Path, Path]] = []
    try:
        for staged, destination, backup in pairs:
            if destination.exists():
                os.replace(destination, backup)
            os.replace(staged, destination)
            published.append((destination, backup))
    except Exception:
        for destination, backup in reversed(published):
            if destination.exists():
                shutil.rmtree(destination)
            if backup.exists():
                os.replace(backup, destination)
        for _, destination, backup in pairs[len(published) :]:
            if not destination.exists() and backup.exists():
                os.replace(backup, destination)
        raise
    finally:
        shutil.rmtree(backup_root, ignore_errors=True)


def run_audit(data_root: Path, repo_root: Path) -> dict[str, Any]:
    """Run both audits and write every versioned aggregate artifact."""

    paths = AuditPaths(
        data_root=data_root,
        derm7pt_dir=(
            data_root / "raw/derm7pt/release_v0/release_v0"
        ),
        pad_ufes20_dir=data_root / "raw/pad_ufes20",
        mapping_path=repo_root
        / "datasets/mappings/melanoma_binary_v0.1.csv",
        report_dir=repo_root / "reports/data_audit/v0.1",
        figure_dir=repo_root
        / "experiments/paper1_concept_dg/figures/data_audit",
    )
    mapping = load_mapping(paths.mapping_path)
    derm = audit_derm7pt(paths.derm7pt_dir, mapping)
    pad = audit_pad_ufes20(paths.pad_ufes20_dir, mapping)
    readiness, readiness_issues = evaluate_baseline_readiness(derm, pad)
    summary: dict[str, Any] = {
        "schema_version": "0.1",
        "inputs": {
            "derm7pt": "raw/derm7pt/release_v0/release_v0",
            "pad_ufes20": "raw/pad_ufes20",
        },
        "derm7pt": derm,
        "pad_ufes20": pad,
        "warnings": _audit_warnings(derm, pad),
        "unresolved_issues": [
            "PAD-UFES-20 authoritative source and license details remain TODO "
            "in the dataset card."
        ],
        "baseline_readiness": readiness,
        "readiness_issues": readiness_issues,
    }

    with tempfile.TemporaryDirectory(
        prefix=".data-audit-stage-", dir=repo_root
    ) as staging_name:
        staging_root = Path(staging_name)
        staged_report_dir = staging_root / "reports"
        staged_figure_dir = staging_root / "figures"
        write_csv_reports(summary, staged_report_dir)
        write_json_summary(summary, staged_report_dir / "audit_summary.json")
        report_path = staged_report_dir / "audit_report.md"
        report_path.write_text(
            build_markdown_report(summary), encoding="utf-8", newline="\n"
        )
        render_figures(summary, staged_figure_dir)
        _validate_artifact_directory(staged_report_dir, REPORT_ARTIFACT_NAMES)
        _validate_artifact_directory(staged_figure_dir, FIGURE_ARTIFACT_NAMES)
        _publish_artifact_directories(
            staged_report_dir,
            paths.report_dir,
            staged_figure_dir,
            paths.figure_dir,
            repo_root,
        )
    return summary


def main() -> int:
    """CLI entry point."""

    repo_root = Path(__file__).resolve().parents[1]
    data_root = resolve_data_root(os.environ, Path.home())
    try:
        summary = run_audit(data_root, repo_root)
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    derm = summary["derm7pt"]
    pad = summary["pad_ufes20"]
    print(
        "Data Audit v0.1 complete: "
        f"Derm7pt={derm['metadata_rows']} rows, "
        f"PAD-UFES-20={pad['metadata_rows']} rows, "
        f"Baseline={summary['baseline_readiness']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
