from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.audit_datasets import (
    AuditError,
    LabelMapping,
    audit_derm7pt,
    binary_distribution,
    load_mapping,
    resolve_data_root,
    validate_official_splits,
)


MAPPING_FIELDS = [
    "dataset",
    "original_label",
    "target_name",
    "target",
    "include",
    "reason",
]

DERM7PT_FIELDS = [
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
]


def _write_mapping(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAPPING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _mapping_row(**overrides: str) -> dict[str, str]:
    row = {
        "dataset": "Derm7pt",
        "original_label": "melanoma",
        "target_name": "melanoma",
        "target": "1",
        "include": "true",
        "reason": "Melanoma diagnosis.",
    }
    row.update(overrides)
    return row


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _make_derm7pt_fixture(root: Path) -> Path:
    dataset_dir = root / "derm7pt"
    base = {
        "seven_point_score": "0",
        "pigment_network": "absent",
        "streaks": "absent",
        "pigmentation": "absent",
        "regression_structures": "absent",
        "dots_and_globules": "absent",
        "blue_whitish_veil": "absent",
        "vascular_structures": "absent",
        "level_of_diagnostic_difficulty": "low",
        "elevation": "flat",
        "location": "arm",
        "sex": "female",
        "management": "excision",
        "case_id": "",
        "notes": "",
    }
    rows = [
        {
            **base,
            "case_num": "1",
            "diagnosis": "melanoma",
            "pigment_network": "atypical",
            "clinic": "A/clinic0.jpg",
            "derm": "A/shared.jpg",
        },
        {
            **base,
            "case_num": "2",
            "diagnosis": "basal cell carcinoma",
            "vascular_structures": "arborizing",
            "clinic": "B/clinic1.jpg",
            "derm": "B/derm1.jpg",
        },
        {
            **base,
            "case_num": "3",
            "diagnosis": "clark nevus",
            "clinic": "A/shared.jpg",
            "derm": "C/derm2.jpg",
        },
    ]
    _write_csv(dataset_dir / "meta/meta.csv", DERM7PT_FIELDS, rows)
    for split, indexes in {"train": [0], "valid": [1], "test": [2]}.items():
        _write_csv(
            dataset_dir / f"meta/{split}_indexes.csv",
            ["indexes"],
            [{"indexes": str(index)} for index in indexes],
        )
    for relative in {
        "A/clinic0.jpg",
        "A/shared.jpg",
        "B/clinic1.jpg",
        "B/derm1.jpg",
        "C/derm2.jpg",
    }:
        image = dataset_dir / "images" / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"fixture-image")
    return dataset_dir


def test_resolve_data_root_prefers_nonempty_environment() -> None:
    result = resolve_data_root(
        {"SKINCANCER_DATA_ROOT": "/srv/skin-cancer"},
        Path("/home/researcher"),
    )

    assert result == Path("/srv/skin-cancer")


def test_resolve_data_root_uses_expanded_default_for_empty_environment() -> None:
    assert resolve_data_root({}, Path("/home/researcher")) == Path(
        "/home/researcher/Datasets/SkinCancer"
    )
    assert resolve_data_root(
        {"SKINCANCER_DATA_ROOT": "   "}, Path("/home/researcher")
    ) == Path("/home/researcher/Datasets/SkinCancer")


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([_mapping_row(), _mapping_row()], "Duplicate mapping"),
        ([_mapping_row(target="2")], "target must be 0 or 1"),
        ([_mapping_row(include="yes")], "include must be true or false"),
        (
            [_mapping_row(target_name="non-melanoma")],
            "target_name does not match target",
        ),
    ],
)
def test_load_mapping_rejects_duplicate_and_invalid_rows(
    tmp_path: Path, rows: list[dict[str, str]], message: str
) -> None:
    mapping_path = tmp_path / "mapping.csv"
    _write_mapping(mapping_path, rows)

    with pytest.raises(AuditError, match=message):
        load_mapping(mapping_path)


def test_binary_distribution_rejects_unmapped_diagnosis() -> None:
    mapping = {
        ("Derm7pt", "melanoma"): LabelMapping(
            dataset="Derm7pt",
            original_label="melanoma",
            target_name="melanoma",
            target=1,
            include=True,
            reason="Melanoma diagnosis.",
        )
    }

    with pytest.raises(
        AuditError, match="Unmapped diagnosis for Derm7pt: unknown-label"
    ):
        binary_distribution(
            dataset="Derm7pt",
            rows=[{"diagnosis": "unknown-label"}],
            diagnosis_column="diagnosis",
            mapping=mapping,
        )


def test_audit_derm7pt_counts_references_concepts_and_splits(
    tmp_path: Path,
) -> None:
    dataset_dir = _make_derm7pt_fixture(tmp_path)
    mapping = load_mapping(
        Path(__file__).parents[1]
        / "datasets/mappings/melanoma_binary_v0.1.csv"
    )

    result = audit_derm7pt(dataset_dir, mapping)

    assert result["metadata_rows"] == 3
    assert result["images"] == {
        "clinical_references": 3,
        "dermoscopy_references": 3,
        "total_references": 6,
        "unique_referenced_images": 5,
        "extracted_images": 5,
        "duplicate_reference_excess": 1,
        "duplicate_referenced_paths": 1,
        "missing_references": 0,
        "unreferenced_images": 0,
    }
    assert result["binary"]["melanoma_count"] == 1
    assert result["binary"]["non_melanoma_count"] == 2
    assert result["concepts"]["pigment_network"] == [
        {"value": "absent", "count": 2, "fraction": 2 / 3},
        {"value": "atypical", "count": 1, "fraction": 1 / 3},
    ]
    assert result["splits"]["sizes"] == {"train": 1, "valid": 1, "test": 1}
    assert result["splits"]["valid"] is True


def test_audit_derm7pt_rejects_case_mismatched_or_missing_image(
    tmp_path: Path,
) -> None:
    dataset_dir = _make_derm7pt_fixture(tmp_path)
    (dataset_dir / "images/A/clinic0.jpg").unlink()

    with pytest.raises(AuditError, match="Image inventory mismatch.*missing=1"):
        audit_derm7pt(
            dataset_dir,
            load_mapping(
                Path(__file__).parents[1]
                / "datasets/mappings/melanoma_binary_v0.1.csv"
            ),
        )


def test_validate_official_splits_rejects_overlap(tmp_path: Path) -> None:
    dataset_dir = _make_derm7pt_fixture(tmp_path)
    _write_csv(
        dataset_dir / "meta/valid_indexes.csv", ["indexes"], [{"indexes": "0"}]
    )

    with pytest.raises(AuditError, match="overlap"):
        validate_official_splits(dataset_dir / "meta", metadata_rows=3)


@pytest.mark.parametrize(
    ("split", "indexes", "message"),
    [
        ("train", [0, 0], "duplicate"),
        ("test", [3], "out-of-range"),
        ("test", [], "do not cover all metadata rows"),
    ],
)
def test_validate_official_splits_rejects_duplicate_out_of_range_and_incomplete_coverage(
    tmp_path: Path, split: str, indexes: list[int], message: str
) -> None:
    dataset_dir = _make_derm7pt_fixture(tmp_path)
    _write_csv(
        dataset_dir / f"meta/{split}_indexes.csv",
        ["indexes"],
        [{"indexes": str(index)} for index in indexes],
    )

    with pytest.raises(AuditError, match=message):
        validate_official_splits(dataset_dir / "meta", metadata_rows=3)


def test_audit_derm7pt_rejects_unmapped_diagnosis(tmp_path: Path) -> None:
    dataset_dir = _make_derm7pt_fixture(tmp_path)
    metadata_path = dataset_dir / "meta/meta.csv"
    rows = _read_csv(metadata_path)
    rows[0]["diagnosis"] = "not-in-mapping"
    _write_csv(metadata_path, DERM7PT_FIELDS, rows)

    with pytest.raises(
        AuditError, match="Unmapped diagnosis for Derm7pt: not-in-mapping"
    ):
        audit_derm7pt(
            dataset_dir,
            load_mapping(
                Path(__file__).parents[1]
                / "datasets/mappings/melanoma_binary_v0.1.csv"
            ),
        )
