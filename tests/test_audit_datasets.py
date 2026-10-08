from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.audit_datasets import (
    AuditError,
    LabelMapping,
    binary_distribution,
    load_mapping,
    resolve_data_root,
)


MAPPING_FIELDS = [
    "dataset",
    "original_label",
    "target_name",
    "target",
    "include",
    "reason",
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
