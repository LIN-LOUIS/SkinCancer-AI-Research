from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from PIL import Image

from scripts.audit_datasets import (
    AuditError,
    LabelMapping,
    audit_derm7pt,
    audit_pad_ufes20,
    binary_distribution,
    build_markdown_report,
    load_mapping,
    render_figures,
    resolve_data_root,
    validate_official_splits,
    write_csv_reports,
    write_json_summary,
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

PAD_FIELDS = [
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


def _make_derm7pt_fixture(
    root: Path, relative: Path = Path("derm7pt")
) -> Path:
    dataset_dir = root / relative
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


def _make_pad_fixture(
    root: Path, relative: Path = Path("pad_ufes20")
) -> Path:
    dataset_dir = root / relative
    base = {
        "smoke": "False",
        "drink": "False",
        "background_father": "EUROPE",
        "background_mother": "EUROPE",
        "age": "60",
        "pesticide": "False",
        "gender": "FEMALE",
        "skin_cancer_history": "False",
        "cancer_history": "False",
        "has_piped_water": "True",
        "has_sewage_system": "True",
        "fitspatrick": "2",
        "region": "ARM",
        "diameter_1": "5",
        "diameter_2": "4",
        "itch": "False",
        "grew": "False",
        "hurt": "False",
        "changed": "False",
        "bleed": "False",
        "elevation": "False",
        "biopsed": "True",
    }
    rows = [
        {
            **base,
            "patient_id": "PAT_ALPHA",
            "lesion_id": "LESION_ALPHA",
            "diagnostic": "MEL",
            "img_id": "image_1.png",
        },
        {
            **base,
            "patient_id": "PAT_ALPHA",
            "lesion_id": "LESION_ALPHA",
            "diagnostic": "BCC",
            "img_id": "image_2.png",
        },
        {
            **base,
            "patient_id": "PAT_BETA",
            "lesion_id": "LESION_BETA",
            "smoke": "",
            "drink": "",
            "diagnostic": "NEV",
            "img_id": "image_3.png",
            "biopsed": "False",
        },
        {
            **base,
            "patient_id": "PAT_GAMMA",
            "lesion_id": "LESION_GAMMA",
            "smoke": "",
            "drink": "",
            "diagnostic": "ACK",
            "img_id": "image_4.png",
            "biopsed": "False",
        },
    ]
    _write_csv(dataset_dir / "metadata.csv", PAD_FIELDS, rows)
    for row in rows:
        image = dataset_dir / "images" / row["img_id"]
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"fixture-image")
    return dataset_dir


def _make_summary(root: Path) -> dict[str, object]:
    mapping = load_mapping(
        Path(__file__).parents[1]
        / "datasets/mappings/melanoma_binary_v0.1.csv"
    )
    return {
        "schema_version": "0.1",
        "inputs": {
            "derm7pt": "raw/derm7pt/release_v0/release_v0",
            "pad_ufes20": "raw/pad_ufes20",
        },
        "derm7pt": audit_derm7pt(_make_derm7pt_fixture(root), mapping),
        "pad_ufes20": audit_pad_ufes20(_make_pad_fixture(root), mapping),
        "warnings": [
            "PAD-UFES-20 has low melanoma prevalence.",
            "Structured metadata missingness requires protocol-level handling.",
        ],
        "unresolved_issues": [],
        "baseline_readiness": "READY",
    }


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


def test_audit_pad_reports_matching_multiplicity_and_structured_missingness(
    tmp_path: Path,
) -> None:
    dataset_dir = _make_pad_fixture(tmp_path)
    mapping = load_mapping(
        Path(__file__).parents[1]
        / "datasets/mappings/melanoma_binary_v0.1.csv"
    )

    result = audit_pad_ufes20(dataset_dir, mapping)

    assert result["metadata_rows"] == 4
    assert result["images"] == {
        "metadata_references": 4,
        "unique_image_ids": 4,
        "extracted_images": 4,
        "duplicate_image_id_excess": 0,
        "duplicate_extracted_filename_excess": 0,
        "missing_images": 0,
        "unreferenced_images": 0,
        "metadata_image_match": True,
    }
    assert result["unique_patients"] == 3
    assert result["unique_lesions"] == 3
    assert result["patient_multiplicity"] == {
        "unique_entities": 3,
        "entities_with_multiple_images": 1,
        "max_images_per_entity": 2,
        "image_count_distribution": [
            {"images_per_entity": 1, "entity_count": 2},
            {"images_per_entity": 2, "entity_count": 1},
        ],
    }
    assert result["lesion_multiplicity"] == result["patient_multiplicity"]
    assert result["binary"]["melanoma_count"] == 1
    assert result["binary"]["non_melanoma_count"] == 3
    assert result["binary"]["melanoma_prevalence"] == 0.25
    missingness = {row["column"]: row for row in result["missingness"]}
    assert missingness["smoke"]["missing_count"] == 2
    assert missingness["drink"]["missing_fraction"] == 0.5
    assert result["structured_missingness"]["shared_groups"] == [
        {
            "group_id": "missing_set_001",
            "columns": ["smoke", "drink"],
            "missing_count": 2,
        }
    ]
    assert result["structured_missingness"]["patterns"] == [
        {
            "missing_columns": [],
            "row_count": 2,
            "diagnosis_counts": {"BCC": 1, "MEL": 1},
            "biopsed_counts": {"True": 2},
        },
        {
            "missing_columns": ["smoke", "drink"],
            "row_count": 2,
            "diagnosis_counts": {"ACK": 1, "NEV": 1},
            "biopsed_counts": {"False": 2},
        },
    ]


def test_pad_summary_does_not_contain_patient_lesion_or_image_identifiers(
    tmp_path: Path,
) -> None:
    dataset_dir = _make_pad_fixture(tmp_path)

    result = audit_pad_ufes20(
        dataset_dir,
        load_mapping(
            Path(__file__).parents[1]
            / "datasets/mappings/melanoma_binary_v0.1.csv"
        ),
    )

    serialized = json.dumps(result, sort_keys=True)
    for identifier in (
        "PAT_ALPHA",
        "PAT_BETA",
        "LESION_ALPHA",
        "LESION_BETA",
        "image_1.png",
        "image_2.png",
    ):
        assert identifier not in serialized


def test_audit_pad_rejects_duplicate_image_ids(tmp_path: Path) -> None:
    dataset_dir = _make_pad_fixture(tmp_path)
    metadata_path = dataset_dir / "metadata.csv"
    rows = _read_csv(metadata_path)
    rows[1]["img_id"] = rows[0]["img_id"]
    _write_csv(metadata_path, PAD_FIELDS, rows)

    with pytest.raises(AuditError, match="Duplicate PAD image IDs"):
        audit_pad_ufes20(
            dataset_dir,
            load_mapping(
                Path(__file__).parents[1]
                / "datasets/mappings/melanoma_binary_v0.1.csv"
            ),
        )


def test_audit_pad_rejects_empty_image_id(tmp_path: Path) -> None:
    dataset_dir = _make_pad_fixture(tmp_path)
    metadata_path = dataset_dir / "metadata.csv"
    rows = _read_csv(metadata_path)
    rows[0]["img_id"] = ""
    _write_csv(metadata_path, PAD_FIELDS, rows)

    with pytest.raises(AuditError, match="Empty PAD image ID"):
        audit_pad_ufes20(
            dataset_dir,
            load_mapping(
                Path(__file__).parents[1]
                / "datasets/mappings/melanoma_binary_v0.1.csv"
            ),
        )


def test_audit_pad_rejects_duplicate_extracted_filenames(tmp_path: Path) -> None:
    dataset_dir = _make_pad_fixture(tmp_path)
    duplicate = dataset_dir / "images/nested/image_1.png"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(b"duplicate-name")

    with pytest.raises(AuditError, match="Duplicate extracted PAD filenames"):
        audit_pad_ufes20(
            dataset_dir,
            load_mapping(
                Path(__file__).parents[1]
                / "datasets/mappings/melanoma_binary_v0.1.csv"
            ),
        )


def test_audit_pad_rejects_missing_or_unreferenced_images(tmp_path: Path) -> None:
    dataset_dir = _make_pad_fixture(tmp_path)
    (dataset_dir / "images/image_1.png").unlink()
    (dataset_dir / "images/orphan.png").write_bytes(b"orphan")

    with pytest.raises(
        AuditError, match="PAD image inventory mismatch: missing=1, unreferenced=1"
    ):
        audit_pad_ufes20(
            dataset_dir,
            load_mapping(
                Path(__file__).parents[1]
                / "datasets/mappings/melanoma_binary_v0.1.csv"
            ),
        )


def test_audit_pad_rejects_unmapped_diagnosis(tmp_path: Path) -> None:
    dataset_dir = _make_pad_fixture(tmp_path)
    metadata_path = dataset_dir / "metadata.csv"
    rows = _read_csv(metadata_path)
    rows[0]["diagnostic"] = "UNKNOWN"
    _write_csv(metadata_path, PAD_FIELDS, rows)

    with pytest.raises(
        AuditError, match="Unmapped diagnosis for PAD-UFES-20: UNKNOWN"
    ):
        audit_pad_ufes20(
            dataset_dir,
            load_mapping(
                Path(__file__).parents[1]
                / "datasets/mappings/melanoma_binary_v0.1.csv"
            ),
        )


def test_write_outputs_creates_all_eight_report_files(tmp_path: Path) -> None:
    summary = _make_summary(tmp_path / "inputs")
    report_dir = tmp_path / "reports"

    csv_paths = write_csv_reports(summary, report_dir)
    json_path = write_json_summary(summary, report_dir / "audit_summary.json")
    markdown_path = report_dir / "audit_report.md"
    markdown_path.write_text(
        build_markdown_report(summary), encoding="utf-8", newline="\n"
    )

    assert {path.name for path in [*csv_paths, json_path, markdown_path]} == {
        "dataset_summary.csv",
        "derm7pt_diagnosis_distribution.csv",
        "derm7pt_concept_distribution.csv",
        "derm7pt_missingness.csv",
        "pad_diagnosis_distribution.csv",
        "pad_missingness.csv",
        "audit_summary.json",
        "audit_report.md",
    }
    for path in csv_paths:
        with path.open(newline="", encoding="utf-8") as handle:
            assert list(csv.DictReader(handle))
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == "0.1"
    assert "# Data Audit v0.1" in markdown_path.read_text(encoding="utf-8")


def test_report_outputs_are_deterministic_and_aggregate_only(tmp_path: Path) -> None:
    summary = _make_summary(tmp_path / "inputs")
    output_dirs = [tmp_path / "first", tmp_path / "second"]

    for output_dir in output_dirs:
        write_csv_reports(summary, output_dir)
        write_json_summary(summary, output_dir / "audit_summary.json")
        (output_dir / "audit_report.md").write_text(
            build_markdown_report(summary), encoding="utf-8", newline="\n"
        )

    first = {
        path.name: path.read_bytes()
        for path in output_dirs[0].iterdir()
        if path.is_file()
    }
    second = {
        path.name: path.read_bytes()
        for path in output_dirs[1].iterdir()
        if path.is_file()
    }
    assert first == second
    aggregate_text = b"\n".join(first.values()).decode("utf-8")
    for identifier in (
        "PAT_ALPHA",
        "LESION_ALPHA",
        "image_1.png",
        str(tmp_path),
    ):
        assert identifier not in aggregate_text


def test_render_figures_creates_five_300_dpi_english_pngs(
    tmp_path: Path,
) -> None:
    summary = _make_summary(tmp_path / "inputs")
    figure_dir = tmp_path / "figures"

    paths = render_figures(summary, figure_dir)

    assert [path.name for path in paths] == [
        "01_dataset_overview.png",
        "02_diagnosis_distribution.png",
        "03_melanoma_prevalence.png",
        "04_derm7pt_concept_distribution.png",
        "05_pad_metadata_missingness.png",
    ]
    for path in paths:
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert max(image.size) >= 1800
            dpi = image.info.get("dpi")
            assert dpi is not None
            assert dpi[0] == pytest.approx(300, abs=1)
            assert dpi[1] == pytest.approx(300, abs=1)


def test_cli_uses_environment_root_and_creates_required_artifacts(
    tmp_path: Path,
) -> None:
    source_root = Path(__file__).parents[1]
    repo_root = tmp_path / "repository"
    script_path = repo_root / "scripts/audit_datasets.py"
    mapping_path = repo_root / "datasets/mappings/melanoma_binary_v0.1.csv"
    script_path.parent.mkdir(parents=True)
    mapping_path.parent.mkdir(parents=True)
    shutil.copy2(source_root / "scripts/audit_datasets.py", script_path)
    shutil.copy2(
        source_root / "datasets/mappings/melanoma_binary_v0.1.csv", mapping_path
    )

    data_root = tmp_path / "data"
    _make_derm7pt_fixture(
        data_root, Path("raw/derm7pt/release_v0/release_v0")
    )
    _make_pad_fixture(data_root, Path("raw/pad_ufes20"))
    env = dict(os.environ)
    env["SKINCANCER_DATA_ROOT"] = str(data_root)
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl-cache")

    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    artifacts = {
        path.name
        for directory in (
            repo_root / "reports/data_audit/v0.1",
            repo_root / "experiments/paper1_concept_dg/figures/data_audit",
        )
        for path in directory.iterdir()
        if path.is_file()
    }
    assert artifacts == {
        "dataset_summary.csv",
        "derm7pt_diagnosis_distribution.csv",
        "derm7pt_concept_distribution.csv",
        "derm7pt_missingness.csv",
        "pad_diagnosis_distribution.csv",
        "pad_missingness.csv",
        "audit_summary.json",
        "audit_report.md",
        "01_dataset_overview.png",
        "02_diagnosis_distribution.png",
        "03_melanoma_prevalence.png",
        "04_derm7pt_concept_distribution.png",
        "05_pad_metadata_missingness.png",
    }
