from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.audit_datasets import load_mapping
from src.data.protocol_v0_1 import (
    ProtocolGateError,
    build_derm7pt_manifest,
    build_pad_manifest,
    validate_derm_split_leakage,
    validate_pad_grouping,
)
from src.metrics.binary import (
    binary_metrics,
    patient_cluster_bootstrap,
    select_source_validation_checkpoint,
    youden_threshold,
)
from src.utils.run_registry import append_run_record
from scripts.run_protocol_p0 import run_p0, write_aggregate_acceptance_report


REPO_ROOT = Path(__file__).parents[1]
MAPPING_PATH = REPO_ROOT / "datasets/mappings/melanoma_binary_v0.1.csv"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_derm(root: Path, *, cross_split_duplicate: bool = False) -> Path:
    dataset = root / "derm"
    fields = [
        "case_num", "diagnosis", "seven_point_score", "pigment_network",
        "streaks", "pigmentation", "regression_structures",
        "dots_and_globules", "blue_whitish_veil", "vascular_structures",
        "level_of_diagnostic_difficulty", "elevation", "location", "sex",
        "management", "clinic", "derm", "case_id", "notes",
    ]
    base = {field: "value" for field in fields}
    rows = []
    for index, diagnosis in enumerate(("melanoma", "clark nevus", "lentigo")):
        row = dict(base)
        row.update(
            case_num=str(index + 1),
            diagnosis=diagnosis,
            clinic=f"clinic_{index}.jpg",
            derm=f"derm_{index}.jpg",
            case_id="",
            notes="",
        )
        rows.append(row)
    if cross_split_duplicate:
        rows[2]["derm"] = rows[0]["derm"]
    _write_csv(dataset / "meta/meta.csv", fields, rows)
    for split, indexes in {"train": [0], "valid": [1], "test": [2]}.items():
        _write_csv(
            dataset / f"meta/{split}_indexes.csv",
            ["indexes"],
            [{"indexes": str(index)} for index in indexes],
        )
    references = {row[domain] for row in rows for domain in ("clinic", "derm")}
    for name in references:
        path = dataset / "images" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("content:" + name).encode())
    return dataset


def _make_pad(root: Path, rows: list[dict[str, str]]) -> Path:
    dataset = root / "pad"
    fields = [
        "patient_id", "lesion_id", "smoke", "drink", "background_father",
        "background_mother", "age", "pesticide", "gender",
        "skin_cancer_history", "cancer_history", "has_piped_water",
        "has_sewage_system", "fitspatrick", "region", "diameter_1",
        "diameter_2", "diagnostic", "itch", "grew", "hurt", "changed",
        "bleed", "elevation", "img_id", "biopsed",
    ]
    complete = {field: "False" for field in fields}
    completed_rows = [{**complete, **row} for row in rows]
    _write_csv(dataset / "metadata.csv", fields, completed_rows)
    for row in completed_rows:
        path = dataset / "images" / row["img_id"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    return dataset


def test_derm_manifest_uses_row_index_and_detects_no_cross_split_leakage(
    tmp_path: Path,
) -> None:
    dataset = _make_derm(tmp_path)
    output = tmp_path / "private/derm_manifest.csv"

    summary = build_derm7pt_manifest(dataset, load_mapping(MAPPING_PATH), output)

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["case_index"] for row in rows] == ["0", "0", "1", "1", "2", "2"]
    assert {row["split"] for row in rows} == {"train", "valid", "test"}
    assert {row["domain"] for row in rows} == {"clinic", "derm"}
    assert all(len(row["image_sha256"]) == 64 for row in rows)
    assert summary["cross_split_conflicting_components"] == 0
    assert summary["cross_split_conflicting_cases"] == 0


def test_derm_cross_split_repeated_reference_blocks_with_aggregate_only_error(
    tmp_path: Path,
) -> None:
    dataset = _make_derm(tmp_path, cross_split_duplicate=True)

    with pytest.raises(ProtocolGateError) as caught:
        build_derm7pt_manifest(
            dataset,
            load_mapping(MAPPING_PATH),
            tmp_path / "private/derm_manifest.csv",
        )

    assert caught.value.code == "DERM_CROSS_SPLIT_LEAKAGE"
    assert caught.value.aggregates == {
        "conflicting_components": 1,
        "conflicting_cases": 2,
        "conflicting_records": 2,
    }
    assert "derm_0.jpg" not in str(caught.value)


def test_derm_digest_duplicate_connects_cases_even_when_paths_differ() -> None:
    rows = [
        {"case_index": "0", "split": "train", "image_path": "a.jpg", "image_sha256": "same"},
        {"case_index": "1", "split": "test", "image_path": "b.jpg", "image_sha256": "same"},
    ]

    result = validate_derm_split_leakage(rows)

    assert result["cross_split_conflicting_components"] == 1
    assert result["cross_split_conflicting_cases"] == 2


def test_pad_grouping_accepts_repeated_consistent_lesion_and_writes_manifest(
    tmp_path: Path,
) -> None:
    rows = [
        {"patient_id": "p1", "lesion_id": "l1", "diagnostic": "MEL", "img_id": "a.png"},
        {"patient_id": "p1", "lesion_id": "l1", "diagnostic": "MEL", "img_id": "b.png"},
        {"patient_id": "p2", "lesion_id": "l2", "diagnostic": "NEV", "img_id": "c.png"},
    ]
    dataset = _make_pad(tmp_path, rows)
    output = tmp_path / "private/pad_manifest.csv"

    summary = build_pad_manifest(dataset, load_mapping(MAPPING_PATH), output)

    assert summary == {
        "rows": 3,
        "patients": 2,
        "raw_lesion_ids": 2,
        "lesion_keys": 2,
        "cross_patient_reused_lesion_ids": 0,
        "missing_patient_id_rows": 0,
        "missing_lesion_id_rows": 0,
        "missing_image_id_rows": 0,
        "missing_identifier_rows": 0,
        "unique_image_keys": 3,
        "duplicate_image_keys": 0,
        "image_key_mapping_conflicts": 0,
        "diagnosis_inconsistent_lesion_keys": 0,
        "binary_target_inconsistent_lesion_keys": 0,
    }
    assert len(output.read_text(encoding="utf-8").splitlines()) == 4


def test_pad_manifest_resolves_unique_image_filename_in_nested_directory(
    tmp_path: Path,
) -> None:
    rows = [
        {"patient_id": "p1", "lesion_id": "l1", "diagnostic": "MEL", "img_id": "a.png"},
    ]
    dataset = _make_pad(tmp_path, rows)
    image = dataset / "images/a.png"
    nested = dataset / "images/subset/a.png"
    nested.parent.mkdir(parents=True)
    image.replace(nested)
    output = tmp_path / "private/pad_manifest.csv"

    summary = build_pad_manifest(dataset, load_mapping(MAPPING_PATH), output)

    assert summary["unique_image_keys"] == 1
    with output.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    assert manifest[0]["image_path"] == str(nested.resolve())


def test_pad_grouping_treats_cross_patient_lesion_id_reuse_as_distinct_lesions() -> None:
    rows = [
        {"patient_id": "p1", "lesion_id": "l1", "diagnostic": "MEL", "img_id": "a.png"},
        {"patient_id": "p2", "lesion_id": "l1", "diagnostic": "NEV", "img_id": "b.png"},
    ]

    summary = validate_pad_grouping(rows, load_mapping(MAPPING_PATH))

    assert summary["raw_lesion_ids"] == 1
    assert summary["lesion_keys"] == 2
    assert summary["cross_patient_reused_lesion_ids"] == 1
    assert summary["diagnosis_inconsistent_lesion_keys"] == 0
    assert summary["binary_target_inconsistent_lesion_keys"] == 0


@pytest.mark.parametrize(
    ("rows", "code", "aggregate_key"),
    [
        (
            [{"patient_id": "", "lesion_id": "l1", "diagnostic": "MEL", "img_id": "a.png"}],
            "PAD_MISSING_IDENTIFIERS",
            "missing_identifier_rows",
        ),
        (
            [
                {"patient_id": "p1", "lesion_id": "l1", "diagnostic": "MEL", "img_id": "a.png"},
                {"patient_id": "p1", "lesion_id": "l1", "diagnostic": "NEV", "img_id": "b.png"},
            ],
            "PAD_LESION_DIAGNOSIS_CONFLICT",
            "diagnosis_inconsistent_lesion_keys",
        ),
        (
            [
                {"patient_id": "p1", "lesion_id": "l1", "diagnostic": "BCC", "img_id": "a.png"},
                {"patient_id": "p1", "lesion_id": "l1", "diagnostic": "SCC", "img_id": "b.png"},
            ],
            "PAD_LESION_DIAGNOSIS_CONFLICT",
            "diagnosis_inconsistent_lesion_keys",
        ),
        (
            [
                {"patient_id": "p1", "lesion_id": "l1", "diagnostic": "MEL", "img_id": "same.png"},
                {"patient_id": "p1", "lesion_id": "l1", "diagnostic": "MEL", "img_id": "same.png"},
            ],
            "PAD_DUPLICATE_IMAGE_KEY",
            "duplicate_image_keys",
        ),
        (
            [
                {"patient_id": "p1", "lesion_id": "l1", "diagnostic": "MEL", "img_id": "same.png"},
                {"patient_id": "p2", "lesion_id": "l2", "diagnostic": "MEL", "img_id": "same.png"},
            ],
            "PAD_IMAGE_KEY_MAPPING_CONFLICT",
            "image_key_mapping_conflicts",
        ),
    ],
)
def test_pad_grouping_fails_closed_without_identifier_disclosure(
    rows: list[dict[str, str]], code: str, aggregate_key: str
) -> None:
    with pytest.raises(ProtocolGateError) as caught:
        validate_pad_grouping(rows, load_mapping(MAPPING_PATH))

    assert caught.value.code == code
    assert caught.value.aggregates[aggregate_key] == 1
    assert "p1" not in str(caught.value)
    assert "l1" not in str(caught.value)


def test_pad_binary_target_conflict_is_counted_independently() -> None:
    rows = [
        {"patient_id": "p1", "lesion_id": "l1", "diagnostic": "MEL", "img_id": "a.png"},
        {"patient_id": "p1", "lesion_id": "l1", "diagnostic": "NEV", "img_id": "b.png"},
    ]

    with pytest.raises(ProtocolGateError) as caught:
        validate_pad_grouping(rows, load_mapping(MAPPING_PATH))

    assert caught.value.aggregates["diagnosis_inconsistent_lesion_keys"] == 1
    assert caught.value.aggregates["binary_target_inconsistent_lesion_keys"] == 1


def test_pad_missing_id_error_still_reports_other_complete_lesion_conflicts() -> None:
    rows = [
        {"patient_id": "", "lesion_id": "missing", "diagnostic": "NEV", "img_id": "a.png"},
        {"patient_id": "p1", "lesion_id": "l1", "diagnostic": "MEL", "img_id": "b.png"},
        {"patient_id": "p1", "lesion_id": "l1", "diagnostic": "NEV", "img_id": "c.png"},
    ]

    with pytest.raises(ProtocolGateError) as caught:
        validate_pad_grouping(rows, load_mapping(MAPPING_PATH))

    assert caught.value.code == "PAD_MISSING_IDENTIFIERS"
    assert caught.value.aggregates["diagnosis_inconsistent_lesion_keys"] == 1
    assert caught.value.aggregates["binary_target_inconsistent_lesion_keys"] == 1


def test_youden_threshold_uses_specificity_then_higher_threshold_ties() -> None:
    result = youden_threshold([0, 0, 1, 1], [0.1, 0.4, 0.4, 0.9])

    assert result["threshold"] == 0.9
    assert result["sensitivity"] == 0.5
    assert result["specificity"] == 1.0


def test_binary_metrics_are_deterministic_and_use_fixed_left_closed_ece_bins() -> None:
    result = binary_metrics(
        [0, 0, 1, 1], [0.0, 0.2, 0.8, 1.0], threshold=0.5, ece_bins=10
    )

    assert result["auroc"] == 1.0
    assert result["average_precision"] == 1.0
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["f1"] == 1.0
    assert result["brier_score"] == pytest.approx(0.02)
    assert result["ece"] == pytest.approx(0.1)
    assert result["ece_binning"] == "[0.0,0.1),...,[0.9,1.0]"
    assert result["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}
    assert result["n_pos"] == 2
    assert result["n_neg"] == 2
    assert result["prevalence"] == 0.5


def test_checkpoint_and_threshold_selection_rejects_non_source_validation() -> None:
    history = [
        {"epoch": 1, "split_role": "source_val", "targets": [0, 1], "scores": [0.3, 0.7], "checkpoint": "one.pt"},
        {"epoch": 2, "split_role": "target", "targets": [0, 1], "scores": [0.2, 0.8], "checkpoint": "two.pt"},
    ]

    with pytest.raises(ValueError, match="source_val"):
        select_source_validation_checkpoint(history)


def test_checkpoint_selection_uses_val_auroc_earliest_epoch_and_val_threshold() -> None:
    history = [
        {"epoch": 2, "split_role": "source_val", "targets": [0, 0, 1, 1], "scores": [0.1, 0.4, 0.4, 0.8], "checkpoint": "two.pt"},
        {"epoch": 1, "split_role": "source_val", "targets": [0, 0, 1, 1], "scores": [0.1, 0.4, 0.4, 0.9], "checkpoint": "one.pt"},
    ]

    selected = select_source_validation_checkpoint(history)

    assert selected["epoch"] == 1
    assert selected["checkpoint"] == "one.pt"
    assert selected["threshold"] == 0.9
    assert selected["threshold_origin"] == "source_val_youden_j"


def test_patient_cluster_bootstrap_is_seeded_and_rejects_one_class_draws() -> None:
    records = [
        {"patient_id": "p1", "target": 1, "score": 0.9},
        {"patient_id": "p1", "target": 1, "score": 0.8},
        {"patient_id": "p2", "target": 0, "score": 0.2},
        {"patient_id": "p3", "target": 0, "score": 0.1},
    ]

    first = patient_cluster_bootstrap(records, "auroc", replicates=50, seed=42)
    second = patient_cluster_bootstrap(records, "auroc", replicates=50, seed=42)

    assert first == second
    assert first["requested_replicates"] == 50
    assert first["valid_replicates"] + first["rejected_one_class"] == 50
    assert first["valid_replicates"] > 0
    assert first["confidence_level"] == 0.95


def test_run_registry_is_private_jsonl_and_contains_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "runs/protocol_v0.1/registry.jsonl"
    record = {
        "run_id": "smoke-a-42",
        "source_domain": "A",
        "tests": "P1 smoke",
        "model": "resnet50",
        "weights_provenance": "local cache",
        "data_version": "manifest-v0.1",
        "protocol_version": "0.1",
        "source_split_hash": "abc",
        "commit_sha": "def",
        "config_hash": "ghi",
        "seed": 42,
        "environment": {"device": "cpu"},
        "epoch": 0,
        "checkpoint": "private.pt",
        "threshold_origin": "source_val_youden_j",
        "threshold": 0.5,
        "timestamp": "2026-10-09T00:00:00Z",
        "metrics": {},
        "artifact_paths": [],
        "state": "PASS",
    }

    append_run_record(path, record)

    parsed = json.loads(path.read_text(encoding="utf-8").strip())
    assert parsed["seed"] == 42
    assert parsed["state"] == "PASS"


def test_local_sensitive_outputs_are_git_ignored() -> None:
    import subprocess

    paths = [
        "data_local/protocol_v0.1/manifests/derm7pt.csv",
        "runs/protocol_v0.1/predictions/a_val.csv",
        "runs/protocol_v0.1/checkpoints/resnet50_A.pt",
    ]
    completed = subprocess.run(
        ["git", "check-ignore", *paths],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert set(completed.stdout.splitlines()) == set(paths)


def test_protocol_yaml_is_machine_validatable_and_preserves_frozen_gates() -> None:
    import yaml

    path = REPO_ROOT / "experiments/paper1_concept_dg/protocol/protocol_v0.1.yaml"
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert protocol["version"] == "0.1"
    assert protocol["target"]["positive"] == "melanoma"
    assert protocol["target"]["negative"] == "non-melanoma"
    assert protocol["target"]["positive_value"] == 1
    assert protocol["target"]["negative_value"] == 0
    assert protocol["datasets"]["PAD-UFES-20"]["role"] == "external_evaluation_only"
    assert protocol["datasets"]["PAD-UFES-20"]["identity_keys"] == {
        "patient_key": "patient_id",
        "lesion_key": ["patient_id", "lesion_id"],
        "image_key": "img_id",
    }
    assert protocol["selection"]["checkpoint"] == {
        "data": "source_validation_only",
        "metric": "auroc",
        "tie_break": "earliest_epoch",
    }
    assert protocol["selection"]["threshold"]["tie_breaks"] == [
        "higher_specificity",
        "higher_threshold",
    ]
    assert protocol["metrics"]["ece_bins"] == 10
    assert protocol["bootstrap"]["unit"] == "patient"


def test_run_p0_writes_only_private_manifests_and_aggregate_summary(tmp_path: Path) -> None:
    data_root = tmp_path / "datasets"
    derm = _make_derm(data_root)
    pad = _make_pad(
        data_root,
        [
            {"patient_id": "P_SECRET_1", "lesion_id": "L_SECRET_1", "diagnostic": "MEL", "img_id": "private_a.png"},
            {"patient_id": "P_SECRET_2", "lesion_id": "L_SECRET_2", "diagnostic": "NEV", "img_id": "private_b.png"},
        ],
    )
    local_root = tmp_path / "data_local/protocol_v0.1"

    result = run_p0(
        derm_dir=derm,
        pad_dir=pad,
        mapping_path=MAPPING_PATH,
        local_root=local_root,
    )

    assert result["status"] == "READY"
    assert (local_root / "manifests/derm7pt_v0.1.csv").is_file()
    assert (local_root / "manifests/pad_ufes20_v0.1.csv").is_file()
    public_summary = json.dumps(result, sort_keys=True)
    for secret in ("P_SECRET_1", "L_SECRET_1", "private_a.png", str(tmp_path)):
        assert secret not in public_summary


def test_aggregate_acceptance_report_records_identity_correction_without_ids(
    tmp_path: Path,
) -> None:
    result = {
        "status": "READY",
        "derm": {
            "manifest_rows": 2022,
            "cross_split_conflicting_components": 0,
            "cross_split_conflicting_cases": 0,
            "cross_split_conflicting_records": 0,
        },
        "pad": {
            "rows": 2298,
            "patients": 1373,
            "raw_lesion_ids": 1641,
            "lesion_keys": 1891,
            "cross_patient_reused_lesion_ids": 250,
            "missing_patient_id_rows": 0,
            "missing_lesion_id_rows": 0,
            "missing_image_id_rows": 0,
            "missing_identifier_rows": 0,
            "unique_image_keys": 2298,
            "duplicate_image_keys": 0,
            "image_key_mapping_conflicts": 0,
            "diagnosis_inconsistent_lesion_keys": 0,
            "binary_target_inconsistent_lesion_keys": 0,
            "debug_identifier_that_must_not_be_rendered": "P_SECRET",
        },
    }
    output = tmp_path / "acceptance.md"

    write_aggregate_acceptance_report(
        result,
        output,
        smoke_status="BLOCKED_BY_ENV",
    )

    report = output.read_text(encoding="utf-8")
    assert "1,891" in report
    assert "250" in report
    assert "Data Audit v0.1" in report
    assert "1,641" in report
    assert "BLOCKED_BY_ENV" in report
    assert "P_SECRET" not in report


def test_p0_script_can_run_directly_as_documented() -> None:
    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "scripts/run_protocol_p0.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Run frozen Protocol v0.1 P0 gates" in completed.stdout
