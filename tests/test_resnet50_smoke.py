from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys

import pytest

import scripts.run_resnet50_smoke as smoke_module
from scripts.run_resnet50_smoke import load_source_records


REPO_ROOT = Path(__file__).parents[1]


def _write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case_index", "split", "domain", "image_path", "target",
                "original_diagnosis", "image_sha256",
            ),
        )
        writer.writeheader()
        writer.writerows(
            [
                {"case_index": "0", "split": "train", "domain": "derm", "image_path": "/private/a.jpg", "target": "1", "original_diagnosis": "melanoma", "image_sha256": "a"},
                {"case_index": "1", "split": "train", "domain": "clinic", "image_path": "/private/b.jpg", "target": "0", "original_diagnosis": "nevus", "image_sha256": "b"},
                {"case_index": "2", "split": "valid", "domain": "derm", "image_path": "/private/c.jpg", "target": "0", "original_diagnosis": "nevus", "image_sha256": "c"},
                {"case_index": "3", "split": "valid", "domain": "derm", "image_path": "/private/d.jpg", "target": "1", "original_diagnosis": "melanoma", "image_sha256": "d"},
            ]
        )


def test_load_source_records_selects_only_requested_derm_domain(tmp_path: Path) -> None:
    manifest = tmp_path / "derm.csv"
    _write_manifest(manifest)

    train, valid = load_source_records(manifest, "derm")

    assert [(row["case_index"], row["target"]) for row in train] == [("0", 1)]
    assert [(row["case_index"], row["target"]) for row in valid] == [
        ("2", 0),
        ("3", 1),
    ]


def test_load_source_records_rejects_unknown_domain_without_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "derm.csv"
    _write_manifest(manifest)

    with pytest.raises(ValueError, match="source domain") as caught:
        load_source_records(manifest, "pad")

    assert "/private/" not in str(caught.value)


def test_resnet50_smoke_cli_help_does_not_require_torch_import() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_resnet50_smoke.py", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "bounded ResNet50 seed=42 smoke" in completed.stdout


def test_smoke_error_summary_does_not_disclose_private_paths() -> None:
    error = FileNotFoundError("/private/P_SECRET/manifest.csv")

    summary = smoke_module.safe_error_summary(error)

    assert summary == "FileNotFoundError"
    assert "P_SECRET" not in summary
