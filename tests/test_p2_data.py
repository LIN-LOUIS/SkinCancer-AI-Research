from __future__ import annotations

import csv
from pathlib import Path
import subprocess

import pytest

from src.p2.data import (
    DataGateError,
    open_evaluation_views,
    prepare_private_views,
    load_training_view,
)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _private_manifests(tmp_path: Path) -> tuple[Path, Path, Path]:
    image = tmp_path / "source.jpg"
    image.write_bytes(b"image")
    derm_rows: list[dict[str, object]] = []
    case = 0
    for split, count in (("train", 413), ("valid", 203), ("test", 395)):
        for _ in range(count):
            target = int(case % 4 == 0)
            for domain in ("derm", "clinic"):
                derm_rows.append(
                    {
                        "case_index": case,
                        "split": split,
                        "domain": domain,
                        "image_path": str(image if split != "test" else tmp_path / "sealed.jpg"),
                        "target": target,
                        "original_diagnosis": "MEL" if target else "NEV",
                        "image_sha256": "a" * 64,
                    }
                )
            case += 1
    derm = tmp_path / "derm.csv"
    _write_csv(derm, list(derm_rows[0]), derm_rows)
    pad_rows = [
        {
            "image_id": f"img-{index}",
            "patient_id": f"patient-{index // 2}",
            "lesion_id": f"lesion-{index}",
            "image_path": str(tmp_path / "pad-sealed.jpg"),
            "target": int(index < 52),
            "original_diagnosis": "MEL" if index < 52 else "NEV",
        }
        for index in range(2298)
    ]
    pad = tmp_path / "pad.csv"
    _write_csv(pad, list(pad_rows[0]), pad_rows)
    return derm, pad, image


def test_prepare_views_enforces_counts_isolation_and_deterministic_hashes(tmp_path: Path) -> None:
    derm, pad, _ = _private_manifests(tmp_path)

    first = prepare_private_views(derm, pad, tmp_path / "private-one")
    second = prepare_private_views(derm, pad, tmp_path / "private-two")

    assert first.counts == {
        "A_train": 413,
        "A_valid": 203,
        "A_test": 395,
        "B_train": 413,
        "B_valid": 203,
        "B_test": 395,
        "C_all": 2298,
    }
    assert first.view_hashes == second.view_hashes
    for source in ("A", "B"):
        views = load_training_view(first, source)
        assert {row["split"] for row in views["train"]} == {"train"}
        assert {row["split"] for row in views["valid"]} == {"valid"}
        assert all("patient_id" not in row for rows in views.values() for row in rows)
        assert all(row["domain"] != "pad" for rows in views.values() for row in rows)


def test_training_views_do_not_open_test_or_pad_and_require_source_images(tmp_path: Path) -> None:
    derm, pad, source_image = _private_manifests(tmp_path)
    seal = prepare_private_views(derm, pad, tmp_path / "private")
    seal.view_paths["A_test"].unlink()
    seal.view_paths["B_test"].unlink()
    seal.view_paths["C_all"].unlink()

    assert len(load_training_view(seal, "A")["train"]) == 413
    source_image.unlink()
    with pytest.raises(DataGateError, match="SOURCE_IMAGE_MISSING") as caught:
        load_training_view(seal, "A")
    assert str(tmp_path) not in str(caught.value)


def test_label_or_pairing_mismatch_fails_closed_without_identifiers(tmp_path: Path) -> None:
    derm, pad, _ = _private_manifests(tmp_path)
    rows = list(csv.DictReader(derm.open(newline="", encoding="utf-8")))
    rows[1]["target"] = "1" if rows[0]["target"] == "0" else "0"
    _write_csv(derm, list(rows[0]), rows)

    with pytest.raises(DataGateError, match="DERM_PAIR_MISMATCH") as caught:
        prepare_private_views(derm, pad, tmp_path / "private")
    assert str(tmp_path) not in str(caught.value)
    assert "patient-" not in str(caught.value)


def test_evaluation_views_require_both_immutable_source_locks(tmp_path: Path) -> None:
    derm, pad, _ = _private_manifests(tmp_path)
    seal = prepare_private_views(derm, pad, tmp_path / "private")
    lock_a = {
        "source": "A",
        "data_seal_sha256": seal.sha256,
        "threshold_origin": "source_val_youden_j",
        "immutable": True,
    }
    lock_b = {**lock_a, "source": "B"}

    with pytest.raises(DataGateError, match="BOTH_SOURCE_LOCKS_REQUIRED"):
        open_evaluation_views(seal, {"A": lock_a})
    opened = open_evaluation_views(seal, {"A": lock_a, "B": lock_b})
    assert {name: len(rows) for name, rows in opened.items()} == {
        "A_test": 395,
        "B_test": 395,
        "C_all": 2298,
    }


def test_private_output_root_is_git_ignored() -> None:
    repo = Path(__file__).resolve().parents[1]
    candidate = repo / "data_local/protocol_v0.1/p2_seed42/views/A_train.csv"
    completed = subprocess.run(
        ["git", "check-ignore", "-q", str(candidate)], cwd=repo, check=False
    )
    assert completed.returncode == 0
