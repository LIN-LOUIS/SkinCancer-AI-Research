from __future__ import annotations

import csv
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.audit_datasets import LabelMapping


class ProtocolGateError(RuntimeError):
    """A fail-closed protocol gate carrying aggregate-only evidence."""

    def __init__(self, code: str, aggregates: Mapping[str, int]) -> None:
        self.code = code
        self.aggregates = dict(aggregates)
        details = ", ".join(f"{key}={value}" for key, value in self.aggregates.items())
        super().__init__(f"{code}: {details}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ProtocolGateError("REQUIRED_CSV_MISSING", {"missing_files": 1})
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target(
    mapping: Mapping[tuple[str, str], LabelMapping], dataset: str, diagnosis: str
) -> int:
    entry = mapping.get((dataset, diagnosis))
    if entry is None or not entry.include:
        raise ProtocolGateError("LABEL_MAPPING_REJECTED", {"rejected_rows": 1})
    return entry.target


def _split_by_case(meta_dir: Path, row_count: int) -> dict[int, str]:
    result: dict[int, str] = {}
    for split in ("train", "valid", "test"):
        rows = _read_csv(meta_dir / f"{split}_indexes.csv")
        for row in rows:
            try:
                index = int(row["indexes"])
            except (KeyError, ValueError) as exc:
                raise ProtocolGateError("DERM_SPLIT_INVALID", {"invalid_indexes": 1}) from exc
            if index in result or not 0 <= index < row_count:
                raise ProtocolGateError("DERM_SPLIT_INVALID", {"invalid_indexes": 1})
            result[index] = split
    if set(result) != set(range(row_count)):
        raise ProtocolGateError(
            "DERM_SPLIT_INVALID",
            {"uncovered_cases": len(set(range(row_count)) - set(result))},
        )
    return result


def validate_derm_split_leakage(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, int]:
    """Find case components linked by repeated path or content digest."""

    case_splits: dict[str, str] = {}
    key_cases: dict[tuple[str, str], set[str]] = defaultdict(set)
    key_record_indexes: dict[tuple[str, str], set[int]] = defaultdict(set)
    for position, row in enumerate(rows):
        case = row["case_index"]
        split = row["split"]
        previous = case_splits.setdefault(case, split)
        if previous != split:
            raise ProtocolGateError("DERM_CASE_SPLIT_CONFLICT", {"conflicting_cases": 1})
        for kind, value in (
            ("path", row["image_path"]),
            ("sha256", row["image_sha256"]),
        ):
            key = (kind, value)
            key_cases[key].add(case)
            key_record_indexes[key].add(position)

    parent = {case: case for case in case_splits}

    def find(case: str) -> str:
        while parent[case] != case:
            parent[case] = parent[parent[case]]
            case = parent[case]
        return case

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    repeated_keys = 0
    for cases in key_cases.values():
        if len(cases) > 1:
            repeated_keys += 1
            ordered = sorted(cases)
            for other in ordered[1:]:
                union(ordered[0], other)

    components: dict[str, set[str]] = defaultdict(set)
    for case in case_splits:
        components[find(case)].add(case)
    conflicting = [
        cases
        for cases in components.values()
        if len({case_splits[case] for case in cases}) > 1
    ]
    conflicting_cases = set().union(*conflicting) if conflicting else set()
    conflict_record_indexes: set[int] = set()
    for key, cases in key_cases.items():
        if len(cases) > 1 and len({case_splits[case] for case in cases}) > 1:
            conflict_record_indexes.update(key_record_indexes[key])

    return {
        "repeated_image_keys": repeated_keys,
        "connected_components": len(components),
        "cross_split_conflicting_components": len(conflicting),
        "cross_split_conflicting_cases": len(conflicting_cases),
        "cross_split_conflicting_records": len(conflict_record_indexes),
    }


def build_derm7pt_manifest(
    dataset_dir: Path,
    mapping: Mapping[tuple[str, str], LabelMapping],
    output_path: Path,
) -> dict[str, int]:
    metadata = _read_csv(dataset_dir / "meta/meta.csv")
    splits = _split_by_case(dataset_dir / "meta", len(metadata))
    manifest: list[dict[str, Any]] = []
    for case_index, row in enumerate(metadata):
        diagnosis = row.get("diagnosis", "")
        for domain, column in (("clinic", "clinic"), ("derm", "derm")):
            reference = row.get(column, "")
            image = dataset_dir / "images" / reference
            if not reference or not image.is_file():
                raise ProtocolGateError("DERM_IMAGE_REFERENCE_INVALID", {"invalid_references": 1})
            manifest.append(
                {
                    "case_index": case_index,
                    "split": splits[case_index],
                    "domain": domain,
                    "image_path": str(image.resolve()),
                    "target": _target(mapping, "Derm7pt", diagnosis),
                    "original_diagnosis": diagnosis,
                    "image_sha256": _sha256(image),
                }
            )
    leakage = validate_derm_split_leakage(manifest)
    if leakage["cross_split_conflicting_components"]:
        raise ProtocolGateError(
            "DERM_CROSS_SPLIT_LEAKAGE",
            {
                "conflicting_components": leakage["cross_split_conflicting_components"],
                "conflicting_cases": leakage["cross_split_conflicting_cases"],
                "conflicting_records": leakage["cross_split_conflicting_records"],
            },
        )
    _write_csv(
        output_path,
        (
            "case_index", "split", "domain", "image_path", "target",
            "original_diagnosis", "image_sha256",
        ),
        manifest,
    )
    return {"rows": len(manifest), **leakage}


def validate_pad_grouping(
    rows: Sequence[Mapping[str, str]],
    mapping: Mapping[tuple[str, str], LabelMapping],
) -> dict[str, int]:
    """Validate PAD identities using the approved patient-scoped lesion key."""

    missing_patient = sum(not row.get("patient_id", "").strip() for row in rows)
    missing_lesion = sum(not row.get("lesion_id", "").strip() for row in rows)
    missing_image = sum(not row.get("img_id", "").strip() for row in rows)
    missing_any = sum(
        not row.get("patient_id", "").strip()
        or not row.get("lesion_id", "").strip()
        or not row.get("img_id", "").strip()
        for row in rows
    )

    patients: set[str] = set()
    raw_lesion_patients: dict[str, set[str]] = defaultdict(set)
    lesion_diagnoses: dict[tuple[str, str], set[str]] = defaultdict(set)
    image_lesions: dict[str, set[tuple[str, str]]] = defaultdict(set)
    image_counts: Counter[str] = Counter()
    for row in rows:
        patient = row.get("patient_id", "").strip()
        lesion = row.get("lesion_id", "").strip()
        image = row.get("img_id", "").strip()
        diagnosis = row.get("diagnostic", "").strip()
        if patient:
            patients.add(patient)
        if lesion and patient:
            raw_lesion_patients[lesion].add(patient)
            lesion_diagnoses[(patient, lesion)].add(diagnosis)
        if image:
            image_counts[image] += 1
            if patient and lesion:
                image_lesions[image].add((patient, lesion))

    diagnosis_conflicts = sum(
        len(diagnoses) > 1 for diagnoses in lesion_diagnoses.values()
    )
    binary_targets: dict[tuple[str, str], set[int]] = defaultdict(set)
    for lesion_key, diagnoses in lesion_diagnoses.items():
        for diagnosis in diagnoses:
            binary_targets[lesion_key].add(
                _target(mapping, "PAD-UFES-20", diagnosis)
            )
    binary_conflicts = sum(len(targets) > 1 for targets in binary_targets.values())
    duplicate_images = sum(count > 1 for count in image_counts.values())
    image_mapping_conflicts = sum(
        len(lesion_keys) > 1 for lesion_keys in image_lesions.values()
    )
    summary = {
        "rows": len(rows),
        "patients": len(patients),
        "raw_lesion_ids": len(raw_lesion_patients),
        "lesion_keys": len(lesion_diagnoses),
        "cross_patient_reused_lesion_ids": sum(
            len(patient_ids) > 1
            for patient_ids in raw_lesion_patients.values()
        ),
        "missing_patient_id_rows": missing_patient,
        "missing_lesion_id_rows": missing_lesion,
        "missing_image_id_rows": missing_image,
        "missing_identifier_rows": missing_any,
        "unique_image_keys": len(image_counts),
        "duplicate_image_keys": duplicate_images,
        "image_key_mapping_conflicts": image_mapping_conflicts,
        "diagnosis_inconsistent_lesion_keys": diagnosis_conflicts,
        "binary_target_inconsistent_lesion_keys": binary_conflicts,
    }
    if missing_any:
        raise ProtocolGateError("PAD_MISSING_IDENTIFIERS", summary)
    if image_mapping_conflicts:
        raise ProtocolGateError("PAD_IMAGE_KEY_MAPPING_CONFLICT", summary)
    if duplicate_images:
        raise ProtocolGateError("PAD_DUPLICATE_IMAGE_KEY", summary)
    if diagnosis_conflicts:
        raise ProtocolGateError("PAD_LESION_DIAGNOSIS_CONFLICT", summary)
    if binary_conflicts:
        raise ProtocolGateError("PAD_LESION_BINARY_TARGET_CONFLICT", summary)
    return summary


def build_pad_manifest(
    dataset_dir: Path,
    mapping: Mapping[tuple[str, str], LabelMapping],
    output_path: Path,
) -> dict[str, int]:
    rows = _read_csv(dataset_dir / "metadata.csv")
    summary = validate_pad_grouping(rows, mapping)
    image_files = [
        path for path in (dataset_dir / "images").rglob("*") if path.is_file()
    ]
    image_name_counts = Counter(path.name for path in image_files)
    duplicate_filenames = sum(count > 1 for count in image_name_counts.values())
    if duplicate_filenames:
        raise ProtocolGateError(
            "PAD_IMAGE_FILENAME_CONFLICT",
            {"duplicate_filenames": duplicate_filenames},
        )
    image_by_name = {path.name: path for path in image_files}
    metadata_names = {row["img_id"] for row in rows}
    missing_images = metadata_names - set(image_by_name)
    unreferenced_images = set(image_by_name) - metadata_names
    if missing_images or unreferenced_images:
        raise ProtocolGateError(
            "PAD_IMAGE_INVENTORY_MISMATCH",
            {
                "missing_images": len(missing_images),
                "unreferenced_images": len(unreferenced_images),
            },
        )
    manifest: list[dict[str, Any]] = []
    for row in rows:
        image = image_by_name[row["img_id"]]
        manifest.append(
            {
                "image_id": row["img_id"],
                "patient_id": row["patient_id"],
                "lesion_id": row["lesion_id"],
                "image_path": str(image.resolve()),
                "target": _target(mapping, "PAD-UFES-20", row["diagnostic"]),
                "original_diagnosis": row["diagnostic"],
            }
        )
    _write_csv(
        output_path,
        ("image_id", "patient_id", "lesion_id", "image_path", "target", "original_diagnosis"),
        manifest,
    )
    return summary
