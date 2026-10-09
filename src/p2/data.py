from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any, Mapping


class DataGateError(RuntimeError):
    """Fail-closed private-data gate with aggregate-only diagnostics."""

    def __init__(self, code: str, **counts: int) -> None:
        detail = ",".join(f"{key}={value}" for key, value in sorted(counts.items()))
        super().__init__(f"{code}" + (f":{detail}" if detail else ""))
        self.code = code
        self.counts = counts


@dataclass
class DataSeal:
    root: Path
    view_paths: dict[str, Path]
    view_hashes: dict[str, str]
    counts: dict[str, int]
    sha256: str


EXPECTED_COUNTS = {
    "A_train": 413,
    "A_valid": 203,
    "A_test": 395,
    "B_train": 413,
    "B_valid": 203,
    "B_test": 395,
    "C_all": 2298,
}


def _read(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return [{key: (value or "").strip() for key, value in row.items()} for row in csv.DictReader(handle)]
    except OSError as exc:
        raise DataGateError("PRIVATE_MANIFEST_UNAVAILABLE", unavailable_files=1) from exc


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise DataGateError("EMPTY_PRIVATE_VIEW", empty_views=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _seal_digest(counts: Mapping[str, int], hashes: Mapping[str, str]) -> str:
    payload = json.dumps(
        {"counts": dict(counts), "view_hashes": dict(hashes)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(payload).hexdigest()


def prepare_private_views(
    derm_manifest: Path,
    pad_manifest: Path,
    output_root: Path,
) -> DataSeal:
    if output_root.exists():
        raise DataGateError("PRIVATE_OUTPUT_ALREADY_EXISTS", existing_outputs=1)
    derm = _read(derm_manifest)
    pad = _read(pad_manifest)
    required_derm = {
        "case_index", "split", "domain", "image_path", "target",
        "original_diagnosis", "image_sha256",
    }
    required_pad = {
        "image_id", "patient_id", "lesion_id", "image_path", "target",
        "original_diagnosis",
    }
    if not derm or not required_derm.issubset(derm[0]):
        raise DataGateError("DERM_MANIFEST_SCHEMA_INVALID", invalid_manifests=1)
    if not pad or not required_pad.issubset(pad[0]):
        raise DataGateError("PAD_MANIFEST_SCHEMA_INVALID", invalid_manifests=1)
    if any(row.get("target") not in {"0", "1"} for row in derm + pad):
        raise DataGateError("TARGET_DIRECTION_INVALID", invalid_targets=1)

    by_case: dict[str, list[dict[str, str]]] = {}
    for row in derm:
        by_case.setdefault(row["case_index"], []).append(row)
    bad_pairs = 0
    for rows in by_case.values():
        if (
            len(rows) != 2
            or {row["domain"] for row in rows} != {"derm", "clinic"}
            or len({row["split"] for row in rows}) != 1
            or len({row["target"] for row in rows}) != 1
        ):
            bad_pairs += 1
    if bad_pairs:
        raise DataGateError("DERM_PAIR_MISMATCH", mismatched_cases=bad_pairs)

    domain_source = {"derm": "A", "clinic": "B"}
    views: dict[str, list[dict[str, str]]] = {name: [] for name in EXPECTED_COUNTS}
    for row in derm:
        if row["split"] not in {"train", "valid", "test"} or row["domain"] not in domain_source:
            raise DataGateError("DERM_SPLIT_OR_DOMAIN_INVALID", invalid_rows=1)
        name = f"{domain_source[row['domain']]}_{row['split']}"
        views[name].append(row)
    views["C_all"] = pad
    counts = {name: len(rows) for name, rows in views.items()}
    if counts != EXPECTED_COUNTS:
        raise DataGateError("PRIVATE_VIEW_COUNT_MISMATCH", mismatched_views=sum(counts.get(k) != v for k, v in EXPECTED_COUNTS.items()))

    staging = output_root.with_name(output_root.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    staged_paths: dict[str, Path] = {}
    for name, rows in views.items():
        staged_paths[name] = staging / "views" / f"{name}.csv"
        _write(staged_paths[name], rows)
    hashes = {name: _digest(path) for name, path in staged_paths.items()}
    seal_sha = _seal_digest(counts, hashes)
    seal_payload = {
        "schema_version": "p2_data_seal_v1",
        "counts": counts,
        "view_hashes": hashes,
        "sha256": seal_sha,
    }
    (staging / "data_seal.json").write_text(
        json.dumps(seal_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging.replace(output_root)
    final_paths = {name: output_root / "views" / f"{name}.csv" for name in views}
    return DataSeal(output_root, final_paths, hashes, counts, seal_sha)


def load_data_seal(root: Path) -> DataSeal:
    try:
        payload = json.loads((root / "data_seal.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataGateError("DATA_SEAL_INVALID", invalid_seals=1) from exc
    counts = {key: int(value) for key, value in payload.get("counts", {}).items()}
    hashes = {key: str(value) for key, value in payload.get("view_hashes", {}).items()}
    if counts != EXPECTED_COUNTS or payload.get("sha256") != _seal_digest(counts, hashes):
        raise DataGateError("DATA_SEAL_INVALID", invalid_seals=1)
    paths = {name: root / "views" / f"{name}.csv" for name in counts}
    return DataSeal(root, paths, hashes, counts, payload["sha256"])


def _validated_view(seal: DataSeal, name: str) -> list[dict[str, str]]:
    path = seal.view_paths[name]
    try:
        if _digest(path) != seal.view_hashes[name]:
            raise DataGateError("PRIVATE_VIEW_HASH_MISMATCH", mismatched_views=1)
    except OSError as exc:
        raise DataGateError("PRIVATE_VIEW_UNAVAILABLE", unavailable_views=1) from exc
    rows = _read(path)
    if len(rows) != seal.counts[name]:
        raise DataGateError("PRIVATE_VIEW_COUNT_MISMATCH", mismatched_views=1)
    return rows


def load_training_view(seal: DataSeal, source: str) -> dict[str, list[dict[str, str]]]:
    if source not in {"A", "B"}:
        raise DataGateError("SOURCE_INVALID", invalid_sources=1)
    result = {
        "train": _validated_view(seal, f"{source}_train"),
        "valid": _validated_view(seal, f"{source}_valid"),
    }
    missing = sum(
        not Path(row["image_path"]).is_file()
        for rows in result.values()
        for row in rows
    )
    if missing:
        raise DataGateError("SOURCE_IMAGE_MISSING", missing_images=missing)
    return result


def open_evaluation_views(
    seal: DataSeal,
    locks: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    if set(locks) != {"A", "B"}:
        raise DataGateError("BOTH_SOURCE_LOCKS_REQUIRED", valid_locks=0)
    for source in ("A", "B"):
        lock = locks[source]
        if (
            lock.get("source") != source
            or lock.get("data_seal_sha256") != seal.sha256
            or lock.get("threshold_origin") != "source_val_youden_j"
            or lock.get("immutable") is not True
        ):
            raise DataGateError("SOURCE_LOCK_INVALID", invalid_locks=1)
    return {
        name: _validated_view(seal, name)
        for name in ("A_test", "B_test", "C_all")
    }
