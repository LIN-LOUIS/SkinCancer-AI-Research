from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_datasets import (
    AuditError,
    audit_derm7pt,
    audit_pad_ufes20,
    evaluate_baseline_readiness,
    load_mapping,
    resolve_data_root,
)
from src.data.protocol_v0_1 import (
    ProtocolGateError,
    build_derm7pt_manifest,
    build_pad_manifest,
)


def write_aggregate_acceptance_report(
    result: dict[str, Any], output_path: Path, *, smoke_status: str
) -> None:
    """Write an allowlisted aggregate-only P0/P1 acceptance summary."""

    derm = result["derm"]
    pad = result["pad"]
    p0_status = result["status"]
    overall_status = "READY" if p0_status == "READY" and smoke_status == "PASS" else "BLOCKED"
    raw_lesions = int(pad["raw_lesion_ids"])
    lesion_keys = int(pad["lesion_keys"])
    report = f"""# Protocol v0.1 补充决议验收报告

本报告仅含聚合统计，不包含患者 ID、病灶 ID、图像 ID、文件名、路径或逐样本预测。

## 身份规则

- `patient_key = patient_id`
- `lesion_key = (patient_id, lesion_id)`
- `image_key = img_id`

## PAD-UFES-20 聚合验收

| 检查项 | 数量 |
|---|---:|
| 元数据行 | {int(pad['rows']):,} |
| 患者键 | {int(pad['patients']):,} |
| 原始 `lesion_id` 唯一值 | {raw_lesions:,} |
| 跨患者复用的 `lesion_id` | {int(pad['cross_patient_reused_lesion_ids']):,} |
| 联合键唯一病灶 | {lesion_keys:,} |
| 联合键内原始诊断冲突 | {int(pad['diagnosis_inconsistent_lesion_keys']):,} |
| 联合键内二分类标签冲突 | {int(pad['binary_target_inconsistent_lesion_keys']):,} |
| 缺失 `patient_id` 的行 | {int(pad['missing_patient_id_rows']):,} |
| 缺失 `lesion_id` 的行 | {int(pad['missing_lesion_id_rows']):,} |
| 缺失 `img_id` 的行 | {int(pad['missing_image_id_rows']):,} |
| 唯一图像键 | {int(pad['unique_image_keys']):,} |
| 重复图像键 | {int(pad['duplicate_image_keys']):,} |
| 图像键到多个病灶键的映射冲突 | {int(pad['image_key_mapping_conflicts']):,} |

PAD 分组门禁：**{p0_status}**。

## 与 Data Audit v0.1 的口径差异

Data Audit v0.1 按原始 `lesion_id` 统计 {raw_lesions:,} 个唯一值。本补充决议按
`(patient_id, lesion_id)` 联合键统计 {lesion_keys:,} 个唯一病灶，差异为
{lesion_keys - raw_lesions:+,}。这是身份统计口径修正；未修改原始 metadata，跨患者复用
`lesion_id` 本身不再构成阻塞。

## Derm7pt 与 Smoke

- Derm7pt manifest 行：{int(derm['manifest_rows']):,}
- 跨 split 冲突组件：{int(derm['cross_split_conflicting_components']):,}
- 跨 split 冲突病例：{int(derm['cross_split_conflicting_cases']):,}
- 跨 split 冲突记录：{int(derm['cross_split_conflicting_records']):,}
- ResNet50 seed=42 Smoke：**{smoke_status}**

## 结论

**{overall_status}**
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(output_path)


def run_p0(
    *,
    derm_dir: Path,
    pad_dir: Path,
    mapping_path: Path,
    local_root: Path,
) -> dict[str, Any]:
    """Run P0 gates and write row-level artifacts only below local_root."""

    mapping = load_mapping(mapping_path)
    derm_audit = audit_derm7pt(derm_dir, mapping)
    pad_audit = audit_pad_ufes20(pad_dir, mapping)
    readiness, issues = evaluate_baseline_readiness(derm_audit, pad_audit)
    if readiness != "READY":
        raise ProtocolGateError("DATA_AUDIT_NOT_READY", {"readiness_issues": len(issues)})

    manifest_dir = local_root / "manifests"
    derm = build_derm7pt_manifest(
        derm_dir, mapping, manifest_dir / "derm7pt_v0.1.csv"
    )
    pad = build_pad_manifest(
        pad_dir, mapping, manifest_dir / "pad_ufes20_v0.1.csv"
    )
    result: dict[str, Any] = {
        "protocol_version": "0.1",
        "status": "READY",
        "derm": {
            "manifest_rows": derm["rows"],
            "repeated_image_keys": derm["repeated_image_keys"],
            "connected_components": derm["connected_components"],
            "cross_split_conflicting_components": derm[
                "cross_split_conflicting_components"
            ],
            "cross_split_conflicting_cases": derm["cross_split_conflicting_cases"],
            "cross_split_conflicting_records": derm[
                "cross_split_conflicting_records"
            ],
        },
        "pad": pad,
        "manifests": {
            "derm7pt": "manifests/derm7pt_v0.1.csv",
            "pad_ufes20": "manifests/pad_ufes20_v0.1.csv",
        },
    }
    local_root.mkdir(parents=True, exist_ok=True)
    acceptance_path = local_root / "p0_acceptance.json"
    acceptance_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen Protocol v0.1 P0 gates")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--local-root", type=Path)
    parser.add_argument("--aggregate-report", type=Path)
    parser.add_argument(
        "--smoke-status",
        choices=("PASS", "NOT_RUN", "BLOCKED", "BLOCKED_BY_ENV"),
        default="NOT_RUN",
    )
    arguments = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    data_root = arguments.data_root or resolve_data_root({}, Path.home())
    local_root = arguments.local_root or repo_root / "data_local/protocol_v0.1"
    try:
        result = run_p0(
            derm_dir=data_root / "raw/derm7pt/release_v0/release_v0",
            pad_dir=data_root / "raw/pad_ufes20",
            mapping_path=repo_root / "datasets/mappings/melanoma_binary_v0.1.csv",
            local_root=local_root,
        )
    except (AuditError, ProtocolGateError) as exc:
        print(f"P0 BLOCKED: {exc}", file=sys.stderr)
        return 2
    if arguments.aggregate_report:
        write_aggregate_acceptance_report(
            result,
            arguments.aggregate_report,
            smoke_status=arguments.smoke_status,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
