# Protocol v0.1 补充决议验收报告

本报告仅含聚合统计，不包含患者 ID、病灶 ID、图像 ID、文件名、路径或逐样本预测。

## 身份规则

- `patient_key = patient_id`
- `lesion_key = (patient_id, lesion_id)`
- `image_key = img_id`

## PAD-UFES-20 聚合验收

| 检查项 | 数量 |
|---|---:|
| 元数据行 | 2,298 |
| 患者键 | 1,373 |
| 原始 `lesion_id` 唯一值 | 1,641 |
| 跨患者复用的 `lesion_id` | 250 |
| 联合键唯一病灶 | 1,891 |
| 联合键内原始诊断冲突 | 0 |
| 联合键内二分类标签冲突 | 0 |
| 缺失 `patient_id` 的行 | 0 |
| 缺失 `lesion_id` 的行 | 0 |
| 缺失 `img_id` 的行 | 0 |
| 唯一图像键 | 2,298 |
| 重复图像键 | 0 |
| 图像键到多个病灶键的映射冲突 | 0 |

PAD 分组门禁：**READY**。

## 与 Data Audit v0.1 的口径差异

Data Audit v0.1 按原始 `lesion_id` 统计 1,641 个唯一值。本补充决议按
`(patient_id, lesion_id)` 联合键统计 1,891 个唯一病灶，差异为
+250。这是身份统计口径修正；未修改原始 metadata，跨患者复用
`lesion_id` 本身不再构成阻塞。

## Derm7pt 与 Smoke

- Derm7pt manifest 行：2,022
- 跨 split 冲突组件：0
- 跨 split 冲突病例：0
- 跨 split 冲突记录：0
- ResNet50 seed=42 Smoke：**PASS**

## 结论

**READY**
