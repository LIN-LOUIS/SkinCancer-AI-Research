# Paper 1 P2 Seed 42 验收报告

- Git SHA: `2761234f9a6a7961454af48135626092b20dcb10`
- config SHA256: `41420a804f30d8eac6bf6ce63440673c0f31563d6e1b73c7148c8ad964b70d25`
- seed: `42`
- Python: `3.12.7`
- PyTorch: `2.5.1+cu121`
- CUDA runtime: `12.1`
- GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`
- 224×224 AMP 门禁峰值显存: `447057920` bytes
- A/B 单 epoch 预检总耗时: `15.738` seconds

## Source selection

| Source | Best epoch (0-based) | source-val AUROC | threshold | origin | epochs run | peak GPU bytes | train seconds |
|---|---:|---:|---:|---|---:|---:|---:|
| A | 3 | 0.804318 | 0.57666016 | source_val_youden_j | 9 | 686331392 | 49.770 |
| B | 7 | 0.780997 | 0.60253906 | source_val_youden_j | 13 | 686331392 | 72.374 |

## Evaluation

完整指标见 `metrics.csv` / `metrics.json`；所有公开内容均为聚合结果。

| Experiment | rows |
|---|---:|
| E1 | 395 |
| E2 | 395 |
| E3 | 2298 |
| E4 | 395 |
| E5 | 2298 |

## Bootstrap

- requested replicates: 2000
- unit: PAD patient cluster; E3/E5 paired patient draws

**READY**
