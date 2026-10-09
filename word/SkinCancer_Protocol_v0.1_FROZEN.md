# SkinCancer-AI-Research — Protocol v0.1 (FROZEN)

**Status:** Architecture frozen; implementation and experimental results pending.
**Date:** 2026-10-09
**Base commit:** `0bd3624aea3c8d5401622885ca58fe7e2fc34c07`
**Implementation branch:** `feature/protocol-v0.1` (branch from `research/data-audit-v0.1`)
**Scope:** Paper 1, melanoma (`1`) versus non-melanoma (`0`), not malignant versus benign.
**Data source of truth:** `datasets/mappings/melanoma_binary_v0.1.csv` and Data Audit v0.1.

## 1. Scientific questions and experiment matrix

| ID | Train | Validation | Test | Question |
|---|---|---|---|---|
| E1 | Derm7pt dermoscopy (A_train) | A_val | A_test | Within-dermoscopy baseline |
| E2 | A_train | A_val | Derm7pt clinical (B_test) | Same-dataset, case-paired modality shift |
| E3 | A_train | A_val | PAD-UFES-20 clinical (C_all) | External acquisition + modality + cohort shift |
| E4 | Derm7pt clinical (B_train) | B_val | B_test | Within-clinical baseline |
| E5 | B_train | B_val | C_all | External clinical-photo shift |

Two **trained** source configurations, A and B, yield five evaluations. B_test and A_test have the same case indexes, but different views. E2 and comparisons on A_test/B_test are paired by case; the groups are not statistically independent. PAD C is external evaluation only; never create a PAD validation or use target labels for training, checkpoint selection, threshold tuning, architecture choice, or iterative method selection. Any exploratory PAD-guided revisions must be declared and subsequently verified on a separate untouched target cohort.

## 2. Dataset identities, modalities, and immutability

- Raw data root: `${SKINCANCER_DATA_ROOT:-$HOME/Datasets/SkinCancer}`; `raw/` immutable and outside Git.
- Derm7pt: 1,011 metadata cases; official train/validation/test index sizes 413/203/395; source modality A=`derm`, B=`clinic`; 2,022 image references but only 2,013 unique referenced files. Nine duplicate-reference excess is **not** missing images; analyze exact duplicates and source/test leakage before READY.
- PAD-UFES-20: 2,298 rows/images; 1,373 patients; 1,641 unique raw `lesion_id` values; 1,891 patient-scoped lesion keys; 52 melanoma images. Identity is frozen as `patient_key=patient_id`, `lesion_key=(patient_id, lesion_id)`, and `image_key=img_id`; identifiers remain in the local/private manifest only. All 2,298 images constitute C evaluation, provided they pass cohort/label integrity checks.
- Melanoma `1`; all other mapped diagnoses `0`. PAD BCC and SCC are `0` for this task; do not claim they are benign.
- Audited labels/maps are authoritative; any unknown/new label => FAIL CLOSED.
- Never assume Derm7pt case equals distinct patient; do not describe case-level CI as patient-level CI.

## 3. Manifest contract and leakage gate

Local/private manifests are versioned by schema, not uploaded row-by-row. Required fields:
- Derm7pt: `case_index`, `split`, `domain`, `image_path`, `target`, `original_diagnosis`, `image_sha256` or validated same-file key. `case_index` is the zero-based **metadata row index**, not `case_num` or mostly-empty `case_id`.
- PAD: `image_id`, `patient_id`, `lesion_id`, `image_path`, `target`, `original_diagnosis`; patient/lesion grouping must be validated.
- Use official Derm7pt split unchanged unless data leakage blocks it. All A/B views of any case must share a split. Validate split index coverage/disjointness and every image ref against files.
- Build a graph of cases connected by a repeated image-file reference (and optionally by confirmed same-content digest). If any such component spans official splits, mark Protocol BLOCKED, report aggregate conflict counts **privately**, and request architectural adjudication; do not silently reassign cases or drop examples.
- PAD raw `lesion_id` values may be reused across patients and do not by themselves imply shared identity. Block on missing identity fields, duplicate or contradictory `img_id` mapping, or inconsistent original diagnosis/binary target within one `(patient_id, lesion_id)` lesion key. Do not infer, delete, relabel, or vote on conflicts. Never use a random image-level split for C.
- Only aggregate figures/statistics in public Git. The manifest with patient IDs, image filenames or per-sample predictions belongs to an ignored local directory.

## 4. Model selection, thresholds, and preprocessing

- First baseline: ImageNet-pretrained ResNet50 **if weights are available and provenance is recorded**; otherwise report clearly and block a purported comparable pretrained baseline. Do not silently substitute random initialization.
- Two separate training runs: `resnet50_A`, `resnet50_B`. A/B same architecture, loss specification, seed, evaluation routines, and transparent preprocessing; domain-specific preprocessing only when explicitly frozen.
- Before training, define deterministic train augmentation and nonstochastic validation/test transforms; no information from C should influence normalization choices. Document weight provenance, resolution, learning rate, optimizer, batch size, epoch budget, early stopping and class weighting.
- All checkpoint selection and threshold selection operate **only on source validation**. The model checkpoint for each source/seed is selected by source-val AUROC (ties: earliest epoch). Obtain final source-val scores from that checkpoint, then choose source threshold by maximizing Youden J = sensitivity + specificity - 1; ties by higher specificity, then higher threshold. Lock threshold before any source test/target evaluation. Do not choose a PAD-specific threshold.
- 0.5 threshold may be secondary only for genuine probability outputs; distinguish uncalibrated confidence from calibrated probability.
- The first smoke run is `seed=42`; final comparative baseline seeds `42,43,44`. Same seed set and train budget for each source. Use deterministic behavior when feasible; record unavoidable nondeterminism.
- All validation/test inference should be deterministic. Augmentation only on source train. Stop training if sample IDs or labels are not consistent across source A/B views.

## 5. Metrics and inference records

- Image-/case-level primary: AUROC, average precision (AUPRC), sensitivity (melanoma recall), specificity; report N_pos, N_neg and prevalence alongside AUPRC. Secondary: balanced accuracy, F1, Brier score, ECE (fixed 10 bins; explicitly specify binning convention), confusion matrix.
- Derm7pt A-test and B-test: one prediction per case per image view. For paired comparison report per-case score delta, error transition table, paired bootstrap CI where appropriate. Do not treat A and B as independent patients.
- PAD C: main table includes image-level metrics evaluated on **all** eligible images; 95% uncertainty intervals must use patient-cluster bootstrap (resample patient IDs with replacement, retain all their images), e.g. 2,000 valid replicates with fixed bootstrap seed. If a sample has only one class, discard it and report the number rejected. Flag inadequate valid replicates. As sensitivity analysis report lesion-level aggregation only if lesions have consistent labels and explain aggregation; do not mechanically create patient-level binary labels if a patient has different diagnoses.
- For E3 vs E5, same PAD images and resampled patient draws enable paired confidence intervals of metric differences. Fixed source-val thresholds apply independently to each trained source model.
- Preserve **private** per-sample predictions for audit and paired inference; public tables and plots must be aggregate only. Never mistake disease-prevalence-induced AP changes for pure robustness changes; report prevalence as AP reference and emphasize AUROC/balanced accuracy/class-specific errors.
- With only 52 PAD melanoma images and possibly fewer melanoma patients, use appropriately wide uncertainty intervals and disclose this limitation.

## 6. Execution phases and strict acceptance

### Phase P0: Protocol and manifest implementation — PRIORITY NOW
Deliver protocol YAML, machine-validatable manifests under ignored local data, leakage checks, deterministic metrics, bootstrap code, unit tests, and a run registry. No full model training. `pytest -q` must pass. Running on real data must produce a clear `READY` or `BLOCKED` reason, without dumping patient-level metadata into logs or Git.

### Phase P1: Local ResNet50 smoke — ONLY AFTER P0 READY
Run approximately 20–30 training batches from A_train and B_train (batch or sample budget adjustable to local VRAM), then validate model save/load, inference on A_val/B_val, metric calculation, checkpoint/threshold selection, reproducible run registry, and no unintended GPU OOM. Smoke metrics are **NOT** research performance results. Do not touch PAD labels for optimization. No full training.

### Phase P2: Local full first seed — AFTER P1 READY
Train A and B seed 42 from fixed configs; E1–E5 inference; report full source-validation history, metrics, and plots. Apply target only after threshold lock. Review results before server transfer.

### Phase P3: Server reproduction — AFTER P2 accepted
Transfer only authorized data under dataset license and privacy policy. Remote server should clone exact commit, validate data inventory/hashes, and record GPU, CUDA, PyTorch, code revision and environment. 3 seeds then run with same protocol and explicit differences in hardware. Preserve private prediction files/weights outside Git; export aggregate results.

### Phase P4: Concept analysis — AFTER trustworthy baselines
Derm7pt concept labels are categorical GT; freeze each concept's class mapping and report per-class metrics. Derm7pt dermoscopy-vs-clinical concept shifts need modality-semantic interpretation: a dermoscopic GT may not be directly visible in ordinary clinical photos. PAD does **not** have equivalent seven-concept GT; do not report PAD concept accuracy. Concept shift can be examined descriptively only.

## 7. Artifacts, plots, tracking

- Versioned spec: `docs/protocols/paper1_protocol_v0.1.md`, `experiments/paper1_concept_dg/protocol/protocol_v0.1.yaml`.
- Outputs (versioned aggregate): dataset/split summaries, duplicate-risk report, training curves, E1–E5 metrics matrix (AUC/AP/sensitivity/specificity/prevalence), bootstrap CI, E2 paired predictions transition plot, E3 vs E5 paired metric-difference plot, calibration curves, source/target prevalence chart, representative privacy-cleared errors only with explicit rights.
- Each run: `run_id`, source domain, tests, model/weights provenance, data/protocol version, source split hash, commit SHA, config hash, seed, environment/GPU, epoch/checkpoint, threshold origin/value, timestamp, metrics/CI, artifact paths, state `PENDING | RUNNING | PASS | FAILED | BLOCKED`.
- Publish only aggregate content. No image paths, patient IDs, lesion IDs, raw metadata, pretrained weights, per-image prediction CSV, credentials, or copyrighted medical imagery in Git.

## 8. Implementation boundaries for Codex

Codex executes code, tests, and reporting only. Do not redesign target labels, source/target use, splits, scientific claims, comparison matrix, threshold policy, statistical unit, or evaluation methodology. When a hard gate fails, stop and return `BLOCKED` with minimal aggregated evidence; the ChatGPT research lead will decide protocol revision.

### Immediate acceptance output format (short)
```
BRANCH:
COMMIT:
TESTS: n/n
DERM SPLITS/LEAKAGE: PASS or BLOCKED + aggregate counts
PAD GROUPING: PASS or BLOCKED + aggregate counts
MANIFESTS: local/private paths (no row-level content)
SMOKE: PASS / NOT RUN / BLOCKED
GIT PRIVACY: PASS/BLOCKED
STATUS: READY/BLOCKED
NEXT ACTION:
```
