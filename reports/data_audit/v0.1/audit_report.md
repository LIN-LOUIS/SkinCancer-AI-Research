# Data Audit v0.1

## Scope and methods

This report was generated from the local metadata and extracted image
inventories by `scripts/audit_datasets.py`. It contains aggregate statistics
only. The target is melanoma versus non-melanoma, not malignant versus benign;
PAD-UFES-20 BCC and SCC are therefore non-melanoma.

## Dataset overview

| Dataset | Metadata rows | Images | Melanoma | Non-melanoma | Prevalence |
|---|---:|---:|---:|---:|---:|
| Derm7pt | 1011 | 2013 | 252 | 759 | 24.93% |
| PAD-UFES-20 | 2298 | 2298 | 52 | 2246 | 2.26% |

## Derm7pt integrity

- Clinical image references: 1011
- Dermoscopy image references: 1011
- Unique referenced images: 2013
- Duplicate-reference excess: 9
- Missing references: 0
- Unreferenced images: 0
- Official split sizes: train=413, valid=203, test=395
- Official split integrity: PASS

## PAD-UFES-20 integrity

- Unique patients: 1373
- Unique lesions: 1641
- Patients with multiple images: 544
- Lesions with multiple images: 512
- Duplicate image-ID excess: 0
- Missing images: 0
- Unreferenced images: 0

## Structured missingness

The following patterns are descriptive and do not establish a missingness
mechanism or justify automatic imputation.

- 804 rows share missingness across: smoke, drink, pesticide, gender, skin_cancer_history, cancer_history, has_piped_water, has_sewage_system, fitspatrick, diameter_1, diameter_2

## Warnings

- Derm7pt contains repeated image references across its clinical and dermoscopy reference columns; these are duplicates, not missing files.
- PAD-UFES-20 melanoma prevalence is below 5%; ordinary accuracy is not suitable as a primary conclusion.
- PAD-UFES-20 contains patients with multiple images; future splits must be patient-aware to avoid leakage.
- PAD-UFES-20 contains lesions with multiple images; lesion-level grouping must be preserved during evaluation.
- PAD-UFES-20 metadata missingness is structured; automatic complete-case filtering or imputation requires a protocol decision.

## Unresolved issues

- PAD-UFES-20 authoritative source and license details remain TODO in the dataset card.

## Baseline v0.1 readiness

**READY**

Readiness covers data inventory, target mapping, and official split integrity.
It does not remove the need for patient-aware evaluation, class-imbalance
handling, calibration analysis, or a frozen experiment protocol.
