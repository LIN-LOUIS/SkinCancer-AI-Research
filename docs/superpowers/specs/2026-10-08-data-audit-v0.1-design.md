# Data Audit v0.1 Design

## Purpose

Data Audit v0.1 establishes a reproducible, aggregate-only description of
the local Derm7pt and PAD-UFES-20 copies used by the Skin Cancer AI Research
Hub. It prepares the evidence boundary for Paper 1, "Clinical Concept
Reliability under Cross-Domain Shift for Skin Cancer Diagnosis", without
training a model or presenting fabricated experimental results.

The Paper 1 target is melanoma versus non-melanoma. It is not malignant
versus benign: PAD-UFES-20 BCC and SCC are non-melanoma for this task.

## Scope

The audit will:

- inspect dataset metadata and extracted image inventories without modifying
  either dataset;
- validate image references, identity multiplicity, clinical-concept labels,
  and Derm7pt official splits;
- publish aggregate CSV, JSON, Markdown, and PNG artifacts;
- publish dataset cards and an explicit raw-label mapping;
- add automated tests for the reusable audit logic and CLI contract; and
- avoid model training, split generation, patient-level exports, and raw-data
  redistribution.

## Runtime and Data Resolution

The entry point is `scripts/audit_datasets.py` and supports:

```text
python3 scripts/audit_datasets.py
```

The data root is resolved in this order:

1. `SKINCANCER_DATA_ROOT`, when non-empty;
2. `~/Datasets/SkinCancer`.

Expected paths below the root are:

- `raw/derm7pt/release_v0/release_v0/meta/meta.csv`
- `raw/derm7pt/release_v0/release_v0/meta/{train,valid,test}_indexes.csv`
- `raw/derm7pt/release_v0/release_v0/images/`
- `raw/pad_ufes20/metadata.csv`
- `raw/pad_ufes20/images/`

The implementation uses the Python standard library for CSV/JSON/path logic
and Matplotlib for figures. It does not require pandas or seaborn.

## Validation Policy

The script fails with a non-zero exit status when:

- a required file or directory is absent;
- a required metadata column is absent;
- a diagnosis label is absent from the versioned mapping;
- an image reference cannot be resolved;
- an extracted image is unreferenced;
- an image ID is duplicated where uniqueness is required;
- a Derm7pt split has duplicate, overlapping, out-of-range, or uncovered
  metadata indexes; or
- an output cannot be written.

Missing optional metadata values are reported rather than treated as a script
failure. Dataset facts that cannot be established from the local data or an
authoritative local source are marked `TODO` in documentation, not inferred.

## Audit Semantics

### Derm7pt

The sample unit is one metadata case. Each case has one clinical image
reference and one dermoscopy image reference.

The audit computes:

- metadata row count;
- raw diagnosis frequency and percentage;
- melanoma and non-melanoma counts and prevalence;
- clinical, dermoscopy, and combined image-reference counts;
- combined unique referenced image count;
- duplicate-reference excess and duplicated reference names;
- missing referenced images and unreferenced extracted images;
- per-column missing counts and percentages;
- long-form distributions for all seven clinical concepts; and
- official split sizes, per-split uniqueness, pairwise disjointness, index
  bounds, and exact coverage of metadata row indexes.

Duplicate image references mean `total references - unique referenced paths`.
They are not described as missing images.

The following raw labels are melanoma (`target=1`):

- `melanoma`
- `melanoma (in situ)`
- `melanoma (less than 0.76 mm)`
- `melanoma (0.76 to 1.5 mm)`
- `melanoma (more than 1.5 mm)`
- `melanoma metastasis`

Every other observed Derm7pt diagnosis is non-melanoma (`target=0`).

### PAD-UFES-20

The sample unit is one image row. Patient and lesion identifiers are used only
for aggregate multiplicity counts; identifier values are never written to
repository outputs.

The audit computes:

- metadata row count and extracted image count;
- unique patient and lesion counts;
- raw diagnosis frequency and percentage;
- melanoma and non-melanoma counts and prevalence;
- metadata/image set equality;
- duplicate image-ID and duplicate extracted-filename counts;
- number of patients and lesions with more than one image, maximum images per
  patient/lesion, and multiplicity distributions;
- per-column missing counts and percentages; and
- row-level missingness-signature counts plus groups of columns sharing the
  same missing-row set.

Structured missingness is descriptive. The report may state that fields share
an identical missing-row set and show aggregate diagnosis/biopsy composition
for a missingness signature. It must not claim a causal mechanism or assume
values are missing at random.

PAD label mapping is `MEL -> melanoma (1)` and
`ACK/BCC/NEV/SEK/SCC -> non-melanoma (0)`.

## Versioned Label Mapping

`datasets/mappings/melanoma_binary_v0.1.csv` contains one row for every
observed raw diagnosis and exactly these columns:

```text
dataset,original_label,target_name,target,include,reason
```

All currently observed labels are included. `target_name` is either
`melanoma` or `non-melanoma`; `target` is integer `1` or `0`; `include` is
`true`. Reasons explicitly distinguish melanoma membership from general
malignancy.

The audit loads this file at runtime and rejects missing, duplicate, or invalid
mapping rows.

## Output Contracts

All tabular files use UTF-8 and deterministic ordering. Percentages are stored
as numeric fractions in CSV/JSON and rendered as percentages in prose and
figures.

`reports/data_audit/v0.1/` contains:

- `dataset_summary.csv`: one row per dataset with sample/image/entity totals
  and binary target counts/prevalence;
- `derm7pt_diagnosis_distribution.csv`: raw label, count, fraction, and binary
  target;
- `derm7pt_concept_distribution.csv`: concept, raw value, count, and within-
  concept fraction;
- `derm7pt_missingness.csv`: every metadata column with missing count and
  fraction;
- `pad_diagnosis_distribution.csv`: raw label, count, fraction, and binary
  target;
- `pad_missingness.csv`: every metadata column with missing count, fraction,
  and shared missingness-group identifier when applicable;
- `audit_summary.json`: machine-readable schema version, relative input
  locations, metrics, integrity checks, warnings, and structured-missingness
  summaries; and
- `audit_report.md`: human-readable methods, findings, warnings, limitations,
  and Baseline v0.1 readiness assessment.

The committed outputs contain no absolute local data-root path, patient ID,
lesion ID, image filename list, or row-level metadata.

## Figures

`experiments/paper1_concept_dg/figures/data_audit/` contains:

1. `01_dataset_overview.png`
2. `02_diagnosis_distribution.png`
3. `03_melanoma_prevalence.png`
4. `04_derm7pt_concept_distribution.png`
5. `05_pad_metadata_missingness.png`

Figures are English-language PNGs, at least 300 DPI, with explicit titles,
axis labels, readable tick labels, colorblind-considerate restrained colors,
and no 3D, truncated misleading axes, or decorative effects. Counts and
percentages are sourced only from the current audit result. The Git ignore
rules will allow only this versioned audit-figure directory while retaining
the general exclusion for experiment outputs.

## Dataset Cards

`datasets/cards/derm7pt.md` and `datasets/cards/pad_ufes20.md` document:

- modality;
- sample unit;
- diagnosis labels;
- concept annotations;
- available metadata;
- split status;
- known limitations; and
- Paper 1 usage across Domains A, B, and C.

The cards separate facts verified from the local copies from source/licensing
facts. Unverified source details are marked `TODO`.

## Testing Strategy

Tests use small synthetic datasets under pytest temporary directories and do
not read or copy the real medical images. Test-first coverage includes:

- environment-variable and default data-root resolution;
- strict mapping validation and melanoma/non-melanoma aggregation;
- Derm7pt reference accounting and duplicate-reference semantics;
- Derm7pt split integrity, including overlap and incomplete coverage failures;
- PAD image matching, identifier uniqueness, multiplicity, and structured
  missingness;
- aggregate-only output contracts; and
- end-to-end CLI artifact creation from a minimal fixture.

After unit/integration tests pass, the script is run against the local real
datasets. Acceptance checks verify required artifact presence, CSV/JSON
parseability, PNG dimensions/DPI metadata, expected label coverage, and the
absence of raw/sensitive/large files in the Git change set.

## Baseline v0.1 Readiness

Data Audit v0.1 may mark Baseline v0.1 `READY` only when:

- both metadata/image inventories match;
- all observed diagnosis labels have an explicit included mapping;
- Derm7pt official splits pass all integrity checks;
- all required aggregate artifacts are generated successfully; and
- no unresolved issue prevents constructing Domains A, B, and C.

Warnings such as severe class imbalance, repeated images across Derm7pt
modalities, patient/lesion multiplicity, and structured metadata missingness
remain documented even when readiness is `READY`. Any unresolved ambiguity
that changes cohort membership, target labels, or split validity makes the
status `BLOCKED`.

## Git and Privacy Gate

Work occurs on `research/data-audit-v0.1`. Before the implementation commit
and push, the workflow runs `git status` and `git diff --stat`, scans the
changed paths and sizes, and confirms that no raw image, archive,
patient-sensitive export, secret, environment file, or model weight is staged.
The branch is pushed normally; `main` is neither merged nor force-pushed.
