# Data Audit v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible, aggregate-only audit of the local Derm7pt and PAD-UFES-20 datasets, then publish its validated reports, figures, dataset cards, and binary-label mapping on the requested research branch.

**Architecture:** `scripts/audit_datasets.py` is a self-contained standard-library audit CLI with small typed functions for input resolution, strict mapping validation, dataset-specific aggregation, artifact writing, and Matplotlib rendering. Pytest fixtures create synthetic dataset trees so every data rule is verified without copying real medical data; the real datasets are read only at the final acceptance gate.

**Tech Stack:** Python 3.10 standard library, Matplotlib 3.5, Pillow 9, pytest 6, Git

**Spec:** `docs/superpowers/specs/2026-10-08-data-audit-v0.1-design.md`

## Global Constraints

- Resolve the data root from non-empty `SKINCANCER_DATA_ROOT`, otherwise `~/Datasets/SkinCancer`.
- Never modify or delete anything below the resolved data root.
- Never emit raw images, row-level metadata, patient IDs, lesion IDs, or image filename lists.
- The Paper 1 target is melanoma versus non-melanoma; PAD BCC and SCC map to non-melanoma.
- Unknown diagnosis labels and integrity violations fail closed with `AuditError`.
- Use UTF-8, deterministic ordering, numeric fractions in CSV/JSON, and English 300-DPI PNG figures.
- Do not train ResNet50 or create model-performance results.
- Mark source facts not verified locally as `TODO`; do not guess.
- Push only `research/data-audit-v0.1`; never merge or force-push `main`.

## Review Focus

- Case-sensitive image names and nested relative paths must match exactly; Task 2 tests a missing case-sensitive reference.
- Empty environment variables must fall back to the expanded default path; Task 1 tests both empty and non-empty values.
- A mapping may be syntactically valid but incomplete for observed labels; Tasks 2 and 3 test rejection of an unmapped diagnosis.
- Missingness signatures must stay aggregate-only even when grouping rows; Task 3 asserts that no identifier value appears in the result.
- Figure generation in a headless/unwritable-home environment must not depend on a user cache; Task 4 runs the CLI with `MPLCONFIGDIR` under a temporary directory.

---

### Task 1: CLI foundations and strict binary mapping

**Files:**
- Create: `scripts/audit_datasets.py`
- Create: `tests/test_audit_datasets.py`
- Create: `datasets/mappings/melanoma_binary_v0.1.csv`

**Interfaces:**
- Produces: `AuditError`, `LabelMapping`, `resolve_data_root(env: Mapping[str, str], home: Path) -> Path`, `load_mapping(path: Path) -> dict[tuple[str, str], LabelMapping]`, and `binary_distribution(dataset: str, rows: Sequence[Mapping[str, str]], diagnosis_column: str, mapping: Mapping[tuple[str, str], LabelMapping]) -> dict[str, Any]`.
- Consumes: the exact mapping schema from the approved spec.

- [ ] **Step 1: Write failing tests for root resolution and mapping validation**

Add `test_resolve_data_root_prefers_nonempty_environment`, `test_resolve_data_root_uses_expanded_default_for_empty_environment`, `test_load_mapping_rejects_duplicate_and_invalid_rows`, and `test_binary_distribution_rejects_unmapped_diagnosis`. Assert exact returned paths and `AuditError` messages that identify the invalid condition without exposing row data.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python3 -m pytest tests/test_audit_datasets.py -k 'resolve_data_root or load_mapping or binary_distribution' -v`

Expected: collection/import failure because `scripts.audit_datasets` does not exist.

- [ ] **Step 3: Implement the minimal foundations**

Create the two immutable dataclasses `LabelMapping` and `AuditPaths`, strict CSV helpers, the four public functions above, and an `AuditError` carrying concise context. Parse `include` only from `true`/`false`, enforce targets `0`/`1`, enforce target names, and reject duplicate `(dataset, original_label)` keys.

- [ ] **Step 4: Add the complete versioned mapping**

Write exactly one mapping row for each of the 20 observed Derm7pt diagnoses and six PAD diagnoses. Map the six specified Derm7pt melanoma labels and PAD `MEL` to `target=1`; map every other label, including PAD `BCC` and `SCC`, to `target=0`.

- [ ] **Step 5: Run focused and full tests and verify GREEN**

Run: `python3 -m pytest tests/test_audit_datasets.py -v`

Expected: all Task 1 tests pass with no warnings.

- [ ] **Step 6: Commit the independently testable mapping foundation**

Run:

```bash
git add scripts/audit_datasets.py tests/test_audit_datasets.py datasets/mappings/melanoma_binary_v0.1.csv
git commit -m "feat(data): define strict melanoma label mapping"
```

### Task 2: Derm7pt audit and official split integrity

**Files:**
- Modify: `scripts/audit_datasets.py`
- Modify: `tests/test_audit_datasets.py`

**Interfaces:**
- Consumes: `LabelMapping`, `binary_distribution`, and an extracted Derm7pt release directory.
- Produces: `audit_derm7pt(dataset_dir: Path, mapping: Mapping[tuple[str, str], LabelMapping]) -> dict[str, Any]` and `validate_official_splits(meta_dir: Path, metadata_rows: int) -> dict[str, Any]`.

- [ ] **Step 1: Add a reusable synthetic Derm7pt fixture and failing happy-path test**

Create three metadata cases, six references, five unique `.jpg` files, all seven concept columns, and disjoint split indexes covering `0..2`. Assert metadata rows, per-modality references, combined duplicate-reference excess of one, zero missing/unreferenced files, concept counts, binary counts, and split sizes.

- [ ] **Step 2: Run the happy-path test and verify RED**

Run: `python3 -m pytest tests/test_audit_datasets.py::test_audit_derm7pt_counts_references_concepts_and_splits -v`

Expected: FAIL because `audit_derm7pt` is absent.

- [ ] **Step 3: Implement Derm7pt aggregation**

Validate the 19 required columns, normalize only surrounding whitespace in metadata values, enumerate regular files below `images/` as POSIX relative paths, keep reference matching case-sensitive, calculate per-column missingness, and return only aggregate counts plus the duplicate-reference excess.

- [ ] **Step 4: Write failing integrity tests**

Add `test_audit_derm7pt_rejects_case_mismatched_or_missing_image`, `test_validate_official_splits_rejects_overlap`, `test_validate_official_splits_rejects_duplicate_out_of_range_and_incomplete_coverage`, and `test_audit_derm7pt_rejects_unmapped_diagnosis`.

- [ ] **Step 5: Run integrity tests and verify RED**

Run: `python3 -m pytest tests/test_audit_datasets.py -k 'derm7pt or official_splits' -v`

Expected: each new negative case fails because the corresponding validation is not implemented.

- [ ] **Step 6: Implement official split and fail-closed validation**

Read the `indexes` column as integers; enforce per-split uniqueness, pairwise disjointness, range `0 <= index < metadata_rows`, and exact union with `range(metadata_rows)`. Raise `AuditError` for image mismatch, unreferenced images, or an unmapped label.

- [ ] **Step 7: Run focused and full tests and verify GREEN**

Run: `python3 -m pytest tests/test_audit_datasets.py -v`

Expected: all Task 1-2 tests pass.

- [ ] **Step 8: Commit the Derm7pt audit**

Run:

```bash
git add scripts/audit_datasets.py tests/test_audit_datasets.py
git commit -m "feat(data): add Derm7pt integrity audit"
```

### Task 3: PAD-UFES-20 audit and structured missingness

**Files:**
- Modify: `scripts/audit_datasets.py`
- Modify: `tests/test_audit_datasets.py`

**Interfaces:**
- Consumes: `LabelMapping`, `binary_distribution`, and an extracted PAD-UFES-20 directory.
- Produces: `audit_pad_ufes20(dataset_dir: Path, mapping: Mapping[tuple[str, str], LabelMapping]) -> dict[str, Any]`, `summarize_multiplicity(values: Iterable[str]) -> dict[str, Any]`, and `summarize_structured_missingness(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]`.

- [ ] **Step 1: Add a synthetic PAD fixture and failing aggregate test**

Create four image rows with three patients, three lesions, one `MEL`, one repeated patient, one repeated lesion, and a shared two-column missingness signature. Assert image-set equality, entity totals, multi-image entity totals/maxima/distributions, binary prevalence, per-column missingness, and grouped signature counts.

- [ ] **Step 2: Run the aggregate test and verify RED**

Run: `python3 -m pytest tests/test_audit_datasets.py::test_audit_pad_reports_matching_multiplicity_and_structured_missingness -v`

Expected: FAIL because `audit_pad_ufes20` is absent.

- [ ] **Step 3: Implement PAD aggregation**

Validate all 26 required columns; compare non-empty `img_id` values with extracted image basenames; compute unique patient/lesion totals, multiplicity histograms, and missingness signatures using row indexes internally; expose only counts, column names, diagnoses, and biopsy-value aggregates.

- [ ] **Step 4: Write failing privacy and integrity tests**

Add `test_pad_summary_does_not_contain_patient_lesion_or_image_identifiers`, `test_audit_pad_rejects_duplicate_image_ids`, `test_audit_pad_rejects_missing_or_unreferenced_images`, and `test_audit_pad_rejects_unmapped_diagnosis`.

- [ ] **Step 5: Run the new tests and verify RED**

Run: `python3 -m pytest tests/test_audit_datasets.py -k 'pad' -v`

Expected: each unsupported validation or privacy assertion fails for the intended reason.

- [ ] **Step 6: Complete PAD fail-closed behavior**

Reject duplicate/empty image IDs, duplicate extracted basenames, unequal metadata/image sets, and unmapped labels. Ensure identifier values never occur in the returned aggregate dictionary.

- [ ] **Step 7: Run focused and full tests and verify GREEN**

Run: `python3 -m pytest tests/test_audit_datasets.py -v`

Expected: all Task 1-3 tests pass.

- [ ] **Step 8: Commit the PAD audit**

Run:

```bash
git add scripts/audit_datasets.py tests/test_audit_datasets.py
git commit -m "feat(data): add PAD-UFES-20 integrity audit"
```

### Task 4: Deterministic reports, figures, and end-to-end CLI

**Files:**
- Modify: `scripts/audit_datasets.py`
- Modify: `tests/test_audit_datasets.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: both dataset audit dictionaries and the approved output schema.
- Produces: `write_csv_reports(summary: Mapping[str, Any], report_dir: Path) -> list[Path]`, `write_json_summary(summary: Mapping[str, Any], path: Path) -> Path`, `build_markdown_report(summary: Mapping[str, Any]) -> str`, `render_figures(summary: Mapping[str, Any], figure_dir: Path) -> list[Path]`, `run_audit(data_root: Path, repo_root: Path) -> dict[str, Any]`, and `main() -> int`.

- [ ] **Step 1: Write failing output-contract tests**

Add `test_write_outputs_creates_all_eight_report_files`, `test_report_outputs_are_deterministic_and_aggregate_only`, and `test_render_figures_creates_five_300_dpi_english_pngs`. Parse every CSV/JSON output and use Pillow to assert PNG format, dimensions of at least 1800 pixels on the long side, and approximately 300 DPI metadata.

- [ ] **Step 2: Run output tests and verify RED**

Run: `python3 -m pytest tests/test_audit_datasets.py -k 'write_outputs or report_outputs or render_figures' -v`

Expected: FAIL because writers and renderers are absent.

- [ ] **Step 3: Implement deterministic writers and report prose**

Use `csv.DictWriter` with fixed field lists, `json.dump(..., sort_keys=True, indent=2)`, stable label/concept ordering, and a fixed `schema_version="0.1"`. Include warnings and a deterministic `READY`/`BLOCKED` evaluation, but omit wall-clock timestamps so reruns with unchanged inputs are byte-stable.

- [ ] **Step 4: Implement restrained publication figures**

Use the non-interactive `Agg` backend, a colorblind-considerate blue/orange palette, zero-based quantitative axes, direct count/percentage labels, `tight_layout`, and `savefig(..., dpi=300)`. Close every figure after writing.

- [ ] **Step 5: Write and run the failing end-to-end CLI test**

Add `test_cli_uses_environment_root_and_creates_required_artifacts`; run the script in a subprocess against the combined synthetic fixture with a temporary `MPLCONFIGDIR`. Assert exit code zero and the exact 13 required artifact names.

Run: `python3 -m pytest tests/test_audit_datasets.py::test_cli_uses_environment_root_and_creates_required_artifacts -v`

Expected: FAIL until `run_audit` and `main` connect all components.

- [ ] **Step 6: Complete CLI wiring and figure ignore exception**

Locate the repository root from `Path(__file__).resolve().parents[1]`, load the versioned mapping, write into the fixed report/figure directories, print a concise aggregate summary, and return exit code two with an `ERROR:` line for `AuditError`. Add a narrow `.gitignore` negation for `experiments/paper1_concept_dg/figures/data_audit/*.png` only.

- [ ] **Step 7: Run focused and full tests and verify GREEN**

Run: `MPLCONFIGDIR=/tmp/skincancer-audit-mpl python3 -m pytest -q`

Expected: all tests pass with zero failures and zero errors.

- [ ] **Step 8: Commit the artifact pipeline**

Run:

```bash
git add .gitignore scripts/audit_datasets.py tests/test_audit_datasets.py
git commit -m "feat(data): generate deterministic audit artifacts"
```

### Task 5: Dataset cards, real-data execution, acceptance, and publication

**Files:**
- Create: `datasets/cards/derm7pt.md`
- Create: `datasets/cards/pad_ufes20.md`
- Create: `reports/data_audit/v0.1/dataset_summary.csv`
- Create: `reports/data_audit/v0.1/derm7pt_diagnosis_distribution.csv`
- Create: `reports/data_audit/v0.1/derm7pt_concept_distribution.csv`
- Create: `reports/data_audit/v0.1/derm7pt_missingness.csv`
- Create: `reports/data_audit/v0.1/pad_diagnosis_distribution.csv`
- Create: `reports/data_audit/v0.1/pad_missingness.csv`
- Create: `reports/data_audit/v0.1/audit_summary.json`
- Create: `reports/data_audit/v0.1/audit_report.md`
- Create: `experiments/paper1_concept_dg/figures/data_audit/01_dataset_overview.png`
- Create: `experiments/paper1_concept_dg/figures/data_audit/02_diagnosis_distribution.png`
- Create: `experiments/paper1_concept_dg/figures/data_audit/03_melanoma_prevalence.png`
- Create: `experiments/paper1_concept_dg/figures/data_audit/04_derm7pt_concept_distribution.png`
- Create: `experiments/paper1_concept_dg/figures/data_audit/05_pad_metadata_missingness.png`

**Interfaces:**
- Consumes: the verified CLI, real read-only datasets, and generated aggregate summaries.
- Produces: the complete Data Audit v0.1 deliverable and the pushed research branch.

- [ ] **Step 1: Write both dataset cards from verified local facts**

Document modality, sample unit, diagnoses, concepts, metadata, split, limitations, and Paper 1 Domain A/B/C usage. Include Derm7pt's locally documented CC BY-NC-ND 4.0/non-redistribution constraint; mark PAD source/license details `TODO: verify against the authoritative dataset publication/repository` rather than guessing.

- [ ] **Step 2: Run the complete test suite fresh**

Run: `MPLCONFIGDIR=/tmp/skincancer-audit-mpl python3 -m pytest -q`

Expected: all tests pass with zero failures and zero errors.

- [ ] **Step 3: Run the audit against the real local data**

Run: `MPLCONFIGDIR=/tmp/skincancer-audit-mpl SKINCANCER_DATA_ROOT=/home/lin/Datasets/SkinCancer python3 scripts/audit_datasets.py`

Expected: exit zero, eight report files and five PNGs produced, with no writes under `/home/lin/Datasets/SkinCancer`.

- [ ] **Step 4: Verify machine-readable outputs and expected invariants**

Run a read-only Python acceptance check that parses all six CSV files and JSON, opens all five PNGs with Pillow, and asserts at minimum: Derm7pt 1,011 rows, 2,013 unique images, nine duplicate-reference excess, zero missing/unreferenced images, 252 melanoma cases, 413/203/395 splits; PAD 2,298 rows/images, 1,373 patients, 1,641 lesions, 52 melanoma cases, zero metadata/image mismatch, zero duplicate image IDs, and 804-row shared structured-missingness signature.

Expected: exit zero and a printed `REAL_DATA_ACCEPTANCE_OK`.

- [ ] **Step 5: Verify reproducibility**

Hash the 13 generated artifacts, rerun the audit, hash them again, and compare manifests outside the repository.

Expected: identical SHA-256 manifests.

- [ ] **Step 6: Run source and repository hygiene checks**

Run `python3 -m compileall -q scripts tests`, `git diff --check`, `git status --short`, `git diff --stat`, list all changed/staged paths with byte sizes, and search changed text for token/password/private-key patterns and patient/lesion identifier values.

Expected: no syntax errors, whitespace errors, secrets, archives, model files, raw images outside the five intended figures, patient-sensitive exports, or unexpectedly large files.

- [ ] **Step 7: Stage the exact final allowlist and inspect it**

Stage only the two cards, eight reports, five figures, mapping if not already committed, script/tests, `.gitignore`, spec, and plan. Run `git status`, `git diff --cached --stat`, `git diff --cached --check`, and inspect `git diff --cached --name-only` before committing.

Expected: the staged set contains only approved Data Audit v0.1 sources and aggregate artifacts.

- [ ] **Step 8: Create the requested implementation commit**

Run:

```bash
git commit -m "feat(data): add reproducible Derm7pt and PAD-UFES-20 audit"
```

Expected: a non-empty commit on `research/data-audit-v0.1`.

- [ ] **Step 9: Re-run the push gate and push normally**

Run `git status`, `git diff --stat`, confirm a clean worktree, then run `git push -u origin research/data-audit-v0.1` without force.

Expected: GitHub accepts the branch; otherwise report the exact transport/authentication failure and leave the verified local commit intact.

- [ ] **Step 10: Report the final evidence**

Return the HEAD commit SHA, branch, push result, changed files, audit warnings/unresolved issues, ten most important findings, and evidence-based `READY` or `BLOCKED` status for Baseline v0.1.
