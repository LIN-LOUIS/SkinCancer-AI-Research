# Skin Cancer AI Research Hub

**Survey · Datasets · Baselines · Experiments · Publications**

An open research hub for dermatology and skin cancer AI.

## Research Areas

- Skin lesion classification
- Lesion segmentation
- Clinical severity and grading
- Multimodal learning
- Clinical concept learning
- Domain generalization and adaptation
- OOD detection and uncertainty
- Foundation models
- Dermatopathology and whole-slide images

## Current Research Project

**Clinical Concept Reliability under Domain Shift**

Initial experimental setting:

- Source: Derm7pt
- Target: PAD-UFES-20
- Task: Melanoma vs Non-melanoma
- Baselines: ResNet50, BiomedCLIP, MONET

Status: Dataset preparation and baseline implementation.

## Repository Structure

- `docs/` — Research overview and methodology
- `survey/` — Literature reviews and paper registry
- `datasets/` — Dataset cards, mappings and split definitions
- `baselines/` — Baseline model documentation
- `experiments/` — Reproducible experimental protocols
- `src/` — Shared research code
- `scripts/` — Data auditing and evaluation utilities
- `reports/` — Research and experiment reports
- `papers/` — Research proposals and publication index

## Data Policy

Original medical images, private clinical records and restricted
datasets are not distributed in this repository.

Dataset licenses, access conditions and attribution must be respected.

## Reproducibility

Experimental results will be published with dataset definitions,
splits, configurations, metrics, and code versions where permitted.
