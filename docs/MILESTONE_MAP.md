# NeuroPrompt-3D Milestone Map

## Purpose

This document provides the human-readable historical milestone map for
NeuroPrompt-3D through the completion of M10.

Existing filenames, folders, milestone identifiers, frozen archives,
checksums, notebooks, and experiment paths are not renamed by this
document. Historical scientific identifiers remain authoritative.

The zero-padded forms M01-M09 may be used in human-facing documentation,
but historical identifiers such as M7KC, M9E7, and M10G remain unchanged.

---

# M1 — Synthetic 3D MRI Pipeline

Established the initial synthetic 3D medical-imaging workflow before
real BraTS patient data was introduced.

Git foundation:

- `4d94eb1` — Build synthetic 3D MRI pipeline

---

# M2 — Four-Modality Synthetic MRI

Extended the synthetic pipeline to a four-modality MRI representation.

Git foundation:

- `ceb37c3` — Add four-modality synthetic MRI pipeline

---

# M3 — Real BraTS Multimodal Case Validation

Transitioned the project from synthetic MRI to real multimodal BraTS
cases.

Historical README stages:

- Milestone 3 Step 1 — strict multimodal case definition
- Milestone 3 Step 2 — separate NIfTI modality loading
- Milestone 3 Step 3 — real BraTS case validation

Git foundations:

- `a27f664` — Add strict multimodal case definition
- `9ce2a02` — Load and validate multimodal NIfTI cases
- `53c0ada` — Document real BraTS validation checkpoint

---

# M4 — Foreground MRI Normalization

Introduced foreground-aware MRI normalization for real multimodal
volumes.

Git foundation:

- `dc0feb3` — Add multimodal MRI normalization

---

# M5 — Leakage-Safe BraTS Cohort Splitting

Established frozen subject-level train, validation, and test
assignments to prevent subject leakage.

Canonical split:

`splits/brats2024_posttreatment_seed42.json`

Git foundation:

- `95ea657` — Add leakage-safe BraTS subject-level splits

---

# M6 — Model-Ready BraTS Preprocessing and 3D Patch Sampling

Extended the frozen cohort assignments into model-ready PyTorch
training samples.

Major work included:

- multimodal MRI preprocessing
- synchronized segmentation targets
- deterministic 3D patch sampling
- model-ready dataset construction from the frozen split

Git foundations:

- `a6acad5` — Complete Milestone 6 BraTS preprocessing and patch sampling
- `f1ade44` — Fix Milestone 6 seeded patch candidate selection

---

# M7 — Baseline 3D U-Net Engineering, Training and Evaluation

M7 was the largest baseline-model development phase and evolved through
multiple engineering, readiness, execution, and evaluation stages.

## M7A — Lightweight 3D U-Net Training Foundation

Implemented the lightweight model, loss functions, and fundamental
training engine.

Git foundation:

- `cb4ac36` — Add lightweight 3D U-Net training foundation

## M7B — Frozen BraTS Train/Validation Loaders

Created dataset loaders tied to the frozen BraTS split.

Git foundation:

- `571852d` — Add frozen BraTS training and validation loaders

## M7C — Sliding-Window Validation Inference

Added full-volume validation inference through sliding windows.

Git foundation:

- `94a2b53` — Add sliding-window validation inference

## M7D — Reproducible Baseline Training Configuration

Established explicit, reproducible baseline training configuration and
setup.

Git foundation:

- `b5da95d` — Add reproducible baseline training configuration

## M7E — CPU/CUDA Device-Aware Training

Added explicit device selection and CPU/CUDA-aware execution.

Git foundation:

- `3f27104` — Add CPU and CUDA device-aware training

## M7F — Training Orchestration, History and Checkpointing

Added baseline-training orchestration, checkpoint handling, history,
runner logic, and workflow integration.

Git foundation:

- `fa61c99` — Add baseline training orchestration and checkpoints

## M7G — Executable Baseline Training CLI

Exposed baseline training through an executable command-line path.

Git foundation:

- `093c1d2` — Add executable baseline training CLI

## M7H — Real BraTS One-Case Smoke-Training Readiness

Historical project-chat records describe M7H as the real-data
one-case training-readiness check, including a real BraTS training case
passing through preprocessing, positive 3D patch extraction, the
lightweight 3D U-Net, loss computation, backward propagation, and an
optimizer update.

This historical label is retained as part of the development record
even though no surviving canonical file currently carries `M7H` in its
filename.

## M7I — Full Train/Validation Dataset Readiness

Historical project-chat records describe M7I as the full frozen
train/validation dataset-preparation and readiness stage before GPU
execution.

This historical label is retained as part of the development record
even though no surviving canonical file currently carries `M7I` in its
filename.

## M7J — GPU Smoke Verification

Verified the baseline-training workflow on GPU before moving to the
full-scale training program.

Canonical notebook:

`notebooks/NeuroPrompt3D_M7J_GPU_Smoke_2026-09-10.ipynb`

## M7K — Full-Scale Baseline Execution Program

M7K expanded into multiple sub-stages rather than remaining a single
training step.

### M7K-A — CUDA Resume Smoke / Interruption-Safe Resume

Added resumable training-state persistence and robust resume handling.

Git foundations:

- `c306454` — Add resumable training state persistence
- `40465c6` — Add robust training resume workflow

Canonical later notebook:

`notebooks/NeuroPrompt3D_7KA_CUDA_Resume_Smoke_2026_09_11.ipynb`

### M7K-B — Feasibility and Staging

Historical evidence indicates a staged feasibility sequence.

- M7K-B1 — early/local feasibility benchmark
- M7K-B2 — T4 feasibility benchmark
- M7K-B3 — multi-case representativeness benchmark
- M7K-B4 — full train/validation dataset staging

Canonical surviving notebooks include:

- `NeuroPrompt3D_7KB2_T4_Feasibility_Benchmark_2026_09_11.ipynb`
- `NeuroPrompt3D_7KB3_MultiCase_Representativeness_Benchmark_2026_09_11.ipynb`
- `NeuroPrompt3D_M7KB4_M7KC_Full_Data_Staging_and_Baseline_Training_2026-09-13.ipynb`

The S4 historical audit did not locate a surviving file explicitly
named M7K-B1. Therefore M7K-B1 is documented as a historical stage, not
as a currently surviving named artifact.

### M7K-C — Full Baseline Training

Performed the main full baseline-training run and training resume.

Canonical locations:

- `checkpoints/M7K_C_Full_Baseline_Training/`
- `notebooks/NeuroPrompt3D_M7KC_Baseline_Training_Resume_2026_09_13.ipynb`

### M7K-D — Baseline Evaluation

M7K-D became the completed baseline-evaluation sequence:

- M7K-D1 — Evaluation Protocol and Metric Verification
- M7K-D2 — One-Case Baseline Evaluation
- M7K-D3 — Full Validation Baseline Evaluation

Canonical output family:

`outputs/M7K_D_Baseline_Evaluation/`

## Earlier M7L / M7M Planning Labels

An earlier development roadmap reserved later labels for baseline
evaluation and closure.

The executed project instead expanded M7K into M7K-C and M7K-D
training/evaluation stages.

No separate completed canonical artifact carrying the literal `M7L`
or `M7M` identifier was identified in the pre-M11 provenance audit.

The documentation therefore preserves the difference between the
earlier plan and the actual execution rather than inventing completed
M7L/M7M stages.

---

# M8 — MC-Dropout Uncertainty and Automatic Prompting

## M8A — One-Case MC-Dropout Verification

Verified stochastic inference and predictive-variance generation on one
case.

## M8B — Representative Development Characterization

### M8B1 — Representative Subset Selection and Staging

Selected and staged the representative development subset.

### M8B2 — 9-Case MC-Dropout Characterization

Characterized MC-Dropout behavior across the selected nine-case
development subset.

Canonical output family:

`outputs/M8_MC_Dropout_Uncertainty/`

## M8C — Automatic Prompt Development

### M8C1 — Prompt Protocol and One-Case Verification

Established and tested the automatic-prompt protocol.

### M8C2 — 9-Case Automatic Prompt Characterization

Characterized prompt behavior across the development subset.

### M8C3 — Prompt Failure Diagnosis and Holdout Freeze

Diagnosed prompt failure modes and established the development/holdout
boundary.

### M8C4 — Dual-Hypothesis Prompt Development

Investigated revised prompt hypotheses using development evidence only.

### M8C5 — V2 Protocol and SAM Interface Freeze

Froze the v2 automatic-prompt protocol and downstream SAM-Med3D
interface contract.

Canonical output family:

`outputs/M8_Automatic_Prompting/`

---

# M9 — SAM-Med3D Integration, Refinement Policy and Final Test

## M9A — SAM-Med3D Integration

### M9A1 — Source and Checkpoint Provenance

Recorded the external SAM-Med3D source and checkpoint identity.

### M9A2 — Source and Interface Audit

Audited the external implementation and model interface.

### M9A3 — Preprocessing and Coordinate Adapter Verification

Verified conversion between NeuroPrompt-3D volumes/prompts and
SAM-Med3D input coordinates.

### M9A4 — One-Case GPU Smoke Test

Verified actual SAM-Med3D GPU inference on a development case.

## M9B — Development-Branch Characterization

### M9B1 — 9-Case Input Payload Freeze

Froze the development-case SAM input payloads before branch inference.

### M9B2 — 9-Case Pre-GT SAM Branch Output Freeze

Generated and froze SAM branch outputs before GT-assisted evaluation.

### M9B3 — Development GT Branch Characterization

Used development ground truth to characterize branch behavior.

## M9C — Localized Refinement Policy Development and Freeze

### M9C1 — Localized Directional SAM Compositing Diagnosis

Diagnosed localized compositing strategies and safe refinement
directions.

### M9C2 — Compositing and Variance-Gate Freeze

Froze the localized compositing protocol and predictive-variance gate.

### M9C3 — Final Automatic Action Policy Freeze

Froze the final automatic action policy and test firewall.

An accidental M9B3 rerun preserved under an explicitly `INVALID`
notebook name is historical evidence only and is not part of the valid
final policy.

### M9C4 — End-to-End Pipeline Freeze Verification

M9C4 verified the complete frozen automatic-refinement pipeline.

Archived evidence includes:

- M9C4A CPU audit
- M9C4B GPU end-to-end smoke
- master freeze record

The frozen source commit used for subsequent final-test work is:

`66edbba5a07144b3b6a2f117d9fec22ea52284ef`

## M9D — 12-Case Frozen Holdout

### M9D1 — Pre-GT Holdout Prediction Freeze

Generated and froze predictions before holdout GT evaluation.

### M9D2 — Post-Freeze GT Evaluation

Evaluated the already-frozen holdout predictions against GT.

The M9D subset was a validation/holdout experiment and is not an
independent final test.

## M9E — Primary 129-Case Frozen Final Test

M9E is the primary final-test experiment.

### M9E1 — Final-Test Protocol Freeze

Defined and froze the final-test execution protocol.

### M9E2 — MRI-Only Final-Test Inference

Executed the frozen pipeline on all 129 test cases before GT access.

A documented Amendment A was applied to the execution procedure before
the final valid pre-GT prediction freeze. Both pre-amendment and
Amendment A execution notebooks are retained for provenance.

### M9E3 — Pre-GT Global Prediction Freeze

Froze the complete 129-case prediction set before first GT access.

### M9E4 — Local Pre-GT Verification

Verified the frozen prediction package locally.

### M9E5 — First GT Access

Recorded the controlled first access to final-test ground truth.

### M9E6 — Final-Test Evaluation

Evaluated the already-frozen predictions.

### M9E7 — Final-Test Evidence Freeze

Packaged and froze final-test evaluation evidence.

### M9E Master Freeze

Created the final experiment-level freeze record for M9E.

M9E must not be silently regenerated, retuned, or rewritten under the
same experiment identity.

---

# M10 — Frozen Final-Test Analysis and Evidence Freeze

M10 performed descriptive and exploratory analysis of the already-frozen
M9E final-test results.

It did not change model weights, predictions, prompts, threshold,
compositing policy, or final masks.

## M10A — Analysis Protocol

Froze the rules for post-final-test analysis.

## M10B — Descriptive Analysis

Summarized frozen final-test performance and action groups.

## M10C — Refinement Case Analysis

Analyzed case-level refinement outcomes.

## M10D — Exploratory Relationship Analysis

Explored associations among frozen result variables.

## M10E — Representative Case Selection

Selected representative cases from the frozen results for presentation.

## M10F — Thesis Tables and Figures

Generated research-reporting tables, figures, and captions.

## M10G — Final Analysis Evidence Freeze

Packaged and froze M10 analysis evidence.

Canonical archive family:

`archives/M10/`

---

# M11 — Application and Interactive Research UI

M11 begins only after the pre-M11 provenance and Git checkpoint is
closed.

M11 is an application and visualization layer built on top of the
frozen research core.

Planned major capabilities include:

- user-uploaded multimodal MRI inference
- FastAPI inference/backend layer
- Gradio research/debug console
- polished React/Vite frontend
- synchronized 2D medical-image viewing
- uncertainty visualization
- hotspot and prompt visualization
- baseline-vs-final mask comparison
- read-only frozen M9E final-test explorer
- optional GT evaluation for research uploads

Frozen M9E results and new live inference must remain explicitly
separated.

Recommended labels:

`FROZEN RESEARCH RESULT | M9E Final Test · Read Only`

and

`LIVE INFERENCE | User-uploaded MRI · New Result`

---

# Historical Naming Policy

1. Existing frozen paths and filenames are never renamed merely for
   cosmetic consistency.
2. Human-facing documentation may use clean top-level names such as M01,
   M02, ..., M10, M11.
3. Historical submilestone identifiers remain authoritative.
4. Planned labels are distinguished from actually executed labels.
5. Later provenance recovery does not rewrite the contemporaneous
   scientific record.
