# NeuroPrompt-3D

**NeuroPrompt-3D: Uncertainty-Aware Automatic Prompting for Robust 3D Brain MRI Tumor Segmentation**

NeuroPrompt-3D is a research and educational prototype for exploring whether uncertainty-guided automatic prompts can improve 3D brain tumor segmentation.

It is **not a medical diagnostic system** and is not intended for clinical use.

## Project idea

The main research pipeline is:

```text
Brain MRI
    ↓
Preprocessing
    ↓
Lightweight 3D U-Net
    ↓
Coarse tumor segmentation
    ↓
MC Dropout uncertainty estimation
    ↓
Uncertainty map
    ↓
Automatic point / optional box prompts
    ↓
Frozen SAM-Med3D
    ↓
Refined 3D tumor mask
    ↓
Postprocessing
    ↓
Dice / IoU / HD95 comparison
```

The central research question is:

> Can uncertainty-guided automatic prompts improve the refinement of coarse 3D brain tumor segmentations?

The first segmentation model will predict **binary whole tumor versus background**.

The original multiclass BraTS segmentation labels are preserved so that multiclass experiments remain possible later.

---

## Current project status

Completed so far:

- deterministic synthetic 3D MRI generation;
- binary synthetic tumor masks;
- NIfTI save and load support;
- correct NIfTI-to-PyTorch axis conversion;
- axial, coronal, and sagittal visualization;
- four-modality synthetic MRI generation;
- shared T1 / T1ce / T2 / FLAIR modality contract;
- strict real-case modality validation;
- separate multimodal NIfTI loading;
- real BraTS 2024 case validation;
- MRI and segmentation geometry checks;
- real tumor visualization;
- per-modality foreground z-score normalization;
- automated preprocessing tests;
- BraTS archive case discovery and required-file validation;
- deterministic subject-level splitting that keeps longitudinal cases together;
- a reproducible BraTS cohort split manifest.
- raw BraTS segmentation NIfTI loading with preserved multiclass labels;
- binary whole-tumor target derivation for the first segmentation model;
- MRI-to-segmentation affine validation;
- model-ready MRI and target preparation;
- frozen split-manifest case loading without downstream reshuffling;
- a PyTorch `BraTSDataset` backed by the frozen cohort assignments;
- synchronized 3D MRI and target cropping;
- tumor-centered and brain-background-centered patch sampling;
- reproducible seeded random training-patch selection.
Current automated test suite:

```text
190 passed
```

---

## MRI tensor convention

NeuroPrompt-3D uses the following in-memory MRI layout:

```text
[C, D, H, W]
```

where:

- `C` = MRI modality/channel;
- `D` = depth;
- `H` = height;
- `W` = width.

The shared modality order is:

```text
T1
T1ce
T2
FLAIR
```

Therefore:

```text
channel 0 = T1
channel 1 = T1ce
channel 2 = T2
channel 3 = FLAIR
```

For example:

```text
[4, 128, 128, 128]
```

means:

- 4 MRI modalities;
- 128 depth slices;
- 128 pixels in height;
- 128 pixels in width.

At one spatial location `(d, h, w)`:

```python
mri[:, d, h, w]
```

contains four MRI intensity measurements from the same voxel location.

---

# Development milestones

## Milestone 1: synthetic 3D MRI pipeline

The first milestone established the basic 3D medical-imaging workflow before using real patient data.

The project can:

- generate deterministic synthetic 3D MRI volumes;
- generate a binary tumor segmentation mask;
- save MRI and segmentation volumes as NIfTI files;
- load saved NIfTI files back into PyTorch;
- preserve spatial values through save/load operations;
- validate MRI and mask shapes;
- display axial, coronal, and sagittal views;
- overlay the tumor mask on the synthetic MRI;
- verify the behavior with automated tests.

This synthetic stage was used to understand the data pipeline safely before introducing real medical images.

---

## Milestone 2: four-modality synthetic MRI

The synthetic pipeline was extended to imitate a BraTS-style multimodal MRI case.

The four modalities are:

- T1;
- T1ce;
- T2;
- FLAIR.

All four synthetic modalities share the same spatial geometry and tumor mask while having different simulated contrast patterns.

Tensor conventions:

- MRI: `[channel, depth, height, width]`, `float32`;
- channel order: `T1`, `T1ce`, `T2`, `FLAIR`;
- mask: `[depth, height, width]`, `uint8`;
- mask labels: `0 = background`, `1 = tumor`.

In memory, MRI uses:

```text
[C, D, H, W]
```

A synthetic four-modality NIfTI stores the corresponding data using NIfTI spatial ordering:

```text
[X, Y, Z, C]
```

The loader converts it back into the shared NeuroPrompt-3D tensor convention.

---

## Milestone 3 Step 1: strict multimodal case definition

`src/data/cases.py` defines how a real MRI case supplies its modality files.

The function:

```python
ordered_modality_paths(...)
```

requires all four modalities:

```text
T1
T1ce
T2
FLAIR
```

It:

- rejects missing modalities;
- rejects `None` paths;
- rejects blank paths;
- converts paths into `pathlib.Path` objects;
- always returns paths in the shared modality order.

This prevents accidental channel-order changes between cases.

---

## Milestone 3 Step 2: separate NIfTI modality loading

`load_multimodal_case` loads four separate 3D NIfTI modality files into one PyTorch tensor.

Example:

```python
from src.data.nifti import load_multimodal_case

mri = load_multimodal_case({
    "T1": "data/example/t1.nii.gz",
    "T1ce": "data/example/t1ce.nii.gz",
    "T2": "data/example/t2.nii.gz",
    "FLAIR": "data/example/flair.nii.gz",
})
```

Each modality must be exactly 3D.

The loader validates:

- number of spatial dimensions;
- matching spatial shapes;
- matching affine geometry.

Affines must match T1 within:

```text
atol = 1e-5
rtol = 0
```

NIfTI stores each separate modality as:

```text
[X, Y, Z]
```

NeuroPrompt-3D converts this into:

```text
[D, H, W] = [Z, Y, X]
```

and stacks the four modalities into:

```text
[4, D, H, W]
```

The result is a contiguous CPU:

```text
torch.float32
```

tensor.

For example, four files with NIfTI shape:

```text
(6, 5, 4)
```

produce:

```text
(4, 4, 5, 6)
```

The loader does not currently reorient or resample images.

It also does not normalize intensities during loading. Preprocessing is intentionally kept as a separate stage.

Run the focused multimodal loader tests with:

```bash
python -m pytest -q tests/test_multimodal_nifti.py
```

---

## Milestone 3 Step 3: real BraTS case validation

The multimodal loader has been validated on a real case from the:

**BraTS 2024 Adult Glioma Post-Treatment training dataset.**

Real medical-image data is intentionally stored outside the Git repository under:

```text
/data/Datasets
```

Patient MRI data is not redistributed with NeuroPrompt-3D.

Dataset checkpoint:

- archive: `BraTS2024-BraTS-GLI-TrainingData.zip`;
- Synapse entity: `syn60086071`;
- verified release: version 2;
- verified MD5:

```text
1d910b17d6cd32e38aa6296b8dfb7c77
```

First inspected real case:

```text
BraTS-GLI-03011-101
```

BraTS modality filenames map into the NeuroPrompt-3D contract as:

```text
t1n → T1
t1c → T1ce
t2w → T2
t2f → FLAIR
seg → segmentation
```

For the inspected case, all four MRI modalities and the segmentation have:

```text
NIfTI shape:   (182, 218, 182)
voxel spacing: (1.0, 1.0, 1.0) mm
orientation:   L, A, S
```

MRI and segmentation affine geometry also match.

The existing multimodal loader successfully produces:

```text
torch.Size([4, 182, 218, 182])
```

with:

```text
dtype:      torch.float32
contiguous: True
```

The inspected segmentation contains the label values:

```text
0, 1, 2, 3
```

The original multiclass segmentation is preserved.

For the first binary segmentation model, a whole-tumor target can later be derived conceptually as:

```python
whole_tumor = segmentation > 0
```

without destroying the original multiclass labels.

Tumor-rich slices were also identified automatically for visualization:

```text
Axial Z:    107
Coronal Y:  117
Sagittal X: 50
```

Real T1ce tumor-overlay and four-modality comparison figures were generated outside the repository for visual validation.

---

## Milestone 4: foreground MRI normalization

NeuroPrompt-3D now supports foreground z-score normalization for real multimodal MRI tensors.

MRI intensities from different sequences can have very different numerical scales.

For the first inspected BraTS case, the raw foreground statistics differed substantially between modalities.

Approximate values included:

```text
T1
mean ≈ 1847.72
std  ≈ 546.24

T1ce
mean ≈ 2456.94
std  ≈ 813.53

T2
mean ≈ 1457.99
std  ≈ 589.41

FLAIR
mean ≈ 800.54
std  ≈ 265.74
```

These different raw scales should not cause the neural network to interpret one modality as inherently more important simply because its numerical values are larger.

### Foreground z-score normalization

Each modality is normalized independently.

For every 3D MRI volume:

1. zero-valued background voxels are excluded from the statistics;
2. the foreground mean is calculated;
3. the foreground population standard deviation is calculated;
4. foreground intensities are converted into z-scores;
5. outside-brain background remains exactly zero.

The normalization equation is:

```text
z = (x - mean) / standard deviation
```

For a multimodal tensor:

```text
[4, D, H, W]
```

the channels are normalized separately:

```text
T1     → own mean and standard deviation
T1ce   → own mean and standard deviation
T2     → own mean and standard deviation
FLAIR  → own mean and standard deviation
```

One global mean and standard deviation are **not** shared across all modalities.

The preprocessing functions validate:

- single-modality inputs are 3D;
- single-modality inputs use a floating-point dtype;
- multimodal inputs follow `[4, D, H, W]`;
- all-zero volumes are handled safely;
- zero-variance foreground is handled safely;
- NaN values are avoided for these edge cases;
- input tensors are not modified in-place.

The normalization pipeline was tested on:

```text
BraTS-GLI-03011-101
```

For all four modalities after normalization:

```text
foreground mean ≈ 0
foreground std  ≈ 1
background      = 0
```

The output retained:

```text
shape:      [4, 182, 218, 182]
dtype:      torch.float32
contiguous: True
```

and all output values were finite.

Run the preprocessing tests with:

```bash
python -m pytest -q tests/test_preprocessing.py
```

---

## Milestone 5: leakage-safe BraTS cohort splitting

`src/data/brats.py` supports case discovery and required-file validation for the BraTS 2024 Adult Glioma Post-Treatment archive, grouping longitudinal cases by subject, and reproducible train / validation / test splitting.

A case ID such as `BraTS-GLI-03011-101` identifies subject `BraTS-GLI-03011` and timepoint `101`. All timepoints belonging to one subject stay in the same split. This prevents the same subject's scans from appearing in both training and evaluation sets.

Subjects are sorted before shuffling with a local random generator using seed `42`, so the assignments do not depend on the input order. The target subject fractions are 80% training, 10% validation, and 10% test. Training and validation counts are rounded down, and the remaining subjects enter the test split. Case proportions can differ because subjects have different numbers of timepoints.

The saved manifest is:

```text
splits/brats2024_posttreatment_seed42.json
```

It records dataset provenance (Synapse ID, release version, and archive MD5), split settings, subject and case IDs, and counts.

| Split | Subjects | Cases |
| --- | ---: | ---: |
| Train | 490 | 1078 |
| Validation | 61 | 143 |
| Test | 62 | 129 |
| Total | 613 | 1350 |

The saved manifest was independently checked for zero subject overlap and zero case overlap between all split pairs, with full coverage of the 613 subjects and 1350 cases. Only IDs and metadata are stored in the manifest; MRI files remain outside the repository.

Run the focused BraTS tests with:

```bash
python -m pytest -q tests/test_brats.py
```

Milestone 5 adds 38 tests to the previous 46, bringing the full suite to **84 tests**.

---

## Milestone 6: model-ready BraTS preprocessing and 3D patch sampling

Milestone 6 extends the BraTS data pipeline from frozen cohort assignments to model-ready PyTorch training samples.

Raw BraTS segmentation NIfTI files are loaded as contiguous `torch.uint8` tensors in the project's internal `[D, H, W]` spatial convention. The original multiclass labels are preserved, while a separate binary whole-tumor target is derived using:

```text
0       -> background
1 / 2 / 3 -> whole tumor
```

The MRI remains in the fixed multimodal layout:

```text
[4, D, H, W]
```

and each MRI modality is independently normalized using nonzero-foreground z-score normalization.

`prepare_model_case()` combines the normalized MRI with the derived whole-tumor target while enforcing matching spatial dimensions.

Segmentation geometry is also validated against the T1 reference affine using:

```text
atol = 1e-5
rtol = 0
```

so meaningful physical-space mismatches are rejected while tiny floating-point rounding differences remain acceptable.

A real BraTS case was successfully processed end to end:

```text
BraTS-GLI-03011-101

Raw MRI:          [4, 182, 218, 182]
Raw segmentation: [182, 218, 182]

Prepared MRI:     [4, 182, 218, 182]
Binary target:    [182, 218, 182]
```

The raw segmentation labels were:

```text
[0, 1, 2, 3]
```

and the derived whole-tumor target contained:

```text
[0, 1]
```

with 135353 whole-tumor voxels.

### Frozen-manifest dataset integration

Downstream datasets do not recompute or reshuffle the train / validation / test assignments.

`load_split_case_ids()` reads the case IDs directly from:

```text
splits/brats2024_posttreatment_seed42.json
```

and preserves their stored order.

`BraTSDataset.from_manifest()` therefore creates datasets using exactly the frozen Milestone 5 assignments:

| Split      | Cases |
| ---------- | ----: |
| Train      |  1078 |
| Validation |   143 |
| Test       |   129 |

`dataset[index]` loads the requested case, validates geometry, normalizes the four MRI modalities, derives the binary target, and returns the model-ready pair.

### 3D training-patch preparation

Full BraTS MRI volumes are too memory-intensive for the initial lightweight 3D U-Net training workflow, so Milestone 6 establishes synchronized 3D patch extraction.

MRI and target always use identical spatial crop coordinates.

The implemented crop utilities support:

- deterministic center cropping;
- cropping around a selected spatial center;
- boundary-safe crops that shift inward near volume edges;
- tumor-centered patches;
- non-tumor patches restricted to nonzero MRI foreground;
- explicit rejection of impossible oversized crops.

The training sampler can choose between positive and background patches using a configurable probability:

```text
positive_probability = 0.5
```

which represents the initial intended 1:1 positive/background sampling strategy.

Positive patches choose a random tumor voxel.

Background patches choose a random non-tumor voxel inside MRI foreground rather than empty zero-valued space outside the brain.

A `torch.Generator` can be supplied so both the branch decision and voxel-center selection are reproducible from a fixed seed.

Random patch sampling is intended for the training split only. Validation and test evaluation will later use deterministic full-volume or sliding-window inference rather than random evaluation patches.

Milestone 6 adds 32 tests to the previous 84, bringing the complete automated suite to:

```text
116 passed
```

## Data safety

Real BraTS MRI data is never stored inside this Git repository.

The project `.gitignore` blocks common medical-image and model artifact formats, including:

```text
*.nii
*.nii.gz
*.mha
*.mhd
*.nrrd
*.pt
*.pth
*.ckpt
BraTS*.zip
```

Real datasets remain under:

```text
/data/Datasets
```

while project source code remains under:

```text
/data/Projects/Project_03
```

The project does not redistribute BraTS patient MRI files.

---

## Run the project

From the repository root, activate the NeuroPrompt-3D Conda environment:

```bash
conda activate /data/Conda/envs/neuroprompt3d
```

Run the complete automated test suite:

```bash
python -m pytest -q
```

Current checkpoint:

```text
190 passed
```

Generate the synthetic demonstration case:

```bash
python -m scripts.generate_synthetic_case
```

The generated synthetic files are stored in ignored development directories such as:

```text
data/synthetic/
outputs/
```

---

## Planned next stages

The current development roadmap includes:

1. lightweight 3D U-Net baseline;
2. binary whole-tumor training and validation;
3. full-volume / sliding-window inference;
4. MC Dropout stochastic inference;
5. voxel-wise uncertainty estimation;
6. uncertainty-guided automatic point prompts;
7. optional uncertainty-guided box prompts;
8. frozen SAM-Med3D refinement;
9. Dice, IoU, and HD95 evaluation;
10. coarse-versus-refined segmentation comparison;
11. external/generalization evaluation;
12. research-focused user interface.

---

## Planned interface

The final application is intended to behave more like a medical-imaging research workstation than a generic machine-learning dashboard.

Planned functionality includes:

- demo cases;
- user-supplied T1 / T1ce / T2 / FLAIR NIfTI files;
- optional ground-truth segmentation;
- axial, coronal, and sagittal viewing;
- modality switching;
- coarse segmentation display;
- uncertainty-map display;
- automatic prompt visualization;
- refined segmentation display;
- Dice, IoU, and HD95 comparison when ground truth is available.

MRI prediction and uncertainty visualization may still be performed without ground-truth segmentation, but evaluation metrics requiring ground truth cannot be calculated.

---

## Research disclaimer

NeuroPrompt-3D is an educational and research prototype.

It is not validated for clinical decision-making, diagnosis, treatment planning, or patient care.
