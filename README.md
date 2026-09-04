# NeuroPrompt-3D

NeuroPrompt-3D is being built as a testable 3D medical-image segmentation project.

## Milestone 2: four-modality synthetic MRI

The current pipeline can:

- generate deterministic T1, T1ce, T2, and FLAIR-like 3D MRI volumes;
- keep all four modalities aligned with one binary tumor-segmentation mask;
- save the four channels together as a compressed 4D NIfTI file (`.nii.gz`);
- load the files back into PyTorch without changing their values;
- save a four-panel modality comparison with the same tumor overlay;
- verify the pipeline with automated tests.

Tensor conventions:

- MRI: `[channel, depth, height, width]`, `float32`
- channel order: `T1`, `T1ce`, `T2`, `FLAIR`
- mask: `[depth, height, width]`, `uint8`
- mask labels: `0 = background`, `1 = tumor`

For example, `[4, 128, 128, 128]` means:

- `4` MRI modalities, or four measurements at every voxel;
- `128` depth slices;
- `128` pixels in height;
- `128` pixels in width.

At one spatial location `(d, h, w)`, `mri[:, d, h, w]` contains four MRI
measurements while `mask[d, h, w]` contains one tumor/background label.

In memory, the MRI uses `[C, D, H, W]`. The synthetic NIfTI file stores the
same data as `[X, Y, Z, C]`. Separate 3D modality files can now be loaded into
this same four-channel memory layout.

## Milestone 3 Step 2: separate NIfTI modalities

`load_multimodal_case` uses the existing ordered case definition to load one
file per modality. For example, given four files already on disk:

```python
from src.data.nifti import load_multimodal_case

mri = load_multimodal_case({
    "T1": "data/example/t1.nii.gz",
    "T1ce": "data/example/t1ce.nii.gz",
    "T2": "data/example/t2.nii.gz",
    "FLAIR": "data/example/flair.nii.gz",
})
```

Every file must be exactly 3D, including rejecting a singleton fourth axis.
Spatial shapes must match exactly. Each affine must match T1 within an
absolute tolerance of `1e-5` (`rtol=0`). Shape describes the voxel array;
the affine maps its indices to physical coordinates, so both must agree.

The result is a contiguous CPU `torch.float32` tensor in `[4, D, H, W]`
order, with channels `T1`, `T1ce`, `T2`, `FLAIR` regardless of dictionary
insertion order. File axes `[X, Y, Z]` become `[D, H, W] = [Z, Y, X]`:
four `(6, 5, 4)` files produce a `(4, 4, 5, 6)` tensor. These labels describe
array axes; loading does not reorient or resample the images. Intensities
are read as float32 without normalization, and no mask file is required.

Run the focused synthetic-file tests with:

```bash
python -m pytest -q tests/test_multimodal_nifti.py
```

## Milestone 3 Step 3: real BraTS case validation

The multimodal loader has now been validated on one real case from the
BraTS 2024 Adult Glioma Post-Treatment training dataset.

Real medical-image data is intentionally stored outside this Git repository
under `/data/Datasets` and is not redistributed with NeuroPrompt-3D.

Dataset checkpoint:

- archive: `BraTS2024-BraTS-GLI-TrainingData.zip`
- Synapse entity: `syn60086071`
- verified release: version 2
- verified MD5: `1d910b17d6cd32e38aa6296b8dfb7c77`
- first inspected case: `BraTS-GLI-03011-101`

BraTS modality names are mapped into the shared NeuroPrompt-3D contract as:

- `t1n` -> `T1`
- `t1c` -> `T1ce`
- `t2w` -> `T2`
- `t2f` -> `FLAIR`

For the inspected case, all four MRI modalities and the segmentation share:

- NIfTI shape: `(182, 218, 182)`
- voxel spacing: `(1.0, 1.0, 1.0)` mm
- orientation codes: `L`, `A`, `S`
- matching MRI/segmentation affine geometry

The existing `load_multimodal_case` function successfully loads the real
case as a contiguous `torch.float32` tensor with shape
`[4, 182, 218, 182]`.

The original segmentation is preserved. The inspected case contains label
values `0`, `1`, `2`, and `3`. A binary whole-tumor target can later be
derived as all nonzero segmentation voxels while retaining the original
multiclass labels separately.

Raw intensity statistics were also inspected independently for T1, T1ce,
T2, and FLAIR. No normalization has been applied yet. Foreground
per-modality normalization is the next preprocessing milestone.

## Run it

From the project root, activate the existing environment and run the tests:

```bash
conda activate neuroprompt3d
python -m pytest -q
```

Generate the demonstration case:

```bash
python -m scripts.generate_synthetic_case
```

This creates:

```text
data/synthetic/synthetic_001_mri.nii.gz
data/synthetic/synthetic_001_mask.nii.gz
outputs/synthetic_001_preview.png
```
