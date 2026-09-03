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
same data as `[X, Y, Z, C]`. A later real-data loader can read separate BraTS
modality files and stack them into this same four-channel memory layout.

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
