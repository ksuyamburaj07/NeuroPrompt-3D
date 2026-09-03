# NeuroPrompt-3D

NeuroPrompt-3D is being built as a testable 3D medical-image segmentation project.

## Milestone 1: synthetic data pipeline

The current pipeline can:

- generate a deterministic synthetic 3D MRI volume;
- generate its binary tumor-segmentation mask;
- save both arrays as compressed NIfTI files (`.nii.gz`);
- load the files back into PyTorch without changing their values;
- save axial, coronal, and sagittal mask-overlay previews;
- verify the pipeline with automated tests.

Tensor conventions:

- MRI: `[channel, depth, height, width]`, `float32`
- mask: `[depth, height, width]`, `uint8`
- mask labels: `0 = background`, `1 = tumor`

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
