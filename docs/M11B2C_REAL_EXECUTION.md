# M11B2C Real Live-Inference Execution Evidence

## Status

M11B2C real execution gate: **PASS**

This gate demonstrated the M11 application backend executing the frozen
NeuroPrompt-3D final automatic pipeline end-to-end using a real four-modality
BraTS MRI case.

No ground-truth segmentation was supplied to the live inference path.

## Source case

- Dataset case: `BraTS-GLI-03011-101`
- Frozen split membership: `train`
- Input modalities: `t1n`, `t1c`, `t2w`, `t2f`
- Input geometry: `182 x 218 x 182`
- Voxel spacing: `1.0 x 1.0 x 1.0 mm`
- Ground truth supplied to M11 inference: **No**

The training-split case was chosen deliberately so this live execution does
not create a new result from an M9E frozen final-test case.

## Application identities

- Live application case ID:
  `case_c4390f9a11d14957a9b4221d95da6eb8`
- Live run ID:
  `run_16a49dbf55664daca7c26fc0cb722124`
- Deterministic scientific inference ID:
  `live_sha256_1dceb308640032e7043e90884eb67d86f82325b7c3c7ba013a889d164dc816ff`
- Frozen MC case seed:
  `1285324160`

The random application case UUID was not used for frozen MC-Dropout seeding.
The deterministic live inference identity was derived from canonical MRI
content and geometry.

## Execution environment

- Execution device: CPU
- PyTorch: `2.14.0+cpu`
- torchvision: `0.29.0+cpu`
- Worker execution: isolated application subprocess
- Approximate execution duration: 3 minutes 22 seconds

## Frozen policy result

- Action: `APPLY_LOCAL_FP`
- Gate state: `FP_ELIGIBLE`
- Frozen variance threshold: `0.21133705228567123`
- Hotspot ZYX: `[71, 98, 50]`
- Hotspot variance: `0.21728354692459106`
- FP prompt ZYX: `[71, 98, 50]`
- FP prompt model XYZ: `[79, 62, 33]`
- SAM used: `true`
- Semantic abstention condition: none

The hotspot variance was strictly greater than the frozen threshold, therefore
the frozen policy correctly entered the localized FP-removal branch.

## Segmentation result

- Baseline foreground voxels: `86649`
- Final foreground voxels: `85975`
- Removed voxels: `674`
- Frozen support voxels: `12841`

Verification confirmed:

- no foreground was added;
- final mask equals baseline minus the removal mask;
- no removal occurred outside frozen localized support;
- generated NIfTI artifacts retain the original input geometry.

No ground-truth metrics were calculated because ground truth was intentionally
not supplied to the live inference path.

## Artifact SHA-256

| Artifact | SHA-256 |
|---|---|
| baseline_mask.nii.gz | `66d25ddbc59f5787afe9f4375984b68148ff8eac58f43e40871482f0d4b69917` |
| baseline_mask.npy | `7cfa90a11e00cb7bf9289e446586eeed5c82875e26e3fc1f1de06c0737fa9711` |
| baseline_probability.nii.gz | `7d89eded2b36031bac8fed21b65e20de3f4284e9f1e1377dc2aadeada0e01e4e` |
| baseline_probability.npy | `e5a0f7277b05ab1ffb1be578bb50653749ee6fbfd8270a86bb15f34de7108190` |
| final_mask.nii.gz | `75590201cea7bfa6d09e2ea61002c591e20a26b5f82fbbb3f73280b1b02e20cc` |
| final_mask.npy | `c98fe150a34a8ee0bc3a03c9efcc6b51b8c4747a6ae8bc0781c22a7e1a7570de` |
| removal_mask.nii.gz | `bd849a2a8408dd68e563470358e3d6a4ee4d496f696d6e552726c0f883ffc2be` |
| removal_mask.npy | `59e812bc8ea2aaed79bfc23789eb888179d9d1a20f6be8895a140701b2669c05` |
| result.json | `1f39e5b592ac12c161b8a1a8a4474e6bbeddfc1786e66af5819704afbc2c219b` |
| support_mask.nii.gz | `4c11ebe5128befbc51ad415ca02e5475b900d514c02d450d8a6e9d87e90863ae` |
| support_mask.npy | `2f13f7e7fdea12974c642bb9a17f168af0b57c1891d81045648a10828b9f43ed` |

## Uncertainty-volume boundary

The frozen top-level automatic result exposes the selected hotspot and hotspot
variance but does not return the complete predictive-variance volume.

M11 therefore did **not** duplicate MC-Dropout execution or modify the frozen
research implementation merely to create a visualization artifact.

`predictive_variance_volume_saved = false`

## Scientific boundary

This run is:

`LIVE INFERENCE | User-uploaded MRI · New Result`

It is not an M9E final-test result and must not be presented as one.

The M9E and M10 frozen archives remained untouched throughout execution.
