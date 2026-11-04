# NeuroPrompt-3D

**Uncertainty-Aware Automatic Prompting for 3D Brain MRI Tumor Segmentation**

NeuroPrompt-3D is a research and educational application exploring whether uncertainty-guided automatic prompts can improve coarse 3D brain tumor segmentation.

The project combines a 3D segmentation pipeline, MC Dropout uncertainty estimation, SAM-Med3D integration, frozen scientific evaluation, interactive MRI visualization, and provisional FreeSurfer cortical quality-control tools.

> **Research prototype — not clinically validated.** This software is not intended for diagnosis, treatment planning, patient care, or other clinical decisions.

## Application overview

The locally accepted application has four integrated modes.

| Mode | Purpose |
| --- | --- |
| **New Case** | Validate authorized multimodal MRI inputs, initiate a configured research run, and inspect resulting visualizations. |
| **Final-Test Explorer** | Examine the frozen 129-case evaluation, decisions, performance metrics, and linked MRI visualization tools. |
| **Research Explorer** | Understand the scientific question, experimental methods, verified outcomes, limitations, and reproducibility evidence. |
| **Cortical QC** | Inspect provisional bilateral FreeSurfer pial surfaces, registered MRI slices, cortical intersections, and patient-oriented 3D camera views. |

All four workspaces are integrated into one application.

**Hosted research application (public production URL):**
https://neuroprompt-3d.vercel.app

**Source repository:**
https://github.com/ksuyamburaj07/NeuroPrompt-3D

**Cloud verification — November 4, 2026:**
- Genuine four-modality MRI upload and interactive 3D visualization.
- Completed frozen scientific inference on Modal CPU infrastructure.
- Eleven persisted artifacts independently verified.
- Scientific decision: ABSTAIN_BASELINE (SAM refinement not applied).
- Authenticated artifact download capability.

The production website is publicly accessible. Visitor-level API
authorization and dataset redistribution clearance are not yet complete.
Do not upload sensitive or identifiable patient data. The application
is for research and educational use only.

**Local development application:** `http://127.0.0.1:8090`.

## Scientific question

Can uncertainty-guided automatic prompts improve the refinement of coarse 3D brain tumor segmentations?

The research pipeline investigates:

1. Multimodal MRI preparation and validation.
2. Coarse 3D U-Net tumor segmentation.
3. MC Dropout uncertainty estimation.
4. Automatic prompt selection.
5. SAM-Med3D refinement under a frozen decision policy.
6. Dice, IoU and HD95 evaluation.

The initial segmentation objective is binary whole tumor versus background. Original BraTS multiclass labels were preserved during dataset preparation.

## Frozen final-test results

The accepted evaluation contains **129 test cases**.

| Metric | Baseline mean | Final mean | Mean change |
| --- | ---: | ---: | ---: |
| Dice | 0.766267 | 0.762512 | -0.003755 |
| IoU | 0.642513 | 0.638683 | -0.003830 |
| HD95 (mm) | 24.298143 | 23.662362 | -0.635782 |

The policy applied local false-positive refinement to **44 cases** and retained the baseline for **85 cases**.

**Interpretation:** The tested policy did not improve mean Dice or IoU. Mean HD95 decreased, which is favorable for that metric. Individual cases showed improvements, unchanged outcomes, and deterioration.

These results must not be presented as uniform segmentation improvement or as clinical validation.

The frozen evaluation, prediction archive and prediction manifest are authenticated separately by SHA-256. The underlying research artifacts are not assumed to be redistributable with the source repository.

## Cortical research QC

The Cortical QC workspace displays genuine FreeSurfer-derived bilateral pial surface meshes in physical RAS+ coordinates.

Features include:

- Gold left and cyan right hemisphere visualization.
- Experimental sulcal shading and hemisphere opacity controls.
- Linked axial, coronal and sagittal T1N MRI views.
- Gold/cyan pial-intersection contours.
- MRI planes inside the 3D reconstruction.
- Patient-oriented camera navigation.
- Technical reconstruction provenance.

The displayed cortical anatomy is **provisional**. Known anatomical uncertainties remain, particularly in the post-treatment region. Visual plausibility is not proof of anatomical correctness.

## Technology

- **Backend:** Python, FastAPI, PyTorch and scientific imaging utilities.
- **Frontend:** React, TypeScript, Vite, Three.js and React Three Fiber.
- **Imaging:** NIfTI-based MRI processing, volumetric rendering and FreeSurfer-derived PLY surfaces.
- **Evaluation:** Frozen research predictions and verified metric artifacts.
- **Testing:** Pytest, frontend lint/build checks, camera geometry and research navigation regressions.

## Running the unified local application

The following commands describe the **previously configured research workstation**. A fresh installation additionally requires the correct research environment, authorized external assets and model dependencies.

The validated development environment is named `neuroprompt3d_m11`.

### Build the frontend

```bash
cd app/frontend
npm ci
npm run build
cd ../..
```

The frontend manifest specifies Node.js 24.21.0 and npm 11.19.0.

### Launch

```bash
conda activate neuroprompt3d_m11
./scripts/start_neuroprompt3d_local.sh
```

Then open:

`http://127.0.0.1:8090`

The launcher binds to `127.0.0.1` only.

To enable local development-run discovery explicitly:

```bash
NP3D_ENABLE_LOCAL_RESEARCH_DISCOVERY=1 \
  ./scripts/start_neuroprompt3d_local.sh
```

This development-only discovery control is not public authentication or authorization.

### External research assets

MRI volumes, model checkpoints, experiment outputs and FreeSurfer reconstruction assets are maintained outside the public source tree.

The pinned cortical QC renderer supports these configuration variables:

- `NP3D_QC_RECONSTRUCTION_ROOT` — verified cortical reconstruction root.
- `NP3D_QC_MRI_FILE` — corresponding original T1N NIfTI image.
- `NEUROPROMPT_CORTEX_REGISTRY` — optional cortical asset registry location.

The renderer verifies the expected MRI source fingerprint and original pial meshes. These checks must remain enabled.

`requirements-m11.txt` documents the M11 application dependency layer. It is **not yet a complete, independently verified fresh-machine installation specification** for the full research pipeline.

## Validation

The accepted local application completed:

```bash
python -m pytest tests/m11 -q

cd app/frontend
npm run lint
npm run build
npm run test:camera
```

**Verified result:** 361 M11 backend tests passed, with two nonfatal dependency/deprecation warnings.

Additional regression scripts cover frozen MRI crosshair alignment, case navigation, research-run identity and research discovery parsing.

The integrated application also passed local smoke tests for its frontend, FastAPI routes, MRI metadata, rendered MRI images, cortical contours and approved PLY assets.

These validations establish research-prototype behavior, not clinical performance or deployment security.

## Research data and sharing

The project uses research MRI data, including BraTS 2024 Adult Glioma Post-Treatment material.

Dataset access and use permissions are separate from redistribution permission. Original MRI volumes, dataset-derived visual assets, case-level records and reconstructed anatomy must not be published or hosted without checking the applicable access and sharing conditions.

The source repository should exclude private inputs, dataset archives, runtime cases, trained weights and generated research assets unless their publication is independently authorized.

## Security and deployment status

**Local unified application: accepted.**

**Hosted deployment: operational and publicly reachable; API security and research-data redistribution review remain open.**

The current local application contains research and writable case/run APIs. Its loopback restrictions are not a substitute for public-facing authentication, authorization, resource controls or secure deployment.

Do not expose the local server through an unrestricted public tunnel.

## Development history and provenance

The original, detailed milestone-by-milestone README is preserved in [Development History](docs/DEVELOPMENT_HISTORY.md).

Additional research methodology, scientific boundaries and application documentation are maintained in the `docs/` directory.

## License and intended use

No open-source software license has been selected for this repository. Public visibility of source code does not by itself grant reuse, modification or redistribution rights.

Third-party model, dataset and dependency licenses must be respected independently.

**NeuroPrompt-3D is provided as an experimental research and educational prototype. It must not be used for clinical diagnosis or patient-care decisions.**
