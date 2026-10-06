# NeuroPrompt-3D M11A — Application and Product Architecture

## Status

M11 begins after the frozen pre-M11 provenance checkpoint:

- Frozen research-core source boundary:
  `66edbba5a07144b3b6a2f117d9fec22ea52284ef`
- Pre-M11 provenance closure:
  `e5ebbd96b67781e2e509f415d0d70aa775775066`
- M11 development branch:
  `m11-app`

M11 is an application and visualization phase.

It must not alter the scientific identity of M9E or M10.

---

# 1. Product Goal

NeuroPrompt-3D should be a polished research application that makes the
entire segmentation process understandable.

The user should be able to inspect:

MRI input
→ baseline segmentation
→ predictive uncertainty
→ uncertainty hotspot
→ automatic action decision
→ optional SAM-Med3D refinement
→ final segmentation.

The application is a research prototype and is not intended for clinical
diagnosis or patient-care decisions.

---

# 2. Application Modes

## 2.1 New Case

Primary application workflow.

Required inputs:

- T1n
- T1c
- T2w
- T2f

Accepted medical-image format:

- `.nii`
- `.nii.gz`

## 2.1.1 Application-to-Research-Core Modality Mapping

The M11 UI/API uses current BraTS-style modality labels while the frozen
research core preserves its historical channel names.

The explicit application boundary is:

| UI/API | Frozen research core |
| --- | --- |
| `t1n` | `T1` |
| `t1c` | `T1ce` |
| `t2w` | `T2` |
| `t2f` | `FLAIR` |

This mapping is an application adapter only.

The research-core channel names and order are not renamed or modified.

Optional input:

- ground-truth segmentation for research evaluation

Ground truth is never required for inference.

Permanent UI label:

`LIVE INFERENCE | User-uploaded MRI · New Result`

---

## 2.2 Frozen Final-Test Explorer

Read-only explorer for the 129-case M9E final test.

This mode uses existing frozen outputs only.

It must not:

- rerun M9E inference
- change the frozen threshold
- change prompts
- rerun SAM-Med3D under the M9E identity
- rewrite final masks
- overwrite M9E evidence

Permanent UI label:

`FROZEN RESEARCH RESULT | M9E Final Test · Read Only`

---

## 2.3 Research Explorer

Technical/debug mode for detailed scientific inspection.

This mode may expose:

- predictive-variance statistics
- policy variables
- prompt coordinates
- model/device information
- timing information
- development-case diagnostics
- intermediate pipeline outputs

The Research Explorer may initially be implemented using Gradio.

---

# 3. New-Case Workflow

The intended workflow is:

1. Upload T1n, T1c, T2w and T2f.
2. Optionally upload GT.
3. Validate file readability.
4. Validate modality completeness.
5. Validate shape and geometry compatibility.
6. Stage the case in temporary application runtime storage.
7. Run canonical NeuroPrompt-3D preprocessing.
8. Run frozen baseline 3D U-Net inference.
9. Run frozen MC-Dropout uncertainty estimation.
10. Select the uncertainty hotspot.
11. Apply the frozen automatic action policy.
12. If the decision is ABSTAIN_BASELINE, preserve the baseline result.
13. If the decision is APPLY_LOCAL_FP, invoke the frozen SAM-Med3D branch.
14. Produce the final segmentation.
15. If GT is supplied, calculate research evaluation metrics.
16. Expose results to the interactive viewer.

Scientific algorithms must be imported from the existing `src/` research
core. They must not be copied into the application layer.

---

# 4. Validation UX

The application must validate the case before inference.

The UI should show individual validation states for:

- T1n loaded
- T1c loaded
- T2w loaded
- T2f loaded
- NIfTI readability
- shape compatibility
- affine/geometry compatibility
- optional GT compatibility

Validation errors must be understandable to the user.

Raw Python tracebacks must not be shown in the normal UI.

The exact scientific validation semantics should reuse the existing
`src.data` implementation rather than introducing a competing validation
policy.

---

# 5. Inference Progress Model

A live inference run should expose a pipeline timeline.

Canonical application stages:

- queued
- validating
- preprocessing
- baseline
- mc_dropout
- hotspot
- policy
- sam_refinement
- finalizing
- complete
- failed

If the automatic policy abstains, `sam_refinement` is explicitly marked as
skipped rather than silently omitted.

---

# 6. Results Workspace

The result workspace is the primary visualization surface.

It should support:

- axial view
- coronal view
- sagittal view
- synchronized slice/crosshair position
- T1n/T1c/T2w/T2f modality switching
- segmentation opacity control
- uncertainty opacity control

Primary overlays:

- baseline mask
- final mask
- predictive uncertainty
- uncertainty hotspot
- automatic prompt

The user should be able to compare baseline and final masks directly.

The interface should make the causal sequence visible:

uncertainty
→ hotspot
→ policy decision
→ prompt/refinement
→ final result.

---

# 7. Policy Summary

For every live result, show an automatic-policy summary containing:

- hotspot variance
- frozen threshold
- comparison result
- action decision
- whether SAM-Med3D was invoked

The application displays frozen scientific policy values.

It does not expose them as ordinary user-tunable controls in New Case mode.

---

# 8. Optional GT Evaluation

When GT is provided for a new uploaded case, the result workspace may show:

- Dice
- IoU
- HD95

GT-derived evaluation must be visibly labeled as research evaluation.

GT remains optional.

---

# 9. API Architecture

The backend uses FastAPI.

Initial API prefix:

`/api/v1`

Planned endpoints:

`GET /api/v1/health`

Application/backend health and provenance metadata.

`POST /api/v1/cases/validate`

Upload and validate a new four-modality MRI case.

Successful validation returns an application-generated `case_id`.

`GET /api/v1/cases/{case_id}`

Return staged live-case metadata and validation state.

`DELETE /api/v1/cases/{case_id}`

Explicitly remove a staged live case.

`POST /api/v1/cases/{case_id}/runs`

Start a NeuroPrompt-3D inference run.

Long-running inference should use a run/job abstraction rather than holding
a browser request open for the entire scientific pipeline.

`GET /api/v1/runs/{run_id}`

Return inference state, progress stage, policy decision and result metadata.

`GET /api/v1/runs/{run_id}/artifacts/{artifact_name}`

Serve approved live result artifacts.

`GET /api/v1/frozen/cases`

List the frozen M9E final-test cases.

`GET /api/v1/frozen/cases/{case_id}`

Return frozen final-test metadata.

`GET /api/v1/frozen/cases/{case_id}/artifacts/{artifact_name}`

Serve approved read-only M9E evidence for visualization.

There will be no M9E mutation endpoint.

---

# 10. Runtime Storage

Live application uploads and outputs must never be written into:

- `archives/M9E/`
- `archives/M10/`
- `outputs/`
- `checkpoints/`

Temporary live application data belongs under:

`app/backend/runtime/`

Runtime data is ignored by Git.

Suggested structure:

`runtime/live_cases/<case_id>/`

and

`runtime/live_runs/<run_id>/`

User-uploaded MRI is not silently copied into the scientific archive.

---

# 11. Privacy

Default application behavior:

- no automatic cloud upload
- no addition of user cases to the BraTS dataset
- no addition of user cases to M9E/M10
- no silent persistence as scientific evidence
- temporary application storage only

A later release may implement automatic expiry/cleanup of live sessions.

---

# 12. Application Layers

The architecture is:

React/Vite frontend
        |
        | HTTP/API
        v
FastAPI application backend
        |
        | imports
        v
Existing frozen Python research core
        |
        +-- data/preprocessing
        +-- baseline inference
        +-- MC Dropout
        +-- uncertainty hotspot
        +-- automatic prompt/policy
        +-- SAM-Med3D adapter/refinement

A Gradio research console may call the same backend/service layer.

Scientific logic must not be duplicated separately in React, Gradio or
FastAPI route handlers.

---

# 13. Repository Structure

Target M11 structure:

app/
├── backend/
│   ├── api/
│   ├── core/
│   ├── schemas/
│   ├── services/
│   ├── runtime/
│   └── main.py
├── research_ui/
└── frontend/

Application tests live under:

`tests/app/`

---

# 14. UI Character

The application should feel:

- scientific
- modern
- restrained
- imaging-focused
- transparent

It should not resemble either a clinical EMR or a generic flashy AI demo.

UI elements must help the user understand at least one of:

- MRI content
- segmentation
- uncertainty
- model decision
- prompt/refinement
- scientific result

---

# 15. Frozen vs Live Boundary

Frozen M9E content and new live inference must never be visually ambiguous.

Frozen:

`FROZEN RESEARCH RESULT | M9E Final Test · Read Only`

Live:

`LIVE INFERENCE | User-uploaded MRI · New Result`

This distinction must persist throughout the frontend, backend metadata and
result artifacts.

---

# 16. Device Strategy

The application architecture must remain device-aware.

The UI/backend foundation can run on CPU.

Heavy live inference may later use an available CUDA-capable execution
environment.

Application code must not silently change frozen scientific parameters
based on device availability.

---

# 17. M11A Acceptance Criteria

M11A is complete when:

1. product modes are documented;
2. live-vs-frozen behavior is explicit;
3. upload/privacy boundaries are documented;
4. API responsibilities are defined;
5. viewer/result requirements are defined;
6. the application/research-core boundary is explicit;
7. no frozen M9E/M10 artifact has been modified.

After M11A, M11B implements the backend foundation and progressively
connects it to the existing research core.

## Live inference identity and frozen MC-Dropout seeding

The M11 application storage identifier (`case_<uuid>`) is not used as the
scientific case identifier passed to the frozen MC-Dropout implementation.
A random upload UUID would otherwise cause identical MRI re-uploads to receive
different frozen per-case MC seeds.

For live inference, M11 derives a deterministic identifier of the form
`live_sha256_<digest>` from the canonical raw four-modality `[4,D,H,W]`
float32 MRI tensor plus the reference 4x4 affine. The hash contract is
versioned (`NeuroPrompt3D|M11|live-content-id|v1`) and canonicalizes numeric
bytes to little-endian float32 MRI values and little-endian float64 affine
values.

This application-layer identifier is passed unchanged to the frozen
case-specific seed function. The frozen MC master seed, T=10 execution,
seed-derivation algorithm, and uncertainty computation are not modified.
Identical canonical MRI data and geometry therefore reproduce the same
live-inference MC seed even when uploaded under different application case
UUIDs.
