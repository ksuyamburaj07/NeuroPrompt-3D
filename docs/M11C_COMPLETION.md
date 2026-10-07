# M11C — Gradio Research Console Completion Evidence

## Status

M11C is complete.

The NeuroPrompt-3D Gradio Research Console has been implemented, tested,
integrated with the real FastAPI backend, and verified with a full
browser-triggered live inference run.

The console is a technical/research interface only.

It is not the polished React/Vite product interface and is not intended for
clinical diagnosis or patient-care decisions.

Permanent live-result label:

`LIVE INFERENCE | User-uploaded MRI · New Result`

---

## 1. M11C implementation checkpoints

M11C was developed on branch:

`m11-app`

Relevant commits:

- `76ad259` — Add Gradio research console dependency
- `fba753c` — Add Gradio research console foundation
- `c2cd059` — Add existing live run inspector
- `acfac1c` — Add safe research artifact downloads

Backend foundation immediately preceding M11C:

- `22d3c45` — Guard live case deletion during active runs
- `d7254f6` — Add safe live result artifact API
- `b9a4fb3` — Document first real M11 live inference

Frozen research-core source remains:

`66edbba5a07144b3b6a2f117d9fec22ea52284ef`

Pre-M11 provenance closure:

`e5ebbd96b67781e2e509f415d0d70aa775775066`

---

## 2. Architecture boundary

The Gradio Research Console communicates with NeuroPrompt-3D through the
public FastAPI HTTP contract.

Architecture:

Gradio
→ `ResearchApiClient`
→ FastAPI `/api/v1`
→ M11 backend services
→ frozen NeuroPrompt-3D research core

The research UI does not import:

- `src.*`
- backend services
- backend workers

Scientific logic is not duplicated in Gradio.

The import-boundary audit was blank.

---

## 3. Gradio runtime

Verified M11 environment:

- Python 3.11
- Gradio 6.29.1
- gradio-client 2.7.2
- FastAPI 0.142.2
- Starlette 1.7.0
- Uvicorn 0.54.0
- HTTPX 0.28.1
- HTTPX2 2.13.1
- Pydantic 2.13.5
- NumPy 2.4.6

`pip check` reported no broken requirements.

Gradio runs locally with public sharing disabled.

Default local address:

`http://127.0.0.1:7860`

FastAPI default local address:

`http://127.0.0.1:8000`

---

## 4. Research Console capabilities

The completed M11C console supports:

- backend/provenance health inspection;
- four required MRI uploads:
  - T1n
  - T1c
  - T2w
  - T2f / FLAIR
- optional ground-truth upload;
- live case validation;
- geometry inspection;
- user-safe validation errors;
- live run creation;
- isolated inference-worker execution;
- run progress/status polling;
- frozen policy-state display;
- uncertainty hotspot display;
- MC case-seed display;
- automatic prompt-coordinate display;
- SAM usage / abstention display;
- existing live-run inspection;
- `result.json` inspection;
- approved live-artifact discovery;
- safe artifact download links through FastAPI;
- staged-case deletion.

The console does not expose raw server filesystem paths.

---

## 5. Existing-run integration proof

Original M11 Gate-3 run:

`run_16a49dbf55664daca7c26fc0cb722124`

Original staged case:

`case_c4390f9a11d14957a9b4221d95da6eb8`

The Existing Run Inspector successfully loaded this completed run through the
public FastAPI contract without rerunning inference.

Observed state:

- status: `complete`
- stage: `complete`
- action: `APPLY_LOCAL_FP`
- gate: `FP_ELIGIBLE`
- hotspot ZYX: `[71, 98, 50]`
- hotspot variance: `0.21728354692459106`
- frozen variance threshold: `0.21133705228567123`
- MC case seed: `1285324160`
- FP prompt model XYZ: `[79, 62, 33]`
- SAM used: `true`

The console also exposed all 11 approved live artifacts.

---

## 6. Safe artifact-download proof

Artifact downloads are generated from logical artifact names and use:

`GET /api/v1/runs/{run_id}/artifacts/{artifact_name}`

The browser never receives application runtime filesystem paths.

Manual browser download testing successfully retrieved live result artifacts
through FastAPI.

Downloaded Gate-3 masks matched the original artifact hashes:

### baseline_mask.nii.gz

SHA-256:

`66d25ddbc59f5787afe9f4375984b68148ff8eac58f43e40871482f0d4b69917`

### final_mask.nii.gz

SHA-256:

`75590201cea7bfa6d09e2ea61002c591e20a26b5f82fbbb3f73280b1b02e20cc`

This confirmed end-to-end delivery of the original persisted artifact bytes
through the public application boundary.

---

## 7. Full Gradio-triggered inference proof

A second upload of the same canonical Gate-3 MRI was performed through the
Gradio Research Console.

New application case:

`case_6f11e5c98a5f436bb33c958f803933e8`

New application run:

`run_7fd922c98cda4fd7b359d61b5c304b5b`

Validation succeeded with:

- shape XYZ: `[182, 218, 182]`
- tensor DHW: `[182, 218, 182]`
- voxel spacing XYZ: `[1.0, 1.0, 1.0]`
- optional GT: not provided

The new application UUIDs differed from the original run, as expected.

The scientific live-inference identity reproduced exactly:

`live_sha256_1dceb308640032e7043e90884eb67d86f82325b7c3c7ba013a889d164dc816ff`

MC case seed reproduced exactly:

`1285324160`

Final automatic-policy state reproduced exactly:

- action: `APPLY_LOCAL_FP`
- gate: `FP_ELIGIBLE`
- frozen variance threshold: `0.21133705228567123`
- hotspot ZYX: `[71, 98, 50]`
- hotspot variance: `0.21728354692459106`
- FP prompt ZYX: `[71, 98, 50]`
- FP prompt model XYZ: `[79, 62, 33]`
- SAM used: `true`
- SAM refinement skipped: `false`
- semantic abstention condition: `null`

All 10 MC-pass hashes reproduced exactly.

---

## 8. End-to-end scientific reproducibility comparison

Original run:

`run_16a49dbf55664daca7c26fc0cb722124`

Gradio-triggered run:

`run_7fd922c98cda4fd7b359d61b5c304b5b`

### NPY arrays

Exact equality was verified for:

- `baseline_probability.npy`
- `baseline_mask.npy`
- `final_mask.npy`
- `removal_mask.npy`
- `support_mask.npy`

All had:

- identical shape;
- `numpy.array_equal == True`;
- maximum absolute difference `0.0`.

### NIfTI outputs

Exact voxel equality and exact affine equality were verified for:

- `baseline_probability.nii.gz`
- `baseline_mask.nii.gz`
- `final_mask.nii.gz`
- `removal_mask.nii.gz`
- `support_mask.nii.gz`

### result.json scientific fields

Exact equality was verified for:

- `action`
- `baseline`
- `case_seed`
- `execution_device`
- `fp_prompt_model_xyz`
- `fp_prompt_zyx`
- `frozen_variance_threshold`
- `gate_state`
- `hotspot_variance`
- `hotspot_zyx`
- `inference_case_id`
- `mc_pass_hashes`
- `predictive_variance_volume_saved`
- `sam`
- `sam_used`
- `semantic_abstention_condition`

Expected differences were limited to application-level identifiers:

- `case_id`
- `run_id`

Final result:

`M11C END-TO-END REPRODUCIBILITY: PASS`

---

## 9. Test state at M11C completion

After M11C5:

- Research UI tests: `15 passed`
- M11 tests: `44 passed`
- Full repository regression: `398 passed`

One pre-existing PyTorch FutureWarning remained:

`torch.jit.interface` is deprecated.

There were no test failures.

The previous Starlette TestClient HTTPX deprecation warning disappeared after
installation of the Gradio dependency set containing HTTPX2.

---

## 10. Scientific integrity boundary

M11C did not modify:

- frozen research algorithms under `src/`;
- `archives/M9E/`;
- `archives/M10/`;
- frozen checkpoints;
- frozen research outputs;
- notebooks.

The Gradio console is an application client of the frozen scientific pipeline,
not an alternative implementation of it.

No post-test scientific tuning was performed.

---

## 11. M11C conclusion

M11C has demonstrated that NeuroPrompt-3D can now be operated through a real
interactive research interface while preserving the frozen scientific
behavior.

A fresh browser upload of the same MRI produced new application UUIDs while
reproducing the same:

- deterministic live inference identity;
- MC-Dropout seed;
- MC-pass hashes;
- baseline probability volume;
- baseline mask;
- uncertainty hotspot;
- automatic policy decision;
- automatic prompt;
- SAM refinement behavior;
- removal/support masks;
- final segmentation;
- model/checkpoint provenance.

M11C is therefore complete.

The next application milestone is M11D:

React/Vite frontend foundation.
