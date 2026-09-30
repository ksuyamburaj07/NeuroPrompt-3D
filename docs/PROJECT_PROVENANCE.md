# NeuroPrompt-3D Project Provenance

## Project Origin

NeuroPrompt-3D originated from an idea and project specification shared
by the project author's friend.

The friend contributed the initial concept/inception and supplied
project-planning materials, including:

- a full NeuroPrompt-3D project specification,
- architecture material,
- and a proposed two-developer implementation plan.

The friend did **not** participate as a co-developer in the subsequent
implementation, experimentation, evaluation, or execution of the
NeuroPrompt-3D research project.

The document titled `NeuroPrompt3D_Two_Developer_Implementation_Plan.md`
is therefore preserved as an original planning artifact and not as a
record of the actual development team.

## Recovered Origin Artifacts

The original project-origin materials were recovered from the project
author's WhatsApp chat history with the friend and preserved in the
repository archive on 2026-09-30.

Canonical local archive hashes:

### Architecture image

`archives/ChatGPT Image Aug 30, 2026, 06_17_00 PM.png`

SHA-256:

`c4060ac6c1e6a90054cfea6cf2d122c849aa5d9cede580a36dfc75cfb2e9e509`

### Full project specification

`archives/NeuroPrompt3D_Full_Project_Specification.md`

SHA-256:

`259dcec77eeddddbd5a810d687c4d95750e705edd18ac199c3444edde582a223`

### Two-developer implementation plan

`archives/NeuroPrompt3D_Two_Developer_Implementation_Plan.md`

SHA-256:

`b974a290443425decead83e7f7c05066b668a7c380360c2a37886b41f9792162`

### WhatsApp architecture image

`archives/WhatsApp Image 2026-09-02 at 10.04.25.jpeg`

SHA-256:

`12c3c4645cf1963724ed5170499bbc0cecc4cbc5250017e66957976fe166c978`

These files preserve the original project conception and planning
material. They must not be retroactively rewritten to match later
research decisions.

---

## Original Budget Goal vs Actual Execution

The original specification targeted a `$0` / free-compute research
workflow.

Actual execution remained low-cost but was not strictly zero-cost.

Approximately **RM45–50** was spent on:

- Google Colab Plus
- Google Drive subscription/storage

NeuroPrompt-3D should therefore be described as a **low-cost,
resource-conscious research project**, not as a literally zero-cost
completed project.

---

## Original Research Direction

The recovered specification proposed a pipeline combining:

1. 3D MRI preprocessing
2. a lightweight 3D U-Net coarse segmenter
3. Monte Carlo dropout uncertainty estimation
4. automatic prompt generation
5. a frozen medical foundation model such as MedSAM2 or SAM-Med3D
6. segmentation refinement
7. evaluation and visualization
8. a research demo/UI

The original document was a starting design rather than a frozen
experimental protocol.

Later experimental work was allowed to reject or modify hypotheses,
prompt types, post-processing ideas, thresholds, and model choices based
on development evidence.

---

## Major Research Evolution

The final implemented research pipeline differs from several early
planning ideas.

Examples include:

- SAM-Med3D was selected as the foundation model.
- The final automatic policy became a gated localized FP-removal
  strategy.
- The final refinement path uses a negative point when the uncertainty
  gate selects refinement.
- The final-test pipeline does not use morphology to conceal errors.
- A strict pre-GT firewall and evidence-freeze process was established.
- The final experiment used a frozen 129-case test cohort.

This evolution is part of the scientific history and should not be
retroactively edited out of the original specification.

---

## Canonical Repository

Canonical project path:

`/data/Projects/Project_03`

Git repository:

`https://github.com/ksuyamburaj07/NeuroPrompt-3D`

Frozen source commit used for the final refinement pipeline:

`66edbba5a07144b3b6a2f117d9fec22ea52284ef`

Existing tags at that commit before the pre-M11 closure:

- `m9c4-freeze-2026-09-23`
- `m9d-holdout-freeze-2026-09-24`

---

## Frozen BraTS Split

Canonical split:

`splits/brats2024_posttreatment_seed42.json`

SHA-256:

`f28bfe9416308a12cd394915c4fe0e74c1928d79bcc6179d27162b5d87590b6b`

Frozen cohort counts:

- train: 490 subjects / 1078 cases
- validation: 61 subjects / 143 cases
- test: 62 subjects / 129 cases

Canonical sorted final-test case-list SHA-256:

`4cbabc1d71f0f3f47589f4a2677982107480c33cb3b98485f932ead052c7bea2`

---

## Frozen Baseline

Canonical baseline checkpoint:

`checkpoints/M7K_C_Full_Baseline_Training/best_baseline.pt`

SHA-256:

`3834e3215d5d6610e8d952df2e61d7e788de6caaa8db8d9e418e9a3240f60771`

The baseline is the lightweight 3D U-Net used by the frozen final
pipeline.

---

## Frozen SAM-Med3D Dependency

External SAM-Med3D source commit:

`f3de1fa10da98e46f49f176773d2b1e306ba131f`

Frozen checkpoint SHA-256:

`899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c`

Registry/model identity:

`vit_b_ori`

---

## Final Automatic Policy

Final automatic policy:

`NeuroPrompt3D_FinalAutomaticPolicy_v1`

Frozen uncertainty threshold:

`0.21133705228567123`

Decision rule:

- `hotspot_variance > threshold` -> `APPLY_LOCAL_FP`
- otherwise -> `ABSTAIN_BASELINE`

Refinement branch:

- one negative point
- label `0`
- no box
- localized directional support
- outside support, final mask equals baseline
- no morphology

The final-test policy is frozen and must not be retuned under the same
experiment identity.

---

## Primary Final Test

M9E is the primary frozen final-test experiment.

129/129 cases were evaluated only after the prediction set had been
frozen before first GT access.

M9E evidence is preserved under:

`archives/M9E/`

The final-test result was mixed/nonuniform and must be reported as
observed rather than reframed as a uniformly beneficial refinement.

---

## M10

M10 analyzes the frozen M9E results.

Canonical archive:

`archives/M10/`

M10 does not authorize:

- model retraining
- threshold tuning
- policy changes
- prediction regeneration
- final-mask modification

---

# Pre-M11 Reconciliation

A pre-M11 storage and provenance audit was performed on 2026-09-30.

## S1 — Frozen Research Backup

Status: COMPLETE

Important frozen-backup SHA-256 identities:

- archive:
  `a8254ab318ec2c68d4152e352b5af80b5266ad91cdb5e1e95e0daa670304a4ee`
- Git bundle:
  `21d225f9fdbc2a34e73e524824fbdee516008f5e6756b4f454758bc4b95cdfc3`
- manifest:
  `3c4bb08561ef45302aee6ea24bac3d491cb2ca3275162b98074ee7132b5593c0`
- provenance:
  `f02c8eeb940a65107864b538a5320a1aceb869f29661208f6c9babfd4a41454a`
- checksums:
  `f999fb042e7828c6edf6c2f463f8c0ecd265754a57c1a6f4b4b57fac7a60d1fd`

## S2 — Drive Artifact Reconciliation

Status: COMPLETE

Canonical reconciliation manifest SHA-256:

`ebc5a092844d3bc4b9f3d6f1697cec517c44eab8941459533b29b95acf6b4e79`

Canonical reconciliation record SHA-256:

`0e8679c1c5a89174d8674064159825b42de6ef1000f841b4d869ec8dc2f312c9`

Recovered items were placed under:

`imports/drive_reconciliation/2026-09-30/`

without overwriting canonical research artifacts.

## S3 — Notebook Reconciliation

Status: COMPLETE

Local notebook fingerprint report SHA-256:

`869a767c80250121671459a212870155686c89ad22c57c887de9a12651dbe838`

Notebook reconciliation manifest SHA-256:

`224d7c766e595a716b23a1d818b1da3c548ec0504c63050d8fb323eb92f24569`

Notebook reconciliation record SHA-256:

`6f5070e3453a2bce2598ef20aa9969707fe1875296712429d5c2e2c91e3ba374`

S3 concluded:

- no frozen notebook loss detected
- no canonical notebook modified
- no notebook deleted during reconciliation

## S4 — Historical Milestone Audit

Status: COMPLETE

S4A cross-checked the detailed M7 development history using Git commit
history, changed files, notebooks, local evidence, reflog information,
and project-history context.

S4B cross-checked the M8-M10 historical label hierarchy against
notebooks, output directories, archives, and Git history.

No scientific result was changed by S4.

---

# Provenance Rule

Recovered or reconciled artifacts under `imports/` preserve historical
context.

They are not automatically canonical research outputs merely because
they were recovered later.

Where historical plans differ from executed experiments, the final
documentation records both without rewriting one into the other.
