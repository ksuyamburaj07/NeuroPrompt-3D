# NeuroPrompt-3D Scientific Boundaries

## Purpose

This document defines the scientific and application boundaries that
must remain intact after completion of M10.

---

# 1. Frozen Source Boundary

The source state used for the frozen automatic-refinement pipeline is:

`66edbba5a07144b3b6a2f117d9fec22ea52284ef`

This commit remains a permanent reproducibility reference even after
later documentation and M11 application-development commits are added.

---

# 2. M9E Final-Test Boundary

M9E is the primary frozen final-test experiment.

Under the M9E experiment identity, do not change:

- test cohort membership
- baseline checkpoint
- MC-Dropout configuration
- uncertainty-map definition
- uncertainty threshold
- automatic action policy
- prompt rule
- SAM-Med3D checkpoint
- compositing support rule
- frozen predictions
- final masks
- final evaluation records

Any future scientific modification requires a new experiment/version
identity.

---

# 3. Ground-Truth Firewall

The M9E workflow separated:

1. MRI-only prediction
2. pre-GT prediction freeze
3. verification
4. first GT access
5. evaluation

This ordering is part of the experiment's scientific integrity.

It must not be rewritten retrospectively.

---

# 4. M10 Boundary

M10 is descriptive/exploratory analysis of already-frozen M9E outputs.

M10 may summarize and visualize the frozen evidence.

M10 must not be used to justify post-test tuning of the original M9E
policy.

---

# 5. Frozen Paths

Existing provenance-sensitive paths remain unchanged, including:

- `archives/M9E/`
- `archives/M10/`
- `archives/M9C4/`
- `archives/M9D/`
- `outputs/M8_MC_Dropout_Uncertainty/`
- `outputs/M8_Automatic_Prompting/`
- `outputs/M9_SAM_Med3D_Integration/`
- `outputs/M7K_D_Baseline_Evaluation/`
- `checkpoints/M7K_C_Full_Baseline_Training/`

Human-readable documentation may explain these paths but should not
rename them merely for cosmetic consistency.

---

# 6. Reconciliation Boundary

The directories:

- `imports/drive_reconciliation/`
- `imports/notebook_reconciliation/`

contain historical/reconciliation material recovered during the
pre-M11 audit.

They must not silently replace canonical artifacts.

---

# 7. M11 Boundary

M11 is a new application and visualization phase built on the frozen
research core.

M11 may:

- expose frozen results in read-only form
- run new user-uploaded cases
- visualize MRI, uncertainty, prompts, and masks
- calculate metrics for new uploaded cases when optional GT is supplied
- provide backend/API/UI layers around the existing pipeline

M11 must not silently regenerate or overwrite M9E/M10 evidence.

---

# 8. Frozen vs Live Presentation

The UI should visibly distinguish frozen experimental evidence from new
user-uploaded inference.

Recommended frozen-result label:

`FROZEN RESEARCH RESULT | M9E Final Test · Read Only`

Recommended live-result label:

`LIVE INFERENCE | User-uploaded MRI · New Result`

---

# 9. User Uploads

The primary M11 live-inference workflow accepts the four required MRI
modalities:

- T1n
- T1c
- T2w
- T2f

Before inference, the application should validate:

- file type
- NIfTI readability
- shape compatibility
- affine/geometry compatibility
- modality completeness

Ground truth is optional and belongs to a separate research/evaluation
mode.

Ground truth must not be required for ordinary inference.

---

# 10. Research-Only Positioning

NeuroPrompt-3D is a research and educational prototype.

It must not be represented as:

- a clinical diagnostic system
- a substitute for a radiologist
- clinically validated for patient care
- an approved medical device

---

# 11. Pre-M11 Closure Rule

Once the pre-M11 documentation/Git checkpoint is created:

- M1-M10 history is treated as closed historical provenance
- future UI/application work proceeds as M11
- later corrections to historical understanding should be additive
  provenance notes, not silent edits to frozen scientific evidence
