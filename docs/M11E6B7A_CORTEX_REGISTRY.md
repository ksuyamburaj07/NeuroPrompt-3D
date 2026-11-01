# M11E6B7A — Read-only cortical reconstruction registry

The optional backend registry binds externally verified cortical reconstructions
to the SHA256 of the **exact staged source T1N file bytes**, not application case
UUIDs, upload filenames, voxel values, dimensions or affines. Reuploading identical
bytes under another case UUID can match; recompressing or rewriting a NIfTI cannot
match until independently registered. No inference run or ground truth is needed.

## Configuration and QC

`configs/cortex_registry.json` is the default metadata-only registry. Override it
with an absolute `NEUROPROMPT_CORTEX_REGISTRY` path before starting FastAPI. Set the
variable to an empty string, or set registry `enabled` to `false`, to disable cortex.
The application reads configuration and assets on each request; it never writes
them. Missing, unreadable or invalid configuration disables this optional feature.

The version 1 schema is defined in `app/backend/schemas/cortex.py`. Each entry
requires a unique reconstruction ID and source T1N SHA256, an absolute external
asset directory, distinct relative LH/RH PLY paths and SHA256 values, explicit
`physical RAS+` coordinates in `mm`, and explicit research QC. Both hemispheres
are required. Duplicate IDs or source hashes invalidate the registry rather than
selecting an ambiguous entry. Add independently verified cases as additional
entries; no allowed application case IDs are configured or hardcoded.

An operator must independently verify patient/source provenance, physical RAS+
alignment and mesh quality before setting `qc.status` to `approved`. Other states
are `provisional` and `rejected`; neither can be displayed. All entries have
`scope: research` and `clinically_validated: false`. Research QC approval is not
clinical validation. Keep the registry under trusted operator control and treat
staged sources and registered assets as immutable while serving requests.

The only seeded entry is experimental FreeSurfer reconstruction `m11e6b5_trial01`.
Its supplied source checksum and both mesh checksums are recorded, with
**provisional research QC**, so it is deliberately **not display-eligible**.
The original MRI and PLY files remain in external storage; no assets are copied
into Git. PLY and medical image files are ignored by Git.

## Read-only API

- `GET /api/v1/cases/{case_id}/viewer/cortex/status`: JSON availability, stable
  reason code, matched reconstruction ID and research QC; no filesystem paths or
  unmatched entry metadata. Returns 200 for unavailable optional cortex, 400 for
  invalid case identifiers, and 404 for absent staged cases.
- `GET /api/v1/cases/{case_id}/viewer/cortex/{hemisphere}`: `lh` or `rh` verified
  PLY bytes as `application/octet-stream`, with `X-Mesh-Coordinates: RAS+
  millimetres`. Invalid hemisphere values return 400. Missing cases or ineligible
  reconstructions return 404 without revealing asset locations.

Successful status and mesh responses use `Cache-Control: no-store`. Both endpoints
require validated ready-case metadata and freshly hash the staged T1N and **both**
mesh payloads. Mesh responses return the actual verified bytes rather than reopening
an unchecked file. Asset paths must remain under the configured external root;
absolute mesh paths, traversal, symbolic links and assets inside the checkout are
rejected. Missing or corrupt assets disable the whole reconstruction.

Status reasons are `available`, `case_not_ready`, `registry_disabled`,
`registry_missing`, `registry_invalid`, `source_unavailable`, `source_unregistered`,
`assets_unavailable`, `asset_integrity_failed`, and `qc_not_approved`.

## Scope and limitations

This backend serves precomputed assets only. It performs no reconstruction,
geometric registration, clinical validation, inference or mask changes. Checksums
prove registered byte identity; they cannot independently establish that an
operator registered the correct patient's assets or coordinate convention.
The scientific core and frontend are unchanged. Existing MRI, brain anatomy and
segmentation viewers continue independently of registry availability.

Verification is deliberately uncached to detect source/mesh changes, at the cost
of source and mesh disk reads on every cortex request. Provision immutable external
storage and trusted configuration; no enrollment, QC editing or asset mutation
endpoint is provided.

M11E6B7C — Validation-time source binding:
Newly validated cases store the original staged T1N SHA-256 in case metadata. Cortex availability requires this fingerprint to match the currently staged T1N and the registered reconstruction source identity. A changed source produces source_identity_mismatch and blocks both hemispheres. Legacy cases without a recorded fingerprint remain accessible through the standard MRI viewer but cannot use registered cortical reconstructions until freshly validated. This protects against unrecorded source substitution, but does not independently protect against a local actor modifying both source bytes and metadata or concurrent filesystem replacement.
