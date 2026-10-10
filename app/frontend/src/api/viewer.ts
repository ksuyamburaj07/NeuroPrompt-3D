const API_BASE = (
  import.meta.env.VITE_API_BASE_URL ||
  'http://127.0.0.1:8000/api/v1'
).replace(/\/+$/, '')

export type MRIPlane = 'axial' | 'coronal' | 'sagittal'
export type MRIModality = 't1n' | 't1c' | 't2w' | 't2f'

export type MRIPlaneInfo = {
  axis: number
  count: number
  center_index: number
  width: number
  height: number
}

export type MRIViewerMetadata = {
  case_id: string
  orientation_convention: string
  original_axis_codes: string[]
  shape_ras_xyz: number[]
  affine_ras: number[][]
  voxel_spacing_ras_mm: number[]
  display_intensity: string
  planes: Record<MRIPlane, MRIPlaneInfo>
}

export async function getViewerMetadata(
  caseId: string,
): Promise<MRIViewerMetadata> {
  let response: Response

  try {
    response = await fetch(
      `${API_BASE}/cases/${encodeURIComponent(caseId)}/viewer/metadata`,
      { cache: 'no-store' },
    )
  } catch {
    throw new Error('Cannot contact the MRI viewer backend.')
  }

  if (!response.ok) {
    throw new Error(
      `Unable to load MRI geometry (HTTP ${response.status}).`,
    )
  }

  const payload: unknown = await response.json()

  if (
    typeof payload !== 'object' ||
    payload === null ||
    !('planes' in payload) ||
    !('shape_ras_xyz' in payload)
  ) {
    throw new Error('Invalid MRI viewer metadata response.')
  }

  return payload as MRIViewerMetadata
}

export function viewerSliceUrl(
  caseId: string,
  modality: MRIModality,
  plane: MRIPlane,
  index: number,
): string {
  return (
    `${API_BASE}/cases/${encodeURIComponent(caseId)}` +
    `/viewer/slices/${modality}/${plane}/${index}`
  )
}


export type MRIOverlayLayer = 'baseline' | 'final' | 'removed'

export function viewerOverlayUrl(
  caseId: string,
  runId: string,
  layer: MRIOverlayLayer,
  plane: MRIPlane,
  index: number,
): string {
  return (
    `${API_BASE}/cases/${encodeURIComponent(caseId)}` +
    `/viewer/overlays/${encodeURIComponent(runId)}` +
    `/${layer}/${plane}/${index}`
  )
}

/* M11E3 — Frozen scientific marker inspection */

export type ViewerSpatialMarker = {
  source_zyx: number[]
  ras_xyz: number[]
  world_ras_mm: number[]
}

export type ViewerRunMarkers = {
  case_id: string
  run_id: string
  orientation_convention: 'RAS+'
  action: string | null
  gate_state: string | null
  hotspot_variance: number | null
  hotspot: ViewerSpatialMarker | null
  negative_prompt: ViewerSpatialMarker | null
  sam_used: boolean | null
}

export async function getViewerRunMarkers(
  caseId: string,
  runId: string,
): Promise<ViewerRunMarkers> {
  let response: Response

  try {
    response = await fetch(
      `${API_BASE}/cases/${encodeURIComponent(caseId)}` +
      `/viewer/markers/${encodeURIComponent(runId)}`,
      { cache: 'no-store' },
    )
  } catch {
    throw new Error('Cannot contact the scientific marker backend.')
  }

  if (!response.ok) {
    throw new Error(
      `Unable to load scientific markers (HTTP ${response.status}).`,
    )
  }

  const payload: unknown = await response.json()

  if (
    typeof payload !== 'object' ||
    payload === null ||
    !('case_id' in payload) ||
    payload.case_id !== caseId ||
    !('run_id' in payload) ||
    payload.run_id !== runId ||
    !('orientation_convention' in payload) ||
    payload.orientation_convention !== 'RAS+'
  ) {
    throw new Error('Invalid scientific marker response.')
  }

  return payload as ViewerRunMarkers
}

/* M11E4 — read-only frozen segmentation mesh */

export type ViewerMeshLayer = 'baseline' | 'final' | 'removed'

export function viewerMeshUrl(
  caseId: string,
  runId: string,
  layer: ViewerMeshLayer,
): string {
  return (
    `${API_BASE}/cases/${encodeURIComponent(caseId)}` +
    `/viewer/meshes/${encodeURIComponent(runId)}/${layer}`
  )
}


/* M11E4G — MRI-derived anatomical visualization context */

export type AnatomyFinish = 'raw' | 'soft'

export function viewerAnatomyUrl(
  caseId: string,
  step: 1 | 2 = 2,
  finish: AnatomyFinish = 'raw',
): string {
  return (
    `${API_BASE}/cases/${encodeURIComponent(caseId)}` +
    `/viewer/anatomy/brain?step=${step}&finish=${finish}`
  )
}
