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
