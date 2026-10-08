const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ||
  'http://127.0.0.1:8000/api/v1'
).replace(/\/+$/, '')

export type ValidationIssue = {
  code: string
  message: string
  field: string | null
}

export type GeometryInfo = {
  shape_xyz: number[]
  tensor_shape_dhw: number[]
  voxel_spacing_xyz: number[]
  affine: number[][]
  affine_tolerance: number
}

export type CaseValidationResponse = {
  valid: boolean
  status: 'ready' | 'invalid'
  case_id: string | null
  modalities: Record<string, {
    field: string
    display_name: string
    research_core_name: string
    filename: string | null
    valid: boolean | null
  }>
  geometry: GeometryInfo | null
  ground_truth: {
    provided: boolean
    filename: string | null
    valid: boolean | null
    tensor_shape_dhw: number[] | null
    labels: number[] | null
  }
  ready_for_inference: boolean
  errors: ValidationIssue[]
}

export type CaseFiles = {
  t1n: File
  t1c: File
  t2w: File
  t2f: File
  segmentation?: File | null
}

function errorMessage(payload: unknown, fallback: string): string {
  if (typeof payload !== 'object' || payload === null) {
    return fallback
  }

  const detail = (payload as { detail?: unknown }).detail

  if (typeof detail === 'string') {
    return detail
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item: unknown) => {
        if (typeof item !== 'object' || item === null) {
          return null
        }
        const message = (item as { msg?: unknown }).msg
        return typeof message === 'string' ? message : null
      })
      .filter((value): value is string => value !== null)

    if (messages.length) return messages.join('; ')
  }

  return fallback
}

async function apiRequest(
  path: string,
  options: RequestInit,
): Promise<unknown> {
  let response: Response

  try {
    response = await fetch(`${API_BASE_URL}/${path}`, options)
  } catch {
    throw new Error(
      `Cannot reach the NeuroPrompt-3D backend at ${API_BASE_URL}. ` +
      'Check that FastAPI is running.',
    )
  }

  let payload: unknown

  try {
    payload = await response.json()
  } catch {
    throw new Error('The backend returned an unreadable response.')
  }

  if (!response.ok) {
    throw new Error(
      errorMessage(payload, `Backend request failed (HTTP ${response.status}).`),
    )
  }

  return payload
}

export async function validateCase(
  files: CaseFiles,
): Promise<CaseValidationResponse> {
  const form = new FormData()

  form.append('t1n', files.t1n, files.t1n.name)
  form.append('t1c', files.t1c, files.t1c.name)
  form.append('t2w', files.t2w, files.t2w.name)
  form.append('t2f', files.t2f, files.t2f.name)

  if (files.segmentation) {
    form.append(
      'segmentation',
      files.segmentation,
      files.segmentation.name,
    )
  }

  const payload = await apiRequest('cases/validate', {
    method: 'POST',
    body: form,
  })

  if (
    typeof payload !== 'object' ||
    payload === null ||
    !('valid' in payload) ||
    typeof payload.valid !== 'boolean' ||
    !('ready_for_inference' in payload) ||
    typeof payload.ready_for_inference !== 'boolean' ||
    !('errors' in payload) ||
    !Array.isArray(payload.errors)
  ) {
    throw new Error('Unexpected MRI validation response from the backend.')
  }

  return payload as CaseValidationResponse
}

export async function deleteStagedCase(
  caseId: string,
): Promise<void> {
  const payload = await apiRequest(
    `cases/${encodeURIComponent(caseId)}`,
    { method: 'DELETE' },
  )

  if (
    typeof payload !== 'object' ||
    payload === null ||
    !('deleted' in payload) ||
    payload.deleted !== true
  ) {
    throw new Error('The backend did not confirm case deletion.')
  }
}
