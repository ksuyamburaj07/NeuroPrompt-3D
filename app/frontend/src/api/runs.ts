const API_BASE = (
  import.meta.env.VITE_API_BASE_URL ||
  'http://127.0.0.1:8000/api/v1'
).replace(/\/+$/, '')

export type RunStatus = 'queued' | 'running' | 'complete' | 'failed'

export type RunStage =
  | 'queued'
  | 'validating'
  | 'loading_models'
  | 'automatic_pipeline'
  | 'preprocessing'
  | 'baseline'
  | 'mc_dropout'
  | 'hotspot'
  | 'policy'
  | 'sam_refinement'
  | 'finalizing'
  | 'complete'
  | 'failed'

export type RunView = {
  run_id: string
  case_id: string
  status: RunStatus
  stage: RunStage
  progress: number
  created_at: string
  updated_at: string
  frozen_variance_threshold: number
  execution_device?: string | null
  worker_pid?: number | null
  inference_case_id?: string | null
  mc_case_seed?: number | null
  action?: string | null
  gate_state?: string | null
  hotspot_zyx?: number[] | null
  hotspot_variance?: number | null
  fp_prompt_zyx?: number[] | null
  fp_prompt_model_xyz?: number[] | null
  sam_used?: boolean | null
  sam_refinement_skipped?: boolean | null
  semantic_abstention_condition?: string | null
  available_artifacts?: string[]
  error?: {
    code: string
    message: string
  } | null
}

function extractError(payload: unknown, fallback: string): string {
  if (typeof payload !== 'object' || payload === null) return fallback

  const detail = (payload as { detail?: unknown }).detail

  if (typeof detail === 'string') return detail

  if (Array.isArray(detail)) {
    const messages = detail.flatMap((item: unknown) => {
      if (typeof item !== 'object' || item === null) return []
      const msg = (item as { msg?: unknown }).msg
      return typeof msg === 'string' ? [msg] : []
    })
    if (messages.length) return messages.join('; ')
  }

  return fallback
}

function isRunView(value: unknown): value is RunView {
  if (typeof value !== 'object' || value === null) return false

  const v = value as Record<string, unknown>

  return (
    typeof v.run_id === 'string' &&
    typeof v.case_id === 'string' &&
    ['queued', 'running', 'complete', 'failed'].includes(String(v.status)) &&
    typeof v.stage === 'string' &&
    typeof v.progress === 'number' &&
    Number.isFinite(v.progress) &&
    v.progress >= 0 &&
    v.progress <= 1 &&
    typeof v.frozen_variance_threshold === 'number'
  )
}

async function requestRun(
  path: string,
  method: 'GET' | 'POST',
): Promise<RunView> {
  let response: Response

  try {
    response = await fetch(`${API_BASE}/${path}`, { method })
  } catch {
    throw new Error(
      'Cannot reach FastAPI. The connection may have been interrupted.',
    )
  }

  let payload: unknown

  try {
    payload = await response.json()
  } catch {
    throw new Error('FastAPI returned an unreadable response.')
  }

  if (!response.ok) {
    throw new Error(
      extractError(payload, `Run request failed: HTTP ${response.status}.`),
    )
  }

  if (!isRunView(payload)) {
    throw new Error('FastAPI returned an unexpected run representation.')
  }

  return payload
}

export function createRun(caseId: string): Promise<RunView> {
  return requestRun(
    `cases/${encodeURIComponent(caseId)}/runs`,
    'POST',
  )
}

export function executeRun(runId: string): Promise<RunView> {
  return requestRun(
    `runs/${encodeURIComponent(runId)}/execute`,
    'POST',
  )
}

export function getRun(runId: string): Promise<RunView> {
  return requestRun(
    `runs/${encodeURIComponent(runId)}`,
    'GET',
  )
}

export function artifactUrl(runId: string, artifactName: string): string {
  return (
    `${API_BASE}/runs/${encodeURIComponent(runId)}` +
    `/artifacts/${encodeURIComponent(artifactName)}`
  )
}
