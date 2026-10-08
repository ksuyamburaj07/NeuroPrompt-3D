export type SavedRunReference = {
  runId: string
  caseId: string
}

const STORAGE_KEY = 'neuroprompt3d.live-run.v1'

function isReference(value: unknown): value is SavedRunReference {
  if (typeof value !== 'object' || value === null) return false

  const ref = value as Record<string, unknown>

  return (
    typeof ref.runId === 'string' &&
    ref.runId.startsWith('run_') &&
    ref.runId.length <= 128 &&
    typeof ref.caseId === 'string' &&
    ref.caseId.startsWith('case_') &&
    ref.caseId.length <= 128
  )
}

export function saveRunRecovery(ref: SavedRunReference): void {
  if (!isReference(ref) || typeof window === 'undefined') return

  try {
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(ref))
  } catch {
    // Storage may be unavailable; manual run-ID recovery still works.
  }
}

export function readRunRecovery(): SavedRunReference | null {
  if (typeof window === 'undefined') return null

  try {
    const raw = window.sessionStorage.getItem(STORAGE_KEY)
    if (!raw) return null

    const parsed: unknown = JSON.parse(raw)
    return isReference(parsed) ? parsed : null
  } catch {
    return null
  }
}

export function clearRunRecovery(): void {
  if (typeof window === 'undefined') return

  try {
    window.sessionStorage.removeItem(STORAGE_KEY)
  } catch {
    // Clearing browser storage must not interfere with backend runs.
  }
}
