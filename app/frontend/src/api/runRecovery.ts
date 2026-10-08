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

/* M11D3B2 — pending idempotent run creation */

const PENDING_CREATION_KEY = 'neuroprompt3d.pending-run-creation.v1'

export type PendingRunCreation = {
  caseId: string
  idempotencyKey: string
}

export function readPendingRunCreation(): PendingRunCreation | null {
  if (typeof window === 'undefined') return null

  try {
    const raw = window.sessionStorage.getItem(PENDING_CREATION_KEY)
    if (!raw) return null

    const parsed: unknown = JSON.parse(raw)

    if (typeof parsed !== 'object' || parsed === null) return null

    const value = parsed as Record<string, unknown>

    if (
      typeof value.caseId !== 'string' ||
      !/^case_[0-9a-f]{32}$/.test(value.caseId) ||
      typeof value.idempotencyKey !== 'string' ||
      !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value.idempotencyKey)
    ) {
      return null
    }

    return {
      caseId: value.caseId,
      idempotencyKey: value.idempotencyKey,
    }
  } catch {
    return null
  }
}

export function getOrCreateRunCreationKey(caseId: string): string {
  const existing = readPendingRunCreation()

  if (existing?.caseId === caseId) {
    return existing.idempotencyKey
  }

  const key = crypto.randomUUID()

  try {
    window.sessionStorage.setItem(
      PENDING_CREATION_KEY,
      JSON.stringify({ caseId, idempotencyKey: key }),
    )
  } catch {
    // The active component also retains the key in memory.
  }

  return key
}

export function clearPendingRunCreation(): void {
  if (typeof window === 'undefined') return

  try {
    window.sessionStorage.removeItem(PENDING_CREATION_KEY)
  } catch {
    // Storage cleanup must not affect backend execution.
  }
}
