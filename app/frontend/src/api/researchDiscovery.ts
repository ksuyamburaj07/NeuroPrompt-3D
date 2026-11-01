import {
  parseRecentResearchRuns,
  type RecentResearchRun,
} from '../utils/recentResearchRuns'

const API_BASE = (
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.DEV ? 'http://127.0.0.1:8000/api/v1' : '/api/v1')
).replace(/\/+$/, '')

export async function getRecentResearchRuns(
  limit = 20,
  signal?: AbortSignal,
): Promise<RecentResearchRun[]> {
  if (
    !Number.isInteger(limit) ||
    limit < 1 ||
    limit > 50
  ) {
    throw new Error('Invalid recent-run listing limit.')
  }

  let response: Response

  try {
    response = await fetch(
      `${API_BASE}/research/runs?limit=${limit}`,
      {
        method: 'GET',
        cache: 'no-store',
        signal,
      },
    )
  } catch (error) {
    if (
      error instanceof Error &&
      error.name === 'AbortError'
    ) {
      throw error
    }

    throw new Error(
      'Cannot reach the local research discovery service.',
      { cause: error },
    )
  }

  if (response.status === 404) {
    throw new Error(
      'Local run discovery is disabled or unavailable. ' +
      'You can still inspect a known run ID manually.',
    )
  }

  if (!response.ok) {
    throw new Error(
      `Research run discovery failed: HTTP ${response.status}.`,
    )
  }

  let payload: unknown

  try {
    payload = await response.json()
  } catch {
    throw new Error(
      'Research discovery returned an unreadable response.',
    )
  }

  return parseRecentResearchRuns(payload)
}
