import { isValidResearchRunId } from './researchRunIdentity.ts'
import type { RunStatus } from '../api/runs'

export type RecentResearchRun = {
  run_id: string
  status: RunStatus
  stage: string
  progress: number
  updated_at: string
}

const statuses = new Set<string>([
  'queued',
  'running',
  'complete',
  'failed',
])

function isRecord(value: unknown): value is Record<string, unknown> {
  return (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value)
  )
}

function isTimestamp(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    /(?:Z|[+-]\d{2}:\d{2})$/i.test(value) &&
    Number.isFinite(Date.parse(value))
  )
}

export function parseRecentResearchRuns(
  payload: unknown,
): RecentResearchRun[] {
  if (!isRecord(payload)) {
    throw new Error('Unexpected research discovery response.')
  }

  const { runs, count, limit } = payload

  if (
    !Array.isArray(runs) ||
    !Number.isInteger(count) ||
    !Number.isInteger(limit) ||
    typeof count !== 'number' ||
    typeof limit !== 'number' ||
    limit < 1 ||
    limit > 50 ||
    count !== runs.length ||
    count > limit
  ) {
    throw new Error('Invalid research discovery inventory.')
  }

  const result: RecentResearchRun[] = []
  const ids = new Set<string>()

  for (const item of runs) {
    if (
      !isRecord(item) ||
      !isValidResearchRunId(String(item.run_id)) ||
      typeof item.run_id !== 'string' ||
      !statuses.has(String(item.status)) ||
      typeof item.stage !== 'string' ||
      item.stage.length === 0 ||
      typeof item.progress !== 'number' ||
      !Number.isFinite(item.progress) ||
      item.progress < 0 ||
      item.progress > 1 ||
      !isTimestamp(item.updated_at) ||
      ids.has(item.run_id)
    ) {
      throw new Error('Invalid saved research run metadata.')
    }

    ids.add(item.run_id)

    result.push({
      run_id: item.run_id,
      status: item.status as RunStatus,
      stage: item.stage,
      progress: item.progress,
      updated_at: item.updated_at,
    })
  }

  for (let index = 1; index < result.length; index += 1) {
    if (
      Date.parse(result[index - 1].updated_at) <
      Date.parse(result[index].updated_at)
    ) {
      throw new Error(
        'Research discovery results are not newest-first.',
      )
    }
  }

  return result
}
