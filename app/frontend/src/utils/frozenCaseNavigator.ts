/**
 * Display-only navigation for the frozen 129-case cohort.
 * Does not load, alter or regenerate scientific evidence.
 */

export const FROZEN_CASE_COUNT = 129
export const FROZEN_CASE_PAGE_SIZE = 20

export function parseFrozenCaseNumber(
  raw: string,
): number | null {
  const text = raw.trim()

  if (!/^[0-9]{1,3}$/.test(text)) {
    return null
  }

  const index = Number(text)

  return index >= 1 && index <= FROZEN_CASE_COUNT
    ? index
    : null
}

export function frozenPageForCase(index: number): number {
  if (
    !Number.isInteger(index) ||
    index < 1 ||
    index > FROZEN_CASE_COUNT
  ) {
    throw new RangeError('Invalid frozen case index.')
  }

  return Math.floor((index - 1) / FROZEN_CASE_PAGE_SIZE)
}

export function adjacentFrozenCase(
  index: number,
  direction: -1 | 1,
): number | null {
  if (
    !Number.isInteger(index) ||
    index < 1 ||
    index > FROZEN_CASE_COUNT ||
    (direction !== -1 && direction !== 1)
  ) {
    throw new RangeError('Invalid frozen case navigation.')
  }

  const next = index + direction

  return next >= 1 && next <= FROZEN_CASE_COUNT
    ? next
    : null
}
