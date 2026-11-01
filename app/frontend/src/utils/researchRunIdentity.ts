/**
 * Research run identifiers are internal development-run IDs,
 * not frozen final-test case identifiers.
 */
export function normalizeResearchRunId(input: string): string {
  return input.trim()
}

export function isValidResearchRunId(value: string): boolean {
  return /^run_[0-9a-f]{32}$/.test(value)
}
