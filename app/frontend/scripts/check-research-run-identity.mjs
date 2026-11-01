import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  isValidResearchRunId,
  normalizeResearchRunId,
} from '../src/utils/researchRunIdentity.ts'

console.log('=== M11E13A PRODUCTION RUN-ID TESTS ===')

const realRunIds = [
  'run_16a49dbf55664daca7c26fc0cb722124',
  'run_7fd922c98cda4fd7b359d61b5c304b5b',
  'run_d38405cee2a156c98a9578757742b131',
]

for (const runId of realRunIds) {
  assert.equal(isValidResearchRunId(runId), true)
}

console.log('PASS: Three existing run identifiers accepted.')

const zeroId = `run_${'0'.repeat(32)}`

assert.equal(isValidResearchRunId(zeroId), true)
assert.equal(zeroId.length, 36)

console.log('PASS: Valid-format missing-run identifier accepted.')

const invalidIds = [
  '',
  'abc',
  'run_',
  `run_${'0'.repeat(31)}`,
  `run_${'0'.repeat(33)}`,
  `RUN_${'0'.repeat(32)}`,
  `run_${'A'.repeat(32)}`,
  `run_${'g'.repeat(32)}`,
  `run_${'0'.repeat(32)}/../`,
  `${zeroId}?execute=true`,
  ` ${zeroId}`,
  `${zeroId} `,
  `${zeroId}\n`,
]

for (const candidate of invalidIds) {
  assert.equal(
    isValidResearchRunId(candidate),
    false,
    `Should reject: ${JSON.stringify(candidate)}`,
  )
}

console.log('PASS: Invalid lengths, case and paths rejected.')

assert.equal(
  normalizeResearchRunId(`  ${realRunIds[0]}  `),
  realRunIds[0],
)

console.log('PASS: Input whitespace normalized as intended.')

for (let index = 0; index < 129; index += 1) {
  const identifier =
    `run_${index.toString(16).padStart(32, '0')}`

  assert.equal(isValidResearchRunId(identifier), true)
}

console.log('PASS: 129 generated valid-format identifiers.')

const scriptDir = dirname(fileURLToPath(import.meta.url))

const component = readFileSync(
  resolve(
    scriptDir,
    '../src/components/ResearchRunInspector.tsx',
  ),
  'utf8',
)

assert.ok(component.includes('isValidResearchRunId(runId)'))
assert.ok(component.includes('normalizeResearchRunId(value)'))
assert.ok(component.includes('void inspectRunId(query)'))
assert.ok(component.includes('void inspectRunId(runId)'))
assert.ok(component.includes('await getRun(runId)'))
assert.ok(component.includes('sequence.current !== requestNumber'))
assert.ok(component.includes("setInspection({ kind: 'idle' })"))

for (const forbidden of [
  'createRun(',
  'executeRun(',
  'artifactUrl(',
]) {
  assert.ok(
    !component.includes(forbidden),
    `Unexpected execution/download action: ${forbidden}`,
  )
}

console.log('PASS: Inspector preserves read-only source safeguards.')
console.log('PASS: Production research identity regression.')
