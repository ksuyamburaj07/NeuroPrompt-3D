import assert from 'node:assert/strict'

import {
  parseRecentResearchRuns,
} from '../src/utils/recentResearchRuns.ts'

console.log('=== M11E13B RESEARCH DISCOVERY PARSER TESTS ===')

const ids = [
  'run_d38405cee2a156c98a9578757742b131',
  'run_7fd922c98cda4fd7b359d61b5c304b5b',
  'run_16a49dbf55664daca7c26fc0cb722124',
]

const timestamps = [
  '2026-10-23T19:37:28.315016Z',
  '2026-10-07T12:57:56.970721Z',
  '2026-10-06T18:43:56.753022Z',
]

const records = ids.map((run_id, index) => ({
  run_id,
  status: 'complete',
  stage: 'complete',
  progress: 1,
  updated_at: timestamps[index],
}))

const valid = {
  runs: records,
  count: 3,
  limit: 20,
}

const parsed = parseRecentResearchRuns(valid)

assert.deepEqual(
  parsed.map((run) => run.run_id),
  ids,
)

console.log('PASS: Three real run identifiers parsed in order.')

assert.deepEqual(
  parseRecentResearchRuns({
    runs: [],
    count: 0,
    limit: 20,
  }),
  [],
)

console.log('PASS: Empty valid inventory accepted.')

const withPrivateFields = {
  runs: [
    {
      ...records[0],
      case_id: 'must-not-appear',
      artifacts: { private: '/tmp/private' },
    },
  ],
  count: 1,
  limit: 20,
}

const sanitized = parseRecentResearchRuns(withPrivateFields)

assert.equal(
  Object.prototype.hasOwnProperty.call(
    sanitized[0],
    'case_id',
  ),
  false,
)

assert.equal(
  Object.prototype.hasOwnProperty.call(
    sanitized[0],
    'artifacts',
  ),
  false,
)

console.log('PASS: Extra private fields are not retained.')

const invalid = [
  null,
  {},
  { runs: [], count: 1, limit: 20 },
  { runs: [], count: 0, limit: 0 },
  { runs: [], count: 0, limit: 51 },
  { runs: [], count: 0, limit: '20' },
  { runs: records, count: 2, limit: 20 },
  { runs: [records[0], records[0]], count: 2, limit: 20 },
  {
    runs: [...records].reverse(),
    count: 3,
    limit: 20,
  },
  {
    runs: [{ ...records[0], run_id: 'invalid' }],
    count: 1,
    limit: 20,
  },
  {
    runs: [{ ...records[0], status: 'invented' }],
    count: 1,
    limit: 20,
  },
  {
    runs: [{ ...records[0], progress: 1.5 }],
    count: 1,
    limit: 20,
  },
  {
    runs: [{ ...records[0], progress: NaN }],
    count: 1,
    limit: 20,
  },
  {
    runs: [{ ...records[0], updated_at: '2026-10-23' }],
    count: 1,
    limit: 20,
  },
]

for (const payload of invalid) {
  assert.throws(
    () => parseRecentResearchRuns(payload),
    Error,
  )
}

console.log(
  `PASS: ${invalid.length} malformed inventory cases rejected.`,
)

console.log('PASS: Production research discovery parser tests.')
