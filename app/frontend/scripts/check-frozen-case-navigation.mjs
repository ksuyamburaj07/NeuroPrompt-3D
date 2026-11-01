import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'

const source = readFileSync(
  new URL('../src/utils/frozenCaseNavigator.ts', import.meta.url),
  'utf8',
)

const compiled = ts.transpileModule(source, {
  fileName: 'frozenCaseNavigator.ts',
  reportDiagnostics: true,
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
})

const errors = (compiled.diagnostics ?? []).filter(
  (item) => item.category === ts.DiagnosticCategory.Error,
)

assert.equal(errors.length, 0)

const url = `data:text/javascript;base64,${
  Buffer.from(compiled.outputText).toString('base64')
}`

const {
  FROZEN_CASE_COUNT,
  FROZEN_CASE_PAGE_SIZE,
  parseFrozenCaseNumber,
  frozenPageForCase,
  adjacentFrozenCase,
} = await import(url)

assert.equal(FROZEN_CASE_COUNT, 129)
assert.equal(FROZEN_CASE_PAGE_SIZE, 20)

console.log('=== M11E12E11 CASE NAVIGATOR TESTS ===')

for (const [text, expected] of [
  ['1', 1],
  ['001', 1],
  ['27', 27],
  ['087', 87],
  ['129', 129],
  [' 052 ', 52],
]) {
  assert.equal(parseFrozenCaseNumber(text), expected)
}

for (const text of [
  '',
  '0',
  '000',
  '130',
  '999',
  '-1',
  '1.5',
  'NaN',
  'Case 27',
  '0010',
]) {
  assert.equal(parseFrozenCaseNumber(text), null)
}

console.log('PASS: Valid and invalid direct-entry inputs')

for (let index = 1; index <= 129; index++) {
  const page = frozenPageForCase(index)

  assert.ok(page >= 0 && page <= 6)
  assert.ok(index > page * 20)
  assert.ok(index <= (page + 1) * 20)

  const previous = adjacentFrozenCase(index, -1)
  const next = adjacentFrozenCase(index, 1)

  assert.equal(previous, index === 1 ? null : index - 1)
  assert.equal(next, index === 129 ? null : index + 1)
}

assert.equal(frozenPageForCase(1), 0)
assert.equal(frozenPageForCase(20), 0)
assert.equal(frozenPageForCase(21), 1)
assert.equal(frozenPageForCase(87), 4)
assert.equal(frozenPageForCase(120), 5)
assert.equal(frozenPageForCase(121), 6)
assert.equal(frozenPageForCase(129), 6)

console.log('PASS: All 129 case-to-page mappings')
console.log('PASS: Previous/Next boundaries and transitions')

for (const invalid of [0, 130, -1, 1.5, NaN]) {
  assert.throws(() => frozenPageForCase(invalid), RangeError)
}

assert.throws(() => adjacentFrozenCase(27, 0), RangeError)

console.log('PASS: Invalid navigation rejected')
console.log('PASS: Production navigation geometry verified')
