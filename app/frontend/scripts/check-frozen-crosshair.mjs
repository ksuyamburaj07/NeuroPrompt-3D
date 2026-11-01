import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'

const source = readFileSync(
  new URL('../src/utils/frozenCrosshairGeometry.ts', import.meta.url),
  'utf8',
)

const compiled = ts.transpileModule(source, {
  fileName: 'frozenCrosshairGeometry.ts',
  reportDiagnostics: true,
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
})

const errors = (compiled.diagnostics ?? []).filter(
  (item) => item.category === ts.DiagnosticCategory.Error,
)

assert.equal(errors.length, 0, 'Geometry transpilation failed')

const moduleUrl = `data:text/javascript;base64,${
  Buffer.from(compiled.outputText).toString('base64')
}`

const { projectCrosshairPixel } = await import(moduleUrl)

const references = [
  {
    xyz: { x: 105, y: 54, z: 35 },
    axial: { column: 105, row: 163 },
    coronal: { column: 105, row: 146 },
    sagittal: { column: 54, row: 146 },
  },
  {
    xyz: { x: 49, y: 85, z: 55 },
    axial: { column: 49, row: 132 },
    coronal: { column: 49, row: 126 },
    sagittal: { column: 85, row: 126 },
  },
  {
    xyz: { x: 128, y: 117, z: 68 },
    axial: { column: 128, row: 100 },
    coronal: { column: 128, row: 113 },
    sagittal: { column: 117, row: 113 },
  },
]

const planes = {
  axial: [182, 218],
  coronal: [182, 182],
  sagittal: [218, 182],
}

console.log('=== M11E12E9 PRODUCTION CROSSHAIR TESTS ===')

for (const reference of references) {
  for (const plane of Object.keys(planes)) {
    assert.deepEqual(
      projectCrosshairPixel(plane, reference.xyz),
      reference[plane],
    )
  }
  console.log('PASS: Reference voxel', reference.xyz)
}

const corners = [
  { x: 0, y: 0, z: 0 },
  { x: 181, y: 217, z: 181 },
]

for (const xyz of corners) {
  for (const [plane, [width, height]] of Object.entries(planes)) {
    const pixel = projectCrosshairPixel(plane, xyz)

    assert.ok(pixel.column >= 0 && pixel.column < width)
    assert.ok(pixel.row >= 0 && pixel.row < height)
  }
}

console.log('PASS: Minimum and maximum voxel corners')

const base = { x: 49, y: 85, z: 55 }

function projections(xyz) {
  return {
    axial: projectCrosshairPixel('axial', xyz),
    coronal: projectCrosshairPixel('coronal', xyz),
    sagittal: projectCrosshairPixel('sagittal', xyz),
  }
}

const initial = projections(base)

const changedX = projections({ ...base, x: 50 })
assert.equal(changedX.axial.column, initial.axial.column + 1)
assert.equal(changedX.coronal.column, initial.coronal.column + 1)
assert.deepEqual(changedX.sagittal, initial.sagittal)

const changedY = projections({ ...base, y: 86 })
assert.equal(changedY.axial.row, initial.axial.row - 1)
assert.equal(changedY.sagittal.column, initial.sagittal.column + 1)
assert.deepEqual(changedY.coronal, initial.coronal)

const changedZ = projections({ ...base, z: 56 })
assert.equal(changedZ.coronal.row, initial.coronal.row - 1)
assert.equal(changedZ.sagittal.row, initial.sagittal.row - 1)
assert.deepEqual(changedZ.axial, initial.axial)

console.log('PASS: X/Y/Z synchronization invariants')

let count = 0

for (let x = 0; x < 182; x++) {
  for (let y = 0; y < 218; y++) {
    for (const z of [0, 35, 55, 91, 181]) {
      const xyz = { x, y, z }

      for (const [plane, [width, height]] of Object.entries(planes)) {
        const { column, row } = projectCrosshairPixel(plane, xyz)

        assert.ok(Number.isInteger(column))
        assert.ok(Number.isInteger(row))

        assert.ok(column >= 0 && column < width)
        assert.ok(row >= 0 && row < height)

        // The SVG intersection uses these pixel centers.
        assert.ok(column + 0.5 > 0 && column + 0.5 < width)
        assert.ok(row + 0.5 > 0 && row + 0.5 < height)

        count++
      }
    }
  }
}

assert.equal(count, 595140)
console.log(`PASS: ${count.toLocaleString()} pixel and center checks`)

for (const xyz of [
  { x: -1, y: 85, z: 55 },
  { x: 182, y: 85, z: 55 },
  { x: 49, y: 218, z: 55 },
  { x: 49, y: 85, z: 182 },
  { x: 49.5, y: 85, z: 55 },
]) {
  assert.throws(
    () => projectCrosshairPixel('axial', xyz),
    RangeError,
  )
}

assert.throws(
  () => projectCrosshairPixel('oblique', base),
  RangeError,
)

console.log('PASS: Invalid coordinates and planes rejected')
console.log('PASS: Production crosshair geometry regression')
