import assert from 'node:assert/strict'
import {
  BufferGeometry,
  Float32BufferAttribute,
} from 'three'

import {
  enhanceScientificSurface,
} from '../src/pages/scientificSurfaceSmoothing.ts'

type Fixture = {
  geometry: BufferGeometry
  largeVertices: number
}

function makeFixture(includeSmall: boolean): Fixture {
  const sides = 24
  const rings = 12
  const positions: number[] = [0, 0, 10]
  const faces: number[] = []

  for (let ring = 1; ring < rings; ring++) {
    const theta = Math.PI * ring / rings

    for (let j = 0; j < sides; j++) {
      const phi = 2 * Math.PI * j / sides
      const radius =
        10 +
        0.22 *
        Math.sin(6 * phi + 0.75 * theta) *
        Math.sin(theta) ** 2

      positions.push(
        radius * Math.sin(theta) * Math.cos(phi),
        radius * Math.sin(theta) * Math.sin(phi),
        radius * Math.cos(theta),
      )
    }
  }

  const south = positions.length / 3
  positions.push(0, 0, -10)

  // Consistent outward-facing, closed triangle surface.
  for (let j = 0; j < sides; j++) {
    faces.push(0, 1 + j, 1 + (j + 1) % sides)
  }

  for (let ring = 1; ring < rings - 1; ring++) {
    for (let j = 0; j < sides; j++) {
      const a = 1 + (ring - 1) * sides + j
      const b = 1 + (ring - 1) * sides + (j + 1) % sides
      const c = 1 + ring * sides + j
      const d = 1 + ring * sides + (j + 1) % sides

      faces.push(a, c, b, b, c, d)
    }
  }

  for (let j = 0; j < sides; j++) {
    faces.push(
      south,
      1 + (rings - 2) * sides + (j + 1) % sides,
      1 + (rings - 2) * sides + j,
    )
  }

  const largeVertices = positions.length / 3

  if (includeSmall) {
    const base = largeVertices

    // A detached four-vertex tetrahedron: intentionally
    // below the processor's 64-vertex smoothing threshold.
    positions.push(
      40, 0, 0,
      42, 0, 0,
      41, 2, 0,
      41, 1, 2,
    )

    faces.push(
      base, base + 2, base + 1,
      base, base + 1, base + 3,
      base + 1, base + 2, base + 3,
      base + 2, base, base + 3,
    )
  }

  const geometry = new BufferGeometry()
  geometry.setAttribute(
    'position',
    new Float32BufferAttribute(positions, 3),
  )
  geometry.setIndex(faces)

  return { geometry, largeVertices }
}

const { geometry: source, largeVertices } =
  makeFixture(true)

const originalPositions = Array.from(
  source.getAttribute('position').array,
)
const originalIndices = Array.from(source.getIndex()!.array)

const started = performance.now()
const enhanced = enhanceScientificSurface(source)
const elapsed = performance.now() - started

assert.equal(
  enhanced.status,
  'enhanced',
  'Synthetic closed surface should pass smoothing safeguards',
)

assert.notEqual(enhanced.geometry, source)

assert.deepEqual(
  Array.from(source.getAttribute('position').array),
  originalPositions,
  'Input geometry must remain unchanged',
)

assert.deepEqual(
  Array.from(enhanced.geometry.getIndex()!.array),
  originalIndices,
  'Triangle connectivity must remain unchanged',
)

let maxDisplacement = 0
let changedLargeVertices = 0

const output = enhanced.geometry.getAttribute('position')

for (let i = 0; i < output.count; i++) {
  const offset = i * 3

  const displacement = Math.hypot(
    output.getX(i) - originalPositions[offset],
    output.getY(i) - originalPositions[offset + 1],
    output.getZ(i) - originalPositions[offset + 2],
  )

  assert.ok(Number.isFinite(displacement))

  if (i < largeVertices) {
    if (displacement > 1e-6) changedLargeVertices++
    maxDisplacement = Math.max(maxDisplacement, displacement)
  } else {
    assert.equal(
      displacement,
      0,
      'Small detached components must remain untouched',
    )
  }
}

assert.ok(changedLargeVertices > 0)
assert.ok(maxDisplacement < 1)

// A valid but unindexed mesh cannot pass this particular
// index-based processor; its original geometry is retained.
const unsuitable = new BufferGeometry()
unsuitable.setAttribute(
  'position',
  new Float32BufferAttribute([
    0, 0, 0,
    1, 0, 0,
    0, 1, 0,
  ], 3),
)

const rejected = enhanceScientificSurface(unsuitable)

assert.equal(rejected.status, 'fallback')
assert.notEqual(rejected.geometry, unsuitable)
assert.deepEqual(
  Array.from(rejected.geometry.getAttribute('position').array),
  Array.from(unsuitable.getAttribute('position').array),
)

source.dispose()
enhanced.geometry.dispose()
unsuitable.dispose()
rejected.geometry.dispose()

console.log('PASS: closed synthetic mesh enhancement accepted')
console.log('PASS: original source coordinates unchanged')
console.log('PASS: original triangle connectivity preserved')
console.log('PASS: small disconnected component preserved exactly')
console.log('PASS: unsupported geometry falls back safely')
console.log(`Maximum synthetic displacement: ${maxDisplacement.toFixed(4)} mm`)
console.log(`Processing time: ${elapsed.toFixed(1)} ms`)
