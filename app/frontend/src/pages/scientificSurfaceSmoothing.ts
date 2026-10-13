import {
  BufferGeometry,
  Float32BufferAttribute,
} from 'three'

export type SurfaceEnhancementStatus = 'original' | 'enhanced' | 'fallback'

export type EnhancedSurfaceResult = {
  geometry: BufferGeometry
  status: SurfaceEnhancementStatus
}

const PAIRS = 4
const LAMBDA = 0.33
const MU = -0.34
const MIN_COMPONENT_VERTICES = 64
const MAX_VERTICES = 500_000
const MAX_FACES = 1_000_000

function originalSurface(source: BufferGeometry): BufferGeometry {
  const geometry = source.clone()
  geometry.computeVertexNormals()
  return geometry
}

export function enhanceScientificSurface(
  source: BufferGeometry,
): EnhancedSurfaceResult {
  const fallback = (): EnhancedSurfaceResult => ({
    geometry: originalSurface(source),
    status: 'fallback',
  })

  const position = source.getAttribute('position')
  const index = source.getIndex()

  if (
    !position ||
    position.itemSize !== 3 ||
    !index ||
    index.count % 3 !== 0 ||
    position.count < 4 ||
    position.count > MAX_VERTICES ||
    index.count / 3 > MAX_FACES
  ) {
    return fallback()
  }

  const n = position.count
  const original = new Float64Array(n * 3)

  for (let i = 0; i < n; i++) {
    const x = position.getX(i)
    const y = position.getY(i)
    const z = position.getZ(i)

    if (![x, y, z].every(Number.isFinite)) return fallback()

    original[3 * i] = x
    original[3 * i + 1] = y
    original[3 * i + 2] = z
  }

  const faces = new Uint32Array(index.count)
  const edges = new Set<number>()

  for (let i = 0; i < index.count; i += 3) {
    const a = index.getX(i)
    const b = index.getX(i + 1)
    const c = index.getX(i + 2)

    if (
      ![a, b, c].every(Number.isInteger) ||
      Math.min(a, b, c) < 0 ||
      Math.max(a, b, c) >= n ||
      a === b || b === c || c === a
    ) return fallback()

    faces[i] = a
    faces[i + 1] = b
    faces[i + 2] = c

    for (const [u, v] of [[a, b], [b, c], [c, a]]) {
      const low = Math.min(u, v)
      const high = Math.max(u, v)
      edges.add(low * n + high)
    }
  }

  // Build a deduplicated, compact undirected adjacency graph.
  const degree = new Uint32Array(n)

  for (const encoded of edges) {
    const a = Math.floor(encoded / n)
    const b = encoded % n
    degree[a]++
    degree[b]++
  }

  if (degree.some((value) => value === 0)) return fallback()

  const offsets = new Uint32Array(n + 1)

  for (let i = 0; i < n; i++) {
    offsets[i + 1] = offsets[i] + degree[i]
  }

  const neighbors = new Uint32Array(offsets[n])
  const cursor = offsets.slice(0, n)

  for (const encoded of edges) {
    const a = Math.floor(encoded / n)
    const b = encoded % n
    neighbors[cursor[a]++] = b
    neighbors[cursor[b]++] = a
  }

  // A component is a mesh-graph component, not necessarily
  // a six-connected component in the original voxel mask.
  const component = new Int32Array(n).fill(-1)
  const sizes: number[] = []

  for (let start = 0; start < n; start++) {
    if (component[start] !== -1) continue

    const id = sizes.length
    const stack = [start]
    component[start] = id

    let size = 0

    while (stack.length) {
      const vertex = stack.pop()!
      size++

      for (
        let k = offsets[vertex];
        k < offsets[vertex + 1];
        k++
      ) {
        const adjacent = neighbors[k]

        if (component[adjacent] === -1) {
          component[adjacent] = id
          stack.push(adjacent)
        }
      }
    }

    sizes.push(size)
  }

  const eligible = sizes.map(
    (size) => size >= MIN_COMPONENT_VERTICES
  )

  if (!eligible.some(Boolean)) return fallback()

  // Uniform-neighbor Laplacian, matching the CPU experiment.
  // Skip small components to retain fragile detached details.
  let current = original.slice()

  for (let pair = 0; pair < PAIRS; pair++) {
    for (const strength of [LAMBDA, MU]) {
      const next = new Float64Array(current.length)

      for (let vertex = 0; vertex < n; vertex++) {
        const base = vertex * 3

        if (!eligible[component[vertex]]) {
          next[base] = current[base]
          next[base + 1] = current[base + 1]
          next[base + 2] = current[base + 2]
          continue
        }

        for (let axis = 0; axis < 3; axis++) {
          let sum = 0

          for (
            let k = offsets[vertex];
            k < offsets[vertex + 1];
            k++
          ) {
            sum += current[3 * neighbors[k] + axis]
          }

          const average = sum / degree[vertex]
          const value = current[base + axis]

          next[base + axis] =
            value + strength * (average - value)
        }
      }

      current = next
    }
  }

  // Quality gate in physical RAS+ mm. No original vertex or
  // triangle index is modified by this procedure.
  const displacements = new Float64Array(n)
  let maximum = 0

  for (let i = 0; i < n; i++) {
    const dx = current[3 * i] - original[3 * i]
    const dy = current[3 * i + 1] - original[3 * i + 1]
    const dz = current[3 * i + 2] - original[3 * i + 2]

    const displacement = Math.hypot(dx, dy, dz)

    if (!Number.isFinite(displacement)) return fallback()

    displacements[i] = displacement
    maximum = Math.max(maximum, displacement)
  }

  displacements.sort()

  const p95 = displacements[
    Math.floor(0.95 * (n - 1))
  ]

  if (maximum >= 1 || p95 >= 0.5) return fallback()

  // Component-relative origins improve enclosed-volume
  // numerical stability for large physical RAS coordinates.
  const centers = new Float64Array(sizes.length * 3)

  for (let i = 0; i < n; i++) {
    for (let axis = 0; axis < 3; axis++) {
      centers[3 * component[i] + axis] += original[3 * i + axis]
    }
  }

  for (let id = 0; id < sizes.length; id++) {
    for (let axis = 0; axis < 3; axis++) {
      centers[3 * id + axis] /= sizes[id]
    }
  }

  const volumesOriginal = new Float64Array(sizes.length)
  const volumesEnhanced = new Float64Array(sizes.length)

  let areaOriginal = 0
  let areaEnhanced = 0

  for (let f = 0; f < faces.length; f += 3) {
    const a = faces[f]
    const b = faces[f + 1]
    const c = faces[f + 2]

    const faceComponent = component[a]

    if (
      component[b] !== faceComponent ||
      component[c] !== faceComponent
    ) return fallback()

    const measure = (vertices: Float64Array) => {
      const ax = vertices[3 * a]
      const ay = vertices[3 * a + 1]
      const az = vertices[3 * a + 2]

      const bx = vertices[3 * b]
      const by = vertices[3 * b + 1]
      const bz = vertices[3 * b + 2]

      const cx = vertices[3 * c]
      const cy = vertices[3 * c + 1]
      const cz = vertices[3 * c + 2]

      const ux = bx - ax
      const uy = by - ay
      const uz = bz - az

      const vx = cx - ax
      const vy = cy - ay
      const vz = cz - az

      const nx = uy * vz - uz * vy
      const ny = uz * vx - ux * vz
      const nz = ux * vy - uy * vx

      const ox = centers[3 * faceComponent]
      const oy = centers[3 * faceComponent + 1]
      const oz = centers[3 * faceComponent + 2]

      return {
        nx, ny, nz,
        area: Math.hypot(nx, ny, nz) * 0.5,
        volume: (
          (ax - ox) * nx +
          (ay - oy) * ny +
          (az - oz) * nz
        ) / 6,
      }
    }

    const before = measure(original)
    const after = measure(current)

    if (
      before.area < 1e-8 ||
      after.area < 1e-8 ||
      before.nx * after.nx +
      before.ny * after.ny +
      before.nz * after.nz <= 0
    ) return fallback()

    areaOriginal += before.area
    areaEnhanced += after.area

    volumesOriginal[faceComponent] += before.volume
    volumesEnhanced[faceComponent] += after.volume
  }

  const originalVolume = Array.from(
    volumesOriginal, Math.abs
  ).reduce((a, b) => a + b, 0)

  const enhancedVolume = Array.from(
    volumesEnhanced, Math.abs
  ).reduce((a, b) => a + b, 0)

  if (originalVolume <= 1e-8 || areaOriginal <= 1e-8) {
    return fallback()
  }

  if (
    Math.abs(enhancedVolume / originalVolume - 1) >= 0.01 ||
    Math.abs(areaEnhanced / areaOriginal - 1) >= 0.07
  ) return fallback()

  // Additional check for each independently smoothed region.
  for (let id = 0; id < sizes.length; id++) {
    if (!eligible[id]) continue

    const v0 = Math.abs(volumesOriginal[id])
    const v1 = Math.abs(volumesEnhanced[id])

    if (
      v0 <= 1e-8 ||
      Math.abs(v1 / v0 - 1) >= 0.05
    ) return fallback()
  }

  const geometry = source.clone()
  geometry.setAttribute(
    'position',
    new Float32BufferAttribute(current, 3),
  )
  geometry.computeVertexNormals()

  return { geometry, status: 'enhanced' }
}
