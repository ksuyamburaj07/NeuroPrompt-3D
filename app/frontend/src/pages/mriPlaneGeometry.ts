import { BufferGeometry, Float32BufferAttribute } from 'three'
import type { MRIPlane, MRIViewerMetadata } from '../api/viewer'

function worldPoint(
  affine: number[][],
  voxel: [number, number, number],
): number[] {
  const [x, y, z] = voxel

  return [0, 1, 2].map(
    (row) =>
      affine[row][0] * x +
      affine[row][1] * y +
      affine[row][2] * z +
      affine[row][3],
  )
}

export function makeSliceGeometry(
  metadata: Pick<MRIViewerMetadata, 'shape_ras_xyz' | 'affine_ras'>,
  plane: MRIPlane,
  index: number,
): BufferGeometry {
  const [nx, ny, nz] = metadata.shape_ras_xyz

  const width = plane === 'sagittal' ? ny : nx
  const height = plane === 'axial' ? ny : nz

  // PNG bottom-left, bottom-right, top-right, top-left.
  // Edges are half a voxel beyond the outermost pixel centres.
  const corners: [number, number][] = [
    [-0.5, -0.5],
    [width - 0.5, -0.5],
    [width - 0.5, height - 0.5],
    [-0.5, height - 0.5],
  ]

  const positions = corners.flatMap(([u, v]) => {
    let voxel: [number, number, number]

    if (plane === 'axial') {
      voxel = [u, v, index]
    } else if (plane === 'coronal') {
      voxel = [u, index, v]
    } else {
      voxel = [index, u, v]
    }

    return worldPoint(metadata.affine_ras, voxel)
  })

  const geometry = new BufferGeometry()

  geometry.setAttribute(
    'position',
    new Float32BufferAttribute(positions, 3),
  )

  // Three.js TextureLoader uses flipY=true by default.
  // Thus the first PNG row maps to the upper edge of the plane.
  geometry.setAttribute(
    'uv',
    new Float32BufferAttribute([
      0, 0,
      1, 0,
      1, 1,
      0, 1,
    ], 2),
  )

  geometry.setIndex([0, 1, 2, 0, 2, 3])
  geometry.computeVertexNormals()

  return geometry
}
