import { useEffect, useMemo, useState } from 'react'
import { useThree } from '@react-three/fiber'
import {
  BufferGeometry,
  DoubleSide,
  Float32BufferAttribute,
  LinearFilter,
  SRGBColorSpace,
  type Texture,
  TextureLoader,
} from 'three'

import {
  viewerSliceUrl,
  type MRIPlane,
  type MRIModality,
  type MRIViewerMetadata,
} from '../api/viewer'

type Props = {
  caseId: string
  modality: MRIModality
  plane: MRIPlane
  index: number
  metadata: MRIViewerMetadata
  opacity: number
}

type LoadedTexture = {
  url: string
  texture: Texture
}

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

function makeSliceGeometry(
  metadata: MRIViewerMetadata,
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

export function MRIPlaneTexture3D({
  caseId,
  modality,
  plane,
  index,
  metadata,
  opacity,
}: Props) {
  const invalidate = useThree((state) => state.invalidate)

  const [loaded, setLoaded] = useState<LoadedTexture | null>(null)
  const [failedUrl, setFailedUrl] = useState<string | null>(null)

  const url = viewerSliceUrl(caseId, modality, plane, index)

  const geometry = useMemo(
    () => makeSliceGeometry(metadata, plane, index),
    [metadata, plane, index],
  )

  useEffect(() => {
    return () => geometry.dispose()
  }, [geometry])

  useEffect(() => {
    let active = true

    const texture = new TextureLoader().load(
      url,
      (result) => {
        if (!active) return

        result.colorSpace = SRGBColorSpace
        result.minFilter = LinearFilter
        result.magFilter = LinearFilter
        result.generateMipmaps = false
        result.needsUpdate = true

        setLoaded({ url, texture: result })
        setFailedUrl(null)
        invalidate()
      },
      undefined,
      () => {
        if (active) {
          setFailedUrl(url)
          invalidate()
        }
      },
    )

    return () => {
      active = false
      texture.dispose()
    }
  }, [url, invalidate])

  const currentTexture = loaded?.url === url
    ? loaded.texture
    : null

  if (failedUrl === url || !currentTexture) {
    return null
  }

  return (
    <mesh geometry={geometry}>
      <meshBasicMaterial
        map={currentTexture}
        transparent
        opacity={opacity}
        side={DoubleSide}
        depthWrite={false}
        toneMapped={false}
      />
    </mesh>
  )
}
