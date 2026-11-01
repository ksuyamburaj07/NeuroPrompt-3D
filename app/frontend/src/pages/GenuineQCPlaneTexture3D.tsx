import { useEffect, useMemo, useState } from 'react'
import { Html } from '@react-three/drei'
import { useThree } from '@react-three/fiber'
import {
  DoubleSide,
  LinearFilter,
  SRGBColorSpace,
  TextureLoader,
  type Texture,
} from 'three'

import { makeSliceGeometry } from './mriPlaneGeometry'
import type {
  QCMetadata,
  QCPlane,
} from './CorticalQCSlicePane'

type Props = {
  metadata: QCMetadata
  plane: QCPlane
  index: number
  opacity: number
}

type Loaded = {
  url: string
  texture: Texture
}

const qcBase = (
  import.meta.env.VITE_QC_MRI_BASE_URL
  || (import.meta.env.DEV ? 'http://127.0.0.1:8013' : '/qc/mri')
)

export function GenuineQCPlaneTexture3D({
  metadata,
  plane,
  index,
  opacity,
}: Props) {
  const invalidate = useThree((state) => state.invalidate)

  // Avoid initiating an image request for every trackpad/slider event.
  const [displayIndex, setDisplayIndex] = useState(index)

  useEffect(() => {
    const timer = window.setTimeout(
      () => setDisplayIndex(index),
      120,
    )
    return () => window.clearTimeout(timer)
  }, [index])

  const [loaded, setLoaded] = useState<Loaded | null>(null)
  const [failedUrl, setFailedUrl] = useState<string | null>(null)

  const url = `${qcBase}/slices3d/${plane}/${displayIndex}`

  const geometry = useMemo(
    () => makeSliceGeometry(metadata, plane, displayIndex),
    [metadata, plane, displayIndex],
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

  const currentTexture =
    loaded?.url === url ? loaded.texture : null

  if (failedUrl === url) {
    return (
      <Html center>
        <div role="alert" style={{
          padding: 12,
          background: '#271b2b',
          color: '#ffd2d2',
          fontSize: 12,
        }}>
          Original 3D MRI slice unavailable.
          Toggle the plane off and on to retry.
        </div>
      </Html>
    )
  }

  if (!currentTexture) return null

  return (
    <mesh geometry={geometry} name="genuine-qc-mri-plane">
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
