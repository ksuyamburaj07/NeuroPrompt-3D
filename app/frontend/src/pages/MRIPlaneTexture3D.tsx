import { useEffect, useMemo, useState } from 'react'
import { useThree } from '@react-three/fiber'
import {
  DoubleSide,
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

import { makeSliceGeometry } from './mriPlaneGeometry'

type Props = {
  caseId: string
  modality: MRIModality
  plane: MRIPlane
  index: number
  metadata: MRIViewerMetadata
  opacity: number
  transparentBackground: boolean
}

type LoadedTexture = {
  url: string
  texture: Texture
}

export function MRIPlaneTexture3D({
  caseId,
  modality,
  plane,
  index,
  metadata,
  opacity,
  transparentBackground,
}: Props) {
  const invalidate = useThree((state) => state.invalidate)

  const [loaded, setLoaded] = useState<LoadedTexture | null>(null)
  const [failedUrl, setFailedUrl] = useState<string | null>(null)

  const baseUrl = viewerSliceUrl(
    caseId, modality, plane, index,
  )
  const url = transparentBackground
    ? `${baseUrl}${baseUrl.includes('?') ? '&' : '?'}transparent_background=true`
    : baseUrl

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
