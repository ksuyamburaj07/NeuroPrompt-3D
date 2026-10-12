import { useEffect, useState } from 'react'
import { DoubleSide, type BufferGeometry } from 'three'
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js'

export type RemovalLoadState =
  | 'idle'
  | 'loading'
  | 'ready'
  | 'unavailable'

type Props = {
  url: string
  onLoadState: (state: RemovalLoadState) => void
}

export function RemovedRefinementSurface({
  url,
  onLoadState,
}: Props) {
  const [geometry, setGeometry] =
    useState<BufferGeometry | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    let active = true
    let ownedGeometry: BufferGeometry | null = null

    onLoadState('loading')

    async function load() {
      try {
        const response = await fetch(url, {
          cache: 'no-store',
          signal: controller.signal,
        })

        if (!response.ok) {
          throw new Error(
            `Removal surface unavailable: HTTP ${response.status}`
          )
        }

        const buffer = await response.arrayBuffer()

        // Browser resource guard; never process an
        // unexpectedly large removal mesh on the Dell.
        if (buffer.byteLength > 32 * 1024 * 1024) {
          throw new Error('Removal mesh exceeds display limit.')
        }

        if (!active) return

        const parsed = new PLYLoader().parse(buffer)

        if (parsed.getAttribute('position').count === 0) {
          parsed.dispose()
          throw new Error('Removal mesh contains no vertices.')
        }

        parsed.computeVertexNormals()

        if (!active) {
          parsed.dispose()
          return
        }

        ownedGeometry = parsed
        setGeometry(parsed)
        onLoadState('ready')
      } catch {
        if (active) {
          onLoadState('unavailable')
        }
      }
    }

    void load()

    return () => {
      active = false
      controller.abort()
      ownedGeometry?.dispose()
    }
  }, [url, onLoadState])

  if (!geometry) return null

  return (
    <mesh geometry={geometry} renderOrder={1}>
      <meshStandardMaterial
        color="#efb44b"
        roughness={0.64}
        metalness={0.04}
        emissive="#382506"
        emissiveIntensity={0.10}
        transparent={false}
        opacity={1}
        depthWrite
        depthTest
        side={DoubleSide}
      />
    </mesh>
  )
}
