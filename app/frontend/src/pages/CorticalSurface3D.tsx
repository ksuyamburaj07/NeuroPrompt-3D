import { useEffect, useMemo } from 'react'
import { useLoader } from '@react-three/fiber'
import { DoubleSide } from 'three'
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js'

type Props = {
  url: string
  hemisphere: 'lh' | 'rh'
  opacity: number
  useSulcalColors?: boolean
}

export function CorticalSurface3D({
  url,
  hemisphere,
  opacity,
  useSulcalColors = false,
}: Props) {
  // Load the verified PLY delivered by our backend.
  const loaded = useLoader(PLYLoader, url)

  // Keep source geometry unchanged.
  // Normals are computed for display lighting only.
  const geometry = useMemo(() => {
    const result = loaded.clone()
    result.computeVertexNormals()
    return result
  }, [loaded])

  // Dispose only the cloned geometry.
  useEffect(() => {
    return () => geometry.dispose()
  }, [geometry])

  const hemisphereColor =
    hemisphere === 'lh' ? '#D6A23F' : '#37C4E0'

  const hasVertexColors =
    geometry.getAttribute('color') !== undefined

  if (useSulcalColors && !hasVertexColors) {
    throw new Error(
      'Experimental sulcal PLY is missing vertex colors',
    )
  }

  return (
    <mesh
      name={`cortical-${hemisphere}`}
      geometry={geometry}
      renderOrder={0}
    >
      <meshStandardMaterial
        key={useSulcalColors ? 'sulcal' : 'uniform'}
        color={useSulcalColors ? '#c7a8b6' : hemisphereColor}
        vertexColors={useSulcalColors}
        roughness={0.86}
        metalness={0}
        transparent={opacity < 1}
        opacity={opacity}
        depthWrite={opacity >= 1}
        side={DoubleSide}
      />
    </mesh>
  )
}
