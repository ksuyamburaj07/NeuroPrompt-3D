import {
  Component,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ComponentRef,
  type ReactNode,
} from 'react'

import { Canvas, useLoader } from '@react-three/fiber'
import { Html, OrbitControls } from '@react-three/drei'
import { DoubleSide } from 'three'
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js'

import {
  viewerMeshUrl,
  viewerAnatomyUrl,
  type MRIPlane,
  type MRIModality,
  type MRIViewerMetadata,
} from '../api/viewer'

import { MRIPlaneTexture3D } from './MRIPlaneTexture3D'
import './MeshViewer3D.css'

type SliceIndices = Record<MRIPlane, number>

type Props = {
  caseId: string
  runId: string
  modality: MRIModality
  metadata: MRIViewerMetadata
  indices: SliceIndices
}

class MeshErrorBoundary extends Component<
  { children: ReactNode },
  { error: string | null }
> {
  state = { error: null as string | null }

  static getDerivedStateFromError(error: unknown) {
    return {
      error: error instanceof Error
        ? error.message
        : 'Unknown 3D rendering error.',
    }
  }

  render() {
    if (this.state.error) {
      return (
        <div className="mesh3d__error" role="alert">
          Unable to display the 3D segmentation.
          <small>{this.state.error}</small>
        </div>
      )
    }

    return this.props.children
  }
}

function SegmentationSurface({
  url,
  opacity,
}: {
  url: string
  opacity: number
}) {
  const loaded = useLoader(PLYLoader, url)

  const geometry = useMemo(() => {
    const result = loaded.clone()
    result.computeVertexNormals()
    return result
  }, [loaded])

  useEffect(() => {
    return () => geometry.dispose()
  }, [geometry])

  return (
    <mesh geometry={geometry} renderOrder={1}>
      <meshStandardMaterial
        color="#58e0ad"
        roughness={0.67}
        metalness={0.08}
        transparent
        opacity={opacity}
        depthWrite={false}
        side={DoubleSide}
      />
    </mesh>
  )
}


function AnatomySurface({
  url,
  opacity,
}: {
  url: string
  opacity: number
}) {
  const loaded = useLoader(PLYLoader, url)

  const geometry = useMemo(() => {
    const result = loaded.clone()
    result.computeVertexNormals()
    return result
  }, [loaded])

  useEffect(() => {
    return () => geometry.dispose()
  }, [geometry])

  return (
    <mesh geometry={geometry} renderOrder={2}>
      <meshStandardMaterial
        color="#e5b8ad"
        roughness={0.94}
        metalness={0}
        transparent={opacity < 1}
        opacity={opacity}
        depthWrite={opacity >= 1}
        side={DoubleSide}
      />
    </mesh>
  )
}

function mriWorldCentre(
  metadata: MRIViewerMetadata,
): [number, number, number] {
  const voxelCentre = metadata.shape_ras_xyz.map(
    (size) => (size - 1) / 2,
  )

  return [0, 1, 2].map(
    (row) =>
      metadata.affine_ras[row][0] * voxelCentre[0] +
      metadata.affine_ras[row][1] * voxelCentre[1] +
      metadata.affine_ras[row][2] * voxelCentre[2] +
      metadata.affine_ras[row][3],
  ) as [number, number, number]
}

export function MeshViewer3D({
  caseId,
  runId,
  modality,
  metadata,
  indices,
}: Props) {
  const controlsRef =
    useRef<ComponentRef<typeof OrbitControls>>(null)

  const [plane, setPlane] =
    useState<MRIPlane | 'none'>('none')

  const [planeOpacity, setPlaneOpacity] = useState(0.65)

  const [showAnatomy, setShowAnatomy] = useState(true)
  const [anatomyOpacity, setAnatomyOpacity] = useState(0.30)
  const [anatomyStep, setAnatomyStep] = useState<1 | 2>(2)

  const [showSegmentation, setShowSegmentation] = useState(true)
  const [segmentationOpacity, setSegmentationOpacity] =
    useState(0.85)

  const centre = useMemo(
    () => mriWorldCentre(metadata),
    [metadata],
  )

  const meshUrl = viewerMeshUrl(caseId, runId, 'final')
  const anatomyUrl = viewerAnatomyUrl(caseId, anatomyStep)

  return (
    <section
      className="mesh3d"
      aria-label="Interactive 3D MRI and segmentation reconstruction"
    >
      <div className="mesh3d__header">
        <div>
          <span className="mesh3d__eyebrow">
            M11E4 / ANATOMICAL RECONSTRUCTION
          </span>
          <h3>3D Segmentation Explorer</h3>
          <p>
            MRI-derived anatomical context + frozen final-mask surface
          </p>
        </div>

        <button
          type="button"
          className="mesh3d__reset"
          onClick={() => controlsRef.current?.reset()}
        >
          Reset camera
        </button>
      </div>



      <div
        className="mesh3d__view-presets"
        aria-label="3D inspection presets"
      >
        <button
          type="button"
          onClick={() => {
            setShowAnatomy(true)
            setAnatomyOpacity(1)
            setAnatomyStep(1)
            setShowSegmentation(false)
            setPlane('none')
          }}
        >
          Inspect brain surface
        </button>

        <button
          type="button"
          onClick={() => {
            setShowAnatomy(true)
            setAnatomyOpacity(0.25)
            setAnatomyStep(2)
            setShowSegmentation(true)
            setSegmentationOpacity(0.85)
            setPlane('none')
          }}
        >
          Inspect prediction
        </button>
      </div>

      <div
        className="mesh3d__layer-controls"
        aria-label="3D scientific layer controls"
      >
        <div className="mesh3d__layer-control">
          <label className="mesh3d__layer-toggle">
            <input
              type="checkbox"
              checked={showAnatomy}
              onChange={(event) => {
                setShowAnatomy(event.target.checked)
              }}
            />
            Brain context
          </label>

          <label>
            Opacity {Math.round(anatomyOpacity * 100)}%
            <input
              type="range"
              min={5}
              max={100}
              step={5}
              value={Math.round(anatomyOpacity * 100)}
              disabled={!showAnatomy}
              onChange={(event) => {
                setAnatomyOpacity(Number(event.target.value) / 100)
              }}
            />
          </label>
        </div>

        <div className="mesh3d__layer-control">
          <label className="mesh3d__layer-toggle">
            <input
              type="checkbox"
              checked={showSegmentation}
              onChange={(event) => {
                setShowSegmentation(event.target.checked)
              }}
            />
            Final prediction
          </label>

          <label>
            Opacity {Math.round(segmentationOpacity * 100)}%
            <input
              type="range"
              min={5}
              max={100}
              step={5}
              value={Math.round(segmentationOpacity * 100)}
              disabled={!showSegmentation}
              onChange={(event) => {
                setSegmentationOpacity(
                  Number(event.target.value) / 100,
                )
              }}
            />
          </label>
        </div>
      </div>

      <div className="mesh3d__detail-control">
        <label htmlFor="mesh-anatomy-detail">
          Anatomical surface detail
        </label>
        <select
          id="mesh-anatomy-detail"
          value={anatomyStep}
          disabled={!showAnatomy}
          onChange={(event) => {
            setAnatomyStep(
              Number(event.target.value) as 1 | 2,
            )
          }}
        >
          <option value={2}>Balanced · 2-voxel step</option>
          <option value={1}>Detailed · 1-voxel step</option>
        </select>
        <span>Visualization geometry only</span>
      </div>

      <p className="mesh3d__context-caption">
        Brain context: nonzero T1N foreground, selectable
        surface sampling. Prediction: unchanged frozen final mask.
      </p>

      <div className="mesh3d__context-controls">
        <label>
          MRI context
          <select
            value={plane}
            onChange={(event) => {
              setPlane(event.target.value as MRIPlane | 'none')
            }}
          >
            <option value="axial">Axial</option>
            <option value="coronal">Coronal</option>
            <option value="sagittal">Sagittal</option>
            <option value="none">No MRI slice</option>
          </select>
        </label>

        <label>
          Plane opacity: {Math.round(planeOpacity * 100)}%
          <input
            type="range"
            min="10"
            max="100"
            step="5"
            value={Math.round(planeOpacity * 100)}
            disabled={plane === 'none'}
            onChange={(event) => {
              setPlaneOpacity(Number(event.target.value) / 100)
            }}
          />
        </label>
      </div>

      <div className="mesh3d__canvas">
        <MeshErrorBoundary key={`${meshUrl}:${anatomyUrl}`}>
          <Canvas
            frameloop="demand"
            dpr={[1, 1.5]}
            gl={{
              antialias: true,
              alpha: false,
              powerPreference: 'low-power',
            }}
            camera={{
              position: [128, -157, 125],
              fov: 43,
              near: 0.1,
              far: 1000,
            }}
            onCreated={({ camera }) => {
              camera.up.set(0, 0, 1)
              camera.lookAt(0, 0, 0)
            }}
          >
            <color attach="background" args={['#0b1420']} />

            <ambientLight intensity={1.3} />

            <directionalLight
              position={[100, -80, 160]}
              intensity={2.5}
            />

            <directionalLight
              position={[-100, 120, -50]}
              intensity={0.7}
            />

            <Suspense
              fallback={
                <Html center>
                  <span className="mesh3d__loading">
                    Loading scientific segmentation…
                  </span>
                </Html>
              }
            >
              <group
                position={[-centre[0], -centre[1], -centre[2]]}
              >
                {showSegmentation && (
                  <SegmentationSurface
                    url={meshUrl}
                    opacity={segmentationOpacity}
                  />
                )}

                {showAnatomy && (
                  <AnatomySurface
                    url={anatomyUrl}
                    opacity={anatomyOpacity}
                  />
                )}

                {plane !== 'none' && (
                  <MRIPlaneTexture3D
                    caseId={caseId}
                    modality={modality}
                    plane={plane}
                    index={indices[plane]}
                    metadata={metadata}
                    opacity={planeOpacity}
                  />
                )}
              </group>
            </Suspense>

            <OrbitControls
              ref={controlsRef}
              makeDefault
              enableDamping
              dampingFactor={0.08}
              enablePan={false}
              minDistance={70}
              maxDistance={600}
              target={[0, 0, 0]}
            />
          </Canvas>
        </MeshErrorBoundary>
      </div>

      <div className="mesh3d__footer">
        <span>Drag to rotate · Scroll to zoom</span>
        <span>
          MRI: {modality.toUpperCase()} ·
          {plane === 'none'
            ? ' No slice'
            : ` ${plane} ${indices[plane]}`}
        </span>
        <span>RAS+ millimetres · Research only</span>
      </div>
    </section>
  )
}
