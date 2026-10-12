import {
  Component,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'

import { Canvas, useLoader } from '@react-three/fiber'
import { Html } from '@react-three/drei'
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
import { AnatomicalCameraControls, type AnatomicalCameraHandle } from './AnatomicalCameraControls'
import {
  ANATOMICAL_LABELS,
  ANATOMICAL_VIEWS,
  mriCameraFrame,
  type CameraFitMode,
  type CameraViewState,
} from './anatomicalCamera'
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

export function MeshViewer3D({
  caseId,
  runId,
  modality,
  metadata,
  indices,
}: Props) {
  const cameraRef = useRef<AnatomicalCameraHandle>(null)
  const [cameraView, setCameraView] = useState<CameraViewState>({
    view: 'home', moving: false,
  })

  const [plane, setPlane] =
    useState<MRIPlane | 'none'>('none')
  const [fitMode, setFitMode] =
    useState<CameraFitMode>('volume')
  const [transparentMRIBackground, setTransparentMRIBackground] =
    useState(false)

  const [planeOpacity, setPlaneOpacity] = useState(0.65)

  const [showAnatomy, setShowAnatomy] = useState(true)
  const [anatomyOpacity, setAnatomyOpacity] = useState(0.30)
  const [anatomyStep, setAnatomyStep] = useState<1 | 2>(2)
  const [anatomyFinish, setAnatomyFinish] =
    useState<'raw' | 'soft'>('raw')

  const [showSegmentation, setShowSegmentation] = useState(true)
  const [segmentationOpacity, setSegmentationOpacity] =
    useState(0.85)

  // Full MRI planes always use full-volume framing.
  const effectiveFit: CameraFitMode =
    plane === 'none' && metadata.foreground_bounds_ras_xyz
      ? fitMode
      : 'volume'

  const cameraFrame = useMemo(
    () => mriCameraFrame(metadata, effectiveFit),
    [metadata, effectiveFit],
  )
  const centre = cameraFrame.centre
  const activeView = ANATOMICAL_VIEWS.find((view) => view.id === cameraView.view)
  const cameraStatus = cameraView.moving
    ? `Moving to ${activeView?.label.toLowerCase() ?? 'overview'} view…`
    : activeView
      ? `${activeView.label} view · from the patient's ${activeView.label.toLowerCase()}`
      : cameraView.view === 'home' ? 'Overview · Superior is up' : 'Free orbit'

  const meshUrl = viewerMeshUrl(caseId, runId, 'final')
  const anatomyUrl = viewerAnatomyUrl(
    caseId,
    anatomyStep,
    anatomyFinish,
  )

  return (
    <section
      className="mesh3d"
      aria-label="Interactive 3D MRI and segmentation reconstruction"
    >
      <div className="mesh3d__header">
        <div>
          <span className="mesh3d__eyebrow">
            M11E5A / ANATOMICAL NAVIGATION
          </span>
          <h3>3D Segmentation Explorer</h3>
          <p>
            MRI-derived anatomical context + frozen final-mask surface
          </p>
        </div>

        <button
          type="button"
          className="mesh3d__reset"
          onClick={() => cameraRef.current?.navigate('home')}
        >
          Reset camera
        </button>
      </div>

      <div className="mesh3d__camera-controls" role="group" aria-label="Anatomical camera views">
        <span>View from patient’s</span>
        {ANATOMICAL_VIEWS.map((view) => (
          <button
            key={view.id}
            type="button"
            aria-pressed={cameraView.view === view.id}
            onClick={() => cameraRef.current?.navigate(view.id)}
          >
            {view.label} <small>{view.code}</small>
          </button>
        ))}
      </div>
      <p className="mesh3d__camera-caption">
        Side views: Superior up · Top and bottom views: Anterior up.
        {' '}L/R always refer to the patient.
      </p>

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
            setAnatomyFinish('raw')
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
            setAnatomyFinish('raw')
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
            const nextStep =
              Number(event.target.value) as 1 | 2

            setAnatomyStep(nextStep)

            if (nextStep === 2) {
              setAnatomyFinish('raw')
            }
          }}
        >
          <option value={2}>Balanced · 2-voxel step</option>
          <option value={1}>Detailed · 1-voxel step</option>
        </select>
        <label htmlFor="mesh-anatomy-finish">
          Surface treatment
        </label>
        <select
          id="mesh-anatomy-finish"
          value={anatomyFinish}
          disabled={!showAnatomy}
          onChange={(event) => {
            const finish =
              event.target.value as 'raw' | 'soft'

            setAnatomyFinish(finish)

            if (finish === 'soft') {
              setAnatomyStep(1)
            }
          }}
        >
          <option value="raw">Original boundary</option>
          <option value="soft">
            Gentle smoothing · σ = 0.85
          </option>
        </select>
        <span>
          Display geometry only · Scientific masks unchanged
        </span>
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

      <div
        className="mesh3d__background-study"
        role="group"
        aria-label="3D MRI plane background"
      >
        <span>MRI plane presentation</span>
        <button
          type="button"
          aria-pressed={!transparentMRIBackground}
          onClick={() => setTransparentMRIBackground(false)}
        >
          Original rectangle
        </button>
        <button
          type="button"
          aria-pressed={transparentMRIBackground}
          onClick={() => setTransparentMRIBackground(true)}
        >
          Foreground only
        </button>
        <small>3D display only · MRI values preserved</small>
      </div>

      <div
        className="mesh3d__fit-controls"
        role="group"
        aria-label="3D anatomical framing"
      >
        <span>Camera framing</span>
        <button
          type="button"
          aria-pressed={effectiveFit === 'volume'}
          onClick={() => setFitMode('volume')}
        >
          Fit volume
        </button>
        <button
          type="button"
          aria-pressed={effectiveFit === 'anatomy'}
          disabled={
            plane !== 'none' ||
            !metadata.foreground_bounds_ras_xyz
          }
          onClick={() => setFitMode('anatomy')}
        >
          Focus anatomy
        </button>
        <small>
          {plane !== 'none'
            ? 'Full-volume framing while MRI slice is displayed'
            : 'Foreground bounds from T1N · Display-only framing'}
        </small>
      </div>

      <div className="mesh3d__canvas">
        <MeshErrorBoundary key={meshUrl}>
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
            onCreated={({ camera }) => camera.up.set(0, 0, 1)}
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
                    transparentBackground={transparentMRIBackground}
                  />
                )}
              </group>
            </Suspense>

            <AnatomicalCameraControls
              ref={cameraRef}
              frame={cameraFrame}
              onViewChange={setCameraView}
            />
          </Canvas>
        </MeshErrorBoundary>
        <div className="mesh3d__camera-status" role="status" aria-live="polite">
          {cameraStatus}
        </div>
        {activeView && !cameraView.moving && (
          <div className="mesh3d__orientation" aria-label={`Screen orientation: top ${ANATOMICAL_LABELS[activeView.edges[0]]}, bottom ${ANATOMICAL_LABELS[activeView.edges[1]]}, left ${ANATOMICAL_LABELS[activeView.edges[2]]}, right ${ANATOMICAL_LABELS[activeView.edges[3]]}`}>
            {(['top', 'bottom', 'left', 'right'] as const).map((edge, index) => (
              <span
                key={edge}
                className={`mesh3d__orientation-${edge}`}
                title={ANATOMICAL_LABELS[activeView.edges[index]]}
              >
                {activeView.edges[index]}
              </span>
            ))}
          </div>
        )}
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
