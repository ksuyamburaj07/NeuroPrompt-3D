import type { ViewerRunMarkers } from '../api/viewer'
import {
  RemovedRefinementSurface,
  type RemovalLoadState,
} from './RemovedRefinementSurface'
import {
  Component,
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'

import { createPortal } from 'react-dom'
import { Canvas, useLoader } from '@react-three/fiber'
import { Html } from '@react-three/drei'
import { DoubleSide } from 'three'
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js'
import { CorticalSurface3D } from './CorticalSurface3D'

import {
  viewerMeshUrl,
  viewerAnatomyUrl,
  getViewerCortexStatus,
  viewerCortexMeshUrl,
  type CortexStatus,
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
  mriPlaneForView,
  type CameraFitMode,
  type CameraViewState,
} from './anatomicalCamera'
import {
  enhanceScientificSurface,
  type SurfaceEnhancementStatus,
} from './scientificSurfaceSmoothing'
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
  layer,
  emphasis,
  treatment,
  onSurfaceStatus,
}: {
  url: string
  opacity: number
  layer: 'final' | 'baseline'
  emphasis: boolean
  treatment: 'original' | 'enhanced'
  onSurfaceStatus: (status: SurfaceEnhancementStatus) => void
}) {
  const loaded = useLoader(PLYLoader, url)

  const surface = useMemo(() => {
    if (treatment === 'enhanced') {
      return enhanceScientificSurface(loaded)
    }

    const geometry = loaded.clone()
    geometry.computeVertexNormals()

    return {
      geometry,
      status: 'original' as const,
    }
  }, [loaded, treatment])

  useEffect(() => {
    onSurfaceStatus(surface.status)
  }, [surface.status, onSurfaceStatus])

  useEffect(() => {
    return () => surface.geometry.dispose()
  }, [surface.geometry])

  const geometry = surface.geometry

  return (
    <mesh geometry={geometry} renderOrder={1}>
      <meshStandardMaterial
        color={
          layer === 'baseline'
            ? '#ed8f99'
            : emphasis ? '#75f2c0' : '#58e0ad'
        }
        roughness={
          treatment === 'enhanced'
            ? 0.79
            : emphasis ? 0.5 : 0.67
        }
        metalness={
          treatment === 'enhanced'
            ? 0
            : emphasis ? 0.04 : 0.08
        }
        emissive={
          layer === 'final' && emphasis
            ? '#124b36'
            : '#000000'
        }
        emissiveIntensity={
          layer === 'final' && emphasis
            ? treatment === 'enhanced' ? 0.06 : 0.16
            : 0
        }
        transparent={opacity < 1}
        opacity={opacity}
        depthWrite={opacity >= 1}
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

class CortexLoadBoundary extends Component<
  { children: ReactNode; onError: () => void },
  { failed: boolean }
> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  componentDidCatch() {
    this.props.onError()
  }

  render() {
    return this.state.failed ? null : this.props.children
  }
}

export function MeshViewer3D({
  caseId,
  runId,
  modality,
  metadata,
  indices,
  markers,
  controlsContainer,
}: Props & {
  markers?: ViewerRunMarkers | null
  controlsContainer?: HTMLElement | null
}) {
  const cameraRef = useRef<AnatomicalCameraHandle>(null)
  const [cameraView, setCameraView] = useState<CameraViewState>({
    view: 'home', moving: false,
  })

  const [plane, setPlane] =
    useState<MRIPlane | 'none'>('none')
  const [fitMode, setFitMode] =
    useState<CameraFitMode>('anatomy')
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
  const [predictionLayer, setPredictionLayer] =
    useState<'final' | 'baseline'>('final')
  const [predictionEmphasis, setPredictionEmphasis] =
    useState(false)
  const [surfaceTreatment, setSurfaceTreatment] =
    useState<'original' | 'enhanced'>('original')
  const [surfaceStatus, setSurfaceStatus] =
    useState<SurfaceEnhancementStatus>('original')

  const [showRemoved, setShowRemoved] = useState(false)
  const [removalStatus, setRemovalStatus] =
    useState<RemovalLoadState>('idle')

  const [cortexRequestedFor, setCortexRequestedFor] =
    useState<string | null>(null)

  const [cortexFailedFor, setCortexFailedFor] =
    useState<string | null>(null)

  const [showLeftCortex, setShowLeftCortex] = useState(true)
  const [showRightCortex, setShowRightCortex] = useState(true)
  const [cortexOpacity, setCortexOpacity] = useState(0.80)

  const [cortexLookup, setCortexLookup] = useState<{
    caseId: string
    status: CortexStatus | null
    error: boolean
  } | null>(null)

  useEffect(() => {
    let active = true

    getViewerCortexStatus(caseId).then(
      (status) => {
        if (active) {
          setCortexLookup({
            caseId,
            status,
            error: false,
          })
        }
      },
      () => {
        if (active) {
          setCortexLookup({
            caseId,
            status: null,
            error: true,
          })
        }
      },
    )

    return () => {
      active = false
    }
  }, [caseId])

  const cortexCheck =
    cortexLookup?.caseId === caseId
      ? cortexLookup
      : null

  const cortexMessage = !cortexCheck
    ? 'Checking cortical reconstruction availability…'
    : cortexCheck.error
      ? 'Status unavailable; standard MRI viewer remains active.'
      : cortexCheck.status?.available
        ? 'Registered reconstruction available for research display; not clinically validated.'
        : cortexCheck.status?.reason === 'qc_not_approved'
          ? 'Reconstruction found; research QC approval pending.'
          : cortexCheck.status?.reason === 'source_identity_mismatch'
            ? 'Cortical access blocked: MRI source identity changed.'
            : cortexCheck.status?.reason === 'source_unavailable'
              ? 'No validation-time source identity; cortical access unavailable.'
              : `Not available: ${cortexCheck.status?.reason ?? 'unknown'}`

  const cortexAvailable =
    cortexCheck?.status?.available === true

  const cortexEnabled =
    cortexAvailable &&
    cortexRequestedFor === caseId &&
    cortexFailedFor !== caseId

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

  const removalEligible =
    markers?.action === 'APPLY_LOCAL_FP' &&
    markers.sam_used === true

  const removalActive =
    showRemoved &&
    removalEligible &&
    showSegmentation &&
    predictionLayer === 'final'

  const removalUrl = viewerMeshUrl(caseId, runId, 'removed')
  const meshUrl = viewerMeshUrl(caseId, runId, predictionLayer)
  const anatomyUrl = viewerAnatomyUrl(
    caseId,
    anatomyStep,
    anatomyFinish,
  )

  function inspectRefinementEvidence() {
    if (!removalEligible) return
    setPredictionLayer('final')
    setPredictionEmphasis(true)
    setShowSegmentation(true)
    setSegmentationOpacity(1)
    setShowAnatomy(true)
    setAnatomyOpacity(0.12)
    setPlane('none')
    setFitMode('anatomy')
    // Preserve the load status if evidence is already mounted.
    // The mesh loader reports loading/ready on actual mount.
    setShowRemoved(true)
  }

  const placeControls = (content: ReactNode) =>
    controlsContainer
      ? createPortal(content, controlsContainer)
      : content

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

      <div className="mesh3d__canvas">
        <MeshErrorBoundary key={`${caseId}:${runId}`}>
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
                    layer={predictionLayer}
                    emphasis={predictionEmphasis}
                    treatment={surfaceTreatment}
                    onSurfaceStatus={setSurfaceStatus}
                  />
                )}

                {removalActive && (
                  <RemovedRefinementSurface
                    key={removalUrl}
                    url={removalUrl}
                    onLoadState={setRemovalStatus}
                  />
                )}

                {showAnatomy && (
                  <AnatomySurface
                    url={anatomyUrl}
                    opacity={anatomyOpacity}
                  />
                )}


                {cortexEnabled && (
                  <CortexLoadBoundary
                    key={caseId}
                    onError={() => setCortexFailedFor(caseId)}
                  >
                    <Suspense fallback={null}>
                      {showLeftCortex && (
                        <CorticalSurface3D
                          url={viewerCortexMeshUrl(caseId, 'lh')}
                          hemisphere="lh"
                          opacity={cortexOpacity}
                        />
                      )}

                      {showRightCortex && (
                        <CorticalSurface3D
                          url={viewerCortexMeshUrl(caseId, 'rh')}
                          hemisphere="rh"
                          opacity={cortexOpacity}
                        />
                      )}
                    </Suspense>
                  </CortexLoadBoundary>
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

      {placeControls(
        <div className="mesh3d__detached-controls">
      <div className="mesh3d__camera-controls" role="group" aria-label="Anatomical camera views">
        <span>View from patient’s</span>
        {ANATOMICAL_VIEWS.map((view) => (
          <button
            key={view.id}
            type="button"
            aria-pressed={cameraView.view === view.id}
            onClick={() => {
              // Match the MRI slice to the selected anatomical view.
              // Prediction-only mode keeps its slice hidden.
              if (plane !== 'none') {
                setPlane(mriPlaneForView(view.id))
              }
              cameraRef.current?.navigate(view.id)
            }}
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
            // Show actual MRI anatomy rather than implying that
            // the foreground envelope is a cortical reconstruction.
            setShowAnatomy(true)
            setAnatomyOpacity(0.15)
            setAnatomyStep(2)
            setAnatomyFinish('raw')
            setShowSegmentation(false)
            setShowRemoved(false)
            setPlane(mriPlaneForView(cameraView.view))
            setTransparentMRIBackground(true)
            setFitMode('volume')
          }}
        >
          Inspect MRI anatomy
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
            setPredictionLayer('final')
            setPredictionEmphasis(false)
            setPlane('none')
            setFitMode('anatomy')
          }}
        >
          Inspect prediction
        </button>
        <button
          type="button"
          onClick={() => {
            setShowAnatomy(true)
            setAnatomyOpacity(0.10)
            setAnatomyStep(2)
            setAnatomyFinish('raw')
            setShowSegmentation(true)
            setSegmentationOpacity(1)
            setPredictionLayer('final')
            setPredictionEmphasis(true)
            setPlane('none')
            setFitMode('anatomy')
          }}
        >
          Spotlight prediction
        </button>

        <button
          type="button"
          className="mesh3d__refinement-preset"
          disabled={!removalEligible}
          onClick={inspectRefinementEvidence}
          title={
            removalEligible
              ? 'Display final prediction and genuinely removed voxels'
              : 'No applied refinement evidence available for this run'
          }
        >
          Show refinement evidence
        </button>
      </div>

      <div
        className="mesh3d__evidence-key"
        aria-label="Visible 3D scientific layers"
      >
        {showAnatomy && (
          <span>
            <i className="mesh3d__key-dot mesh3d__key-dot--anatomy"
               aria-hidden="true" />
            MRI foreground envelope
          </span>
        )}
        {showSegmentation && (
          <span>
            <i
              className={
                predictionLayer === 'final'
                  ? 'mesh3d__key-dot mesh3d__key-dot--final'
                  : 'mesh3d__key-dot mesh3d__key-dot--baseline'
              }
              aria-hidden="true"
            />
            {predictionLayer === 'final'
              ? 'Final model prediction'
              : 'Baseline model prediction'}
          </span>
        )}
        {removalActive && (
          <span>
            <i className="mesh3d__key-dot mesh3d__key-dot--removed"
               aria-hidden="true" />
            Voxels removed by refinement
          </span>
        )}
        <small>Read-only display · Not clinical ground truth</small>
      </div>

      <p role="status" aria-live="polite">
        <strong>Registered cortical reconstruction:</strong>
        {' '}{cortexMessage}
      </p>

      <div
        className="mesh3d__layer-controls"
        aria-label="Registered cortical reconstruction controls"
      >
        <div className="mesh3d__layer-control">
          <label className="mesh3d__layer-toggle">
            <input
              type="checkbox"
              checked={cortexEnabled}
              disabled={
                !cortexAvailable ||
                cortexFailedFor === caseId
              }
              onChange={(event) => {
                setCortexRequestedFor(
                  event.target.checked ? caseId : null
                )
              }}
            />
            FreeSurfer cortex · experimental
          </label>

          <label>
            Opacity {Math.round(cortexOpacity * 100)}%
            <input
              type="range"
              min={10}
              max={100}
              step={5}
              value={Math.round(cortexOpacity * 100)}
              disabled={!cortexEnabled}
              onChange={(event) => {
                setCortexOpacity(Number(event.target.value) / 100)
              }}
            />
          </label>
        </div>

        <div className="mesh3d__layer-control">
          <label>
            <input
              type="checkbox"
              checked={showLeftCortex}
              disabled={!cortexEnabled}
              onChange={(event) =>
                setShowLeftCortex(event.target.checked)
              }
            />
            Left hemisphere
          </label>

          <label>
            <input
              type="checkbox"
              checked={showRightCortex}
              disabled={!cortexEnabled}
              onChange={(event) =>
                setShowRightCortex(event.target.checked)
              }
            />
            Right hemisphere
          </label>
        </div>
      </div>

      {cortexFailedFor === caseId && (
        <p role="alert">
          Optional cortical mesh loading failed.
          The standard MRI viewer remains available.
        </p>
      )}

      <details className="mesh3d__advanced">
        <summary>
          <strong>Advanced 3D controls</strong>
          <span>
            Geometry · prediction comparison · opacity · MRI planes
          </span>
        </summary>
        <div className="mesh3d__advanced-body">
          <section className="mesh3d__advanced-group mesh3d__advanced-group--prediction" aria-label="Prediction & refinement">
            <h4>Prediction & refinement</h4>
      <div
        className="mesh3d__surface-treatment"
        role="group"
        aria-label="Scientific prediction surface treatment"
      >
        <span>Surface treatment</span>

        <button
          type="button"
          aria-pressed={surfaceTreatment === 'original'}
          onClick={() => setSurfaceTreatment('original')}
        >
          Original geometry
        </button>

        <button
          type="button"
          aria-pressed={surfaceTreatment === 'enhanced'}
          onClick={() => setSurfaceTreatment('enhanced')}
        >
          Enhanced display
        </button>

        <small role="status" aria-live="polite">
          {surfaceTreatment === 'original'
            ? 'Voxel-faithful marching-cubes surface'
            : surfaceStatus === 'fallback'
              ? 'Enhancement rejected by geometry safeguards · Original retained'
              : surfaceStatus === 'enhanced'
                ? '4 smoothing pairs · Display-only · Frozen mask unchanged'
                : 'Preparing enhanced geometry…'}
        </small>
      </div>

      <div
        className="mesh3d__prediction-compare"
        role="group"
        aria-label="Compare frozen model prediction surfaces"
      >
        <span>Model surface</span>
        <button
          type="button"
          aria-pressed={predictionLayer === 'final'}
          onClick={() => {
            setPredictionLayer('final')
            setPredictionEmphasis(false)
            setShowRemoved(false)
            setShowSegmentation(true)
          }}
        >
          Final · green
        </button>
        <button
          type="button"
          aria-pressed={predictionLayer === 'baseline'}
          onClick={() => {
            setPredictionLayer('baseline')
            setPredictionEmphasis(false)
            setShowRemoved(false)
            setShowSegmentation(true)
          }}
        >
          Baseline · coral
        </button>
        <small>
          Frozen model outputs · Same RAS geometry ·
          Display colors only
        </small>
      </div>

      <div
        className="mesh3d__refinement-evidence"
        role="group"
        aria-label="Frozen automatic refinement evidence"
      >
        <span>Refinement evidence</span>

        <button
          type="button"
          aria-pressed={removalActive}
          disabled={!removalEligible}
          onClick={inspectRefinementEvidence}
        >
          Final + removed
        </button>

        <button
          type="button"
          disabled={!showRemoved}
          onClick={() => {
            setShowRemoved(false)
            setRemovalStatus('idle')
          }}
        >
          Hide amber
        </button>

        <small role="status" aria-live="polite">
          {!markers
            ? 'Awaiting frozen run evidence'
            : !removalEligible
              ? markers.action === 'ABSTAIN'
                ? 'Policy abstained · No removal expected'
                : 'No applied SAM refinement recorded'
              : !removalActive
                ? 'Amber identifies voxels removed by refinement'
                : removalStatus === 'loading'
                  ? 'Loading genuine removal mask surface…'
                  : removalStatus === 'unavailable'
                    ? 'Removal surface unavailable · Inspect artifact'
                    : removalStatus === 'ready'
                      ? 'Green: final · Amber: removed by refinement'
                      : 'Preparing refinement evidence'}
        </small>
      </div>
          </section>

          <section className="mesh3d__advanced-group mesh3d__advanced-group--anatomy" aria-label="Anatomy & surfaces">
            <h4>Anatomy & surfaces</h4>
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
            {predictionLayer === 'final'
              ? 'Final prediction'
              : 'Baseline prediction'}
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
          </section>

          <section className="mesh3d__advanced-group mesh3d__advanced-group--mri" aria-label="MRI presentation">
            <h4>MRI presentation</h4>
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
          </section>

          <section className="mesh3d__advanced-group mesh3d__advanced-group--camera" aria-label="Camera & framing">
            <h4>Camera & framing</h4>
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
          </section>

        </div>
      </details>
        </div>,
      )}

    </section>
  )
}
