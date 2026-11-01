import { Component, Suspense, useEffect, useRef, useState, type ReactNode, type RefObject } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Bounds, Html, OrbitControls } from '@react-three/drei'

import { CorticalSurface3D } from './CorticalSurface3D'
import { GenuineQCPlaneTexture3D } from './GenuineQCPlaneTexture3D'
import { Quaternion, Vector3 } from 'three'
import type { OrbitControls as OrbitImpl } from 'three-stdlib'
import '../np3d-b7d-real-qc.css'
import {
  CorticalQCSlicePane,
  type QCMetadata,
  type QCPlane,
} from './CorticalQCSlicePane'
import {
  getGenuineQCMetadata,
} from './genuineQCMetadata'

const assetBase = (
  import.meta.env.VITE_QC_MESH_BASE_URL
  || (import.meta.env.DEV ? 'http://127.0.0.1:8012' : '/qc/meshes')
)


type MeshErrorBoundaryProps = {
  children: ReactNode
}

type MeshErrorBoundaryState = {
  failed: boolean
}

class CorticalMeshErrorBoundary extends Component<
  MeshErrorBoundaryProps,
  MeshErrorBoundaryState
> {
  state: MeshErrorBoundaryState = { failed: false }

  static getDerivedStateFromError(): MeshErrorBoundaryState {
    return { failed: true }
  }

  render() {
    if (this.state.failed) {
      return (
        <div className="np3d-qc__mesh-error" role="alert">
          <strong>3D cortical mesh unavailable</strong>
          <p>
            Unable to load the provisional cortical geometry.
            Check the local allowlisted mesh service on
            port 8012.
          </p>
          <p>
            No anatomical approval or clinical validation
            is implied.
          </p>
          <button
            type="button"
            onClick={() => this.setState({ failed: false })}
          >
            Retry 3D loading
          </button>
        </div>
      )
    }

    return this.props.children
  }
}


type QCView =
  | 'left'
  | 'right'
  | 'anterior'
  | 'posterior'
  | 'superior'
  | 'inferior'

type QCRequest = {
  view: QCView
  sequence: number
}

const cameraViews: Array<{ view: QCView; label: string }> = [
  { view: 'left', label: 'Left · L' },
  { view: 'right', label: 'Right · R' },
  { view: 'anterior', label: 'Anterior · A' },
  { view: 'posterior', label: 'Posterior · P' },
  { view: 'superior', label: 'Superior · S' },
  { view: 'inferior', label: 'Inferior · I' },
]

const cameraDirections: Record<QCView, [number, number, number]> = {
  left: [-1, 0, 0],
  right: [1, 0, 0],
  anterior: [0, 1, 0],
  posterior: [0, -1, 0],
  superior: [0, 0, 1],
  inferior: [0, 0, -1],
}

function QCCameraDirector({
  request,
  orbitRef,
}: {
  request: QCRequest | null
  orbitRef: RefObject<OrbitImpl | null>
}) {
  const { camera, invalidate } = useThree()

  const motion = useRef<{
    elapsed: number
    center: Vector3
    start: Vector3
    rotation: Quaternion
    distance: number
    startUp: Vector3
    endUp: Vector3
  } | null>(null)

  useEffect(() => {
    const controls = orbitRef.current

    if (!request || !controls) {
      motion.current = null
      return
    }

    const center = controls.target.clone()
    const offset = camera.position.clone().sub(center)
    const distance = Math.max(offset.length(), 70)

    const start = offset.lengthSq() > 0
      ? offset.normalize()
      : new Vector3(0, -1, 0)

    const end = new Vector3(
      ...cameraDirections[request.view],
    )

    const pole =
      request.view === 'superior' ||
      request.view === 'inferior'

    motion.current = {
      elapsed: 0,
      center,
      start,
      rotation: new Quaternion().setFromUnitVectors(start, end),
      distance,
      startUp: camera.up.clone().normalize(),
      endUp: pole
        ? new Vector3(0, 1, 0)
        : new Vector3(0, 0, 1),
    }

    invalidate()
  }, [request, orbitRef, camera, invalidate])

  useFrame((_, delta) => {
    const movement = motion.current
    const controls = orbitRef.current

    if (!movement || !controls) return

    movement.elapsed += Math.min(delta, 0.1)

    const t = Math.min(movement.elapsed / 0.65, 1)
    const eased = t * t * (3 - 2 * t)

    const rotation = new Quaternion()
      .identity()
      .slerp(movement.rotation, eased)

    const direction = movement.start
      .clone()
      .applyQuaternion(rotation)

    camera.position
      .copy(movement.center)
      .addScaledVector(direction, movement.distance)

    camera.up
      .copy(movement.startUp)
      .lerp(movement.endUp, eased)
      .normalize()

    camera.lookAt(movement.center)
    controls.update()

    if (t < 1) {
      invalidate()
    } else {
      motion.current = null
    }
  })

  return null
}

export function CorticalQCPreview() {
  const orbitRef = useRef<OrbitImpl | null>(null)
  const [cameraRequest, setCameraRequest] =
    useState<QCRequest | null>(null)

  const [showLeft, setShowLeft] = useState(true)
  const [showRight, setShowRight] = useState(true)
  const [opacity, setOpacity] = useState(1)
  const [sulcalContrast, setSulcalContrast] = useState(false)
  const [showContours, setShowContours] = useState(true)
  const [showAxial3D, setShowAxial3D] = useState(false)
  const [axial3DOpacity, setAxial3DOpacity] = useState(0.8)

  const [extraPlaneVisible, setExtraPlaneVisible] = useState({
    coronal: false,
    sagittal: false,
  })
  const [extraPlaneOpacity, setExtraPlaneOpacity] = useState({
    coronal: 0.8,
    sagittal: 0.8,
  })


  const [metadata, setMetadata] = useState<QCMetadata | null>(null)
  const [metadataError, setMetadataError] = useState<string | null>(null)

  const [indices, setIndices] = useState<Record<QCPlane, number>>({
    axial: 91,
    coronal: 98,
    sagittal: 65,
  })

  useEffect(() => {
    let active = true

    getGenuineQCMetadata()
      .then((result) => {
        if (!active) return
        setMetadata(result)
        setMetadataError(null)
      })
      .catch((error: unknown) => {
        if (!active) return
        setMetadata(null)
        setMetadataError(
          error instanceof Error
            ? error.message
            : 'Unable to load genuine MRI QC metadata.',
        )
      })

    return () => {
      active = false
    }
  }, [])

  function applyDisplayPreset(
    preset: 'cortex' | QCPlane | 'mri-three',
  ) {
    const mriOnly = preset === 'mri-three'
    const cortexOnly = preset === 'cortex'

    setShowLeft(!mriOnly)
    setShowRight(!mriOnly)
    setOpacity(cortexOnly || mriOnly ? 1 : 0.35)

    setShowAxial3D(mriOnly || preset === 'axial')
    setExtraPlaneVisible({
      coronal: mriOnly || preset === 'coronal',
      sagittal: mriOnly || preset === 'sagittal',
    })

    if (!cortexOnly) {
      setAxial3DOpacity(1)
      setExtraPlaneOpacity({ coronal: 1, sagittal: 1 })
    }
  }

  function changeIndex(plane: QCPlane, index: number) {
    setIndices((previous) => ({
      ...previous,
      [plane]: index,
    }))
  }

  return (
    <main
      style={{
        background: '#081923',
        color: '#edf7fa',
        minHeight: '100vh',
        padding: 24,
        fontFamily: 'system-ui, sans-serif',
      }}
    >
      <div style={{ maxWidth: 1540, margin: '0 auto' }}>
        <header style={{ marginBottom: 20 }}>
          <p style={{ color: '#6ac8da' }}>
            NEUROPROMPT-3D · CORTICAL RESEARCH QC
          </p>

          <h1>Genuine FreeSurfer Cortical Reconstruction</h1>

          <p>
            Original bilateral pial geometry · Physical RAS+ mm
            · No decimation · Read-only diagnostic preview
          </p>

          <p style={{ color: '#e6b38e' }}>
            PROVISIONAL ANATOMICAL QC — NOT CLINICALLY VALIDATED.
            Known uncertainties remain in the post-treatment region.
          </p>
        </header>




        {metadata && (
          <section
            className="np3d-qc__provenance"
            aria-label="Research reconstruction provenance"
          >
            <div>
              <span>3D anatomical source</span>
              <strong className="np3d-qc__source-label">
                FreeSurfer bilateral pial surfaces
              </strong>
              <details className="np3d-qc__source-detail">
                <summary>Technical provenance</summary>
                <code>{metadata.reconstruction_id}</code>
              </details>
            </div>
            <div>
              <span>Pinned T1N source SHA-256 · metadata check</span>
              <code>{metadata.source_sha256}</code>
            </div>
            <div>
              <span>Anatomical QC</span>
              <strong>PROVISIONAL · Research only</strong>
            </div>
          </section>
        )}

        {metadataError && (
          <p role="alert" style={{ color: '#f1c19c' }}>
            MRI QC service: {metadataError}
            {' '}The 3D cortical preview remains available.
          </p>
        )}

        <div className={`np3d-qc__stage${metadata ? ' np3d-qc__stage--mri-ready' : ''}`}>
          {metadata && (['axial', 'coronal', 'sagittal'] as const).map(
            (plane) => (
              <div
                key={plane}
                className={`np3d-qc__slot--${plane}`}
              >
                <CorticalQCSlicePane
                  plane={plane}
                  metadata={metadata}
                  indices={indices}
                  showContours={showContours}
                  onIndexChange={changeIndex}
                />
              </div>
            ),
          )}

          <section className="np3d-qc__hero">
            <details className="np3d-qc__viewer-guide">
              <summary>What am I viewing?</summary>
              <div className="np3d-qc__viewer-guide-content">
                <strong>Bilateral pial cortical surfaces</strong>
                <p>
                  These FreeSurfer-derived meshes approximate
                  the outer cortical boundary, including the
                  visible gyri and sulci.
                </p>
                <p>
                  {sulcalContrast
                    ? 'Experimental sulcal shading is active; normal hemisphere fill colors are replaced.'
                    : 'Gold identifies the left hemisphere; cyan identifies the right.'}
                </p>
                <p>
                  Drag to rotate, scroll to zoom, or choose
                  a patient-oriented camera direction below.
                </p>
                <p>
                  Axial, coronal and sagittal MRI planes can
                  be displayed in the same RAS+ coordinate
                  space. Opacity changes affect appearance
                  only, not anatomical data.
                </p>
                <p className="np3d-qc__guide-caution">
                  Provisional reconstruction. Visual appearance
                  does not establish anatomical accuracy or
                  clinical suitability.
                </p>
              </div>
            </details>
        <div className="np3d-qc__canvas-shell">
          <CorticalMeshErrorBoundary>
          <Canvas
            frameloop="demand"
            dpr={1}
            camera={{
              position: [0, -340, 130],
              fov: 44,
              near: 0.1,
              far: 2000,
            }}
            onCreated={({ camera }) => {
              camera.up.set(0, 0, 1)
            }}
            gl={{
              antialias: false,
              powerPreference: 'low-power',
            }}
          >
            <color
              attach="background"
              args={['#091c27']}
            />

            <ambientLight intensity={1.0} />

            <directionalLight
              position={[110, -90, 160]}
              intensity={2.0}
            />

            <directionalLight
              position={[-120, 90, -30]}
              intensity={0.6}
            />

            <OrbitControls
              ref={orbitRef}
              makeDefault
              enableDamping
              minDistance={70}
              maxDistance={1100}
              onStart={() => setCameraRequest(null)}
            />

            <QCCameraDirector
              request={cameraRequest}
              orbitRef={orbitRef}
            />

            {(showLeft || showRight) && (
              <Suspense
                fallback={
                  <Html center>
                    <span style={{ color: 'white' }}>
                      Loading genuine cortical meshes…
                    </span>
                  </Html>
                }
              >
                <Bounds fit clip observe margin={1.10}>
                  {showLeft && (
                    <CorticalSurface3D
                      key={sulcalContrast ? 'lh-sulcal' : 'lh-uniform'}
                      url={`${assetBase}/lh.pial.ras${sulcalContrast ? '.sulc-qc' : ''}.ply`}
                      hemisphere="lh"
                      opacity={opacity}
                      useSulcalColors={sulcalContrast}
                    />
                  )}

                  {showRight && (
                    <CorticalSurface3D
                      key={sulcalContrast ? 'rh-sulcal' : 'rh-uniform'}
                      url={`${assetBase}/rh.pial.ras${sulcalContrast ? '.sulc-qc' : ''}.ply`}
                      hemisphere="rh"
                      opacity={opacity}
                      useSulcalColors={sulcalContrast}
                    />
                  )}
                </Bounds>
              </Suspense>
            )}
            {metadata && showAxial3D && (
              <GenuineQCPlaneTexture3D
                metadata={metadata}
                plane="axial"
                index={indices.axial}
                opacity={axial3DOpacity}
              />
            )}
            {metadata && (['coronal', 'sagittal'] as const).map(
              (plane) => extraPlaneVisible[plane] ? (
                <GenuineQCPlaneTexture3D
                  key={plane}
                  metadata={metadata}
                  plane={plane}
                  index={indices[plane]}
                  opacity={extraPlaneOpacity[plane]}
                />
              ) : null,
            )}
          </Canvas>
          </CorticalMeshErrorBoundary>
        </div>
          </section>

          {metadata && (
            <aside className="np3d-qc__context">
              <strong>Genuine anatomical inspection</strong>
              <p>
                Original T1N MRI · 182 × 218 × 182 · RAS+
              </p>
              <p>
                Linked axial, coronal and sagittal slice positions.
                The 3D camera remains independently navigable.
              </p>
              <p className="np3d-qc__legend">
                {sulcalContrast ? (
                  <>
                    Experimental sulcal shading enabled.
                    Normal gold/cyan surface fills are replaced.
                  </>
                ) : (
                  <>
                    Gold: left pial surface<br />
                    Cyan: right pial surface
                  </>
                )}
              </p>
              <p style={{ color: '#e6b38e' }}>
                PROVISIONAL QC. Anatomical concerns remain unresolved.
                No clinical validation or production approval.
              </p>
            </aside>
          )}
        </div>

        <section
          className="np3d-qc__camera-navigation"
          aria-label="Patient-oriented anatomical camera"
        >
          <header>
            <strong>Anatomical camera</strong>
            <span>Patient RAS+ orientation</span>
          </header>

          <div className="np3d-qc__camera-buttons">
            {cameraViews.map(({ view, label }) => (
              <button
                key={view}
                type="button"
                aria-pressed={cameraRequest?.view === view}
                onClick={() => setCameraRequest((previous) => ({
                  view,
                  sequence: (previous?.sequence ?? 0) + 1,
                }))}
              >
                {label}
              </button>
            ))}
          </div>

          <p>
            Patient-relative directions · Side views: superior up
            · Top/bottom views: anterior up · Drag to rotate freely
          </p>
        </section>

        <section
          className="np3d-qc__camera-navigation"
          aria-label="Registered MRI inside 3D brain"
        >
          <header>
            <strong>Original MRI inside 3D brain</strong>
            <span>Canonical RAS+ · research inspection</span>
          </header>

          <div
            role="group"
            aria-label="Research display presets"
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: 8,
              marginBottom: 16,
            }}
          >
            <button type="button"
              onClick={() => applyDisplayPreset('cortex')}>
              Cortex only
            </button>
            <button type="button"
              onClick={() => applyDisplayPreset('axial')}>
              Cortex + Axial
            </button>
            <button type="button"
              onClick={() => applyDisplayPreset('coronal')}>
              Cortex + Coronal
            </button>
            <button type="button"
              onClick={() => applyDisplayPreset('sagittal')}>
              Cortex + Sagittal
            </button>
            <button type="button"
              onClick={() => applyDisplayPreset('mri-three')}>
              MRI three-plane
            </button>
          </div>

          <label>
            <input
              type="checkbox"
              checked={showAxial3D}
              disabled={!metadata}
              onChange={(event) =>
                setShowAxial3D(event.target.checked)
              }
            />
            {' '}Show registered axial T1N plane
          </label>

          <label>
            Plane opacity: {Math.round(axial3DOpacity * 100)}%
            <input
              type="range"
              min={10}
              max={100}
              step={5}
              value={Math.round(axial3DOpacity * 100)}
              disabled={!showAxial3D}
              onChange={(event) =>
                setAxial3DOpacity(Number(event.target.value) / 100)
              }
            />
          </label>

          {(['coronal', 'sagittal'] as const).map((plane) => (
            <div key={plane} style={{ display: 'grid', gap: 8, marginTop: 12 }}>
              <label>
                <input
                  type="checkbox"
                  checked={extraPlaneVisible[plane]}
                  disabled={!metadata}
                  onChange={(event) =>
                    setExtraPlaneVisible((previous) => ({
                      ...previous,
                      [plane]: event.target.checked,
                    }))
                  }
                />
                {` Show registered ${plane} T1N plane`}
              </label>

              <label>
                {plane} plane opacity: {Math.round(extraPlaneOpacity[plane] * 100)}%
                <input
                  type="range"
                  min={10}
                  max={100}
                  step={5}
                  value={Math.round(extraPlaneOpacity[plane] * 100)}
                  disabled={!extraPlaneVisible[plane]}
                  onChange={(event) =>
                    setExtraPlaneOpacity((previous) => ({
                      ...previous,
                      [plane]: Number(event.target.value) / 100,
                    }))
                  }
                />
              </label>
            </div>
          ))}

          <p>
            Uses the original MRI and its existing linked slice
            positions. To inspect the plane through the cortical
            shell, reduce hemisphere opacity under Research
            display controls. The original meshes are unchanged.
          </p>
        </section>

        <section
          className="np3d-qc__research-controls"
          aria-label="Research visualization controls"
        >
          <header className="np3d-qc__controls-heading">
            <strong>Research display controls</strong>
            <span>
              Provisional anatomy · Display-only adjustments
            </span>
          </header>
          <label>
            <input
              type="checkbox"
              checked={showLeft}
              onChange={(event) =>
                setShowLeft(event.target.checked)
              }
            />
            {' '}Left hemisphere
          </label>

          <label>
            <input
              type="checkbox"
              checked={showRight}
              onChange={(event) =>
                setShowRight(event.target.checked)
              }
            />
            {' '}Right hemisphere
          </label>

          <label>
            <input
              type="checkbox"
              checked={sulcalContrast}
              onChange={(event) =>
                setSulcalContrast(event.target.checked)
              }
            />
            {' '}Sulcal Contrast A · experimental
          </label>

          <label>
            <input
              type="checkbox"
              checked={showContours}
              disabled={!metadata}
              onChange={(event) =>
                setShowContours(event.target.checked)
              }
            />
            {' '}Gold/cyan MRI boundaries
          </label>

          <label>
            Opacity {Math.round(opacity * 100)}%
            {' '}
            <input
              type="range"
              min={25}
              max={100}
              step={5}
              value={Math.round(opacity * 100)}
              onChange={(event) =>
                setOpacity(Number(event.target.value) / 100)
              }
            />
          </label>
        </section>

        <p style={{ color: '#b8d0db' }}>
          Drag to rotate · Scroll to zoom · Patient superior is +Z.
          The 3D view shows provisional pial geometry. The MRI panels
          display registered cortical intersections for research QC only.
        </p>
      </div>
    </main>
  )
}
