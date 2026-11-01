import { useState } from 'react'

import {
  getFrozenPlaneSliceUrl,
  type FrozenInitialPosition,
  type FrozenAxialLayer,
  type FrozenMriModality,
  type FrozenMriPlane,
} from '../api/frozenTest'

import { projectCrosshairPixel } from '../utils/frozenCrosshairGeometry'

import './FrozenThreePlanePreview.css'

type Coordinates = {
  x: number
  y: number
  z: number
}

type LoadStatus = 'loading' | 'ready' | 'error'

type ImageLoad = {
  url: string
  status: LoadStatus
}

type PlaneInfo = {
  plane: FrozenMriPlane
  axis: keyof Coordinates
  width: number
  height: number
  left: string
  right: string
  top: string
  bottom: string
}

const PLANES: PlaneInfo[] = [
  {
    plane: 'axial',
    axis: 'z',
    width: 182,
    height: 218,
    left: 'R',
    right: 'L',
    top: 'A',
    bottom: 'P',
  },
  {
    plane: 'coronal',
    axis: 'y',
    width: 182,
    height: 182,
    left: 'R',
    right: 'L',
    top: 'S',
    bottom: 'I',
  },
  {
    plane: 'sagittal',
    axis: 'x',
    width: 218,
    height: 182,
    left: 'P',
    right: 'A',
    top: 'S',
    bottom: 'I',
  },
]

const AXES: {
  axis: keyof Coordinates
  label: string
  maximum: number
}[] = [
  { axis: 'z', label: 'Axial Z', maximum: 181 },
  { axis: 'y', label: 'Coronal Y', maximum: 217 },
  { axis: 'x', label: 'Sagittal X', maximum: 181 },
]

function ThreePlaneImages({
  caseIndex,
  coordinates,
  modality,
  layer,
  showCrosshairs,
}: {
  caseIndex: number
  coordinates: Coordinates
  modality: FrozenMriModality
  layer: FrozenAxialLayer
  showCrosshairs: boolean
}) {
  const [statuses, setStatuses] = useState<
    Record<FrozenMriPlane, ImageLoad | null>
  >({
    axial: null,
    coronal: null,
    sagittal: null,
  })

  const [attempt, setAttempt] = useState(0)

  function urlFor(info: PlaneInfo): string {
    const index = coordinates[info.axis]

    const source = getFrozenPlaneSliceUrl(
      caseIndex,
      info.plane,
      index,
      modality,
      layer,
      'volume',
    )

    return `${source}&threeplane_retry=${attempt}`
  }

  function statusFor(info: PlaneInfo): LoadStatus {
    const currentUrl = urlFor(info)
    const recorded = statuses[info.plane]

    return recorded?.url === currentUrl
      ? recorded.status
      : 'loading'
  }

  const allReady = PLANES.every(
    (info) => statusFor(info) === 'ready',
  )

  const failed = PLANES.some(
    (info) => statusFor(info) === 'error',
  )

  function mark(
    plane: FrozenMriPlane,
    url: string,
    status: LoadStatus,
  ) {
    setStatuses((previous) => ({
      ...previous,
      [plane]: { url, status },
    }))
  }

  function retry() {
    setStatuses({
      axial: null,
      coronal: null,
      sagittal: null,
    })

    setAttempt((value) => value + 1)
  }

  return (
    <>
      <div className="frozen-threeplane__grid">
        {PLANES.map((info) => {
          const index = coordinates[info.axis]
          const url = urlFor(info)
          const status = statusFor(info)
          const { column, row } = projectCrosshairPixel(
            info.plane,
            coordinates,
          )

          return (
            <section
              key={info.plane}
              className="frozen-threeplane__panel"
            >
              <div className="frozen-threeplane__panel-title">
                <strong>{info.plane.toUpperCase()}</strong>
                <span>
                  {info.axis.toUpperCase()} = {index}
                </span>
              </div>

              <span className="frozen-threeplane__direction">
                {info.top}
              </span>

              <div className="frozen-threeplane__image-row">
                <span>{info.left}</span>

                <div
                  className="frozen-threeplane__image-frame"
                  style={{
                    aspectRatio: `${info.width} / ${info.height}`,
                  }}
                >
                  <img
                    key={url}
                    src={url}
                    width={info.width}
                    height={info.height}
                    decoding="async"
                    alt={
                      `Case ${caseIndex}, ${info.plane} ` +
                      `slice ${index}, ${modality}, ` +
                      `${layer} overlay`
                    }
                    style={{
                      visibility: status === 'ready'
                        ? 'visible'
                        : 'hidden',
                    }}
                    onLoad={() => mark(info.plane, url, 'ready')}
                    onError={() => mark(info.plane, url, 'error')}
                  />

                  {showCrosshairs && status === 'ready' && (
                    <svg
                      className="frozen-threeplane__crosshair"
                      viewBox={`0 0 ${info.width} ${info.height}`}
                      preserveAspectRatio="none"
                      aria-hidden="true"
                      focusable="false"
                    >
                      <line
                        x1={column + 0.5}
                        y1={0}
                        x2={column + 0.5}
                        y2={info.height}
                      />
                      <line
                        x1={0}
                        y1={row + 0.5}
                        x2={info.width}
                        y2={row + 0.5}
                      />
                      <circle
                        cx={column + 0.5}
                        cy={row + 0.5}
                        r={2.5}
                      />
                    </svg>
                  )}
                </div>

                <span>{info.right}</span>
              </div>

              <span className="frozen-threeplane__direction">
                {info.bottom}
              </span>
            </section>
          )
        })}
      </div>

      {failed ? (
        <div role="alert" className="frozen-threeplane__message">
          <p>
            One or more verified MRI views are unavailable.
            The unavailable panel is hidden.
            Other successfully loaded views remain visible.
          </p>
          <button type="button" onClick={retry}>
            Retry all three views
          </button>
        </div>
      ) : !allReady ? (
        <p role="status" className="frozen-threeplane__message">
          Loading updated MRI views…
        </p>
      ) : (
        <p role="status" className="frozen-threeplane__message">
          All three authenticated MRI views loaded.
        </p>
      )}
    </>
  )
}

export function FrozenThreePlanePreview({
  caseIndex,
  modality,
  layer,
  coordinates,
  recommendation,
  onCoordinatesChange,
}: {
  caseIndex: number
  modality: FrozenMriModality
  layer: FrozenAxialLayer
  coordinates: Coordinates
  recommendation: FrozenInitialPosition
  onCoordinatesChange: (next: Coordinates) => void
}) {
  const [showCrosshairs, setShowCrosshairs] = useState(true)

  const recommended = recommendation.coordinates

  return (
    <section
      className="frozen-threeplane"
      aria-label="Synchronized three-plane MRI evidence"
    >
      <header className="frozen-threeplane__header">
        <strong>Three-plane MRI</strong>
        <span>
          Case {String(caseIndex).padStart(3, '0')}
        </span>
      </header>

      <p className="frozen-threeplane__description">
        Original MRI · Shared volume contrast ·
        Read-only frozen prediction evidence
      </p>

      <div className="frozen-threeplane__controls">
        {AXES.map(({ axis, label, maximum }) => (
          <label key={axis}>
            <span>
              {label}: {coordinates[axis]} / {maximum}
            </span>

            <input
              type="range"
              min={0}
              max={maximum}
              step={1}
              value={coordinates[axis]}
              onChange={(event) => {
                const next = Number(event.target.value)

                onCoordinatesChange({
                  ...coordinates,
                  [axis]: next,
                })
              }}
            />
          </label>
        ))}
        <button
          type="button"
          disabled={
            coordinates.x === recommended.x &&
            coordinates.y === recommended.y &&
            coordinates.z === recommended.z
          }
          onClick={() => onCoordinatesChange({ ...recommended })}
        >
          Reset to recommended position
        </button>
      </div>

      <label className="frozen-threeplane__crosshair-toggle">
        <input
          type="checkbox"
          checked={showCrosshairs}
          onChange={(event) =>
            setShowCrosshairs(event.target.checked)
          }
        />
        Show anatomical intersection crosshairs
      </label>

      <p className="frozen-threeplane__description">
        Recommended native XYZ: ({recommended.x}, {recommended.y},
        {recommended.z}) · Source:{' '}
        {recommendation.mask_source === 'volume_center'
          ? 'geometric volume center'
          : `${recommendation.mask_source} prediction mask`}
        . Display positioning only, not clinical localization.
      </p>

      <ThreePlaneImages
        caseIndex={caseIndex}
        coordinates={coordinates}
        modality={modality}
        layer={layer}
        showCrosshairs={showCrosshairs}
      />

      <p className="frozen-threeplane__description">
        The three positions refer to one native MRI voxel
        coordinate (X, Y, Z). Orientation letters indicate
        patient directions. Images are display-only and
        do not perform inference or ground-truth comparison.
      </p>
    </section>
  )
}
