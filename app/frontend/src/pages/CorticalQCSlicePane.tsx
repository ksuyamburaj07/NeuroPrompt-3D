import { useEffect, useState } from 'react'
import './CorticalQCSlicePane.css'

export type QCPlane = 'axial' | 'coronal' | 'sagittal'

type QCPlaneInfo = {
  axis: number
  count: number
  center_index: number
  width: number
  height: number
}

export type QCMetadata = {
  orientation_convention: string
  shape_ras_xyz: number[]
  affine_ras: number[][]
  source_sha256: string
  reconstruction_id: string
  qc_status: string
  clinically_validated: boolean
  planes: Record<QCPlane, QCPlaneInfo>
}

const qcBase = (
  import.meta.env.VITE_QC_MRI_BASE_URL
  || (import.meta.env.DEV ? 'http://127.0.0.1:8013' : '/qc/mri')
)


type Props = {
  plane: QCPlane
  metadata: QCMetadata
  indices: Record<QCPlane, number>
  showContours: boolean
  onIndexChange: (plane: QCPlane, index: number) => void
}

function percentage(index: number, count: number): number {
  return 100 * (index + 0.5) / count
}

export function CorticalQCSlicePane({
  plane,
  metadata,
  indices,
  showContours,
  onIndexChange,
}: Props) {
  const [frame, setFrame] = useState<{
    index: number
    mriSrc: string
    contourSrc: string | null
  } | null>(null)

  const [frameError, setFrameError] = useState<string | null>(null)
  const [retry, setRetry] = useState(0)

  const info = metadata.planes[plane]
  const index = indices[plane]
  const [sizeX, sizeY, sizeZ] = metadata.shape_ras_xyz

  const framePending =
    frame === null ||
    frame.index !== index ||
    (showContours && frame.contourSrc === null)

  const mriUrl = `${qcBase}/slices/${plane}/${index}`
  const contourUrl = `${qcBase}/contours/${plane}/${index}`

  // Hold the previous complete frame during rapid slider movements.
  // MRI and contour blobs are decoded before being displayed together.
  useEffect(() => {
    const controller = new AbortController()
    let active = true
    let uncommittedUrls: string[] = []

    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          const requests = showContours
            ? [mriUrl, contourUrl]
            : [mriUrl]

          const responses = await Promise.all(
            requests.map((url) =>
              fetch(url, {
                signal: controller.signal,
                cache: 'no-store',
              }),
            ),
          )

          if (responses.some((response) => !response.ok)) {
            throw new Error('MRI or contour request failed.')
          }

          const blobs = await Promise.all(
            responses.map((response) => response.blob()),
          )

          const objectUrls = blobs.map((blob) =>
            URL.createObjectURL(blob),
          )

          uncommittedUrls = objectUrls

          await Promise.all(
            objectUrls.map(async (url) => {
              const image = new Image()
              image.src = url
              await image.decode()
            }),
          )

          if (!active) return

          setFrame({
            index,
            mriSrc: objectUrls[0],
            contourSrc: showContours ? objectUrls[1] : null,
          })

          // Ownership moves to the frame cleanup effect.
          uncommittedUrls = []
          setFrameError(null)
        } catch {
          if (active) {
            setFrameError('Unable to load the requested MRI slice.')
          }
        }
      })()
    }, 120)

    return () => {
      active = false
      controller.abort()
      window.clearTimeout(timer)

      for (const url of uncommittedUrls) {
        URL.revokeObjectURL(url)
      }
    }
  }, [mriUrl, contourUrl, index, showContours, retry])

  // Release the previous decoded images after a frame replacement.
  useEffect(() => {
    return () => {
      if (frame) {
        URL.revokeObjectURL(frame.mriSrc)
        if (frame.contourSrc) {
          URL.revokeObjectURL(frame.contourSrc)
        }
      }
    }
  }, [frame])

  const horizontal =
    plane === 'sagittal'
      ? percentage(indices.coronal, sizeY)
      : percentage(indices.sagittal, sizeX)

  const vertical =
    plane === 'axial'
      ? 100 - percentage(indices.coronal, sizeY)
      : 100 - percentage(indices.axial, sizeZ)

  const leftDirection = plane === 'sagittal' ? 'P' : 'L'
  const rightDirection = plane === 'sagittal' ? 'A' : 'R'
  const topDirection = plane === 'axial' ? 'A' : 'S'

  return (
    <section className="qc-slice">
      <header className="qc-slice__heading">
        <strong>{plane.toUpperCase()}</strong>
        <span>
          {plane === 'axial' ? 'Z' : plane === 'coronal' ? 'Y' : 'X'}
          {' · shown '}{frame?.index ?? '—'} / {info.count - 1}
          {framePending && (
            <span className="qc-slice__requested">
              {' · requested '}{index}
            </span>
          )}
        </span>
      </header>

      <div
        className="qc-slice__image-area"
        style={{ aspectRatio: `${info.width} / ${info.height}` }}
      >
        {frame ? (
          <>
            <img
              className="qc-slice__image"
              src={frame.mriSrc}
              alt={`Original T1N ${plane} MRI slice ${frame.index}`}
              draggable={false}
            />

            {showContours && frame.contourSrc && (
              <img
                className="qc-slice__contours"
                src={frame.contourSrc}
                alt=""
                aria-hidden="true"
                draggable={false}
              />
            )}
          </>
        ) : (
          <div className="qc-slice__error" role="status">
            {frameError ?? 'Loading original T1N MRI…'}
            {frameError && (
              <button
                type="button"
                onClick={() => setRetry((value) => value + 1)}
              >
                Retry
              </button>
            )}
          </div>
        )}

        {frame && framePending && !frameError && (
          <span className="qc-slice__updating" role="status">
            Updating slice…
          </span>
        )}

        {frameError && frame && (
          <div className="qc-slice__load-warning" role="alert">
            <span>
              Requested slice unavailable.
              Showing {frame.index}.
            </span>
            <button
              type="button"
              onClick={() => setRetry((value) => value + 1)}
            >
              Retry
            </button>
          </div>
        )}

        <div
          className="qc-slice__crosshair qc-slice__crosshair--horizontal"
          style={{ top: `${vertical}%` }}
          aria-hidden="true"
        />
        <div
          className="qc-slice__crosshair qc-slice__crosshair--vertical"
          style={{ left: `${horizontal}%` }}
          aria-hidden="true"
        />

        <span className="qc-slice__direction qc-slice__direction--left">
          {leftDirection}
        </span>
        <span className="qc-slice__direction qc-slice__direction--right">
          {rightDirection}
        </span>
        <span className="qc-slice__direction qc-slice__direction--top">
          {topDirection}
        </span>
      </div>

      <label className="qc-slice__slider">
        <span>Slice position</span>
        <input
          type="range"
          aria-label={`${plane} MRI slice position`}
          min={0}
          max={info.count - 1}
          step={1}
          value={index}
          onChange={(event) => {
            onIndexChange(plane, Number(event.target.value))
          }}
        />
      </label>
    </section>
  )
}
