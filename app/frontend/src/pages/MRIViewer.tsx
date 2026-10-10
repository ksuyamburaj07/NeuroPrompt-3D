import { useEffect, useState } from 'react'
import {
  getViewerMetadata,
  viewerSliceUrl,
  viewerOverlayUrl,
  getViewerRunMarkers,
  type ViewerRunMarkers,
  type ViewerSpatialMarker,
  type MRIModality,
  type MRIPlane,
  type MRIViewerMetadata,
  type MRIOverlayLayer,
} from '../api/viewer'
import './MRIViewer.css'

type Props = {
  caseId: string
  completedRunId: string | null
}

type SliceIndices = Record<MRIPlane, number>

const planes: MRIPlane[] = [
  'axial',
  'coronal',
  'sagittal',
]

const modalities: { key: MRIModality; name: string }[] = [
  { key: 't2f', name: 'FLAIR' },
  { key: 't1n', name: 'T1' },
  { key: 't1c', name: 'T1ce' },
  { key: 't2w', name: 'T2' },
]

function percentage(index: number, count: number): number {
  if (count <= 1) return 50
  return Math.max(0, Math.min(100, 100 * (index + 0.5) / count))
}


function ScientificMarker({
  kind,
  marker,
  plane,
  indices,
  metadata,
}: {
  kind: 'hotspot' | 'negative-prompt'
  marker: ViewerSpatialMarker | null
  plane: MRIPlane
  indices: SliceIndices
  metadata: MRIViewerMetadata
}) {
  if (!marker) return null

  const [x, y, z] = marker.ras_xyz
  const [sizeX, sizeY, sizeZ] = metadata.shape_ras_xyz

  if (marker.ras_xyz[metadata.planes[plane].axis] !== indices[plane]) {
    return null
  }

  const horizontal = plane === 'sagittal'
    ? percentage(y, sizeY)
    : percentage(x, sizeX)

  const vertical = plane === 'axial'
    ? 100 - percentage(y, sizeY)
    : 100 - percentage(z, sizeZ)

  const description = kind === 'hotspot'
    ? 'Frozen uncertainty hotspot'
    : 'Negative SAM prompt'

  return (
    <span
      role="img"
      aria-label={description}
      title={`${description}: RAS voxel (${x}, ${y}, ${z})`}
      className={`mri-viewer__scientific-marker mri-viewer__scientific-marker--${kind}`}
      style={{
        left: `${horizontal}%`,
        top: `${vertical}%`,
      }}
    />
  )
}

function SlicePane({
  caseId,
  modality,
  plane,
  metadata,
  indices,
  completedRunId,
  overlayLayer,
  overlayOpacity,
  markers,
  onIndexChange,
}: {
  caseId: string
  modality: MRIModality
  plane: MRIPlane
  metadata: MRIViewerMetadata
  indices: SliceIndices
  completedRunId: string | null
  overlayLayer: MRIOverlayLayer | 'none'
  overlayOpacity: number
  markers: ViewerRunMarkers | null
  onIndexChange: (plane: MRIPlane, index: number) => void
}) {
  const [failedUrl, setFailedUrl] = useState<string | null>(null)
  const [failedOverlayUrl, setFailedOverlayUrl] = useState<string | null>(null)

  const info = metadata.planes[plane]
  const index = indices[plane]
  const [sizeX, sizeY, sizeZ] = metadata.shape_ras_xyz

  const url = viewerSliceUrl(caseId, modality, plane, index)

  const overlayUrl =
    completedRunId && overlayLayer !== 'none'
      ? viewerOverlayUrl(
          caseId, completedRunId, overlayLayer, plane, index,
        )
      : null

  const horizontal = plane === 'sagittal'
    ? percentage(indices.coronal, sizeY)
    : percentage(indices.sagittal, sizeX)

  const vertical = plane === 'axial'
    ? 100 - percentage(indices.coronal, sizeY)
    : 100 - percentage(indices.axial, sizeZ)

  const leftLabel = plane === 'sagittal' ? 'P' : 'L'
  const rightLabel = plane === 'sagittal' ? 'A' : 'R'
  const topLabel = plane === 'axial' ? 'A' : 'S'

  return (
    <section className="mri-viewer__pane">
      <header className="mri-viewer__pane-heading">
        <strong>{plane.toUpperCase()}</strong>
        <span>
          {plane === 'axial' ? 'Z' : plane === 'coronal' ? 'Y' : 'X'}
          {' '}· {index} / {info.count - 1}
        </span>
      </header>

      <div className="mri-viewer__image-frame">
        <div
          className="mri-viewer__image-area"
          style={{ aspectRatio: `${info.width} / ${info.height}` }}
        >
          {failedUrl === url ? (
            <div className="mri-viewer__image-error">
              Slice unavailable.
              <button
                type="button"
                onClick={() => setFailedUrl(null)}
              >
                Retry image
              </button>
            </div>
          ) : (
            <img
              key={url}
              className="mri-viewer__image"
              src={url}
              alt={`Actual ${modality.toUpperCase()} ${plane} MRI slice ${index}`}
              draggable={false}
              onError={() => setFailedUrl(url)}
            />
          )}

          {overlayUrl && failedUrl !== url && (
            failedOverlayUrl === overlayUrl ? (
              <div className="mri-viewer__overlay-error" role="status">
                Overlay unavailable
                <button
                  type="button"
                  onClick={() => setFailedOverlayUrl(null)}
                >
                  Retry
                </button>
              </div>
            ) : (
              <img
                key={overlayUrl}
                className="mri-viewer__overlay"
                src={overlayUrl}
                alt=""
                aria-hidden="true"
                draggable={false}
                style={{ opacity: overlayOpacity / 100 }}
                onError={() => setFailedOverlayUrl(overlayUrl)}
              />
            )
          )}

          {markers && (
            <>
              <ScientificMarker
                kind="hotspot"
                marker={markers.hotspot}
                plane={plane}
                indices={indices}
                metadata={metadata}
              />
              <ScientificMarker
                kind="negative-prompt"
                marker={markers.negative_prompt}
                plane={plane}
                indices={indices}
                metadata={metadata}
              />
            </>
          )}

          <div
            className="mri-viewer__crosshair-horizontal"
            style={{ top: `${vertical}%` }}
            aria-hidden="true"
          />
          <div
            className="mri-viewer__crosshair-vertical"
            style={{ left: `${horizontal}%` }}
            aria-hidden="true"
          />
          <span className="mri-viewer__direction mri-viewer__direction--left">
            {leftLabel}
          </span>
          <span className="mri-viewer__direction mri-viewer__direction--right">
            {rightLabel}
          </span>
          <span className="mri-viewer__direction mri-viewer__direction--top">
            {topLabel}
          </span>
        </div>
      </div>

      <label className="mri-viewer__slider">
        <span>Slice position</span>
        <input
          type="range"
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

export function MRIViewer({
  caseId,
  completedRunId,
}: Props) {
  const [metadata, setMetadata] = useState<MRIViewerMetadata | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [modality, setModality] = useState<MRIModality>('t2f')
  const [overlayLayer, setOverlayLayer] =
    useState<MRIOverlayLayer | 'none'>('final')
  const [overlayOpacity, setOverlayOpacity] = useState(45)
  const [markerResult, setMarkerResult] =
    useState<ViewerRunMarkers | null>(null)
  const [markerFailure, setMarkerFailure] =
    useState<{ runId: string; message: string } | null>(null)

  const [indices, setIndices] = useState<SliceIndices>({
    axial: 0,
    coronal: 0,
    sagittal: 0,
  })

  useEffect(() => {
    let active = true

    getViewerMetadata(caseId)
      .then((result) => {
        if (!active) return

        setMetadata(result)
        setIndices({
          axial: result.planes.axial.center_index,
          coronal: result.planes.coronal.center_index,
          sagittal: result.planes.sagittal.center_index,
        })
      })
      .catch((error: unknown) => {
        if (!active) return

        setLoadError(
          error instanceof Error
            ? error.message
            : 'Unable to initialize MRI viewer.',
        )
      })

    return () => {
      active = false
    }
  }, [caseId])


  useEffect(() => {
    if (!completedRunId) return

    let active = true

    getViewerRunMarkers(caseId, completedRunId)
      .then((data) => {
        if (!active) return
        setMarkerResult(data)
        setMarkerFailure(null)
      })
      .catch((error: unknown) => {
        if (!active) return
        setMarkerFailure({
          runId: completedRunId,
          message: error instanceof Error
            ? error.message
            : 'Unable to retrieve scientific markers.',
        })
      })

    return () => {
      active = false
    }
  }, [caseId, completedRunId])

  const displayedMarkers =
    markerResult?.case_id === caseId &&
    markerResult.run_id === completedRunId
      ? markerResult
      : null

  const displayedMarkerError =
    markerFailure?.runId === completedRunId
      ? markerFailure.message
      : null

  function jumpToMarker(marker: ViewerSpatialMarker | null) {
    if (!marker || !metadata) return

    const [x, y, z] = marker.ras_xyz
    const [sizeX, sizeY, sizeZ] = metadata.shape_ras_xyz

    if (
      ![x, y, z].every(Number.isInteger) ||
      x < 0 || x >= sizeX ||
      y < 0 || y >= sizeY ||
      z < 0 || z >= sizeZ
    ) {
      return
    }

    setIndices({
      sagittal: x,
      coronal: y,
      axial: z,
    })
  }

  function changeIndex(plane: MRIPlane, index: number) {
    setIndices((previous) => ({
      ...previous,
      [plane]: index,
    }))
  }

  return (
    <section className="mri-viewer">
      <header className="mri-viewer__header">
        <div>
          <span className="mri-viewer__eyebrow">
            NEUROPROMPT-3D / VOLUME EXPLORER
          </span>
          <h2>Brain MRI Imaging Workspace</h2>
          <p>Validated NIfTI data · Read-only anatomical inspection</p>
        </div>

        <div className="mri-viewer__header-control">
          <label htmlFor="mri-modality">MRI sequence</label>
          <select
            id="mri-modality"
            value={modality}
            onChange={(event) => {
              setModality(event.target.value as MRIModality)
            }}
          >
            {modalities.map((item) => (
              <option key={item.key} value={item.key}>
                {item.name}
              </option>
            ))}
          </select>
        </div>
      </header>

      {metadata && (
        <div className="mri-viewer__overlay-controls">
          <label htmlFor="mri-overlay-layer">
            Segmentation
            <select
              id="mri-overlay-layer"
              value={overlayLayer}
              disabled={!completedRunId}
              onChange={(event) => {
                setOverlayLayer(
                  event.target.value as MRIOverlayLayer | 'none',
                )
              }}
            >
              <option value="final">Final prediction</option>
              <option value="baseline">Baseline prediction</option>
              <option value="removed">Removed voxels</option>
              <option value="none">MRI only</option>
            </select>
          </label>

          <label htmlFor="mri-overlay-opacity">
            Opacity: {overlayOpacity}%
            <input
              id="mri-overlay-opacity"
              type="range"
              min={0}
              max={100}
              step={5}
              value={overlayOpacity}
              disabled={!completedRunId || overlayLayer === 'none'}
              onChange={(event) => {
                setOverlayOpacity(Number(event.target.value))
              }}
            />
          </label>

          <span
            className="mri-viewer__overlay-legend"
            data-layer={overlayLayer}
          >
            {!completedRunId
              ? 'Overlays require a completed inference run'
              : overlayLayer === 'none'
                ? 'MRI only'
                : overlayLayer === 'removed'
                  ? 'Removed predicted voxels — not verified errors'
                  : overlayLayer === 'baseline'
                    ? 'Original model prediction'
                    : 'Final refined prediction'}
          </span>
        </div>
      )}


      {completedRunId && metadata && (
        <section
          className="mri-viewer__scientific-controls"
          aria-label="Uncertainty-guided refinement inspection"
        >
          <div className="mri-viewer__scientific-heading">
            <strong>Uncertainty-guided inspection</strong>
            {displayedMarkers && (
              <span>
                {displayedMarkers.action ?? 'No recorded action'}
                {' · '}
                {displayedMarkers.gate_state ?? 'No recorded gate'}
              </span>
            )}
          </div>

          <div className="mri-viewer__scientific-actions">
            <button
              type="button"
              disabled={!displayedMarkers?.hotspot}
              onClick={() => {
                jumpToMarker(displayedMarkers?.hotspot ?? null)
              }}
            >
              <span className="mri-viewer__legend-dot mri-viewer__legend-dot--hotspot" />
              Jump to Hotspot
            </button>

            <button
              type="button"
              disabled={!displayedMarkers?.negative_prompt}
              onClick={() => {
                jumpToMarker(displayedMarkers?.negative_prompt ?? null)
              }}
            >
              <span className="mri-viewer__legend-dot mri-viewer__legend-dot--prompt" />
              Jump to Negative Prompt
            </button>

            <span className="mri-viewer__scientific-note">
              {displayedMarkerError
                ? displayedMarkerError
                : !displayedMarkers
                  ? 'Retrieving frozen run markers…'
                  : displayedMarkers.hotspot
                    ? `Hotspot variance: ${
                        displayedMarkers.hotspot_variance ?? 'unavailable'
                      }`
                    : 'No uncertainty hotspot recorded for this run'}
            </span>
          </div>
        </section>
      )}

      {loadError ? (
        <div className="mri-viewer__message" role="alert">
          {loadError}
        </div>
      ) : !metadata ? (
        <div className="mri-viewer__message" role="status">
          Loading actual MRI volume geometry…
        </div>
      ) : (
        <>
          <div className="mri-viewer__info">
            <span>
              RAS+ · {metadata.shape_ras_xyz.join(' × ')} voxels
            </span>
            <span>
              Spacing {metadata.voxel_spacing_ras_mm
                .map((value) => value.toFixed(2))
                .join(' × ')} mm
            </span>
            <span>
              XYZ {indices.sagittal}, {indices.coronal}, {indices.axial}
            </span>
          </div>

          <div className="mri-viewer__planes">
            {planes.map((plane) => (
              <SlicePane
                key={plane}
                caseId={caseId}
                modality={modality}
                plane={plane}
                metadata={metadata}
                indices={indices}
                completedRunId={completedRunId}
                overlayLayer={overlayLayer}
                overlayOpacity={overlayOpacity}
                markers={displayedMarkers}
                onIndexChange={changeIndex}
              />
            ))}
          </div>

          <footer className="mri-viewer__footer">
            Source: validated staged MRI · Percentile-scaled grayscale
            · RAS+ voxel orientation · Research use only
          </footer>
        </>
      )}
    </section>
  )
}
