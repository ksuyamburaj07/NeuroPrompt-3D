import { useEffect, useState } from 'react'
import {
  getViewerMetadata,
  viewerSliceUrl,
  type MRIModality,
  type MRIPlane,
  type MRIViewerMetadata,
} from '../api/viewer'
import './MRIViewer.css'

type Props = {
  caseId: string
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
  return Math.max(0, Math.min(100, 100 * index / (count - 1)))
}

function SlicePane({
  caseId,
  modality,
  plane,
  metadata,
  indices,
  onIndexChange,
}: {
  caseId: string
  modality: MRIModality
  plane: MRIPlane
  metadata: MRIViewerMetadata
  indices: SliceIndices
  onIndexChange: (plane: MRIPlane, index: number) => void
}) {
  const [failedUrl, setFailedUrl] = useState<string | null>(null)

  const info = metadata.planes[plane]
  const index = indices[plane]
  const [sizeX, sizeY, sizeZ] = metadata.shape_ras_xyz

  const url = viewerSliceUrl(caseId, modality, plane, index)

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

export function MRIViewer({ caseId }: Props) {
  const [metadata, setMetadata] = useState<MRIViewerMetadata | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [modality, setModality] = useState<MRIModality>('t2f')
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
