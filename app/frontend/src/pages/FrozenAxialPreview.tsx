import { useEffect, useRef, useState } from 'react'

import { FrozenAxialComparison } from './FrozenAxialComparison'
import { FrozenThreePlanePreview } from './FrozenThreePlanePreview'

import {
  getFrozenAxialSliceUrl,
  getFrozenInitialPosition,
  getFrozenRemovedFocus,
  type FrozenInitialPosition,
  type FrozenAxialLayer,
  type FrozenMriModality,
} from '../api/frozenTest'

import {
  FrozenOverlayLegend,
  FrozenPositionReadout,
} from '../components/FrozenScientificGuide'

import './FrozenAxialPreview.css'

type ImageState = {
  url: string
  status: 'ready' | 'error'
}

export function FrozenAxialPreview({
  caseIndex,
  refinementApplied,
}: {
  caseIndex: number
  refinementApplied: boolean
}) {
  const [recommendation, setRecommendation] =
    useState<FrozenInitialPosition | null>(null)

  const [coordinates, setCoordinates] = useState<
    { x: number; y: number; z: number } | null
  >(null)

  const [positionError, setPositionError] = useState(false)
  const [positionAttempt, setPositionAttempt] = useState(0)

  const [removedFocusStatus, setRemovedFocusStatus] = useState<
    'idle' | 'loading' | 'ready' | 'unavailable' | 'error'
  >('idle')

  const [removedFocusCount, setRemovedFocusCount] = useState(0)
  const removedFocusController = useRef<AbortController | null>(null)
  const [comparison, setComparison] = useState(false)
  const [threePlane, setThreePlane] = useState(false)

  const [modality, setModality] =
    useState<FrozenMriModality>('t2f')

  const [layer, setLayer] =
    useState<FrozenAxialLayer>('removed')

  const [retry, setRetry] = useState(0)

  const [imageState, setImageState] =
    useState<ImageState | null>(null)

  useEffect(() => {
    const controller = new AbortController()

    getFrozenInitialPosition(caseIndex, controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return

        setRecommendation(result)
        setCoordinates({ ...result.coordinates })
        setPositionError(false)
      })
      .catch(() => {
        if (controller.signal.aborted) return
        setPositionError(true)
      })

    return () => controller.abort()
  }, [caseIndex, positionAttempt])

  useEffect(() => {
    return () => removedFocusController.current?.abort()
  }, [])

  async function focusOnRemovedRegion() {
    if (!refinementApplied || removedFocusStatus === 'loading') {
      return
    }

    removedFocusController.current?.abort()
    const controller = new AbortController()
    removedFocusController.current = controller

    setRemovedFocusStatus('loading')

    try {
      const result = await getFrozenRemovedFocus(
        caseIndex,
        controller.signal,
      )

      if (controller.signal.aborted) return

      if (!result.available || result.coordinates === null) {
        setRemovedFocusStatus('unavailable')
        return
      }

      setCoordinates({ ...result.coordinates })
      setRemovedFocusCount(result.removed_voxels)
      setLayer('removed')
      setComparison(false)
      setThreePlane(true)
      setRemovedFocusStatus('ready')
    } catch {
      if (!controller.signal.aborted) {
        setRemovedFocusStatus('error')
      }
    } finally {
      if (removedFocusController.current === controller) {
        removedFocusController.current = null
      }
    }
  }

  function retryInitialPosition() {
    setRecommendation(null)
    setCoordinates(null)
    setPositionError(false)
    setPositionAttempt((current) => current + 1)
  }

  if (
    recommendation === null ||
    coordinates === null ||
    recommendation.case_index !== caseIndex
  ) {
    return (
      <section
        className="frozen-axial-preview"
        aria-label="Original frozen MRI prediction evidence"
      >
        {positionError ? (
          <div role="alert">
            <p>
              The verified initial MRI position is unavailable.
              All MRI viewing modes are withheld until verification succeeds.
            </p>
            <button type="button" onClick={retryInitialPosition}>
              Retry verified position
            </button>
          </div>
        ) : (
          <p role="status">
            Authenticating the frozen MRI position…
          </p>
        )}
      </section>
    )
  }

  const z = coordinates.z

  const baseUrl = getFrozenAxialSliceUrl(
    caseIndex,
    z,
    modality,
    layer,
  )

  const imageUrl = `${baseUrl}&preview_retry=${retry}`

  const status = imageState?.url === imageUrl
    ? imageState.status
    : 'loading'

  return (
    <section
      className="frozen-axial-preview"
      aria-label="Original frozen MRI prediction evidence"
    >
      <div className="frozen-axial-preview__heading">
        <div>
          <h4>Original MRI prediction evidence</h4>
          <p>
            Authenticated saved predictions · {threePlane ? 'Three-plane view' : 'Axial view'} · Read only
          </p>
        </div>
        <span>Case {String(caseIndex).padStart(3, '0')}</span>
      </div>

      <FrozenPositionReadout
        current={coordinates}
        recommended={recommendation.coordinates}
      />

      <div
        className="frozen-axial-preview__focus"
        role="group"
        aria-label="Frozen prediction navigation"
      >
        <button
          type="button"
          disabled={
            !refinementApplied ||
            removedFocusStatus === 'loading' ||
            removedFocusStatus === 'unavailable'
          }
          onClick={() => {
            void focusOnRemovedRegion()
          }}
        >
          {removedFocusStatus === 'loading'
            ? 'Authenticating removed region…'
            : 'Focus on removed region'}
        </button>

        {!refinementApplied && (
          <p>
            Baseline retained. No refinement removal region
            is available for this case.
          </p>
        )}

        {removedFocusStatus === 'ready' && (
          <p role="status">
            Authenticated focus selected from {removedFocusCount}
            {' '}removed voxels. Three-plane MRI and the Removed
            Voxels overlay are now selected.
          </p>
        )}

        {removedFocusStatus === 'unavailable' && (
          <p role="status">
            The verified frozen removal mask contains no
            removed voxels.
          </p>
        )}

        {removedFocusStatus === 'error' && (
          <p role="alert">
            The verified removed-region focus is unavailable.
            Your existing MRI coordinates were preserved.
            You can retry the focus request.
          </p>
        )}
      </div>

      <FrozenOverlayLegend selected={layer} />

      <div className="frozen-axial-preview__controls">
        <label>
          MRI sequence
          <select
            value={modality}
            onChange={(event) =>
              setModality(event.target.value as FrozenMriModality)
            }
          >
            <option value="t2f">T2-FLAIR</option>
            <option value="t1c">T1 contrast-enhanced</option>
            <option value="t1n">T1 native</option>
            <option value="t2w">T2-weighted</option>
          </select>
        </label>

        <label style={{ display: comparison ? 'none' : undefined }}>
          Display overlay
          <select
            value={layer}
            onChange={(event) =>
              setLayer(event.target.value as FrozenAxialLayer)
            }
          >
            <option value="none">MRI only</option>
            <option value="baseline">Baseline prediction</option>
            <option value="final">Final prediction</option>
            <option value="removed">Removed voxels</option>
            <option value="support">Refinement support</option>
          </select>
        </label>
      </div>

      <div
        className="frozen-axial-preview__mode"
        role="group"
        aria-label="MRI viewing mode"
      >
        <button
          type="button"
          aria-pressed={!comparison && !threePlane}
          onClick={() => {
            setComparison(false)
            setThreePlane(false)
            setImageState(null)
          }}
        >
          Single overlay
        </button>
        <button
          type="button"
          aria-pressed={comparison && !threePlane}
          onClick={() => {
              setComparison(true)
              setThreePlane(false)
            }}
        >
          Baseline vs Final
        </button>
        <button
          type="button"
          aria-pressed={threePlane}
          onClick={() => {
            setComparison(false)
            setThreePlane(true)
            setImageState(null)
          }}
        >
          Three-plane MRI
        </button>
      </div>

      {threePlane ? (
        <FrozenThreePlanePreview
          key={caseIndex}
          caseIndex={caseIndex}
          modality={modality}
          layer={layer}
          coordinates={coordinates}
          recommendation={recommendation}
          onCoordinatesChange={setCoordinates}
        />
      ) : comparison ? (
        <FrozenAxialComparison
          key={`${caseIndex}:${z}:${modality}`}
          caseIndex={caseIndex}
          z={z}
          modality={modality}
        />
      ) : (
        <>
      <div className="frozen-axial-preview__viewport">
        <span className="frozen-axial-preview__orientation">
          Anterior (A)
        </span>

        <div className="frozen-axial-preview__image-row">
          <span title="Patient's right">R</span>

          <img
            key={imageUrl}
            src={imageUrl}
            alt={`Case ${caseIndex}, axial slice ${z}, ${modality}, ${layer} overlay`}
            width={182}
            height={218}
            onLoad={() =>
              setImageState({
                url: imageUrl,
                status: 'ready',
              })
            }
            onError={() =>
              setImageState({
                url: imageUrl,
                status: 'error',
              })
            }
            style={{
              visibility:
                status === 'ready' ? 'visible' : 'hidden',
            }}
          />

          <span title="Patient's left">L</span>
        </div>

        <span className="frozen-axial-preview__orientation">
          Posterior (P)
        </span>
      </div>

      {status === 'loading' && (
        <p role="status">
          Loading authenticated MRI slice…
        </p>
      )}

      {status === 'error' && (
        <div role="alert">
          <p>
            The verified MRI preview is unavailable.
            The original evidence has not been replaced.
          </p>
          <button
            type="button"
            onClick={() => setRetry((value) => value + 1)}
          >
            Retry MRI preview
          </button>
        </div>
      )}

        </>
      )}

      {!threePlane && (
      <div className="frozen-axial-preview__navigation">
        <label htmlFor="frozen-axial-slice">
          Axial slice Z = {z} / 181
        </label>

        <input
          id="frozen-axial-slice"
          type="range"
          min={0}
          max={181}
          step={1}
          value={z}
          onChange={(event) =>
            setCoordinates({
              ...coordinates,
              z: Number(event.target.value),
            })
          }
        />

        <div className="frozen-axial-preview__steps">
          <button
            type="button"
            disabled={z === 0}
            onClick={() =>
              setCoordinates({
                ...coordinates,
                z: Math.max(0, z - 1),
              })
            }
          >
            Previous slice
          </button>

          <button
            type="button"
            disabled={z === 181}
            onClick={() =>
              setCoordinates({
                ...coordinates,
                z: Math.min(181, z + 1),
              })
            }
          >
            Next slice
          </button>
        </div>
      </div>
      )}

      <p className="frozen-axial-preview__note">
        Orientation labels refer to the patient.
        The image and overlays use the original frozen voxel grid.
        No inference or ground-truth comparison is performed here.
      </p>
    </section>
  )
}
