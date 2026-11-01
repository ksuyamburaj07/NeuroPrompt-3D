import { useState } from 'react'

import {
  getFrozenAxialSliceUrl,
  type FrozenMriModality,
} from '../api/frozenTest'

import './FrozenAxialComparison.css'

type Panel = 'baseline' | 'final'
type LoadStatus = 'loading' | 'ready' | 'error'

const PANELS: {
  key: Panel
  title: string
  description: string
}[] = [
  {
    key: 'baseline',
    title: 'Baseline prediction',
    description: 'Before automatic refinement',
  },
  {
    key: 'final',
    title: 'Final prediction',
    description: 'After the frozen refinement policy',
  },
]

export function FrozenAxialComparison({
  caseIndex,
  z,
  modality,
}: {
  caseIndex: number
  z: number
  modality: FrozenMriModality
}) {
  const [statuses, setStatuses] = useState<
    Record<Panel, LoadStatus>
  >({
    baseline: 'loading',
    final: 'loading',
  })

  const [attempt, setAttempt] = useState(0)

  const bothReady =
    statuses.baseline === 'ready' &&
    statuses.final === 'ready'

  const failed =
    statuses.baseline === 'error' ||
    statuses.final === 'error'

  function mark(panel: Panel, status: LoadStatus) {
    setStatuses((previous) => ({
      ...previous,
      [panel]: status,
    }))
  }

  function retry() {
    setStatuses({
      baseline: 'loading',
      final: 'loading',
    })
    setAttempt((value) => value + 1)
  }

  return (
    <section
      className="frozen-comparison"
      aria-label="Synchronized baseline and final MRI comparison"
    >
      <div className="frozen-comparison__heading">
        <strong>Baseline vs Final</strong>
        <span>
          Case {String(caseIndex).padStart(3, '0')}
          {' · '}Axial Z={z}
        </span>
      </div>

      <p>
        Both panels use the same authenticated MRI sequence
        and voxel coordinates.
      </p>

      <div className="frozen-comparison__grid">
        {PANELS.map((panel) => {
          const imageUrl = (
            getFrozenAxialSliceUrl(
              caseIndex,
              z,
              modality,
              panel.key,
            ) + `&comparison_attempt=${attempt}`
          )

          return (
            <section
              key={panel.key}
              className="frozen-comparison__panel"
            >
              <strong>{panel.title}</strong>
              <span>{panel.description}</span>

              <div className="frozen-comparison__orientation">
                Anterior (A)
              </div>

              <div className="frozen-comparison__image-row">
                <span>R</span>

                <img
                  key={imageUrl}
                  src={imageUrl}
                  width={182}
                  height={218}
                  decoding="async"
                  alt={`${panel.title}, Case ${caseIndex}, axial Z=${z}, ${modality}`}
                  style={{
                    visibility:
                      bothReady ? 'visible' : 'hidden',
                  }}
                  onLoad={() => mark(panel.key, 'ready')}
                  onError={() => mark(panel.key, 'error')}
                />

                <span>L</span>
              </div>

              <div className="frozen-comparison__orientation">
                Posterior (P)
              </div>
            </section>
          )
        })}
      </div>

      {failed ? (
        <div role="alert">
          <p>
            The verified comparison is unavailable.
            Both panels are hidden until the evidence
            can be loaded again.
          </p>
          <button type="button" onClick={retry}>
            Retry comparison
          </button>
        </div>
      ) : !bothReady ? (
        <p role="status">
          Loading both authenticated MRI images…
        </p>
      ) : (
        <p role="status">
          Both frozen prediction views loaded.
        </p>
      )}

      <p className="frozen-comparison__note">
        Cyan represents the original baseline prediction;
        purple represents the final prediction.
        For an exact voxel-removal visualization, use
        Removed voxels in Single overlay mode.
      </p>
    </section>
  )
}
