import type { FrozenAxialLayer } from '../api/frozenTest'
import './FrozenScientificGuide.css'

type NativeXYZ = Readonly<{
  x: number
  y: number
  z: number
}>

const OVERLAYS = [
  {
    id: 'baseline',
    title: 'Baseline',
    description: 'Saved prediction before the frozen policy.',
  },
  {
    id: 'final',
    title: 'Final',
    description: 'Saved prediction after the frozen policy.',
  },
  {
    id: 'removed',
    title: 'Removed voxels',
    description: 'Baseline voxels absent from the final mask.',
  },
  {
    id: 'support',
    title: 'Refinement support',
    description: 'Localized region permitting the frozen edit.',
  },
] as const

export function FrozenMetricGuide() {
  return (
    <details className="frozen-science-guide">
      <summary>How to interpret these evaluation metrics</summary>

      <div className="frozen-science-guide__content">
        <p>
          These measurements come from the completed,
          frozen held-out evaluation. No predictions,
          ground-truth comparisons, or measurements
          are recomputed by this page.
        </p>

        <dl>
          <div>
            <dt>Dice coefficient</dt>
            <dd>
              Segmentation overlap with the reference mask.
              Higher values indicate better overlap.
            </dd>
          </div>

          <div>
            <dt>Intersection over Union (IoU)</dt>
            <dd>
              Intersection of predicted and reference
              foreground voxels divided by their union.
              Higher values are better.
            </dd>
          </div>

          <div>
            <dt>HD95</dt>
            <dd>
              A 95th-percentile boundary-distance measure,
              reported in millimeters. Lower values are better.
            </dd>
          </div>
        </dl>

        <p>
          <strong>Paired Δ = final − baseline.</strong>
          {' '}For Dice and IoU, positive changes are favorable.
          For HD95, negative changes are favorable.
        </p>

        <p>
          The cohort summaries describe the frozen test set.
          A favorable result on one metric or case does not
          establish overall improvement, statistical
          significance, or clinical benefit.
        </p>
      </div>
    </details>
  )
}

export function FrozenOverlayLegend({
  selected,
}: {
  selected: FrozenAxialLayer
}) {
  return (
    <section
      className="frozen-overlay-legend"
      aria-label="MRI prediction overlay legend"
    >
      <div className="frozen-overlay-legend__heading">
        <strong>MRI overlay legend</strong>
        <span>
          Colors represent saved prediction evidence,
          not ground truth.
        </span>
      </div>

      <ul className="frozen-overlay-legend__items">
        {OVERLAYS.map((item) => (
          <li
            key={item.id}
            data-selected={selected === item.id ? 'true' : 'false'}
          >
            <span
              className={`frozen-overlay-legend__swatch frozen-overlay-legend__swatch--${item.id}`}
              aria-hidden="true"
            />

            <span>
              <strong>{item.title}</strong>
              <small>{item.description}</small>
            </span>
          </li>
        ))}
      </ul>

      <p className="frozen-overlay-legend__note">
        The highlighted legend entry is the selected overlay.
        A view without an overlay shows original MRI intensity only.
        Removed voxels are changes made by the frozen policy;
        they are not automatically proven false positives.
      </p>
    </section>
  )
}

export function FrozenPositionReadout({
  current,
  recommended,
}: {
  current: NativeXYZ
  recommended: NativeXYZ
}) {
  const atRecommendation =
    current.x === recommended.x &&
    current.y === recommended.y &&
    current.z === recommended.z

  return (
    <section
      className="frozen-position-readout"
      aria-label="Current and recommended MRI coordinates"
    >
      <div>
        <span>Current native XYZ</span>
        <strong>
          ({current.x}, {current.y}, {current.z})
        </strong>
      </div>

      <div>
        <span>Recommended native XYZ</span>
        <strong>
          ({recommended.x}, {recommended.y}, {recommended.z})
        </strong>
      </div>

      <p>
        {atRecommendation
          ? 'Currently at the saved prediction-focus coordinate.'
          : 'Current position differs from the saved prediction focus. This may result from manual navigation or a removed-region focus request.'}
        {' '}Coordinates are display-only, not clinical localization.
      </p>
    </section>
  )
}
