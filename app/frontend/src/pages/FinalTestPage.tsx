import { useEffect, useState } from 'react'

import { ModeBadge } from '../components/ModeBadge'
import './FrozenTestControls.css'
import { FrozenCaseExplorer } from './FrozenCaseExplorer'
import { FrozenMetricGuide } from '../components/FrozenScientificGuide'
import {
  getFrozenTestSummary,
  type FrozenTestSummary,
} from '../api/frozenTest'

type LoadState =
  | { status: 'loading' }
  | { status: 'ready'; data: FrozenTestSummary }
  | { status: 'error'; message: string }

type MetricProps = {
  name: string
  baseline: number
  final: number
  delta: number
  unit?: string
  lowerIsBetter?: boolean
}

function MetricTile({
  name,
  baseline,
  final,
  delta,
  unit = '',
  lowerIsBetter = false,
}: MetricProps) {
  const decimals = unit ? 2 : 4
  const deltaDecimals = unit ? 3 : 5
  const favorable = lowerIsBetter
    ? delta < 0
    : delta > 0

  const signedDelta = delta > 0
    ? `+${delta.toFixed(deltaDecimals)}`
    : delta.toFixed(deltaDecimals)

  return (
    <div
      style={{
        padding: 16,
        border: '1px solid var(--border)',
        borderRadius: 8,
      }}
    >
      <span className="eyebrow">{name}</span>

      <strong
        style={{
          display: 'block',
          marginTop: 12,
          fontSize: 27,
          fontWeight: 600,
        }}
      >
        {final.toFixed(decimals)}{unit}
      </strong>

      <p>
        Baseline: {baseline.toFixed(decimals)}{unit}
      </p>

      <p>
        Mean paired Δ: {signedDelta}{unit}
      </p>

      <small>
        {favorable ? 'Favorable' : delta === 0
          ? 'No mean change' : 'Unfavorable'}
        {' · '}
        {lowerIsBetter ? 'lower is better' : 'higher is better'}
      </small>
    </div>
  )
}

export function FinalTestPage() {
  const [state, setState] = useState<LoadState>({
    status: 'loading',
  })
  const [requestNumber, setRequestNumber] = useState(0)

  useEffect(() => {
    const controller = new AbortController()

    getFrozenTestSummary(controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) {
          setState({ status: 'ready', data })
        }
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setState({
            status: 'error',
            message: error instanceof Error
              ? error.message
              : 'Unable to verify frozen evidence.',
          })
        }
      })

    return () => controller.abort()
  }, [requestNumber])

  const summary = state.status === 'ready'
    ? state.data
    : null

  const metrics = summary?.overall_summary

  function retry() {
    setState({ status: 'loading' })
    setRequestNumber((previous) => previous + 1)
  }

  return (
    <main className="workspace frozen-test-page">
      <section className="workspace-heading">
        <div>
          <ModeBadge tone="frozen">
            FROZEN RESEARCH RESULT
          </ModeBadge>

          <h2>Final-Test Explorer</h2>

          <p>
            Read-only evaluation of held-out MRI test cases.
            All displayed measurements require
            verified frozen evidence.
          </p>
        </div>

        <div className="result-identity result-identity--frozen">
          Held-out Test Evaluation · Read Only
        </div>
      </section>

      <section className="workspace-grid workspace-grid--explorer">
        <article className="panel metric-summary">
          <span className="eyebrow">Frozen cohort</span>

          {summary ? (
            <>
              <strong className="large-number">
                {summary.case_count}
              </strong>
              <span>final-test cases</span>
            </>
          ) : (
            <p>Awaiting verified evidence</p>
          )}
        </article>

        <article className="panel">
          <div className="panel__heading">
            <div>
              <span className="eyebrow">
                Scientific boundary
              </span>
              <h3>Read-only evidence</h3>
            </div>
          </div>

          <div className="boundary-list">
            <span>No inference reruns</span>
            <span>No threshold changes</span>
            <span>No prompt changes</span>
            <span>No refinement model reruns</span>
            <span>No frozen-output rewrites</span>
          </div>
        </article>

        <article className="panel panel--wide">
          <div className="panel__heading">
            <div>
              <span className="eyebrow">
                Verified final-test results
              </span>
              <h3>Baseline vs final policy</h3>
            </div>

            <span className="panel-status">
              {summary ? 'Verified source' : 'Verification pending'}
            </span>
          </div>

          {state.status === 'loading' && (
            <p role="status">
              Loading and verifying frozen evaluation…
            </p>
          )}

          {state.status === 'error' && (
            <div role="alert">
              <p>
                Verified frozen results are unavailable.
                No fallback metrics are displayed.
              </p>

              <p>{state.message}</p>

              <button type="button" onClick={retry}>
                Retry verification
              </button>
            </div>
          )}

          {summary && metrics && (
            <>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns:
                    'repeat(auto-fit, minmax(190px, 1fr))',
                  gap: 12,
                  marginTop: 16,
                }}
              >
                <MetricTile
                  name="Mean Dice"
                  baseline={metrics.baseline_dice_mean}
                  final={metrics.final_dice_mean}
                  delta={metrics.delta_dice_mean}
                />

                <MetricTile
                  name="Mean IoU"
                  baseline={metrics.baseline_iou_mean}
                  final={metrics.final_iou_mean}
                  delta={metrics.delta_iou_mean}
                />

                <MetricTile
                  name="Mean HD95"
                  baseline={metrics.baseline_hd95_mm_mean}
                  final={metrics.final_hd95_mm_mean}
                  delta={metrics.delta_hd95_mm_mean}
                  unit=" mm"
                  lowerIsBetter
                />
              </div>

              <div style={{ marginTop: 24 }}>
                <FrozenMetricGuide />
                <h3>Automatic refinement decisions</h3>

                <p>
                  Refinement applied:{' '}
                  <strong>
                    {summary.action_counts.APPLY_LOCAL_FP}
                  </strong>
                  {' '}cases
                </p>

                <p>
                  Abstained and retained baseline:{' '}
                  <strong>
                    {summary.action_counts.ABSTAIN_BASELINE}
                  </strong>
                  {' '}cases
                </p>
              </div>

              <p style={{ marginTop: 20 }}>
                These are descriptive held-out measurements,
                not evidence of clinical benefit or
                statistical significance.
              </p>

              <small
                style={{
                  display: 'block',
                  marginTop: 16,
                  overflowWrap: 'anywhere',
                }}
              >
                Frozen source SHA-256:{' '}
                <code>{summary.evaluation_sha256}</code>
              </small>
            </>
          )}
        </article>
        {summary && (
          <FrozenCaseExplorer
            expectedSha={summary.evaluation_sha256}
          />
        )}
      </section>
    </main>
  )
}
