import { useEffect, useRef, useState } from 'react'

import { FrozenAxialPreview } from './FrozenAxialPreview'
import {
  FROZEN_CASE_PAGE_SIZE,
  adjacentFrozenCase,
  frozenPageForCase,
  parseFrozenCaseNumber,
} from '../utils/frozenCaseNavigator'

import {
  getFrozenTestCase,
  getFrozenTestCases,
  type FrozenAction,
  type FrozenCase,
  type FrozenCaseList,
} from '../api/frozenTest'

const PAGE_SIZE = FROZEN_CASE_PAGE_SIZE

type ListState =
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; data: FrozenCaseList }

type DetailState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'error'; message: string }
  | { status: 'ready'; data: FrozenCase }

function errorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : 'Unable to retrieve verified frozen evidence.'
}

function friendlyAction(action: FrozenAction): string {
  return action === 'APPLY_LOCAL_FP'
    ? 'Refinement applied'
    : 'Baseline retained'
}

function caseLabel(index: number): string {
  return `Case ${String(index).padStart(3, '0')}`
}

function metricOutcome(
  delta: number,
  higherIsBetter: boolean,
): string {
  if (Math.abs(delta) <= 1e-12) return 'Unchanged'

  return (higherIsBetter ? delta > 0 : delta < 0)
    ? 'Improved'
    : 'Worsened'
}

export function FrozenCaseExplorer({
  expectedSha,
}: {
  expectedSha: string
}) {
  const [action, setAction] = useState<FrozenAction | null>(null)
  const [page, setPage] = useState(0)
  const [retryCount, setRetryCount] = useState(0)
  const [detailRetry, setDetailRetry] = useState(0)
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null)
  const [list, setList] = useState<ListState>({ status: 'loading' })
  const [detail, setDetail] = useState<DetailState>({ status: 'idle' })
  const [jumpInput, setJumpInput] = useState('')
  const [jumpError, setJumpError] = useState<string | null>(null)
  const selectedHeadingRef = useRef<HTMLHeadingElement>(null)

  useEffect(() => {
    const controller = new AbortController()

    getFrozenTestCases(
      page * PAGE_SIZE,
      PAGE_SIZE,
      action,
      controller.signal,
    )
      .then((data) => {
        if (!controller.signal.aborted) {
          setList({ status: 'ready', data })
        }
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setList({
            status: 'error',
            message: errorMessage(error),
          })
        }
      })

    return () => controller.abort()
  }, [action, page, retryCount])

  useEffect(() => {
    if (selectedIndex === null) return

    const controller = new AbortController()

    getFrozenTestCase(selectedIndex, controller.signal)
      .then((response) => {
        if (response.evaluation_sha256 !== expectedSha) {
          throw new Error('Frozen evidence identity mismatch.')
        }

        if (!controller.signal.aborted) {
          setDetail({ status: 'ready', data: response.case })
        }
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setDetail({
            status: 'error',
            message: errorMessage(error),
          })
        }
      })

    return () => controller.abort()
  }, [selectedIndex, expectedSha, detailRetry])

  useEffect(() => {
    if (
      detail.status === 'ready' &&
      list.status === 'ready' &&
      detail.data.case_index === selectedIndex
    ) {
      selectedHeadingRef.current?.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      })
    }
  }, [detail, list, selectedIndex])

  function resetView() {
    setSelectedIndex(null)
    setDetail({ status: 'idle' })
    setList({ status: 'loading' })
    setJumpInput('')
    setJumpError(null)
  }

  function changeAction(value: string) {
    resetView()
    setPage(0)
    setAction(
      value === 'ALL' ? null : value as FrozenAction,
    )
  }

  function changePage(next: number) {
    resetView()
    setPage(next)
  }

  function selectCase(index: number) {
    setSelectedIndex(index)
    setDetail({ status: 'loading' })
    setJumpInput(String(index).padStart(3, '0'))
    setJumpError(null)
    if (selectedIndex === index) {
      setDetailRetry((n) => n + 1)
    }
  }

  function navigateToCase(index: number) {
    if (!Number.isInteger(index) || index < 1 || index > 129) {
      setJumpError('Enter a case number from 1 to 129.')
      return
    }

    const targetPage = frozenPageForCase(index)

    // Numeric navigation always uses the complete cohort.
    // Filtering the table does not restrict direct navigation.
    if (action !== null || page !== targetPage) {
      setAction(null)
      setPage(targetPage)
      setList({ status: 'loading' })
    }

    selectCase(index)
  }

  function submitCaseJump() {
    const index = parseFrozenCaseNumber(jumpInput)

    if (index === null) {
      setJumpError('Enter a whole-number case index from 1 to 129.')
      return
    }

    navigateToCase(index)
  }

  function navigateAdjacent(direction: -1 | 1) {
    if (selectedIndex === null) return

    const next = adjacentFrozenCase(selectedIndex, direction)

    if (next !== null) {
      navigateToCase(next)
    }
  }

  function retryList() {
    resetView()
    setRetryCount((n) => n + 1)
  }

  const listing = list.status === 'ready' ? list.data : null
  const selected = detail.status === 'ready' ? detail.data : null

  const metricRows = selected ? [
    {
      name: 'Dice',
      baseline: selected.baseline_dice,
      final: selected.final_dice,
      delta: selected.delta_dice,
      higherIsBetter: true,
    },
    {
      name: 'IoU',
      baseline: selected.baseline_iou,
      final: selected.final_iou,
      delta: selected.delta_iou,
      higherIsBetter: true,
    },
    {
      name: 'HD95 (mm)',
      baseline: selected.baseline_hd95_mm,
      final: selected.final_hd95_mm,
      delta: selected.delta_hd95_mm,
      higherIsBetter: false,
    },
  ] : []

  return (
    <article className="panel panel--wide">
      <div className="panel__heading">
        <div>
          <span className="eyebrow">Frozen case evidence</span>
          <h3>Explore individual test cases</h3>
        </div>
        <span className="panel-status">Read only</span>
      </div>

      <p>
        Recorded held-out test measurements. Select a case to
        inspect its original frozen results.
      </p>

      <div style={{ margin: '18px 0' }}>
        <label htmlFor="frozen-action-filter">
          Automatic decision:{' '}
        </label>
        <select
          id="frozen-action-filter"
          value={action ?? 'ALL'}
          onChange={(event) => changeAction(event.target.value)}
          style={{
            padding: 8,
            background: 'transparent',
            color: 'inherit',
            border: '1px solid var(--border)',
          }}
        >
          <option value="ALL">All cases</option>
          <option value="APPLY_LOCAL_FP">Refinement applied</option>
          <option value="ABSTAIN_BASELINE">Abstained</option>
        </select>
      </div>

      <section
        className="frozen-case-navigator"
        aria-label="Direct case navigator"
      >
        <div>
          <strong>Find any frozen test case</strong>
          <p className="frozen-case-navigator__hint">
            Jump directly to Case 001–129 without paging manually.
            Direct navigation uses the full cohort and clears
            the table filter when necessary.
          </p>
        </div>

        <form
          className="frozen-case-navigator__controls"
          onSubmit={(event) => {
            event.preventDefault()
            submitCaseJump()
          }}
        >
          <label htmlFor="frozen-case-jump">
            Case number
          </label>
          <input
            id="frozen-case-jump"
            type="text"
            inputMode="numeric"
            autoComplete="off"
            maxLength={3}
            value={jumpInput}
            placeholder="001–129"
            onChange={(event) => {
              setJumpInput(event.target.value)
              setJumpError(null)
            }}
          />
          <button type="submit">Go to case</button>
        </form>

        {jumpError && <p role="alert">{jumpError}</p>}

        <nav
          className="frozen-case-navigator__controls"
          aria-label="Sequential frozen case navigation"
        >
          <button
            type="button"
            disabled={selectedIndex === null || selectedIndex <= 1}
            onClick={() => navigateAdjacent(-1)}
          >
            Previous case
          </button>

          <span aria-live="polite">
            {selectedIndex === null
              ? 'No case selected'
              : `${caseLabel(selectedIndex)} / 129`}
          </span>

          <button
            type="button"
            disabled={selectedIndex === null || selectedIndex >= 129}
            onClick={() => navigateAdjacent(1)}
          >
            Next case
          </button>
        </nav>
      </section>

      {list.status === 'loading' && (
        <p role="status">Loading verified frozen cases…</p>
      )}

      {list.status === 'error' && (
        <div role="alert">
          <p>Case evidence unavailable: {list.message}</p>
          <button type="button" onClick={retryList}>
            Retry case list
          </button>
        </div>
      )}

      {listing && (
        <>
          <p role="status">
            {listing.total_matching} matching cases.
            {' '}Showing {listing.cases.length} on this page.
          </p>

          <div style={{ overflowX: 'auto' }}>
            <table style={{
              width: '100%',
              borderCollapse: 'collapse',
              textAlign: 'left',
            }}>
              <thead>
                <tr>
                  <th>Case</th>
                  <th>Decision</th>
                  <th>Δ Dice</th>
                  <th>Δ HD95</th>
                </tr>
              </thead>
              <tbody>
                {listing.cases.map((row) => (
                  <tr key={row.case_index}>
                    <td>
                      <button
                        type="button"
                        aria-pressed={selectedIndex === row.case_index}
                        onClick={() => selectCase(row.case_index)}
                      >
                        {caseLabel(row.case_index)}
                      </button>
                    </td>
                    <td>{friendlyAction(row.action)}</td>
                    <td>{row.delta_dice.toFixed(5)}</td>
                    <td>{row.delta_hd95_mm.toFixed(3)} mm</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{
            display: 'flex',
            gap: 12,
            alignItems: 'center',
            marginTop: 18,
          }}>
            <button
              type="button"
              disabled={page === 0}
              onClick={() => changePage(page - 1)}
            >
              Previous page
            </button>
            <span>
              Page {page + 1} of{' '}
              {Math.max(1, Math.ceil(
                listing.total_matching / PAGE_SIZE,
              ))}
            </span>
            <button
              type="button"
              disabled={
                (page + 1) * PAGE_SIZE >= listing.total_matching
              }
              onClick={() => changePage(page + 1)}
            >
              Next page
            </button>
          </div>
        </>
      )}

      <div style={{
        marginTop: 24,
        paddingTop: 18,
        borderTop: '1px solid var(--border)',
      }}>
        <h3
          ref={selectedHeadingRef}
          className="frozen-case-navigator__detail-heading"
        >
          Selected case
        </h3>
        <nav
          className="frozen-case-navigator__controls"
          aria-label="Selected case navigation"
        >
          <button
            type="button"
            disabled={selectedIndex === null || selectedIndex <= 1}
            onClick={() => navigateAdjacent(-1)}
          >
            Previous case
          </button>
          <span>
            {selectedIndex === null
              ? 'No case selected'
              : caseLabel(selectedIndex)}
          </span>
          <button
            type="button"
            disabled={selectedIndex === null || selectedIndex >= 129}
            onClick={() => navigateAdjacent(1)}
          >
            Next case
          </button>
        </nav>

        {detail.status === 'idle' && (
          <p>Select a case above to inspect its measurements.</p>
        )}

        {detail.status === 'loading' && (
          <p role="status">Verifying selected case…</p>
        )}

        {detail.status === 'error' && (
          <div role="alert">
            <p>{detail.message}</p>
            <button
              type="button"
              onClick={() => {
                setDetail({ status: 'loading' })
                setDetailRetry((n) => n + 1)
              }}
            >
              Retry case details
            </button>
          </div>
        )}

        {selected && (
          <>
            <p>
              <strong>{caseLabel(selected.case_index)}</strong>
              {' · '}{friendlyAction(selected.action)}
              {' · '}Decision status: {selected.gate_state === 'FP_ELIGIBLE' ? 'Eligible for refinement' : 'Refinement not applied'}
            </p>
            <p>
              Refinement model used: {selected.sam_used ? 'Yes' : 'No'}
              {' · '}Hotspot variance:{' '}
              {selected.hotspot_variance.toFixed(8)}
            </p>
            <div style={{ overflowX: 'auto' }}>
              <table style={{
                width: '100%',
                borderCollapse: 'collapse',
                textAlign: 'left',
              }}>
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Baseline</th>
                    <th>Final</th>
                    <th>Paired Δ</th>
                    <th>Outcome</th>
                  </tr>
                </thead>
                <tbody>
                  {metricRows.map((metric) => (
                    <tr key={metric.name}>
                      <td>{metric.name}</td>
                      <td>{metric.baseline.toFixed(4)}</td>
                      <td>{metric.final.toFixed(4)}</td>
                      <td>{metric.delta.toFixed(5)}</td>
                      <td>
                        {metricOutcome(
                          metric.delta,
                          metric.higherIsBetter,
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p>
              Recorded outcomes are descriptive.
              No prediction masks are regenerated here.
            </p>

            <FrozenAxialPreview
              key={selected.case_index}
              caseIndex={selected.case_index}
              refinementApplied={selected.action === 'APPLY_LOCAL_FP'}
            />

          </>
        )}
      </div>
    </article>
  )
}
