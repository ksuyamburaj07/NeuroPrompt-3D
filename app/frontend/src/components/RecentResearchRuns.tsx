import {
  useEffect,
  useRef,
  useState,
} from 'react'

import { getRecentResearchRuns } from '../api/researchDiscovery'
import type { RecentResearchRun } from '../utils/recentResearchRuns'

import './RecentResearchRuns.css'

type ListingState =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'ready'; runs: RecentResearchRun[] }
  | { kind: 'error'; message: string }

type RecentResearchRunsProps = {
  onInspect: (runId: string) => void
}

function formatUpdatedAt(value: string): string {
  return new Date(value).toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
    timeZone: 'UTC',
  })
}

export function RecentResearchRuns({
  onInspect,
}: RecentResearchRunsProps) {
  const [listing, setListing] = useState<ListingState>({
    kind: 'idle',
  })

  const sequence = useRef(0)

  useEffect(() => {
    return () => {
      sequence.current += 1
    }
  }, [])

  async function loadRuns() {
    const requestNumber = ++sequence.current

    setListing({ kind: 'loading' })

    try {
      const runs = await getRecentResearchRuns(20)

      if (sequence.current !== requestNumber) {
        return
      }

      setListing({
        kind: 'ready',
        runs,
      })
    } catch (error) {
      if (sequence.current !== requestNumber) {
        return
      }

      setListing({
        kind: 'error',
        message:
          error instanceof Error
            ? error.message
            : 'Unable to load saved development runs.',
      })
    }
  }

  return (
    <section
      className="panel recent-research-runs"
      aria-labelledby="recent-research-runs-heading"
    >
      <div className="panel__heading">
        <div>
          <span className="eyebrow">
            Saved development executions
          </span>
          <h3 id="recent-research-runs-heading">
            Recent Research Runs
          </h3>
        </div>

        <span className="panel-status">
          Cloud saved runs
        </span>
      </div>

      <p className="recent-research-runs__introduction">
        Discover previously saved development executions,
        ordered by their last recorded update. Selecting a run
        opens the existing read-only Research Run Inspector.
      </p>

      <div className="recent-research-runs__actions">
        <button
          type="button"
          disabled={listing.kind === 'loading'}
          onClick={() => {
            void loadRuns()
          }}
        >
          {listing.kind === 'loading'
            ? 'Loading saved runs…'
            : listing.kind === 'idle'
              ? 'Load saved runs'
              : 'Refresh runs'}
        </button>
      </div>

      {listing.kind === 'idle' && (
        <p className="recent-research-runs__note">
          Discovery requires the local development feature to be
          enabled. Manual inspection by run ID remains available
          below.
        </p>
      )}

      {listing.kind === 'loading' && (
        <p role="status">
          Reading the saved development run inventory…
        </p>
      )}

      {listing.kind === 'error' && (
        <p
          className="recent-research-runs__error"
          role="alert"
        >
          {listing.message}
        </p>
      )}

      {listing.kind === 'ready' && (
        <>
          <p role="status">
            {listing.runs.length === 0
              ? 'No saved development runs were found.'
              : `${listing.runs.length} saved development runs found.`}
          </p>

          {listing.runs.length > 0 && (
            <ol className="recent-research-runs__list">
              {listing.runs.map((run, index) => (
                <li
                  className="recent-research-runs__item"
                  key={run.run_id}
                >
                  <div className="recent-research-runs__details">
                    <strong>
                      Run {index + 1}
                    </strong>

                    <code>{run.run_id}</code>

                    <span>
                      {run.status} · {run.stage} ·
                      {' '}{Math.round(run.progress * 100)}%
                    </span>

                    <span>
                      Updated (UTC): {formatUpdatedAt(run.updated_at)}
                    </span>
                  </div>

                  <button
                    type="button"
                    onClick={() => onInspect(run.run_id)}
                    aria-label={`Inspect run ${run.run_id}`}
                  >
                    Inspect
                  </button>
                </li>
              ))}
            </ol>
          )}
        </>
      )}

      <p className="recent-research-runs__boundary">
        Development-run history only. These records are not
        held-out final-test results. Loading the inventory
        does not start inference or download artifacts.
      </p>
    </section>
  )
}
