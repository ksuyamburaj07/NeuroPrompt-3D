import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from 'react'

import {
  getRun,
  type RunView,
} from '../api/runs'

import {
  isValidResearchRunId,
  normalizeResearchRunId,
} from '../utils/researchRunIdentity'

import { RecentResearchRuns } from './RecentResearchRuns'

import './ResearchRunInspector.css'

type Inspection =
  | { kind: 'idle' }
  | { kind: 'loading'; runId: string }
  | { kind: 'error'; message: string }
  | { kind: 'ready'; run: RunView }


function displayValue(
  value: string | number | boolean | null | undefined,
): string {
  if (value === null || value === undefined || value === '') {
    return 'Not recorded'
  }

  if (typeof value === 'boolean') {
    return value ? 'Yes' : 'No'
  }

  return String(value)
}

function displayCoordinates(
  value: number[] | null | undefined,
): string {
  if (
    !Array.isArray(value) ||
    value.length !== 3 ||
    !value.every(Number.isInteger)
  ) {
    return 'Not recorded'
  }

  return `(${value.join(', ')})`
}

function ResearchField({
  label,
  value,
}: {
  label: string
  value: string
}) {
  return (
    <div className="research-run-inspector__field">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  )
}

export function ResearchRunInspector() {
  const [query, setQuery] = useState('')
  const [inspection, setInspection] = useState<Inspection>({
    kind: 'idle',
  })

  const sequence = useRef(0)

  useEffect(() => {
    return () => {
      sequence.current += 1
    }
  }, [])

  async function inspectRunId(value: string) {
    const runId = normalizeResearchRunId(value)
    const requestNumber = ++sequence.current

    setQuery(runId)

    if (!isValidResearchRunId(runId)) {
      setInspection({
        kind: 'error',
        message:
          'Enter run_ followed by 32 lowercase hexadecimal characters.',
      })
      return
    }

    setInspection({
      kind: 'loading',
      runId,
    })

    try {
      const run = await getRun(runId)

      if (sequence.current !== requestNumber) {
        return
      }

      if (run.run_id !== runId) {
        throw new Error(
          'The backend returned a different run identifier.',
        )
      }

      setInspection({
        kind: 'ready',
        run,
      })
    } catch (error) {
      if (sequence.current !== requestNumber) {
        return
      }

      setInspection({
        kind: 'error',
        message:
          error instanceof Error
            ? error.message
            : 'Unable to inspect this saved run.',
      })
    }
  }

  function inspect(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    void inspectRunId(query)
  }

  const run = inspection.kind === 'ready'
    ? inspection.run
    : null

  return (
    <>
      <RecentResearchRuns
        onInspect={(runId) => {
          void inspectRunId(runId)
        }}
      />

      <section
        className="panel research-run-inspector"
      aria-labelledby="research-run-inspector-heading"
    >
      <div className="panel__heading">
        <div>
          <span className="eyebrow">
            Existing execution evidence
          </span>
          <h3 id="research-run-inspector-heading">
            Research Run Inspector
          </h3>
        </div>

        <span className="panel-status">
          Read only
        </span>
      </div>

      <p className="research-run-inspector__introduction">
        Inspect a previously recorded live or development
        execution run. This does not launch inference,
        alter predictions, or access the frozen final-test
        cohort.
      </p>

      <form
        className="research-run-inspector__form"
        onSubmit={inspect}
      >
        <label htmlFor="research-run-identifier">
          Existing run identifier
        </label>

        <div className="research-run-inspector__form-row">
          <input
            id="research-run-identifier"
            value={query}
            type="text"
            placeholder="run_ followed by 32 hex characters"
            autoComplete="off"
            spellCheck={false}
            aria-describedby="research-run-identifier-hint"
            onChange={(event) => {
              sequence.current += 1
              setQuery(event.target.value)
              setInspection({ kind: 'idle' })
            }}
          />

          <button
            type="submit"
            disabled={inspection.kind === 'loading'}
          >
            {inspection.kind === 'loading'
              ? 'Inspecting…'
              : 'Inspect saved run'}
          </button>
        </div>

        <p id="research-run-identifier-hint">
          Only existing saved run IDs are accepted.
          No execution is started.
        </p>
      </form>

      {inspection.kind === 'idle' && (
        <p className="research-run-inspector__message">
          Enter an existing run identifier to inspect
          its persisted execution metadata.
        </p>
      )}

      {inspection.kind === 'loading' && (
        <p role="status">
          Authenticating saved metadata for
          {' '}{inspection.runId}…
        </p>
      )}

      {inspection.kind === 'error' && (
        <p
          className="research-run-inspector__error"
          role="alert"
        >
          {inspection.message}
        </p>
      )}

      {run && (
        <section
          className="research-run-inspector__result"
          aria-label="Saved research run details"
        >
          <div className="research-run-inspector__result-heading">
            <strong>{run.run_id}</strong>
            <span data-run-status={run.status}>
              {run.status}
            </span>
          </div>

          <label htmlFor="research-run-progress">
            Saved execution progress
          </label>
          <progress
            id="research-run-progress"
            value={run.progress}
            max={1}
          />
          <p>
            {Math.round(run.progress * 100)}% recorded progress
            · Stage: {run.stage}
          </p>

          <dl className="research-run-inspector__fields">
            <ResearchField
              label="Case identifier"
              value={run.case_id}
            />
            <ResearchField
              label="Execution status"
              value={run.status}
            />
            <ResearchField
              label="Created"
              value={run.created_at}
            />
            <ResearchField
              label="Last updated"
              value={run.updated_at}
            />
            <ResearchField
              label="Execution device"
              value={displayValue(run.execution_device)}
            />
            <ResearchField
              label="Frozen variance threshold"
              value={String(run.frozen_variance_threshold)}
            />
            <ResearchField
              label="MC case seed"
              value={displayValue(run.mc_case_seed)}
            />
            <ResearchField
              label="Policy action"
              value={displayValue(run.action)}
            />
            <ResearchField
              label="Gate state"
              value={displayValue(run.gate_state)}
            />
            <ResearchField
              label="Hotspot variance"
              value={displayValue(run.hotspot_variance)}
            />
            <ResearchField
              label="Hotspot (ZYX)"
              value={displayCoordinates(run.hotspot_zyx)}
            />
            <ResearchField
              label="FP prompt (ZYX)"
              value={displayCoordinates(run.fp_prompt_zyx)}
            />
            <ResearchField
              label="FP prompt (model XYZ)"
              value={displayCoordinates(run.fp_prompt_model_xyz)}
            />
            <ResearchField
              label="SAM used"
              value={displayValue(run.sam_used)}
            />
            <ResearchField
              label="SAM refinement skipped"
              value={displayValue(run.sam_refinement_skipped)}
            />
            <ResearchField
              label="Abstention condition"
              value={displayValue(
                run.semantic_abstention_condition,
              )}
            />
          </dl>

          {run.error && (
            <p
              className="research-run-inspector__error"
              role="alert"
            >
              Recorded execution failure:
              {' '}{run.error.code} — {run.error.message}
            </p>
          )}

          <div className="research-run-inspector__artifacts">
            <h4>Recorded artifact names</h4>

            {run.available_artifacts?.length ? (
              <ul>
                {run.available_artifacts.map((name) => (
                  <li key={name}>
                    <code>{name}</code>
                  </li>
                ))}
              </ul>
            ) : (
              <p>No artifact names recorded for this run.</p>
            )}

            <p>
              Artifact names are shown for inspection only.
              Downloads will be implemented separately,
              using the existing allowlisted endpoint.
            </p>
          </div>

          <p className="research-run-inspector__boundary">
            This is persisted development-run metadata,
            not a held-out final-test measurement.
            Fields marked Not recorded are unavailable
            in this saved run.
          </p>
        </section>
      )}
      </section>
    </>
  )
}
