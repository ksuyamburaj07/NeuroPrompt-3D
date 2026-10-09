import { MRIViewer } from './MRIViewer'
import { useEffect, useRef, useState } from 'react'
import {
  artifactUrl,
  createRun,
  executeRun,
  getRun,
  type RunStage,
  type RunView,
} from '../api/runs'
import './ProcessPage.css'
import {
  clearPendingRunCreation,
  getOrCreateRunCreationKey,
  saveRunRecovery,
} from '../api/runRecovery'

type Props = {
  caseId: string
  existingRunId: string | null
  onRunCreated: (runId: string) => void
  onBack: () => void
}

const stageLabels: Record<RunStage, string> = {
  queued: 'Awaiting execution',
  validating: 'Verifying case integrity',
  loading_models: 'Loading frozen models',
  automatic_pipeline: 'Running automatic pipeline',
  preprocessing: 'Preparing MRI tensors',
  baseline: 'Baseline segmentation',
  mc_dropout: 'Monte Carlo uncertainty estimation',
  hotspot: 'Uncertainty hotspot selection',
  policy: 'Applying frozen decision policy',
  sam_refinement: 'SAM-Med3D refinement',
  finalizing: 'Finalizing scientific artifacts',
  complete: 'Inference complete',
  failed: 'Inference failed',
}

type Phase = 'idle' | 'creating' | 'executing' | 'monitoring' | 'settled' | 'attention'

type ObservedStage = {
  stage: RunStage
  updatedAt: string
}

function messageFromError(error: unknown): string {
  return error instanceof Error ? error.message : 'Unknown request error.'
}

function InferencePanel({
  caseId,
  existingRunId,
  onRunCreated,
  onBack,
}: Props) {
  const [run, setRun] = useState<RunView | null>(null)
  const [phase, setPhase] = useState<Phase>(
    existingRunId ? 'monitoring' : 'idle',
  )
  const [notice, setNotice] = useState<string | null>(null)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const [observedStages, setObservedStages] = useState<ObservedStage[]>([])
  const launchLock = useRef(false)
  const creationKeyRef = useRef<string | null>(null)

  const runId = run?.run_id ?? existingRunId
  const terminal = run?.status === 'complete' || run?.status === 'failed'
  const percentage = run ? Math.round(run.progress * 100) : null

  function acceptRun(next: RunView) {
    setRun(next)
    setObservedStages((previous) => {
      if (previous.at(-1)?.stage === next.stage) return previous
      return [
        ...previous,
        { stage: next.stage, updatedAt: next.updated_at },
      ]
    })
  }

  async function startInference() {
    if (
      launchLock.current ||
      run !== null ||
      (phase !== 'idle' && phase !== 'attention')
    ) return

    launchLock.current = true
    setNotice(null)
    setConnectionError(null)
    setPhase('creating')

    let created: RunView

    try {
      const key = creationKeyRef.current ??
        getOrCreateRunCreationKey(caseId)

      creationKeyRef.current = key

      created = await createRun(caseId, key)

      // Save confirmed identity before execution.
      saveRunRecovery({
        runId: created.run_id,
        caseId: created.case_id,
      })

      onRunCreated(created.run_id)
      acceptRun(created)

      clearPendingRunCreation()
    } catch (error) {
      setNotice(
        `Run creation could not be confirmed: ${messageFromError(error)} ` +
        'Do not repeatedly submit while the backend state is uncertain.',
      )
      setPhase('attention')
      launchLock.current = false
      return
    }

    setPhase('executing')

    try {
      const launched = await executeRun(created.run_id)
      acceptRun(launched)
    } catch (error) {
      setNotice(
        `Worker launch response was not confirmed: ${messageFromError(error)} ` +
        'The existing run will be inspected. Execution will not be requested again automatically.',
      )
    }

    setPhase('monitoring')
    launchLock.current = false
  }

  async function refreshRunStatus() {
    if (!runId || launchLock.current) return

    setConnectionError(null)

    try {
      const latest = await getRun(runId)

      if (latest.case_id !== caseId) {
        throw new Error(
          'The server returned a run associated with another case.',
        )
      }

      acceptRun(latest)

      if (latest.status === 'complete' || latest.status === 'failed') {
        setNotice(null)
        setPhase('settled')
      } else if (latest.status === 'queued') {
        setNotice(
          'The existing run is still queued. No worker was launched.',
        )
        setPhase('attention')
      } else {
        setNotice(null)
        setPhase('monitoring')
      }
    } catch (error) {
      setConnectionError(
        `Unable to inspect the existing run: ${messageFromError(error)}`,
      )
    }
  }

  useEffect(() => {
    if (phase !== 'monitoring' || !runId) return

    let stopped = false
    let timer: ReturnType<typeof setTimeout>

    async function poll() {
      try {
        const latest = await getRun(runId!)

        if (stopped) return

        acceptRun(latest)
        setConnectionError(null)

        if (latest.status === 'complete' || latest.status === 'failed') {
          setNotice(null)
          setPhase('settled')
          return
        }

        if (latest.status === 'queued') {
          setNotice(
            'This existing run is queued. No worker will be started ' +
            'automatically. Check its server status before taking action.',
          )
          setPhase('attention')
          return
        }

        setNotice(null)
      } catch (error) {
        if (stopped) return
        setConnectionError(
          `Unable to refresh run status: ${messageFromError(error)}`,
        )
      }

      if (!stopped) {
        timer = setTimeout(poll, 2500)
      }
    }

    timer = setTimeout(poll, 1500)

    return () => {
      stopped = true
      clearTimeout(timer)
    }
  }, [phase, runId])

  const backDisabled =
    phase === 'creating' ||
    phase === 'executing' ||
    (phase === 'monitoring' && !terminal)

  return (
    <main className="process">
      <header className="process__header">
        <div>
          <span className="process__eyebrow">02 / LIVE INFERENCE</span>
          <h2>From volume to insight.</h2>
          <p>
            Uncertainty-aware automatic prompting using the frozen
            NeuroPrompt-3D research pipeline.
          </p>
        </div>

        <button
          type="button"
          className="process__back"
          onClick={onBack}
          disabled={backDisabled}
        >
          ← {terminal ? 'Return to Prepare' : 'Back to Prepare'}
        </button>
      </header>

      <div className="process__workspace">
        <section className="process__primary" aria-label="Inference workspace">
          <div className="process__workspace-top">
            <span>PROCESS WORKSPACE</span>
            <span>
              {run ? run.status.toUpperCase() : 'AWAITING START'}
            </span>
          </div>

          <div className="process__focus" aria-live="polite">
            <span className="process__focus-icon" aria-hidden="true">
              {run?.status === 'complete'
                ? '✓'
                : run?.status === 'failed'
                  ? '!'
                  : '◈'}
            </span>

            <span className="process__eyebrow">
              {run ? run.stage.replaceAll('_', ' ').toUpperCase() : 'READY'}
            </span>

            <h3>
              {run
                ? stageLabels[run.stage]
                : phase === 'creating'
                  ? 'Creating an inference run'
                  : existingRunId
                    ? 'Retrieving existing inference run'
                    : 'Ready to begin inference'}
            </h3>

            <p>
              {run?.status === 'complete'
                ? 'The backend has completed this live scientific run.'
                : run?.status === 'failed'
                  ? 'The backend reported a run failure. Inspect the details below.'
                  : run
                    ? 'Stage and progress are reported directly by FastAPI.'
                    : 'Start a new run using the validated MRI case.'}
            </p>

            {run && (
              <div className="process__progress">
                <div className="process__progress-label">
                  <span>BACKEND-REPORTED PROGRESS</span>
                  <strong>{percentage}%</strong>
                </div>

                <progress
                  max={100}
                  value={percentage ?? 0}
                  aria-label="Backend-reported inference progress"
                />
              </div>
            )}

            {phase === 'idle' && (
              <button
                type="button"
                className="process__start"
                onClick={startInference}
              >
                Start scientific pipeline <span>→</span>
              </button>
            )}

            {phase === 'attention' && !runId && (
              <button
                type="button"
                className="process__refresh"
                onClick={() => void startInference()}
              >
                Retry creation with same request key ↻
              </button>
            )}

            {phase === 'attention' && runId && (
              <button
                type="button"
                className="process__refresh"
                onClick={() => void refreshRunStatus()}
              >
                Check existing run status ↻
              </button>
            )}
          </div>

          <div className="process__scientific-note">
            Frozen research inference · No post-test model tuning ·
            No clinical use
          </div>
        </section>

        <aside className="process__inspector">
          <div className="process__inspector-heading">
            <span>RUN INSPECTOR</span>
            <strong>Live case</strong>
          </div>

          <div className="process__datum">
            <span>STAGED CASE</span>
            <code>{caseId}</code>
          </div>

          <div className="process__datum">
            <span>RUN ID</span>
            <code>{runId ?? 'Not yet created'}</code>
          </div>

          <div className="process__datum">
            <span>EXECUTION STATUS</span>
            <strong>{run?.status ?? phase}</strong>
          </div>

          {run && (
            <>
              <div className="process__datum">
                <span>DEVICE</span>
                <strong>{run.execution_device ?? 'Not yet reported'}</strong>
              </div>

              <div className="process__datum">
                <span>FROZEN VARIANCE THRESHOLD</span>
                <code>{run.frozen_variance_threshold}</code>
              </div>

              <div className="process__observed">
                <span>OBSERVED SERVER STAGES</span>
                <ol>
                  {observedStages.map((item, index) => (
                    <li key={`${item.stage}-${index}`}>
                      {stageLabels[item.stage]}
                    </li>
                  ))}
                </ol>
                <small>
                  Only stages observed by this browser session are listed.
                </small>
              </div>
            </>
          )}

          {run?.status === 'complete' && (
            <div className="process__result">
              <span>FROZEN SCIENTIFIC DECISION</span>
              <strong>{run.action ?? 'Not reported'}</strong>
              <small>Gate: {run.gate_state ?? 'Not reported'}</small>
              {run.hotspot_variance != null && (
                <small>
                  Hotspot variance: {run.hotspot_variance}
                </small>
              )}
              {run.semantic_abstention_condition && (
                <small>
                  Abstention: {run.semantic_abstention_condition}
                </small>
              )}
            </div>
          )}

          {run?.status === 'failed' && (
            <div className="process__failure" role="alert">
              <strong>{run.error?.code ?? 'Run failed'}</strong>
              <p>{run.error?.message ?? 'No failure details were reported.'}</p>
            </div>
          )}
        </aside>
      </div>

      {(notice || connectionError) && (
        <div className="process__notice" role="alert">
          {notice && <p>{notice}</p>}
          {connectionError && <p>{connectionError}</p>}
        </div>
      )}

      {run?.status === 'complete' && (
        <section className="process__artifacts">
          <div>
            <span className="process__eyebrow">03 / EXPLORE</span>
            <h3>Your scientific outputs are ready.</h3>
            <p>
              The immersive NIfTI viewer is our next frontend milestone.
              For now, verified backend artifacts can be retrieved below.
            </p>
          </div>

          <div className="process__downloads">
            {(run.available_artifacts ?? []).map((name) => (
              <a
                key={name}
                href={artifactUrl(run.run_id, name)}
                target="_blank"
                rel="noopener noreferrer"
              >
                {name} ↗
              </a>
            ))}
          </div>
        </section>
      )}

      <footer className="process__footer">
        <span>01 PREPARE <b>✓</b></span>
        <span className="process__footer-active">02 PROCESS</span>
        <span>03 EXPLORE {terminal && run?.status === 'complete' ? '→' : '·'}</span>
      </footer>
    </main>
  )
}


// Imaging first; original inference controls remain unchanged.
export function ProcessPage(props: Props) {
  return (
    <div className="mri-workspace">
      <MRIViewer key={props.caseId} caseId={props.caseId} />
      <InferencePanel {...props} />
    </div>
  )
}
