import { useState, type ChangeEvent, type DragEvent } from 'react'
import './PreparePage.css'
import { ProcessPage } from './ProcessPage'
import { createRun, getRun } from '../api/runs'
import {
  clearRunRecovery,
  clearPendingRunCreation,
  readPendingRunCreation,
  readRunRecovery,
  saveRunRecovery,
  type SavedRunReference,
} from '../api/runRecovery'
import {
  deleteStagedCase,
  validateCase,
  type CaseValidationResponse,
} from '../api/cases'

type ModalityKey = 't1n' | 't1c' | 't2w' | 't2f'

type Modality = {
  key: ModalityKey
  label: string
  core: string
  description: string
}

const modalities: Modality[] = [
  { key: 't1n', label: 'T1n', core: 'T1', description: 'Native T1-weighted' },
  { key: 't1c', label: 'T1c', core: 'T1ce', description: 'Contrast-enhanced T1' },
  { key: 't2w', label: 'T2w', core: 'T2', description: 'T2-weighted' },
  { key: 't2f', label: 'T2-FLAIR', core: 'FLAIR', description: 'Fluid-attenuated inversion recovery' },
]

const isNifti = (name: string) => /\.nii(\.gz)?$/i.test(name)

const fileSize = (bytes: number) =>
  `${(bytes / (1024 * 1024)).toFixed(1)} MB`

export function PreparePage() {
  const [files, setFiles] = useState<Partial<Record<ModalityKey, File>>>({})
  const [groundTruth, setGroundTruth] = useState<File | null>(null)
  const [activeDrop, setActiveDrop] = useState<ModalityKey | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [requestError, setRequestError] = useState<string | null>(null)
  const [validation, setValidation] = useState<CaseValidationResponse | null>(null)
  const [isValidating, setIsValidating] = useState(false)
  const [isDiscarding, setIsDiscarding] = useState(false)
  const [processCaseId, setProcessCaseId] = useState<string | null>(null)
  const [existingRunId, setExistingRunId] = useState<string | null>(null)
  const [savedRun, setSavedRun] = useState<SavedRunReference | null>(
    () => readRunRecovery(),
  )
  const [recoveryInput, setRecoveryInput] = useState('')
  const [isRecovering, setIsRecovering] = useState(false)
  const [recoveryError, setRecoveryError] = useState<string | null>(null)

  const selectedCount = modalities.filter(({ key }) => Boolean(files[key])).length
  const complete = selectedCount === modalities.length
  const ready = validation?.valid === true &&
    validation.ready_for_inference === true &&
    Boolean(validation.case_id)
  const locked = ready || isValidating || isDiscarding

  function selectFile(key: ModalityKey, file?: File) {
    if (locked || !file) return

    if (!isNifti(file.name)) {
      setError(`${file.name}: select a .nii or .nii.gz file.`)
      return
    }

    setFiles((previous) => ({ ...previous, [key]: file }))
    setValidation(null)
    setRequestError(null)
    setError(null)
  }

  function selectGroundTruth(file?: File) {
    if (locked || !file) return

    if (!isNifti(file.name)) {
      setError('Ground truth must be a .nii or .nii.gz file.')
      return
    }

    setGroundTruth(file)
    setValidation(null)
    setRequestError(null)
    setError(null)
  }

  function handleChange(
    event: ChangeEvent<HTMLInputElement>,
    key: ModalityKey,
  ) {
    selectFile(key, event.currentTarget.files?.[0])
    event.currentTarget.value = ''
  }

  function handleDrop(event: DragEvent<HTMLDivElement>, key: ModalityKey) {
    event.preventDefault()
    setActiveDrop(null)

    if (locked) return

    if (event.dataTransfer.files.length !== 1) {
      setError('Drop exactly one MRI file into each modality slot.')
      return
    }

    selectFile(key, event.dataTransfer.files[0])
  }

  function removeFile(key: ModalityKey) {
    if (locked) return
    setValidation(null)
    setRequestError(null)

    setFiles((previous) => {
      const updated = { ...previous }
      delete updated[key]
      return updated
    })
    setError(null)
  }

  async function handleValidate() {
    if (!complete || locked) return

    setError(null)
    setRequestError(null)
    setValidation(null)
    setIsValidating(true)

    try {
      const response = await validateCase({
        t1n: files.t1n!,
        t1c: files.t1c!,
        t2w: files.t2w!,
        t2f: files.t2f!,
        segmentation: groundTruth,
      })

      setValidation(response)
    } catch (cause) {
      setRequestError(
        cause instanceof Error ? cause.message : 'MRI validation failed.',
      )
    } finally {
      setIsValidating(false)
    }
  }

  async function handleDiscard() {
    if (!validation?.case_id || !ready || isDiscarding) return

    setIsDiscarding(true)
    setRequestError(null)

    try {
      await deleteStagedCase(validation.case_id)
      setFiles({})
      setGroundTruth(null)
      setValidation(null)
      setProcessCaseId(null)
      setExistingRunId(null)
      setError(null)
    } catch (cause) {
      setRequestError(
        cause instanceof Error ? cause.message : 'Unable to discard the case.',
      )
    } finally {
      setIsDiscarding(false)
    }
  }

  async function recoverPendingCreation() {
    const pending = readPendingRunCreation()

    if (!pending || isRecovering || ready) return

    setIsRecovering(true)
    setRecoveryError(null)

    try {
      // Same request key; no worker execution.
      const recovered = await createRun(
        pending.caseId,
        pending.idempotencyKey,
      )

      if (recovered.case_id !== pending.caseId) {
        throw new Error('Recovered run belongs to another case.')
      }

      const reference = {
        runId: recovered.run_id,
        caseId: recovered.case_id,
      }

      saveRunRecovery(reference)
      clearPendingRunCreation()

      setSavedRun(reference)
      setExistingRunId(recovered.run_id)
      setProcessCaseId(recovered.case_id)
    } catch (cause) {
      setRecoveryError(
        cause instanceof Error
          ? cause.message
          : 'Unable to recover creation request.',
      )
    } finally {
      setIsRecovering(false)
    }
  }

  async function handleRecoverRun(candidate: string) {
    if (isRecovering || ready) return

    const requestedId = candidate.trim()

    if (!requestedId) {
      setRecoveryError('Enter an existing inference run ID.')
      return
    }

    setIsRecovering(true)
    setRecoveryError(null)

    try {
      // A GET request only: never create or execute a run on reconnect.
      const recovered = await getRun(requestedId)

      if (recovered.run_id !== requestedId) {
        throw new Error('The backend returned a different run identity.')
      }

      const reference: SavedRunReference = {
        runId: recovered.run_id,
        caseId: recovered.case_id,
      }

      saveRunRecovery(reference)
      setSavedRun(reference)
      setExistingRunId(recovered.run_id)
      setProcessCaseId(recovered.case_id)
    } catch (cause) {
      setRecoveryError(
        cause instanceof Error
          ? cause.message
          : 'Unable to reconnect to the run.',
      )
    } finally {
      setIsRecovering(false)
    }
  }

  if (processCaseId) {
    return (
      <ProcessPage
        caseId={processCaseId}
        existingRunId={existingRunId}
        onRunCreated={setExistingRunId}
        onBack={() => {
          setProcessCaseId(null)
          setSavedRun(readRunRecovery())
        }}
      />
    )
  }

  return (
    <main className="prepare">
      <div className="prepare__heading">
        <div>
          <div className="prepare__eyebrow">
            <span className="prepare__live-dot" />
            NEW CASE <span className="prepare__separator">/</span> LIVE INPUT
          </div>

          <h2>
            Begin with the scans.
            <span>Discover what the model sees.</span>
          </h2>

          <p>
            Four MRI sequences. One uncertainty-aware segmentation workflow.
            Prepare a new case for NeuroPrompt-3D.
          </p>
        </div>

        <div className="prepare__session">
          <span>SESSION TYPE</span>
          <strong>New inference</strong>
          <small>Research use only</small>
        </div>
      </div>

      <div className="prepare__layout">
        <section className="prepare__acquisition" aria-labelledby="acquisition-title">
          <div className="prepare__section-header">
            <div>
              <span className="prepare__kicker">01 / ACQUISITION</span>
              <h3 id="acquisition-title">MRI input sequence</h3>
            </div>
            <span className="prepare__counter">{selectedCount} / 4 ATTACHED</span>
          </div>

          <div className="prepare__progress" aria-label={`${selectedCount} of 4 modalities selected`}>
            {modalities.map(({ key }) => (
              <span
                key={key}
                className={files[key] ? 'prepare__progress-segment is-filled' : 'prepare__progress-segment'}
              />
            ))}
          </div>

          <p className="prepare__instruction">
            Select or drop one NIfTI volume into each sequence.
          </p>

          <div className="prepare__file-list">
            {modalities.map((modality, index) => {
              const file = files[modality.key]
              const inputId = `prepare-${modality.key}`

              return (
                <div
                  key={modality.key}
                  className={[
                    'prepare__file-row',
                    file ? 'is-selected' : '',
                    activeDrop === modality.key ? 'is-dragging' : '',
                  ].filter(Boolean).join(' ')}
                  onDragOver={(event) => {
                    event.preventDefault()
                    if (!locked) setActiveDrop(modality.key)
                  }}
                  onDragLeave={() => setActiveDrop(null)}
                  onDrop={(event) => handleDrop(event, modality.key)}
                >
                  <span className="prepare__file-index">
                    {String(index + 1).padStart(2, '0')}
                  </span>

                  <div className="prepare__file-identity">
                    <strong>{modality.label}</strong>
                    <span>{modality.description}</span>
                    <small>Core channel: {modality.core}</small>
                  </div>

                  <div className="prepare__file-action">
                    <input
                      id={inputId}
                      className="prepare__native-input"
                      type="file"
                      accept=".nii,.nii.gz"
                      disabled={locked}
                      aria-label={`Choose ${modality.label} MRI volume`}
                      onChange={(event) => handleChange(event, modality.key)}
                    />

                    <label htmlFor={inputId} className="prepare__file-picker">
                      <span className="prepare__file-picker-icon" aria-hidden="true">
                        {file ? '✓' : '+'}
                      </span>

                      <span className="prepare__file-picker-text">
                        {file ? file.name : 'Choose or drop file'}
                      </span>
                    </label>

                    {file && (
                      <>
                        <span className="prepare__file-size">{fileSize(file.size)}</span>
                        <button
                          type="button"
                          className="prepare__remove"
                          onClick={() => removeFile(modality.key)}
                          disabled={locked}
                          aria-label={`Remove ${modality.label} volume`}
                        >
                          ×
                        </button>
                      </>
                    )}
                  </div>
                </div>
              )
            })}
          </div>

          <div className="prepare__optional">
            <div className="prepare__optional-heading">
              <div>
                <strong>Ground-truth segmentation</strong>
                <span>Optional · research evaluation only</span>
              </div>
              <small>OPTIONAL</small>
            </div>

            <div className="prepare__optional-picker">
              <input
                id="prepare-gt"
                className="prepare__native-input"
                type="file"
                accept=".nii,.nii.gz"
                disabled={locked}
                aria-label="Choose optional ground truth segmentation"
                onChange={(event) => {
                  selectGroundTruth(event.currentTarget.files?.[0])
                  event.currentTarget.value = ''
                }}
              />
              <label htmlFor="prepare-gt">
                {groundTruth ? groundTruth.name : '+ Add ground truth volume'}
              </label>

              {groundTruth && (
                <button
                  type="button"
                  onClick={() => {
                    if (locked) return
                    setGroundTruth(null)
                    setValidation(null)
                    setRequestError(null)
                  }}
                  disabled={locked}
                  aria-label="Remove ground truth segmentation"
                >
                  Remove
                </button>
              )}
            </div>
          </div>

          {error && (
            <p className="prepare__error" role="alert">{error}</p>
          )}

          <div className="prepare__validation">
            <div aria-live="polite">
              <strong>
                {isValidating
                  ? 'Validating MRI volumes...'
                  : isDiscarding
                    ? 'Discarding staged case...'
                    : ready
                      ? 'MRI case validated and staged'
                      : validation?.status === 'invalid'
                        ? 'Scientific validation failed'
                        : complete
                          ? 'All four sequences selected'
                          : 'Awaiting complete MRI set'}
              </strong>
              <span>
                {ready
                  ? 'FastAPI confirmed MRI readiness. Inference has not started.'
                  : isValidating
                    ? 'Checking image readability, modality completeness and geometry.'
                    : complete
                      ? 'Ready to submit for scientific validation.'
                      : 'Select all four MRI volumes before validation.'}
              </span>
            </div>

            {ready ? (
              <div className="prepare__validation-actions">
                <button
                  type="button"
                  onClick={() => {
                    if (validation?.case_id) {
                      setProcessCaseId(validation.case_id)
                    }
                  }}
                >
                  Continue to Process →
                </button>
                <button
                  type="button"
                  onClick={handleDiscard}
                  disabled={isDiscarding}
                >
                  {isDiscarding ? 'Discarding...' : 'Discard staged case'}
                </button>
              </div>
            ) : (
              <button
                type="button"
                disabled={!complete || isValidating}
                onClick={handleValidate}
              >
                {isValidating ? 'Validating...' : 'Validate case'}
                <span aria-hidden="true">→</span>
              </button>
            )}
          </div>

          {ready && validation?.case_id && (
            <div className="prepare__validated-result" role="status">
              <strong>VALIDATED CASE</strong>
              <code>{validation.case_id}</code>
              {validation.geometry && (
                <span>
                  Shape XYZ: {validation.geometry.shape_xyz.join(' × ')}
                  {' · '}
                  Spacing XYZ: {validation.geometry.voxel_spacing_xyz.join(' × ')} mm
                </span>
              )}
              <span>
                Ground truth: {validation.ground_truth.provided
                  ? 'Provided'
                  : 'Not provided'}
              </span>
            </div>
          )}

          {validation?.errors && validation.errors.length > 0 && (
            <ul className="prepare__validation-errors" role="alert">
              {validation.errors.map((issue, index) => (
                <li key={`${issue.code}-${issue.field ?? 'case'}-${index}`}>
                  {issue.field ? `${issue.field}: ` : ''}
                  {issue.message}
                </li>
              ))}
            </ul>
          )}

          {requestError && (
            <p className="prepare__error" role="alert">{requestError}</p>
          )}

          <p className="prepare__implementation-note">
            Validated cases are staged temporarily by FastAPI.
            No inference runs during validation.
          </p>
        </section>

        <aside className="prepare__visual" aria-label="Imaging workspace preview">
          <div className="prepare__visual-header">
            <span>VOLUME WORKSPACE</span>
            <span className="prepare__visual-status">
              <span /> NO VOLUME LOADED
            </span>
          </div>

          <div className="prepare__scan-field">
            <div className="prepare__scan-ring prepare__scan-ring--outer" />
            <div className="prepare__scan-ring prepare__scan-ring--inner" />
            <div className="prepare__crosshair prepare__crosshair--horizontal" />
            <div className="prepare__crosshair prepare__crosshair--vertical" />
            <span className="prepare__scan-origin" />

            <div className="prepare__scan-label">
              <span>AWAITING IMAGE DATA</span>
              <strong>Your MRI workspace starts here.</strong>
              <p>
                Once the pipeline and viewer are connected, real anatomical
                slices and segmentation results will appear in this space.
              </p>
            </div>

            <div className="prepare__scan-coordinates">
              <span>X —</span><span>Y —</span><span>Z —</span>
            </div>
          </div>

          <div className="prepare__visual-footnote">
            <span>Accepted formats</span>
            <strong>.nii / .nii.gz</strong>
            <span>Volume geometry verified by the backend</span>
          </div>
        </aside>
      </div>

      {!ready && (
        <section className="prepare__recovery" aria-label="Run recovery">
          <div className="prepare__recovery-intro">
            <span className="prepare__kicker">RECONNECT / EXISTING RUN</span>
            <strong>Continue an existing inference session</strong>
            <p>
              Inspect a previously created run without uploading MRI files
              again or starting another worker.
            </p>
          </div>

          <div className="prepare__recovery-controls">
            {readPendingRunCreation() && (
              <div className="prepare__saved-run">
                <button
                  type="button"
                  disabled={isRecovering}
                  onClick={() => void recoverPendingCreation()}
                >
                  Recover unconfirmed creation ↻
                </button>
                <span>
                  Reuses the original key. No worker execution.
                </span>
              </div>
            )}
            {savedRun && (
              <div className="prepare__saved-run">
                <button
                  type="button"
                  disabled={isRecovering}
                  onClick={() => void handleRecoverRun(savedRun.runId)}
                >
                  Reconnect saved run →
                </button>
                <code>{savedRun.runId}</code>
                <button
                  type="button"
                  disabled={isRecovering}
                  onClick={() => {
                    clearRunRecovery()
                    setSavedRun(null)
                    setExistingRunId(null)
                  }}
                >
                  Clear browser shortcut
                </button>
              </div>
            )}

            <form
              className="prepare__recovery-form"
              onSubmit={(event) => {
                event.preventDefault()
                void handleRecoverRun(recoveryInput)
              }}
            >
              <input
                type="text"
                value={recoveryInput}
                onChange={(event) => setRecoveryInput(event.target.value)}
                placeholder="run_..."
                aria-label="Existing inference run ID"
                autoComplete="off"
                spellCheck={false}
                disabled={isRecovering}
              />
              <button
                type="submit"
                disabled={isRecovering || !recoveryInput.trim()}
              >
                {isRecovering ? 'Connecting...' : 'Inspect existing run'}
              </button>
            </form>

            {recoveryError && (
              <p role="alert" className="prepare__error">
                {recoveryError}
              </p>
            )}
          </div>
        </section>
      )}

      <footer className="prepare__timeline" aria-label="Case workflow">
        <div className="prepare__timeline-step is-current">
          <span>01</span>
          <strong>Prepare</strong>
          <small>Select MRI volumes</small>
        </div>
        <span className="prepare__timeline-line" />
        <div className="prepare__timeline-step">
          <span>02</span>
          <strong>Process</strong>
          <small>Frozen inference</small>
        </div>
        <span className="prepare__timeline-line" />
        <div className="prepare__timeline-step">
          <span>03</span>
          <strong>Explore</strong>
          <small>Inspect segmentation</small>
        </div>
      </footer>
    </main>
  )
}
