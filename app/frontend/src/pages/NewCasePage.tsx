import { ModeBadge } from '../components/ModeBadge'

const modalities = [
  {
    name: 'T1n',
    core: 'T1',
  },
  {
    name: 'T1c',
    core: 'T1ce',
  },
  {
    name: 'T2w',
    core: 'T2',
  },
  {
    name: 'T2-FLAIR',
    core: 'FLAIR',
  },
]

const pipeline = [
  'Validate MRI set',
  'Baseline 3D U-Net',
  'MC Dropout uncertainty',
  'Hotspot + policy',
  'SAM-Med3D if eligible',
  'Final segmentation',
]

export function NewCasePage() {
  return (
    <main className="workspace">
      <section className="workspace-heading">
        <div>
          <ModeBadge tone="live">
            LIVE INFERENCE
          </ModeBadge>

          <h2>New MRI Case</h2>

          <p>
            Upload a complete four-modality MRI case,
            validate its geometry, and run the frozen
            NeuroPrompt-3D automatic pipeline.
          </p>
        </div>

        <div className="result-identity result-identity--live">
          User-uploaded MRI · New Result
        </div>
      </section>

      <section className="workspace-grid workspace-grid--new-case">
        <article className="panel panel--inputs">
          <div className="panel__heading">
            <div>
              <span className="eyebrow">
                Input set
              </span>
              <h3>MRI modalities</h3>
            </div>

            <span className="panel-status">
              4 required
            </span>
          </div>

          <div className="modality-grid">
            {modalities.map((modality) => (
              <div
                className="modality-card"
                key={modality.name}
              >
                <div className="modality-card__top">
                  <span className="modality-card__name">
                    {modality.name}
                  </span>

                  <span className="required-tag">
                    Required
                  </span>
                </div>

                <span className="modality-card__mapping">
                  Frozen core: {modality.core}
                </span>

                <div className="modality-card__empty">
                  No file selected
                </div>
              </div>
            ))}
          </div>

          <div className="optional-input">
            <div>
              <strong>
                Ground-truth segmentation
              </strong>
              <span>
                Optional · research evaluation only
              </span>
            </div>

            <span className="optional-tag">
              Optional
            </span>
          </div>

          <button
            className="primary-action"
            type="button"
            disabled
          >
            Validate MRI Case
          </button>

          <p className="control-note">
            Upload controls and backend validation will
            be connected in a later M11D integration step.
          </p>
        </article>

        <article className="panel panel--viewer-placeholder">
          <div className="panel__heading">
            <div>
              <span className="eyebrow">
                Imaging workspace
              </span>
              <h3>MRI viewer</h3>
            </div>

            <span className="panel-status">
              Planned
            </span>
          </div>

          <div className="viewer-placeholder">
            <div
              className="viewer-placeholder__crosshair"
              aria-hidden="true"
            />

            <div className="viewer-placeholder__content">
              <div
                className="viewer-placeholder__brain"
                aria-hidden="true"
              >
                <span />
                <span />
                <span />
              </div>

              <strong>
                Interactive imaging workspace
              </strong>

              <p>
                Axial, coronal and sagittal views with
                segmentation, uncertainty and prompt
                overlays will be implemented in M11E.
              </p>
            </div>

            <div className="viewer-placeholder__axes">
              <span>AXIAL</span>
              <span>CORONAL</span>
              <span>SAGITTAL</span>
            </div>
          </div>
        </article>

        <article className="panel panel--pipeline">
          <div className="panel__heading">
            <div>
              <span className="eyebrow">
                Frozen execution path
              </span>
              <h3>Automatic pipeline</h3>
            </div>
          </div>

          <ol className="pipeline-list">
            {pipeline.map((stage, index) => (
              <li key={stage}>
                <span className="pipeline-list__index">
                  {String(index + 1).padStart(2, '0')}
                </span>

                <span>
                  {stage}
                </span>
              </li>
            ))}
          </ol>

          <div className="policy-lock">
            <span
              className="policy-lock__icon"
              aria-hidden="true"
            >
              ◇
            </span>

            <div>
              <strong>
                Scientific policy locked
              </strong>

              <p>
                Thresholds, model parameters and automatic
                prompting rules are displayed but are not
                ordinary user controls.
              </p>
            </div>
          </div>
        </article>
      </section>
    </main>
  )
}
