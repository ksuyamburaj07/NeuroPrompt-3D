import { ModeBadge } from '../components/ModeBadge'

const diagnostics = [
  'Predictive-variance statistics',
  'Frozen policy variables',
  'Prompt coordinates',
  'Model and device provenance',
  'MC case seed and hashes',
  'Intermediate pipeline artifacts',
]

export function ResearchPage() {
  return (
    <main className="workspace">
      <section className="workspace-heading">
        <div>
          <ModeBadge tone="research">
            RESEARCH EXPLORER
          </ModeBadge>

          <h2>Technical Inspection</h2>

          <p>
            Development and scientific diagnostics for
            inspecting NeuroPrompt-3D execution state.
          </p>
        </div>

        <div className="result-identity result-identity--research">
          Technical · Development
        </div>
      </section>

      <section className="workspace-grid workspace-grid--research">
        <article className="panel">
          <div className="panel__heading">
            <div>
              <span className="eyebrow">
                Diagnostics
              </span>
              <h3>Scientific state</h3>
            </div>
          </div>

          <div className="diagnostic-grid">
            {diagnostics.map((item) => (
              <div
                className="diagnostic-item"
                key={item}
              >
                <span
                  className="diagnostic-item__marker"
                  aria-hidden="true"
                />
                {item}
              </div>
            ))}
          </div>
        </article>

        <article className="panel">
          <div className="panel__heading">
            <div>
              <span className="eyebrow">
                Existing interface
              </span>
              <h3>Gradio console</h3>
            </div>

            <span className="panel-status">
              Available
            </span>
          </div>

          <div className="research-console-card">
            <strong>
              Research console complete
            </strong>

            <p>
              The Gradio interface remains the current
              technical console for live validation,
              inference, existing-run inspection and
              safe artifact retrieval.
            </p>

            <code>
              http://127.0.0.1:7860
            </code>
          </div>
        </article>
      </section>
    </main>
  )
}
