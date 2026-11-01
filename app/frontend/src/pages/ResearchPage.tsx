import { ModeBadge } from '../components/ModeBadge'
import { ResearchRunInspector } from '../components/ResearchRunInspector'
import { ResearchStudyOverview } from '../components/ResearchStudyOverview'
import './ResearchTechnicalWorkspace.css'

const diagnostics = [
  'Execution status and progress',
  'Frozen variance threshold',
  'MC case seed, when recorded',
  'Policy action and gate state',
  'Hotspot and prompt coordinates',
  'Available artifact names',
]

export function ResearchPage() {
  return (
    <main className="workspace">
      <section className="workspace-heading">
        <div>
          <ModeBadge tone="research">
            RESEARCH EXPLORER
          </ModeBadge>

          <h2>Research &amp; Scientific Evidence</h2>

          <p>
            Research motivation, methodology, frozen experimental
            findings and reproducible technical evidence.
          </p>
        </div>

        <div className="result-identity result-identity--research">
          Scientific · Research
        </div>
      </section>

      <ResearchStudyOverview />

      <section
        className="research-tech"
        aria-labelledby="research-tech-heading"
      >
        <header className="research-tech__header">
          <span className="eyebrow">
            Supporting technical evidence
          </span>
          <h3 id="research-tech-heading">
            Technical Research Workspace
          </h3>
          <p>
            Optional development-run inspection and local
            diagnostic tools, kept separate from the frozen
            scientific findings above.
          </p>
        </header>

        <details className="research-tech__disclosure">
          <summary className="research-tech__summary">
            <span>
              <strong>Saved Runs &amp; Inspection</strong>
              <small>
                Recent executions, policy decisions, provenance
                and recorded artifacts
              </small>
            </span>
          </summary>
          <div className="research-tech__body">
            <ResearchRunInspector />
          </div>
        </details>

        <details className="research-tech__disclosure">
          <summary className="research-tech__summary">
            <span>
              <strong>Diagnostics &amp; Local Console</strong>
              <small>
                Metadata reference and optional Gradio interface
              </small>
            </span>
          </summary>
          <div className="research-tech__body">
            <section className="workspace-grid workspace-grid--research">
        <article className="panel">
          <div className="panel__heading">
            <div>
              <span className="eyebrow">
                Diagnostics
              </span>
              <h3>Recorded metadata fields</h3>
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
              Not checked
            </span>
          </div>

          <div className="research-console-card">
            <strong>
              Separate local research console
            </strong>

            <p>
              The Gradio interface is a separate local
              development tool for validation, inference,
              run inspection and artifact retrieval.
              Its availability is not checked by this page.
            </p>

            <code>
              http://127.0.0.1:7860
            </code>
          </div>
        </article>
            </section>
          </div>
        </details>
      </section>
    </main>
  )
}
