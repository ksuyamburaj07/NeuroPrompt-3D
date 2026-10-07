import { ModeBadge } from '../components/ModeBadge'

export function FinalTestPage() {
  return (
    <main className="workspace">
      <section className="workspace-heading">
        <div>
          <ModeBadge tone="frozen">
            FROZEN RESEARCH RESULT
          </ModeBadge>

          <h2>Final-Test Explorer</h2>

          <p>
            Read-only exploration of the frozen
            M9E final-test evidence and outputs.
          </p>
        </div>

        <div className="result-identity result-identity--frozen">
          M9E Final Test · Read Only
        </div>
      </section>

      <section className="workspace-grid workspace-grid--explorer">
        <article className="panel metric-summary">
          <span className="eyebrow">
            Frozen cohort
          </span>

          <strong className="large-number">
            129
          </strong>

          <span>
            final-test cases
          </span>
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
            <span>No SAM reruns under M9E identity</span>
            <span>No frozen-output rewrites</span>
          </div>
        </article>

        <article className="panel panel--wide">
          <div className="panel__heading">
            <div>
              <span className="eyebrow">
                Explorer workspace
              </span>
              <h3>Frozen cases and results</h3>
            </div>

            <span className="panel-status">
              Upcoming
            </span>
          </div>

          <div className="empty-workspace">
            <strong>
              M9E result explorer foundation
            </strong>

            <p>
              Case navigation, frozen segmentation
              visualization and final-test metrics will
              be connected without modifying the M9E
              research archive.
            </p>
          </div>
        </article>
      </section>
    </main>
  )
}
