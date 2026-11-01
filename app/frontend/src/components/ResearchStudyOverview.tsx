import './ResearchStudyOverview.css'

const methodology = [
  {
    title: 'Multimodal MRI',
    detail:
      'Use the available T1n, T1c, T2w and T2f sequences with consistent spatial preprocessing.',
  },
  {
    title: 'Baseline segmentation',
    detail:
      'Generate a baseline 3D tumor prediction for comparison with subsequent refinement.',
  },
  {
    title: 'Uncertainty estimation',
    detail:
      'Use Monte Carlo dropout to identify predictive variance and candidate uncertain regions.',
  },
  {
    title: 'Policy-controlled prompting',
    detail:
      'Evaluate frozen decision criteria and candidate prompt coordinates before optional refinement.',
  },
  {
    title: 'Conservative refinement',
    detail:
      'Use SAM-Med3D within the prescribed local refinement path, or retain the baseline when the policy abstains.',
  },
  {
    title: 'Frozen evaluation',
    detail:
      'Compare previously frozen final and baseline predictions against held-out ground truth without changing the decisions.',
  },
]

const milestones = [
  {
    name: 'M5 — Dataset discipline',
    detail:
      'Subject-level leakage-safe dataset partitioning and reproducible split records.',
  },
  {
    name: 'M9 — Scientific evidence',
    detail:
      'Pre-ground-truth prediction freeze, recorded evaluation protocol and final-test measurements.',
  },
  {
    name: 'M11 — Research application',
    detail:
      'Scientific visualization, frozen-case inspection and separate development-run diagnostics.',
  },
]

export function ResearchStudyOverview() {
  return (
    <section
      className="research-study"
      aria-labelledby="research-study-heading"
    >
      <div className="panel research-study__lead">
        <span className="eyebrow">
          Scientific research overview
        </span>

        <h3 id="research-study-heading">
          Uncertainty-guided 3D brain tumor segmentation
        </h3>

        <p>
          NeuroPrompt-3D investigates whether predictive
          uncertainty can guide automatic spatial prompts
          and conservative local refinement of a baseline
          3D brain MRI segmentation.
        </p>

        <p>
          The research question is not simply whether an
          additional model can modify a segmentation, but
          whether those modifications produce measurable,
          reproducible improvements on held-out cases.
        </p>
      </div>

      <article className="panel">
        <span className="eyebrow">Project motivation</span>
        <h3>Why NeuroPrompt-3D?</h3>

        <p>
          NeuroPrompt-3D grew from the goal of advancing beyond
          earlier tabular prediction projects into a deeper
          computer-vision and research-engineering problem.
          Three-dimensional medical imaging offers a demanding
          setting for studying segmentation, uncertainty and
          reproducible model evaluation.
        </p>

        <p>
          The central hypothesis was that uncertainty-guided
          prompts and conservative local refinement might
          improve a baseline segmentation. Whether this
          hypothesis actually holds must be determined by
          experimental evidence rather than visual appearance
          or an assumed benefit from adding another model.
        </p>
      </article>

      <div className="research-study__grid">
        <article className="panel">
          <span className="eyebrow">Dataset provenance</span>
          <h3>Research data</h3>

          <dl className="research-study__facts">
            <div>
              <dt>Dataset</dt>
              <dd>BraTS 2024 Adult Glioma Post-Treatment</dd>
            </div>
            <div>
              <dt>Source</dt>
              <dd>Synapse · syn60086071</dd>
            </div>
            <div>
              <dt>Selected release</dt>
              <dd>Version 2</dd>
            </div>
            <div>
              <dt>MRI modalities</dt>
              <dd>T1n, T1c, T2w, T2f</dd>
            </div>
          </dl>

          <p>
            The 2024 release was selected for its
            multimodal MRI data and post-treatment glioma
            segmentation labels. It provided an accessible
            research source when the newer 2025 data was
            unavailable. Subject-level separation protects
            against leakage between training and evaluation.
          </p>
        </article>

        <article className="panel">
          <span className="eyebrow">Research design</span>
          <h3>What we tested</h3>

          <p>
            A baseline segmentation is compared with a
            policy-controlled refinement result.
            Uncertainty informs candidate regions and
            prompts, while abstention allows the baseline
            to remain unchanged.
          </p>

          <p>
            All held-out measurements reported here come
            from the previously frozen evaluation,
            not from the separate live development runs.
          </p>
        </article>
      </div>

      <article className="panel">
        <span className="eyebrow">Experimental methodology</span>
        <h3>Research pipeline</h3>

        <ol className="research-study__method">
          {methodology.map((stage) => (
            <li key={stage.title}>
              <strong>{stage.title}</strong>
              <p>{stage.detail}</p>
            </li>
          ))}
        </ol>
      </article>

      <article className="panel">
        <span className="eyebrow">Frozen held-out evaluation</span>
        <h3>Measured findings</h3>

        <div className="research-study__metrics">
          <div>
            <strong>129</strong>
            <span>Frozen test cases</span>
          </div>
          <div>
            <strong>44</strong>
            <span>Local FP-removal decisions</span>
          </div>
          <div>
            <strong>85</strong>
            <span>Baseline abstentions</span>
          </div>
        </div>

        <h4>Mean baseline versus final performance</h4>

        <div
          className="research-study__table-wrap"
          role="region"
          aria-label="Frozen evaluation metric comparison"
          tabIndex={0}
        >
          <table className="research-study__results-table">
            <caption>
              Mean measurements across 129 held-out cases
            </caption>
            <thead>
              <tr>
                <th scope="col">Metric</th>
                <th scope="col">Baseline</th>
                <th scope="col">Final</th>
                <th scope="col">Paired change</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th scope="row">Dice ↑</th>
                <td>0.76627</td>
                <td>0.76251</td>
                <td>−0.00376</td>
              </tr>
              <tr>
                <th scope="row">IoU ↑</th>
                <td>0.64251</td>
                <td>0.63868</td>
                <td>−0.00383</td>
              </tr>
              <tr>
                <th scope="row">HD95 ↓</th>
                <td>24.29814 mm</td>
                <td>23.66236 mm</td>
                <td>−0.63578 mm</td>
              </tr>
            </tbody>
          </table>
        </div>

        <h4>Case-level metric outcomes</h4>

        <ul className="research-study__outcomes">
          <li>
            <strong>Dice</strong>
            <span>
              31 improved · 85 unchanged · 13 worsened
            </span>
          </li>
          <li>
            <strong>IoU</strong>
            <span>
              31 improved · 85 unchanged · 13 worsened
            </span>
          </li>
          <li>
            <strong>HD95</strong>
            <span>
              21 improved · 86 unchanged · 22 worsened
            </span>
          </li>
        </ul>

        <p>
          Metric outcome counts are not the same as
          policy-action counts. The 44 local FP-removal
          decisions and 85 abstentions describe the
          policy, not guaranteed improvements.
        </p>

        <p>
          Differences are final minus baseline.
          The negative Dice and IoU differences are
          unfavorable, while the negative HD95 difference
          is favorable because lower boundary distance
          is preferable.
        </p>

        <p className="research-study__conclusion">
          The frozen cohort does not establish an average
          Dice improvement from the refinement policy.
          A decision to apply local refinement must not
          be interpreted as proof of a better prediction.
        </p>
      </article>

      <div className="research-study__grid">
        <article className="panel">
          <span className="eyebrow">Scientific interpretation</span>
          <h3>Limitations</h3>

          <p>
            Findings are specific to the studied dataset,
            frozen protocol and model configuration.
            An uncertainty hotspot does not guarantee
            that refinement will improve segmentation.
          </p>

          <p>
            These results do not establish clinical
            effectiveness or generalization to other
            populations and MRI acquisition settings.
          </p>
        </article>

        <article className="panel">
          <span className="eyebrow">Evidence integrity</span>
          <h3>Reproducibility</h3>

          <p>
            The project retains subject-level split records,
            frozen predictions prepared before ground-truth
            evaluation, a prediction manifest and a
            separately recorded final-test evaluation.
          </p>

          <p>
            SHA-256 verification protects archived evidence
            against unnoticed changes. Development-run
            diagnostics are kept separate from the held-out
            research conclusions.
          </p>
        </article>
      </div>

      <article className="panel">
        <span className="eyebrow">Research development</span>
        <h3>Project milestones</h3>

        <ol className="research-study__timeline">
          {milestones.map((milestone) => (
            <li key={milestone.name}>
              <strong>{milestone.name}</strong>
              <p>{milestone.detail}</p>
            </li>
          ))}
        </ol>
      </article>

      <p className="research-study__disclaimer">
        Research prototype only. Not intended for clinical
        diagnosis, treatment planning or patient-care decisions.
      </p>
    </section>
  )
}
