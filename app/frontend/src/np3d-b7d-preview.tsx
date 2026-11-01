import { createRoot } from 'react-dom/client'
import { MRIViewer } from './pages/MRIViewer'
import './index.css'

const caseId = 'case_8ba65840d9454aa68c82095010705aa6'
const runId = 'run_5e6f7388faa94267958d26f20025d605'

createRoot(document.getElementById('root')!).render(
  <main
    style={{
      maxWidth: 1500,
      margin: '0 auto',
      padding: 24,
    }}
  >
    <header
      style={{
        marginBottom: 24,
        padding: 18,
        border: '1px solid #95667b',
        borderRadius: 12,
      }}
    >
      <h1>NeuroPrompt-3D · Synthetic Cortex Preview</h1>
      <p>
        Disposable software visualization test only.
        Artificial folded geometry and simulated segmentation.
        No real inference or anatomical QC.
      </p>
    </header>

    <MRIViewer
      caseId={caseId}
      completedRunId={runId}
    />
  </main>,
)
