import { createRoot } from 'react-dom/client'
import { CorticalQCPreview } from './pages/CorticalQCWorkspace'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <CorticalQCPreview />,
)
