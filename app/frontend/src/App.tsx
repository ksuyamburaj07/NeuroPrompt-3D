import { useState } from 'react'

import './App.css'
import {
  AppHeader,
  type AppMode,
} from './components/AppHeader'
import { ResearchDisclaimer } from './components/ResearchDisclaimer'
import { FinalTestPage } from './pages/FinalTestPage'
import { PreparePage } from './pages/PreparePage'
import { ResearchPage } from './pages/ResearchPage'

function App() {
  const [
    activeMode,
    setActiveMode,
  ] = useState<AppMode>('new-case')

  return (
    <div className="app-shell">
      <AppHeader
        activeMode={activeMode}
        onModeChange={setActiveMode}
      />

      <div className="app-content">
        <div hidden={activeMode !== 'new-case'}>
          <PreparePage />
        </div>

        {activeMode === 'final-test' && (
          <FinalTestPage />
        )}

        {activeMode === 'research' && (
          <ResearchPage />
        )}
      </div>

      <ResearchDisclaimer />
    </div>
  )
}

export default App
