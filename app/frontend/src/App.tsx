import { useState } from 'react'

import './App.css'
import {
  AppHeader,
  type AppMode,
} from './components/AppHeader'
import { ResearchDisclaimer } from './components/ResearchDisclaimer'
import { FinalTestPage } from './pages/FinalTestPage'
import { NewCasePage } from './pages/NewCasePage'
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
        {activeMode === 'new-case' && (
          <NewCasePage />
        )}

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
