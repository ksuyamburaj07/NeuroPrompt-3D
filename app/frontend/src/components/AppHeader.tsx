export type AppMode =
  | 'new-case'
  | 'final-test'
  | 'research'

type AppHeaderProps = {
  activeMode: AppMode
  onModeChange: (mode: AppMode) => void
}

const navigation: Array<{
  id: AppMode
  label: string
}> = [
  {
    id: 'new-case',
    label: 'New Case',
  },
  {
    id: 'final-test',
    label: 'Final-Test Explorer',
  },
  {
    id: 'research',
    label: 'Research',
  },
]

export function AppHeader({
  activeMode,
  onModeChange,
}: AppHeaderProps) {
  return (
    <header className="app-header">
      <div className="app-header__identity">
        <div
          className="app-mark"
          aria-hidden="true"
        >
          <span className="app-mark__core" />
          <span className="app-mark__ring app-mark__ring--one" />
          <span className="app-mark__ring app-mark__ring--two" />
        </div>

        <div>
          <div className="app-header__title-row">
            <h1>NeuroPrompt-3D</h1>
            <span className="prototype-tag">
              Research
            </span>
          </div>

          <p>
            Uncertainty-Aware Automatic Prompting for
            3D Brain MRI Tumor Segmentation
          </p>
        </div>
      </div>

      <nav
        className="primary-nav"
        aria-label="Application modes"
      >
        {navigation.map((item) => {
          const active =
            activeMode === item.id

          return (
            <button
              key={item.id}
              type="button"
              className={
                active
                  ? 'primary-nav__item primary-nav__item--active'
                  : 'primary-nav__item'
              }
              aria-current={
                active
                  ? 'page'
                  : undefined
              }
              onClick={() => {
                onModeChange(item.id)
              }}
            >
              {item.label}
            </button>
          )
        })}
      </nav>
    </header>
  )
}
