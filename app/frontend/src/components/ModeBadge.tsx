type ModeBadgeProps = {
  tone: 'live' | 'frozen' | 'research'
  children: React.ReactNode
}

export function ModeBadge({
  tone,
  children,
}: ModeBadgeProps) {
  return (
    <span className={`mode-badge mode-badge--${tone}`}>
      <span
        className="mode-badge__dot"
        aria-hidden="true"
      />
      {children}
    </span>
  )
}
