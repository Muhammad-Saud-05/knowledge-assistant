import type { Theme } from '../hooks/useTheme'
import { MoonIcon, SunIcon } from './Icons'

interface Props {
  theme: Theme
  onToggle: () => void
}

/** Switches between the light and dark theme (and remembers the choice). */
export default function ThemeToggle({ theme, onToggle }: Props) {
  const next = theme === 'dark' ? 'light' : 'dark'
  return (
    <button type="button" className="foot-item" onClick={onToggle}>
      {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
      <span>Switch to {next} theme</span>
    </button>
  )
}
