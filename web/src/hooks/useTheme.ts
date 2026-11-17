import { useCallback, useEffect, useState } from 'react'
import { THEME_KEY, readSetting, writeSetting } from '../storage'

export type Theme = 'light' | 'dark'

function systemTheme(): Theme {
  try {
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  } catch {
    return 'light'
  }
}

function storedTheme(): Theme | null {
  const value = readSetting(THEME_KEY)
  return value === 'light' || value === 'dark' ? value : null
}

/**
 * The colour theme. By default it follows the system setting; the toggle picks light or dark explicitly and
 * remembers the choice (in localStorage, if it is available). An explicit choice is set as data-theme on <html>;
 * with no choice the stylesheet follows prefers-color-scheme by itself.
 */
export function useTheme() {
  const [chosen, setChosen] = useState<Theme | null>(storedTheme)
  const [system, setSystem] = useState<Theme>(systemTheme)

  useEffect(() => {
    const root = document.documentElement
    if (chosen) root.dataset.theme = chosen
    else delete root.dataset.theme
  }, [chosen])

  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return
    const query = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = () => setSystem(query.matches ? 'dark' : 'light')
    query.addEventListener?.('change', onChange)
    return () => query.removeEventListener?.('change', onChange)
  }, [])

  const theme: Theme = chosen ?? system
  const toggleTheme = useCallback(() => {
    const next: Theme = theme === 'dark' ? 'light' : 'dark'
    setChosen(next)
    writeSetting(THEME_KEY, next)
  }, [theme])

  return { theme, toggleTheme }
}
