import { useEffect, useState } from 'react'

function matches(query: string): boolean {
  return typeof window !== 'undefined' && typeof window.matchMedia === 'function' ? window.matchMedia(query).matches : false
}

/** Whether a CSS media query currently matches (false where the browser cannot tell). */
export function useMediaQuery(query: string): boolean {
  const [value, setValue] = useState(() => matches(query))

  useEffect(() => {
    if (typeof window.matchMedia !== 'function') return
    const list = window.matchMedia(query)
    const onChange = () => setValue(list.matches)
    list.addEventListener?.('change', onChange)
    return () => list.removeEventListener?.('change', onChange)
  }, [query])

  return value
}
