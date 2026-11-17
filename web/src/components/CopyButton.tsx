import { useEffect, useRef, useState } from 'react'
import { copyText } from '../clipboard'
import { CheckIcon, CopyIcon } from './Icons'

interface Props {
  text: string
  label: string // the accessible name, for example "Copy answer"
  /** Visible text next to the icon; omit for an icon-only button. */
  caption?: string
  className?: string
}

export default function CopyButton({ text, label, caption, className = 'action' }: Props) {
  const [copied, setCopied] = useState(false)
  const timer = useRef<number | undefined>(undefined)

  useEffect(() => () => window.clearTimeout(timer.current), [])

  async function copy() {
    if (await copyText(text)) {
      setCopied(true)
      window.clearTimeout(timer.current)
      timer.current = window.setTimeout(() => setCopied(false), 1500)
    }
  }

  return (
    <button type="button" className={className} aria-label={copied ? 'Copied' : label} title={copied ? 'Copied' : label} onClick={() => void copy()}>
      {copied ? <CheckIcon /> : <CopyIcon />}
      {caption && <span>{copied ? 'Copied' : caption}</span>}
    </button>
  )
}
