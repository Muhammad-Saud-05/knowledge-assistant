import { createContext, isValidElement, useContext } from 'react'
import type { ComponentProps, ReactElement, ReactNode } from 'react'
import Markdown from 'react-markdown'
import type { Components } from 'react-markdown'
import CitationChip from './CitationChip'
import { remarkCitations } from '../citations'
import { remarkTables } from '../markdownTables'
import CopyButton from './CopyButton'

// ---------------------------------------------------------------- code blocks

function textOf(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(textOf).join('')
  if (isValidElement(node)) return textOf((node as ReactElement<{ children?: ReactNode }>).props.children)
  return ''
}

function CodeBlock({ children }: { children?: ReactNode }) {
  const code = isValidElement(children) ? (children as ReactElement<{ className?: string }>) : null
  const language = /language-([\w+-]+)/.exec(code?.props.className ?? '')?.[1]
  return (
    <div className="codeblock">
      <div className="codebar">
        <span>{language ?? 'code'}</span>
        <CopyButton text={textOf(children).replace(/\n$/, '')} label="Copy code" caption="Copy" />
      </div>
      <pre>{children}</pre>
    </div>
  )
}

// --------------------------------------------------------------------- answer

/** The function to call when a citation chip is clicked. Passed through context so that the Markdown components
 *  below can be the same functions on every render: new functions would make React rebuild every chip, and a
 *  chip that was clicked would no longer be on the page to give the focus back to. */
const CiteContext = createContext<((number: number, trigger: HTMLElement) => void) | undefined>(undefined)

function ChipFromMarkdown({ n }: { n?: string }) {
  const onCite = useContext(CiteContext)
  return <CitationChip number={Number(n)} onOpen={(number, trigger) => onCite?.(number, trigger)} />
}

const COMPONENTS = {
  a: (props: ComponentProps<'a'>) => (
    <a href={props.href} target="_blank" rel="noopener noreferrer">
      {props.children}
    </a>
  ),
  pre: (props: ComponentProps<'pre'>) => <CodeBlock>{props.children}</CodeBlock>,
  table: (props: ComponentProps<'table'>) => (
    <div className="table-wrap">
      <table>{props.children}</table>
    </div>
  ),
  'citation-chip': ChipFromMarkdown,
} as unknown as Components

interface Props {
  text: string
  /** How many sources the answer has. With this and onCite, valid [n] markers become clickable chips. */
  sourceCount?: number
  onCite?: (number: number, trigger: HTMLElement) => void
  /** The answer is still being written: a soft caret follows the last word. */
  streaming?: boolean
}

/**
 * Shows an answer written in Markdown, safely.
 *
 * react-markdown builds real elements itself and does NOT render raw HTML: a "<script>" or
 * "<img onerror=...>" in the text appears as plain text. Images are also switched off, so
 * an answer can never make the browser fetch something from the internet, and links open
 * in a new tab without access to this page.
 */
export default function AnswerText({ text, sourceCount = 0, onCite, streaming = false }: Props) {
  const chips = sourceCount > 0 && onCite !== undefined
  return (
    <CiteContext.Provider value={onCite}>
      <div className={streaming ? 'answer-text streaming' : 'answer-text'}>
        <Markdown
          disallowedElements={['img']}
          unwrapDisallowed
          remarkPlugins={chips ? [remarkTables, [remarkCitations, { max: sourceCount }]] : [remarkTables]}
          components={COMPONENTS}
        >
          {text}
        </Markdown>
      </div>
    </CiteContext.Provider>
  )
}
