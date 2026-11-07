import Markdown from 'react-markdown'

/**
 * Shows an answer written in Markdown, safely.
 *
 * react-markdown builds real elements itself and does NOT render raw HTML: a "<script>" or
 * "<img onerror=...>" in the text appears as plain text. Images are also switched off, so
 * an answer can never make the browser fetch something from the internet, and links open
 * in a new tab without access to this page.
 */
export default function AnswerText({ text }: { text: string }) {
  return (
    <div className="answer-text">
      <Markdown
        disallowedElements={['img']}
        unwrapDisallowed
        components={{
          a: (props) => (
            <a href={props.href} target="_blank" rel="noopener noreferrer">
              {props.children}
            </a>
          ),
        }}
      >
        {text}
      </Markdown>
    </div>
  )
}
