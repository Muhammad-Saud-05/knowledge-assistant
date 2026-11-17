import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'
import type { SourceItem } from '../api/types'
import { askQuestion, doneEvent, installFakeServer, sse } from '../test-helpers'
import AnswerText from './AnswerText'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

const EXECUTABLE = 'script, iframe, object, embed, img, svg, style, link, meta, base, form, input[type="image"], video, audio, source'
const UNSAFE_URL = /^\s*(javascript|data|vbscript):/i

/** Nothing in this DOM can run code or load anything: no such elements, no event-handler attributes, no unsafe URLs. */
function expectInert(root: ParentNode) {
  // The app's own button icons and its composer form are the only such elements it contains.
  const ownIcon = (el: Element) => el.tagName.toLowerCase() === 'svg' && el.classList.contains('icon') && el.closest('button, .avatar') !== null
  const ownForm = (el: Element) => el.tagName === 'FORM' && el.classList.contains('composer')
  const found = [...root.querySelectorAll(EXECUTABLE)].filter((el) => !ownIcon(el) && !ownForm(el))
  expect(found.map((el) => el.tagName)).toEqual([])
  for (const element of root.querySelectorAll('*')) {
    for (const name of element.getAttributeNames()) {
      expect(name.toLowerCase().startsWith('on'), `${element.tagName} has the attribute ${name}`).toBe(false)
      if (['href', 'src', 'action', 'formaction', 'xlink:href', 'srcdoc', 'data'].includes(name.toLowerCase())) {
        expect(element.getAttribute(name) ?? '').not.toMatch(UNSAFE_URL)
      }
    }
  }
}

describe('raw HTML in an answer is shown as text, never rendered', () => {
  const CASES: [string, string][] = [
    ['an img with onerror', '<img src=x onerror=alert(1)>'],
    ['an img with quotes', '<img src="x" onerror="alert(document.cookie)">'],
    ['a script tag', '<script>alert(1)</script>'],
    ['an uppercase script tag', '<SCRIPT SRC=//evil.example/x.js></SCRIPT>'],
    ['an iframe', '<iframe src="javascript:alert(1)"></iframe>'],
    ['an svg with onload', '<svg onload=alert(1)>'],
    ['a style tag', '<style>body{background:url(//evil.example)}</style>'],
    ['an anchor with an event handler', '<a href="x" onclick="alert(1)">x</a>'],
    ['an HTML comment hiding a tag', '<!-- --><img src=x onerror=alert(1)>'],
    ['an entity-encoded tag', '&lt;script&gt;alert(1)&lt;/script&gt;'],
  ]

  it.each(CASES)('%s', (_name, html) => {
    const { container } = render(<AnswerText text={`Before ${html} after`} sourceCount={2} onCite={() => {}} />)
    expectInert(container)
    expect(container.textContent).toContain('Before')
    expect(container.textContent).toContain('after')
  })

  it('shows the dangerous text literally, so the reader can see what the answer contained', () => {
    const { container } = render(<AnswerText text="<img src=x onerror=alert(1)> and <script>alert(2)</script>" />)
    expect(container.textContent).toContain('<img src=x onerror=alert(1)>')
    expect(container.textContent).toContain('<script>alert(2)</script>')
  })

  it('is also safe while the answer is only partly written', () => {
    const { container } = render(<AnswerText text={'Streaming <img src=x onerr'} streaming />)
    expectInert(container)
    const again = render(<AnswerText text={'Streaming <scr'} streaming />)
    expectInert(again.container)
  })

  it('does not load Markdown images either', () => {
    const { container } = render(<AnswerText text={'![x](http://evil.example/p.png) ![y](data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=)'} />)
    expectInert(container)
  })

  it('does not run HTML that is wrapped in Markdown formatting, a list, a heading or a quote', () => {
    const text = '# <img src=x onerror=alert(1)>\n\n- <script>alert(1)</script>\n\n> <iframe src=x></iframe>\n\n**<svg onload=alert(1)>**'
    const { container } = render(<AnswerText text={text} />)
    expectInert(container)
  })
})

describe('links in an answer', () => {
  const BAD = [
    '[click](javascript:alert(1))',
    '[click](JaVaScRiPt:alert(1))',
    '[click](  javascript:alert(1))',
    '[click](java\tscript:alert(1))',
    '[click](&#106;avascript:alert(1))',
    '[click](data:text/html,<script>alert(1)</script>)',
    '[click](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)',
    '[click](vbscript:msgbox(1))',
    '<javascript:alert(1)>',
    '[click][ref]\n\n[ref]: javascript:alert(1)',
    '[click](//evil.example/x.js)', // protocol-relative: allowed as a link, but never as a script (checked below)
  ]

  it.each(BAD)('never produces an unsafe URL: %s', (markdown) => {
    const { container } = render(<AnswerText text={markdown} />)
    expectInert(container)
    for (const link of container.querySelectorAll('a')) expect(link.getAttribute('href') ?? '').not.toMatch(UNSAFE_URL)
  })

  it('gives an unsafe link no destination at all', () => {
    const { container } = render(<AnswerText text="[click me](javascript:alert(1))" />)
    const link = container.querySelector('a')
    expect(link?.getAttribute('href') ?? '').toBe('')
    expect(container.textContent).toContain('click me')
  })

  it('opens every link in a new tab without access to this page', () => {
    const { container } = render(
      <AnswerText text={'[a](https://example.com/a) and [b](http://example.com/b) and [c](mailto:me@example.com) and [d](/relative)'} />,
    )
    const links = [...container.querySelectorAll('a')]
    expect(links).toHaveLength(4)
    for (const link of links) {
      expect(link.getAttribute('target')).toBe('_blank')
      const rel = (link.getAttribute('rel') ?? '').split(/\s+/)
      expect(rel).toContain('noopener')
      expect(rel).toContain('noreferrer')
    }
  })
})

describe('the table plugin only ever produces table elements with text', () => {
  const TABLE_TAGS = new Set(['TABLE', 'THEAD', 'TBODY', 'TR', 'TH', 'TD', 'DIV'])

  /** Whatever the input, the answer is either an ordinary paragraph or a table made only of table elements whose
   *  cells hold only text. (A table whose cells contain inline markup is read as several pieces, so the plugin leaves
   *  it as plain paragraph text: safe, if less pretty.) */
  function expectSafeTableOrText(container: HTMLElement) {
    expectInert(container)
    if (!container.querySelector('table')) return
    for (const element of container.querySelectorAll('.answer-text *')) {
      expect(TABLE_TAGS.has(element.tagName), `unexpected <${element.tagName.toLowerCase()}> in a table answer`).toBe(true)
    }
    for (const cell of container.querySelectorAll('th, td')) {
      expect(cell.children, 'a cell holds only text').toHaveLength(0)
      expect([...cell.childNodes].every((node) => node.nodeType === Node.TEXT_NODE)).toBe(true)
    }
    for (const element of container.querySelectorAll('table, thead, tbody, tr, th, td')) {
      expect(element.getAttributeNames().filter((n) => !['style', 'align'].includes(n))).toEqual([])
    }
  }

  const MALFORMED = [
    '| a | b |\n| --- | --- |\n| <script>alert(1)</script> | <img src=x onerror=alert(1)> |',
    '| <img src=x onerror=alert(1)> | <b onclick=alert(1)>b</b> |\n| --- | --- |\n| 1 | 2 |',
    '| a | b | c |\n| --- | --- | --- |\n| only one',
    '| a |\n| --- |\n| 1 | 2 | 3 | 4 | 5 |',
    '|||\n|---|---|\n|||',
    '| a | b |\n|:---|---:|\n| javascript:alert(1) | [x](javascript:alert(1)) |',
    '| a | b |\n| --- | --- |\n| `<script>` | **bold** |',
    '| a | b |\n| --- | --- |\n|' + ' x |'.repeat(500),
    '| a | b |\n| --- | --- |\n' + '| 1 | 2 |\n'.repeat(300),
    '| a | b |\n| --- | --- |\n| javascript:alert(1) | data:text/html,<script>alert(1)</script> |',
  ]

  it.each(MALFORMED)('case %#: a safe table or plain text, nothing executable', (markdown) => {
    for (const chips of [false, true]) {
      const { container, unmount } = render(<AnswerText text={markdown} sourceCount={chips ? 3 : 0} onCite={chips ? () => {} : undefined} />)
      expectInert(container)
      if (!chips) expectSafeTableOrText(container)
      unmount()
    }
  })

  it('makes real tables out of plain-text cells, with only table elements and text', () => {
    const { container } = render(<AnswerText text={'| a | b |\n| --- | --- |\n| javascript:alert(1) | 1 < 2 > 0 |'} />)
    expect(container.querySelector('table')).toBeTruthy()
    expectSafeTableOrText(container)
    expect([...container.querySelectorAll('td')].map((c) => c.textContent)).toEqual(['javascript:alert(1)', '1 < 2 > 0'])
    expect(container.querySelector('a')).toBeNull() // not a link, only text
  })

  it('shows markup in a cell as literal text, never as elements', () => {
    const { container } = render(<AnswerText text={'| a |\n| --- |\n| <img src=x onerror=alert(1)> |'} />)
    expectInert(container)
    expect(container.textContent).toContain('<img src=x onerror=alert(1)>')
  })

  it('leaves text that is not clearly a table as an ordinary paragraph', () => {
    for (const text of ['| a | b |\n| -- | -- |\n| 1 | 2 |', '| a |\n| --- |\nnot a row', '|', '| | |\n']) {
      const { container, unmount } = render(<AnswerText text={text} />)
      expectInert(container)
      expect(container.querySelector('table')).toBeNull()
      unmount()
    }
  })

  it('may put a citation chip in a cell, and that chip is a plain numbered button', () => {
    const { container } = render(<AnswerText text={'| a |\n| --- |\n| claim [1] |'} sourceCount={2} onCite={() => {}} />)
    expectInert(container)
    const children = [...container.querySelectorAll('td *')]
    expect(children.map((e) => e.tagName)).toEqual(['BUTTON'])
    expect(children[0].textContent).toBe('1')
  })
})

describe('the whole page', () => {
  const HOSTILE: SourceItem[] = [
    {
      number: 1, chunk_id: 1, title: '<img src=x onerror=alert(1)>', path: 'x.md', page_number: null,
      section_heading: '<script>alert(2)</script>', text: '<iframe src="javascript:alert(3)"></iframe> [x](javascript:alert(4)) '.repeat(4),
      vector_rank: 1, keyword_rank: null, rerank_rank: 1, rerank_score: 1,
    },
  ]
  const ANSWER =
    'Look <img src=x onerror=alert(1)> and <script>alert(2)</script> and [click](javascript:alert(3)) [1].\n\n| a |\n| --- |\n| <svg onload=alert(4)> |'

  beforeEach(() => {
    localStorage.clear()
  })

  it('shows hostile answers and hostile source text as plain text, in the thread and in the sources panel', async () => {
    const server = installFakeServer()
    const stream = server.nextStream()
    render(<App />)
    await askQuestion(server, 'What <script>alert(0)</script> is this?')
    await stream.pipe().send(sse('sources', HOSTILE) + sse('token', { text: ANSWER }) + sse('done', doneEvent({ cited: [1] }, ANSWER)))
    await stream.pipe().end()

    fireEvent.click(screen.getByRole('button', { name: 'Source 1' })) // open the panel on the hostile source
    expect(screen.getByRole('region', { name: 'Sources' }).textContent).toContain('<img src=x onerror=alert(1)>') // as text
    expectInert(document.body)
    expect(document.body.textContent).toContain('<script>alert(2)</script>')
  })

  it('stays safe after the history is saved and loaded again', async () => {
    const server = installFakeServer()
    const stream = server.nextStream()
    const first = render(<App />)
    await askQuestion(server, 'A question')
    await stream.pipe().send(sse('sources', HOSTILE) + sse('token', { text: ANSWER }) + sse('done', doneEvent({ cited: [1] }, ANSWER)))
    await stream.pipe().end()
    first.unmount()

    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'A question' }))
    fireEvent.click(screen.getByRole('button', { name: 'Sources (1)' }))
    expectInert(document.body)
  })
})
