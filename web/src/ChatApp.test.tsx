import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { HISTORY_KEY, THEME_KEY } from './storage'
import { SUGGESTIONS } from './suggestions'
import { COMPOSER_HINT, EMPTY_HEADING } from './texts'
import { SOURCES, askAndFinish, askQuestion, doneEvent, installFakeServer, questionBox, sse } from './test-helpers'
import type { FakeServer } from './test-helpers'

let server: FakeServer

beforeEach(() => {
  localStorage.clear()
  document.documentElement.removeAttribute('data-theme')
  server = installFakeServer()
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

const sidebar = () => within(screen.getByRole('complementary', { name: 'Sidebar' }))
const thread = () => within(document.querySelector('.thread') as HTMLElement)
const turnCount = () => document.querySelectorAll('.thread article.turn').length

function mockMatchMedia(matching: (query: string) => boolean) {
  vi.stubGlobal(
    'matchMedia',
    vi.fn((query: string) => ({
      matches: matching(query),
      media: query,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
    })),
  )
}

// -------------------------------------------------------------------- empty state

describe('the empty state', () => {
  it('shows the heading, the suggestion chips and the composer hint', () => {
    render(<App />)
    expect(screen.getByRole('heading', { name: EMPTY_HEADING })).toBeTruthy()
    expect(SUGGESTIONS).toHaveLength(3)
    for (const text of SUGGESTIONS) expect(screen.getByRole('button', { name: text })).toBeTruthy()
    expect(screen.getByText(COMPOSER_HINT)).toBeTruthy()
    expect(COMPOSER_HINT).toMatch(/Each question is answered on its own/) // the back end keeps no memory
  })

  it('sends a suggestion when its chip is clicked', async () => {
    server.nextStream()
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: SUGGESTIONS[1] }))
    await waitFor(() => expect(server.askCalls).toHaveLength(1))
    expect(server.askCalls[0].body).toEqual({ question: SUGGESTIONS[1] })
    expect(thread().getByText(SUGGESTIONS[1])).toBeTruthy()
    expect(screen.queryByRole('heading', { name: EMPTY_HEADING })).toBeNull()
  })
})

// ------------------------------------------------------------------ conversations

describe('conversations', () => {
  it('"New chat" starts a fresh thread and keeps the old one in the history', async () => {
    render(<App />)
    await askAndFinish(server, 'First question', 'First answer.')
    expect(sidebar().getByRole('button', { name: 'First question' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'New chat' }))
    expect(screen.getByRole('heading', { name: EMPTY_HEADING })).toBeTruthy()
    expect(turnCount()).toBe(0)

    await askAndFinish(server, 'Second question', 'Second answer.')
    expect(turnCount()).toBe(1)
    expect(sidebar().getByRole('button', { name: 'First question' })).toBeTruthy()
    expect(sidebar().getByRole('button', { name: 'Second question' })).toBeTruthy()
    expect(sidebar().getByRole('button', { name: 'Second question' }).getAttribute('aria-current')).toBe('true') // the active one
  })

  it('selecting an old conversation shows it, and a new question is added to the same thread', async () => {
    render(<App />)
    await askAndFinish(server, 'First question', 'First answer.')
    fireEvent.click(screen.getByRole('button', { name: 'New chat' }))
    await askAndFinish(server, 'Second question', 'Second answer.')

    fireEvent.click(sidebar().getByRole('button', { name: 'First question' }))
    expect(thread().getByText('First answer.')).toBeTruthy()
    expect(thread().queryByText('Second answer.')).toBeNull()

    await askAndFinish(server, 'Third question', 'Third answer.')
    expect(turnCount()).toBe(2) // First and Third, in one thread
    expect(thread().getByText('First answer.')).toBeTruthy()
    expect(thread().getByText('Third answer.')).toBeTruthy()
    expect(document.querySelectorAll('.conv-item')).toHaveLength(2) // still two conversations
    // each question was still sent on its own: only that question, nothing from the thread
    expect(server.askCalls[2].body).toEqual({ question: 'Third question' })
  })

  it('titles a conversation with its first question, trimmed', async () => {
    render(<App />)
    await askAndFinish(server, `${'long '.repeat(40)}question`)
    const title = document.querySelector('.conv-open')!.textContent!
    expect(title.endsWith('...')).toBe(true)
    expect(title.length).toBeLessThanOrEqual(64)
  })

  it('groups the history under Today', async () => {
    render(<App />)
    await askAndFinish(server, 'First question')
    expect(sidebar().getByRole('region', { name: 'Today' })).toBeTruthy()
  })

  it('deletes one conversation and leaves the others', async () => {
    render(<App />)
    await askAndFinish(server, 'First question')
    fireEvent.click(screen.getByRole('button', { name: 'New chat' }))
    await askAndFinish(server, 'Second question')

    fireEvent.click(sidebar().getByRole('button', { name: 'Delete conversation: First question' }))
    expect(sidebar().queryByRole('button', { name: 'First question' })).toBeNull()
    expect(sidebar().getByRole('button', { name: 'Second question' })).toBeTruthy()
    expect(localStorage.getItem(HISTORY_KEY)).not.toContain('First question')
    expect(localStorage.getItem(HISTORY_KEY)).toContain('Second question')
  })

  it('deleting the open conversation returns to an empty chat', async () => {
    render(<App />)
    await askAndFinish(server, 'Only question')
    fireEvent.click(sidebar().getByRole('button', { name: 'Delete conversation: Only question' }))
    expect(screen.getByRole('heading', { name: EMPTY_HEADING })).toBeTruthy()
    expect(sidebar().getByText('Your conversations will appear here.')).toBeTruthy()
  })

  it('clears all history only after an in-page confirmation (no browser dialogs)', async () => {
    const confirm = vi.spyOn(window, 'confirm')
    const alert = vi.spyOn(window, 'alert')
    render(<App />)
    await askAndFinish(server, 'First question')
    fireEvent.click(screen.getByRole('button', { name: 'New chat' }))
    await askAndFinish(server, 'Second question')

    fireEvent.click(screen.getByRole('button', { name: 'Clear all history' }))
    const group = screen.getByRole('group', { name: 'Confirm clearing history' })
    expect(within(group).getByText(/Delete all conversations from this browser\?/)).toBeTruthy()
    expect(document.querySelectorAll('.conv-item')).toHaveLength(2) // nothing deleted yet

    fireEvent.click(within(group).getByRole('button', { name: 'Cancel' }))
    expect(screen.queryByRole('group', { name: 'Confirm clearing history' })).toBeNull()
    expect(document.querySelectorAll('.conv-item')).toHaveLength(2)

    fireEvent.click(screen.getByRole('button', { name: 'Clear all history' }))
    fireEvent.click(screen.getByRole('button', { name: 'Yes, clear all' }))
    expect(document.querySelectorAll('.conv-item')).toHaveLength(0)
    expect(screen.getByRole('heading', { name: EMPTY_HEADING })).toBeTruthy()
    expect(JSON.parse(localStorage.getItem(HISTORY_KEY) ?? '{"conversations":[]}').conversations).toEqual([])
    expect(confirm).not.toHaveBeenCalled()
    expect(alert).not.toHaveBeenCalled()
  })

  it('says under "Clear all history" that history stays on this device', () => {
    render(<App />)
    expect(sidebar().getByText('History stays on this device.')).toBeTruthy()
  })
})

// ----------------------------------------------------------------------- persistence

describe('history in this browser', () => {
  it('survives a reload of the page', async () => {
    const first = render(<App />)
    await askAndFinish(server, 'Remember me', 'A remembered answer [1].')
    first.unmount()

    render(<App />)
    expect(screen.getByRole('heading', { name: EMPTY_HEADING })).toBeTruthy() // a reload opens a fresh chat
    fireEvent.click(sidebar().getByRole('button', { name: 'Remember me' }))
    expect(thread().getByText(/A remembered answer/)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Sources (3)' })).toBeTruthy() // the sources came back too
  })

  it('shows an answer that was cut off by closing the page as interrupted', () => {
    localStorage.setItem(
      HISTORY_KEY,
      JSON.stringify({
        version: 1,
        conversations: [
          {
            id: 'x', title: 'Cut off', createdAt: Date.now(), updatedAt: Date.now(),
            turns: [{ id: 1, question: 'Cut off', compare: false, status: 'streaming', text: 'Half an ans', sources: [], done: null, error: null }],
          },
        ],
      }),
    )
    render(<App />)
    fireEvent.click(sidebar().getByRole('button', { name: 'Cut off' }))
    expect(screen.getByText('Half an ans')).toBeTruthy()
    expect(screen.getByRole('alert').textContent).toContain('stopped before it was finished')
    expect(screen.getByRole('button', { name: 'Try again' })).toBeTruthy()
  })

  it('keeps working in memory when storage is unavailable, and says so', async () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    render(<App />)
    expect(sidebar().getByText(/can't be saved in this browser/)).toBeTruthy()
    await askAndFinish(server, 'In memory only', 'Still answered.')
    expect(thread().getByText('Still answered.')).toBeTruthy()
    expect(sidebar().getByRole('button', { name: 'In memory only' })).toBeTruthy() // in the list while the page is open

    fireEvent.click(screen.getByRole('button', { name: 'New chat' }))
    fireEvent.click(sidebar().getByRole('button', { name: 'In memory only' }))
    expect(thread().getByText('Still answered.')).toBeTruthy()
  })

  it('notices when saving fails later (for example the storage is full)', async () => {
    render(<App />)
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('full', 'QuotaExceededError')
    })
    await askAndFinish(server, 'Too big to keep')
    await waitFor(() => expect(sidebar().getByText(/can't be saved in this browser/)).toBeTruthy())
    expect(sidebar().getByRole('button', { name: 'Too big to keep' })).toBeTruthy()
  })
})

// ---------------------------------------------------------------------------- composer

describe('the composer', () => {
  it('Enter sends, Shift+Enter inserts a line break, and composing text does not send', () => {
    server.nextStream()
    render(<App />)
    fireEvent.change(questionBox(), { target: { value: 'A line' } })

    expect(fireEvent.keyDown(questionBox(), { key: 'Enter', shiftKey: true })).toBe(true) // not prevented: the browser adds the new line
    expect(fireEvent.keyDown(questionBox(), { key: 'Enter', isComposing: true })).toBe(true)
    expect(server.askCalls).toHaveLength(0)

    expect(fireEvent.keyDown(questionBox(), { key: 'Enter' })).toBe(false) // prevented: it sends instead
    expect(server.askCalls).toHaveLength(1)
    expect(questionBox().value).toBe('')
  })

  it('swaps Send for Stop while an answer is being written, and back afterwards', async () => {
    const stream = server.nextStream()
    render(<App />)
    expect((screen.getByRole('button', { name: 'Send' }) as HTMLButtonElement).disabled).toBe(true) // empty box
    await askQuestion(server, 'A question')
    expect(screen.queryByRole('button', { name: 'Send' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Stop' })).toBeTruthy()

    await stream.pipe().send(sse('token', { text: 'Words' }) + sse('done', doneEvent({}, 'Words')))
    await stream.pipe().end()
    expect(screen.queryByRole('button', { name: 'Stop' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Send' })).toBeTruthy()
  })

  it('keeps the "Compare across documents (slower)" toggle under the text box and sends the split mode', async () => {
    server.nextStream()
    render(<App />)
    const toggle = screen.getByLabelText('Compare across documents (slower)') as HTMLInputElement
    expect(toggle.checked).toBe(false)
    expect(toggle.closest('.composer')).toBeTruthy()
    fireEvent.click(toggle)
    await askQuestion(server, 'Two things?')
    expect(server.askCalls[0].body).toEqual({ question: 'Two things?', mode: 'rerank-split-llm' })
  })
})

// ------------------------------------------------------------ citations and the panel

describe('citation chips and the sources panel', () => {
  async function answerWithCitations(text = 'A claim [2] and another [3]. Odd [9].', cited = [2, 3]) {
    const stream = server.nextStream()
    render(<App />)
    await askQuestion(server, 'Which sources?')
    await stream.pipe().send(sse('sources', SOURCES) + sse('token', { text }) + sse('done', doneEvent({ cited }, text)))
    await stream.pipe().end()
  }

  it('turns valid [n] markers into chips and leaves out-of-range ones as text', async () => {
    await answerWithCitations()
    expect(screen.getByRole('button', { name: 'Source 2' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Source 3' })).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Source 9' })).toBeNull()
    expect(thread().getByText(/Odd \[9\]/)).toBeTruthy()
    expect(screen.queryByRole('region', { name: 'Sources' })).toBeNull() // nothing opens by itself
  })

  it('opens the panel on the clicked source and highlights it', async () => {
    await answerWithCitations()
    fireEvent.click(screen.getByRole('button', { name: 'Source 3' }))

    const panel = screen.getByRole('region', { name: 'Sources' })
    const cards = within(panel).getAllByRole('listitem')
    expect(cards).toHaveLength(3)
    expect(cards[2].classList.contains('highlight')).toBe(true)
    expect(cards[2].getAttribute('aria-current')).toBe('true')
    expect(cards[0].classList.contains('highlight')).toBe(false)
    expect(within(cards[2]).getByText('Fixed Window Notes')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Source 2' })) // another chip moves the highlight
    expect(within(panel).getAllByRole('listitem')[1].classList.contains('highlight')).toBe(true)
    expect(within(panel).getAllByRole('listitem')[2].classList.contains('highlight')).toBe(false)
  })

  it('shows only fields the server sent: rerank rank and score when present, nothing invented when absent', async () => {
    await answerWithCitations()
    fireEvent.click(screen.getByRole('button', { name: 'Source 2' }))
    const cards = within(screen.getByRole('region', { name: 'Sources' })).getAllByRole('listitem')
    expect(within(cards[0]).getByText(/rerank #1 · score 6\.42/)).toBeTruthy()
    expect(within(cards[1]).getByText('p. 5 - 4. Retrieval')).toBeTruthy()
    expect(within(cards[2]).queryByText(/rerank/)).toBeNull() // that source had no rerank fields
    expect(within(cards[2]).queryByText(/p\. /)).toBeNull() // and no page
    expect(within(cards[0]).queryAllByText('cited')).toHaveLength(0) // [1] is not among the cited
    expect(within(cards[1]).getByText('cited')).toBeTruthy()
  })

  it('shows a long passage shortened, with a button to see all of it', async () => {
    await answerWithCitations()
    fireEvent.click(screen.getByRole('button', { name: 'Source 2' }))
    const first = within(screen.getByRole('region', { name: 'Sources' })).getAllByRole('listitem')[0]
    const text = first.querySelector('.source-text')!
    expect(text.textContent!.endsWith('...')).toBe(true)
    fireEvent.click(within(first).getByRole('button', { name: 'Show more' }))
    expect(text.textContent).toBe(SOURCES[0].text)
    fireEvent.click(within(first).getByRole('button', { name: 'Show less' }))
    expect(text.textContent!.endsWith('...')).toBe(true)
  })

  it('closes with Escape and gives the focus back to the chip that opened it', async () => {
    await answerWithCitations()
    const chip = screen.getByRole('button', { name: 'Source 2' })
    chip.focus()
    fireEvent.click(chip)
    const close = screen.getByRole('button', { name: 'Close sources' })
    expect(document.activeElement).toBe(close) // focus moves into the panel

    fireEvent.keyDown(document, { key: 'Escape' })
    expect(screen.queryByRole('region', { name: 'Sources' })).toBeNull()
    expect(document.activeElement).toBe(chip)
  })

  it('closes with its close button and returns the focus', async () => {
    await answerWithCitations()
    const chip = screen.getByRole('button', { name: 'Source 3' })
    fireEvent.click(chip)
    fireEvent.click(screen.getByRole('button', { name: 'Close sources' }))
    expect(screen.queryByRole('region', { name: 'Sources' })).toBeNull()
    expect(document.activeElement).toBe(chip)
  })

  it('the "Sources (n)" button opens and closes the panel', async () => {
    await answerWithCitations()
    const toggle = screen.getByRole('button', { name: 'Sources (3)' })
    expect(toggle.getAttribute('aria-expanded')).toBe('false')
    fireEvent.click(toggle)
    expect(screen.getByRole('region', { name: 'Sources' })).toBeTruthy()
    expect(toggle.getAttribute('aria-expanded')).toBe('true')
    expect(within(screen.getByRole('region', { name: 'Sources' })).getAllByRole('listitem').some((c) => c.classList.contains('highlight'))).toBe(false)
    fireEvent.click(toggle)
    expect(screen.queryByRole('region', { name: 'Sources' })).toBeNull()
  })

  it('has the sources toggle as soon as the sources arrive, before any word of the answer', async () => {
    const stream = server.nextStream()
    render(<App />)
    await askQuestion(server, 'Q')
    expect(screen.queryByRole('button', { name: /^Sources/ })).toBeNull()
    await stream.pipe().send(sse('sources', SOURCES))
    expect(screen.getByRole('button', { name: 'Sources (3)' })).toBeTruthy()
    expect(screen.getByText(/Reading your sources/)).toBeTruthy()
    await stream.pipe().end()
  })

  it('does not make chips out of code or links', async () => {
    await answerWithCitations('Use `values[1]` or [a link [1]](https://example.com).\n\n```\nx = [1]\n```\n\nReal [2].', [2])
    expect(screen.getAllByRole('button', { name: /^Source \d$/ })).toHaveLength(1) // only the real [2]
    expect(screen.getByRole('button', { name: 'Source 2' })).toBeTruthy()
  })
})

// ---------------------------------------------------------------------------- the states

describe('the answer states', () => {
  async function finishWith(done: Parameters<typeof doneEvent>[0], text: string) {
    const stream = server.nextStream()
    render(<App />)
    await askQuestion(server, 'Q')
    await stream.pipe().send(sse('sources', SOURCES) + sse('token', { text }) + sse('done', doneEvent(done, text)))
    await stream.pipe().end()
  }

  it('shows a refusal as a calm note, without a warning badge', async () => {
    await finishWith({ refused: true, cited: [], uncited: false }, "I couldn't find this in your documents.")
    expect(screen.getByRole('note').textContent).toMatch(/don't seem to contain an answer/)
    expect(document.querySelector('.flag')).toBeNull()
    expect(document.querySelector('.notice.warning')).toBeNull()
  })

  it('shows the unverified badge when an answer cites no source, and keeps the answer and sources', async () => {
    await finishWith({ cited: [], uncited: true }, 'A claim with nothing behind it.')
    expect(screen.getByText('Unverified: this answer cites no source')).toBeTruthy()
    expect(screen.getByText('A claim with nothing behind it.')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Sources (3)' })).toBeTruthy()
  })

  it('shows the invalid-citation badge', async () => {
    await finishWith({ invalid_citations: ['0', '9'] }, 'A claim [1] and [9].')
    expect(screen.getByText("Some citation numbers don't match a source")).toBeTruthy()
  })

  it('shows an error with a Try again button', async () => {
    server.onAsk = () => Promise.reject(new TypeError('Failed to fetch'))
    render(<App />)
    await askQuestion(server, 'Q')
    expect((await screen.findByRole('alert')).textContent).toMatch(/couldn't reach the server/)
    expect(screen.getByRole('button', { name: 'Try again' })).toBeTruthy()
  })
})

// ---------------------------------------------------------------------- message actions

describe('actions under an answer', () => {
  it('copies the answer', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
    render(<App />)
    await askAndFinish(server, 'Q', 'The whole answer.')
    fireEvent.click(screen.getByRole('button', { name: 'Copy answer' }))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith('The whole answer.'))
    expect(await screen.findByRole('button', { name: 'Copied' })).toBeTruthy()
  })

  it('copies the code of a code block with its own button', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true })
    render(<App />)
    await askAndFinish(server, 'Q', 'Use this:\n\n```python\nprint("hi")\n```\n')
    fireEvent.click(screen.getByRole('button', { name: 'Copy code' }))
    await waitFor(() => expect(writeText).toHaveBeenCalledWith('print("hi")'))
    expect(document.querySelector('.codebar span')?.textContent).toBe('python')
  })

  it('regenerates the answer in the same place', async () => {
    render(<App />)
    await askAndFinish(server, 'Ask me again', 'First try.')
    const stream = server.nextStream()
    fireEvent.click(screen.getByRole('button', { name: 'Regenerate answer' }))
    await waitFor(() => expect(server.askCalls).toHaveLength(2))
    expect(server.askCalls[1].body).toEqual({ question: 'Ask me again' })
    expect(turnCount()).toBe(1) // the same turn, not a second one
    expect(screen.queryByText('First try.')).toBeNull()
    await stream.pipe().send(sse('token', { text: 'Second try.' }) + sse('done', doneEvent({}, 'Second try.')))
    await stream.pipe().end()
    expect(screen.getByText('Second try.')).toBeTruthy()
  })
})

// ---------------------------------------------------------------------------- scrolling

describe('scrolling', () => {
  function scroller() {
    const element = document.querySelector('.thread-scroll') as HTMLElement
    let top = 0
    Object.defineProperty(element, 'scrollHeight', { value: 2000, configurable: true })
    Object.defineProperty(element, 'clientHeight', { value: 500, configurable: true })
    function setTop(value: number) {
      top = value
    }
    Object.defineProperty(element, 'scrollTop', { get: () => top, set: setTop, configurable: true })
    return { element, top: () => top, set: setTop }
  }

  it('follows the newest words while the reader is at the bottom', async () => {
    const stream = server.nextStream()
    render(<App />)
    await askQuestion(server, 'Q')
    const s = scroller()
    s.set(1500) // at the bottom (2000 - 500)
    fireEvent.scroll(s.element)
    await stream.pipe().send(sse('token', { text: 'more words' }))
    expect(s.top()).toBe(2000)
    await stream.pipe().end()
  })

  it('stops following when the reader scrolls up, and offers "Jump to latest"', async () => {
    const stream = server.nextStream()
    render(<App />)
    await askQuestion(server, 'Q')
    const s = scroller()
    s.set(0)
    fireEvent.scroll(s.element)
    expect(screen.getByRole('button', { name: 'Jump to latest' })).toBeTruthy()

    await stream.pipe().send(sse('token', { text: 'words that arrive meanwhile' }))
    expect(s.top()).toBe(0) // not dragged back down

    fireEvent.click(screen.getByRole('button', { name: 'Jump to latest' }))
    expect(s.top()).toBe(2000)
    expect(screen.queryByRole('button', { name: 'Jump to latest' })).toBeNull()
    await stream.pipe().end()
  })
})

// --------------------------------------------------------------------------- the theme

describe('the theme', () => {
  it('toggles, applies the choice to the page and remembers it', () => {
    const first = render(<App />)
    expect(document.documentElement.dataset.theme).toBeUndefined() // follows the system by default
    fireEvent.click(screen.getByRole('button', { name: 'Switch to dark theme' }))
    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(localStorage.getItem(THEME_KEY)).toBe('dark')
    expect(screen.getByRole('button', { name: 'Switch to light theme' })).toBeTruthy()

    first.unmount()
    document.documentElement.removeAttribute('data-theme')
    render(<App />)
    expect(document.documentElement.dataset.theme).toBe('dark') // restored
    fireEvent.click(screen.getByRole('button', { name: 'Switch to light theme' }))
    expect(document.documentElement.dataset.theme).toBe('light')
    expect(localStorage.getItem(THEME_KEY)).toBe('light')
  })

  it('starts from the system setting when nothing was chosen', () => {
    mockMatchMedia((query) => query.includes('prefers-color-scheme: dark'))
    render(<App />)
    expect(screen.getByRole('button', { name: 'Switch to light theme' })).toBeTruthy()
    expect(document.documentElement.dataset.theme).toBeUndefined()
  })

  it('still toggles when storage is unavailable', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('blocked')
    })
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Switch to dark theme' }))
    expect(document.documentElement.dataset.theme).toBe('dark')
  })
})

// ------------------------------------------------------------------------ the sidebar

describe('the sidebar on a wide screen', () => {
  it('is open, and can be hidden and shown again', () => {
    render(<App />)
    const aside = screen.getByRole('complementary', { name: 'Sidebar' })
    expect(aside.classList.contains('collapsed')).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: 'Hide sidebar' }))
    expect(aside.classList.contains('collapsed')).toBe(true)
    expect(aside.hasAttribute('inert')).toBe(true) // not reachable with the keyboard while hidden
    fireEvent.click(screen.getByRole('button', { name: 'Show sidebar' }))
    expect(aside.classList.contains('collapsed')).toBe(false)
  })
})

describe('the sidebar drawer on a phone', () => {
  beforeEach(() => mockMatchMedia((query) => query.includes('max-width: 767px')))

  it('starts closed and opens from the menu button', () => {
    render(<App />)
    const aside = screen.getByRole('complementary', { name: 'Sidebar' })
    expect(aside.classList.contains('open')).toBe(false)
    expect(aside.hasAttribute('inert')).toBe(true)
    expect(document.querySelector('.backdrop')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Show sidebar' }))
    expect(aside.classList.contains('open')).toBe(true)
    expect(aside.hasAttribute('inert')).toBe(false)
    expect(document.querySelector('.backdrop')).toBeTruthy()
  })

  it('closes with Escape and returns the focus to the menu button', () => {
    render(<App />)
    const menu = screen.getByRole('button', { name: 'Show sidebar' })
    menu.focus()
    fireEvent.click(menu)
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(screen.getByRole('complementary', { name: 'Sidebar' }).classList.contains('open')).toBe(false)
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Show sidebar' }))
  })

  it('closes with its close button and with the backdrop', () => {
    render(<App />)
    fireEvent.click(screen.getByRole('button', { name: 'Show sidebar' }))
    fireEvent.click(screen.getByRole('button', { name: 'Close menu' }))
    expect(screen.getByRole('complementary', { name: 'Sidebar' }).classList.contains('open')).toBe(false)

    fireEvent.click(screen.getByRole('button', { name: 'Show sidebar' }))
    fireEvent.click(document.querySelector('.backdrop') as HTMLElement)
    expect(screen.getByRole('complementary', { name: 'Sidebar' }).classList.contains('open')).toBe(false)
  })

  it('closes after picking a conversation or starting a new chat', async () => {
    render(<App />)
    await askAndFinish(server, 'Phone question')
    fireEvent.click(screen.getByRole('button', { name: 'Show sidebar' }))
    fireEvent.click(sidebar().getByRole('button', { name: 'Phone question' }))
    expect(screen.getByRole('complementary', { name: 'Sidebar' }).classList.contains('open')).toBe(false)

    fireEvent.click(screen.getByRole('button', { name: 'Show sidebar' }))
    fireEvent.click(screen.getByRole('button', { name: 'New chat' }))
    expect(screen.getByRole('complementary', { name: 'Sidebar' }).classList.contains('open')).toBe(false)
    expect(screen.getByRole('heading', { name: EMPTY_HEADING })).toBeTruthy()
  })
})

// ----------------------------------------------------------------------- accessibility

describe('accessibility basics', () => {
  it('has a live region only for the waiting line, not for streamed text', async () => {
    const stream = server.nextStream()
    render(<App />)
    await askQuestion(server, 'Q')
    await stream.pipe().send(sse('sources', SOURCES))
    const live = [...document.querySelectorAll('[aria-live]')].filter((el) => el.getAttribute('aria-live') !== 'off')
    expect(live).toHaveLength(1)
    expect(live[0].textContent).toMatch(/Reading your sources/)

    await stream.pipe().send(sse('token', { text: 'Streaming words' }))
    expect(document.querySelector('.answer-text')!.closest('[aria-live]:not([aria-live="off"])')).toBeNull()
    await act(async () => stream.pipe().end())
  })

  it('gives icon-only buttons accessible names', () => {
    render(<App />)
    for (const button of screen.getAllByRole('button')) {
      expect((button.textContent ?? '').trim() || button.getAttribute('aria-label') || button.getAttribute('title')).toBeTruthy()
    }
  })
})
