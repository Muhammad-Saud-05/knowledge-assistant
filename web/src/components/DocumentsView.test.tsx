import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'
import type { DocumentItem, Health, IngestResult } from '../api/types'
import { EMPTY_MESSAGE, describeUpload } from '../documents'
import DocumentsView from './DocumentsView'

const DOCS: DocumentItem[] = [
  { id: 7, title: 'Rate Limiting Notes', source_type: 'markdown', display_name: 'rate.md', doc_date: '2026-10-05', ingested_at: '2026-11-14T08:00:00Z', chunk_count: 3 },
  { id: 8, title: 'A Beginner Guide', source_type: 'pdf', display_name: 'guide.pdf', doc_date: null, ingested_at: '2026-11-14T08:01:00Z', chunk_count: 12 },
]

const HEALTHY: Health = {
  status: 'ok',
  database: { ok: true, documents: 2, chunks: 15, detail: null },
  ollama: { ok: true, model: 'llama3.2:3b', model_installed: true, detail: null },
}

type Call = { method: string; url: string }
let calls: Call[]
let documents: DocumentItem[]
let listFails: boolean
let onUpload: () => Response | Promise<Response>
let onDelete: (id: number) => Response | Promise<Response>

function json(body: unknown, status = 200) {
  return Response.json(body, { status })
}

beforeEach(() => {
  localStorage.clear()
  calls = []
  documents = [...DOCS]
  listFails = false
  onUpload = () => json({ status: 'new', chunks: 4, document_id: 9, filename: 'notes.md', error: null } satisfies IngestResult)
  onDelete = (id) => {
    documents = documents.filter((d) => d.id !== id)
    return new Response(null, { status: 204 })
  }
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      const method = init?.method ?? 'GET'
      calls.push({ method, url })
      if (url === '/api/health') return json(HEALTHY)
      if (url === '/api/documents' && method === 'GET') return listFails ? json({ detail: 'nope' }, 500) : json(documents)
      if (url === '/api/ingest' && method === 'POST') return onUpload()
      const match = /^\/api\/documents\/(\d+)$/.exec(url)
      if (match && method === 'DELETE') return onDelete(Number(match[1]))
      throw new Error(`unexpected request: ${method} ${url}`)
    }),
  )
  vi.spyOn(window, 'confirm').mockReturnValue(true)
  vi.spyOn(window, 'alert').mockImplementation(() => {})
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

const listCalls = () => calls.filter((c) => c.method === 'GET' && c.url === '/api/documents').length

function chooseFile(name = 'notes.md') {
  const input = screen.getByLabelText('Choose a file to upload')
  fireEvent.change(input, { target: { files: [new File(['# Notes\n\nhello'], name, { type: 'text/markdown' })] } })
}

describe('the document list', () => {
  it('shows title, type, date and chunks for each document, and only file names', async () => {
    render(<DocumentsView />)
    const table = await screen.findByRole('table')
    const rows = within(table).getAllByRole('row')
    expect(rows).toHaveLength(3) // header + 2 documents
    const first = within(rows[1])
    expect(first.getByText('Rate Limiting Notes')).toBeTruthy()
    expect(first.getByText('rate.md')).toBeTruthy()
    expect(first.getByText('Markdown')).toBeTruthy()
    expect(first.getByText('2026-10-05')).toBeTruthy()
    expect(first.getByText('3')).toBeTruthy()
    const second = within(rows[2])
    expect(second.getByText('PDF')).toBeTruthy()
    expect(second.getByText('-')).toBeTruthy() // no date
    expect(second.getByText('12')).toBeTruthy()
    expect(table.textContent).not.toMatch(/[\\/]/) // no path anywhere
  })

  it('shows an empty state when there are no documents', async () => {
    documents = []
    render(<DocumentsView />)
    expect(await screen.findByText(EMPTY_MESSAGE)).toBeTruthy()
    expect(screen.queryByRole('table')).toBeNull()
    expect(screen.getByRole('button', { name: 'Upload a document' })).toBeTruthy()
  })

  it('explains a failure to load, with a button that tries again', async () => {
    listFails = true
    render(<DocumentsView />)
    const alert = await screen.findByRole('alert')
    expect(alert.textContent).toContain('The server ran into a problem')
    listFails = false
    fireEvent.click(within(alert).getByRole('button', { name: 'Try again' }))
    expect(await screen.findByRole('table')).toBeTruthy()
    expect(screen.queryByRole('alert')).toBeNull()
  })
})

describe('uploading', () => {
  it('shows progress while the file is being added, then the result, and refreshes the list', async () => {
    let finish!: (r: Response) => void
    onUpload = () => new Promise<Response>((resolve) => (finish = resolve))
    render(<DocumentsView />)
    await screen.findByRole('table')
    expect(listCalls()).toBe(1)

    chooseFile('notes.md')
    expect((await screen.findByRole('status')).textContent).toMatch(/Uploading and indexing notes\.md/)
    expect((screen.getByRole('button', { name: 'Upload a document' }) as HTMLButtonElement).disabled).toBe(true)

    documents = [...DOCS, { ...DOCS[0], id: 9, title: 'Notes', display_name: 'notes.md', chunk_count: 4 }]
    await act(async () => finish(json({ status: 'new', chunks: 4, document_id: 9, filename: 'notes.md', error: null })))
    expect(await screen.findByText('Added notes.md (4 passages).')).toBeTruthy()
    await waitFor(() => expect(listCalls()).toBe(2))
    expect(await screen.findByText('Notes')).toBeTruthy()
    expect((screen.getByRole('button', { name: 'Upload a document' }) as HTMLButtonElement).disabled).toBe(false)
  })

  it('shows the server\'s message when the upload is refused', async () => {
    onUpload = () => json({ detail: 'Only .pdf and .md files can be uploaded.' }, 415)
    render(<DocumentsView />)
    await screen.findByRole('table')
    chooseFile('picture.png')
    const alert = await screen.findByRole('alert')
    expect(alert.textContent).toBe('Only .pdf and .md files can be uploaded.')
    expect(screen.queryByText(/Uploading and indexing/)).toBeNull()
    await waitFor(() => expect(listCalls()).toBe(2))
  })

  it('reports a file that could not be indexed (a failed result) as an error', async () => {
    onUpload = () => json({ status: 'failed', chunks: 0, document_id: null, filename: 'bad.pdf', error: 'The file could not be read or stored.' })
    render(<DocumentsView />)
    await screen.findByRole('table')
    chooseFile('bad.pdf')
    expect((await screen.findByRole('alert')).textContent).toBe('The file could not be read or stored.')
  })

  it('explains a network failure in plain words', async () => {
    onUpload = () => {
      throw new TypeError('Failed to fetch')
    }
    render(<DocumentsView />)
    await screen.findByRole('table')
    chooseFile()
    expect((await screen.findByRole('alert')).textContent).toMatch(/couldn't reach the server/)
  })

  it('describes every kind of result', () => {
    const base = { chunks: 5, document_id: 1, filename: 'a.md', error: null }
    expect(describeUpload({ ...base, status: 'new' }).text).toBe('Added a.md (5 passages).')
    expect(describeUpload({ ...base, status: 'updated' }).text).toBe('Updated a.md (5 passages).')
    expect(describeUpload({ ...base, status: 'unchanged' }).text).toMatch(/already stored/)
    expect(describeUpload({ ...base, status: 'failed', error: null })).toEqual({ kind: 'error', text: 'a.md could not be added.' })
  })
})

describe('deleting', () => {
  async function openConfirmation(title = 'Rate Limiting Notes') {
    render(<DocumentsView />)
    await screen.findByRole('table')
    fireEvent.click(screen.getByRole('button', { name: `Delete ${title}` }))
    return screen.getByRole('group', { name: `Confirm deleting ${title}` })
  }

  it('asks in the page first and does not delete until confirmed (no browser dialogs)', async () => {
    const group = await openConfirmation()
    expect(group.textContent).toMatch(/Delete this document and its 3 chunks\? This cannot be undone\./)
    expect(calls.some((c) => c.method === 'DELETE')).toBe(false)
    expect(window.confirm).not.toHaveBeenCalled()
    expect(window.alert).not.toHaveBeenCalled()
  })

  it('cancelling leaves the document alone and sends nothing', async () => {
    const group = await openConfirmation()
    fireEvent.click(within(group).getByRole('button', { name: 'Cancel' }))
    expect(screen.queryByRole('group')).toBeNull()
    expect(screen.getByText('Rate Limiting Notes')).toBeTruthy()
    expect(calls.some((c) => c.method === 'DELETE')).toBe(false)
  })

  it('confirming deletes that one document, refreshes the list and says so', async () => {
    const group = await openConfirmation()
    fireEvent.click(within(group).getByRole('button', { name: 'Yes, delete' }))
    expect(await screen.findByText('Deleted Rate Limiting Notes.')).toBeTruthy()
    expect(calls.filter((c) => c.method === 'DELETE')).toEqual([{ method: 'DELETE', url: '/api/documents/7' }])
    await waitFor(() => expect(screen.queryByText('rate.md')).toBeNull())
    expect(screen.getByText('A Beginner Guide')).toBeTruthy() // the other one stays
    expect(window.confirm).not.toHaveBeenCalled()
  })

  it('shows the last document being deleted as the empty state', async () => {
    documents = [DOCS[0]]
    const group = await openConfirmation()
    fireEvent.click(within(group).getByRole('button', { name: 'Yes, delete' }))
    expect(await screen.findByText(EMPTY_MESSAGE)).toBeTruthy()
  })

  it('shows a message when the document is already gone, and refreshes the list', async () => {
    onDelete = () => {
      documents = documents.filter((d) => d.id !== 7)
      return json({ detail: 'That document does not exist.' }, 404)
    }
    const group = await openConfirmation()
    fireEvent.click(within(group).getByRole('button', { name: 'Yes, delete' }))
    expect((await screen.findByRole('alert')).textContent).toBe('That document does not exist.')
    await waitFor(() => expect(screen.queryByText('rate.md')).toBeNull())
  })

  it('shows a server error without technical detail and keeps the document', async () => {
    onDelete = () => json({ detail: 'Traceback (most recent call last)' }, 500)
    const group = await openConfirmation()
    fireEvent.click(within(group).getByRole('button', { name: 'Yes, delete' }))
    const alert = await screen.findByRole('alert')
    expect(alert.textContent).toBe('The server ran into a problem. Please try again.')
    expect(alert.textContent).not.toMatch(/Traceback/)
    expect(screen.getByText('Rate Limiting Notes')).toBeTruthy()
  })
})

describe('the Documents view in the app', () => {
  it('opens from the sidebar and the chat is still there when coming back', async () => {
    render(<App />)
    expect(screen.queryByRole('region', { name: 'Documents' })).toBeNull()
    fireEvent.change(screen.getByLabelText('Your question'), { target: { value: 'half-typed question' } })

    fireEvent.click(screen.getByRole('button', { name: 'Documents' }))
    expect(await screen.findByRole('region', { name: 'Documents' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Documents' }).getAttribute('aria-current')).toBe('page')
    expect(await screen.findByRole('table')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'New chat' }))
    expect(screen.queryByRole('region', { name: 'Documents' })).toBeNull()
    expect((screen.getByLabelText('Your question') as HTMLTextAreaElement).value).toBe('half-typed question')
  })
})

describe('drag and drop', () => {
  const dropOn = (zone: HTMLElement, files: File[]) => fireEvent.drop(zone, { dataTransfer: { files } })

  it('shows a drop area that highlights while a file is dragged over it', async () => {
    render(<DocumentsView />)
    await screen.findByRole('table')
    const zone = screen.getByTestId('dropzone')
    expect(zone.textContent).toMatch(/Drag a PDF or Markdown file here/)
    fireEvent.dragEnter(zone)
    expect(zone.classList.contains('dragging')).toBe(true)
    fireEvent.dragLeave(zone)
    expect(zone.classList.contains('dragging')).toBe(false)
  })

  it('uploads a dropped file with the same progress and result as the button', async () => {
    let finish!: (r: Response) => void
    onUpload = () => new Promise<Response>((resolve) => (finish = resolve))
    render(<DocumentsView />)
    await screen.findByRole('table')

    dropOn(screen.getByTestId('dropzone'), [new File(['# Notes'], 'dropped.md', { type: 'text/markdown' })])
    expect((await screen.findByRole('status')).textContent).toMatch(/Uploading and indexing dropped\.md/)
    expect(calls.filter((c) => c.method === 'POST' && c.url === '/api/ingest')).toHaveLength(1)

    await act(async () => finish(json({ status: 'new', chunks: 2, document_id: 9, filename: 'dropped.md', error: null })))
    expect(await screen.findByText('Added dropped.md (2 passages).')).toBeTruthy()
    await waitFor(() => expect(listCalls()).toBe(2)) // the list refreshed
    expect(screen.getByTestId('dropzone').classList.contains('dragging')).toBe(false)
  })

  it('adds only the first file when several are dropped, and says so', async () => {
    render(<DocumentsView />)
    await screen.findByRole('table')
    dropOn(screen.getByTestId('dropzone'), [new File(['a'], 'one.md'), new File(['b'], 'two.md')])
    expect((await screen.findByText(/Added notes\.md/)).textContent).toMatch(/Only the first file was added/)
    expect(calls.filter((c) => c.method === 'POST' && c.url === '/api/ingest')).toHaveLength(1)
  })

  it('shows the error when a dropped file is refused', async () => {
    onUpload = () => json({ detail: 'Only .pdf and .md files can be uploaded.' }, 415)
    render(<DocumentsView />)
    await screen.findByRole('table')
    dropOn(screen.getByTestId('dropzone'), [new File(['x'], 'picture.png')])
    expect((await screen.findByRole('alert')).textContent).toBe('Only .pdf and .md files can be uploaded.')
  })

  it('ignores a drop that holds no files', async () => {
    render(<DocumentsView />)
    await screen.findByRole('table')
    dropOn(screen.getByTestId('dropzone'), [])
    expect(calls.some((c) => c.method === 'POST')).toBe(false)
  })
})
