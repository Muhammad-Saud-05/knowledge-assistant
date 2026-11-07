import { describe, expect, it } from 'vitest'
import { SseParser } from './sse'

function parseAll(...chunks: string[]) {
  const parser = new SseParser()
  return chunks.flatMap((chunk) => parser.feed(chunk))
}

const STREAM =
  'event: sources\ndata: [{"number":1}]\n\n' +
  'event: token\ndata: {"text":"Hel"}\n\n' +
  'event: token\ndata: {"text":"lo"}\n\n' +
  'event: done\ndata: {"text":"Hello"}\n\n'

const EXPECTED = [
  { event: 'sources', data: '[{"number":1}]' },
  { event: 'token', data: '{"text":"Hel"}' },
  { event: 'token', data: '{"text":"lo"}' },
  { event: 'done', data: '{"text":"Hello"}' },
]

describe('SseParser', () => {
  it('parses one event', () => {
    expect(parseAll('event: token\ndata: {"text":"hi"}\n\n')).toEqual([{ event: 'token', data: '{"text":"hi"}' }])
  })

  it('parses several events that arrive in one chunk', () => {
    expect(parseAll(STREAM)).toEqual(EXPECTED)
  })

  it('does not emit an event until its blank line has arrived', () => {
    const parser = new SseParser()
    expect(parser.feed('event: token\ndata: {"text":"hi"}\n')).toEqual([])
    expect(parser.feed('\n')).toEqual([{ event: 'token', data: '{"text":"hi"}' }])
  })

  it('handles an event split across chunks at every possible position', () => {
    for (let cut = 0; cut <= STREAM.length; cut++) {
      expect(parseAll(STREAM.slice(0, cut), STREAM.slice(cut)), `split at ${cut}`).toEqual(EXPECTED)
    }
  })

  it('handles a stream delivered one character at a time', () => {
    expect(parseAll(...STREAM.split(''))).toEqual(EXPECTED)
  })

  it('handles CRLF line endings, including a CRLF split between two chunks', () => {
    const crlf = STREAM.replaceAll('\n', '\r\n')
    expect(parseAll(crlf)).toEqual(EXPECTED)
    for (let cut = 0; cut <= crlf.length; cut++) {
      expect(parseAll(crlf.slice(0, cut), crlf.slice(cut)), `CRLF split at ${cut}`).toEqual(EXPECTED)
    }
    expect(parseAll(...crlf.split(''))).toEqual(EXPECTED)
  })

  it('handles lone CR line endings', () => {
    // The final "\r" might be half of "\r\n", so the parser waits for the next piece...
    const parser = new SseParser()
    expect(parser.feed('event: token\rdata: {"text":"a"}\r\r')).toEqual([])
    expect(parser.feed('event: token\rdata: {"text":"b"}\r\r')).toEqual([{ event: 'token', data: '{"text":"a"}' }])
    // ...unless the stream ends, which settles it.
    expect(parser.finish()).toEqual([{ event: 'token', data: '{"text":"b"}' }])
  })

  it('finish() drops an event that never got its blank line, and does nothing when nothing is pending', () => {
    const parser = new SseParser()
    expect(parser.feed('event: token\ndata: {"text":"partial"}\n')).toEqual([])
    expect(parser.finish()).toEqual([])
    expect(new SseParser().finish()).toEqual([])
  })

  it('joins multi-line data with newlines', () => {
    expect(parseAll('event: note\ndata: first\ndata: second\ndata:\ndata: fourth\n\n')).toEqual([
      { event: 'note', data: 'first\nsecond\n\nfourth' },
    ])
  })

  it('handles multi-line data split across chunks and with CRLF', () => {
    expect(parseAll('data: one\r\nda', 'ta: two\r', '\n\r\n')).toEqual([{ event: 'message', data: 'one\ntwo' }])
  })

  it('uses "message" when the event has no name', () => {
    expect(parseAll('data: plain\n\n')).toEqual([{ event: 'message', data: 'plain' }])
  })

  it('passes an unknown event name through unchanged', () => {
    expect(parseAll('event: something-new\ndata: {}\n\n')).toEqual([{ event: 'something-new', data: '{}' }])
  })

  it('delivers an error event like any other', () => {
    expect(parseAll('event: error\ndata: {"message":"Start Ollama"}\n\n')).toEqual([
      { event: 'error', data: '{"message":"Start Ollama"}' },
    ])
  })

  it('removes only one leading space from a value', () => {
    expect(parseAll('data:no space\n\ndata:  two spaces\n\n')).toEqual([
      { event: 'message', data: 'no space' },
      { event: 'message', data: ' two spaces' },
    ])
  })

  it('keeps colons inside the value', () => {
    expect(parseAll('data: {"url":"http://x:1"}\n\n')).toEqual([{ event: 'message', data: '{"url":"http://x:1"}' }])
  })

  it('ignores comments, ids, retry hints and fields it does not know', () => {
    expect(parseAll(': keep-alive\n\nid: 7\nretry: 3000\nmystery: yes\nevent: token\ndata: ok\n\n')).toEqual([
      { event: 'token', data: 'ok' },
    ])
  })

  it('does not deliver an event that has no data', () => {
    expect(parseAll('event: ping\n\n\n\n')).toEqual([])
  })

  it('does not let the event name of one event leak into the next', () => {
    expect(parseAll('event: token\ndata: a\n\ndata: b\n\n')).toEqual([
      { event: 'token', data: 'a' },
      { event: 'message', data: 'b' },
    ])
  })

  it('ignores a byte-order mark at the start', () => {
    expect(parseAll(String.fromCharCode(0xfeff) + 'data: x\n\n')).toEqual([{ event: 'message', data: 'x' }])
  })

  it('keeps unfinished text for the next chunk and never delivers a half event', () => {
    const parser = new SseParser()
    expect(parser.feed('event: tok')).toEqual([])
    expect(parser.feed('en\ndata: {"te')).toEqual([])
    expect(parser.feed('xt":"x"}\n\n')).toEqual([{ event: 'token', data: '{"text":"x"}' }])
  })

  it('handles empty chunks', () => {
    expect(parseAll('', 'data: a\n\n', '', '')).toEqual([{ event: 'message', data: 'a' }])
  })
})
