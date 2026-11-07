// A small parser for Server-Sent Events (text/event-stream), with no browser dependencies.
//
// The browser's built-in EventSource can only make GET requests, but we need to POST the
// question, so we read the response ourselves and parse it here. The format is lines of
// "field: value", with a blank line ending each event:
//
//   event: token
//   data: {"text": "Hello"}
//
// Network chunks can cut the stream anywhere (in the middle of a word, or between the
// "\r" and "\n" of a line ending), so the parser keeps what it has not finished yet.

export interface SseEvent {
  /** The event name; "message" when the server did not give one. */
  event: string
  /** The data lines joined with "\n". */
  data: string
}

export class SseParser {
  private buffer = ''
  private eventName = ''
  private dataLines: string[] = []
  private started = false

  /** Add the next piece of text; returns every event that is now complete. */
  feed(chunk: string): SseEvent[] {
    if (!this.started && chunk.length > 0) {
      this.started = true
      if (chunk.charCodeAt(0) === 0xfeff) chunk = chunk.slice(1) // a byte-order mark at the start
    }
    this.buffer += chunk

    const events: SseEvent[] = []
    let start = 0
    for (;;) {
      const end = this.findLineEnd(start)
      if (end === -1) break
      const line = this.buffer.slice(start, end)
      const isCrLf = this.buffer[end] === '\r' && this.buffer[end + 1] === '\n'
      start = end + (isCrLf ? 2 : 1)
      const event = this.processLine(line)
      if (event) events.push(event)
    }
    this.buffer = this.buffer.slice(start)
    return events
  }

  /**
   * Call when the stream has ended. A "\r" that was held back, in case "\n" followed,
   * is now known to be a complete line ending. Text that never got its blank line is
   * an unfinished event and is dropped, as the format requires.
   */
  finish(): SseEvent[] {
    return this.buffer.endsWith('\r') ? this.feed('\n') : []
  }

  /** Index of the next line ending at or after `from`, or -1 if the line is not finished yet. */
  private findLineEnd(from: number): number {
    for (let i = from; i < this.buffer.length; i++) {
      const char = this.buffer[i]
      if (char === '\n') return i
      if (char === '\r') {
        // A "\r" at the very end might be the first half of "\r\n": wait for the next piece.
        return i === this.buffer.length - 1 ? -1 : i
      }
    }
    return -1
  }

  private processLine(line: string): SseEvent | null {
    if (line === '') return this.dispatch()
    if (line.startsWith(':')) return null // a comment (often used as a keep-alive)

    const colon = line.indexOf(':')
    const field = colon === -1 ? line : line.slice(0, colon)
    let value = colon === -1 ? '' : line.slice(colon + 1)
    if (value.startsWith(' ')) value = value.slice(1) // only one leading space is dropped

    if (field === 'event') this.eventName = value
    else if (field === 'data') this.dataLines.push(value)
    // "id" and "retry" are valid fields that this app does not use.
    return null
  }

  private dispatch(): SseEvent | null {
    const hadData = this.dataLines.length > 0
    const event = { event: this.eventName || 'message', data: this.dataLines.join('\n') }
    this.eventName = ''
    this.dataLines = []
    return hadData ? event : null // an event with no data lines is not delivered
  }
}
