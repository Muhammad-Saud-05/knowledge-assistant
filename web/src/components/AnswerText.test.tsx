import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import AnswerText from './AnswerText'

afterEach(cleanup)

describe('AnswerText', () => {
  it('shows a <script> tag as plain text and creates no script element', () => {
    const { container } = render(<AnswerText text="<script>alert(1)</script>" />)
    expect(container.querySelector('script')).toBeNull()
    expect(container.textContent).toContain('<script>alert(1)</script>')
  })

  it('shows an <img onerror> tag as plain text and creates no img element', () => {
    const { container } = render(<AnswerText text={'<img src=x onerror="alert(1)">'} />)
    expect(container.querySelector('img')).toBeNull()
    expect(container.querySelector('[onerror]')).toBeNull()
    expect(container.textContent).toContain('<img src=x onerror="alert(1)">')
  })

  it('does not create script or img elements when the HTML is mixed into a sentence', () => {
    const { container } = render(
      <AnswerText text={'Text <script>alert(1)</script> and <img src=x onerror=alert(2)> more <b onclick="x()">bold</b>'} />,
    )
    expect(container.querySelector('script, img, b, [onclick], [onerror]')).toBeNull()
    expect(container.textContent).toContain('Text')
    expect(container.textContent).toContain('more')
  })

  it('does not load images written in Markdown', () => {
    const { container } = render(<AnswerText text="![tracker](http://evil.example/pixel.png)" />)
    expect(container.querySelector('img')).toBeNull()
  })

  it('neutralises javascript: links', () => {
    const { container } = render(<AnswerText text="[click me](javascript:alert(1))" />)
    const link = container.querySelector('a')
    expect(link?.getAttribute('href') ?? '').not.toMatch(/^javascript:/i)
  })

  it('opens ordinary links in a new tab without access to this page', () => {
    const { container } = render(<AnswerText text="[docs](https://example.com/page)" />)
    const link = container.querySelector('a')
    expect(link?.getAttribute('href')).toBe('https://example.com/page')
    expect(link?.getAttribute('target')).toBe('_blank')
    expect(link?.getAttribute('rel')).toContain('noopener')
  })

  it('still renders ordinary Markdown', () => {
    const { container } = render(<AnswerText text={'A **bold** claim [1].\n\n- one\n- two\n\n`code`'} />)
    expect(container.querySelector('strong')?.textContent).toBe('bold')
    expect(container.querySelectorAll('li')).toHaveLength(2)
    expect(container.querySelector('code')?.textContent).toBe('code')
    expect(container.textContent).toContain('[1]')
  })
})
