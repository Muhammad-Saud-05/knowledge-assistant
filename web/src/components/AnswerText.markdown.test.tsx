import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { citationsToNodes } from '../citations'
import { tableFrom } from '../markdownTables'
import AnswerText from './AnswerText'

afterEach(cleanup)

describe('pipe tables', () => {
  const TABLE = '| Algorithm | Bursts |\n| --- | :---: |\n| Token bucket | Yes |\n| Fixed window | At the edge |'

  it('renders a table with a header row and body rows', () => {
    const { container } = render(<AnswerText text={`Before.\n\n${TABLE}\n\nAfter.`} />)
    expect(container.querySelectorAll('table')).toHaveLength(1)
    expect([...container.querySelectorAll('th')].map((c) => c.textContent)).toEqual(['Algorithm', 'Bursts'])
    expect([...container.querySelectorAll('tbody tr')].map((r) => [...r.querySelectorAll('td')].map((c) => c.textContent))).toEqual([
      ['Token bucket', 'Yes'],
      ['Fixed window', 'At the edge'],
    ])
    expect(container.querySelector('.table-wrap > table')).toBeTruthy() // scrolls sideways inside its own box on a phone
    expect(container.textContent).toContain('Before.')
    expect(container.textContent).toContain('After.')
  })

  it('shows a table that is still arriving (the header and separator only) without breaking', () => {
    const { container } = render(<AnswerText text={'| A | B |\n| --- | --- |\n| 1'} />)
    expect(container.querySelectorAll('table')).toHaveLength(1)
    expect(container.querySelectorAll('tbody tr')).toHaveLength(1)
    expect(container.querySelectorAll('tbody td')).toHaveLength(2) // a short row is padded
  })

  it('leaves ordinary text with a pipe alone', () => {
    const { container } = render(<AnswerText text="Use a | b to mean a or b." />)
    expect(container.querySelector('table')).toBeNull()
    expect(container.textContent).toContain('a | b')
  })

  it('recognises only well-formed tables', () => {
    expect(tableFrom('| a | b |\n| --- | --- |\n| 1 | 2 |')).toMatchObject({ type: 'table' })
    expect(tableFrom('| a | b |\n| -- | -- |\n| 1 | 2 |')).toBeNull() // separator too short
    expect(tableFrom('| a | b |\n| --- |\n| 1 | 2 |')).toBeNull() // different column count
    expect(tableFrom('| a | b |\n| --- | --- |\nnot a row')).toBeNull()
    expect(tableFrom('just text')).toBeNull()
  })

  it('does not run HTML placed in a cell', () => {
    const { container } = render(<AnswerText text={'| a |\n| --- |\n| <img src=x onerror=alert(1)> |'} />)
    expect(container.querySelector('img, [onerror]')).toBeNull()
    expect(container.textContent).toContain('<img src=x onerror=alert(1)>')
  })
})

describe('citation chips', () => {
  it('turns [n] into chips for the numbers 1 to the number of sources only', () => {
    const onCite = vi.fn()
    render(<AnswerText text="Fact [1][2, 3]. Odd [4] and [0]." sourceCount={3} onCite={onCite} />)
    expect(screen.getAllByRole('button').map((b) => b.getAttribute('aria-label'))).toEqual(['Source 1', 'Source 2', 'Source 3'])
    expect(document.body.textContent).toContain('[4]')
    expect(document.body.textContent).toContain('[0]')
    fireEvent.click(screen.getByRole('button', { name: 'Source 2' }))
    expect(onCite).toHaveBeenCalledWith(2, screen.getByRole('button', { name: 'Source 2' }))
  })

  it('makes no chips when the sources are not known', () => {
    render(<AnswerText text="Fact [1]." />)
    expect(screen.queryAllByRole('button')).toHaveLength(0)
    expect(document.body.textContent).toContain('[1]')
  })

  it('keeps the same chip elements while the text grows (so focus is not lost)', () => {
    const onCite = vi.fn()
    const { rerender } = render(<AnswerText text="Fact [1]." sourceCount={2} onCite={onCite} />)
    const chip = screen.getByRole('button', { name: 'Source 1' })
    rerender(<AnswerText text="Fact [1]. More words." sourceCount={2} onCite={() => onCite()} />)
    expect(screen.getByRole('button', { name: 'Source 1' })).toBe(chip)
  })

  it('finds citation groups in plain text', () => {
    expect(citationsToNodes('no markers', 3)).toBeNull()
    expect(citationsToNodes('a [1] b', 3)?.map((n) => n.type)).toEqual(['text', 'citation', 'text'])
    expect(citationsToNodes('a [5] b', 3)).toBeNull() // out of range stays text
    expect(citationsToNodes('[1, 2]', 3)?.filter((n) => n.type === 'citation')).toHaveLength(2)
  })
})

describe('the caret while streaming', () => {
  it('marks the answer as streaming so the stylesheet can show a caret after the last word', () => {
    const { container, rerender } = render(<AnswerText text="Some words" streaming />)
    expect(container.querySelector('.answer-text.streaming')).toBeTruthy()
    rerender(<AnswerText text="Some words" />)
    expect(container.querySelector('.streaming')).toBeNull()
  })
})
