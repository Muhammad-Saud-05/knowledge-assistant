// Pipe tables ("| a | b |" with a "| --- | --- |" line under the header) in answers.
//
// react-markdown does not read tables by itself (that needs the remark-gfm plugin, which is not a dependency
// of this project), so this small plugin turns a paragraph that is a pipe table into a table node. Cells are
// plain text. Anything that is not clearly a table is left as the paragraph it was.

interface MdNode {
  type: string
  value?: string
  children?: MdNode[]
  align?: (string | null)[]
}

const SEPARATOR_CELL = /^:?-{3,}:?$/

function cellsOf(line: string): string[] {
  let text = line.trim()
  if (text.startsWith('|')) text = text.slice(1)
  if (text.endsWith('|')) text = text.slice(0, -1)
  return text.split('|').map((cell) => cell.trim())
}

function row(cells: string[], columns: number): MdNode {
  const padded = Array.from({ length: columns }, (_, i) => cells[i] ?? '')
  return {
    type: 'tableRow',
    children: padded.map((value) => ({ type: 'tableCell', children: value ? [{ type: 'text', value }] : [] })),
  }
}

/** A table node for the text of a paragraph, or null if the text is not a pipe table. */
export function tableFrom(text: string): MdNode | null {
  const lines = text.split('\n').filter((line) => line.trim() !== '')
  if (lines.length < 2 || !lines[0].includes('|')) return null
  const header = cellsOf(lines[0])
  const separator = cellsOf(lines[1])
  if (separator.length !== header.length || !separator.every((cell) => SEPARATOR_CELL.test(cell))) return null
  const body = lines.slice(2)
  if (body.some((line) => !line.includes('|'))) return null
  const align = separator.map((cell) => (cell.startsWith(':') ? (cell.endsWith(':') ? 'center' : 'left') : cell.endsWith(':') ? 'right' : null))
  return {
    type: 'table',
    align,
    children: [row(header, header.length), ...body.map((line) => row(cellsOf(line), header.length))],
  }
}

function walk(node: MdNode): void {
  if (!node.children) return
  node.children = node.children.map((child) => {
    if (child.type === 'paragraph' && child.children?.length === 1 && child.children[0].type === 'text') {
      return tableFrom(child.children[0].value ?? '') ?? child
    }
    walk(child)
    return child
  })
}

/** A remark plugin that reads pipe tables. */
export function remarkTables() {
  return (tree: MdNode) => walk(tree)
}
