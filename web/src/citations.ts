// Turning citation markers such as [2] in an answer into nodes the Markdown renderer can show as chips.

const CITATION_GROUP = /\[\s*(\d{1,3}(?:\s*,\s*\d{1,3})*)\s*\]/g

interface MdNode {
  type: string
  value?: string
  children?: MdNode[]
  data?: { hName: string; hProperties: Record<string, string> }
}

/** Replace "text [1][2, 3] text" in Markdown text nodes by citation nodes, for the numbers 1..max. Groups that
 *  contain a number outside that range, and everything inside code or links, are left as they were. */
export function citationsToNodes(text: string, max: number): MdNode[] | null {
  const out: MdNode[] = []
  let last = 0
  let found = false
  for (const match of text.matchAll(CITATION_GROUP)) {
    const numbers = match[1].split(',').map((n) => Number(n.trim()))
    if (numbers.some((n) => n < 1 || n > max)) continue
    found = true
    if (match.index > last) out.push({ type: 'text', value: text.slice(last, match.index) })
    for (const n of numbers) out.push({ type: 'citation', data: { hName: 'citation-chip', hProperties: { n: String(n) } } })
    last = match.index + match[0].length
  }
  if (!found) return null
  if (last < text.length) out.push({ type: 'text', value: text.slice(last) })
  return out
}

function walk(node: MdNode, max: number): void {
  if (!node.children || node.type === 'link' || node.type === 'linkReference') return
  const next: MdNode[] = []
  for (const child of node.children) {
    const replaced = child.type === 'text' && child.value ? citationsToNodes(child.value, max) : null
    if (replaced) next.push(...replaced)
    else {
      walk(child, max)
      next.push(child)
    }
  }
  node.children = next
}

/** A remark plugin: turns valid [n] markers into citation nodes. Used only when the sources are known. */
export function remarkCitations({ max }: { max: number }) {
  return (tree: MdNode) => walk(tree, max)
}
