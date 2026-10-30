/* A deliberately tiny, SAFE markdown subset for assistant replies: paragraphs, bullet and numbered lists, fenced code, #-headings (shown as
   bold lines), **bold** and `inline code`. It produces a plain data tree; the UI renders it as React text nodes, so nothing the model writes
   can become HTML, a link, an image or a script. Anything it doesn't recognise stays ordinary text. */

export type Inline = { t: 'text' | 'bold' | 'code'; v: string }
export type Block =
  | { t: 'p'; inline: Inline[] }
  | { t: 'h'; inline: Inline[] }
  | { t: 'ul'; items: Inline[][] }
  | { t: 'ol'; items: Inline[][]; start: number }
  | { t: 'code'; v: string }

export function inline(src: string): Inline[] {
  const out: Inline[] = []
  const re = /(\*\*[^*\n]+?\*\*|`[^`\n]+?`)/g
  let last = 0
  for (const m of src.matchAll(re)) {
    if (m.index > last) out.push({ t: 'text', v: src.slice(last, m.index) })
    const tok = m[0]
    out.push(tok.startsWith('**') ? { t: 'bold', v: tok.slice(2, -2) } : { t: 'code', v: tok.slice(1, -1) })
    last = m.index + tok.length
  }
  if (last < src.length) out.push({ t: 'text', v: src.slice(last) })
  return out.length ? out : [{ t: 'text', v: src }]
}

export function parse(md: string): Block[] {
  const lines = md.replace(/\r\n?/g, '\n').split('\n')
  const blocks: Block[] = []
  let para: string[] = []
  const flush = () => { if (para.length) { blocks.push({ t: 'p', inline: inline(para.join(' ').trim()) }); para = [] } }
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    if (/^\s*```/.test(line)) {                               // fenced code: everything up to the closing fence, verbatim
      flush()
      const body: string[] = []
      i++
      while (i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i++])
      blocks.push({ t: 'code', v: body.join('\n') })
      continue
    }
    if (!line.trim()) { flush(); continue }
    const h = /^\s{0,3}#{1,6}\s+(.+)$/.exec(line)
    if (h) { flush(); blocks.push({ t: 'h', inline: inline(h[1].trim()) }); continue }
    const ul = /^\s{0,3}[-*•]\s+(.+)$/.exec(line)
    const ol = /^\s{0,3}(\d{1,3})[.)]\s+(.+)$/.exec(line)
    if (ul || ol) {
      flush()
      const ordered = !!ol
      const items: Inline[][] = []
      const start = ol ? Number(ol[1]) : 1
      let j = i
      for (; j < lines.length; j++) {
        const m = ordered ? /^\s{0,3}\d{1,3}[.)]\s+(.+)$/.exec(lines[j]) : /^\s{0,3}[-*•]\s+(.+)$/.exec(lines[j])
        if (!m) break
        items.push(inline(m[1].trim()))
      }
      blocks.push(ordered ? { t: 'ol', items, start } : { t: 'ul', items })
      i = j - 1
      continue
    }
    para.push(line.trim())
  }
  flush()
  return blocks
}
