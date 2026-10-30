import type { ReactNode } from 'react'
import { parse, type Inline } from '../../lib/miniMarkdown'

function Inl({ parts }: { parts: Inline[] }): ReactNode {
  return parts.map((p, i) =>
    p.t === 'bold' ? <strong key={i} className="font-semibold text-fg">{p.v}</strong>
      : p.t === 'code' ? <code key={i} className="rounded-md bg-black/40 px-1.5 py-0.5 font-mono text-[0.85em] text-cyan">{p.v}</code>
        : <span key={i}>{p.v}</span>)
}

/** Renders the safe markdown subset as React elements. Model text only ever becomes text nodes: no HTML, no links, no images. */
export function Markdown({ text }: { text: string }) {
  const blocks = parse(text)
  return (
    <div className="space-y-2.5 break-words [overflow-wrap:anywhere]" data-testid="markdown">
      {blocks.map((b, i) => {
        if (b.t === 'p') return <p key={i}><Inl parts={b.inline} /></p>
        if (b.t === 'h') return <p key={i} className="font-display font-bold text-fg"><Inl parts={b.inline} /></p>
        if (b.t === 'ul') return <ul key={i} className="list-disc space-y-1 pl-5 marker:text-fg3">{b.items.map((it, j) => <li key={j}><Inl parts={it} /></li>)}</ul>
        if (b.t === 'ol') return <ol key={i} start={b.start} className="list-decimal space-y-1 pl-5 marker:text-fg3">{b.items.map((it, j) => <li key={j}><Inl parts={it} /></li>)}</ol>
        return <pre key={i} tabIndex={0} aria-label="Code" className="overflow-x-auto rounded-xl border border-line bg-black/40 p-3 font-mono text-[0.8rem] leading-relaxed text-fg2"><code>{b.v}</code></pre>
      })}
    </div>
  )
}
