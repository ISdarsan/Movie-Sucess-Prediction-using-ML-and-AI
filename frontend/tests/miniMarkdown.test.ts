import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { inline, parse } from '../src/lib/miniMarkdown.ts'

describe('mini markdown (safe subset)', () => {
  it('paragraphs, with single newlines joined and blank lines separating', () => {
    assert.deepEqual(parse('one\ntwo\n\nthree'), [{ t: 'p', inline: [{ t: 'text', v: 'one two' }] }, { t: 'p', inline: [{ t: 'text', v: 'three' }] }])
  })
  it('bullet and numbered lists, keeping the start number', () => {
    const b = parse('- a\n- **b**\n\n3. x\n4. y')
    assert.deepEqual(b[0], { t: 'ul', items: [[{ t: 'text', v: 'a' }], [{ t: 'bold', v: 'b' }]] })
    assert.deepEqual(b[1], { t: 'ol', start: 3, items: [[{ t: 'text', v: 'x' }], [{ t: 'text', v: 'y' }]] })
  })
  it('bold, inline code, and headings as bold lines', () => {
    assert.deepEqual(inline('use **trees** and `max_depth`'), [{ t: 'text', v: 'use ' }, { t: 'bold', v: 'trees' }, { t: 'text', v: ' and ' }, { t: 'code', v: 'max_depth' }])
    assert.deepEqual(parse('## Title')[0], { t: 'h', inline: [{ t: 'text', v: 'Title' }] })
  })
  it('fenced code is kept verbatim, including markdown and html inside it', () => {
    assert.deepEqual(parse('```python\nx = **y** <b>\n```')[0], { t: 'code', v: 'x = **y** <b>' })
  })
  it('an unclosed fence does not swallow nothing or crash', () => {
    assert.deepEqual(parse('```\nabc')[0], { t: 'code', v: 'abc' })
  })
  it('HTML, links and images stay plain text: there is no way to produce them', () => {
    const tree = JSON.stringify(parse('<script>alert(1)</script> [click](javascript:alert(1)) ![x](http://evil/x.png) <img src=x onerror=alert(1)>'))
    assert.ok(tree.includes('<script>') && tree.includes('javascript:'))                     // present only as TEXT nodes
    assert.ok(!/"t":"(a|img|html|link)"/.test(tree))
  })
  it('unmatched markers are left alone, and empty input is empty', () => {
    assert.deepEqual(inline('2 * 3 ** 4 and `open'), [{ t: 'text', v: '2 * 3 ** 4 and `open' }]); assert.deepEqual(parse(''), []); assert.deepEqual(parse('   \n\n  '), [])
  })
  it('handles Windows line endings and very long input without hanging', () => {
    assert.equal(parse('a\r\n\r\nb').length, 2)
    const t = Date.now(); parse(('- item **bold** `c`\n').repeat(5000) + '**'.repeat(5000)); assert.ok(Date.now() - t < 2000)
  })
})
