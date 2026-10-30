import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { normalizeRun } from '../src/lib/agentRun.ts'
import { createAnalysisRegistry, fromRun, MAX_ANALYSES, parseSummary } from '../src/lib/analyses.ts'
import { completed, failedRun, research, running, RUN_ID, withResearch } from './fixtures.ts'

const ref = (i: number) => ({ runId: i.toString(16).padStart(32, '0'), title: `Film ${i}`, predictedClass: 'Hit', year: 2000 + i, genres: 'Drama' })

describe('fromRun', () => {
  it('describes a completed run with a prediction, using the research title when there is one', () => {
    const run = normalizeRun(withResearch(research({ query: { title: 'Inception', genres: ['Action'] } })))!
    assert.deepEqual(fromRun(run), { runId: RUN_ID, title: 'Inception', predictedClass: 'Hit', year: 2010, genres: 'Drama / Romance' })
  })
  it('has no title when none was given, and never invents one', () => {
    assert.equal(fromRun(normalizeRun(completed())!)!.title, null)
    assert.equal(fromRun(normalizeRun(withResearch(research({ query: { title: null, genres: [] } })))!)!.title, null)
    assert.equal(fromRun(normalizeRun(withResearch(research({ query: { title: '   ', genres: [] } })))!)!.title, null)
  })
  it('refuses runs that are unfinished, failed, without a prediction, or without a valid id', () => {
    for (const raw of [running(), failedRun(), completed({ prediction: undefined }), completed({ prediction: { label: 'Blockbuster' } }), completed({ run_id: 'not-an-id' })]) assert.equal(fromRun(normalizeRun(raw)), null)
    assert.equal(fromRun(null), null); assert.equal(fromRun(undefined), null)
  })
  it('carries no scores, evidence or research text', () => {
    const blob = JSON.stringify(fromRun(normalizeRun(completed())!))
    for (const k of ['probabilit', 'excerpt', 'cohort', 'comparable', 'risks', '0.6375']) assert.ok(!blob.includes(k), k)
  })
})

describe('parseSummary (the server summary)', () => {
  const ok = { run_id: 'c'.repeat(32), title: 'Inception', predicted_class: 'Hit', year: 2010, genres: 'Action / Sci-Fi', country: 'USA', class_scores: { Flop: 0, Average: 0, Hit: 1 } }
  it('reads the display fields only', () => assert.deepEqual(parseSummary(ok), { runId: 'c'.repeat(32), title: 'Inception', predictedClass: 'Hit', year: 2010, genres: 'Action / Sci-Fi' }))
  it('rejects anything that is not a recognisable summary', () => {
    for (const bad of [null, [], 'x', {}, { ...ok, run_id: 'zz' }, { ...ok, predicted_class: 'Blockbuster' }, { ...ok, run_id: 5 }]) assert.equal(parseSummary(bad), null)
  })
  it('tolerates missing optional fields', () => assert.deepEqual(parseSummary({ run_id: RUN_ID, predicted_class: 'Flop' }), { runId: RUN_ID, title: null, predictedClass: 'Flop', year: null, genres: null }))
})

describe('analysis registry', () => {
  it('lists newest first, once per run, capped, and ignores invalid entries', () => {
    let t = 0; const reg = createAnalysisRegistry(() => ++t)
    reg.add(null); reg.add({ ...ref(1), runId: 'bad' })
    for (let i = 1; i <= MAX_ANALYSES + 3; i++) reg.add(ref(i))
    reg.add(ref(5))                                                          // seen again: moves to the front, not duplicated
    const l = reg.getState()
    assert.equal(l.length, MAX_ANALYSES); assert.equal(l[0].runId, ref(5).runId); assert.equal(l.filter((a) => a.runId === ref(5).runId).length, 1); assert.ok(!l.some((a) => a.runId === ref(1).runId), 'oldest dropped'); assert.ok(l[0].at > l[1].at)
  })
  it('notifies subscribers and looks up by id', () => {
    const reg = createAnalysisRegistry(); let n = 0; const off = reg.subscribe(() => n++)
    reg.add(ref(2)); assert.equal(n, 1); assert.equal(reg.get(ref(2).runId)!.title, 'Film 2'); assert.equal(reg.get('d'.repeat(32)), null); off(); reg.add(ref(3)); assert.equal(n, 1)
  })
})
