import assert from 'node:assert/strict'
import { describe, it } from 'node:test'
import { buildBody, emptyValues, numericFields, validateForm } from '../src/lib/filmForm.ts'

// shape of GET /api/predict/schema (only the parts the form reads)
const schema = { numeric: ['title_year', 'duration', 'budget', 'aspect_ratio'], genres: ['Drama', 'Romance'], categorical: { country: ['USA'], content_rating: ['PG-13'], language: ['English'], color: ['Color'] } } as never

describe('researcher form', () => {
  it('asks only for the schema’s pre-release numeric fields', () => {
    assert.deepEqual(numericFields(schema).map((f) => f.key), ['title_year', 'duration', 'budget', 'aspect_ratio'])
  })

  it('lists fields in the Predictor page order whatever order the schema uses, keeping unknown fields after them', () => {
    const shuffled = { ...(schema as object), numeric: ['aspect_ratio', 'new_field', 'duration', 'budget', 'title_year'] } as never
    assert.deepEqual(numericFields(shuffled).map((f) => f.key), ['title_year', 'duration', 'budget', 'aspect_ratio', 'new_field'])
  })

  it('requires year, runtime, genres, country and rating, and nothing else', () => {
    assert.deepEqual(Object.keys(validateForm(schema, emptyValues(), [])).sort(), ['content_rating', 'country', 'duration', 'genres', 'title_year'])
    assert.deepEqual(validateForm(schema, { ...emptyValues(), title_year: '2010', duration: '120', country: 'USA', content_rating: 'PG-13' }, ['Drama']), {})
  })

  it('gives readable hints for bad numbers (the server stays the authority)', () => {
    const v = { ...emptyValues(), title_year: '20.5', duration: '0', budget: 'abc', country: 'USA', content_rating: 'PG-13' }
    const e = validateForm(schema, v, ['Drama'])
    assert.match(e.title_year, /whole number/); assert.match(e.duration, /between 1 and 600/); assert.match(e.budget, /number/)
  })

  it('builds the exact POST /api/predict body: numbers as numbers, blanks omitted, nothing post-release', () => {
    const body = buildBody(schema, { ...emptyValues(), title_year: '2010', duration: '120.5', budget: '', aspect_ratio: '2.35', country: 'USA', content_rating: 'PG-13', language: '' }, ['Drama', 'Romance'])
    assert.deepEqual(body, { genres: ['Drama', 'Romance'], title_year: 2010, duration: 120.5, aspect_ratio: 2.35, country: 'USA', content_rating: 'PG-13' })
    for (const banned of ['gross', 'imdb_score', 'num_voted_users', 'num_critic_for_reviews', 'movie_facebook_likes']) assert.ok(!(banned in body))
  })

  it('adds the research-only title when given, trimmed and single-spaced, and omits it when blank', () => {
    const base = { ...emptyValues(), title_year: '2010', duration: '120', country: 'USA', content_rating: 'PG-13' }
    assert.equal(buildBody(schema, { ...base, research_title: '  Tom   &  Jerry ' }, ['Drama']).research_title, 'Tom & Jerry')
    assert.ok(!('research_title' in buildBody(schema, { ...base, research_title: '   ' }, ['Drama'])))
    assert.ok(!('research_title' in buildBody(schema, base, ['Drama'])))
  })

  it('rejects hostile or oversized research titles before sending', () => {
    const base = { ...emptyValues(), title_year: '2010', duration: '120', country: 'USA', content_rating: 'PG-13' }
    for (const bad of ['<script>alert(1)</script>', 'x'.repeat(121), 'A', 'a|b', '{{x}}']) assert.ok(validateForm(schema, { ...base, research_title: bad }, ['Drama']).research_title, bad)
    for (const good of ['Inception', 'Tom & Jerry', "Schindler's List", 'Amélie', '2001: A Space Odyssey']) assert.equal(validateForm(schema, { ...base, research_title: good }, ['Drama']).research_title, undefined, good)
  })

  it('ignores stray keys in the form state', () => {
    const body = buildBody(schema, { ...emptyValues(), gross: '1000000', title_year: '2010', duration: '100', country: 'USA', content_rating: 'R' } as never, ['Drama'])
    assert.ok(!('gross' in body))
  })
})
