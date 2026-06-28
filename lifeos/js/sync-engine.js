// sync-engine.js — SYNC v0.1 (KYI middle door). The COMPUTED read, not the echo-prop.
// Deterministic · sovereign (client-side, no server, no model) · light · non-fabricating.
// Spec: airlock-coordination/reports/2026-06-09-sync-engine-v0-spec.md
//
// Pipeline: (1) answers → DECF vector  (2) vector → archetype  (3) birthday → chart
//           (4) compose  (4.5) CMTR cadence boundaries (parallel, sealed, no cross-inference)
//
// Determinism: NO Math.random. Template variety is chosen by a stable hash of the input.

// ── DATA TABLES (tunable — "archetype set = data table", per the review) ───────────────
// v0 MAGS-6 anchors as DECF 4-vectors (1..10). ESTIMATES — to be calibrated, not validated.
export const MAGS6 = {
  Analyst:    { D: 3, E: 2, C: 7, F: 8 },
  Strategist: { D: 7, E: 4, C: 6, F: 7 },
  Executor:   { D: 8, E: 5, C: 3, F: 4 },
  Connector:  { D: 4, E: 8, C: 5, F: 3 },
  Guardian:   { D: 4, E: 3, C: 7, F: 8 },
  Architect:  { D: 6, E: 3, C: 7, F: 7 },
}

// Default embedded signal subset (real vocabulary from decf-signals.json). In production,
// inject the FULL file via syncRead(input, { signals }). Kept small here to stay self-contained.
export const DEFAULT_SIGNALS = {
  high_D: ['must','immediately','ship','now','decisive','bold','act','move','push','own','lead','drive','execute','launch','cut','prioritize','take charge','make it happen'],
  low_D:  ['maybe','we could','i guess','if that works','whatever you think','no rush','eventually','i tend to wait','let it'],
  high_E: ['we','talk','people','together','share','out loud','everyone','connect','chat','room','friends','social','meet'],
  low_E:  ['alone','quiet','myself','inward','private','on my own','reflect','solo','i think to myself'],
  high_C: ['steady','patient','consistent','routine','over time','finish','follow through','calm','step by step','stable','slowly'],
  low_C:  ['fast','quick','bored','switch','jump','restless','new thing','pivot','change it up','impatient','right away'],
  high_F: ['precise','exact','correct','plan','structure','careful','detail','rules','accurate','proper','organized','double-check'],
  low_F:  ['rough','wing it','improvise','loose','vibe','whatever','casual','figure it out','play it by ear','messy'],
}

const DRIVES = ['D','E','C','F']

const GLOSS = {
  D: { hi: 'moving first and deciding fast',      lo: 'letting things settle before you act' },
  E: { hi: 'thinking out loud, with people',      lo: 'processing inward before you share' },
  C: { hi: 'a steady pace and follow-through',    lo: 'a fast tempo and quick pivots' },
  F: { hi: 'precision and structure',             lo: 'improvising and keeping it loose' },
}
const DRIVE_NAME = { D: 'Dominance', E: 'Extraversion', C: 'Patience', F: 'Formality' }
// What a LOW value on each drive might be quietly downplaying (for the novelty line).
const DOWNPLAY = {
  D: 'how much you actually want to call the shot',
  E: 'how much you lean on the room around you',
  C: 'a need for things to hold still longer than you admit',
  F: 'how much the details actually matter to you',
}

// ── helpers ────────────────────────────────────────────────────────────────────────
function norm(s) { return (s || '').toLowerCase() }
function countMatches(text, words) {
  const t = norm(text)
  let n = 0
  for (const w of words) {
    const needle = w.toLowerCase()
    if (needle.includes(' ')) { if (t.includes(needle)) n++ }              // phrase
    else { const re = new RegExp(`\\b${needle.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')}\\b`,'g'); const m = t.match(re); if (m) n += m.length }
  }
  return n
}
function hashStr(s) { let h = 0; for (let i = 0; i < (s||'').length; i++) h = (31*h + s.charCodeAt(i)) | 0; return Math.abs(h) }

// ── Step 1: answers → DECF vector (INVERT decf-signals: estimate unknown persona) ─────
export function scoreDECF(answers, signals = DEFAULT_SIGNALS) {
  const joined = (answers || []).join('  \n  ')
  const out = {}
  for (const d of DRIVES) {
    const hi = countMatches(joined, signals['high_' + d] || [])
    const lo = countMatches(joined, signals['low_' + d] || [])
    const total = hi + lo
    if (total === 0) { out[d] = { value: null, confidence: 0, nMatches: 0 }; continue }  // declare-gap, never default 0.5
    const ratio = hi / total
    const value = Math.round(1 + ratio * 9)                                  // 1..10
    const confidence = Math.min(1, total / 6)                                // thin signal → low confidence
    out[d] = { value, confidence: Math.round(confidence * 100) / 100, nMatches: total }
  }
  return out
}

// ── Step 2: vector → nearest archetype (skip null dims) ───────────────────────────────
export function nearestArchetype(decf, anchors = MAGS6) {
  const dims = DRIVES.filter(d => decf[d] && decf[d].value != null)
  if (dims.length === 0) return { primary: null, secondary: null, confidence: 0 }
  const scored = Object.entries(anchors).map(([name, a]) => {
    let sq = 0
    for (const d of dims) sq += Math.pow((decf[d].value - a[d]) / 9, 2)       // normalized distance
    return { name, dist: Math.sqrt(sq / dims.length) }
  }).sort((x, y) => x.dist - y.dist)
  const density = dims.reduce((s, d) => s + decf[d].confidence, 0) / dims.length
  const sep = scored.length > 1 ? Math.min(1, (scored[1].dist - scored[0].dist) * 3) : 1
  const confidence = Math.round(Math.max(0, (1 - scored[0].dist) * 0.6 + density * 0.25 + sep * 0.15) * 100) / 100
  return { primary: scored[0].name, secondary: scored[1] ? scored[1].name : null, confidence }
}

// ── Step 3: birthday → Sun sign + element (date math; rising only if time given) ──────
const SIGNS = [ // [name, element, untilMonth(1-12), untilDay]
  ['Capricorn','earth',1,19],['Aquarius','air',2,18],['Pisces','water',3,20],['Aries','fire',4,19],
  ['Taurus','earth',5,20],['Gemini','air',6,20],['Cancer','water',7,22],['Leo','fire',8,22],
  ['Virgo','earth',9,22],['Libra','air',10,22],['Scorpio','water',11,21],['Sagittarius','fire',12,21],
  ['Capricorn','earth',12,31],
]
export function sunSign(birthday) {
  if (!birthday) return { sun: null, element: null, gap: 'no birthday' }
  const d = new Date(birthday); if (isNaN(d)) return { sun: null, element: null, gap: 'unreadable date' }
  const mo = d.getUTCMonth() + 1, day = d.getUTCDate()
  for (const [name, element, um, ud] of SIGNS) { if (mo < um || (mo === um && day <= ud)) return { sun: name, element } }
  return { sun: 'Capricorn', element: 'earth' }
}

// ── Step 4 + 4.5: compose, with CMTR boundaries (parallel · sealed · no cross-inference) ─
const T = {
  behavioral: [
    (a,g) => `You come through closest to ${a}. There's a steadiness in how you describe yourself, and a preference for ${g}. A clear signal, even from three lines.`,
    (a,g) => `Your answers lean toward ${a}. The pattern is consistent: ${g}. Not a verdict — the shape that shows up most.`,
    (a,g) => `Closest match: ${a}. The way you phrase things points to ${g}. A light read, but a real one.`,
  ],
  chart: [
    (s,e) => `Alongside that, your birthday puts your Sun in ${s}, an ${e} sign. A parallel read — not the same signal as the behavioral one.`,
    (s,e) => `Separately, your Sun is in ${s}. An ${e} signature. It sits next to the behavioral read, not inside it.`,
    (s,e) => `In parallel, your chart starts with ${s}, an ${e} tone. A second lens, not a blend.`,
  ],
  gap: [
    (x) => `Here's what I can't see yet: ${x}. I'm not filling that in.`,
    (x) => `There's a gap — ${x}. I'm naming it instead of guessing.`,
    (x) => `One thing I can't read from this is ${x}. That stays open until you add it.`,
  ],
  novelty: [
    (x) => `One thing you didn't say, but your pattern hints at: you may downplay ${x}.`,
    (x) => `Something to watch: a trace of ${x} in how you phrase things, even though you didn't name it.`,
    (x) => `A small counter-signal: ${x}. Not dominant, but it's there.`,
  ],
}

export function syncRead(input, opts = {}) {
  const { answers = [], birthday = null, time = null } = input || {}
  const signals = opts.signals || DEFAULT_SIGNALS
  const pick = arr => arr[hashStr(answers.join('|')) % arr.length]           // stable, deterministic

  const decf = scoreDECF(answers, signals)
  const arch = nearestArchetype(decf, opts.anchors || MAGS6)
  const chart = sunSign(birthday)
  if (time) chart.risingNote = 'birth time given — rising computable in v1'
  else chart.rising = null, chart.gap = chart.gap || 'no birth time'

  // gloss for the strongest-confidence non-null drive
  const strongest = DRIVES.filter(d => decf[d].value != null)
    .sort((a,b) => decf[b].confidence - decf[a].confidence)[0]
  const gloss = strongest ? (decf[strongest].value >= 6 ? GLOSS[strongest].hi : GLOSS[strongest].lo) : 'a mix that needs more signal'

  // gaps: missing time + any null drive
  const nullDrives = DRIVES.filter(d => decf[d].value == null).map(d => DRIVE_NAME[d])
  const gapBits = []
  if (!time) gapBits.push('your rising (that needs a birth time)')
  if (nullDrives.length) gapBits.push(`enough signal on ${nullDrives.join(', ')} yet`)
  const gapText = gapBits.length ? gapBits.join('; and ') : 'nothing major — but this is still an early read'

  // novelty: the lowest-value non-null drive = the quietly-downplayed thing
  const lowest = DRIVES.filter(d => decf[d].value != null)
    .sort((a,b) => decf[a].value - decf[b].value)[0]
  const noveltyTrait = lowest ? DOWNPLAY[lowest] : 'something the three lines didn\'t reach'

  // bars (per-dimension confidence; null → "unknown")
  const bars = DRIVES.map(d => ({
    drive: DRIVE_NAME[d],
    level: decf[d].value == null ? 'unknown' : decf[d].confidence >= 0.6 ? 'high' : decf[d].confidence >= 0.33 ? 'medium' : 'low',
    value: decf[d].value, confidence: decf[d].confidence,
  }))

  // 4.5: each segment composed independently, sealed, NO cross-segment reference.
  const read = {
    headline: arch.primary ? pick(T.behavioral)(arch.primary, gloss) : 'Not enough signal yet for a read — tell me a little more.',
    bars,
    chart: chart.sun ? pick(T.chart)(chart.sun, chart.element) : null,       // parallel beam, never blended
    gapNote: pick(T.gap)(gapText),
    noveltyLine: arch.primary ? pick(T.novelty)(noveltyTrait) : null,
  }

  return {
    decf, archetype: arch,
    chart: { sun: chart.sun, element: chart.element, rising: chart.rising ?? null, gap: chart.gap ?? null },
    read,
    meta: { engine: 'sync-v0.1', deterministic: true, lowSignal: arch.confidence < 0.5, cmtrBoundaries: true },
  }
}
