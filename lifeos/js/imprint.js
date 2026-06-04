// LifeOS Imprint — DETERMINISTIC DECF scorer (zero-LLM, zero-network, cannot drain a key).
// "Get the array from imprint ⇒ LifeOS," the free-tier way: match the DECF signal-words
// against the KYI answers directly instead of asking an LLM. Same source of truth as
// the Pro extractor (airlock-persona inference/drive-signals.yaml + constellation-bench
// data/signal-words/decf-signals.json) — pattern-matched, not classified.
//
// D=Dominance  E=Extraversion  C=Patience/Consistency  F=Formality
// Per airlock-persona rule: store the array internally; NEVER show raw DECF numbers in UI.

const SIGNALS = {
  D: {
    high: ["must","immediately","ship","now","decisive","bold","act","move","push","own","lead","drive","execute","deploy","launch","cut","prioritize","non-negotiable","this is how","demand","command","take charge","decide","force","override","authority"],
    low:  ["perhaps","maybe","consider","might","could","suggest","if possible","it depends","we should explore","option","alternative"],
  },
  E: {
    high: ["team","collaborate","together","discuss","share","align","communicate","rally","engage","everyone","across","stakeholder","feedback","sync","broadcast","meeting","huddle","connect"],
    low:  ["alone","independently","solo","private","focused","quiet","deep work","uninterrupted","heads down"],
  },
  C: {
    high: ["careful","thorough","systematic","methodical","patience","deliberate","steady","sustained","long-term","step-by-step","incremental","measured","review","verify","validate","audit","document","specification","diligent","meticulous","gradual"],
    low:  ["urgently","quickly","fast","sprint","rush","asap","immediately","right now","ship it","iterate fast","mvp","rapid","speed"],
  },
  F: {
    high: ["process","standard","compliance","documentation","procedure","protocol","guideline","schema","contract","format","versioning","changelog","checklist","approval","regulation","policy","framework","governance"],
    low:  ["skip","bypass","just do it","move fast","iterate","prototype","rough","hacky","good enough","wing it","scrappy"],
  },
};

const Imprint = {
  // Count how many signal words/phrases appear in the text (presence per signal).
  _count(text, words) {
    let n = 0;
    for (const w of words) {
      if (w.includes(' ')) { if (text.includes(w)) n++; continue; }
      const esc = w.replace(/[-]/g, '\\-');
      // Words >=5 chars match inflections (careful→carefully); short words stay strict
      // to avoid false positives (act→actually, now→nowhere).
      const re = w.length >= 5 ? new RegExp(`\\b${esc}`) : new RegExp(`\\b${esc}\\b`);
      if (re.test(text)) n++;
    }
    return n;
  },

  // Score one drive: lean 0..1 (0.5 neutral), plus the raw match counts.
  _drive(text, drive) {
    const s = SIGNALS[drive];
    const high = this._count(text, s.high);
    const low = this._count(text, s.low);
    const total = high + low;
    const lean = total === 0 ? 0.5 : high / total;   // 0 = low pole, 1 = high pole
    return { lean: Math.round(lean * 100) / 100, high, low, total };
  },

  // Score free text → the DECF array (deterministic, no key).
  score(text) {
    const t = (text || '').toLowerCase();
    const drives = {};
    let signalSum = 0;
    for (const d of ['D', 'E', 'C', 'F']) {
      drives[d] = this._drive(t, d);
      signalSum += drives[d].total;
    }
    // Confidence rises with evidence; ~6 matched signals → fully confident.
    const confidence = Math.min(1, Math.round((signalSum / 6) * 100) / 100);
    return {
      decf: { D: drives.D.lean, E: drives.E.lean, C: drives.C.lean, F: drives.F.lean },
      detail: drives,
      confidence,
      signals_matched: signalSum,
      method: 'deterministic-lexical-v1',
      updated_at: new Date().toISOString(),
    };
  },

  // Pull every KYI answer from memory, score the concatenation, write back state.imprint.
  recompute() {
    if (typeof window === 'undefined' || !window.LifeOSMemory) return null;
    const state = window.LifeOSMemory.load();
    const answers = (state.canon_pages || [])
      .flatMap((p) => (p.prompts || []).map((q) => q.answer))
      .filter((a) => a && a !== '(skipped)');
    const name = (localStorage.getItem('lifeos.name') || '');
    const blob = [name, ...answers].join('  ').trim();
    state.imprint = blob ? this.score(blob) : null;
    window.LifeOSMemory.save(state);
    return state.imprint;
  },

  read() {
    if (typeof window === 'undefined' || !window.LifeOSMemory) return null;
    return window.LifeOSMemory.load().imprint || null;
  },

  // Descriptive, never-numeric phrasing (honors the no-raw-DECF-in-copy rule).
  describe(imprint) {
    if (!imprint) return '';
    const labels = {
      D: ['takes a gentler, collaborative lead', 'leans bold and decisive'],
      E: ['prefers focused, quiet space', 'is energized by people and connection'],
      C: ['moves fast and iterative', 'works careful and methodical'],
      F: ['keeps it informal and scrappy', 'values structure and process'],
    };
    const parts = [];
    for (const d of ['D', 'E', 'C', 'F']) {
      const lean = imprint.decf[d];
      if (imprint.detail[d].total === 0) continue;        // no evidence → say nothing
      parts.push(labels[d][lean >= 0.5 ? 1 : 0]);
    }
    return parts.length ? parts.join('; ') : '';
  },
};

if (typeof window !== 'undefined') window.LifeOSImprint = Imprint;
