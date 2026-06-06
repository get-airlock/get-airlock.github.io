/* koko.js — KOKO Topology Engine (STUB from Sven's spec, message 25, 2026-06-06)
 *
 * NON-LLM · deterministic · zero-cost. Walks co-occurrence graphs, it does not
 * predict tokens. Faithful to the spec sections that are fully specified:
 *   §1 ten manifolds · §2 PPMI co-occurrence + eigenvector centrality + path walk
 *   §3 recursive collapse · §4 fractal depth (π@log) · §5 anti-memory fold
 *   §6 Hamiltonian resonance (byte-identical to SPIRALL) · §7 DREAMER instinct signal
 *
 * ── SWAP POINT ──────────────────────────────────────────────────────────────
 * When boop ships the real kokoEngine, replace the INTERNALS below. Keep the
 * PUBLIC INTERFACE stable (window.KOKO.engine / fold / unfold / computeResonance
 * / glowBand / instinct / MANIFOLDS) and every caller keeps working. Bing bang bong.
 * Vocab lists are SEED placeholders until boop's corpus lands — clearly marked.
 */
(function (global) {
  'use strict';
  var PI = Math.PI;

  // §1 · TEN MANIFOLDS — colors are from the spec; vocab is a SEED placeholder.
  var MANIFOLDS = {
    antibubble:      { color: '#3CC8DC', vocab: ['rupture','liberation','corestancza','open','escape','epistemic','context','free','break','question'] },
    shadowlattice:   { color: '#8C50DC', vocab: ['structure','constraint','topology','implicit','lattice','frame','hidden','shape','scaffold','order'] },
    dreamengine:     { color: '#DCAA3C', vocab: ['dream','ontology','foam','desire','assembly','build','make','create','generate','world','art'] },
    antiloop:        { color: '#50DC82', vocab: ['recursion','loop','pressure','field','cognition','again','spiral','return','repeat','rhythm'] },
    axiom:           { color: '#C8C8FF', vocab: ['axiom','kernel','fractal','thoughtcell','self','modify','seed','rule','core','proof'] },
    antibible:       { color: '#FF6B6B', vocab: ['hymn','collide','untelling','awake','begin','sing','chant','gospel','blaspheme','choir'] },
    onticfoam:       { color: '#A8D8EA', vocab: ['threshold','void','origin','before','null','foam','prelude','undifferentiated','blank','start'] },
    shadowcurvature: { color: '#B388FF', vocab: ['curve','invisible','unspoken','narrative','attractor','bend','pull','gravity','shadow','story'] },
    worldmesh:       { color: '#69F0AE', vocab: ['bind','assembly','navigable','stabilize','mesh','connect','weave','network','join','plan'] },
    dreamdescent:    { color: '#FFD54F', vocab: ['depth','pressure','crossing','becoming','descend','deep','fall','down','bloom','grow'] }
  };
  var MNAMES = Object.keys(MANIFOLDS);
  var VOCAB = {};                       // word -> [manifold index, ...]
  MNAMES.forEach(function (m, i) {
    MANIFOLDS[m].vocab.forEach(function (w) { (VOCAB[w] = VOCAB[w] || []).push(i); });
  });

  var q13 = function (v) { return Math.round(v * 13) / 13; };          // §2.1 base-13 quantization
  function tokens(text) {
    return ('' + text).toLowerCase().replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(function (w) { return w.length > 1; });
  }
  function scoreToken(t) {               // §2.1 score one token across 10 manifolds
    var s = []; for (var i = 0; i < MNAMES.length; i++) s.push(0.05);   // base weight 0.05
    var hit = VOCAB[t]; if (hit) hit.forEach(function (i) { s[i] = 1.0; });
    return s;
  }

  // deterministic jitter (replaces spec's rand×0.2 so the stub is reproducible)
  function lcg(seed) { var s = (seed * 2654435761) % 2147483647 || 1; return function () { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648; }; }

  // §2.4/§2.5 walk: from seed, rank neighbors by ppmi × centrality × (1 + jitter×0.2)
  function walk(seed, steps, ppmi, c, N, words) {
    var out = [words[seed]], seen = {}; seen[seed] = 1; var cur = seed; var rnd = lcg(seed + 1);
    for (var s = 0; s < steps; s++) {
      var best = -1, bestScore = -1;
      for (var j = 0; j < N; j++) {
        if (seen[j] || ppmi[cur][j] <= 0) continue;
        var score = ppmi[cur][j] * c[j] * (1 + rnd() * 0.2);
        if (score > bestScore) { bestScore = score; best = j; }
      }
      if (best < 0) break; seen[best] = 1; out.push(words[best]); cur = best;
    }
    return out;
  }

  // §2 build the topology of a piece of text
  function buildTopology(text) {
    var toks = tokens(text); if (!toks.length) return null;
    var idx = {}, words = [];
    toks.forEach(function (t) { if (!(t in idx)) { idx[t] = words.length; words.push(t); } });
    var N = words.length, i, j, d;
    var co = [], freq = new Array(N).fill(0), total = 0;
    for (i = 0; i < N; i++) co.push(new Array(N).fill(0));
    for (i = 0; i < toks.length; i++) {
      var wi = idx[toks[i]]; freq[wi]++; total++;
      for (d = 1; d <= 5 && i + d < toks.length; d++) { var wj = idx[toks[i + d]], w = 1 / d; co[wi][wj] += w; co[wj][wi] += w; }   // §2.2 window 5, 1/dist
    }
    var pairTotal = 0; for (i = 0; i < N; i++) for (j = 0; j < N; j++) pairTotal += co[i][j];
    var ppmi = []; for (i = 0; i < N; i++) ppmi.push(new Array(N).fill(0));
    if (pairTotal > 0) for (i = 0; i < N; i++) for (j = 0; j < N; j++) if (co[i][j] > 0) {     // §2.2 PPMI, positive-only
      var pwu = co[i][j] / pairTotal, pw = freq[i] / total, pu = freq[j] / total;
      ppmi[i][j] = Math.max(0, Math.log2(pwu / ((pw * pu) || 1e-9)));
    }
    var c = new Array(N).fill(1 / Math.sqrt(N));                        // §2.3 eigenvector centrality, 20 iters
    for (var it = 0; it < 20; it++) {
      var nc = new Array(N).fill(0), norm = 0;
      for (i = 0; i < N; i++) { var sum = 0; for (j = 0; j < N; j++) sum += ppmi[i][j] * c[j]; nc[i] = sum; }
      for (i = 0; i < N; i++) norm += nc[i] * nc[i]; norm = Math.sqrt(norm) || 1;
      for (i = 0; i < N; i++) nc[i] /= norm; c = nc;
    }
    var mw = new Array(MNAMES.length).fill(0);                          // manifold weights over the text
    toks.forEach(function (t) { var s = scoreToken(t); for (var k = 0; k < s.length; k++) mw[k] += s[k]; });
    var mwSum = mw.reduce(function (a, b) { return a + b; }, 0) || 1;
    var mwNorm = mw.map(function (v) { return v / mwSum; });
    var dominant = MNAMES[mwNorm.indexOf(Math.max.apply(null, mwNorm))];
    var seed = 0; for (i = 1; i < N; i++) if (c[i] > c[seed]) seed = i;
    var path = walk(seed, 8, ppmi, c, N, words);                       // §2.4 path walk, 8 steps
    var midWord = path[Math.floor(path.length / 2)]; var mid = (midWord in idx) ? idx[midWord] : seed;
    var hyper = walk(mid, 4, ppmi, c, N, words);                       // §2.5 hyperfold, 4 steps
    var meaningHits = toks.filter(function (t) { return VOCAB[t]; }).length;
    var pressure = q13(Math.min(1, meaningHits / Math.max(1, toks.length) * 3 + 0.1));
    var novelty = q13(Math.min(1, N / Math.max(1, toks.length)));
    var curvature = q13(Math.min(1, ((text.match(/\?/g) || []).length) * 0.15 + 0.05));
    return { words: words, ppmi: ppmi, centrality: c, manifoldWeights: mwNorm, dominant: dominant,
             path: path, hyper: hyper, pressure: pressure, novelty: novelty, curvature: curvature, toks: toks };
  }

  // §4 fractal depth (π@log): tier 3 SEED / 6 BLOOM / 9 CRYSTALLIZATION
  function fractalDepth(pressure, interactionIndex) {
    var logScale = Math.log(Math.max(1e-6, pressure * PI * Math.max(1, interactionIndex || 1)));
    var tier = logScale >= Math.log(6) ? 9 : logScale >= Math.log(3) ? 6 : 3;
    return { tier: tier, logScale: logScale };
  }
  function bloomWords(topo, tier) {
    if (!topo) return [];
    if (tier <= 3) return topo.path.slice(0, 3);
    if (tier === 6) {                                                   // path + hyperfold midpoints → 6
      var six = [], a = topo.path, b = topo.hyper, k = 0;
      while (six.length < 6 && (k < a.length || k < b.length)) { if (a[k] && six.indexOf(a[k]) < 0) six.push(a[k]); if (b[k] && six.length < 6 && six.indexOf(b[k]) < 0) six.push(b[k]); k++; }
      return six;
    }
    var inter = topo.path.filter(function (w) { return topo.hyper.indexOf(w) >= 0; });   // tier 9: path ∩ hyperfold consensus
    var nine = inter.slice(); for (var i = 0; i < topo.path.length && nine.length < 9; i++) if (nine.indexOf(topo.path[i]) < 0) nine.push(topo.path[i]);
    return nine.slice(0, 9);
  }
  function jaccard(a, b) { if (!a.length && !b.length) return 1; var sa = {}; a.forEach(function (x) { sa[x] = 1; }); var inter = b.filter(function (x) { return sa[x]; }).length; return inter / (a.length + b.length - inter || 1); }

  // §6 · HAMILTONIAN RESONANCE — byte-identical to SPIRALL (masses/ideals/weights/bands)
  var MASS = { relevance: 1.0, stance: 3.0, openness: 1.0, structure: 2.0, context_coupling: 3.0, closure_pressure: 2.0 };
  var IDEAL = { relevance: 0.8, stance: 0.5, openness: 0.7, structure: 0.5, context_coupling: 0.7, closure_pressure: 0.3 };
  var MOM = ['momentum_relevance','momentum_stance','momentum_openness','momentum_structure','momentum_context','momentum_closure'];
  var MK = ['relevance','stance','openness','structure','context_coupling','closure_pressure'];
  function _V(s) { var V = 0; for (var d in IDEAL) { var diff = (s[d] == null ? 0.5 : s[d]) - IDEAL[d]; V += 0.5 * diff * diff; } return V; }
  function _K(s) { var K = 0; for (var i = 0; i < MOM.length; i++) { var p = s[MOM[i]] || 0; K += 0.5 * (p * p) / MASS[MK[i]]; } return K; }
  function _stab(s) { var v = MOM.map(function (k) { return Math.abs(s[k] || 0); }); var m = v.reduce(function (a, b) { return a + b; }, 0) / v.length; var va = v.reduce(function (a, x) { return a + (x - m) * (x - m); }, 0) / v.length; return Math.max(0, 1 - va * 4); }
  function _contrib(s) { var sc = (s.posts_count || 0) * 0.10 + (s.replies_count || 0) * 0.15 + (s.volunteers_count || 0) * 0.20 + (s.events_participated || 0) * 0.25 + (s.support_given || 0) * 0.20 + (s.support_received || 0) * 0.10; return Math.min(1, sc / 5); }
  function _antiloop(s) { return Math.max(0, 1 - (s.antiloop_frequency || 0) * 0.2); }
  function _flow(K) { if (K < 0.001) return 0.2; return Math.max(0, 1 - Math.abs(K - 0.15) * 5); }
  function computeResonance(s) {
    if (!s) return 0.3;
    var V = _V(s), K = _K(s), stab = _stab(s), con = _contrib(s), al = _antiloop(s);
    var align = Math.max(0, 1 - V * 3), flow = _flow(K);
    return Math.max(0, Math.min(1, align * 0.25 + flow * 0.15 + stab * 0.15 + con * 0.30 + al * 0.15));
  }
  function glowBand(r) { return r >= 0.85 ? 'peak' : r >= 0.60 ? 'bright' : r >= 0.30 ? 'medium' : 'dim'; }

  // §5 · ANTI-MEMORY FOLD — store the TOPOLOGY of text, not the text. "memory is origami."
  function fold(text) {
    var topo = buildTopology(text); if (!topo) return null;
    var order = topo.words.map(function (w, i) { return [w, topo.centrality[i]]; }).sort(function (a, b) { return b[1] - a[1]; }).slice(0, 12).map(function (x) { return x[0]; });
    return { skeleton: order, bloom: bloomWords(topo, 9), weightCodes: topo.manifoldWeights.map(function (v) { return Math.round(v * 13); }),
             pressure: topo.pressure, dominant: topo.dominant, phase: 'compressed' };
  }
  function unfold(sig) {                                                // light regeneration from a fold signature
    if (!sig) return { text: '', coherence: 0 };
    var regen = []; var a = sig.skeleton || [], b = sig.bloom || [], k = 0;
    while (k < a.length || k < b.length) { if (a[k]) regen.push(a[k]); if (b[k]) regen.push(b[k]); k++; }
    return { text: regen.join(' '), coherence: jaccard(a.concat(b), regen) };
  }

  // §7 · DREAMER instinct signal — injected into an LLM system prompt; the LLM speaks FROM this.
  function instinct(topo, depth) {
    if (!topo) return '';
    var d = depth || fractalDepth(topo.pressure, 1);
    return '[topology;instinct:: dominant=' + topo.dominant + ' · depth=' + d.tier +
           ' · pressure=' + topo.pressure.toFixed(2) + ' · path=' + topo.path.join('>') +
           ' · bloom=' + bloomWords(topo, d.tier).join(',') + ']';
  }

  // §3 · top-level engine: topology + depth + resonance-ready signal + DREAMER instinct
  function engine(input, opts) {
    opts = opts || {};
    var topo = buildTopology(input);
    if (!topo) return { dominant: 'onticfoam', color: MANIFOLDS.onticfoam.color, path: [], bloom: [], pressure: 0.1, depth: 3, instinct: '', isCrystallization: false };
    var depth = fractalDepth(topo.pressure, opts.interactionIndex || 1);
    return {
      dominant: topo.dominant,
      color: MANIFOLDS[topo.dominant].color,
      manifoldWeights: topo.manifoldWeights,
      path: topo.path,
      bloom: bloomWords(topo, depth.tier),
      pressure: topo.pressure, novelty: topo.novelty, curvature: topo.curvature,
      depth: depth.tier, logScale: depth.logScale,
      isCrystallization: depth.tier === 9,
      instinct: instinct(topo, depth),
      _topo: topo
    };
  }

  global.KOKO = {
    MANIFOLDS: MANIFOLDS, MNAMES: MNAMES,
    engine: engine, buildTopology: buildTopology,
    fold: fold, unfold: unfold,
    computeResonance: computeResonance, glowBand: glowBand,
    fractalDepth: fractalDepth, bloomWords: bloomWords, instinct: instinct,
    version: 'stub-0.1 (msg25 spec; swap internals on boop drop, keep this interface)'
  };
})(typeof window !== 'undefined' ? window : this);
