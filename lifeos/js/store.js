/* LifeOS — lightweight data store (localStorage-backed).
   Reads the REAL KYI onboarding (lifeos.memory.v1) when present, and only falls
   back to the seeded demo ("Maya") if no one has onboarded yet. */
(function () {
  const KEY = 'lifeos.state.v2';
  const MEM_KEY = 'lifeos.memory.v1';   // where our KYI onboarding writes

  // ── DREAM (Constellation Credits) ↔ SIS (Proof-of-Creation value) — DEMO economics, TUNE THESE ──
  // DREAM = compute fuel, debited per creation (the POMR routing cost; on-chain form of Constellation Credits). SIS = minted PoC value.
  // The multiplier is a STUB for ConstellationBench L26-L44 depth/meaning scoring (the real exchange rate).
  const CREDIT_ALLOWANCE = 100;          // starting demo credit grant
  const CREDIT_COST_PER_CREATION = 3;    // fuel spent per validated creation
  const SIS_BASE = 10;                   // base SIS minted per validated creation
  const SIS_REFLECTION_MULT = 1.5;       // depth multiplier when a reflection is completed (L26-L44 proxy)

  function cap(s) { return s ? s[0].toUpperCase() + s.slice(1) : s; }

  // ── Real onboarding bridge ──────────────────────────────────────────────
  function realOnboarding() {
    try {
      const mem = JSON.parse(localStorage.getItem(MEM_KEY) || 'null');
      if (mem && Array.isArray(mem.canon_pages) && mem.canon_pages.length) return mem;
    } catch (e) {}
    return null;
  }

  function fromOnboarding(mem, keepCreations) {
    const ln = localStorage.getItem('lifeos.name');
    const canon = mem.canon_pages.map((p) => ({
      session: cap(p.session),
      ipfs_cid: p.ipfs_cid || 'bafy…local',
      created_at: Date.parse(p.created_at) || Date.now(),
      prompts: (p.prompts || p.prompts_and_answers || []).map((x) =>
        ({ dimension: cap(x.dimension), question: x.question, answer: x.answer })),
    }));
    let fam = [];
    if (mem.family && Array.isArray(mem.family.members)) {
      fam = mem.family.members
        .filter((m) => m.role !== 'guardian' && (m.shared_memory || []).length)
        .map((m) => ({ member: m.name, role: m.role, shared: (m.shared_memory || []).join(' ') }));
    }
    return {
      demo: false,
      name: (ln && ln !== 'friend') ? ln : 'friend',
      model: localStorage.getItem('lifeos.model') || 'DEEP',
      memoryMode: localStorage.getItem('lifeos.memory.mode.v1') || 'requested',
      unlocked: { create: true, wallet: true },
      canon_pages: canon,
      creations: keepCreations || mem.creations || [],
      family_shared: fam,
    };
  }

  function demoSeed() {
    const ln = localStorage.getItem('lifeos.name');
    return {
      demo: true,
      name: (ln && ln !== 'friend') ? ln : 'Maya',
      model: localStorage.getItem('lifeos.model') || 'DEEP',
      memoryMode: 'requested',
      unlocked: { create: true, wallet: true },
      canon_pages: [
        { session: 'Arrive', ipfs_cid: 'bafy…k3q9', created_at: Date.now() - 1000 * 60 * 60 * 26,
          prompts: [
            { dimension: 'Meaning', question: 'What pulled you here today?', answer: 'I wanted somewhere quiet to think that wasn’t a notes app.' },
            { dimension: 'Time', question: 'What does this season of life feel like?', answer: 'Transitional. A lot is half-finished and that’s okay.' },
          ] },
        { session: 'Continue', ipfs_cid: 'bafy…r7m2', created_at: Date.now() - 1000 * 60 * 60 * 5,
          prompts: [
            { dimension: 'Relation', question: 'Who have you been thinking about?', answer: 'My sister. We keep missing each other’s calls.' },
            { dimension: 'Aspiration', question: 'What would “a good week” look like?', answer: 'Shipping one real thing and going for two long walks.' },
          ] },
      ],
      creations: [
        { id: 'c1', kind: 'note', mock_cid: 'cid:Qm…a1f', created_at: Date.now() - 1000 * 60 * 90,
          body: 'A small idea: keep a “done” list next to the to-do list. Restless days look more productive in hindsight than they feel.',
          reflection: 'It reminds me that momentum is quieter than I expect.' },
        { id: 'c2', kind: 'note', mock_cid: 'cid:Qm…9e0', created_at: Date.now() - 1000 * 60 * 30,
          body: 'Voice memo, transcribed: the bridge near the river at dusk — that exact light is what calm feels like to me.',
          reflection: '(skipped)' },
      ],
      family_shared: [
        { member: 'Dad', role: 'family', shared: 'Has been feeling more settled this month; enjoying morning walks.' },
        { member: 'Sister', role: 'family', shared: 'Busy with a work deadline — would welcome a low-key check-in.' },
      ],
    };
  }

  function load() {
    let cached = null;
    try { cached = JSON.parse(localStorage.getItem(KEY) || 'null'); } catch (e) {}
    const real = realOnboarding();
    if (cached) {
      // Upgrade the demo seed to the real person once they've onboarded.
      if (real && (cached.demo || cached.name === 'Maya')) {
        return save(fromOnboarding(real, cached.creations || []));
      }
      return cached;
    }
    return save(real ? fromOnboarding(real, []) : demoSeed());
  }
  function save(s) { try { localStorage.setItem(KEY, JSON.stringify(s)); } catch (e) {} return s; }

  function signals() {
    const s = load();
    return {
      creations_validated: s.creations.length,
      reflections_completed: s.creations.filter((c) => c.reflection && c.reflection !== '(skipped)').length,
    };
  }

  function addCreation({ kind = 'note', body, reflection }) {
    const s = load();
    const n = (Math.random().toString(36).slice(2, 5) + Math.random().toString(36).slice(2, 5));
    const c = { id: 'c' + Date.now(), kind, body, reflection: reflection || '(skipped)',
                mock_cid: 'cid:Qm…' + n.slice(0, 3), created_at: Date.now() };
    s.creations.push(c);
    save(s);
    return c;
  }

  // Constellation Credits — the fuel ledger (spent per creation).
  function credits() {
    const s = load();
    const spent = s.creations.length * CREDIT_COST_PER_CREATION;
    return { allowance: CREDIT_ALLOWANCE, spent, remaining: Math.max(0, CREDIT_ALLOWANCE - spent), costPer: CREDIT_COST_PER_CREATION };
  }
  // SIS — the Proof-of-Creation mint (value = base × quality multiplier, per creation).
  function sisValue() {
    const s = load();
    let total = 0;
    const per = s.creations.map((c) => {
      const mult = (c.reflection && c.reflection !== '(skipped)') ? SIS_REFLECTION_MULT : 1.0;
      const value = Math.round(SIS_BASE * mult);
      total += value;
      return { id: c.id, base: SIS_BASE, mult, value, credits: CREDIT_COST_PER_CREATION };
    });
    return { total, per, base: SIS_BASE };
  }

  const LifeOSStore = {
    load, save, signals, addCreation, credits, sisValue,
    getMode() { return load().memoryMode; },
    setMode(m) { const s = load(); s.memoryMode = m; save(s); },
    isUnlocked(surface) { return !!(load().unlocked && load().unlocked[surface]); },
    familyShared() { return load().family_shared || []; },
    reset() { localStorage.removeItem(KEY); },
    isDemo() { return !!load().demo; },
    familySummary(name) {
      return `Here’s what ${name} chose to share, gently summarised:\n\n` +
        `• Feeling a bit restless but steady this week.\n` +
        `• Has been thinking about family and wants to reconnect with their sister.\n` +
        `• A “good week” means shipping one real thing and taking two long walks.\n\n` +
        `Nothing private was opened to write this — only what ${name} marked as shareable.`;
    },
  };

  window.LifeOSStore = LifeOSStore;
})();
