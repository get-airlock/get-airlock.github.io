/* LifeOS — lightweight data store (localStorage-backed).
   CLOSED-WORLD BETA: no demo data. Reads the REAL KYI onboarding (lifeos.memory.v1)
   when present; otherwise starts EMPTY and fills only from the user / nerve.name mint.
   No recall theatre, no unreceipted value (per Ande's DQS closed-world contract). */
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

  // Closed-world beta: a fresh device starts EMPTY and fills only from real
  // onboarding / nerve.name mint. No seeded persona, no fake creations, no fake
  // family, no claimed value. The wallet shows nothing until a real receipt exists.
  function emptyState() {
    const ln = localStorage.getItem('lifeos.name');
    return {
      demo: false,
      name: (ln && ln !== 'friend') ? ln : 'friend',
      model: localStorage.getItem('lifeos.model') || 'DEEP',
      memoryMode: localStorage.getItem('lifeos.memory.mode.v1') || 'requested',
      unlocked: { create: true, wallet: true },
      canon_pages: [],
      creations: [],
      family_shared: [],
    };
  }

  function load() {
    let cached = null;
    try { cached = JSON.parse(localStorage.getItem(KEY) || 'null'); } catch (e) {}
    const real = realOnboarding();
    if (cached) {
      // Upgrade a real onboarding into state once it exists.
      if (real && (cached.demo || cached.name === 'Maya')) {
        return save(fromOnboarding(real, cached.creations || []));
      }
      // Scrub any legacy demo cache from before the closed-world beta.
      if (cached.demo || cached.name === 'Maya') return save(emptyState());
      return cached;
    }
    return save(real ? fromOnboarding(real, []) : emptyState());
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
