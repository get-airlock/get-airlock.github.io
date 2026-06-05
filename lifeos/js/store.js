/* LifeOS — lightweight mock data store (localStorage-backed, seeded for demo) */
(function () {
  const KEY = 'lifeos.state.v2';

  function seed() {
    const ln = localStorage.getItem('lifeos.name');
    return {
      name: (ln && ln !== 'friend') ? ln : 'Maya',
      model: localStorage.getItem('lifeos.model') || 'DEEP',
      memoryMode: 'requested',
      unlocked: { create: true, wallet: true }, // demo: surfaces open
      canon_pages: [
        {
          session: 'Arrive', ipfs_cid: 'bafy…k3q9',
          created_at: Date.now() - 1000 * 60 * 60 * 26,
          prompts: [
            { dimension: 'Meaning', question: 'What pulled you here today?', answer: 'I wanted somewhere quiet to think that wasn’t a notes app.' },
            { dimension: 'Time', question: 'What does this season of life feel like?', answer: 'Transitional. A lot is half-finished and that’s okay.' },
          ],
        },
        {
          session: 'Continue', ipfs_cid: 'bafy…r7m2',
          created_at: Date.now() - 1000 * 60 * 60 * 5,
          prompts: [
            { dimension: 'Relation', question: 'Who have you been thinking about?', answer: 'My sister. We keep missing each other’s calls.' },
            { dimension: 'Aspiration', question: 'What would “a good week” look like?', answer: 'Shipping one real thing and going for two long walks.' },
          ],
        },
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
    try {
      const raw = localStorage.getItem(KEY);
      if (raw) return JSON.parse(raw);
    } catch (e) {}
    const s = seed();
    save(s);
    return s;
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

  const LifeOSStore = {
    load, save, signals, addCreation,
    getMode() { return load().memoryMode; },
    setMode(m) { const s = load(); s.memoryMode = m; save(s); },
    isUnlocked(surface) { return !!(load().unlocked && load().unlocked[surface]); },
    familyShared() { return load().family_shared || []; },
    reset() { localStorage.removeItem(KEY); },
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
