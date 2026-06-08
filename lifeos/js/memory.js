// LifeOS Memory Layer — localStorage-backed continuity store
// Per Sisyphus DAG L4.1, invariants G1 (continuity not deterministic) + G4 (consented, visible, explainable)
// Built 2026-05-30 for Rika June 1 demo

const MEMORY_KEY = 'lifeos.memory.v1';
const MODE_KEY = 'lifeos.memory.mode.v1'; // 'nothing' | 'requested' | 'all'

// ── Memory modes per Sisyphus L4.2 ──
const MEMORY_MODES = {
  nothing: { label: 'Nothing', desc: 'LifeOS forgets between visits.' },
  requested: { label: 'Only requested', desc: 'LifeOS remembers what you ask it to.' },
  all: { label: 'All', desc: 'LifeOS carries the whole conversation forward.' }
};

// ── Default seed state ──
function defaultState() {
  return {
    version: '0.1.0',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    canon_pages: [],          // each page = { id, session, prompts, answers, created_at }
    progress: {
      arrive_complete: false,
      continue_complete: false,
      contribute_complete: false
    },
    family: defaultFamilyMock(),
    creations: [],
    signals: { creations_validated: 0, reflections_completed: 0 },
    imprint: null            // deterministic DECF lean, recomputed from KYI answers (zero-LLM)
  };
}

// ── Family continuity mock per Sisyphus L7.1 ──
function defaultFamilyMock() {
  return {
    family_name: 'Demo Family',
    members: [
      {
        name: 'You',
        role: 'guardian',
        private_memory: [],
        shared_memory: []
      },
      {
        name: 'Carrie',
        role: 'teen',
        private_memory: ['Working on a reflection project, prefers quiet time.'],
        shared_memory: ['Calm support after school works best for Carrie.']
      }
    ]
  };
}

// ── Public API ──
const Memory = {
  load() {
    try {
      const raw = localStorage.getItem(MEMORY_KEY);
      if (!raw) {
        const fresh = defaultState();
        localStorage.setItem(MEMORY_KEY, JSON.stringify(fresh));
        return fresh;
      }
      return JSON.parse(raw);
    } catch (e) {
      return defaultState();
    }
  },

  save(state) {
    state.updated_at = new Date().toISOString();
    localStorage.setItem(MEMORY_KEY, JSON.stringify(state));
    return state;
  },

  reset() {
    localStorage.removeItem(MEMORY_KEY);
    localStorage.removeItem(MODE_KEY);
    return defaultState();
  },

  // ── Memory mode (Nothing / Only requested / All) ──
  getMode() { return localStorage.getItem(MODE_KEY) || 'requested'; },
  setMode(mode) {
    if (!MEMORY_MODES[mode]) return;
    localStorage.setItem(MODE_KEY, mode);
  },
  modes() { return MEMORY_MODES; },

  // ── KYI session canon-page writing ──
  addCanonPage({ session, prompts_and_answers }) {
    const state = Memory.load();
    const page = {
      id: `page_${Date.now()}`,
      session,
      prompts: prompts_and_answers,
      created_at: new Date().toISOString(),
      // Mock CID — per L0.3 mock-boundary, no real IPFS for demo
      ipfs_cid: `bafy${Math.random().toString(36).slice(2, 12)}mockdemo`
    };
    state.canon_pages.push(page);
    if (state.progress[`${session}_complete`] !== undefined) {
      state.progress[`${session}_complete`] = true;
    }
    Memory.save(state);
    // Deterministic imprint refresh — zero-LLM, no key. Recomputes the DECF array from answers.
    if (typeof window !== 'undefined' && window.LifeOSImprint) window.LifeOSImprint.recompute();
    return Memory.load();
  },

  // ── Creation/reflection writing ──
  addCreation({ kind, body, reflection }) {
    const state = Memory.load();
    const creation = {
      id: `creation_${Date.now()}`,
      kind, // 'note' | 'voice' | 'drawing'
      body,
      reflection,
      created_at: new Date().toISOString(),
      mock_cid: `bafy${Math.random().toString(36).slice(2, 12)}mockcreate`,
      signals: {
        validated_poc: true,
        reflection_completed: !!reflection,
        value_assigned: 'pending-demo'
      }
    };
    state.creations.push(creation);
    state.signals.creations_validated += 1;
    if (reflection) state.signals.reflections_completed += 1;
    return Memory.save(state);
  },

  // ── Allowed memory snapshot — for companion to use ──
  allowedMemorySnapshot() {
    const state = Memory.load();
    const mode = Memory.getMode();
    if (mode === 'nothing') return { mode, canon_pages: [], creations: [] };
    // For demo: 'requested' and 'all' both show the same — the consent UI is the surface
    return {
      mode,
      canon_pages: state.canon_pages,
      creations: state.creations,
      progress: state.progress,
      signals: state.signals,
      imprint: state.imprint || null
    };
  },

  // ── Progress checks for progressive unlock per Sisyphus L2.3 ──
  isUnlocked(surface) {
    const state = Memory.load();
    switch (surface) {
      case 'home':
      case 'kyi':
        return true;
      case 'memory':
        return state.progress.arrive_complete;
      case 'create':
        return state.progress.continue_complete;
      case 'wallet':
        return state.progress.contribute_complete;
      default:
        return false;
    }
  },

  // ── Family continuity per Sisyphus L7.2 ──
  familyShared() {
    const state = Memory.load();
    const shared = [];
    state.family.members.forEach((m) => {
      m.shared_memory.forEach((s) => shared.push({ member: m.name, role: m.role, shared: s }));
    });
    return shared;
  }
};

// Make available globally for vanilla HTML pages
if (typeof window !== 'undefined') {
  window.LifeOSMemory = Memory;
}
