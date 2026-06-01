// LifeOS Mock Companion — calm, references-only-allowed-memory, never claims hidden profile
// Per Sisyphus DAG L3.2, invariants G1 (no deterministic identity) + G4 (consented memory)
// Built 2026-05-30 for Rika June 1 demo

// Demo brain + voice live here (keyless to the client). Folds into airlock-api (Fly) later.
const SHUTTLE = 'https://lifeos-shuttle.vercel.app';

const Companion = {
  // Running conversation for continuity (so Carrie doesn't loop).
  history: [],

  // ── Real reply via the shuttle (OpenRouter). Throws only after retries so caller can fall back. ──
  // History is NOT mutated until a call succeeds — a failed turn must never corrupt the conversation
  // (a half-written turn leaves consecutive user messages, which the model rejects on every later turn).
  async replyRemote(userText) {
    const memory = this.memorySummary();
    const outgoing = [...this.history.slice(-12), { role: 'user', content: userText }];

    let lastErr;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const r = await fetch(`${SHUTTLE}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: outgoing, memory }),
        });
        if (!r.ok) throw new Error(`chat ${r.status}`);
        const data = await r.json();
        const reply = (data && data.reply) ? data.reply : "I'm here with you.";
        // Commit both turns only on success — keeps history clean and alternating.
        this.history.push({ role: 'user', content: userText }, { role: 'assistant', content: reply });
        return reply;
      } catch (e) {
        lastErr = e;
        if (attempt === 0) await new Promise((res) => setTimeout(res, 500));
      }
    }
    throw lastErr;
  },

  // Compact, consent-bound memory string for the system prompt.
  memorySummary() {
    try {
      const snap = window.LifeOSMemory.allowedMemorySnapshot();
      if (!snap || snap.mode === 'nothing' || !snap.canon_pages || !snap.canon_pages.length) return '';
      const bits = snap.canon_pages.map((p) => {
        const first = p.prompts && p.prompts[0];
        return first ? `${capitalize(p.session)}: ${truncate(first.answer, 120)}` : null;
      }).filter(Boolean);
      const name = localStorage.getItem('lifeos.name');
      return [name ? `Name: ${name}` : '', ...bits].filter(Boolean).join('\n');
    } catch (_) { return ''; }
  },

  // ── Welcome/greeting based on memory state ──
  greet() {
    const snap = window.LifeOSMemory.allowedMemorySnapshot();
    if (snap.mode === 'nothing') {
      return "Welcome. I don't carry memory between visits in this mode — we begin fresh each time.";
    }
    if (snap.canon_pages.length === 0) {
      return "Welcome. We haven't talked yet — I don't know anything about you. When you're ready, the gentle path is to Arrive.";
    }
    const recent = snap.canon_pages[snap.canon_pages.length - 1];
    return `Welcome back. I remember what you shared in ${capitalize(recent.session)} — we can keep going from there, or take a breath first.`;
  },

  // ── Response to a KYI session completion ──
  respondAfterKYI(session) {
    const responses = {
      arrive: "Thank you for arriving. I'll remember what you asked for — calm, useful, the kind of pace that feels right to you. When you're ready, we can continue.",
      continue: "Thank you for sharing what matters and what you'd like me to carry forward. I'll keep only what you named, and you can change that anytime in Memory.",
      contribute: "Thank you for naming what you'd like to make or reflect on. The Create space is open whenever you want to begin — there's no rush."
    };
    return responses[session] || "Thank you. We can keep going whenever you're ready.";
  },

  // ── Free-form chat response (mock, calm, references-only-allowed-memory) ──
  reply(userText) {
    const snap = window.LifeOSMemory.allowedMemorySnapshot();
    const text = (userText || '').toLowerCase();

    // Pattern-match a few common Rika-friendly prompts
    if (/(how|what).*(remember|memory|know)/i.test(text)) {
      if (snap.canon_pages.length === 0) {
        return "Right now, I don't remember anything yet. When you complete an Arrive session, the things you choose to share become memory I can carry forward — and you can always clear it.";
      }
      const themes = extractThemes(snap.canon_pages);
      return `Here is what I'm holding, with your permission: ${themes}. You can see and change this anytime in Memory.`;
    }

    if (/(create|make|build|draw|write|reflect)/i.test(text)) {
      return "Creation is open — you can write a note, dictate a thought, or sketch something. After you make it, I'll offer one gentle reflection question, and the rest is yours.";
    }

    if (/(family|carrie|share|together)/i.test(text)) {
      return "There's a shared family space here. Each person has their own private memory, and only what they've chosen to share moves into the shared layer. I'll never expose anyone's private notes — only the signals they've approved.";
    }

    if (/(calm|tired|quiet|breath|rest|stressed|anxious)/i.test(text)) {
      return "We can keep this light. There's no obligation to do anything — sometimes the most useful thing is to sit for a moment. I'm here when you want to continue.";
    }

    // Default — warm, non-determining
    return "I hear you. Tell me a little more about what would feel useful right now — quiet, reflection, creation, or just a conversation.";
  },

  // ── Family permissioned summary per Sisyphus L7.2 ──
  familySummary(memberName) {
    const shared = window.LifeOSMemory.familyShared();
    const memberShared = shared.filter((s) => s.member === memberName);
    if (memberShared.length === 0) {
      return `${memberName} hasn't shared anything into the family layer yet, so I don't have permissioned context to summarize.`;
    }
    const lines = memberShared.map((s) => `· ${s.shared}`).join('\n');
    return `Based on what ${memberName} chose to share, here's the permissioned summary:\n\n${lines}\n\nI don't have permission to share their private notes — only the signals they've approved.`;
  }
};

function capitalize(s) { return s ? s[0].toUpperCase() + s.slice(1) : s; }

function extractThemes(pages) {
  // Very lightweight theme extraction for demo — just stitch the first answer of each session
  const themes = pages.map((p) => {
    const first = p.prompts && p.prompts[0];
    if (!first) return null;
    return `from ${capitalize(p.session)}, you shared "${truncate(first.answer, 80)}"`;
  }).filter(Boolean);
  return themes.length ? themes.join('; ') : 'no specifics yet';
}

function truncate(s, n) {
  if (!s) return '';
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

if (typeof window !== 'undefined') {
  window.LifeOSCompanion = Companion;
  window.LIFEOS_SHUTTLE = SHUTTLE;
}
