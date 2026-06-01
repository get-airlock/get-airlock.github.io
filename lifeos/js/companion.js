// LifeOS Mock Companion — calm, references-only-allowed-memory, never claims hidden profile
// Per Sisyphus DAG L3.2, invariants G1 (no deterministic identity) + G4 (consented memory)
// Built 2026-05-30 for Rika June 1 demo

// Demo brain + voice live here (keyless to the client). Folds into airlock-api (Fly) later.
const SHUTTLE = 'https://lifeos-shuttle.vercel.app';

const Companion = {
  // Running conversation for continuity (so the companion doesn't loop).
  history: [],

  // Which companion soul is active (server picks the matching persona). 'carrie' | 'sparky'
  companionId: 'carrie',
  setCompanion(id) { if (id) this.companionId = String(id).toLowerCase(); },

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
          body: JSON.stringify({ messages: outgoing, memory, companion: this.companionId }),
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

  // ── Offline fallback ONLY (used when the live brain can't be reached). ──
  // Deliberately NOT an onboarding menu — a brief reconnect line that holds the thread,
  // so a momentary network blip never pulls the user back into "onboarding" feeling.
  reply() {
    const lines = [
      "Sorry — I lost the connection for a second there. Say that once more and I'll pick right back up.",
      "I didn't quite catch that — could you say it again? I'm still right here with you.",
      "One moment, my connection hiccuped. Go ahead and repeat that and we'll keep going.",
    ];
    this._fallbackIdx = ((this._fallbackIdx || 0) + 1) % lines.length;
    return lines[this._fallbackIdx];
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
