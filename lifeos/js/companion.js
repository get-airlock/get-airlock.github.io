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

  // Sticky: once a help request fires (intent-collapse), assistance stays on for the session.
  assist: false,
  _HELP_RE: /\b(help me|plan|find|organize|organise|schedule|study|studying|tutor|homework|remind|fix|figure out|work on|decide|build|make)\b/i,

  // ── Real reply via the shuttle (OpenRouter). Throws only after retries so caller can fall back. ──
  // History is NOT mutated until a call succeeds — a failed turn must never corrupt the conversation
  // (a half-written turn leaves consecutive user messages, which the model rejects on every later turn).
  async replyRemote(userText) {
    if (!this.assist && this._HELP_RE.test(userText)) this.assist = true;   // intent-collapse latch
    const memory = this.memorySummary();
    const outgoing = [...this.history.slice(-12), { role: 'user', content: userText }];

    let lastErr;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const r = await fetch(`${SHUTTLE}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ messages: outgoing, memory, companion: this.companionId, assist: this.assist }),
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
      // Deterministic imprint → a tone hint the brain uses to match energy (never a label said aloud).
      let style = '';
      try {
        if (snap.imprint && window.LifeOSImprint) {
          const d = window.LifeOSImprint.describe(snap.imprint);
          if (d) style = `Tone hint (match their energy; never say this aloud, never label them): ${d}`;
        }
      } catch (_) {}
      return [name ? `Name: ${name}` : '', ...bits, style].filter(Boolean).join('\n');
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

  // ── Deterministic brain (FREE tier · zero-LLM · cannot drain a key) ───────────
  // Ported from Sven's Autonomous Pattern Engine — but warm companion register, not
  // topology reports. Handles greetings, get-to-know, feelings, small talk + fallback.
  // Genuine assistance (plan/find/organize/study…) latches `assist` → routes to the shuttle.
  _recent: [],
  _detIdx: {},

  _pick(key, arr) {
    const i = (this._detIdx[key] || 0) % arr.length;
    this._detIdx[key] = i + 1;
    return arr[i];
  },

  // Loop detection (ANTILOOP idea): same utterance twice → vary + nudge toward help.
  _isLoop(low) {
    const norm = low.replace(/[^a-z0-9 ]/g, '').trim();
    const looped = norm.length > 0 && this._recent.includes(norm);
    this._recent.push(norm);
    if (this._recent.length > 4) this._recent.shift();
    return looped;
  },

  // Pick a word to reflect back, so the user feels heard without an LLM.
  _salient(text) {
    const stop = new Set('the a an and or but to of in on for with i you me my your we it is am are was were be been do did so just really that this what how can could would about have has had not no yes ok okay here there at as if into your my our'.split(' '));
    const words = (text.toLowerCase().match(/[a-z][a-z'-]{2,}/g) || []).filter((w) => !stop.has(w));
    return words.length ? words.sort((a, b) => b.length - a.length)[0] : null;
  },

  // The deterministic responder. Warm, grounded, forward — never a metrics dump.
  autonomousReply(userText) {
    const t = (userText || '').trim();
    const low = t.toLowerCase();
    const name = (localStorage.getItem('lifeos.name') || '').trim();
    const looped = this._isLoop(low);
    const salient = this._salient(t);

    const isGreeting = t.length < 40 && /\b(hi|hey|hello|good (morning|afternoon|evening)|yo|hiya|howdy)\b/i.test(low);
    const isThanks   = /\b(thanks|thank you|appreciate|ty)\b/i.test(low);
    const isQuestion = /\?\s*$/.test(t) || /^(who|what|when|where|why|how|can|could|do|does|are|is|will|would)\b/i.test(low);
    const feeling    = (low.match(/\b(tired|exhausted|sad|down|happy|glad|excited|anxious|nervous|stressed|overwhelmed|lonely|scared|worried|angry|frustrated|calm|good|okay|fine|great|grateful|hopeful|bored)\b/) || [])[0];

    const offer = this._pick('offer', [
      "If you'd like, I can help you plan something, study a bit, or sort out your day — just say the word.",
      "Whenever you want, I can help you plan, find something, or work through a problem together.",
      "I'm happy to just talk — and the moment anything needs doing, say “help me…” and I'll jump in.",
    ]);

    if (looped) {
      return this._pick('loop', [
        "I hear you. Want me to actually help with that? Say “help me…” and I'll take it from here.",
        "Still with you. If you'd like me to do something about it, just ask — “help me plan,” “help me find…,” anything.",
      ]);
    }
    if (isGreeting) {
      return name
        ? this._pick('greetn', [`Hi ${name}. Good to hear you — what's on your mind?`, `Hey ${name}, I'm right here. Where do you want to start?`])
        : this._pick('greet', ["Hi, I'm right here. What's on your mind?", "Hey — good to hear you. What's going on today?"]);
    }
    if (isThanks) {
      return this._pick('thanks', ["Anytime. I'm right here whenever you need me.", "Of course — we can keep going whenever you like."]);
    }
    if (feeling) {
      return this._pick('feel', [
        `Thank you for telling me you're feeling ${feeling}. I'm here with you. ${offer}`,
        `${capitalize(feeling)} — I hear that, and I'm glad you said it. ${offer}`,
      ]);
    }
    if (isQuestion) {
      // Deterministic can't answer factual questions honestly — route to the real brain.
      return this._pick('q', [
        "Good question — let me actually help with that. Say “help me…” and I'll dig in properly.",
        "I want to give you a real answer, not a guess — ask me to help (“help me find…,” “help me figure out…”) and I'm on it.",
      ]);
    }
    if (salient) {
      return this._pick('reflect', [
        `${capitalize(salient)} — I'm listening. Tell me a little more, or ${offer}`,
        `I caught that, ${salient}. Say more if you'd like — ${offer}`,
      ]);
    }
    return this._pick('default', [`I'm here with you. ${offer}`, `Go on — I'm listening. ${offer}`]);
  },

  // ── Dispatcher: deterministic for get-to-know, shuttle (LLM) only for real help ──
  // This is the free/Pro line: no help intent = no key touched.
  async respond(userText) {
    const t = (userText || '').trim();
    if (!t) return "I'm here with you.";
    if (!this.assist && this._HELP_RE.test(t)) this.assist = true;     // intent-collapse latch

    if (!this.assist) {
      const reply = this.autonomousReply(t);                            // FREE: zero-LLM, no key
      this.history.push({ role: 'user', content: t }, { role: 'assistant', content: reply });
      if (this.history.length > 24) this.history.splice(0, this.history.length - 24);
      return reply;
    }
    try {
      return await this.replyRemote(t);                                 // PRO: gated LLM via shuttle
    } catch (_) {
      return this.autonomousReply(t);                                   // graceful: never dead-air
    }
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
