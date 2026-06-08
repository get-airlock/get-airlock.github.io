// LifeOS Voice — talk to it, it talks back.
// Web Speech API: webkitSpeechRecognition (listen) + speechSynthesis (speak).
// Works on iPad Safari (iOS 14.5+) and Chrome. Graceful typed fallback elsewhere.
// Per Rika spec: voice-first, calm, family/teen friendly. Built for June 1 demo.

const Voice = {
  rec: null,
  listening: false,
  speaking: false,
  level: 0,          // 0..1 animation energy
  _raf: null,
  _canvas: null,
  _ctx: null,
  cb: {},            // { onStatus, onTranscript, onReply }

  // Hands-free conversation: one tap to begin, then listen→reply→listen with no taps.
  conversationMode: false,
  _gotResult: false,
  _errStreak: 0,
  greeting: "Hi, I'm right here. What's on your mind?",

  // Smart endpointing — end the turn on COMPLETION + her rhythm, not raw silence.
  _utterance: '',        // accumulated final transcript for the current turn
  _endpointTimer: null,  // fires when she's genuinely finished a thought
  _lastResultAt: 0,
  _baseEndpoint: 1400,   // ms of grace before ending a turn; adapts to the speaker
  _pauseEMA: 0,          // learned average mid-thought pause length

  supportsSTT() {
    return !!(window.SpeechRecognition || window.webkitSpeechRecognition);
  },
  supportsTTS() {
    return 'speechSynthesis' in window;
  },

  init({ canvas, onStatus, onTranscript, onReply } = {}) {
    this.cb = { onStatus, onTranscript, onReply };
    if (canvas) {
      this._canvas = canvas;
      this._ctx = canvas.getContext('2d');
      this._sizeCanvas();
      window.addEventListener('resize', () => this._sizeCanvas());
      this._animate();
    }
    // Warm up voices (Safari loads them async)
    if (this.supportsTTS()) {
      window.speechSynthesis.getVoices();
      window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
    }
    // Restore her learned speaking rhythm so endpointing fits this person from the start.
    try {
      const p = parseInt(localStorage.getItem('lifeos.voice.pauseMs') || '0', 10);
      if (p > 400 && p < 4000) {
        this._pauseEMA = p;
        this._baseEndpoint = Math.max(1100, Math.min(2600, Math.round(p * 1.35)));
      }
    } catch (_) {}
  },

  // ── Hands-free conversation ────────────────────────────────
  // Called from the one "begin" tap (user gesture → unlocks mic + audio on iOS).
  startConversation() {
    this.conversationMode = true;
    this._errStreak = 0;
    this._clearEndpoint();
    this._utterance = '';
    this.speak(this.greeting);   // greeting ends → _afterSpeak → auto-listen
  },

  stopConversation() {
    this.conversationMode = false;
    this.stop();
    this._cancelSpeech();
    this._status('paused');
  },

  // After Carrie finishes speaking, re-open the ear automatically.
  _afterSpeak() {
    this.speaking = false;
    this._rampLevel(0);
    this._status('idle');
    if (this.conversationMode) {
      setTimeout(() => { if (this.conversationMode && !this.speaking && !this.listening) this.start(); }, 250);
    }
  },

  // ── Listening ──────────────────────────────────────────────
  toggle() {
    if (this.listening) { this.stop(); return; }
    this.start();
  },

  start() {
    if (this.speaking) this._cancelSpeech();
    if (!this.supportsSTT()) {
      this._status('Type to talk — voice input needs Safari or Chrome.');
      if (this.cb.onStatus) this.cb.onStatus('no-stt');
      return;
    }
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const rec = new SR();
    rec.lang = window.LIFEOS_LANG
      || (function () { try { return localStorage.getItem('lifeos.lang'); } catch (_) { return null; } })()
      || 'en-US';
    rec.interimResults = true;
    rec.continuous = true;       // keep the ear open across pauses; WE decide when she's done
    rec.maxAlternatives = 1;

    rec.onstart = () => {
      this.listening = true;
      this._gotResult = false;
      this._status('listening');
      this._rampLevel(0.7);
    };
    rec.onresult = (e) => {
      let interim = '', final = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript;
        if (e.results[i].isFinal) final += t; else interim += t;
      }
      // Learn her rhythm: a gap she paused through but recovered from is a pause
      // that should NEVER have ended her turn. Calibrate toward it.
      const now = Date.now();
      const gap = this._lastResultAt ? (now - this._lastResultAt) : 0;
      this._lastResultAt = now;
      if (gap > 500 && gap < 4000) {
        this._pauseEMA = this._pauseEMA ? (this._pauseEMA * 0.8 + gap * 0.2) : gap;
        this._baseEndpoint = Math.max(1100, Math.min(2600, Math.round(this._pauseEMA * 1.35)));
        try { localStorage.setItem('lifeos.voice.pauseMs', String(Math.round(this._pauseEMA))); } catch (_) {}
      }
      // Accumulate finals across pauses instead of firing a reply on each one.
      if (final) { this._utterance = (this._utterance + ' ' + final).trim(); this._errStreak = 0; }
      const live = (this._utterance + ' ' + interim).trim();
      if (live && this.cb.onTranscript) this.cb.onTranscript(live, false); // interim → UI; final fires on flush
      this.level = Math.min(1, 0.5 + live.length / 60);
      this._scheduleEndpoint(interim);   // any activity resets the clock; it only fires once she stops
    };
    rec.onerror = (e) => {
      this.listening = false;
      this._rampLevel(0);
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
        this.conversationMode = false;
        this._status('Tap to allow the microphone, then we can talk.');
      } else if (e.error === 'no-speech' || e.error === 'aborted') {
        // silence is fine in a conversation — onend will re-arm the ear
      } else {
        this._errStreak++;
      }
    };
    rec.onend = () => {
      this.listening = false;
      if (this.speaking || this._gotResult) return;     // a reply is coming; speak flow re-arms
      if (this.conversationMode && this._errStreak < 4) {
        // silence — keep listening, hands-free
        setTimeout(() => { if (this.conversationMode && !this.listening && !this.speaking) this.start(); }, 400);
      } else {
        if (this._errStreak >= 4) { this.conversationMode = false; this._status('paused'); }
        this._rampLevel(0); this._status(this.conversationMode ? 'idle' : 'paused');
      }
    };

    this.rec = rec;
    try { rec.start(); } catch (_) { /* already started */ }
  },

  stop() {
    this._clearEndpoint();
    this._utterance = '';
    this._lastResultAt = 0;
    if (this.rec) { try { this.rec.stop(); } catch (_) {} }
    this.listening = false;
    this._rampLevel(0);
    this._status('idle');
  },

  // ── Smart endpointing ──────────────────────────────────────
  // Fire the turn only after she's been quiet long enough AND the thought looks
  // complete. Mid-sentence pauses (and trailing "and…/because…") extend the wait.
  _scheduleEndpoint(interim) {
    this._clearEndpoint();
    const text = (this._utterance + ' ' + (interim || '')).trim();
    if (!text) return;
    const wait = this._looksComplete(text) ? this._baseEndpoint : (this._baseEndpoint + 800);
    this._endpointTimer = setTimeout(() => this._flush(), wait);
  },

  _looksComplete(text) {
    const words = text.toLowerCase().replace(/[^\w'\s]/g, ' ').split(/\s+/).filter(Boolean);
    if (words.length < 2) return false;             // too short — probably just getting started
    const CONT = new Set(['and','but','so','because','or','with','to','the','a','an','of','for','i',
      'we','my','our','um','uh','er','like','that','if','when','then','as','is','are','was','were',
      'at','in','on','it','this','these','those','you','your','its','about','into','over','than',
      'also','still','just','really','very','gonna','wanna','want','need','have','had','what','which']);
    return !CONT.has(words[words.length - 1]);       // trailing connector → she's mid-thought, wait
  },

  _flush() {
    this._clearEndpoint();
    const text = (this._utterance || '').trim();
    this._utterance = '';
    this._lastResultAt = 0;
    if (!text) return;
    this._gotResult = true; this._errStreak = 0;     // onend won't re-arm; a reply is coming
    if (this.cb.onTranscript) this.cb.onTranscript(text, true);  // now it's final for the UI
    if (this.rec) { try { this.rec.stop(); } catch (_) {} }      // close the ear so we don't hear Carrie
    this.listening = false;
    this._handleFinal(text);
  },

  _clearEndpoint() {
    if (this._endpointTimer) { clearTimeout(this._endpointTimer); this._endpointTimer = null; }
  },

  // Route a final utterance (voice OR typed) through the companion and speak back.
  submitText(text) {
    const t = (text || '').trim();
    if (!t) return;
    if (this.cb.onTranscript) this.cb.onTranscript(t, true);
    this._handleFinal(t);
  },

  async _handleFinal(text) {
    this._status('thinking');
    let reply;
    try {
      reply = await window.LifeOSCompanion.respond(text);       // deterministic (free) → shuttle (Pro, on help)
    } catch (_) {
      reply = (window.LifeOSCompanion && window.LifeOSCompanion.autonomousReply)
        ? window.LifeOSCompanion.autonomousReply(text)          // zero-LLM fallback
        : "I'm here with you.";
    }
    if (this.cb.onReply) this.cb.onReply(reply, text);
    this.speak(reply);
  },

  // ── Speaking ───────────────────────────────────────────────
  // Try the high-quality shuttle voice; fall back to the browser's built-in speech.
  async speak(text) {
    const base = window.LIFEOS_SHUTTLE;
    if (base) {
      try {
        this._cancelSpeech();
        const r = await fetch(`${base}/api/tts`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        });
        if (!r.ok) throw new Error('tts ' + r.status);
        const url = URL.createObjectURL(await r.blob());
        const audio = new Audio(url);
        this._audio = audio;
        this.speaking = true; this._status('speaking'); this._rampLevel(0.85);
        audio.onended = () => { URL.revokeObjectURL(url); this._afterSpeak(); };
        audio.onerror = () => { URL.revokeObjectURL(url); this._browserSpeak(text); };
        await audio.play();
        return;
      } catch (_) {
        this._browserSpeak(text);
        return;
      }
    }
    this._browserSpeak(text);
  },

  _browserSpeak(text) {
    if (!this.supportsTTS()) { this._status('idle'); return; }
    this._cancelSpeech();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 0.96;
    u.pitch = 1.04;
    u.volume = 1;
    const v = this._pickVoice();
    if (v) u.voice = v;
    u.onstart = () => { this.speaking = true; this._status('speaking'); this._rampLevel(0.85); };
    u.onend = () => { this._afterSpeak(); };
    u.onerror = () => { this._afterSpeak(); };
    window.speechSynthesis.speak(u);
  },

  _pickVoice() {
    const voices = window.speechSynthesis.getVoices() || [];
    if (!voices.length) return null;
    // Prefer a warm, natural English voice; Safari has "Samantha", Chrome has "Google US English".
    const prefer = ['Samantha', 'Google US English', 'Karen', 'Moira', 'Allison', 'Ava'];
    for (const name of prefer) {
      const hit = voices.find((v) => v.name.includes(name));
      if (hit) return hit;
    }
    return voices.find((v) => /en[-_]US/i.test(v.lang)) || voices.find((v) => /^en/i.test(v.lang)) || voices[0];
  },

  _cancelSpeech() {
    if (this._audio) { try { this._audio.pause(); } catch (_) {} this._audio = null; }
    if (this.supportsTTS()) { try { window.speechSynthesis.cancel(); } catch (_) {} }
    this.speaking = false;
  },

  _status(s) { if (this.cb.onStatus) this.cb.onStatus(s); },

  // ── Waveform animation ─────────────────────────────────────
  _sizeCanvas() {
    if (!this._canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = this._canvas.getBoundingClientRect();
    this._canvas.width = rect.width * dpr;
    this._canvas.height = rect.height * dpr;
    this._ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this._cssW = rect.width;
    this._cssH = rect.height;
  },

  _rampLevel(target) { this._levelTarget = target; },

  _animate() {
    let t = 0;
    const draw = () => {
      t += 0.018;
      // ease level toward target
      const target = this._levelTarget != null ? this._levelTarget : 0;
      this.level += (target - this.level) * 0.08;
      const ctx = this._ctx;
      if (!ctx) { this._raf = requestAnimationFrame(draw); return; }
      const w = this._cssW, h = this._cssH;
      ctx.clearRect(0, 0, w, h);

      const cx = w / 2, cy = h / 2;
      const R = Math.min(w, h) / 2 - 4;

      // outer ring
      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(110, 196, 214, 0.35)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // clip to circle so waves stay inside the ring
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, R - 2, 0, Math.PI * 2);
      ctx.clip();

      const base = 0.12 + this.level * 0.9;          // idle breathes gently
      const lines = [
        { amp: 26 * base, freq: 1.6, phase: t * 1.1, color: 'rgba(110, 196, 214, 0.9)', lw: 2.4 },
        { amp: 20 * base, freq: 2.3, phase: t * 1.6 + 1, color: 'rgba(120, 175, 220, 0.6)', lw: 2 },
        { amp: 14 * base, freq: 3.1, phase: t * 0.8 + 2, color: 'rgba(176, 156, 222, 0.5)', lw: 1.6 }
      ];
      lines.forEach((L) => {
        ctx.beginPath();
        for (let x = -R; x <= R; x += 3) {
          const norm = x / R;
          const envelope = Math.cos(norm * Math.PI / 2); // taper at edges
          const y = Math.sin(norm * Math.PI * L.freq + L.phase) * L.amp * envelope;
          const px = cx + x, py = cy + y;
          if (x === -R) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.strokeStyle = L.color;
        ctx.lineWidth = L.lw;
        ctx.lineJoin = 'round';
        ctx.stroke();
      });
      ctx.restore();

      this._raf = requestAnimationFrame(draw);
    };
    this._levelTarget = 0;
    this._raf = requestAnimationFrame(draw);
  }
};

if (typeof window !== 'undefined') {
  window.LifeOSVoice = Voice;
}
