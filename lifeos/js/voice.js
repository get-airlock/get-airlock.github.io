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
    rec.lang = 'en-US';
    rec.interimResults = true;
    rec.continuous = false;
    rec.maxAlternatives = 1;

    rec.onstart = () => {
      this.listening = true;
      this._status('listening');
      this._rampLevel(0.7);
    };
    rec.onresult = (e) => {
      let interim = '', final = '';
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript;
        if (e.results[i].isFinal) final += t; else interim += t;
      }
      const shown = (final || interim).trim();
      if (shown && this.cb.onTranscript) this.cb.onTranscript(shown, !!final);
      this.level = Math.min(1, 0.5 + shown.length / 60);
      if (final) this._handleFinal(final.trim());
    };
    rec.onerror = (e) => {
      this.listening = false;
      this._rampLevel(0);
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
        this._status('Microphone permission is needed to talk.');
      } else if (e.error === 'no-speech') {
        this._status('idle');
      } else {
        this._status('idle');
      }
    };
    rec.onend = () => {
      this.listening = false;
      if (!this.speaking) { this._rampLevel(0); this._status('idle'); }
    };

    this.rec = rec;
    try { rec.start(); } catch (_) { /* already started */ }
  },

  stop() {
    if (this.rec) { try { this.rec.stop(); } catch (_) {} }
    this.listening = false;
    this._rampLevel(0);
    this._status('idle');
  },

  // Route a final utterance (voice OR typed) through the companion and speak back.
  submitText(text) {
    const t = (text || '').trim();
    if (!t) return;
    if (this.cb.onTranscript) this.cb.onTranscript(t, true);
    this._handleFinal(t);
  },

  _handleFinal(text) {
    const reply = (window.LifeOSCompanion && window.LifeOSCompanion.reply)
      ? window.LifeOSCompanion.reply(text)
      : "I'm here with you.";
    if (this.cb.onReply) this.cb.onReply(reply, text);
    this.speak(reply);
  },

  // ── Speaking ───────────────────────────────────────────────
  speak(text) {
    if (!this.supportsTTS()) { this._status('idle'); return; }
    this._cancelSpeech();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 0.96;
    u.pitch = 1.04;
    u.volume = 1;
    const v = this._pickVoice();
    if (v) u.voice = v;
    u.onstart = () => { this.speaking = true; this._status('speaking'); this._rampLevel(0.85); };
    u.onend = () => { this.speaking = false; this._rampLevel(0); this._status('idle'); };
    u.onerror = () => { this.speaking = false; this._rampLevel(0); this._status('idle'); };
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
