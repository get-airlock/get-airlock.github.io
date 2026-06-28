/* LifeOS — nerve.name citizen client.
   Mints / resolves a sovereign identity via the shuttle (server-to-server to otto-cockpit).
   Identity mint is GASLESS + FREE — never token-gated. did:sonr:<name> is the one-life anchor.
   Stores the minted citizen locally (lifeos.nerve.v1) so onboarding is the only mint moment. */
(function () {
  const SHUTTLE = (typeof window !== 'undefined' && window.LIFEOS_SHUTTLE) || 'https://lifeos-shuttle.vercel.app';
  const KEY = 'lifeos.nerve.v1';

  function load() {
    try { return JSON.parse(localStorage.getItem(KEY) || 'null'); } catch (e) { return null; }
  }
  function save(citizen) {
    try { localStorage.setItem(KEY, JSON.stringify(citizen)); } catch (e) {}
    return citizen;
  }

  const LifeOSNerve = {
    // The currently-minted citizen for this device, or null.
    current() { return load(); },
    did() { const c = load(); return c && (c.id ? `did:sonr:${c.id}` : c.did) || null; },

    // Resolve a name. Returns { registered, citizen } or { registered:false }.
    async resolve(name) {
      const r = await fetch(`${SHUTTLE}/api/nerve?name=${encodeURIComponent(String(name || '').toLowerCase())}`);
      if (r.status === 404) return { registered: false };
      if (!r.ok) throw new Error(`resolve ${r.status}`);
      return r.json();
    },

    // Mint a citizen (the onboarding gate). Returns the full register result.
    // { status, citizen, on_chain, uri }. on_chain:false today = local-only (Sonr node down) — fine for beta.
    async register(name, opts = {}) {
      const payload = { name: String(name || '').toLowerCase() };
      if (opts.wallet) payload.wallet = opts.wallet;
      if (opts.pod_url) payload.pod_url = opts.pod_url;
      const r = await fetch(`${SHUTTLE}/api/nerve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await r.json().catch(() => ({}));
      if (r.status === 409) { const e = new Error('taken'); e.taken = true; throw e; }
      if (!r.ok) throw new Error(data.error || `register ${r.status}`);
      if (data.citizen) save(data.citizen);
      return data;
    },
  };

  if (typeof window !== 'undefined') window.LifeOSNerve = LifeOSNerve;
  if (typeof module !== 'undefined' && module.exports) module.exports = LifeOSNerve;
})();
