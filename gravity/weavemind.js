// weavemind.js — the empty vessel.
//
// There is no mind at rest. The vessel holds nothing until two connections meet
// on ONE device, derive the same point in the 10D handshake space, and a channel
// collapses into being there — transient, on-device, gone when the link drops.
//
// nerve.name was a 10D Hamiltonian field (the first thought of the night). ANTILOOP
// is its 6D sibling. The handshake doesn't EXCHANGE a blob — both ends COMPUTE the
// same coordinate from a shared gremlin + a public nonce. Same point ⇒ handshook.
// No server saw it. No database stored it. The vessel is recomputed, never kept.
//
// Security rests on the gremlin: a PRIVATE shared phrase makes the derived point a
// real pre-shared session key (the nonce can be public, e.g. in a QR). A PUBLIC
// gremlin (like "kekeke") makes a public rendezvous, not a private channel — same
// honesty as the bearer-token gate.

const DIM = 10;
const _crypto = (globalThis.crypto && globalThis.crypto.subtle) ? globalThis.crypto : null;

function enc(s) { return new TextEncoder().encode(s); }
async function sha256(bytes) { return new Uint8Array(await _crypto.subtle.digest('SHA-256', bytes)); }

// Deterministic coordinate in the 10D handshake space. Both devices, same inputs,
// same point. Hash → 10 floats in [-1, 1] via counter-expansion.
export async function derivePoint(gremlinHash, nonce) {
  const out = new Float64Array(DIM);
  let need = DIM, counter = 0, idx = 0;
  while (need > 0) {
    const h = await sha256(enc(`${gremlinHash}:${nonce}:${counter++}`));
    for (let b = 0; b + 4 <= h.length && need > 0; b += 4, need--) {
      const u = ((h[b] << 24) | (h[b + 1] << 16) | (h[b + 2] << 8) | h[b + 3]) >>> 0;
      out[idx++] = (u / 0xffffffff) * 2 - 1;
    }
  }
  return out;
}

export function distance(a, b) {
  let s = 0;
  for (let i = 0; i < DIM; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}

// The vessel. Empty until converge() meets two coincident points.
export class Weavemind {
  constructor(eps = 1e-9) { this.eps = eps; this._seed = null; }

  // True while nothing has collapsed here.
  get isEmpty() { return this._seed === null; }

  // Two connections meet on THIS device. If their points coincide, the vessel
  // fills with a transient channel seed (the hash of the shared coordinate) that
  // bootstraps the encrypted P2P link. If they don't, it stays empty.
  async converge(myPoint, theirPoint) {
    if (!myPoint || !theirPoint || distance(myPoint, theirPoint) > this.eps) {
      this._seed = null;
      return null;
    }
    this._seed = await sha256(new Uint8Array(new Float64Array(myPoint).buffer));
    return this._seed;
  }

  // The channel seed, only while the vessel is full.
  get seed() { return this._seed; }

  // Link drops → the vessel empties. Nothing persists.
  release() { this._seed = null; }
}
