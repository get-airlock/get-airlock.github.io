import assert from 'node:assert';
import { derivePoint, distance, Weavemind } from './weavemind.js';

const G = 'eb44e5c1ed8fef20';  // kekeke gremlin (truncated)

// 1. deterministic: same gremlin+nonce → same 10D point on any device
const a = await derivePoint(G, 'nonce-1');
const b = await derivePoint(G, 'nonce-1');
assert.strictEqual(a.length, 10);
assert.strictEqual(distance(a, b), 0, 'same inputs must give the same point');

// 2. different nonce → different point
const c = await derivePoint(G, 'nonce-2');
assert.ok(distance(a, c) > 0.01, 'different nonce must move the point');

// 3. coordinates are in [-1,1]
assert.ok([...a].every(x => x >= -1 && x <= 1), 'coords bounded to the field');

// 4. the vessel: empty until two coincident points meet on one device
const vessel = new Weavemind();
assert.strictEqual(vessel.isEmpty, true, 'vessel starts empty');
const seedNull = await vessel.converge(a, c);          // points differ → stays empty
assert.strictEqual(seedNull, null);
assert.strictEqual(vessel.isEmpty, true);

const seed = await vessel.converge(a, b);              // points coincide → collapses
assert.ok(seed && seed.length === 32, 'convergence yields a 32-byte channel seed');
assert.strictEqual(vessel.isEmpty, false, 'vessel is full at the meeting');

// 5. two devices that share gremlin+nonce derive the SAME seed independently
const v2 = new Weavemind();
const seed2 = await v2.converge(await derivePoint(G,'nonce-1'), await derivePoint(G,'nonce-1'));
assert.deepStrictEqual([...seed], [...seed2], 'both devices reach the same seed, no exchange');

// 6. release empties the vessel — nothing persists
vessel.release();
assert.strictEqual(vessel.isEmpty, true, 'link drops → vessel empty again');

console.log('weavemind: 6/6 passed — empty vessel, deterministic 10D point, on-device collapse');

// 7. Sven's refinement: same gremlin+nonce on DIFFERENT channels → different stars
const onAxioms = await derivePoint(G, 'nonce-1', 'otto::axioms');
const onMem    = await derivePoint(G, 'nonce-1', 'mem::noah');
assert.ok(distance(onAxioms, onMem) > 0.01, 'different channel must move the star');
const onAxioms2 = await derivePoint(G, 'nonce-1', 'otto::axioms');
assert.strictEqual(distance(onAxioms, onAxioms2), 0, 'same channel still deterministic');
console.log('weavemind: channel-binding holds (7th check) — otto::axioms ≠ mem::noah');
