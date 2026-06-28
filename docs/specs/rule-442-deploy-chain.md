# Rule 442 Deploy Chain — Corrected Canon

**Date:** 2026-04-30
**Status:** Canonical. Supersedes any prior mapping that placed Railway upstream of Fly or that described a "direct-to-Railway" deploy path.
**Author:** Zachary Holwerda

---

## The correction in one sentence

The substrate deploys **skills library → Fly → Railway → public surface**, not skills library → Railway. Fly is the *processor* that acts on raw materials and emits signed compounds; Railway is the *emission collection and control system* that consumes Fly's emissions and routes them to production at the required efficiency.

Anything that bypasses Fly and writes directly to Railway is a substrate-discipline violation by construction — the architectural analog of emitting VOCs without 85% capture-and-control under Rule 442.

---

## Mapping table

| Rule 442 element | Substrate component | Role |
|---|---|---|
| **VOC-containing material** (the source) | Otto skills library at `/Volumes/OttoVault/repos/airlock-skills-library/` | Raw materials: `otter-sound-v2.html`, `otto-identity-exploration.html`, `otto-command-center.html`, etc. |
| **Equipment that uses solvents / acts on materials** | Fly.io — `otto-cockpit.fly.dev` | The processor. Runs the Untangler closure-band gate. Emits substrate-signed compounds (the VCs). |
| **Emission collection and control system** (90% capture × 95% destruction = **85% overall efficiency**) | Railway | Consumes Fly's emissions. Routes them controlled. Moves apps to production at the required efficiency. |
| **Recordkeeping (Rule 109)** | BCODE audit ledger | Every Fly → Railway handoff signs a closure. No unsigned propagation. |
| **Storage and disposal — closed containers** | Provenance Filter | No leak path. Failed compounds never reach atmosphere. |
| **Permanent Total Enclosure (PTE)** | The full substrate boundary | `otto-cockpit.fly.dev` → Railway → public surface. No auth bypass. |
| **Aerosol products exemption** | Public-only static (`airlocklabs.io`) | No auth gate. No controlled emission required. Outside the regulated flow. |

---

## Operator path

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Otto skills library                                              │
│    /Volumes/OttoVault/repos/airlock-skills-library/                 │
│    Source material — the substrate's VOC-containing HTMLs           │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. Fly.io  /  otto-cockpit.fly.dev                                  │
│    Acts on the materials                                            │
│    Runs the Untangler closure-band gate                             │
│    Emits substrate-signed VCs (the compounds)                       │
│    422 gate must clear before any emission propagates               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. Railway                                                          │
│    Consumes Fly's emissions                                         │
│    85% capture + control efficiency requirement                     │
│    Moves apps to production                                         │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. Public Railway URLs                                              │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. Threema channel feeds reference Railway URLs                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. Brain Brigade operators                                          │
│    Subscribed to Threema → pick up the signal                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Why this is load-bearing (not just deploy plumbing)

The substrate's discipline — closure-band stability, provenance filtering, untangler rejection of anomalous closures — is **embedded in the deploy pipeline itself.** Nothing reaches Railway production until it has been Fly-processed (run through the Untangler at `otto-cockpit.fly.dev`) and signed. There is no architectural path for un-gated material to reach operators.

Three direct consequences:

1. **A bypass Railway deploy that skips Fly is a substrate-discipline violation.** It is the architectural analog of emitting VOCs without 85% capture-and-control under Rule 442. The discipline is not optional and not enforced by policy — it is enforced by the topology of the chain.

2. **The 422 gate at `otto-cockpit.fly.dev` is the structural firewall between source materials and production emissions.** A 422 from Fly means: "this material did not pass the closure-band check; no compound is emitted; nothing propagates downstream." Railway never sees rejected material because rejected material never leaves Fly.

3. **Brain Brigade operators receive only Fly-signed, Railway-served content — by architectural construction.** Operator trust in the Threema feed is not based on social or operational discipline; it is based on the fact that the chain has no upstream entry point that bypasses the gate.

---

## What this rules out

The following are all **prohibited** under the canonical chain — not because policy says so, but because each represents an unenclosed emission path:

- ❌ Direct `railway up` from a local working tree without a Fly-signed closure
- ❌ Manually pasting an HTML file from the skills library into a Railway service
- ❌ Pulling a build artifact from a non-Fly source (CI runner output, scratch container) into Railway
- ❌ Allowing `otto-cockpit.fly.dev` to forward un-gated requests through to Railway (the gate must be live, not bypassed in dev mode)
- ❌ Operating a "shadow Railway service" that accepts pushes from any source other than Fly

If any of these is observed in practice, treat it the way a compliance audit would treat a venting solvent tank: **stop production, identify the bypass, restore enclosure, log the closure for the recovery action.**

---

## Aerosol exemption (the one legitimate sidecar)

`airlocklabs.io` is public-only static. It runs **outside** the regulated chain by design — no auth gate, no operator-targeted material, no controlled emission. This is the equivalent of Rule 442's aerosol-products exemption: a category of output that does not require the capture-and-control infrastructure because it is structurally incapable of carrying gated material.

The exemption is narrow. It applies only to:
- Marketing surface
- Open-access documentation
- Public-readable artifact mirrors (where the content has already cleared the gate as part of a Fly → Railway flow and is being re-served statically)

It does **not** apply to anything operator-targeted, anything carrying credentials or session state, or anything that would meaningfully change behavior for a Brain Brigade operator. Those flows route through Fly → Railway, always.

---

## Provenance check (one-line audit)

For any URL an operator receives, the canonical question is:

> **Did this content's closure pass the 422 gate at `otto-cockpit.fly.dev` and arrive at this URL via Railway?**

If yes → trusted by construction.
If no, and it is on `airlocklabs.io` → exempt by aerosol rule, validate manually if it carries any operator-relevant payload.
If no, and it is on a Railway URL → **bypass detected.** Treat as untrusted. Log a rejection closure. Investigate the upstream gap.

---

## Reference cross-links

- **Untangler gate spec:** closure-band stability, $\|C\|_F \le d$ acceptance rule (Architecture Standards Manual)
- **Provenance filter:** six-rule policy engine governing ACODE → BCODE crossings (Architecture Standards Manual)
- **BCODE operator:** $C = P \wedge L \oplus R \oplus W$ — the closure that gets signed at the Fly → Railway handoff
- **PACT envelope:** the typed operator $\Pi: X \to X \times Y \times Z$ that carries the request through the pipeline
- **V3 Governance:** Truth Config + Evidence Pack + DID + Runtime Gate — the four requirements that make the chain auditable end-to-end

---

**Bottom line:** Fly is the processor. Railway is the emission control system. The skills library is the source. The chain only runs in that order. A direct-to-Railway deploy is not a shortcut — it is a venting tank.
