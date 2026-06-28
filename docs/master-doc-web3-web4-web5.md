# Web3–Web5 Master Document

> **Status:** Master canonical source-of-truth. All downstream artifacts (NeurIPS paper, PRDs, investor decks, grants, editorial, landing copy) derive by SLICE + TRANSLATE from this document. Last updated: 2026-04-29.

## Purpose

This document is the dense master source for the Web3, Web4, and Web5 body of work. It is intended to be split downstream into product requirement documents, academic papers, investor deck narratives, onboarding docs, canon files, and implementation runbooks. The organizing rule is simple: preserve everything once in a high-density source of truth, then derive thinner artifacts by audience and use case. The current stack already contains the necessary ingredients for that approach: a replayable protocol specification, an established public-paper voice, pricing and identity tiering, canon framing, and a layered security/economics model.

A second purpose is architectural alignment. The operating environment is not a single app. Claude Desktop is positioned as the relay surface between VS Code and the CLI, with VS Code serving the backend and systems-development role while the CLI handles binary and task-heavy operations. That means the UI story should be authored as orchestration, not as a monolith, which fits the corrected Web5 framing as an orchestration layer rather than a replacement for all lower layers.

## Core thesis

The full stack resolves into three layers that should remain distinct in every derivative artifact. **Web3** is the ownership and identity layer, where DID, key custody, eSIM-backed assurance, and user-owned rails live. **Web4** is the hidden compute and security layer, where BCODE, PACT, closure verification, audit sidecars, side-channel discipline, and HiddenLayer-like functions reside. **Web5** is the orchestration layer, the operator-visible shell that coordinates identity, compute, and tooling into one coherent control surface.

This corrected layering matters because prior drafts drifted into treating Web5 as a sovereignty or subsumption claim. The canon now explicitly rejects that. Web5 is not the thing that swallows Web3 and Web4; it is the routing and presentation shell that makes them usable together. HiddenLayer belongs in Web4 as a compute-adjacent security slot, not as the shell. That correction should be applied consistently across papers, PRDs, investor material, and UI narratives.

## Audience splits

The same source material must support at least five downstream outputs, each with a different truth-density and rhetorical posture.

| Output | Primary audience | What it pulls from this master doc | Tone |
|---|---|---|---|
| NeurIPS / research paper | reviewers, researchers | protocol, methods, safety, side-channels, behavioral theory, limitations | empirical, citation-heavy |
| Product requirements docs | internal builders, contractors | UI flows, service boundaries, endpoints, acceptance criteria, tier ladders | precise, implementation-first |
| Investor deck | investors, strategic partners | thesis, market wedge, architecture moat, tier ladder, roadmap, economics | compressed, narrative, legible |
| Canon / vision docs | internal alignment, public narrative | Web3/Web4/Web5 framing, Plot of Time, Sovereign-Warrior guardrails | myth-aware but factual |
| Runbooks / onboarding | operators | service ports, relays, smoke tests, wiring, config, bootstrap steps | procedural |

The important rule is that every one of these outputs must inherit the same canonical layer map. If the paper calls Web5 orchestration but the deck calls it sovereignty, drift begins immediately. This master doc exists to stop that drift before it spreads.

## Architecture map

The architecture is best understood as a relayed workspace rather than a single app. Claude Desktop is the operator-facing conversation and relay surface. VS Code is the primary backend and implementation environment where services, configs, schemas, and application logic are authored. The CLI is the binary-operations and task-execution layer used for repo surgery, smoke tests, validation, process control, and other terminal-native work. This division should become explicit in the platform docs and deck architecture slides because it explains why orchestration is the correct word for Web5.

That orchestration shell sits over a substrate of services. The protocol specification already gives the right mental model: calls are wrapped in auditable structures, BCODE closures serialize behaviorally meaningful state, and sidecar artifacts permit replay and verification independent of the original execution environment. In product terms, the UI is not the system. The UI is the lens through which the operator interacts with a signed, replayable substrate.

### Workspace relay roles

| Layer | Main tool | Job | Canon mapping |
|---|---|---|---|
| Conversational relay | Claude Desktop | coordinate work across artifacts, UIs, and sessions | Web5 shell |
| Application/backend authoring | VS Code | services, routes, state service, business logic, configs | Web4 implementation surface |
| Binary/task execution | CLI | git repair, process start, curl, validation, file ops, smoke runs | Web4 operational surface |
| Identity + custody rails | Skeet / DID / keyring | user-owned access and assurance | Web3 |

This map is useful beyond engineering. In an investor deck, it becomes the product architecture slide. In a PRD, it becomes a systems-boundary section. In onboarding, it becomes the explanation for why different classes of work happen in different tools.

## Web3 layer

The Web3 layer is the ownership tier. It is where the user owns identity, where the keyring story lives, and where assurance can be raised without collapsing into a platform-custody model. The Skeet canon already supplies the right primitives: ACODE-free social-auth identity as the base layer, BCODE-paid or earned carrier-backed eSIM identity as the stronger layer, and future self-issued carrier surfaces as the enterprise direction.

This layer should not be oversold as abstract decentralization. In the current stack it has concrete jobs: bind an operator to a DID, attach a real auth surface, support portable key custody, and later bridge on-chain value such as $MOTION into off-chain internal credits. The bridge matters, but the deeper value is that identity and permissioning begin from user-owned rails rather than platform-issued accounts.

### Web3 components to preserve

- Sonr DID and DID-linked signing path.
- Skeet free/pro identity ladder and social-to-carrier assurance path.
- Keyring / Spacebar identity surface and phone-anchored burner identity canon.
- $MOTION as public incentive layer, distinct from internal credit accounting.
- Chainlink-related cross-chain and attestation ideas, currently deferred rather than removed.

For derivative artifacts, the investor deck should present Web3 as the ownership moat, the PRD should describe exact auth flows and assurance checks, and the paper should treat Web3 only where it affects protocol guarantees, identity provenance, or reproducibility.

## Web4 layer

Web4 is the hidden compute layer. This is where most of the substantive defensibility sits. It includes BCODE, PACT envelopes, closure operators, signed JSONL sidecars, replay verifiers, state services, side-channel controls, and the broader logic of verifiable execution. HiddenLayer belongs here conceptually because it occupies the same market slot: security and observability adjacent to the model-compute surface.

The protocol specification is especially important because it converts vague platform claims into inspectable machinery. For each scored call, the system computes a closure tuple from persona vector, model-derived vector, response-distributed fidelity vector, and rolling context vector, serializes it deterministically, and writes a sidecar record that can be replayed and verified later. This is the best current concrete artifact for showing how the substrate turns behavioral state into an auditable object.

### Web4 components to preserve

- PACT envelopes as protocol wrappers that carry zone, payload, and audit context.
- BCODE closure operator and signed sidecar output.
- Replay verification and acceptance-band logic.
- State service and cross-service wire validation.
- Side-channel discipline, especially rational-reconstruction protection via rounding.
- HiddenLayer mapping as external validation of the compute-security category, not as the shell itself.

### Side-channel and pricing discipline

The rounding-surplus artifact is a cornerstone because it is simultaneously a pricing primitive, a security primitive, and a ledger simplification rule. A true compute cost may be internally estimated at a fractional value like 0.0037, but the substrate rounds it to a clean external unit such as 0.01, charges the rounded amount, and stores the delta as operator-bound surplus. That surplus can later fund promotional credits without damaging margin, while also preventing raw high-precision telemetry from leaking internal ratios or bounded-integer structures that could be recovered by Stern–Brocot rational reconstruction.

This belongs in the master doc because it can be split three ways. The paper version treats it as side-channel hardening and audit hygiene. The PRD version treats it as pricing and ledger behavior. The deck version treats it as a unit-economics insight: promotional capacity funded structurally, not by venture subsidy.

## Web5 layer

Web5 is the orchestration shell. It is the layer that coordinates the identity tier, the hidden compute layer, and the operator tools into one usable product surface. The canon correction matters here: Web5 is not "more sovereign than Web3" and not a residency doctrine. It is the shell that composes lower layers into workflows an operator can actually run.

Spacebar is the clearest current candidate for the Web5 expression. The notes around the Spacebar keyring wireframe, magic-link plus QR onboarding, and Skeet handoff all indicate a shell whose primary job is orchestration: identity handoff, session start, key access, service routing, and operator-visible state. That is what the deck should show, what the PRD should scope, and what the canon should defend.

### Web5 responsibilities

- Provide the operator-visible shell across identity, compute, and memory surfaces.
- Relay work between Claude Desktop, VS Code, and CLI roles.
- Surface approvals, warm starts, and persona state without leaking raw hidden-layer internals.
- Coordinate onboarding, magic-link/QR handoff, and keyring access.

## Product surfaces

The stack currently implies several distinct but related product surfaces. These should be named and scoped in the master doc so they can each become their own PRD without re-litigating architecture.

| Surface | Role | Layer center |
|---|---|---|
| Spacebar | operator shell / orchestration UI | Web5 |
| Skeet | identity, auth, keyring, onboarding rail | Web3 + Web5 handoff |
| State service | persona/state coordination | Web4 |
| ConstellationBench | research and proof substrate | Web4 |
| Airlock coordination backend | wiring, relay, config, service topology | Web4 |
| Claude Desktop relay pattern | conversational command surface | Web5 |

This framing helps prevent confusion when talking to different audiences. The deck can present them as stack layers and product modules. The paper can mention only the ones needed to ground empirical claims. The PRDs can then branch: Skeet PRD, Spacebar PRD, state service PRD, and research substrate PRD.

## Protocol and proof assets

The protocol spec is the most mature technical anchor for the master doc because it already contains objective, countable, auditable elements: model roster, benchmark tasks, cost estimates, JSONL sidecars, replay verifier, and explicit stripped claims that must not be reintroduced without evidence. It should become the "proof spine" for any research-grade derivative artifact.

The public white paper contributes a different value: it shows the long-form voice, the broader biological and behavioral thesis, and the deck-friendly conceptual framing around topological intelligence, architecture splits, and failure transparency. But it also contains claims that need discipline in academic contexts. So the master doc should preserve its structure and narrative strengths while tagging which parts are public-story, which parts are measured, and which parts are conjectural.

### Evidence classes

| Class | Meaning | Example use |
|---|---|---|
| Measured | backed by code/data artifact | benchmark counts, sidecar verifier, cost totals |
| Implemented | built in code or system notes, not fully validated | state service wiring, config centralization, service ports |
| Canonical | internal alignment truth | Web5 orchestration correction, HiddenLayer mapping, Plot of Time frame locks |
| Conjectural | framing or future thesis | full topological-intelligence implications, future commitment primitives |

This separation is essential if the document is going to serve paper, deck, and product work simultaneously. Investors tolerate conjecture if it is clearly marked as thesis. Reviewers do not. PRDs need actionable implementation truth, not philosophical spread.

## Narrative spine for decks

The strongest investor narrative is not "AI but decentralized." It is that current AI products optimize capability, cost, and speed while leaving behavioral fidelity, operator control, auditability, and trust surfaces under-specified. The stack answers that gap with a layered system: user-owned identity rails, a replayable hidden compute substrate, and an orchestration shell that lets operators actually govern the system they are using.

A strong deck can therefore be built out of seven repeated moves:

1. Name the market failure: today's AI is high-capability but low-control and weakly auditable.
2. Show the scientific wedge: ConstellationBench turns behavioral claims into measurable protocol artifacts.
3. Show the security wedge: BCODE + PACT + sidecars + side-channel discipline make the substrate inspectable.
4. Show the product wedge: Skeet + Spacebar make the system usable at the identity and operator layers.
5. Show the pricing wedge: Constellation Credits and rounding-surplus produce legible economics.
6. Show the architecture wedge: Claude Desktop relay, VS Code backend, CLI binary tasks explain why orchestration is the right UI frame.
7. Show the roadmap wedge: from free social-auth access to stronger identity, richer orchestration, and enterprise coordination.

## PRD split map

The master doc should split into PRDs along system boundaries, not along brainstorm clusters. At minimum, the downstream document set should include the following.

| PRD | Scope |
|---|---|
| Spacebar PRD | orchestration shell, keyring UI, warm start, approval flows, operator dashboard |
| Skeet PRD | auth ladder, social DM bot, carrier eSIM tier, magic-link + QR onboarding |
| State Service PRD | persona profiles, warm-start endpoints, health checks, config centralization |
| BCODE/PACT PRD | closure generation, signing, verification, zone logic, audit sidecars |
| Credits & Billing PRD | CC ledger, $MOTION bridge, rounding-surplus dial, promotions |
| Claude Relay PRD | Claude Desktop ↔ VS Code ↔ CLI handoff model, task boundaries, session conventions |
| Canon Governance PRD | source-of-truth files, layer corrections, evidence classes, update workflow |

Each of these can be derived from this master doc without rethinking the whole platform. That is the main leverage of writing the dense version first.

## Paper split map

The academic paper should pull only the segments that survive empirical scrutiny. The likely section map is already visible in the current work: introduction and claim boundary from the public white paper, methods and protocol from the ConstellationBench spec, safety and side-channel analysis from the rounding and layered-trust notes, and discussion sections that frame Web4/Web5 carefully as systems categories rather than marketing slogans.

A safe paper split would look like this:

- Problem framing: behavioral fidelity and auditability gaps in current AI evaluation.
- Methods: DECF, benchmark suite, closure operator, sidecar artifacts, replay verification.
- Findings: only measured benchmark claims with pinned artifacts.
- Safety: side-channel discipline, layered trust, audit posture.
- Systems interpretation: Web4 hidden compute and Web5 orchestration as conceptual layers, stated cautiously.
- Limitations: roster drift, non-deterministic APIs, conjectural parts removed or marked.

## Canon and tone governance

The Plot of Time and frame-lock documents matter because they prevent the project from drifting into either paranoia or generic startup language. They establish a factual, citation-heavy, non-naive posture that recognizes the surrounding cyberpunk world without centering it. They also encode the counter-atom rule: every proven dark-side historical pattern should generate a corresponding protective or user-first design response in the substrate.

This is especially useful for decks and public writing. Instead of speaking in vague anti-establishment slogans, the material can say: there are historically documented patterns of surveillance, extraction, and tool capture; this stack responds with user-owned identity, replayable audits, and orchestration that keeps the operator in control. That tone is stronger, more legible, and more defensible than broad ideology.

## Implementation and operations backlog

Several implementation items already sit close enough to spec to include in the master doc as operational near-term work. These include centralizing state-service config in `config.py`, patching `server.py` and `validate-wires.py`, syncing ports, locking `ONBOARDING_MAX_TURNS=10`, starting the state service on port 8100, exercising `/health`, `/persona/profiles`, and `/persona/warm-start`, and confirming all four wires green. These are not abstract roadmap points; they are the direct near-term actions that stabilize the Web4 substrate beneath the future Web5 shell.

Likewise, the git-recovery work, backup-remote establishment, and file-format mismatch investigation belong in an operational appendix of the master doc. They are not deck material, but they matter for internal execution and source integrity. The `.gitignore` repairs, repo cleanup, and `reports/*.yaml` versus `reports/*.json` mismatch are examples of issues that can quietly undermine confidence if they are not tracked centrally.

## Immediate next artifacts

This master document should immediately feed four first-generation derivatives.

1. A **master outline deck** with one slide per section in the narrative spine, written from this document rather than from scratch.
2. A **Spacebar PRD** focused on the orchestration shell, keyring, warm-start, and operator approval flows.
3. A **Skeet onboarding PRD** covering social-auth, carrier-backed tiers, magic link, and QR handoff.
4. A **paper source outline** that maps measured claims only, with conjectural material marked for exclusion or future work.

## Editorial rule

The standing editorial rule for every derivative should be: preserve density in the source, reduce density only in the derivative. Do not create separate truths for deck, product, and paper. Write one dense canonical object, then cut it cleanly by audience. The current research base is already rich enough to support that workflow, and the corrected Web3/Web4/Web5 framing provides the stable backbone needed to do it without losing important work.
