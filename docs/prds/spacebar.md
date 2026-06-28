# Spacebar PRD — Orchestration Shell

> **Derived from:** `docs/master-doc-web3-web4-web5.md` (sections: Architecture map, Web5 layer, Product surfaces).
> **Status:** v0 specification, drafted 2026-04-29.
> **Scope:** Web5 orchestration UI for the substrate — operator-facing shell only. Does not own substrate logic.

## 0. Summary

Spacebar is the operator shell for the substrate: a Web5 orchestration UI that sits on top of Web3 identity (Skeet, DID, $MOTION) and Web4 hidden compute (BCODE/PACT, state service, credits). Its job is to give operators a single place to see sessions, approve risky actions, manage keyrings, launch warm-starts, and inspect audit trails — without ever bypassing the protocol.

Spacebar does not call models directly. It only talks to substrate services (state service, PACT gateway, credits service, Skeet identity) and renders their state. It is the visible face of Web5 orchestration.

Source: drafted in-chat, archived 2026-04-29. Path of record will be `docs/prds/spacebar.md` once the substrate repo is in place.

## 1. Goals & non-goals

### 1.1 Goals

Provide a single operator surface for:

- Session list and status across zones and personas.
- Keyring and identity overview (Skeet tiers, DID, $MOTION/credit balance).
- Warm-start launch for personas via state service.
- Approval of PACTs that change risk, tier, or dial state.
- Inspection and replay verification of BCODE closures.

Implement Web5 as orchestration, not compute:

- All compute flows through PACT → BCODE → services; Spacebar only orchestrates.

Be the operator-grade UI that investors and users experience first:

- Representative of the substrate's glass-box thesis: legible budgets, legible audit, legible permissions.

### 1.2 Non-goals

- No direct model API calls (OpenRouter, vendor APIs) from Spacebar.
- No custom prompt-engineering UI in v0; Spacebar launches pre-defined benchmark and workflow flows, not arbitrary chat.
- No inline editing of Skeet identity details beyond launch and status; deeper identity management lives in Skeet flows.

## 2. Personas & jobs-to-be-done

### 2.1 Solo operator / researcher

Profile: technical user (VS Code + CLI fluent), running ConstellationBench, exploring routing, validating behavioral results.

Jobs:

- See which runs are active, which zones are in use (green/yellow/red).
- Launch new persona/model runs with a warm-start.
- Inspect closures and sidecars for a given run.
- Verify that credit consumption matches expectations.

### 2.2 Small team / integrator

Profile: early customer or partner integrating substrate into their own tools.

Jobs:

- Monitor aggregate credit usage and tier status for their org.
- Approve or reject high-risk PACTs (e.g., red-zone calls, dial changes).
- Export or view audit trails and verifier results.

## 3. Core concepts

### 3.1 Session

A session is a named sequence of PACT calls sharing:

- operator (DID)
- persona profile
- zone (green/yellow/red)
- model roster / workflow type (e.g., "full ConstellationBench", "Spacebar warm-start flow")
- credit budget and current spend

Spacebar must display:

- session name, status (`idle`, `running`, `paused`, `completed`, `error`)
- persona name and DECF shorthand
- zone (color + label)
- model family or workflow label
- credits used / remaining, with warning thresholds

### 3.2 Keyring

The keyring is the operator's consolidated identity view:

- Skeet identity (Free / Pro / Enterprise tier)
- DID(s) in use, including Sonr DID and any other supported schemes
- Linked accounts (Discord/Telegram social for Free, carrier line for Pro)
- $MOTION and Constellation Credits summary (balances and recent activity)

Spacebar shows the keyring read-only in v0; management flows are launched into Skeet or external wallets.

### 3.3 PACT & closures

Every substrate call Spacebar initiates or approves is a PACT:

- envelope header: persona, zone, budget ceiling, tier, routing hints
- payload: benchmark/workflow parameters or operator action

Each PACT produces a BCODE closure stored in sidecar files and/or a closure service:

- Spacebar must retrieve and display key closure metadata (timestamp, persona id, model id, norm, status).
- Spacebar must expose a "verify" action that triggers the BCODE verifier and surfaces pass/fail.

## 4. Features & requirements

### 4.1 Dashboard: sessions & status

Feature: A dashboard view listing active and recent sessions.

Requirements:

- **R4.1.1**: Show table with columns: Name, Persona, Zone, Workflow, Status, Credits used / limit, Last activity.
- **R4.1.2**: Support filtering by zone, persona, status.
- **R4.1.3**: Clicking a session opens a detail pane with:
  - session metadata (ids, zone, tier, created/updated timestamps)
  - list of recent PACTs and closures (see 4.4)
- **R4.1.4**: Sessions should update in near-real-time via polling or websocket (implementation detail) with at least a manual refresh.

### 4.2 Keyring view

Feature: A "Keyring" view summarizing identity and credit.

Requirements:

- **R4.2.1**: Display Skeet tier (Free/Pro/Enterprise) and linked auth surface(s) (Discord/Telegram/eSIM).
- **R4.2.2**: Display primary DID(s) and verification status (e.g., "Sonr: verified").
- **R4.2.3**: Display balances:
  - Constellation Credits (CC) balance and last 5 debits/credits.
  - $MOTION (if applicable) as pulled from wallet or bridge service.
- **R4.2.4**: Provide links/actions:
  - "Manage identity in Skeet" → open Skeet onboarding/management flow.
  - "Top up credits" → open appropriate billing or $MOTION-bridge UI.

### 4.3 Warm-start launcher

Feature: Launch persona-specific warm-start sequences for sessions.

Requirements:

- **R4.3.1**: Query state service `/persona/profiles` to list available persona profiles with name, DECF, and description.
- **R4.3.2**: Provide a "New session" wizard:
  - Step 1: Choose persona.
  - Step 2: Choose workflow (e.g., "Smoke test", "Full ConstellationBench", "Custom workflow").
  - Step 3: Set zone (default green) and credit budget ceiling.
- **R4.3.3**: On submit, Spacebar must:
  - construct a PACT warm-start envelope with persona, zone, budget, workflow id
  - POST to a PACT gateway endpoint
  - create a new session record and display status
- **R4.3.4**: For warm-start flows that require initial persona context, call `/persona/warm-start` and surface status.

### 4.4 PACT & closure inspection

Feature: Per-session view of PACTs and closures with verify action.

Requirements:

- **R4.4.1**: For a selected session, show a chronological list of PACT calls with:
  - timestamp
  - zone
  - model / service identifier
  - status (`pending`, `completed`, `rejected`)
  - credit delta for the call
- **R4.4.2**: Selecting a PACT should show:
  - header fields (persona, zone, tier, budget before/after)
  - payload summary (workflow type, parameters)
  - linked closure id
- **R4.4.3**: Selecting a closure should show:
  - closure JSON (pretty-printed)
  - Frobenius norm, status (`accepted`/`rejected`)
  - signature verification status
- **R4.4.4**: Provide a "Verify closure" button that:
  - calls the BCODE verifier (local CLI or service) on the closure/sidecar entry
  - displays result `VERIFIED` vs `MISMATCH` / `BAD_SIG`

### 4.5 Approval console

Feature: Operators must be able to approve or reject PACTs that change risk posture, tier, or pricing dials.

Requirements:

- **R4.5.1**: Expose a "Pending approvals" panel listing:
  - requested PACT type (e.g., `zone_change`, `tier_upgrade`, `dial_adjust`)
  - requested new values (e.g., yellow→red zone, Free→Pro, rounding dial change)
  - rationale / originator when available
- **R4.5.2**: For each item, provide Approve and Reject actions that:
  - construct and send an approval PACT envelope
  - log result as a closure (even for rejection)
- **R4.5.3**: Tier upgrades and dial changes must never be executed without an approval closure signed under an Admin DID.
- **R4.5.4**: The UI must clearly explain the economic impact of dial changes (e.g., "round to 0.10 increases promotional capacity at the cost of higher per-call charges").

### 4.6 Credits & budget visibility

Feature: Clear credit consumption, budget, and dial state.

Requirements:

- **R4.6.1**: Per session, show a budget meter:
  - initial budget, used credits, remaining credits
  - predicted calls remaining for the current workflow (if possible)
- **R4.6.2**: Globally, show:
  - operator's credit balance
  - operator's current humid dial state (`drier` / `default` / `wetter`)
- **R4.6.3**: Warn the operator when:
  - a session is projected to exceed budget
  - nightly or global credit caps are nearing

## 5. APIs & integration contracts

### 5.1 State service

Endpoints used:

- `GET /health` → health indicator
- `GET /persona/profiles` → list of persona profiles
- `POST /persona/warm-start` → initialize warm-start for a persona

Requirements:

- Spacebar must handle 5xx and degraded health gracefully and surface status.

### 5.2 PACT gateway

Endpoints used:

- `POST /pact/execute` → execute a PACT envelope, returning PACT status + closure reference
- `GET /pact/session/:id` → session details and PACT history
- `GET /closure/:id` → closure JSON and metadata

Contracts:

- Spacebar sends typed PACT envelopes; gateway handles execution and logging.
- Gateway must not allow direct model calls without PACT; Spacebar must not attempt them.

### 5.3 Credits service

Endpoints used:

- `GET /credits/balance` → operator CC balance
- `GET /credits/history` → recent debits/credits
- `GET /credits/dial` → current rounding dial position

### 5.4 Skeet identity service

Endpoints used:

- `GET /identity/profile` → Skeet tier, linked accounts, assurance level

Spacebar is a client of all these services; it does not own any of the core substrate logic.

## 6. UX & constraints

### 6.1 UX principles

- Show zones and risk explicitly (color + label) whenever a PACT is initiated.
- Favor read-only clarity over in-place mutation: changes to identity, dialing, or routing are initiated from Spacebar but executed and recorded by substrate services.
- Keep the UI understandable to a sophisticated operator but legible enough for an investor demo.

### 6.2 Hard constraints

All destructive or risk-changing actions must result in:

- a PACT envelope
- a BCODE closure
- an approval closure if required by policy

Spacebar must not:

- write directly to model providers
- mutate ledger or dial state without going through PACT/BCODE

## 7. Open questions / future work

- Should Spacebar eventually host a lightweight prompt console for debugging, or should that live in a separate "lab" tool?
- Where to surface NSI / non-separability metrics once they are defined — per session or per model?
- How tightly to integrate $MOTION wallets in v1 (if at all), versus keeping that in an external wallet/bridge UI?
