# Otto Translation-Layer Provenance Filter Spec

> **Derived from:** `docs/master-doc-web3-web4-web5.md` + PACT canon (`memory/project_pact_protocol_for_agents_context_trace_executionrouter_runtime.md`).
> **Web layer:** Web4 = filter / express (the substrate's filtering verb).
> **Status:** v1 specification, locked 2026-04-30 (operator-authored).
> **Reference implementation:** `pact/provenance.py` (Pydantic model + ProvenanceFilter policy engine).

## Purpose

This spec defines the provenance filter that sits at the ACODE to BCODE translation boundary for Otto and related substrate services. Its job is not to guess whether text "sounds AI" after the fact. Its job is to attach origin, trust, and retention semantics at the moment a request crosses from application-facing state into audited substrate state.

This filter exists to separate sovereign human signal, delegated agent work, and synthetic or derived material before the payload is allowed to enter PACT and BCODE flows.

## Canonical framing

- The filter lives at the **translation layer**, not the scrape layer.
- It operates on **provenance**, not style-detection.
- It is evaluated **before** BCODE closure emission.
- It writes a provenance block into the **PACT envelope**.
- BCODE receives the resulting typed envelope plus a provenance hash or pointer for audit continuity.

## Why this exists

Once human and synthetic text are mixed on the public web, post hoc filtering becomes unreliable and destructive. Aggressive filters drop valuable human material; weak filters allow synthetic contamination. The substrate avoids that failure mode by classifying requests at ingress, where the router knows the session origin, acting principal, and transformation path.

## Required PACT extension

Every PACT envelope that can lead to a BCODE closure must include a `provenance` block.

```python
from typing import Literal, Optional
from pydantic import BaseModel

OriginType = Literal["human", "agent", "synthetic", "mixed"]
TranslationMode = Literal["direct", "summarized", "transformed", "replayed"]
TrustClass = Literal["sovereign", "delegated", "unverified"]
RetainPolicy = Literal["ephemeral", "closure_only", "full_audit"]

class ProvenanceBlock(BaseModel):
    origin_type: OriginType
    origin_actor: str
    translation_mode: TranslationMode
    trust_class: TrustClass
    retain_policy: RetainPolicy
    source_ref: Optional[str] = None
    parent_provenance_id: Optional[str] = None
```

## Field semantics

### `origin_type`

- `human`: input comes directly from an authenticated human session, such as Spacebar, CLI, or a first-party app surface.
- `agent`: input is produced by an internal agent, workflow, judge, or delegated assistant.
- `synthetic`: input originates from known synthetic corpora or external AI output without an intervening authenticated human authorship step.
- `mixed`: input contains both human and synthetic lineage, or synthetic material materially edited by a human.

### `origin_actor`

- For `human`, this should be the operator DID or another authenticated principal ID.
- For `agent`, this should be the stable agent ID or runtime identity.
- For `synthetic` and `mixed`, this should identify the feed, uploader, ingestion job, or external source handle when available.

### `translation_mode`

- `direct`: payload passed through without semantic rewrite.
- `summarized`: payload compressed or summarized before entering BCODE.
- `transformed`: payload structurally changed, parsed, classified, normalized, or rewritten.
- `replayed`: previously logged or archived material re-entered a fresh session.

### `trust_class`

- `sovereign`: directly from the authenticated operator session.
- `delegated`: from a trusted agent or delegated workflow operating on behalf of the operator.
- `unverified`: from a source that has not been authenticated or whose authorship cannot be elevated.

### `retain_policy`

- `ephemeral`: may exist in volatile runtime memory only; raw payload cannot be durably written.
- `closure_only`: may contribute derived features or closure metadata, but raw payload is not retained in durable audit logs.
- `full_audit`: raw payload and closure metadata may both be retained for replay and inspection.

### `source_ref`

Optional pointer to upload ID, file hash, URL, job ID, queue item, or storage object key.

### `parent_provenance_id`

Optional lineage pointer used when one request is derived from another request or prior artifact.

## Policy engine

The runtime component is a `ProvenanceFilter` invoked by the Otto router before any PACT is accepted for execution.

```python
from dataclasses import dataclass
from typing import Literal

Decision = Literal["allow", "allow_with_downgrade", "deny"]

@dataclass
class ProvenanceDecision:
    decision: Decision
    reason: str
    normalized_retain_policy: str
    normalized_trust_class: str
```

```python
class ProvenanceFilter:
    def evaluate(self, provenance: ProvenanceBlock, zone: str, chamber: str) -> ProvenanceDecision:
        if provenance.origin_type == "human" and provenance.trust_class == "sovereign":
            return ProvenanceDecision("allow", "authenticated human request", provenance.retain_policy, provenance.trust_class)

        if provenance.origin_type == "agent":
            if provenance.trust_class == "delegated":
                policy = "closure_only" if provenance.retain_policy == "full_audit" else provenance.retain_policy
                return ProvenanceDecision("allow_with_downgrade", "agent requests default to closure-only unless whitelisted", policy, provenance.trust_class)
            return ProvenanceDecision("deny", "agent request missing delegated trust", "ephemeral", "unverified")

        if provenance.origin_type == "synthetic":
            if provenance.retain_policy == "full_audit":
                return ProvenanceDecision("allow_with_downgrade", "synthetic input cannot enter full-audit raw retention", "closure_only", "unverified")
            return ProvenanceDecision("allow_with_downgrade", "synthetic input restricted", provenance.retain_policy, "unverified")

        if provenance.origin_type == "mixed":
            if zone in {"red", "high_integrity"}:
                return ProvenanceDecision("allow_with_downgrade", "mixed input restricted in high-integrity zone", "closure_only", provenance.trust_class)
            return ProvenanceDecision("allow", "mixed input allowed in non-critical zone", provenance.retain_policy, provenance.trust_class)

        return ProvenanceDecision("deny", "unrecognized provenance combination", "ephemeral", "unverified")
```

## Required routing rules

### Rule 1: Sovereign human path

If `origin_type=human` and `trust_class=sovereign`, the request may proceed according to chamber policy. This is the only class eligible for default elevation to `full_audit` without additional justification.

### Rule 2: Delegated agent path

If `origin_type=agent`, the request must carry `trust_class=delegated` and a stable `origin_actor`. Agent-produced material does not inherit sovereign trust merely because it acts for a sovereign user. Unless explicitly whitelisted, raw payload retention is downgraded to `closure_only`.

### Rule 3: Synthetic path

If `origin_type=synthetic`, the request cannot be admitted as `full_audit` raw content. Synthetic material may be used for bounded derived operations, scoring, or feature extraction, but must not be silently promoted to equal standing with sovereign human data.

### Rule 4: Mixed path

If `origin_type=mixed`, the router must treat the request as an explicitly composite object. High-integrity zones should force `closure_only`; permissive zones may allow more latitude.

### Rule 5: Missing provenance is a protocol error

If a request would cross ACODE into BCODE without a provenance block, the router must reject it before model execution.

### Rule 6: Zone and chamber overrides

Each chamber or risk zone may further restrict allowed provenance combinations.

Examples:

- `Control` may forbid `synthetic` requests from triggering mutating operations.
- `Signal` may allow `mixed` inputs for summarization but not durable replay.
- `Orchestrate` may allow `agent` lineage for delegated workflows but still downgrade retention.

## Otto integration

This plugs directly into tasks 1.3 to 2.4 as follows:

### Task 1.3 — Intent Resolver

Intent resolution must output both an execution tier and a provenance expectation.

Example additions:

- `Signal`: expects `human/sovereign/direct` by default.
- `Orchestrate`: may accept `agent/delegated/transformed` for long-running work.
- `Control`: should require stricter trust semantics for state mutation and approvals.

### Task 1.4 — Execution Router

The execution router is responsible for:

1. Constructing or validating the `ProvenanceBlock`.
2. Running `ProvenanceFilter.evaluate(...)`.
3. Applying policy downgrades before PACT serialization.
4. Rejecting invalid requests before BCODE closure creation.
5. Storing a provenance hash or ID in the emitted BCODE audit record.

### Tasks 2.1 to 2.4 — Sub-agents / tier handoffs

Every sub-agent handoff must preserve or derive lineage from the parent provenance block.

Required invariants:

- Child agent calls must not erase parent provenance.
- Derived outputs should set `parent_provenance_id`.
- Trust can be downgraded during handoff, but not silently upgraded.
- Synthetic lineage remains synthetic or mixed unless a human-authenticated authorship step explicitly reclassifies it.

## Recommended PACT envelope shape

```python
class PactEnvelope(BaseModel):
    pact_id: str
    session_id: str
    zone: str
    chamber: str
    role: str
    token_budget: int
    payload: dict
    provenance: ProvenanceBlock
```

## BCODE audit requirement

BCODE should not need the raw payload to enforce provenance discipline. It only needs the normalized provenance block or its hash, plus the routing decision taken at ingress.

Minimum BCODE-side additions:

- `provenance_id` or `provenance_hash`
- `origin_type`
- `trust_class`
- `retain_policy`
- `router_decision`

This preserves replay and forensic capability without forcing all payload classes into durable raw storage.

## Default policy matrix

| origin_type | trust_class | default result | default retention |
| --- | --- | --- | --- |
| human | sovereign | allow | full_audit |
| agent | delegated | allow_with_downgrade | closure_only |
| synthetic | unverified | allow_with_downgrade | closure_only |
| mixed | delegated/unverified | allow_with_downgrade | closure_only |
| any | missing | deny | ephemeral |

## Engineering notes

- Do not implement this as a classifier over text style.
- Do not let agents silently inherit sovereign trust.
- Do not persist synthetic raw payloads into full-audit stores by default.
- Do not permit provenance-free PACT execution.
- Treat provenance normalization as part of the router contract, not a logging afterthought.

## One-line canon

> The filter between human signal and synthetic slop is not a detector over contaminated text; it is a provenance policy enforced at the ACODE to BCODE translation boundary, where the substrate knows who is acting, from what trust class, and under what retention rules.
