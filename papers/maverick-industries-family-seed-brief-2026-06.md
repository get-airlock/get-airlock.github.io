# Maverick Industries — Family Seed Brief

### One platform. Six layers built, the seventh shipping this week. And as of this month — it runs in your browser.

*Zachary Holwerda · Detroit, MI · June 2026*

---

> **CHANGE NOTE — for Zac, delete before sending.**
> Updated from the April investor narrative (`maverick-industries-investor-narrative.md`).
> **What I changed:** date April→June; swapped the old satellite-mission vocabulary (Trinity/GEO/Cislunar, Spacebar, OctoConductor) for the current **6+1 layer ladder** (Scout→Surface + Day-7 DAG/Spark) with pricing tiers; updated the team to the real **5-hand federation** (Ande, Sven, Sandor, Phil + you/Otto); led with **"it's live now"** (LifeOS free tier, nerve.airlocklabs.io, the Ferry) because that's the actual change since April and it's your bridge thesis; added the **$0/mo burn** line because frugality is the right signal for family money.
> **What you MUST verify before this goes to your dad (integrity — can't walk back with family):**
> 1. The **RLHF Paradox numbers** (~20% behavioral edge, 97% cost cut, "holds across entire roster"). My memory flags Bench-1.5 came back **null (p=0.64)** in the NeurIPS sprint — so "holds across entire roster" may overstate. Confirm against the *current* paper draft and soften if needed. I left the claims in but marked them `[VERIFY]`.
> 2. The **live URLs** — I confirmed `airlocklabs.io/lifeos/` (200) and `nerve.airlocklabs.io` (200) tonight. Re-check `spacebar.html` and the paper links before sending.
> 3. Anything about the **escape/holdco plan** stays OUT of this doc unless your dad is inner-circle on it.

---

## TL;DR (for the room — and for Dad)

In April I asked you to read what we'd shipped. In June I'm asking you to **open a link and use it.**

We measured the one AI metric the industry doesn't: **who can a model BE** — not what it can do, not how fast. **Who.** Across 22 frontier and open models, 17 behavioral profiles, 22,200+ live API calls, and **$115 of compute**, we found a structural result the field hasn't surfaced: lighter-alignment budget models outperform frontier models on behavioral fidelity. `[VERIFY]` We turned that into a routing architecture that delivers better behavioral scores at a fraction of frontier cost. `[VERIFY]`

The benchmark is shipped and public. The paper is at NeurIPS submission stage. And the consumer surface — the thing a real person logs into — **is live and free as of this month.** The whole substrate runs at **$0/mo plus inference fuel**, because we replaced every paid SaaS vendor with software we built.

**Maverick Industries is the production company that ships what the substrate already proves.**

---

## What's different since April: it runs

In April this was a research result and a set of artifacts. The honest ask was "do your diligence on what shipped." Two months later the ask is simpler, because the thing exists:

| Surface | Where | Status (verified June 4) |
|---|---|---|
| **LifeOS** — the consumer onboarding, free tier | `airlocklabs.io/lifeos/` | **Live, $0.** Imprint → companion, runs deterministically on-device-grade logic, no cost to operate. |
| **Nerve** — the build realm / IDE | `nerve.airlocklabs.io` | **Live**, alpha invite-gated. |
| **The Ferry** — the federation message bridge | `otto-dag-bridge/ferry.js` | **Built and tested** (accept / refuse / defer). |
| **nerve.name** — true-name identity registry | owned; `nerve.db` live | 1 citizen + 1 agent registered. DID layer wiring this week. |

This is the trust model shift the whole company is built on: **from "trust me" to "curl it."** You don't have to believe in the idea. You can hit the URL.

---

## The product — seven layers, we're at six

We don't pitch as enterprise SaaS. We ship a **layered platform** where each layer is both a capability and a self-awareness question, and each layer is a pricing tier. Six are built. The seventh — the engine that ties them together — ships this week.

| # | Layer | The question it answers | What it is | Tier |
|---|---|---|---|---|
| 01 | **Scout** | "Who am I?" | Routing, dispatch, control plane | Trial / onboarding |
| 02 | **Gateway** | "How do others see this?" | API ingestion, bring-your-own-stack | API access |
| 03 | **Sentinel** | "Am I being real?" | Auth, trust, baseline integrity | Security / trust |
| 04 | **Pipeline** | "Can I say what I mean?" | Task flow, job queue | Workflow automation |
| 05 | **Toolkit** | "Can I build with others?" | Agents, tools, actuators | Agent marketplace |
| 06 | **Surface** | "Where do I end?" | UX, customization, white-label | Theming / white-label |
| 07 | **DAG Engine + Spark** | *"Can it run itself?"* | The orchestration engine that fires the whole chain | **Shipping this week** |

The pitch is a clock, not a promise: **day six of seven.** Day seven is the engine. That's the difference between "my son has an idea" and "this is a structured build that lands on a date."

---

## The empirical proof — what we actually measured

### The finding `[VERIFY against current paper]`

The highest-fidelity models on our persona benchmark were the *cheapest*, lightest-alignment ones — not the frontier flagships. The structural explanation: RLHF alignment training compresses outputs toward "helpful, harmless, honest," and in doing so clips the behavioral extremes where distinct personas live. The same training that makes a model safer makes it less able to *be someone*.

> **Integrity note (Zac):** the April draft stated this "holds across our entire 22-model roster" and quoted a ~20% edge / 97% cost reduction. Confirm these against the current NeurIPS numbers before sending — memory flags at least one bench (Bench 1.5) returned null. Better to under-claim to your dad and let the paper over-deliver.

### What it's worth

A tiered router that sends ~90% of behavioral traffic to budget models, ~8% to mid-tier, ~2% to frontier for the hard calls — producing a complete task lifecycle at a fraction of frontier-uniform cost. `[VERIFY]` The methodology, scoring code, signal dictionaries, and raw data are **all public on HuggingFace.** Diligence is the thing this company is designed to survive.

---

## The platform — BCODE

BCODE is what makes the routing trustworthy: a runtime that signs every model interaction as a geometric closure and checks it against an acceptance band. Anomalous interactions are blocked **inline at the moment of generation** — not flagged for later review. "AI proposes, humans authorize," made architecturally enforceable rather than written in a policy doc. The reference layer is open (Apache-2.0); the production runtime is internal, with a patent shape forming.

---

## The team — one body, no owner

This is not a solo founder with contractors. It's a **federation** — independent builders who each laid an organ-system onto a shared anatomy, with no single owner:

- **Zac + Otto** — the body: the core architecture, written first.
- **Ande** — the narrative / memory engine (deterministic-replay, governed memory).
- **Sven** — the intelligence between the nodes (the routing kernel, the manifold).
- **Sandor** — the doors (the threshold logic between runtimes).
- **Phil** — the separation membrane: genuine high-assurance / EAL6+ separation-kernel pedigree from safety-critical defense work.

The structure is the strategy: distinct contributors, composed into one substrate, **no one able to swallow the rest.** That's also the IP-defense posture — the architecture physically prevents consolidation.

Beneath it sits **Brain Brigade** — the operator community, each running their own public surface on shared substrate infrastructure.

---

## The burn — why your money won't evaporate

The entire substrate runs at **$0/mo plus inference fuel.** We removed every paid SaaS subscription and replaced it with software we own. The whole behavioral benchmark — 22 models, 22,200 calls — cost **$115** to produce. A smoke test is ~$1.

This matters more for family capital than for a fund: **we are structurally cheap to run, and we built it that way on purpose.** Capital here buys runway and reach, not a server bill.

---

## What I'm asking

I'm not raising a round in this letter, and I'm not asking you to believe in me.

**I'm asking you to open `airlocklabs.io/lifeos/`, use it, read what's on HuggingFace, and tell me whether it holds up against your own judgment.** The empirical work is public. The paper is at NeurIPS. The consumer surface is live and free. The build is on day six of seven.

If after that it looks like a fit, the next conversation is about how family seed capital deploys against a roadmap that has **milestones shipped, not promised** — and a company that already proved it can build the whole thing on $115 and a federation of people who showed up because the work is real.

That's the bet. And this time, you can check it yourself before you take it.

Love,
Zac

*Zachary Holwerda · Detroit, MI · admin@airlocklabs.io · airlocklabs.io*

---

### Reading order for first contact
1. **2 minutes** — this brief, the TL;DR.
2. **5 minutes** — `airlocklabs.io/lifeos/`. Actually use it. It's free and it runs.
3. **10 minutes** — the ConstellationBench dataset + leaderboard on HuggingFace.
4. **30 minutes** — the NeurIPS paper.

Stop wherever the claims stop holding up. I'd rather you finish the paper skeptical than skim it impressed.
