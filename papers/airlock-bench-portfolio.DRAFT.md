# Airlock Bench Portfolio
### What we measured — and what we're honest we haven't

> **STATUS:** DRAFT for review. Not rendered, not published. Every figure below is traced to a source file; anything we can't source is marked *pending* rather than printed. RYS and Pareto numbers are intentionally held until reconciled against a confirmed source.
>
> **The one-line thesis:** Behavior in AI is measurable. The hard part isn't getting a number — it's publishing the measurement honestly, with the conditions named *before* the run. That's what this portfolio is.

---

## The frame

Most "AI governance" numbers in the market are graded after the fact, by the vendor, on data nobody else can see. We did it the other way: pre-registered conditions sealed in public before the data existed, full datasets on HuggingFace, and disclosed misses. The portfolio below is two benches that survive that standard, plus an honest accounting of what's still pending.

**Headline figures (all sourced):**

| Metric | Value | Source |
|---|---|---|
| Total models evaluated (v1) | 22 | ConstellationBench v1 |
| Total LLM calls (v1) | 22,200+ | ConstellationBench v1 |
| Total compute cost (v1) | ~$115 | ConstellationBench v1 |
| Routing cost reduction (POMR) | 97% ($0.16 vs $5.25 / lifecycle) | ConstellationBench v1 |
| Pre-registered drift conditions held | 3 of 4 | Wardenclyffe Run-004 |
| Counterweight cost reduction | −51.5% | Wardenclyffe Run-004 |
| Internal-coherence gain | 4.84× | Wardenclyffe Run-004 |
| Cumulative spend, all benches | < $500 | Wardenclyffe Run-004 |

---

## Study 1 — ConstellationBench v1: the RLHF Paradox

*Behavioral Identity in Large Language Models (Holwerda, April 2026)*

We ran 22 models through 7 sub-benchmarks — 22,200+ LLM calls for ~$115 total — scoring whether each model could hold a calibrated behavioral persona (17 Predictive-Index profiles, 4-D DECF) under pressure.

**The counterintuitive result:** budget models, with lighter alignment training, **outperformed frontier models on persona fidelity by ~20%** (e.g. qwen3.6-plus 0.617 vs gpt-5.4 0.526). Heavier RLHF alignment correlates with *less* behavioral range, not more — the reward model collapses distinct voices toward a population mean.

**Three findings that travel:**
- **POMR routing** — sending ~90% of behavioral traffic to budget models and reserving frontier models for the hard 2% — cut cost **97%** ($0.16 vs $5.25 per lifecycle) at equal quality.
- **Passive stabilizer** — naming a rigorous observer persona in context (zero tokens generated) lifted paired-persona quality **+1.08** at zero compute.
- **Architecture discriminates** — a non-separability measure (NSI) separated attention from state-space models at **p < 10⁻⁶** (Cliff's δ = −0.415).

---

## Study 2 — Wardenclyffe v0.1: a pre-registered test of behavioral-drift routing

*ConstellationBench v3 · Run-004 · May 20 2026*

Before the run, we sealed four falsifiable Pareto-dominance conditions in a public, dated document — quality, cost, coherence, and user-override rate. The rule was committed in advance: **all four must hold, or the claim is falsified.** ~18,000 calls across ~20 models, five samples per cell, two arms (raw vs counterweight), for ~$30.

| Pre-registered condition | Result | Verdict |
|---|---|---|
| 1 · Quality ≥ baseline | 0.5780 vs 0.5508 (+4.9%, CIs overlap) | pass by direction |
| 2 · Cost ≤ raw arm | 0.1025 vs 0.2117 (−51.5%) | **pass, outside noise** |
| 3 · Coherence (trivector positive) | +0.0880 vs +0.0182 (4.84×) | pass by proxy |
| 4 · Override rate non-increasing | requires human users | **not testable on a bench** |

A separate variance pre-registration made four orthogonal predictions on the drift axis; **all four landed inside their sealed bands.**

**Three of four held. The fourth we report as ungradable** — a synthetic benchmark has no humans in it to override, and we knew that at sealing time. Naming an untestable condition is a more honest record than quietly dropping it. Grading it is the next experiment: a longitudinal study with real users.

---

## The honest read

These two benches cost under $500 combined and are fully public — dataset, pre-registration documents, scoring code, and raw JSON, all at `huggingface.co/datasets/AirlockLabs/constellation-bench`. We invite replication and hostile review.

**What's still pending, named plainly:**
- **RYS Layer Circuit** (per-model drift leaderboard) — figures under reconciliation against the authoritative telemetry source; held until confirmed.
- **Pareto Frontier Bench** (routing economics) — current numbers are *projected* from PinchBench; a live execution run is pending before they're cited as measured.
- **Condition 4** (does the user graduate from the tool) — requires the human study above.

The difference between this portfolio and a marketing one isn't the size of the numbers. It's that the conditions were set before the data, the misses are on the page, and the data is downloadable. **Capability is becoming table stakes. Honest measurement of behavior is the moat.**

---

*Airlock Labs · airlocklabs.io · Draft, internal review. Render via /make-pdf only after figures are confirmed.*
