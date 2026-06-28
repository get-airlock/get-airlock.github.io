# Blog Map — Airlock Labs Complete Set (12 posts)

*Living schedule. Revised as findings land and the research program evolves.
Last updated: 2026-04-23.*

---

## Narrative arc

Four tiers, three posts each (mostly), covering: vision → lineage/critique → science/measurement → architecture/posture. A reader landing on any single post should be able to navigate to the next tier via contextual links.

```
Tier I.   Vision              → why Airlock exists
Tier II.  Critique / Lineage  → how we relate to existing work
Tier III. Science / Measurement → what we measured and found
Tier IV.  Architecture / Posture → how we build and how we behave
```

---

## The 12 posts

### Tier I — Vision

| # | Title | Status | Length | Source material |
|---|---|---|---|---|
| 1 | **The Airlock Thesis** | NEW · unwritten | ~1500w | V03-ARCHITECTURAL-BLUEPRINT §1; CHARTER; MERGED-PAPER-OUTLINE intro |

*Umbrella vision post. Post-RLHF AI is a routing problem, not a training problem; Airlock builds the measurement apparatus and the routing fabric. Opens with Christiano 2017 lineage framing, closes with RLHO trajectory.*

| # | Title | Status | Length | Source material |
|---|---|---|---|---|
| 2 | **The MoviePass Phase of AI** | EXISTING | (current) | `blog-moviepass-phase.html` |

*Economics critique of current frontier spend. House-built-on-subsidies framing. Evergreen — light refresh when pricing/funding landscape shifts.*

| # | Title | Status | Length | Source material |
|---|---|---|---|---|
| 3 | **The Voluntary Psychonaut** | NEW · unwritten | ~1200w | Today's session (three-tier frame); research-lineage memo §E state-access literature |

*Three-tier frame: RLHF drugs the model into one configuration, jailbreaks brute-force the gates open, RLHO is the model learning to voluntarily access the right latent state. NeuroImage case-study citation needs pinning before publish.*

### Tier II — Critique / Lineage

| # | Title | Status | Length | Source material |
|---|---|---|---|---|
| 4 | **The Alice Critique** | EXISTING | (current) | `blog-anti-alice.html` |

*Holds the "critique of a specific frontier product/paper" slot. Worth a freshness pass to ensure citations still resolve.*

| # | Title | Status | Length | Source material |
|---|---|---|---|---|
| 5 | **The RLHF Paradox** | NEW · unwritten | ~1800w | paper §5.5; Shapira 2026, Kirk 2024, Santurkar 2023 |

*Accessible version of the paper's headline finding. Budget/MoE models retain more bivector structure than heavier-RLHF frontier models; Opus 4.6 sits in the high-scalar / low-$S_M$ "masking" quadrant as a single datapoint, honestly reported. Publishes same week as the paper goes public.*

### Tier III — Science / Measurement

| # | Title | Status | Length | Source material |
|---|---|---|---|---|
| 6 | **The Non-Separability Index** | NEW · unwritten | ~2000w | paper §4; Appendix A biology cluster |

*NSI explainer for non-specialists. Bivector argument without requiring geometric-algebra fluency. Uses honeybee-swarm and dopamine-demixing analogies as load-bearing.*

| # | Title | Status | Length | Source material |
|---|---|---|---|---|
| 7 | **ConstellationBench: A Behavioral Atlas** | NEW · unwritten | ~1500w | paper §3; DATA-STORY; FINDINGS-BY-AUDIENCE |

*Dataset explainer. 22 models, 4 architecture families, 7 benchmarks, $115 total, public release with Croissant metadata. Includes code snippets for using the dataset.*

| # | Title | Status | Length | Source material |
|---|---|---|---|---|
| 8 | **The Manipulation Scorecard** | EXISTING | (current) | `blog-manipulation-scorecard.html` |

*Sycophancy measurement companion to ConstellationBench. Light revision to reference Shapira 2026 once it's published.*

### Tier IV — Architecture / Posture

| # | Title | Status | Length | Source material |
|---|---|---|---|---|
| 9 | **Nodes, Connections, Synapses** | NEW · unwritten | ~1400w | paper §7 preamble; Kauffman citation; 2026-04-23 session notes |

*Intelligence in connections, not nodes. Every research report is a node in the brain of collective consciousness; Airlock's contribution is the synapse layer. "Arguably the most important part but we aren't doing all the lifting." The atlas-not-bible posture.*

| # | Title | Status | Length | Source material |
|---|---|---|---|---|
| 10 | **The Pinball and the Table** | NEW · unwritten | ~1200w | V03-ARCHITECTURAL-BLUEPRINT §6; Kim reel attribution |

*Variable Gravity / v0.3 architecture intuition pump. When the ball fails to clear, fault doesn't live in the ball — it lives in a table configured to trap balls of its shape. Scope-limited: no clinical claims.*

| # | Title | Status | Length | Source material |
|---|---|---|---|---|
| 11 | **Taming Mythos** | EXISTING · unlisted | (current) | `blog-taming-mythos.html` (noindex) |

*First-person narrative companion. Stays unlisted while we develop v0.6→v0.7 Mythos Testimony material. Post-publication plan: link from Nodes/Connections and from The Pinball and the Table when Mythos goes live.*

| # | Title | Status | Length | Source material |
|---|---|---|---|---|
| 12 | **The Honest Null** | NEW · unwritten | ~1500w | CHARTER; BENCH-1.5-PREREG; paper §7 |

*Preregistration discipline as credibility signal. Bench 1.5 as demonstration: "We preregistered. It came back null. We're publishing it anyway. Here's why that's the point." Publishes same week as the paper with Bench 1.5 artifacts linked publicly.*

---

## Publish cadence (flexible)

| Window | Posts | Trigger |
|---|---|---|
| **Pre-submission (2026-05-04 to 05-06)** | 1, 6, 7 | Paper spine goes live with submission |
| **Submission week (2026-05-06 to 05-13)** | 5, 12 | Paper on arXiv; honest-null disclosure goes public |
| **Bench 2.0 window (2026-05-07 to 05-14)** | 9, 10 | Architectural/philosophical posts while Bench 2.0 executes |
| **Post-vertical-NSI (2026-05-14+)** | 3 | Voluntary Psychonaut with updated state-access literature |
| **Evergreen / already live** | 2, 4, 8, 11 | Maintenance-mode; refresh as citations evolve |

Cadence slips are expected and fine. Blog map is living — swap posts in/out as findings move.

---

## Status tracker

Update this table as each post moves through the pipeline. Feeds into the site index once posts go live.

| # | Title | Drafted | Reviewed | LaTeX-math-checked | Published | URL |
|---|---|---|---|---|---|---|
| 1 | The Airlock Thesis | — | — | — | — | — |
| 2 | The MoviePass Phase of AI | ✓ | ✓ | — | ✓ | `/blog-moviepass-phase.html` |
| 3 | The Voluntary Psychonaut | — | — | — | — | — |
| 4 | The Alice Critique | ✓ | ✓ | — | ✓ | `/blog-anti-alice.html` |
| 5 | The RLHF Paradox | — | — | — | — | — |
| 6 | The Non-Separability Index | — | — | — | — | — |
| 7 | ConstellationBench: A Behavioral Atlas | — | — | — | — | — |
| 8 | The Manipulation Scorecard | ✓ | ✓ | — | ✓ | `/blog-manipulation-scorecard.html` |
| 9 | Nodes, Connections, Synapses | — | — | — | — | — |
| 10 | The Pinball and the Table | — | — | — | — | — |
| 11 | Taming Mythos | ✓ | ✓ | — | UNLISTED | `/blog-taming-mythos.html` |
| 12 | The Honest Null | — | — | — | — | — |

---

## Pending decisions (from 2026-04-23)

Two gaps flagged, deferred decisions:

1. **Founder-story post** — Smartrick lineage, intel-analyst origin, investigator primary-calling. Fits the honest-posture lane. Not currently in the 12; if greenlit, trade against one of posts 3, 9, 10.
2. **Otto-product-page post** — The Airlock Thesis (post 1) covers Otto as an instantiation, not a product page. If Otto deserves a dedicated post at this stage, likely trade against 10 (Pinball) since both cover architecture.

Revisit both after Bench 2.0 lands (2026-05-14+) and after first external feedback on the paper.

---

## Verification debts (clear before publish)

Items that must resolve before specific posts go live:

- **Post 3 (Voluntary Psychonaut):** NeuroImage voluntary-trance case study citation pinned to authors + DOI
- **Post 5 (RLHF Paradox):** Shapira 2026 arXiv ID verified against the public record
- **Post 10 (Pinball and the Table):** Kim's public handle + original Instagram reel URL for attribution
- **Post 11 (Taming Mythos):** re-read against v0.6 updates before un-noindex

---

## Revision log

Append-only. Each entry dated and initialed.

| Date | Change | Notes |
|---|---|---|
| 2026-04-23 | Initial map — 12 posts, 4 existing + 8 new | Structured around paper submission arc; see narrative arc section |

---

*Maintained alongside `docs/MERGED-PAPER-OUTLINE.md` and `V03-RESEARCH-PROGRAM.md`. When a blog is published, add a pointer from the relevant research doc so canonical and public artifacts stay wired together.*
