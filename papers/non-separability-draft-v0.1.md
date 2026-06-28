---
title: "Non-Separability as a Design Principle for Behavioral AI Systems"
subtitle: "Why pairwise interactions in LLM deployments require bivector-valued representations, and what that means for routing, consistency, and consent"
author: "Zachary Holwerda"
affiliation: "Airlock Labs"
date: "2026-04-22 (working draft v0.1)"
paper_class: position paper — target venue NeurIPS Position Track, ML4H, ICML-AIES, or arXiv preprint
---

## Abstract

Modern behavioral AI systems — recommendation engines, conversational agents, router-based deployments — routinely treat user interactions as *scalar* events: a compatibility score, a match rate, an engagement metric, a sentiment polarity. We argue that this scalarization is not merely lossy; it is **structurally incapable** of representing the information content of an interaction between two distinct behavioral entities (user↔agent, user↔user, agent↔agent). Drawing on geometric algebra (Hestenes), the manifold hypothesis, and the quantum-foundations literature on non-separable states, we propose that the minimal faithful representation of a pairwise interaction is *bivector-valued*: a scalar component (alignment magnitude) plus an oriented plane-of-interaction (the bivector) that preserves the geometric information classical inner products discard.

We then show that three otherwise disconnected phenomena in applied ML — (i) LLM *sycophancy* under adversarial pressure, (ii) *decoherence* of long-horizon reasoning chains, and (iii) the *surveillance-residue* problem in data deletion — are the same structural failure under different names: the system has collapsed a non-separable joint state into a separable scalar representation. We propose a unifying benchmark, the Non-Separability Index (NSI), operationalized through an extension to the existing ConstellationBench methodology (Holwerda, 2026), and argue that routing layers that preserve bivector structure are *not* an optimization technique but a structural necessity for any deployed system with commercial, regulatory, or safety-sensitive interaction requirements.

The position is deliberately narrow in claim and broad in consequence. We are not claiming LLMs are quantum systems. We *are* claiming that the mathematics of non-separability — originally formalized in quantum foundations but equally applicable in classical geometric algebra and in network theory — provides the cleanest available vocabulary for describing what breaks in sycophantic, decohered, or surveilled AI systems, and therefore the cleanest available design principle for fixing them.

## 1. Introduction

The canonical deployment pattern for behavioral AI systems today is:

1. Encode each entity (user, agent, item) as a vector in some feature space.
2. Score pairwise compatibility via an inner product or learned similarity function.
3. Use the scalar output to drive a decision (match, rank, route, deny, personalize).

This pipeline works well for bulk recommendation and retrieval where the relevant signal is aggregate. It *fails systematically* when the interaction itself carries information that cannot be recovered from either entity's unilateral state — precisely the situation in conversational agents, adversarial robustness, behavioral benchmarking, and any setting where a user's question and an agent's answer are *jointly* what the system is optimizing for.

This failure has been observed, named, and partially addressed under separate headings:

- **Sycophancy** in language models (Sharma et al., 2024; Perez et al., 2022): the agent's output-state becomes correlated with user-framing rather than with ground-truth, producing model responses that flip under rephrasing or adversarial pressure without any change in underlying facts.
- **Decoherence** in long-horizon agent chains: information that should persist across turns degrades because the joint state of the interaction is approximated turn-by-turn as a sequence of scalar updates.
- **Surveillance residue**: deleting a user's scalar identifiers (cookies, accounts, passwords) from a system does not decouple the user from downstream behavioral predictions, because the joint state of user-and-system persists in the data distribution even after the scalar trace is scrubbed.

We argue these are the *same failure*. Each is a case of a non-separable joint state being compressed into a scalar representation, and then the downstream system being surprised when the discarded geometric information was actually load-bearing.

### 1.1 The proposed frame

Given two vectors $a, b \in \mathbb{R}^n$, the standard inner product $a \cdot b$ returns a scalar. In geometric algebra, the *geometric product* returns:

$$a \otimes_g b \ = \ a \cdot b \ + \ a \wedge b$$

where $a \wedge b$ is the bivector, an oriented plane spanned by the two vectors. For orthogonal unit vectors, the scalar part vanishes and only the bivector remains — i.e., the interaction is *entirely* information the inner product would discard. For collinear vectors, the bivector vanishes and the scalar is sufficient. Most real interactions sit between these extremes; current systems model them as if they were purely collinear.

A *non-separable* joint state is, by definition, one that cannot be written as a tensor product of single-entity states. In geometric-algebra terms, it is one whose bivector component is non-vanishing. In applied ML terms, it is one whose interaction-specific information cannot be recovered from either participant's embedding alone.

### 1.2 What this paper claims

- **Claim 1 (descriptive):** Sycophancy, decoherence, and surveillance-residue are the same phenomenon: scalar collapse of a non-separable joint state.
- **Claim 2 (measurement):** The Non-Separability Index (NSI), defined as the ratio of the bivector norm to the scalar norm of the geometric product of the system's representation of an interaction, quantifies how much of the interaction's information the system is throwing away.
- **Claim 3 (design):** Routing layers that select models based on NSI rather than scalar capability benchmarks structurally outperform on interaction-sensitive workloads (trading advice, clinical triage, legal counsel, education, therapy — any setting where the *dialogue* is the product).
- **Claim 4 (regulatory):** User consent frameworks predicated on scalar identifiers (GDPR right-to-deletion, CCPA) are structurally insufficient because deletion of the scalar trace does not collapse the non-separable joint state. True consent requires non-entanglement at the outset.

## 2. Related Work

- **Geometric algebra and ML.** Hestenes (1966, 2015), Doran & Lasenby (2003); recent resurgence in geometric deep learning (Bronstein et al., 2021) which argues architecture should respect the geometry of data.
- **LLM behavioral benchmarks.** Sharma et al. 2024 (Anthropic) on sycophancy; Perez et al. 2022 on in-context adversarial robustness; Holwerda 2026 (ConstellationBench) on adversarial consistency across 22 models.
- **Non-separability in quantum foundations.** Bell (1964), Aspect (1982), Horodecki et al. (2009). Note: we borrow the mathematical vocabulary, not the physical claim. LLMs are classical systems; their non-separability is algebraic, not quantum-mechanical.
- **Consent and data infrastructure.** Veale & Edwards (2018); Zuboff (2019) on surveillance capitalism; emerging work on differential privacy in the presence of joint-state leakage.

## 3. The Non-Separability Index (NSI)

### 3.1 Definition

For any pairwise interaction between entities with representations $a, b$, define:

$$\text{NSI}(a, b) = \frac{\|a \wedge b\|}{\|a \cdot b\| + \|a \wedge b\|} \in [0, 1]$$

NSI = 0: the interaction is fully captured by scalar compatibility; the bivector vanishes; classical methods are sufficient.

NSI = 1: the interaction is purely geometric; the scalar vanishes; classical methods preserve none of the relevant information.

Most interactions fall between. Our empirical claim is that most *commercially valuable* interactions sit at NSI > 0.3, and that existing production systems that operate as if NSI = 0 are leaving substantial predictive power and safety margin on the table.

### 3.2 Operationalization via ConstellationBench

ConstellationBench (Holwerda, 2026) measures LLM consistency under adversarial prompt perturbations across 22 frontier and open models. We note that ConstellationBench's measured "sycophancy gap" (42% vs 89% hold-rate under "are you sure?" perturbation) is isomorphic to an NSI estimate: models with high hold-rate are preserving bivector information across perturbations; models with low hold-rate are allowing bivector collapse. A proposed extension, ConstellationBench-NSI, would report NSI explicitly per-model per-domain, turning a behavioral benchmark into a geometric-algebra measurement.

## 4. The Router as Non-Separability Preserver

The Airlock Router (internal, production) selects among candidate LLMs based on behavioral profile match rather than raw capability score. We now claim this is not an optimization heuristic but a structural requirement: a router that routes based on scalar benchmarks will systematically select for high-capability low-NSI-preservation models, producing the sycophancy-under-pressure phenomenon the market is beginning to notice.

**Claim 3 reformulated:** routing is the act of *preserving* bivector structure across the model-selection boundary. A router that cannot do this is not a router; it is a load balancer.

## 5. Regulatory Consequences — The Consent Problem

GDPR Article 17, CCPA §1798.105, and similar frameworks grant users the right to delete their *scalar trace* — account, records, identifiers. This framework assumes user and system are separable at the level of the user's scalar representation. Non-separability shows this assumption is false: user-system interactions generate joint-state information that persists in the data distribution even after the user's scalar identifiers are removed.

This is not a critique of the existing frameworks; it is a specification gap. Future consent regimes must address non-entanglement (the right to never have entered the joint state) in addition to deletion (the right to scrub the scalar trace).

This has practical implications for any system handling sensitive interactions: a platform that routes a user's conversation through an LLM and then "deletes the user's data" is making a claim about separability that the mathematics does not support.

## 6. Experimental Program

1. **ConstellationBench-NSI extension.** Instrument the existing 22-model benchmark with explicit NSI scoring. Target: 8 weeks.
2. **Router A/B.** Compare NSI-preserving vs. scalar-optimized routing on three real workloads: trading-signal generation, clinical triage, legal summarization. Target: 12 weeks.
3. **Consent leakage study.** On a deployed conversational system, measure the predictive information retained about deleted users from their downstream interaction partners' data. Target: 16 weeks, pending IRB.

## 7. Limitations and Non-Claims

We are **not** claiming:

- That LLMs are quantum-mechanical systems.
- That non-separability implies any faster-than-light-style communication between user and model.
- That this framework resolves AI alignment, interpretability, or safety in general.

We **are** claiming:

- That the mathematics of non-separability, borrowed from geometric algebra and quantum foundations, provides a precise vocabulary for three currently-named failures in applied ML.
- That this vocabulary suggests a measurable, testable extension to existing benchmarks.
- That routing architectures should be evaluated on NSI-preservation, not scalar benchmarks alone.

## 8. Author's Note on Research Posture

This paper is the theoretical spine for a broader research program at Airlock Labs on behaviorally-aware AI infrastructure. Companion work-in-progress:

- The Kernel Hypothesis (behavioral manifold geometry, in draft)
- The ConstellationBench dataset (published, HuggingFace)
- The DECF profile framework (production, internal)
- The POMR router (production, internal)

Licensing, partnership, and research-collaboration inquiries: `zac@airlocklabs.io`.

---

## References (to be filled in full at submission)

[Sharma et al. 2024], [Perez et al. 2022], [Holwerda 2026 — ConstellationBench], [Hestenes 1966], [Doran & Lasenby 2003], [Bronstein et al. 2021], [Bell 1964], [Aspect 1982], [Horodecki et al. 2009], [Veale & Edwards 2018], [Zuboff 2019], [Anokhin — Cognitome theory, 2023], [Gusev et al. — Evolutionary Trajectories of Consciousness, 2024].
