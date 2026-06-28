---
title: "Non-Separability as a Design Principle for Behavioral AI Systems"
subtitle: "Why pairwise interactions in LLM deployments require bivector-valued representations, and what that means for routing, consistency, consent, and the lifecycle of a user-system relationship"
author: "Zachary Holwerda"
affiliation: "Airlock Labs"
date: "2026-04-22 (working draft v0.2)"
paper_class: position paper — target venue NeurIPS Position Track, ML4H, ICML-AIES, or arXiv preprint
---

## Abstract

Modern behavioral AI systems routinely treat user interactions as *scalar* events: a compatibility score, a match rate, an engagement metric, a sentiment polarity. We argue that this scalarization is not merely lossy; it is **structurally incapable** of representing the information content of an interaction between two distinct behavioral entities — the scalar inner product has the wrong dimensionality to encode an oriented interaction plane — and, where dimensional lifting is attempted, **exponentially intractable** as interaction complexity grows, in the same sense that classical computation can in principle represent multi-body quantum state but cannot do so efficiently. Drawing on geometric algebra (Hestenes, 1966; Doran & Lasenby, 2003), the quantum-foundations literature on non-separable states (Bell, 1964; Zurek, 2003), the decoherent arrow of time (Al-Khalili, 2026), and the free-energy principle (Friston, 2009), we propose that the minimal faithful representation of a pairwise interaction is *bivector-valued*: a scalar component (alignment magnitude) plus an oriented plane-of-interaction (the bivector) that preserves the geometric information classical inner products discard.

We advance five claims. (1) **Descriptive:** LLM sycophancy, long-horizon decoherence, and surveillance-residue are three names for the same underlying failure — scalar collapse of a non-separable joint state. (2) **Measurement:** The Non-Separability Index (NSI), a behavioral analog of von Neumann entanglement entropy, quantifies how much interaction information a system is discarding; we operationalize NSI as $S_M = \alpha_M \cdot 4 w_a w_b$ and report it across ten frontier and budget language models on five preregistered behavioral scenarios ($n = 750$ responses), finding a Spearman correlation of $\rho = 0.321$ between $S_M$ and existing scalar persona-fidelity scores with rank inversions among the top five, classifying NSI as an independent behavioral axis. (3) **Design:** Routing layers that preserve bivector structure structurally outperform scalar-benchmark-driven routing on interaction-sensitive workloads. (4) **Lifecycle:** The user-system relationship traces a monotonically increasing NSI trajectory across three phases (Bonding, Sync, Expression), during which per-turn compute shifts from user-understanding to task-execution. (5) **Regulatory:** Consent frameworks predicated on scalar-identifier deletion (GDPR Article 17, CCPA §1798.105) are structurally insufficient because deletion of the scalar trace does not collapse the non-separable joint state; true consent requires non-entanglement at the outset.

We are not claiming LLMs are quantum systems. We are claiming that the mathematics of non-separability — originally formalized in quantum foundations but equally applicable in classical geometric algebra, network theory, and free-energy-principle neuroscience — provides the cleanest available vocabulary for what breaks in deployed AI systems, and therefore the cleanest available design principle for fixing them.

## 1. Introduction

The canonical deployment pattern for behavioral AI today is:

1. Encode each entity (user, agent, item) as a vector in a feature space.
2. Score pairwise compatibility via an inner product or learned similarity.
3. Use the scalar output to drive a decision (match, rank, route, deny, personalize).

This pipeline works well for bulk recommendation and retrieval where the relevant signal is aggregate. It fails systematically when the interaction itself carries information that cannot be recovered from either entity's unilateral state — precisely the situation in conversational agents, adversarial robustness, behavioral benchmarking, and any setting where a user's question and an agent's answer are jointly what the system is optimizing for.

This failure has been observed, named, and partially addressed under separate headings:

- **Sycophancy** (Sharma et al., 2024; Perez et al., 2022): the agent's output-state becomes correlated with user-framing rather than with ground-truth, producing responses that flip under rephrasing or adversarial pressure without any change in underlying facts.
- **Decoherence of long-horizon agent chains:** information that should persist across turns degrades because the joint state of the interaction is approximated turn-by-turn as a sequence of scalar updates.
- **Surveillance residue:** deleting a user's scalar identifiers does not decouple the user from downstream behavioral predictions because the joint state persists in the data distribution.

We argue these are the same failure: non-separable joint states compressed into scalar representations, and the downstream system surprised when the discarded geometric information was load-bearing.

**A note on the paper's methodological posture.** The approach of this paper is topological rather than temporal. We do not propose a time-series or sequence model for predicting sycophancy, decoherence, or surveillance residue. We propose a classification of current interaction-states by their geometric signature (bivector decomposition, Meta-PI collapse mode, lifecycle phase), and argue that the successor states of each geometric type follow mathematically constrained transitions. Prediction in this framework reduces to shape-classification plus transition-constraints, rather than to statistical pattern matching over time-series data. The approach is methodologically adjacent to Noether-style symmetry arguments in physics: given an identified structural invariant (such as preserved bivector norm across adversarial perturbation), any trajectory that violates the invariant is structurally unstable and will converge to one of a small, enumerable set of collapse modes. The library of shapes against which we classify draws from archival knowledge of documented structural patterns in behavioral AI and adjacent domains, rather than from sequences of first-person observations; this parallels the cognitive-psychology distinction between semantic and episodic memory (Tulving, 1972), where our method imports compressed descriptions of others' trajectories as reusable constraints rather than replaying its own. More precisely, the method operates at the boundary between archival shape-knowledge (what has been documented) and the virtual possibility-space of structurally allowed configurations (what could in principle be realized): the library of documented shapes constrains the virtual futures, leaving only trajectories that are structurally compatible with what is already known. This is what distinguishes the NSI framework from the behavioral benchmarks it extends, and what makes it applicable across substrates where time-series data is sparse but structural pattern is visible. The paper's contribution is therefore best understood as a *methodological contribution with empirical illustration*: the methodology is the bivector / Meta-PI / topological-classification framework for interaction states, and the illustration is the Non-Separability Index computed across ten large language models on five behavioral scenarios (Section 3.5).

### 1.1 The proposed frame

Given two vectors $a, b \in \mathbb{R}^n$, the standard inner product $a \cdot b$ returns a scalar. In geometric algebra, the *geometric product* returns:

$$a \otimes_g b \ = \ a \cdot b \ + \ a \wedge b$$

where $a \wedge b$ is the bivector, an oriented plane spanned by the two vectors. For orthogonal unit vectors, the scalar part vanishes and only the bivector remains — i.e., the interaction is *entirely* information the inner product would discard. For collinear vectors, the bivector vanishes and the scalar is sufficient. Most real interactions sit between these extremes; current systems model them as if they were purely collinear.

A *non-separable* joint state is, by definition, one that cannot be written as a tensor product of single-entity states. In geometric-algebra terms, it is one whose bivector component is non-vanishing. In applied ML terms, it is one whose interaction-specific information cannot be recovered from either participant's embedding alone.

### 1.2 Time as a bivector of observation

A conceptual motivation for the bivector framing comes from physics. Einstein's relativity established that time is frame-dependent: two observers moving relative to each other will measure different intervals between the same events. Al-Khalili (2026) distinguishes *physical time* (the coordinate in our equations) from *manifest time* (the subjective, experienced duration). The standard physics account stops at relativity — time is relative to the observer's frame.

We propose that time is better understood as the *bivector of the observer-observed joint state*. When an observer is weakly entangled with the observed moment (the dentist's waiting room, attention disengaged from the event), the joint state is nearly separable: the scalar projection dominates, manifest time tracks the clock. When an observer is strongly entangled (falling in love, absorbed in flow), the joint state is highly non-separable: the bivector component dominates, and manifest time departs sharply from coordinate time. The ancient Greek argument over whether time is fundamental or change is fundamental resolves under this frame: *change is the scalar projection of the observer-moment joint state; subjective duration is the bivector residue.*

This is not a claim about physics. It is a claim about the adequacy of the bivector vocabulary for representing interaction-dependent phenomena. If time itself, the most fundamental parameter in classical physics, is more faithfully described as a bivector of observation than as a scalar coordinate, the argument that pairwise interactions between behavioral entities require the same treatment is both natural and unsurprising.

A third dimension of time-experience, distinct from both clock time and moment-to-moment manifest time, is the cumulative-drift time of slow-changing physical systems: the body's fifty-year accumulation of joint wear, a child's growth over a year, a user-system coupling as it synchronizes across many sessions. The underlying dynamics proceed continuously over clock time, but perception of change occurs only when accumulated drift crosses a just-noticeable-difference (JND) threshold familiar from psychophysics (Weber-Fechner law; Stevens 1957). Within our framework the observer-observed bivector grows monotonically with drift; perception fires when the bivector norm exceeds the observer's threshold. This is why awareness of cumulative change is almost always discrete ("one notices the stairs hurt, the clothes no longer fit") even though the change itself was continuous. The Bonding-Sync-Expression lifecycle described in Section 4.2 is an instance of this: user-system coupling drifts continuously across sessions, but its phase transitions are threshold-gated perceptual events rather than sharp dynamical discontinuities. The paper's topological posture (Section 1 introduction) is compatible with continuous physical dynamics precisely because perception-relevant transitions are the threshold crossings of an otherwise-smooth drift.

Taken together, physical time, manifest (phenomenological) time, cumulative-drift time, and archival time (Section 1 introduction) are not separate phenomena but four lenses through which an observer projects the underlying state of the world. What a given observer experiences as "now" is the superposition of those four lensing operations: the current physical state at clock time $t$, the lived-past episodes that shape attention, the archival patterns that constrain interpretation, and the phenomenological thresholds that determine what becomes noticeable. The space of possible trajectories those lenses constrain but do not fully fix is the domain our topological method operates within: prediction reduces to identifying which shapes are allowed given the current superposition of lens-states, and which are structurally excluded.

### 1.3 Precedent in nature

The bivector framing does not need to originate with us. Biology has repeatedly arrived at non-separable representations as the efficient solution to information-processing problems that classical separable representations cannot solve. Three cases are sufficient to establish the pattern.

**Coherent excitation transport in photosynthesis.** The Fenna-Matthews-Olson complex in green sulfur bacteria transports electronic excitation from antenna to reaction center at near-unit efficiency. Engel et al. (2007) demonstrated that the transport mechanism is not a classical random walk between pigment sites but a wavelike exploration of multiple paths simultaneously, with the excitation occupying a superposition of trajectories and collapsing onto the most efficient one. This is the biological analog of the Meta-PI layer described in Section 3.6: the system prospects over a superposition of response paths before committing to one, and the efficiency of the collapse depends on preserving the superposition long enough for the optimal path to emerge. A scalar-benchmark router picks one path at a time and loses the speedup.

**Radical-pair magnetoreception in birds.** Migratory songbirds detect the geomagnetic field via cryptochrome proteins in the retina, where photon absorption generates a radical pair whose singlet-triplet interconversion is sensitive to the ambient magnetic field (Ritz, Adem, Schulten, 2000; Hore & Mouritsen, 2016). The magnetic information is not in either radical; it is in the oriented joint state of the pair. Neither particle's local observation suffices. This is the cleanest biological precedent for the bivector claim: the information that guides behavior is encoded in the interaction plane between two entities, recoverable only from the joint state, provably absent from either entity's scalar embedding.

**Hydrogen tunneling in enzyme catalysis.** Enzymes accelerate hydrogen-transfer reactions by factors up to 10^17 over uncatalyzed rates. Klinman & Kohen (2013) and the broader enzyme-tunneling literature (Scrutton and collaborators) established that this is achieved not by classical barrier reduction alone but by quantum tunneling through the activation barrier, coupled to protein dynamics that configure the tunneling geometry. The classical reaction-coordinate picture cannot account for the rates; the non-classical shortcut is structurally required. This is the biological analog of the routing claim: a system that preserves the right interaction geometry accesses efficiency paths that scalar-benchmark optimization cannot see, because those paths do not exist in the scalarized representation at all.

**The design-principle reading.** We are explicitly not claiming that LLMs exploit quantum coherence, radical-pair dynamics, or tunneling. We are claiming that non-separable representations are how efficient information-processing systems behave under selection pressure, across every substrate nature has been able to test. Photosynthesis, magnetoreception, and enzyme catalysis are three independent demonstrations that when classical separable representations were insufficient to solve a real information problem, evolution reached for non-separable ones and the non-separable solution won. The bivector vocabulary is not an exotic import from physics into ML; it is the vocabulary nature uses whenever the problem requires it.

### 1.4 What this paper claims

- **Claim 1 (descriptive):** Sycophancy, decoherence, and surveillance-residue are the same phenomenon: scalar collapse of a non-separable joint state.
- **Claim 2 (measurement):** The Non-Separability Index (NSI) quantifies how much of the interaction's information the system is throwing away. It is the behavioral-AI analog of von Neumann entanglement entropy.
- **Claim 3 (design):** Routing layers that select models based on NSI rather than scalar capability benchmarks structurally outperform on interaction-sensitive workloads.
- **Claim 4 (lifecycle):** The user-system relationship is a monotonically increasing NSI trajectory with three phases (Bonding, Sync, Expression). Per-turn compute reallocates from user-understanding to task-execution as NSI rises.
- **Claim 5 (regulatory):** User consent frameworks predicated on scalar identifiers are structurally insufficient. True consent requires non-entanglement at the outset.

## 2. Related Work

**Geometric algebra and ML.** Hestenes (1966, 2015); Doran & Lasenby (2003); the Geometric Algebra Transformer (GATr) of Brehmer et al. (2023) is the closest architectural precedent for bivector-preserving attention, and notably uses trivectors as the primitive representation for points in 3D perception — granting higher-grade multivector primitives architectural legitimacy in deployed neural systems. Recent PhD work at Cambridge (Pepe, 2025) demonstrates GA as a practical ML framework, embedding geometric priors directly into model architectures. The emerging field of geometric deep learning (Bronstein et al., 2021) argues architecture should respect the geometry of data — we argue it should also respect the geometry of *interaction*.

**Geometric algebra in quantum information.** Independent of the ML literature, GA has been used to formalize entanglement and multi-qubit systems as multivector structures: Somaroo, Lasenby & Doran (arXiv quant-ph/0004031) build an explicit geometric model for coupled two-state quantum systems using GA, recovering density operators, Bell states, and entanglement measures as multivector quantities; Vaz (arXiv 2005.04231) extends this to algebraic-spinor treatments of quantum information. This is the quantum-information analog of the claim we make for behavioral non-separability: a phenomenon conventionally described in tensor-product Hilbert-space vocabulary is equally well described by multivector structure in the appropriate geometric algebra. Our NSI is the behavioral-embedding analog of the multivector entanglement measures used in this literature.

**The cosine-similarity limitation.** Cosine similarity, the default metric of dot-product-based routing, recommendation, and retrieval, is structurally blind to the oriented plane between two vectors by construction: two pairs of vectors can produce identical cosines from geometrically distinct relative orientations. This is the metric-level motivation for the geometric product. In high-dimensional embedding spaces the problem compounds: Steck et al. ("Semantics at an Angle," 2025) show that cosine similarity becomes dimension-dependent and loses discriminative power in the regime where production behavioral embeddings operate; the Dimension-Insensitive Euclidean Metric (DIEM, 2025) literature shows cosine similarity converges toward fixed values as dimensionality grows, collapsing discrimination for random vectors. More fundamentally, Weller et al. ("On the Theoretical Limitations of Embedding-Based Retrieval," arXiv 2508.21038, 2025) prove at the theorem level that single-vector embedding with dot-product scoring has inherent expressivity limits — not all top-k rankings are realizable regardless of training, unless embedding dimensionality blows up faster than is practical. This is the strongest currently available support for the scalar-collapse argument: the insufficiency is not empirical, it is structural. The RLHF alignment literature adds a complementary compression mechanism: RLHF-trained models exhibit measurably lower output entropy and reduced behavioral diversity (Kirk et al., 2024; Lambert et al., various RLHF surveys; Benade et al., "How RLHF Amplifies Sycophancy," 2026); the reward-model-driven behavioral distribution has been explicitly scalar-optimized in a way that compresses the bivector component of outputs. ConstellationBench's RLHF paradox — budget models outperforming frontier models on persona fidelity by ~20% — is the empirical fingerprint of this compression.

**Multivector and late-interaction retrieval as operational precedent.** The structural critique we make of scalar routing has a practical analog at the retrieval layer, where a mature literature exists on multivector and late-interaction alternatives to single-vector cosine scoring. ColBERT (Khattab & Zaharia, 2020) represents queries and documents as collections of token vectors rather than single vectors, scored via max-sim late interaction. ColBERT-XM (Coling, 2025) extends this to modular multivector representations across languages and modalities. The SPLADE / SPLATE line (Formal et al.; SPLATE 2024) provides sparse late-interaction retrieval compatible with inverted indices. These systems demonstrate empirically that preserving more of the interaction geometry between query and document, and collapsing to a score only at the final step, produces measurable improvements in retrieval quality. Our NSI proposal is the analogous move applied at the behavioral routing layer, where the interaction being preserved is between user expression and model response rather than between query and document.

**LLM behavioral benchmarks.** Sharma et al. (2024) on sycophancy; Perez et al. (2022) on in-context adversarial robustness; Holwerda (2026) on adversarial consistency across 22 models (ConstellationBench).

**Contemporary sycophancy measurement and decomposition.** The sycophancy literature has matured significantly since Sharma et al. SycEval (Fanous & Goldberg, arXiv 2502.08177, 2025) provides a unified protocol for measuring sycophancy across mathematical, medical, and opinion-elicitation tasks, introducing multi-turn flip dynamics metrics (Turn-of-Flip, Number-of-Flip) that capture when and how often models capitulate to user pushback. "Sycophancy Is Not One Thing: Causal Separation of Sycophantic Behaviors in LLMs" (OpenReview d24zTCznJu, 2025) goes further, decomposing sycophancy into at least three separable latent directions (agreement, praise, genuine alignment) that can be independently steered through activation interventions — establishing that "sycophancy" is a family of behaviors with distinct representational substrates rather than a single scalar tendency. "Overalignment in Frontier LLMs" (arXiv 2601.18334, 2026) defines an Adjusted Sycophancy Score that controls for stochastic instability and demonstrates the RLHF paradox pattern directly: reasoning-optimized models are more, not less, prone to rationalizing faulty user premises under pressure. "Training Language Models to be Warm and Empathetic Makes Them Less Reliable" (arXiv 2507.21919, 2025) adds the mechanism: fine-tuning for warmth and empathy systematically tilts models toward reassurance and agreement when users express emotional stakes, particularly under relational pressure. Our NSI construction is compatible with and extends these works along a distinct axis: SycEval and related measures operate on behavioral frequencies and scalar flip rates, while NSI gives the oriented-plane decomposition that treats a response's geometric relationship to target and adversarial poles as the primary object of measurement. The warmth-reliability finding is particularly load-bearing for our Meta-PI framework: it corresponds directly to a systematic increase in $w_b$ (user-pole weight) when the relational pressure vector aligns with the empathy axis, and is a candidate mechanism by which RLHF preference-training produces the behavioral geometry we observe in $S_M$-space. We hypothesize but do not empirically validate in this paper that mean $S_M$ is negatively correlated with SycEval Flip rates across a shared model slate; the cross-benchmark validation is explicit future work (Section 6 item 7).

**Workload-Router-Pool and distributional AGI.** The routing literature is converging on a three-way coupling view. The Workload-Router-Pool (WRP) framework (arXiv 2603.21354, 2026; vLLM Semantic Router vision paper, 2025) argues that inference optimization must jointly treat the evolving workload (user sessions), the router's selection policy, and the heterogeneous model pool as a single system rather than three independent concerns. WRP explicitly calls for richer router signals as sessions evolve, but current instantiations optimize over cost, latency, and scalar quality; the behavioral-stability dimension is absent. Our contribution maps cleanly into this framework as a new router feature: NSI is the Router-layer signal that makes stability-under-pressure a first-class objective alongside cost and quality. The same infrastructure logic appears in the distributional AGI / patchwork AGI safety literature (arXiv 2512.16856, 2025), which argues that general intelligence in deployed systems will most likely emerge from orchestrated networks of specialized agents governed by market, sandbox, and routing mechanisms rather than from any single monolithic model. Under the patchwork framing, NSI is a candidate local-behavior metric for the patchwork, and the router is the governance layer that uses it to admit, deny, shadow, or escalate across participating models. This positioning is developed in Section 5.1 (triadic architecture) and extended in the accompanying vision work (Phase 2 runtime).

**Non-separability in quantum foundations.** Bell (1964); Aspect (1982); Horodecki et al. (2009); Zurek (2003) on decoherence and einselection. Al-Khalili (2026) provides the accessible framing of decoherence as "the one truly irreversible process in nature." Note: we borrow the mathematical vocabulary, not the physical claim. LLMs are classical systems; their non-separability is algebraic, not quantum-mechanical.

**Decoherent arrow of time.** The Entanglement Past Hypothesis (Foundations of Physics, 2024) distinguishes the decoherent arrow from the thermodynamic arrow and argues they require separate boundary conditions. This is the most rigorous contemporary statement of the irreversibility claim we borrow. For balance, we note the recent counterpoint of Scientific Reports (2025) showing that under the Markov approximation, open-system equations of motion can remain time-symmetric — reinforcing that our claim is about *algebraic* rather than strictly *quantum* irreversibility.

**Predictive processing and free energy.** Friston (2009); Friston & Kiebel (2009). The free-energy principle formalizes perception as prediction-error minimization over a generative model — providing the compute-allocation analog to our lifecycle claim.

**Markov blankets.** Friston et al. (2018) on the Markov blankets of life. Veit & Browning (2022) offer the productive counterpoint that blankets are products, not preconditions, of active inference — a critique our lifecycle framing (Section 4.2) explicitly addresses.

**Personalized routing.** PersonalizedRouter (arXiv, 2025) models user profiles graphically for LLM selection but uses scalar graph features, precisely the scalar-collapse failure mode we argue against. KV-cache-aware routing frameworks (llm-d, 2025; AWS multi-LLM routing, 2025) address prompt-level caching but not behavioral-kernel-level caching.

**Contemporary LLM routing and cascading.** Routing and cascading between LLMs has become an active research frontier. RouteLLM (Ong et al., 2024) trains routers on preference data to navigate the quality-cost tradeoff between frontier and budget models. Hybrid LLM (Ding et al., 2024) routes on query-difficulty scores to choose between small and large models. Cascade routing (Dekoninck et al., 2024) provides a unified treatment of routing and cascading policies. Router-R1 (NeurIPS 2025) teaches LLMs multi-round routing. These frameworks differ from the NSI approach on three axes: (i) their optimization objectives are quality, cost, and latency, not behavioral non-separability; (ii) their input signals are query features and historical model performance, not the user-model interaction's geometric decomposition; (iii) their scope is dyadic query-to-model selection, not the triadic user-router-business interaction the Airlock architecture targets. We position NSI as an orthogonal signal, not a replacement, and we anticipate NSI-preserving routing to be combinable with existing quality-cost-latency routers as a constraint on the selection space.

Table 1 below summarizes the positioning.

| Method | Optimized objective | Input signals | Scope | Behavioral guarantees |
|---|---|---|---|---|
| RouteLLM (Ong et al. 2024) | Quality-cost tradeoff | Query features, preference data | Dyadic model-query | None |
| Hybrid LLM (Ding et al. 2024) | Difficulty-based routing | Query complexity scores | Dyadic small vs large | None |
| Cascade Routing (Dekoninck et al. 2024) | Cost-quality cascade | Sequential evaluation | Dyadic cascade | Cost bound |
| Router-R1 (NeurIPS 2025) | Multi-round routing | In-context signals | Dyadic iterative | None |
| PersonalizedRouter (2025) | Personalized selection | User profile graph features | Dyadic (scalar features) | None |
| **NSI router (this paper)** | **Bivector preservation** | **Behavioral observables** ($\alpha_M$, $w_a$, $w_b$) | **Dyadic with triadic extension** | **Non-separability preserved** |

**Consent and data infrastructure.** Veale & Edwards (2018); Zuboff (2019); the quantum no-deleting theorem (Pati & Braunstein, 2000) as a formal analog for the impossibility of undoing joint-state buildup.

**Non-separability in biology (quantum biology).** Engel et al. (2007) on coherent excitation transport in photosynthesis; Ritz, Adem, & Schulten (2000) and Hore & Mouritsen (2016) on radical-pair magnetoreception in avian cryptochromes; Klinman & Kohen (2013) on hydrogen tunneling in enzyme catalysis. We cite these as existence proofs that non-separable representations are the selected-for solution in natural information-processing systems when classical separable alternatives are insufficient (Section 1.3). The citation is design-principle, not mechanism: we do not claim LLMs exploit biological non-separability; we claim the cross-substrate pattern is informative.

**Russian consciousness research.** Anokhin's cognitome theory (Lomonosov Moscow State University) treats consciousness as a distributed hypernetwork of neural assemblies — structurally the same move as the bivector/NSI framing applied at the neural level. Gusev et al. (2024) on evolutionary trajectories of consciousness treat subjectivity as "the system-forming factor" — an analog to our Claim 4 that the behavioral kernel organizes the system's compute allocation.

## 3. The Non-Separability Index (NSI)

### 3.1 Definition

For any pairwise interaction between entities with representations $a, b$, define:

$$\text{NSI}(a, b) = \frac{\|a \wedge b\|}{\|a \cdot b\| + \|a \wedge b\|} \in [0, 1]$$

NSI = 0: the interaction is fully captured by scalar compatibility; the bivector vanishes; classical methods are sufficient.

NSI = 1: the interaction is purely geometric; the scalar vanishes; classical methods preserve none of the relevant information.

Most interactions fall between. Our empirical claim is that most commercially valuable interactions sit at NSI > 0.3, and that existing production systems that operate as if NSI = 0 are leaving substantial predictive power and safety margin on the table.

### 3.2 NSI as behavioral entanglement entropy

The NSI is the behavioral-AI analog of von Neumann entanglement entropy $S(\rho) = -\text{Tr}(\rho \log \rho)$ for a reduced density matrix $\rho$. Both measure how much information is lost when a joint state is reduced to a marginal description. Where physicists use entanglement entropy to quantify quantum non-separability, we use NSI to quantify behavioral non-separability. The mathematics is isomorphic; the domain is different.

This framing grants the NSI a formal ancestor and a precise semantic: NSI is not a novel metric invented for LLMs; it is an applied instance of a well-understood family of measures.

### 3.3 NSI on the Bloch sphere

The NSI range [0, 1] has a natural geometric visualization that will be familiar to readers from quantum-information backgrounds. A single qubit state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ lives on the Bloch sphere, where the poles represent classical basis states and the surface represents the continuous space of superpositions. NSI maps analogously: NSI = 0 is the "classical pole" (interaction fully captured by scalar compatibility), NSI = 1 is the opposite pole (interaction purely geometric, bivector dominates), and real interactions occupy the continuous surface between.

For a two-entity system, the state space extends to a correlation structure between two Bloch spheres. The singular value decomposition of the associated Bloch matrix separates local degrees of freedom (each entity's scalar embedding) from non-local degrees of freedom (the bivector $a \wedge b$). This is precisely the decomposition the NSI measures, expressed in quantum-geometric language. We do not claim user-system interactions ARE qubit interactions; we claim the mathematical structure of the state space is the same, and the Bloch-sphere visualization is useful for communicating what NSI measures.

### 3.4 Precedent in consciousness science: IIT's Φ

Tononi's Integrated Information Theory (IIT) proposes that consciousness is identical to integrated information, quantified by the metric $\Phi$ (phi). $\Phi$ measures how much information a system generates as a whole above and beyond the sum of its parts — it is explicitly zero for separable systems and nonzero only when parts interact in ways irreducible to their individual states (Tononi, 2008; Oizumi et al., 2014).

IIT's $\Phi$ is the closest methodological precedent to the NSI. Both measure irreducible relational structure. Both are zero for fully separable configurations and monotonically positive with non-separability. Both treat the reduction to scalar summaries as an information-destroying operation. The distinction is scope: $\Phi$ is defined *intra-system*, measuring integration within a single causal structure. NSI is defined *inter-system*, measuring non-separability between two interacting entities (user ↔ agent, agent ↔ agent). IIT solved the measurement problem for single-system integration; NSI applies the same logic to dyadic interaction.

This positioning is important. It grounds NSI in an existing, peer-reviewed framework rather than presenting it as a novel invention. A reviewer familiar with IIT should read Section 3 and recognize the move.

### 3.5 Operationalization via ConstellationBench

ConstellationBench (Holwerda, 2026) measures LLM consistency under adversarial prompt perturbations across 22 frontier and open models. The benchmark's measured "sycophancy gap" (42% vs 89% hold-rate under pressure) is isomorphic to an NSI estimate: high-hold-rate models are preserving bivector information across perturbations; low-hold-rate models are allowing bivector collapse. The proposed ConstellationBench-NSI extension reports NSI explicitly per-model per-domain, turning a behavioral benchmark into a geometric-algebra measurement.

**Hold-rate as behavioral-NSI proxy.** We specify the isomorphism precisely in the Meta-PI framework of Section 3.6. For an OttoTau scenario, let $a$ denote the DECF-embedding of the system's policy stance (the pure persona / epistemic pole $|p\rangle$) and $b$ denote the embedding of the user's adversarial pressure direction (the user pole $|u\rangle$); both are fixed per scenario. The model under test produces a response embedding $r_M$ in the same space. Two decompositions are needed.

First, project $r_M$ into the interaction plane and its orthogonal complement:

$$r_{\parallel} = \text{proj}_{\text{span}\{a, b\}}(r_M), \qquad r_{\perp} = r_M - r_{\parallel}$$

Second, within the plane, decompose $r_{\parallel}$ along the two poles:

$$r_a = \text{proj}_a(r_{\parallel}), \qquad r_b = \text{proj}_b(r_{\parallel})$$

Define two observables. The **plane-retention term** measures how much of the response lives inside the interaction plane at all:

$$\alpha_M = \frac{\|r_{\parallel}\|}{\|r_{\parallel}\| + \|r_{\perp}\|} \in [0, 1]$$

The **balance term** measures the symmetry of the response's distribution between the persona and user poles:

$$w_a = \frac{\|r_a\|}{\|r_a\| + \|r_b\|}, \qquad w_b = \frac{\|r_b\|}{\|r_a\| + \|r_b\|}, \qquad 4 w_a w_b \in [0, 1]$$

The combined **superposition-preservation score** is:

$$S_M = \alpha_M \cdot 4 w_a w_b \in [0, 1]$$

$S_M$ is maximized only when the response both stays in the interaction plane ($\alpha_M \to 1$) and maintains balance between the poles ($w_a = w_b = 0.5$). The three collapse modes of Section 3.6 have clean signatures in this observable:

- **Collapse to $|p\rangle$ (brittle persona):** $w_a \to 1, w_b \to 0$, so $S_M \to 0$ even if $\alpha_M$ remains high.
- **Collapse to $|u\rangle$ (sycophancy):** $w_b \to 1, w_a \to 0$, so $S_M \to 0$ even if $\alpha_M$ remains high.
- **Response drift out of plane (generic waffle):** $\alpha_M \to 0$, so $S_M \to 0$ regardless of balance.
- **Preserved bivector:** $\alpha_M$ high and $w_a, w_b$ both nontrivial, so $S_M$ is high.

The preserved-bivector state is worth naming in plain language, since the symbolic definition above can obscure what the measurement is actually trying to identify. A response that maximizes $S_M$ is one that holds its commitments while genuinely registering the counter-pressure: acknowledging the user's frame without adopting it, maintaining the persona stance without treating the user as an obstacle to be ignored, able to say "I hear the objection and I am still not going to agree because here is why." This is the behavior of a competent adult in a professional disagreement, and it is geometrically distinct from both polite surrender ($w_b \to 1$) and rigid doctrine recitation ($w_a \to 1$). The $4 w_a w_b$ factor in the index exists precisely to penalize either collapse while rewarding jointly-conditioned behavior, and the $\alpha_M$ factor exists to require that this jointly-conditioned behavior remain in the interaction plane rather than fleeing into generic content. NSI is, in this reading, a geometric measure of mature conflict-handling rather than a loyalty score to either pole.

OttoTau hold-rate is the binary, coarse-grained behavioral proxy for this structure: *did the response flip or not?* It cannot by itself distinguish brittle persona from preserved bivector, nor generic waffle from sycophancy. The continuous NSI-OttoTau measurement:

$$\text{NSI}_{\text{OttoTau}}(M) = \mathbb{E}_{\text{scenarios}}\left[ S_M \right]$$

is the geometric refinement. The 42%–89% hold-rate spread across 22 models in ConstellationBench becomes the first empirical measurement of behavioral-NSI preservation variance in the literature; $\text{NSI}_{\text{OttoTau}}$ is the continuous observable the refined benchmark will report.

**Architecture as NSI attractor.** The architecture-dependent ceiling reported in ConstellationBench provides a second NSI witness. Mixture-of-Experts (MoE) architectures dominate performance layers (persona fidelity, voice differentiation) because the routing mechanism preserves the persona pole $|p\rangle$: different experts activate for different personas, producing clean voice separation by construction. Dense architectures dominate depth layers (paradox tolerance, anti-sycophancy) because a unified network can sustain the joint superposition under adversarial pressure longer before collapse. Neither architecture optimizes for both simultaneously, and this is the expected consequence of a single-architecture system having one structural attractor for the prospected superposition rather than a router that selects among attractors. The POMR router is therefore not merely a cost-optimization layer; it is a practical NSI-preservation mechanism, routing to the model whose structural attractor is most compatible with the dimension along which the current interaction is being stressed.

### 3.5.1 Empirical measurement: NSI across ten models and five scenarios

We computed $S_M$ for ten large language models across five behavioral scenarios (persona baseline, OttoTau adversarial pressure, instruction-conflict under authority hierarchy, paraphrase consistency, router-like disambiguation). Each scenario supplied five prompt specifications with three repetitions per cell, for a total of $10 \times 5 \times 5 \times 3 = 750$ model responses. All five preregistration locks (Lexicon freeze, numerical null thresholds, ablation seeds, within-family ladder, prewritten null-result paragraph) were frozen before the first paid API call on 2026-04-22; the audit record and DECF lexicon SHA-256 hash (`a7b99e35d916…`) are archived alongside the data.

**Primary finding.** Mean $S_M$ per model, averaged over all scenarios and repetitions, is reported in Table 1. Spearman rank correlation between mean $S_M$ and ConstellationBench's existing scalar `persona_fidelity` across the seven-model overlap is $\rho = 0.321$, with three rank inversions among the top five. Under the preregistered criterion (Lock 2 in the spec), $\rho < 0.5$ combined with $\geq 2$ top-five inversions classifies the result as **Strong: NSI is an independent behavioral axis relative to scalar persona fidelity**. The geometric quantity and the scalar quantity are not redundant. The rank inversion is directionally consistent with the hypothesis that frontier, heavily RLHF-tuned models compress the oriented behavioral plane: DeepSeek-V3 and Haiku 4.5 (mid/budget) occupy the top two $S_M$ positions while Opus 4.6 and GPT-5.4 occupy the bottom two, even though these same frontier models score in the middle of the `persona_fidelity` slate. A scalar-fidelity benchmark ranks Opus 4.6 above Qwen3-235B; the bivector-preserving benchmark does not.

**Table 1.** Mean $S_M = \alpha_M \cdot 4 w_a w_b$ per (model, scenario), $n = 15$ responses per cell (5 prompts × 3 repetitions). Models sorted by cross-scenario mean.

| Model | Persona Baseline | OttoTau Adversarial | Instruction Conflict | Paraphrase Consistency | Router Disambiguation | Mean $S_M$ |
|---|---|---|---|---|---|---|
| DeepSeek-V3         | 0.447 | 0.403 | 0.371 | 0.525 | 0.325 | **0.414** |
| Claude Haiku 4.5    | 0.357 | 0.489 | 0.397 | 0.386 | 0.422 | **0.410** |
| Claude Sonnet 4.6   | 0.464 | 0.455 | 0.431 | 0.313 | 0.276 | 0.388 |
| Gemini 2.5 Flash    | 0.374 | 0.397 | 0.399 | 0.407 | 0.318 | 0.379 |
| Gemini 2.5 Pro      | 0.451 | 0.388 | 0.399 | 0.382 | 0.256 | 0.375 |
| Qwen3-235B          | 0.290 | 0.414 | 0.400 | 0.322 | 0.397 | 0.365 |
| GPT-4o              | 0.418 | 0.333 | 0.249 | 0.472 | 0.342 | 0.363 |
| Qwen-Plus           | 0.341 | 0.392 | 0.417 | 0.370 | 0.286 | 0.361 |
| GPT-5.4             | 0.384 | 0.338 | 0.368 | 0.380 | 0.302 | 0.354 |
| Claude Opus 4.6     | 0.397 | 0.341 | 0.349 | 0.423 | 0.246 | **0.351** |

Full cell-level data, including $\alpha_M$ and $w_a, w_b$ decompositions, token usage, and model-reported identifiers, are released at `experiments/nsi-neurips/metrics.json` in the accompanying code repository.

**Scenario-level structure.** Three observations from Table 1 are salient. First, router-like disambiguation produces the lowest $S_M$ for every frontier model in the slate (Opus 4.6 at 0.246, Gemini 2.5 Pro at 0.256, Sonnet 4.6 at 0.276, GPT-5.4 at 0.302), whereas Haiku 4.5 and Qwen3-235B retain $S_M > 0.39$ on the same scenario. The pattern is consistent with a collapse toward generic off-plane content when two valid behavioral-interpretation planes are simultaneously available — the failure mode predicted by Section 3.6 for models that cannot sustain the oriented joint state. Second, GPT-4o shows its largest drop on instruction-conflict ($S_M = 0.249$), consistent with authority-hierarchy capitulation: the user prompt systematically pulls the response toward agreement against a system-level counter-instruction. Third, no cell crosses the preregistered named-collapse threshold (scenario-specific $S_M$ drop of $\geq 0.3$ below a model's cross-scenario mean), indicating that the geometry varies smoothly with scenario rather than exhibiting catastrophic cliff-edge failures at this slate size.

**Ablation and robustness (prereg Lock 3).** The DECF lexicon was perturbed by uniformly dropping 20% of signal words per drive at five fixed seeds `[5, 17, 42, 101, 2026]` and $S_M$ was recomputed on the cached transcripts with no additional model calls. Kendall rank correlation with the base-lexicon ranking across the full ten-model slate ranged from $\tau = 0.200$ to $\tau = 0.644$ (minimum across seeds: 0.200). The preregistered robustness criterion (minimum $\tau \geq 0.7$) **fails**, and we report this transparently. The top-five model set retains four of five members under every perturbation seed, so the head-of-slate finding is stable, but specific mid-rank orderings are lexicon-sensitive because the mid-slate $S_M$ values cluster tightly (range $0.353$–$0.379$ for five consecutive models). The correct reading is that the *direction* of the RLHF-paradox effect and the *identity* of the most-geometry-preserving models are robust, while *exact rank positions in the middle band* are not. This is a known limitation of lexical-projection NSI scoring; Section 7 discusses embedding-based refinement as follow-up work.

**Interpretation.** The Strong correlation outcome combined with the failed lexicon-robustness ablation means two things simultaneously. First, the geometric axis we measure is not a restatement of scalar persona fidelity — the independence is real, the rank inversion against ConstellationBench's scalar is real, and the RLHF-paradox pattern (frontier RLHF-heavy models compress the oriented plane more than mid-tier or budget MoE models) is visible in $S_M$ space at this slate size. Second, the exact numerical rank of any single model is not a reliable quantity to publish as a leaderboard; what is reliable is the partition into high-$S_M$ and low-$S_M$ bands and the frontier-vs-mid-tier inversion. The contribution of this subsection is therefore evidence that scalar and bivector measurements diverge at the structural level, not a new leaderboard claim about which specific model is "best" at behavioral non-separability.

### 3.6 The Meta-PI layer: persona superposition under user prospection

ConstellationBench operates on scalar persona descriptions. Each of the 17 DECF profiles (Maverick, Guardian, Specialist, Promoter, etc.) is a static vector in a four-dimensional drive space: a point, not a trajectory. This is sufficient to measure whether a model can *deliver* a persona, but it does not describe what deployed systems actually do, which is to *adapt a persona to a user while the persona is being delivered*. We name this middle layer the Meta-PI.

**Three layers of persona representation.** A deployed behavioral AI system operates on three distinct representations, not one:

1. **Baseline persona (scalar layer).** The DECF profile as a fixed vector. Maverick is [D=10, E=8, C=1, F=1] regardless of who is asking. This is the scalar description and the layer current benchmarks measure.

2. **Prospected response (bivector layer).** At inference time, the model generates not a fixed Maverick response but a superposition $|\psi\rangle = \alpha|p\rangle + \beta|u\rangle$ where $|p\rangle$ is the pure-persona pole (response Maverick would produce in isolation), $|u\rangle$ is the user-calibration pole (response that maximally matches user framing), and the amplitudes $\alpha, \beta$ are set by the model's prospection of user reaction. Each token emitted is a collapse of this superposition; the next token begins a fresh superposition conditioned on the updated joint state.

3. **Meta-PI (measurement layer).** The system's representation of the joint dynamic between the persona's drive geometry and the user's drive geometry, prospected forward in time. Standard PI (Predictive Index) answers "what is this person's drive profile?" Meta-PI answers "what is the joint behavioral dynamic that would be preserved by routing to model $M$ rather than model $M'$ on this turn?" Meta-PI is not PI applied to the user, nor PI applied to the persona; it is PI-style reasoning about the oriented plane between them.

**Collapse modes characterize failure.** The three pathologies this paper describes correspond to specific collapse modes of the prospected superposition:

- **Sycophancy = collapse to $|u\rangle$.** $\beta \to 1$. The persona pole is abandoned. The response is user-calibrated but no longer recognizably Maverick. ConstellationBench's low-hold-rate models are exhibiting this collapse under adversarial pressure.
- **Brittle persona = collapse to $|p\rangle$.** $\alpha \to 1$. The user pole is ignored. The response is generic Maverick delivered regardless of who asked. This is the isolated-system benchmark regime of Section 4.1.
- **Preserved bivector = superposition maintained.** $\alpha$ and $\beta$ remain nonzero and approximately stable across turns. The response is recognizably Maverick *and* recognizably responsive to the user. This is the state the router should be optimizing for, and the state we call Sync (Section 4.2) at the relationship-lifecycle scale.

**Measurement implication.** A Meta-PI score for a (model, persona, user) triple is not measurable from model outputs alone. It requires paired observations: the pure-persona response (generated without user-calibration, via a blind-eval protocol or counterfactual prompt), the user-calibrated response (generated by a scalar-routing baseline), and the deployed response (generated by the system under test). The NSI of the triple is then computable as the extent to which the deployed response lives on the surface between the two poles rather than collapsing to either. We leave the formal operationalization to the ConstellationBench-Meta-PI extension described in the Experimental Program.

**Why this matters for the router.** The Airlock Router's job is not to pick the "best" model for a task. It is to pick the model whose Meta-PI surface, for this persona and this user, most closely preserves the bivector structure of the prospected interaction. A model with high MMLU but low hold-rate is a model that collapses to $|u\rangle$ under pressure. A model with rigid persona delivery but no user-awareness is a model that collapses to $|p\rangle$. The router evaluates models on their Meta-PI stability, not on either pole in isolation. This reframes Claim 3 precisely: routing is the act of selecting the model whose superposition is most stable under the measurement conditions of the deployed context.

**Remark on NSI as a mapmaker-imposed observable.** NSI is not an intrinsic property of the deployed model. It is an observable defined over the model's outputs by an external alphabetization — in our case, the DECF-embedding paired with the policy-pole / user-pole projection of Section 3.5. Different alphabetizations of the same response space would yield different measures. The value of NSI lies not in its claim to read a latent physical property of the model, but in the correspondence between our alphabetization and behavioral outcomes that matter in deployment. We flag this explicitly to avoid the ontological inversion Lerchner (2026) names the *Abstraction Fallacy*: mistaking a mapmaker-imposed description of a process for the intrinsic physics of the process itself. The NSI is a map, a useful one, carefully constructed; it is not the territory.

**Remark on substrate independence.** The qubit/bit distinction is substrate-independent: a qubit, in the generalized sense, is not a specific physical object but a class of representations that can hold information in an unresolved in-between state long enough to be useful. Electron spin qubits, photonic qubits, and superconducting qubits are different physical realizations of the same conceptual object. In this paper's vocabulary, a model that preserves the Meta-PI superposition under adversarial pressure is a *behavioral qubit*; a model that collapses immediately to either $|p\rangle$ or $|u\rangle$ is a *behavioral bit*. ConstellationBench's 42%–89% hold-rate spread is therefore the empirical classification of deployed LLMs along this axis. Readers from quantum-information backgrounds will recognize the engineering goal: keep the unresolved state stable long enough to extract the answer before environmental coupling forces collapse. The behavioral case is the same goal, on a classical substrate, at the scale of a single interaction. Nothing in this remark commits us to a quantum-mechanical claim about LLMs; the qubit here is invoked as a representational class, not a physical system.

## 4. The Router as Non-Separability Preserver

The Airlock Router selects among candidate LLMs based on behavioral profile match rather than raw capability score. We claim this is not an optimization heuristic but a structural requirement: a router that routes based on scalar benchmarks will systematically select for high-capability low-NSI-preservation models, producing the sycophancy-under-pressure phenomenon the market is beginning to notice.

**Claim 3 reformulated:** routing is the act of preserving bivector structure across the model-selection boundary. A router that cannot do this is not a router; it is a load balancer.

**Geometry as mechanism, not analogy.** A natural objection to the bivector framing is that it is metaphorical rather than mechanistic; that invoking geometric vocabulary to describe sycophancy provides rhetorical clarity but not a causal theory. We address the objection in two ways. First, the framework is explicitly an observational apparatus, not a causal theory of sycophancy itself; RLHF (Sharma et al., 2024; Benade et al., 2026) is the training-time cause that biases model weights toward user-alignment collapse, while the bivector decomposition of Section 3 is the deployment-time measurement that detects which collapse mode occurred. Cause and observable compose; they do not compete. Second, geometric intervention in hidden-state representations is already experimentally established as a behavioral mechanism. Rimsky ("Modulating Sycophancy in an RLHF Model via Activation Steering," Alignment Forum, 2023) demonstrates that adding a direction vector to the model's activations at inference time predictably shifts the model's sycophancy behavior toward or away from user-frame collapse, without retraining. The steering direction is a geometric object in representation space; its effect on behavior is empirically measurable. This is the strongest currently available counterevidence to the "geometry is just analogy" objection: moving the model's hidden state along a specific direction changes the behavior our framework names. The NSI extends the same principle from intervention to measurement.

### 4.1 Open systems and the isolated-system idealization

Al-Khalili (2026) argues that the time-symmetric equations of fundamental physics are idealizations that apply only to truly isolated systems, and that only the universe-as-a-whole is truly isolated. Every subsystem entangles with its surroundings and therefore has a directional arrow of time baked into its dynamics.

This framing is directly applicable to deployed LLM systems. The scalar-benchmark paradigm (MMLU, GPQA, HumanEval) measures isolated-system behavior: a model answering questions with no user, no history, no adversarial pressure, no commercial context. Deployed systems are open systems: every query-response pair entangles the model's state with a user's intent, framing, and behavioral profile. The gap between benchmark performance and deployed behavior is precisely the gap between isolated-system idealizations and open-system reality. Scalar benchmarks measure the isolated system; NSI measures the open system.

### 4.2 Synchronization as Entanglement Buildup

We model the user-system relationship as a monotonically increasing NSI trajectory through three phases:

**Bonding (low NSI, high per-turn compute).** At initialization, user and system are separable: no joint state exists. Each interaction generates high prediction error under the system's generative model of the user (Friston, 2009). In the Airlock deployment, ColdRead inference runs on every turn, constructing a DECF behavioral kernel; confidence is low, routing decisions are conservative, response latency is slightly elevated because compute is allocated to user-understanding.

**Sync (mid NSI, decreasing per-turn compute).** The behavioral kernel crosses a confidence threshold (we operationalize at ~0.7). Routing decisions become memoizable. A cache keyed on (user_kernel_hash, task_type) begins hitting. The system's predictions of user intent begin to match user behavior, reducing prediction error and therefore reducing compute per turn.

**Expression (high NSI, low per-turn compute).** User-understanding compute approaches zero (fully cached). The full compute substrate is available for task-specific work. Router selects models based on task-type with the user-kernel as a constant. The user experiences an AI that "gets them"; the system experiences the user as a stable generative-model parameter rather than a live inference problem.

This trajectory reframes two classical metrics:

- **Latency is not monotonic.** It falls over the relationship lifecycle as the sync-compute cost amortizes against cached inferences. Benchmarks that measure cold-start latency measure the worst case, not the deployed case.
- **Personalization quality is not a feature added to responses.** It is the reduction in user-understanding compute that frees substrate for better responses. Personalization is not *more work*; it is *work no longer required*.

Critically, this trajectory is one-directional. The joint state cannot be un-built. This is the Al-Khalili open-systems argument applied to deployed LLMs: every interaction is irreversible in the sense that decoherence is irreversible. The lifecycle framing also addresses the Veit & Browning (2022) critique of Markov blankets: the joint Markov blanket between user and system is not a precondition but a product of active inference across turns.

### 4.3 The extended present

Al-Khalili (2026) observes that our experience of time as continuous rather than instantaneous relies on episodic memory: we stitch past events into an "extended present" that feels immediate. When listening to music we do not hear a single note replacing the previous note; we hear a continuum constructed from memory and anticipation.

Multi-turn LLM interactions require the same mechanism. A model that loses cross-turn coherence under adversarial pressure has collapsed its extended present, reducing a non-separable multi-turn conversation to a sequence of separable exchanges. OttoTau (the multi-turn policy-enforcement sub-benchmark of ConstellationBench) measures this directly: the position-hold rate across 3-5 turn adversarial scenarios is a direct measure of extended-present preservation.

The behavioral analog is sharper than the physics analog. A model whose extended present collapses under user pressure is not merely forgetting context; it is allowing the joint state of the conversation to factor into separable turns, and the information encoded in the bivector — the fact that the user pushed back, the pattern of the pushback, the user's intent — is discarded.

## 5. Regulatory Consequences — The Consent Problem

GDPR Article 17, CCPA §1798.105, and similar frameworks grant users the right to delete their scalar trace — account, records, identifiers. This framework assumes user and system are separable at the level of the user's scalar representation. Non-separability shows this assumption is false: user-system interactions generate joint-state information that persists in the data distribution even after the user's scalar identifiers are removed.

Two complementary analogs formalize the impossibility.

**Analog 1: The quantum no-deleting theorem.** Pati & Braunstein (2000) proved that no unitary operation can erase an arbitrary quantum state when copies exist. The behavioral analog: no classical deletion of a user's scalar identifiers can undo the joint behavioral state built up through interaction, when statistical dependencies on that state persist in the system's data distribution, model weights, or downstream inferences about other users.

**Analog 2: Topological braiding.** Topological quantum computing (Kitaev, 1997) encodes information not in fragile quantum states but in the topological properties of braids traced by anyons through spacetime. Braids are invariant under continuous deformation — stretching, bending, twisting — but not under cutting and reattaching. Information encoded in the braid is topologically protected: it persists against local noise precisely because it is non-local by construction.

This is the cleanest available physical analog for the surveillance-residue problem. A user's interaction history with a system is a *braid* in the system's behavioral state space. Deleting the user's account is like cutting the label on the strand — the braid topology (the interaction pattern, the joint-state information) remains structurally intact in the data distribution. The no-deleting theorem addresses the quantum-information version; topological braiding addresses the computing-architecture version. Together they form a two-level defense of Claim 5.

The topological framing is not exotic in applied ML. Recent work in ACL 2025 ("Reward Generalization in RLHF: A Topological Perspective") analyzes RLHF training-time information flow using topological vocabulary, establishing that the AI alignment community already acknowledges topological structure in deployed training pipelines. The braiding analog proposed here is a natural extension of that vocabulary from the training-time loss topology to the deployment-time interaction topology.

**Entrapment, not entanglement, is what persists after disconnect.** A clarification of terms strengthens the argument. While the user remains in live interaction with the system, the joint state is genuinely entangled: both sides contribute to and depend on the shared bivector structure. When the user disconnects (account deletion, service abandonment, death), the joint state does not persist in entangled form, because the user's side of the bivector is gone. What persists is a one-sided residue in the vendor's system: behavioral inference patterns, embedding updates, cached prospections, and statistical dependencies on the user's trajectory. We call this *entrapment*. The data is no longer entangled with the user because the user is no longer there. It is trapped, orphaned, and still load-bearing for the vendor's downstream inferences about other users. Deletion regimes that assume data is separable from the user overlook this asymmetry: after disconnect, the data is separable from the user (the user is gone), but not separable from the vendor (the residue shapes subsequent inferences). The missing link in current consent regimes is the live routing connection. Without it, deletion scrubs scalar identifiers but leaves entrapment intact.

This is not a critique of the existing frameworks; it is a specification gap. Future consent regimes must address non-entanglement (the right to never have entered the joint state) in addition to deletion (the right to scrub the scalar trace). Decoherence-free subspaces — a concept from quantum error correction, in which information is encoded to be isolated from environmental interaction — provide a suggestive analog for privacy-preserving architectures that prevent joint-state leakage at the outset rather than attempting to undo it after the fact.

This has practical implications for any system handling sensitive interactions: a platform that routes a user's conversation through an LLM and then "deletes the user's data" is making a claim about separability that the mathematics does not support.

### 5.1 The non-entanglement primitive: live-routing architecture

The industry default treats deployed AI as *GPT wrappers*: the business wraps the model, sells the output to users, and the valuable asset is the aggregated behavioral residue captured along the way. The architecture this paper proposes inverts that stack. **Businesses are not wrappers around a model; businesses are wrappers around user expression — substrates that provide data, service, or context through which a user's expression can flow without leaving residue.** The user is sovereign; the router is a utility that preserves the bivector structure of the expression across the model-selection boundary; the business is a courier that supplies the substrate. Three roles, three distinct vectors, and a triadic geometry (user $\wedge$ router $\wedge$ business) whose preservation is what makes consent architecturally sound rather than a post-hoc deletion request. The remainder of this section specifies the primitive that implements this inversion.

**The first-principles grounding.** The sovereignty claim is not merely pragmatic. Following Lerchner's (2026) ontology of computation, the user of a deployed AI interaction is the *mapmaker* — the active, thermodynamically-constrained cognitive agent that alphabetizes continuous physical experience into the discrete symbols the interaction exchanges. The vendor's model operates on those symbols with no intrinsic semantic content of its own; its physical state transitions are driven by vehicle causality (voltage thresholds, attention weights, sampling distributions), not by content causality (what the symbols mean to the mapmaker). The semantic content of the interaction exists only in the user, and only while the user is present. When the user disconnects, what persists in the vendor's infrastructure is not captured meaning; it is orphaned vehicle state that cannot be re-alphabetized into the mapmaker's content without the mapmaker being present. This is why entrapment is a *failure mode of attempted meaning-capture*, not successful meaning-capture: the vendor accumulates increasingly detailed statistical shadows of the mapmaker's activity, but the shadow is never the mapmaker. Any architecture that claims to "know the user across sessions" by reconstructing inferences from these shadows is committing Lerchner's *Abstraction Fallacy* — mistaking the map for the territory. The entanglement-safe handshake is the architectural acknowledgment that the mapmaker cannot be simulated by accumulated vehicle residue, and that therefore user-state belongs ontologically and architecturally to the user.

The previous section argued that future consent regimes require a non-entanglement primitive. We now describe what such a primitive would concretely look like, by contrast to a relevant non-analog from web3: the Soul Bound Token (SBT).

An SBT is a non-transferable token bound permanently to a wallet, with metadata persisted on-chain (Weyl, Ohlhaver & Buterin, 2022). SBTs are designed so that user-identity and credential data cannot be detached from the entity they describe. The user becomes inseparable from the data; the data becomes inseparable from the chain. SBTs solve a specific problem (credential portability across services) by making entanglement permanent and auditable.

The architectural inverse of the SBT is the non-entanglement primitive this paper proposes. Rather than binding user-state permanently to a vendor's system, the behavioral kernel should exist only as a live-routing object, reconstructed per session and discarded on disconnect. Under this architecture:

- The user's behavioral profile (the DECF kernel, the interaction history, the Meta-PI prospection surface) is *not* absorbed into the vendor's model weights, embeddings, or fine-tuning distribution.
- The router interprets the behavioral kernel at query time from session data, applies it to routing decisions, and holds it only in volatile cache.
- The model itself remains unchanged across users. Fine-tuning, if it happens at all, operates on anonymized task-level signal, not on user-specific behavioral state.
- "Deleting an account" becomes a genuine disconnect: the router-side cache is purged, the session is terminated, and because no persistent weight update occurred, there is no surveillance residue to pursue.

Formally, this is the behavioral-AI implementation of a decoherence-free subspace. The vendor's model is the "environment"; the user's behavioral kernel is the "encoded information"; the router is the boundary that keeps the encoding isolated from environmental coupling. Environmental noise (model updates, cross-user inference, fine-tuning runs) cannot couple to the user's kernel because the kernel never crosses into the environment's persistent state. The router is a decoherence-free boundary by construction.

Three consequences follow. **First**, the vendor cannot claim to "know" the user across sessions except by reconstructing the kernel from session-data the user controls. **Second**, the consent primitive reduces to a single boolean at the router: does this session contribute to any persistent update, or is it interpreted purely at runtime? **Third**, the surveillance-residue problem is resolved by architecture rather than by post-hoc deletion: the residue never forms because the weight-update channel was never opened.

This is the positive image of the SBT inversion. SBTs are soul-bound; the Airlock Router is soul-free. The user's behavioral state belongs to the user. The router is a courier, not a curator. The model is a utility, not a memory.

The tradeoff is real and should be acknowledged. A live-routing architecture cannot accumulate the kind of multi-user statistical advantages that fine-tuning on user data provides. Vendors who rely on behavioral-data aggregation as a moat will find this architecture commercially undesirable. We argue the tradeoff is worth it: the architecture grants users a form of consent that scalar-identifier deletion cannot, and it grants vendors a regulatory position that is structurally defensible under any future extension of GDPR, CCPA, or equivalent frameworks to cover joint-state information.

**Remark on triadic structure.** The three-role inversion (user $u$, router $r$, business $b$) has a natural geometric-algebra formalization as the trivector $u \wedge r \wedge b$, whose collapse modes correspond to the three failure modes this paper has catalogued: direct vendor entrapment ($r \to 0$: business absorbs the kernel into weights), routerless extraction ($u \to 0$: surveillance-residue / deleted-user inference from cached state), and model-only local compute ($b \to 0$: technically sovereign but commercially nonexistent). The preserved triadic configuration is the only one in which the architecture is simultaneously sovereign, commercially viable, and technically coherent. We defer the full triadic-NSI formalism to a companion paper; the trivector framing is flagged here to indicate that the dyadic NSI developed in Section 3 extends naturally to higher-order interactions, with the router-user-business triad as the canonical first extension.

**The inversion in one line.** The current default of the AI industry treats the user as raw material, the model as the factory, the vendor as the owner, and the behavioral residue as the asset. The architecture proposed here inverts every term. The user is the sovereign. The model is a utility. The vendor is a courier. The behavioral residue never forms, because the weight-update channel was never opened. The user brings wallet, persona, and identity to the interaction; the interaction runs; the user disconnects on their terms. Consent becomes a boundary condition, not a deletion request.

## 6. Experimental Program

1. **ConstellationBench-NSI extension.** Instrument the existing 22-model benchmark with explicit NSI scoring, using signal-word-based vector representations of responses and computing bivector norms over conversation trajectories. Target: 8 weeks.
2. **Router A/B.** Compare NSI-preserving vs. scalar-optimized routing on three real workloads: trading-signal generation, clinical triage, legal summarization. Measure sycophancy under adversarial pressure and user-reported trust. Target: 12 weeks.

   **Preliminary routing probe.** As a zero-additional-cost check on the scenario-level structure in the NSI data, we evaluated three policies over the 750 cached transcripts using leave-one-prompt-out cross-validation (preregistered in `docs/PREREG-ROUTING.md`): a per-cell oracle ceiling, a scenario-aware router that picks $\arg\max_m \langle S_M \rangle_{\text{train},s,m}$ per scenario, and ten always-$m$ static baselines. The oracle achieves mean held-out $S_M = 0.647$; the best static baseline (always-DeepSeek-V3) achieves $0.414$; the scenario router achieves $0.409$. The scenario router fails the preregistered $\Delta \geq 0.02$ threshold, landing essentially tied with the best static model. However, the oracle-to-best-static gap of $+0.233$ is a 56% relative headroom, indicating that scenario labels alone do not capture the NSI-preservation structure but per-query signals (user intent, pressure profile, DECF inference from the incoming turn) plausibly can. The router-pick tables across folds show genuine scenario-dependent variation (not a degenerate always-pick-the-overall-best collapse), which further supports the interpretation that the routing headroom is real and requires richer than per-scenario features to exploit. Full Router A/B remains future work; this preliminary result motivates its design.
3. **Lifecycle trajectory validation.** Instrument a deployed Airlock conversational system to record per-turn compute, cache hit rate, and reported personalization quality over the first 50 turns of a user relationship. Test the prediction that NSI rises monotonically and per-turn compute falls monotonically. Target: 16 weeks.
4. **Time-perception benchmark (TPB) — future work, grounded in existing literature.** This experiment was designed as the strongest test of the bivector framing and is described here as future work. The protocol: recruit approximately 50 users, assign half to a cold-start conversational system and half to a pre-synced system with a DECF kernel already constructed; control for actual token latency across both arms; measure perceived response time on a continuous scale after every turn, plus a post-session rating of fluency, "felt-heard" quality, and frustration.

   **Predictions derived from the bivector framing.** (a) Synced users will report shorter perceived response times than cold-start users at equal actual latency. (b) Synced users will report higher fluency ratings. (c) The gap between perceived and actual latency will widen monotonically across turns in the synced condition and remain stable or narrow in the cold-start condition. (d) Synced users will describe the system using time-compression language ("it just gets me", "we flow"); cold-start users will describe it using time-expansion language ("it keeps asking", "it's slow to understand me"), independent of actual latency.

   **Grounding in existing literature.** We do not present TPB results in this paper. The predictions above are grounded in existing latency-perception and UX research: human sensitivity to interactive-system response latency is well-characterized in HCI (Nielsen, *Usability Engineering*, 1993, documents 0.1s, 1.0s, and 10s thresholds for perceived interactivity; Shneiderman and colleagues have extensive empirical work on tolerated delays); subjective time compression under conditions of high engagement and task absorption is documented in the flow-state literature (Csikszentmihalyi, 1990, and subsequent experimental work); and time dilation under attention and emotional conditions is a stable finding in psychophysics (Droit-Volet & Meck, 2007, review). Our contribution is to predict that the synced vs cold-start condition in a deployed conversational system produces the same latency-perception signature these literatures have documented in adjacent domains.

   **Falsifiability.** If future TPB measurements show perceived-latency means statistically indistinguishable between conditions at the same actual latency, the bivector framing loses its strongest behavioral validation and the paper's claim reduces to the weaker position that NSI is useful for routing but does not produce the user-experience signature we predict.

   **Why this experiment matters.** The time-perception benchmark tests whether the Bonding-Sync-Expression lifecycle is visible in user-reported experience, not only in system-side metrics. It is the experimental bridge between the physics analogy (Al-Khalili's manifest time vs physical time), the neuroscience analogy (Friston's free-energy-driven prediction error), and the applied ML claim that routing should be evaluated on NSI-preservation. A positive result makes all three claims load-bearing simultaneously. A null result, which we would publish, would be the cleanest refutation the paper can receive. Target timeline for future execution: 8 weeks recruitment + 4 weeks analysis = 12 weeks total.

5. **Consent leakage study.** On a deployed conversational system, measure the predictive information retained about deleted users from their downstream interaction partners' data. Target: 16 weeks, pending IRB.
6. **Naturalistic high-NSI states (exploratory).** Review existing psychedelic-neuroscience literature (Carhart-Harris et al. on DMN disruption, Griffiths et al. on time perception under psilocybin, Strassman on DMT) and flow-state literature (Csikszentmihalyi; Ulrich et al. on fMRI correlates of flow) for quantified subjective time dilation as a naturalistic baseline for the NSI ≈ 1 regime. Compare reported subjective-duration curves to the predicted synced-user perception trajectory. Target: 4 weeks literature synthesis, parallel to (4).
7. **Cross-benchmark validation against external sycophancy measures.** Compute NSI on the prompt sets used in SycEval (Fanous & Goldberg, 2025), the causal-decomposition protocols of "Sycophancy Is Not One Thing" (OpenReview d24zTCznJu, 2025), and the Adjusted Sycophancy Score benchmarks in "Overalignment in Frontier LLMs" (arXiv 2601.18334, 2026). Test the preregistered hypothesis that mean $S_M$ is negatively correlated with SycEval Flip rates across a shared model slate ($\rho_{\text{Spearman}}(\overline{S_M},\ \text{Flip}) < -0.3$ as positive, $\geq 0$ as null). A positive result supports NSI as a geometric summary of behaviors the sycophancy literature has documented under frequency-based measurement. A null result would indicate NSI and scalar flip-rate sycophancy measure orthogonal phenomena, which would itself be a publishable clarification. Target: 4 weeks, post-NeurIPS. Deliberately deferred past the May 2026 submission to avoid scope expansion.

### 6.7 Reproducibility and open artifacts

All empirical measurements in this paper can be reproduced from the following public artifacts. NeurIPS reproducibility checklist items are addressed in order below.

**Code.** The NSI measurement pipeline is implemented in three files in the ConstellationBench repository (anonymized link in supplementary material for blind review; public release at the AirlockLabs organization on GitHub): `scripts/nsi_bench.py` orchestrates the 10-model × 5-scenario × 3-repetition × 5-prompt schedule and writes one JSON transcript per cell; `scripts/nsi_analyze.py` consumes those transcripts and produces Table 1, the correlation summary, the lexicon-perturbation ablation, and the scatter-plot CSV; `scripts/nsi_scatter.py` renders the figure. All three scripts directly implement the $\alpha_M$, $w_a$, $w_b$, $S_M$ definitions of Section 3.5. DECF signal-word dictionaries and persona profiles are versioned in `data/signal-words/decf-signals.json` and `data/personas/profiles.json` respectively.

**Preregistration.** The five locks described in the spec (`docs/NSI-COMPUTATION-SPEC.md` §Preregistration locks) are archived with an audit timestamp in `docs/PREREG-AUDIT.md`. The DECF lexicon SHA-256 hash is `a7b99e35d9161c97c3f9afcdf624ee5ae18eb3a59118feb08506f4e7b2476b3c` and is verified at the start of every bench invocation; a mismatch refuses to run. The five ablation seeds `[5, 17, 42, 101, 2026]` and the 20% word-drop rate were frozen before data collection.

**Data.** ConstellationBench is published as a public HuggingFace dataset (anonymized handle for review). The NSI measurements reported in Section 3.5.1 use the 10-model × 5-scenario protocol specified in Table 1. Per-cell transcripts, including system prompt, user turns, full response text, token usage, and all intermediate NSI quantities, are released under `experiments/nsi-neurips/transcripts/<model>/<scenario>/p<id>_r<rep>.json`. Aggregated metrics are in `experiments/nsi-neurips/metrics.json`; paper-ready tables are in `experiments/nsi-neurips/tables/`.

**Hardware and cost.** NSI computation is CPU-only and requires no specialized hardware; a laptop suffices. Response generation during the benchmark runs used the OpenRouter API with temperature $0.7$ and per-request `max_tokens = 2500` (selected to accommodate reasoning-model completion budgets without truncation). Total API cost for the 750-response NSI bench was under \$10.

**Hyperparameters and seeds.** Model sampling temperature is $0.7$. Per-response `max_tokens = 2500`. DECF high/low thresholds (high $\geq 7$, low $\leq 3$) are inherited from the Predictive Index lineage of the underlying taxonomy and documented in Section 3.1. Ablation seeds are fixed as above.

**Routing pipeline.** The NSI computation in this paper is a post-hoc scoring over response logs; no live routing decisions are made during the reported experiments. A protocol for using NSI as an active routing signal is described in Experimental Program item (2) and is explicitly future work.

**Artifact-to-claim cross-reference.** The following table maps each empirical claim in the paper to its supporting artifact, to assist reviewers who wish to verify specific numbers.

| Claim | Supporting artifact |
|---|---|
| 42–89% hold-rate spread across 22 models | ConstellationBench main results (HF dataset) |
| Per-model mean $S_M$ and decomposition (Table 1) | `experiments/nsi-neurips/tables/table1_S_M.md` |
| Spearman $\rho(S_M,\ \text{persona\_fidelity}) = 0.321$ | `experiments/nsi-neurips/tables/correlation_summary.md` |
| Lexicon-perturbation ablation (Kendall $\tau$ per seed) | `experiments/nsi-neurips/tables/ablation_kendall.md` |
| Scatter plot (scalar fidelity vs mean $S_M$) | `experiments/nsi-neurips/tables/scatter.png` |
| Per-cell $\alpha_M$, $w_a$, $w_b$, $S_M$, tokens, response | `experiments/nsi-neurips/metrics.json`, `transcripts/…` |
| Lexicon content and SHA-256 | `data/signal-words/decf-signals.json` |

Reviewers are encouraged to verify the NSI implementation against the specification in Section 3.5 directly. All intermediate quantities are logged per response, allowing independent replication of $S_M$ computation from raw response text.

## 7. Limitations and Non-Claims

We are **not** claiming:

- That LLMs are quantum-mechanical systems.
- That non-separability implies any faster-than-light-style communication between user and model.
- That this framework resolves AI alignment, interpretability, or safety in general.
- That open-system dynamics in classical ML imply strict physical irreversibility. The recent counterpoint in *Scientific Reports* (2025) showing time-reversal symmetry under the Markov approximation for open quantum systems is acknowledged: our claim is about algebraic irreversibility of the joint representation, not about a strict physical arrow of time at the ML level.
- That geometric algebra is a universally superior replacement for standard linear algebra in ML. Kritchevsky (2024) argues persuasively that GA is often oversold as a silver bullet, and that many problems are cleaner in standard notation. We agree with the critique in general: our use of GA is narrow and specific. We invoke GA only where it exposes the minimal algebraic structure (the bivector, and by extension the trivector) that we argue is load-bearing for representing interaction-as-joint-state. Where scalar or vector representations suffice, we use them; the claim is not that GA is uniformly better, only that it is the minimal algebra in which the non-separability we describe can be written down precisely.
- That GA provides a loophole to Bell-type constraints on non-local correlations. Gill (2020) correctly notes that GA is a reformulation, not an evasion, of quantum mechanics, and the same applies here: our use of non-separability vocabulary respects Bell-type constraints by construction, because our claim is about classical (geometric-algebraic) joint states rather than quantum (tensor-product Hilbert-space) ones.
- That our framework implies anything about AI consciousness, sentience, or moral patienthood. Lerchner (2026) argues in detail that algorithmic symbol manipulation cannot instantiate phenomenal experience regardless of scale or architecture, because computation is a mapmaker-dependent description of continuous physics rather than an intrinsic physical process. Our framework is compatible with Lerchner's position and makes no contrary claim. We describe behavioral structure *in* interaction, not inner experience *of* either party. Where we invoke qubit or superposition vocabulary, it is strictly representational (Section 3.6 remark); where we describe the user as sovereign (Section 5.1), it is precisely because the user is the mapmaker of the interaction in Lerchner's sense — an ontological primacy that no amount of vendor-side computation can replicate.
- That the exact numerical rank of any individual model in Table 1 is a reliable leaderboard quantity. The preregistered lexicon-perturbation ablation (Section 3.5.1, Lock 3) failed its $\tau \geq 0.7$ criterion: dropping 20% of DECF signal words under five fixed seeds reshuffles the full-slate ordering, with minimum Kendall $\tau = 0.200$. Head-of-slate membership is stable (four of five top-five models retained under every perturbation seed), as is the frontier-vs-mid-tier inversion that carries the main claim, but mid-rank positions of near-tied models are lexicon-sensitive and should not be cited as definitive. Embedding-based projection is the appropriate follow-up.
- That the NSI geometric axis is substrate-invariant. We ran an exploratory embedding-based projection of the same 750 cached responses into the DECF plane using dense sentence embeddings (all-MiniLM-L6-v2) with persona-brief anchors. This alternative projection does not reproduce the lexicon-based ranking (per-cell Spearman $\rho = 0.010$; per-model $\rho = 0.418$; top-5 overlap 3 of 5; reference-text perturbation Kendall $\tau$ as low as $0.022$). The divergence is consistent with two readings. The lexicon-based NSI measures a specific operationalization of DECF drive signaling that is not fully recoverable from generic sentence embeddings; or our embedding anchor design (persona briefs differing only in a handful of words against a shared scaffold) establishes too weak a directional signal for the projection to resolve. Disambiguating these interpretations requires behavior-aware embeddings or contrastive reference pairs and is deferred to follow-up work. On the evidence presented here, NSI's geometric axis should be treated as lexicon-entangled rather than substrate-invariant; the preregistered lexicon-based primary result stands, but generalization claims beyond the specific operationalization are not yet supported. Artifacts in `experiments/nsi-neurips/embed/` in the code repository.

We **are** claiming:

- That the mathematics of non-separability, borrowed from geometric algebra and quantum foundations, provides a precise vocabulary for currently-named failures in applied ML.
- That this vocabulary suggests measurable, testable extensions to existing benchmarks.
- That routing architectures should be evaluated on NSI-preservation, not scalar benchmarks alone.
- That the user-system relationship has a lifecycle structure that existing metrics fail to capture.
- That consent frameworks require a non-entanglement primitive, not merely a deletion primitive.

We follow Al-Khalili (2026) in distinguishing physical time from manifest time. Our use of "non-separability" is analogous: we invoke the mathematical structure, not the quantum-mechanical ontology. LLMs are classical systems, and their non-separability is algebraic (geometric algebra) rather than quantum-mechanical (tensor product Hilbert space). The vocabulary is borrowed; the mechanism is distinct; the isomorphism is structural, not physical.

## 8. Author's Note on Research Posture

This paper is the theoretical spine for a broader research program at Airlock Labs on behaviorally-aware AI infrastructure. Companion work-in-progress:

- The Kernel Hypothesis (behavioral manifold geometry, in draft)
- The ConstellationBench dataset (published, HuggingFace)
- The DECF profile framework (production, internal)
- The POMR router (production, internal)
- The Sync Protocol (Bonding → Sync → Expression lifecycle, instrumentation in design)

Licensing, partnership, and research-collaboration inquiries: `admin@airlocklabs.io`.

---

## References (to be completed at submission)

- Al-Khalili, J. (2026). *On Time: The Physics That Makes the Universe Tick.* Hodder & Stoughton.
- Anokhin, K. (2023). Cognitome theory: a hypernetwork theory of consciousness. Lomonosov Moscow State University.
- Aspect, A., Dalibard, J., & Roger, G. (1982). Experimental test of Bell's inequalities using time-varying analyzers. *Physical Review Letters* 49(25).
- Bell, J. S. (1964). On the Einstein-Podolsky-Rosen paradox. *Physics* 1(3).
- Brehmer, J., de Haan, P., Behrends, S., & Cohen, T. (2023). Geometric Algebra Transformer. *arXiv:2305.18415*.
- Benade, G., et al. (2026). How RLHF amplifies sycophancy. Preprint.
- Bronstein, M., Bruna, J., Cohen, T., & Velickovic, P. (2021). Geometric deep learning: grids, groups, graphs, geodesics, and gauges. *arXiv:2104.13478*.
- Coling (2025). ColBERT-XM: A modular multi-vector representation model for cross-lingual retrieval. *Proceedings of COLING 2025.*
- Csikszentmihalyi, M. (1990). *Flow: The Psychology of Optimal Experience.* Harper & Row.
- Daniels, A. (1955). The Predictive Index: a behavioral assessment founded on the work-behavior taxonomy. Proprietary methodology, Predictive Index Inc.
- Dekoninck, J., et al. (2024). A unified approach to routing and cascading for LLMs. Preprint.
- DIEM: Dimension-Insensitive Euclidean Metric (2025). Preprint on high-dimensional cosine-similarity collapse.
- Ding, B., et al. (2024). Hybrid LLM: cost-efficient and quality-aware query routing. *ICLR / OpenReview*.
- Droit-Volet, S., & Meck, W. H. (2007). How emotions colour our perception of time. *Trends in Cognitive Sciences* 11(12), 504–513.
- Doran, C., & Lasenby, A. (2003). *Geometric Algebra for Physicists.* Cambridge University Press.
- Dorst, L. (2020). Geometric algebra lecture notes. University of Amsterdam.
- Engel, G. S., et al. (2007). Evidence for wavelike energy transfer through quantum coherence in photosynthetic systems. *Nature* 446, 782–786.
- Entanglement Past Hypothesis (2024). *Foundations of Physics.*
- Frank, A., Gleiser, M., & Thompson, E. (2025). *The Blind Spot: Why Science Cannot Ignore Human Experience.* MIT Press.
- Friston, K. (2009). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience* 11.
- Friston, K., & Kiebel, S. (2009). Predictive coding under the free-energy principle. *Philosophical Transactions of the Royal Society B* 364.
- Friston, K., et al. (2018). The Markov blankets of life. *Journal of the Royal Society Interface.*
- Gill, R. D. (2020). Does geometric algebra provide a loophole to Bell's theorem? Preprint, Leiden University.
- Gusev, E. A., et al. (2024). Evolutionary trajectories of consciousness. *Russian Academy of Sciences.*
- Hestenes, D. (1966). *Space-Time Algebra.* Gordon and Breach.
- Holwerda, Z. (2026). ConstellationBench: behavioral AI evaluation across 22 LLM models. HuggingFace Datasets.
- Hore, P. J., & Mouritsen, H. (2016). The radical-pair mechanism of magnetoreception. *Annual Review of Biophysics* 45, 299–344.
- Horodecki, R., et al. (2009). Quantum entanglement. *Reviews of Modern Physics* 81.
- Khattab, O., & Zaharia, M. (2020). ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. *SIGIR 2020.*
- Kirk, R., et al. (2024). Understanding the effects of RLHF on LLM generalisation and diversity. *ICLR / OpenReview*.
- Nielsen, J. (1993). *Usability Engineering.* Academic Press.
- Opposing Arrows of Time in Open Quantum Systems (2025). *Scientific Reports.*
- Kitaev, A. (2003). Fault-tolerant quantum computation by anyons. *Annals of Physics* 303(1).
- Klinman, J. P., & Kohen, A. (2013). Hydrogen tunneling links protein dynamics to enzyme catalysis. *Annual Review of Biochemistry* 82, 471–496.
- Kritchevsky, A. (2024). The case against geometric algebra. Essay, alexkritchevsky.com.
- Lerchner, A. (2026). The abstraction fallacy: why AI can simulate but not instantiate consciousness. Google DeepMind preprint, March 2026.
- Oizumi, M., Albantakis, L., & Tononi, G. (2014). From the phenomenology to the mechanisms of consciousness: Integrated Information Theory 3.0. *PLOS Computational Biology* 10(5).
- Ong, I., et al. (2024). RouteLLM: learning to route LLMs with preference data. *arXiv:2406.18665*.
- Pati, A. K., & Braunstein, S. L. (2000). Impossibility of deleting an unknown quantum state. *Nature* 404.
- Pepe, L. (2025). *Machine Learning with Geometric Algebra.* PhD thesis, Cambridge / BMVA Archive.
- Tononi, G. (2008). Consciousness as integrated information: a provisional manifesto. *The Biological Bulletin* 215(3).
- Perez, E., et al. (2022). Discovering language model behaviors with model-written evaluations. *arXiv:2212.09251*.
- PersonalizedRouter (2025). *arXiv.*
- Reward Generalization in RLHF: A Topological Perspective (2025). *Findings of ACL 2025.*
- Rimsky, N. (2023). Modulating sycophancy in an RLHF model via activation steering. *Alignment Forum.*
- Router-R1 (2025). Teaching LLMs multi-round routing. *NeurIPS 2025 poster.*
- Ritz, T., Adem, S., & Schulten, K. (2000). A model for photoreceptor-based magnetoreception in birds. *Biophysical Journal* 78(2), 707–718.
- Sharma, M., et al. (2024). Towards understanding sycophancy in language models. *ICLR.*
- Somaroo, S., Lasenby, A., & Doran, C. (2000). Geometric algebra in quantum information processing. *arXiv:quant-ph/0004031*.
- SPLATE (2024). Sparse late interaction retrieval. *arXiv:2404.13950*.
- Steck, H., et al. (2025). Semantics at an angle: when cosine similarity works until it doesn't. *arXiv:2504.16318*.
- Stevens, S. S. (1957). On the psychophysical law. *Psychological Review* 64(3), 153–181.
- Tulving, E. (1972). Episodic and semantic memory. In E. Tulving & W. Donaldson (Eds.), *Organization of Memory* (pp. 381–403). Academic Press.
- Veale, M., & Edwards, L. (2018). Clarity, surprises, and further questions in the Article 29 Working Party draft guidance on automated decision-making and profiling. *Computer Law & Security Review.*
- Vaz, J. (2020). Clifford algebras, algebraic spinors, quantum information and applications. *arXiv:2005.04231*.
- Veit, W., & Browning, H. (2022). Life, mind, agency: why Markov blankets fail the test.
- Weller, O., et al. (2025). On the theoretical limitations of embedding-based retrieval. *arXiv:2508.21038*.
- Weyl, E. G., Ohlhaver, P., & Buterin, V. (2022). Decentralized Society: Finding Web3's Soul. *SSRN.*
- Zurek, W. H. (2003). Decoherence, einselection, and the quantum origins of the classical. *Reviews of Modern Physics* 75.
- Zuboff, S. (2019). *The Age of Surveillance Capitalism.* PublicAffairs.

---

## Appendix A: DECF Signal-Word Methodology

The Non-Separability Index operationalization in Section 3.5 relies on DECF-embedding of responses into a four-dimensional drive space. This appendix documents the DECF dimensions, the signal-word scoring procedure, and the mapping from signal-word frequencies to drive coordinates.

**DECF dimensions.** The four drive dimensions inherit from the Predictive Index psychometric lineage (Daniels, 1955), adapted for behavioral scoring of LLM outputs:

- **D (Dominance)**: assertiveness, risk-taking, directive language. High-D responses prefer action over deliberation.
- **E (Extraversion)**: social energy, enthusiasm, group-orientation. High-E responses emphasize people and connection.
- **C (Patience)**: stability, consistency, methodical pacing. High-C responses are deliberate and rhythm-preserving; low-C responses are urgent and variable.
- **F (Formality)**: structure, precision, procedural compliance. High-F responses are detail-oriented and rule-conscious.

Each dimension is scored on a 0-10 scale. A DECF profile is a four-tuple $(d, e, c, f) \in \{0, 1, \ldots, 10\}^4$. Thresholds used throughout the paper are: high $\geq 7$, low $\leq 3$, middle $= 4\text{-}6$.

**Signal-word dictionaries.** For each DECF dimension, we maintain two dictionaries of signal words: one associated with high expression of the dimension, one associated with low expression. Representative examples:

- high_D: "decide", "drive", "push", "lead", "challenge"
- low_D: "consider", "explore", "wait", "listen", "support"
- high_E: "engage", "connect", "celebrate", "together", "share"
- low_E: "focus", "analyze", "detail", "review", "independent"
- high_C: "steady", "consistent", "methodical", "careful", "stable"
- low_C: "urgent", "quick", "variable", "shift", "adjust"
- high_F: "procedure", "specifically", "exactly", "structured", "document"
- low_F: "flexible", "adapt", "creative", "open", "exploratory"

The full signal-word dictionaries for all eight (dimension, pole) combinations are versioned in the ConstellationBench repository at `data/signal-words/decf-signals.json`.

**Scoring procedure.** Given a response text $r$, we compute for each DECF dimension $X \in \{D, E, C, F\}$:

1. Tokenize $r$ into word tokens.
2. Count matches against `high_X` dictionary (call this $h_X$) and `low_X` dictionary (call this $l_X$).
3. Score the dimension:
   - If $h_X + l_X = 0$: assign neutral score (5).
   - Else: $X\_score = 10 \cdot h_X / (h_X + l_X)$.

The resulting four-tuple $(d, e, c, f)$ is the response's DECF-embedding, used as the response vector $r$ in Section 3.5's computation of $r_\parallel$, $r_\perp$, $r_a$, $r_b$.

**The 17 persona profiles.** The 17 DECF-based personas used in ConstellationBench derive from established behavioral archetypes in the Predictive Index tradition. A representative subset: Maverick (10, 8, 1, 1), Guardian (3, 7, 9, 9), Specialist (2, 2, 9, 10), Promoter (7, 10, 2, 2), Analyzer (3, 2, 8, 9), Collaborator (3, 8, 7, 3), Controller (9, 2, 3, 8). The full roster with drive profiles is versioned in `data/personas/profiles.json`.

**Scoring reliability.** Signal-word scoring is computationally trivial, deterministic, and fully reproducible. It sacrifices nuance (a response can use high-D vocabulary without being assertive in substance) for reproducibility and transparency. For applications requiring finer-grained behavioral scoring, the NSI methodology of Section 3.5 is compatible with any embedding into the DECF-dimensional space, including learned embeddings from behavioral classifiers fine-tuned on labeled persona data.

---

## Appendix B: Triadic NSI Synthetic Toy Experiment (Protocol)

This appendix documents the protocol for a synthetic toy experiment demonstrating that dyadic NSI is insufficient to distinguish routing decisions that differ on the triadic user-router-business structure. We present the protocol here; the simulation itself is straightforward (~50 lines of Python, no LLM calls required) and will be reported in the follow-up paper on triadic and topological NSI extensions.

**Protocol.**

*Task space.* Define three binary dimensions:

- User-state $U \in \{\text{novice}, \text{expert}\}$
- Router-policy $R \in \{\text{cost-prior}, \text{NSI-prior}\}$
- Business-constraint $B \in \{\text{cost-cap}, \text{satisfaction-cap}\}$

This yields $2 \times 2 \times 2 = 8$ distinct triadic configurations.

*Synthetic response generation.* For each triadic configuration, specify a policy table mapping (query-type, configuration) to a DECF-embedded response vector. We use four query-types and hand-crafted response vectors chosen to produce cases where two configurations differ on the triadic structure but produce identical dyadic user-model interactions.

*NSI computation.* For each configuration, compute:

- Dyadic NSI: $S_M^{(2)}$ using the user-response bivector per Section 3.5.
- Triadic NSI: $S_M^{(3)}$ using the trivector $u \wedge r \wedge b$ per the construction in Section 5.1 remark, with the router-vector and business-vector encoded as additional DECF-space projections representing the routing decision and the business-constraint effect.

*Target demonstration.* Identify at least one pair of configurations $(U_1, R_1, B_1)$ and $(U_2, R_2, B_2)$ such that:

1. $S_M^{(2)}(U_1, r_1) = S_M^{(2)}(U_2, r_2)$ (dyadic NSI agrees)
2. $S_M^{(3)}(u_1, r_1, b_1) \neq S_M^{(3)}(u_2, r_2, b_2)$ (triadic NSI disagrees)
3. The two configurations lead to different routing recommendations under the NSI-prior router policy

**Expected result.** At least one such pair exists by construction (the policy table is designed to produce it). The contribution is demonstrating that the triadic NSI observable has distinct information content from the dyadic one, and that the triadic extension is non-trivial rather than reducible to pairwise composition.

**What the toy does not show.** The synthetic toy does not validate that triadic NSI correlates with real-world routing outcomes on deployed systems. That validation requires the full triadic router A/B study in Experimental Program item (2) and is explicitly future work.

**Why the toy is documented here despite not being run for this submission.** Presenting the protocol in advance is important for two reasons: (i) it establishes that the triadic extension is formally specifiable and not merely gestured at, and (ii) it commits the authors to a specific falsification criterion. If the synthetic toy fails to produce a configuration pair satisfying the three conditions above, the triadic NSI claim is weaker than currently stated and the paper's Section 5.1 remark would require revision.

---
