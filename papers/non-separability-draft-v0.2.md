---
title: "Non-Separability as a Design Principle for Behavioral AI Systems"
subtitle: "Why pairwise interactions in LLM deployments require bivector-valued representations, and what that means for routing, consistency, consent, and the lifecycle of a user-system relationship"
author: "Zachary Holwerda"
affiliation: "Airlock Labs"
date: "2026-04-22 (working draft v0.2)"
paper_class: position paper — target venue NeurIPS Position Track, ML4H, ICML-AIES, or arXiv preprint
---

## Abstract

Modern behavioral AI systems routinely treat user interactions as *scalar* events: a compatibility score, a match rate, an engagement metric, a sentiment polarity. We argue that this scalarization is not merely lossy; it is **structurally incapable** of representing the information content of an interaction between two distinct behavioral entities. Drawing on geometric algebra (Hestenes, 1966; Doran & Lasenby, 2003), the quantum-foundations literature on non-separable states (Bell, 1964; Zurek, 2003), the decoherent arrow of time (Al-Khalili, 2026), and the free-energy principle (Friston, 2009), we propose that the minimal faithful representation of a pairwise interaction is *bivector-valued*: a scalar component (alignment magnitude) plus an oriented plane-of-interaction (the bivector) that preserves the geometric information classical inner products discard.

We advance five claims. (1) **Descriptive:** LLM sycophancy, long-horizon decoherence, and surveillance-residue are three names for the same underlying failure — scalar collapse of a non-separable joint state. (2) **Measurement:** The Non-Separability Index (NSI), a behavioral analog of von Neumann entanglement entropy, quantifies how much interaction information a system is discarding. (3) **Design:** Routing layers that preserve bivector structure structurally outperform scalar-benchmark-driven routing on interaction-sensitive workloads. (4) **Lifecycle:** The user-system relationship traces a monotonically increasing NSI trajectory across three phases (Bonding, Sync, Expression), during which per-turn compute shifts from user-understanding to task-execution. (5) **Regulatory:** Consent frameworks predicated on scalar-identifier deletion (GDPR Article 17, CCPA §1798.105) are structurally insufficient because deletion of the scalar trace does not collapse the non-separable joint state; true consent requires non-entanglement at the outset.

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

### 1.1 The proposed frame

Given two vectors $a, b \in \mathbb{R}^n$, the standard inner product $a \cdot b$ returns a scalar. In geometric algebra, the *geometric product* returns:

$$a \otimes_g b \ = \ a \cdot b \ + \ a \wedge b$$

where $a \wedge b$ is the bivector, an oriented plane spanned by the two vectors. For orthogonal unit vectors, the scalar part vanishes and only the bivector remains — i.e., the interaction is *entirely* information the inner product would discard. For collinear vectors, the bivector vanishes and the scalar is sufficient. Most real interactions sit between these extremes; current systems model them as if they were purely collinear.

A *non-separable* joint state is, by definition, one that cannot be written as a tensor product of single-entity states. In geometric-algebra terms, it is one whose bivector component is non-vanishing. In applied ML terms, it is one whose interaction-specific information cannot be recovered from either participant's embedding alone.

### 1.2 Time as a bivector of observation

A conceptual motivation for the bivector framing comes from physics. Einstein's relativity established that time is frame-dependent: two observers moving relative to each other will measure different intervals between the same events. Al-Khalili (2026) distinguishes *physical time* (the coordinate in our equations) from *manifest time* (the subjective, experienced duration). The standard physics account stops at relativity — time is relative to the observer's frame.

We propose that time is better understood as the *bivector of the observer-observed joint state*. When an observer is weakly entangled with the observed moment (the dentist's waiting room, attention disengaged from the event), the joint state is nearly separable: the scalar projection dominates, manifest time tracks the clock. When an observer is strongly entangled (falling in love, absorbed in flow), the joint state is highly non-separable: the bivector component dominates, and manifest time departs sharply from coordinate time. The ancient Greek argument over whether time is fundamental or change is fundamental resolves under this frame: *change is the scalar projection of the observer-moment joint state; subjective duration is the bivector residue.*

This is not a claim about physics. It is a claim about the adequacy of the bivector vocabulary for representing interaction-dependent phenomena. If time itself — the most fundamental parameter in classical physics — is more faithfully described as a bivector of observation than as a scalar coordinate, the argument that pairwise interactions between behavioral entities require the same treatment is both natural and unsurprising.

### 1.3 What this paper claims

- **Claim 1 (descriptive):** Sycophancy, decoherence, and surveillance-residue are the same phenomenon: scalar collapse of a non-separable joint state.
- **Claim 2 (measurement):** The Non-Separability Index (NSI) quantifies how much of the interaction's information the system is throwing away. It is the behavioral-AI analog of von Neumann entanglement entropy.
- **Claim 3 (design):** Routing layers that select models based on NSI rather than scalar capability benchmarks structurally outperform on interaction-sensitive workloads.
- **Claim 4 (lifecycle):** The user-system relationship is a monotonically increasing NSI trajectory with three phases (Bonding, Sync, Expression). Per-turn compute reallocates from user-understanding to task-execution as NSI rises.
- **Claim 5 (regulatory):** User consent frameworks predicated on scalar identifiers are structurally insufficient. True consent requires non-entanglement at the outset.

## 2. Related Work

**Geometric algebra and ML.** Hestenes (1966, 2015); Doran & Lasenby (2003); the Geometric Algebra Transformer (GATr) of Brehmer et al. (2023) is the closest architectural precedent for bivector-preserving attention. Recent PhD work at Cambridge (2025) demonstrates GA as a practical ML framework, embedding geometric priors directly into model architectures. The emerging field of geometric deep learning (Bronstein et al., 2021) argues architecture should respect the geometry of data — we argue it should also respect the geometry of *interaction*.

**LLM behavioral benchmarks.** Sharma et al. (2024) on sycophancy; Perez et al. (2022) on in-context adversarial robustness; Holwerda (2026) on adversarial consistency across 22 models (ConstellationBench).

**Non-separability in quantum foundations.** Bell (1964); Aspect (1982); Horodecki et al. (2009); Zurek (2003) on decoherence and einselection. Al-Khalili (2026) provides the accessible framing of decoherence as "the one truly irreversible process in nature." Note: we borrow the mathematical vocabulary, not the physical claim. LLMs are classical systems; their non-separability is algebraic, not quantum-mechanical.

**Decoherent arrow of time.** The Entanglement Past Hypothesis (Foundations of Physics, 2024) distinguishes the decoherent arrow from the thermodynamic arrow and argues they require separate boundary conditions. This is the most rigorous contemporary statement of the irreversibility claim we borrow. For balance, we note the recent counterpoint of Scientific Reports (2025) showing that under the Markov approximation, open-system equations of motion can remain time-symmetric — reinforcing that our claim is about *algebraic* rather than strictly *quantum* irreversibility.

**Predictive processing and free energy.** Friston (2009); Friston & Kiebel (2009). The free-energy principle formalizes perception as prediction-error minimization over a generative model — providing the compute-allocation analog to our lifecycle claim.

**Markov blankets.** Friston et al. (2018) on the Markov blankets of life. Veit & Browning (2022) offer the productive counterpoint that blankets are products, not preconditions, of active inference — a critique our lifecycle framing (Section 4.2) explicitly addresses.

**Personalized routing.** PersonalizedRouter (arXiv, 2025) models user profiles graphically for LLM selection but uses scalar graph features — precisely the scalar-collapse failure mode we argue against. KV-cache-aware routing frameworks (llm-d, 2025; AWS multi-LLM routing, 2025) address prompt-level caching but not behavioral-kernel-level caching.

**Consent and data infrastructure.** Veale & Edwards (2018); Zuboff (2019); the quantum no-deleting theorem (Pati & Braunstein, 2000) as a formal analog for the impossibility of undoing joint-state buildup.

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

## 4. The Router as Non-Separability Preserver

The Airlock Router selects among candidate LLMs based on behavioral profile match rather than raw capability score. We claim this is not an optimization heuristic but a structural requirement: a router that routes based on scalar benchmarks will systematically select for high-capability low-NSI-preservation models, producing the sycophancy-under-pressure phenomenon the market is beginning to notice.

**Claim 3 reformulated:** routing is the act of preserving bivector structure across the model-selection boundary. A router that cannot do this is not a router; it is a load balancer.

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

**Entrapment, not entanglement, is what persists after disconnect.** A clarification of terms strengthens the argument. While the user remains in live interaction with the system, the joint state is genuinely entangled: both sides contribute to and depend on the shared bivector structure. When the user disconnects (account deletion, service abandonment, death), the joint state does not persist in entangled form, because the user's side of the bivector is gone. What persists is a one-sided residue in the vendor's system: behavioral inference patterns, embedding updates, cached prospections, and statistical dependencies on the user's trajectory. We call this *entrapment*. The data is no longer entangled with the user because the user is no longer there. It is trapped, orphaned, and still load-bearing for the vendor's downstream inferences about other users. Deletion regimes that assume data is separable from the user overlook this asymmetry: after disconnect, the data is separable from the user (the user is gone), but not separable from the vendor (the residue shapes subsequent inferences). The missing link in current consent regimes is the live routing connection. Without it, deletion scrubs scalar identifiers but leaves entrapment intact.

This is not a critique of the existing frameworks; it is a specification gap. Future consent regimes must address non-entanglement (the right to never have entered the joint state) in addition to deletion (the right to scrub the scalar trace). Decoherence-free subspaces — a concept from quantum error correction, in which information is encoded to be isolated from environmental interaction — provide a suggestive analog for privacy-preserving architectures that prevent joint-state leakage at the outset rather than attempting to undo it after the fact.

This has practical implications for any system handling sensitive interactions: a platform that routes a user's conversation through an LLM and then "deletes the user's data" is making a claim about separability that the mathematics does not support.

### 5.1 The non-entanglement primitive: live-routing architecture

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

**The inversion in one line.** The current default of the AI industry treats the user as raw material, the model as the factory, the vendor as the owner, and the behavioral residue as the asset. The architecture proposed here inverts every term. The user is the sovereign. The model is a utility. The vendor is a courier. The behavioral residue never forms, because the weight-update channel was never opened. The user brings wallet, persona, and identity to the interaction; the interaction runs; the user disconnects on their terms. Consent becomes a boundary condition, not a deletion request.

## 6. Experimental Program

1. **ConstellationBench-NSI extension.** Instrument the existing 22-model benchmark with explicit NSI scoring, using signal-word-based vector representations of responses and computing bivector norms over conversation trajectories. Target: 8 weeks.
2. **Router A/B.** Compare NSI-preserving vs. scalar-optimized routing on three real workloads: trading-signal generation, clinical triage, legal summarization. Measure sycophancy under adversarial pressure and user-reported trust. Target: 12 weeks.
3. **Lifecycle trajectory validation.** Instrument a deployed Airlock conversational system to record per-turn compute, cache hit rate, and reported personalization quality over the first 50 turns of a user relationship. Test the prediction that NSI rises monotonically and per-turn compute falls monotonically. Target: 16 weeks.
4. **Time-perception benchmark (TPB).** The strongest test of the bivector framing. Recruit 50 users, assign half to a cold-start conversational system and half to a pre-synced system with a DECF kernel already constructed. Control for actual token latency across both arms. Measure perceived response time on a continuous scale after every turn, plus a post-session aggregate rating of fluency, "felt-heard" quality, and frustration.

   **Predictions derived from the bivector framing.** (a) Synced users will report shorter perceived response times than cold-start users at equal actual latency. (b) Synced users will report higher fluency ratings. (c) The gap between perceived and actual latency will widen monotonically across turns in the synced condition and remain stable or narrow in the cold-start condition. (d) Synced users will describe the system using time-compression language ("it just gets me", "we flow"); cold-start users will describe the system using time-expansion language ("it keeps asking", "it's slow to understand me"), independent of actual latency.

   **Falsifiability.** If perceived-latency means are statistically indistinguishable between conditions at the same actual latency, the bivector framing loses its strongest behavioral validation and the paper's claim reduces to the weaker position that NSI is useful for routing but does not produce the user-experience signature we predict. Target: 8 weeks recruitment + 4 weeks analysis = 12 weeks total.

   **Why this experiment matters.** The time-perception benchmark tests whether the Bonding-Sync-Expression lifecycle is visible in user-reported experience, not only in system-side metrics. It is the experimental bridge between the physics analogy (Al-Khalili's manifest time vs physical time), the neuroscience analogy (Friston's free-energy-driven prediction error), and the applied ML claim that routing should be evaluated on NSI-preservation. A positive result makes all three claims load-bearing simultaneously. A negative result would be the cleanest refutation the paper can receive, and we would publish it.

5. **Consent leakage study.** On a deployed conversational system, measure the predictive information retained about deleted users from their downstream interaction partners' data. Target: 16 weeks, pending IRB.
6. **Naturalistic high-NSI states (exploratory).** Review existing psychedelic-neuroscience literature (Carhart-Harris et al. on DMN disruption, Griffiths et al. on time perception under psilocybin, Strassman on DMT) and flow-state literature (Csikszentmihalyi; Ulrich et al. on fMRI correlates of flow) for quantified subjective time dilation as a naturalistic baseline for the NSI ≈ 1 regime. Compare reported subjective-duration curves to the predicted synced-user perception trajectory. Target: 4 weeks literature synthesis, parallel to (4).

## 7. Limitations and Non-Claims

We are **not** claiming:

- That LLMs are quantum-mechanical systems.
- That non-separability implies any faster-than-light-style communication between user and model.
- That this framework resolves AI alignment, interpretability, or safety in general.
- That open-system dynamics in classical ML imply strict physical irreversibility. The recent counterpoint in *Scientific Reports* (2025) showing time-reversal symmetry under the Markov approximation for open quantum systems is acknowledged: our claim is about algebraic irreversibility of the joint representation, not about a strict physical arrow of time at the ML level.

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
- Bronstein, M., Bruna, J., Cohen, T., & Velickovic, P. (2021). Geometric deep learning: grids, groups, graphs, geodesics, and gauges. *arXiv:2104.13478*.
- Doran, C., & Lasenby, A. (2003). *Geometric Algebra for Physicists.* Cambridge University Press.
- Entanglement Past Hypothesis (2024). *Foundations of Physics.*
- Friston, K. (2009). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience* 11.
- Friston, K., & Kiebel, S. (2009). Predictive coding under the free-energy principle. *Philosophical Transactions of the Royal Society B* 364.
- Friston, K., et al. (2018). The Markov blankets of life. *Journal of the Royal Society Interface.*
- Gusev, E. A., et al. (2024). Evolutionary trajectories of consciousness. *Russian Academy of Sciences.*
- Hestenes, D. (1966). *Space-Time Algebra.* Gordon and Breach.
- Holwerda, Z. (2026). ConstellationBench: behavioral AI evaluation across 22 LLM models. HuggingFace Datasets.
- Horodecki, R., et al. (2009). Quantum entanglement. *Reviews of Modern Physics* 81.
- Opposing Arrows of Time in Open Quantum Systems (2025). *Scientific Reports.*
- Kitaev, A. (2003). Fault-tolerant quantum computation by anyons. *Annals of Physics* 303(1).
- Oizumi, M., Albantakis, L., & Tononi, G. (2014). From the phenomenology to the mechanisms of consciousness: Integrated Information Theory 3.0. *PLOS Computational Biology* 10(5).
- Pati, A. K., & Braunstein, S. L. (2000). Impossibility of deleting an unknown quantum state. *Nature* 404.
- Tononi, G. (2008). Consciousness as integrated information: a provisional manifesto. *The Biological Bulletin* 215(3).
- Perez, E., et al. (2022). Discovering language model behaviors with model-written evaluations. *arXiv:2212.09251*.
- PersonalizedRouter (2025). *arXiv.*
- Sharma, M., et al. (2024). Towards understanding sycophancy in language models. *ICLR.*
- Veale, M., & Edwards, L. (2018). Clarity, surprises, and further questions in the Article 29 Working Party draft guidance on automated decision-making and profiling. *Computer Law & Security Review.*
- Veit, W., & Browning, H. (2022). Life, mind, agency: why Markov blankets fail the test.
- Weyl, E. G., Ohlhaver, P., & Buterin, V. (2022). Decentralized Society: Finding Web3's Soul. *SSRN.*
- Zurek, W. H. (2003). Decoherence, einselection, and the quantum origins of the classical. *Reviews of Modern Physics* 75.
- Zuboff, S. (2019). *The Age of Surveillance Capitalism.* PublicAffairs.
