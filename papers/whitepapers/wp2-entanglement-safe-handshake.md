---
title: "The Entanglement-Safe Handshake"
subtitle: "A threat model and protocol sketch for post-disconnect inference safety, borrowing documentation structure from cryptographic hash analysis"
author: "Zachary Holwerda"
affiliation: "Airlock Labs"
date: "2026-04-22 (working draft)"
audience: "ML privacy researchers, cryptographers, engineers"
paper_class: "Technical whitepaper"
---

## Abstract

The entire engineering battle of behavioral AI routing is one sentence: **keep the behavioral superposition unresolved long enough to route to the right model before pressure forces the response to collapse.** This is the engineering analog, on a classical substrate, of the central challenge of quantum computing: keep the qubit unresolved long enough to compute before nature forces decoherence.

The commercial and regulatory corollary is that the user's behavioral state must not cross into the vendor's persistent infrastructure, because once it does, the analog of decoherence occurs: the joint state collapses into a one-sided residue that is no longer entangled with the user but remains load-bearing for the vendor's downstream inferences about other users. We call this entrapment. It is the actual substance of what GDPR-style "right to deletion" was written to prevent, and it is the actual substance of what scalar-identifier deletion fails to deliver.

This whitepaper specifies the primitive that addresses the problem: an **entanglement-safe handshake** for user-system interaction, structured so that the user's behavioral kernel cannot be reconstructed from the post-session transcript or from downstream model state, even by an adversary substantially more powerful than the one the protocol was originally designed against. The threat model, protocol sketch, and open problems are structured by borrowing the documentation genre of cryptographic hash analysis (SHA-256 cryptanalysis literature) and mapping it onto the ML privacy attack catalog (membership inference, attribute inference, embedding inversion, model inversion, linkability).

## 1. Problem statement

Deployed behavioral AI systems routinely claim to respect user privacy by offering scalar-identifier deletion, the GDPR Article 17 right to be forgotten and its analogs in CCPA. These guarantees are structurally insufficient. Deleting a user's account, records, and persistent identifiers does not decouple the user from the system's inferences, because user-system interactions generate joint-state information that persists in the data distribution, the model weights, and the embeddings of downstream interaction partners. The companion paper on non-separability argues this in detail. This whitepaper takes the argument as given and focuses on the engineering question that falls out of it.

If scalar-identifier deletion is insufficient, what primitive is sufficient?

We argue that the right primitive is an **entanglement-safe handshake**: a protocol for user-system interaction designed so that the user's behavioral kernel cannot be reconstructed from the post-session transcript or from downstream model state, even by an adversary substantially more powerful than the one the protocol was originally designed against. The analog we borrow from is post-quantum cryptography, where a signature scheme is considered quantum-safe if it remains secure even against a hypothetical future quantum adversary. The analog we borrow the documentation structure from is cryptographic hash analysis, specifically the body of work on SHA-256, where a mature literature has developed for reasoning about primitive safety through structural attacks, probabilistic attacks, partial attacks, complexity bounds, and formal verification.

This whitepaper is not a proof. It is a threat model and a protocol sketch. It names the attack surface, maps it onto existing work in the ML privacy literature, proposes a handshake architecture that addresses it, and identifies the open problems that would need to be solved to make the handshake fully quantitative.

## 2. Threat model

**Ontological framing.** Before cataloguing specific attacks we state what the handshake is and is not trying to protect. Following Lerchner (2026), computation is the manipulation of physical vehicle states (voltages, attention weights, embedding coordinates) that carry no intrinsic semantic content. Meaning in an interaction is contributed by the user, who is the mapmaker alphabetizing continuous experience into discrete symbols the session exchanges. When the user disconnects, the vendor retains vehicle states — cache entries, embedding deltas, model activations, weight updates — but does not retain the user's content. The adversary's task, therefore, is not to "recover the user's thoughts" (which are constitutively unavailable to any system that is not the user) but to re-alphabetize the retained vehicle states into a *statistical shadow* sufficient for commercially or personally sensitive downstream inference. The handshake's goal is to bound the adversary's capacity for this re-alphabetization by preventing the vendor from accumulating the vehicle states that make the shadow possible. We are protecting against successful simulation of the mapmaker, not against successful capture of the mapmaker's experience — because the latter is structurally impossible and does not need to be defended against. This framing matters because it tells us precisely what to measure: the residual vehicle state, not the (undefined, unmeasurable) "user meaning."

**The adversary.** The adversary in this threat model is not a malicious external actor. It is the future behavioral inference capability of any system that retains state from a user's session. This includes the vendor whose service the user interacted with, any third party that receives data derivatives from the vendor, and any entity that could in the future gain access to the vendor's model weights or embeddings through acquisition, breach, or regulatory compulsion.

The adversary's goal is to reconstruct the user's behavioral kernel, or any commercially or personally sensitive projection of it, from the information that persists after the user disconnects. This kernel may include the user's drive profile, interaction patterns, domain preferences, decision history, or any latent state useful for profiling, targeted advertising, insurance underwriting, credit assessment, employment screening, or similar downstream applications.

We categorize the attacks borrowing the taxonomy of the SHA-256 cryptanalysis literature.

### 2.1 Structural attacks

Structural attacks exploit the specific architecture of the system to reconstruct user state. In SHA-256, these are attacks on the message schedule or the compression function rounds that leak information about the input. In behavioral inference, the analogs are well documented.

**Embedding inversion** (Song et al. 2020; Morris et al. 2023): given the embedding of a user's interaction history, reconstruct the underlying text. Demonstrated feasible against standard sentence-embedding models with high fidelity. In the behavioral handshake context, this is the attack where a vendor holds cached embeddings of a user's session and an adversary recovers the session text.

**Model inversion** (Fredrikson et al. 2015): given access to a model's query interface and partial knowledge of the input, reconstruct the training data. In the handshake context, this is the attack where a fine-tuned model leaks user-specific training signal through its outputs.

**Kernel reconstruction from persona signatures.** Specific to behavioral systems: given the model's outputs across a session, infer the DECF drive profile that was used to condition those outputs. This is not a documented attack in the ML privacy literature yet, because DECF-conditioned models are not yet a widely deployed class. It is an expected attack surface as behavioral AI deployments proliferate.

### 2.2 Probabilistic attacks

Probabilistic attacks use statistical inference rather than exact reconstruction. In SHA-256, these include SAT-solver-based preimage attacks and Bayesian approaches that exploit the structured nature of the message schedule. In behavioral inference, the dominant probabilistic attack is membership inference.

**Membership inference** (Shokri et al. 2017; Carlini et al. 2022): given access to a model, determine whether a specific user's data was in the training set. Demonstrated feasible against large language models with nontrivial success rates, especially for users whose data appears frequently. In the handshake context, this is the attack where an adversary wants to know whether a specific user ever interacted with the vendor's system, even if no per-user identifier remains.

**Attribute inference**: given partial information about a user and access to the model, infer a protected attribute. In behavioral systems this includes inferring drive-profile attributes that the user never explicitly disclosed.

**Linkability attacks** (Narayanan & Shmatikov 2008): given two anonymized datasets, link records across them using behavioral signatures. In the handshake context, this is the attack where a user's presence in one session is linked to their presence in another, across purportedly stateless systems, via the behavioral fingerprint their interactions leave behind.

### 2.3 Partial attacks

Partial attacks recover only a portion of the target but enough to be damaging. In SHA-256, the 3XOR paper of Boyar et al. shows how to construct three inputs whose hash outputs partially correlate, controlling half the output bits. In behavioral inference, the analog is partial-kernel reconstruction.

**Partial-kernel leakage through side channels.** Even if the full behavioral kernel cannot be reconstructed, specific projections may leak through timing, response length, routing decisions, or model-selection patterns visible to an observer of the system's public API. A vendor that routes to different models based on user kernel leaks kernel information through the routing decision itself if the routing is observable.

**Cached-prompt attacks.** If the system caches prompts or intermediate representations for efficiency, an attacker with access to the cache can recover partial kernel information even without direct model access.

### 2.4 Complexity-bound attacks

Complexity bounds are the quantitative safety claims that cryptanalytic literature provides for well-designed primitives. In SHA-256, the complexity of a brute-force preimage is 2^256 operations, which is infeasible by known physics even at planetary scales of compute. For behavioral inference, the analog is differential privacy.

**Differential privacy bounds** (Dwork 2006): a mechanism is (ε, δ)-differentially private if the probability distribution over outputs is similar whether any one user is present or absent from the input. DP provides a quantitative bound on membership and attribute inference attacks. The epsilon is the knob. Low epsilon means strong privacy; high epsilon means weak privacy.

The honest statement for the handshake context is that DP is a necessary but not sufficient tool. DP with reasonable utility typically requires epsilon values (on the order of 1 to 10) that do not provide strong bounds against sophisticated adversaries. The handshake must combine DP-style bounds with architectural barriers that make the attacker's job harder in the first place.

### 2.5 Verification

Cryptographic primitives are taken seriously only when their implementations have been formally verified. Appel et al. machine-checked the OpenSSL SHA-256 implementation against the spec. For behavioral handshake protocols, the analogous verification target is:

**Verified absence of persistent state.** A formal proof, ideally mechanized, that the system's code does not persist user behavioral state across sessions through any channel: disk, cache, model weights, embedding updates, or training-data aggregation. Static analysis and information-flow type systems are the right tools. There is existing work in this direction in the formally verified privacy-preserving systems literature (Opacus for PyTorch, Tumult Labs for differential privacy) but it has not yet been applied to behavioral AI deployments.

## 3. Protocol sketch

The entanglement-safe handshake is a protocol for user-system interaction that addresses the threat model. We describe it at the level of a sketch, not a full spec.

### 3.1 Per-session reconstruction

The user's behavioral kernel is never stored by the vendor. It is reconstructed at session start from data the user controls. The user may keep their own kernel in local storage, in a personal data vault, on a user-controlled blockchain, or in any mechanism that grants the vendor read access scoped to the session.

At session start, the vendor's router receives the kernel, holds it in volatile memory, and uses it for routing and response generation. At session end, the router zeroes the memory. No disk flush, no persistent embedding update, no cache entry keyed on user identity.

### 3.2 Stateless routing with in-session cache

Within a session, the router may cache routing decisions to avoid recomputing the kernel at every turn. The cache is keyed on (session_id, kernel_hash, task_type) and lives only for the session's duration. At session end, the cache is purged.

Cross-session reuse of the cache is not permitted. This is a deliberate architectural constraint. It sacrifices efficiency for non-entanglement. The efficiency cost is bounded because the cost of reconstructing the kernel at session start is amortized across the session's turns.

### 3.3 No weight updates on user data

The models the router calls are foundation models or specialized fine-tunes that are updated on signals that do not include user-specific behavioral state. Aggregate task-level signals (this class of prompts benefited from this class of models) are permitted. User-specific signals are not.

This is the single most consequential architectural constraint. It is also the one that most directly conflicts with current industry practice, where RLHF and continued fine-tuning on user interaction data is the default. The argument is that the statistical advantages of this practice do not justify the consent cost, and that the advantages can be recovered through task-level signal aggregation that does not require user-specific capture.

### 3.4 Observable routing limited by DP noise

When the router's routing decision is observable to an external party, for example through response latency, model identity disclosed in API headers, or side channels, the decision itself may leak kernel information. To bound this leakage, the routing decision is made with differentially private noise added to the kernel-aware scoring function.

This is the DP-style quantitative safety layer. The epsilon is a deployment parameter. For high-sensitivity verticals (healthcare, finance) the epsilon should be tight. For lower-sensitivity verticals the epsilon can be loose to favor routing quality.

### 3.5 Audit trail

The vendor maintains an audit trail of what information was held in memory during each session, for what duration, and what was zeroed at session end. The audit trail is user-readable on request. This is the accountability primitive that makes the architectural guarantees legible to users and regulators.

## 4. Comparison to existing privacy primitives

The entanglement-safe handshake is not a replacement for existing privacy primitives. It composes with them.

**Differential privacy** provides quantitative bounds on membership and attribute inference. The handshake uses DP for observable routing decisions. DP alone is insufficient because it does not address the architectural question of whether user state is captured in the first place.

**Federated learning** keeps user data on the user's device during training. This is a related but distinct architecture. Federated learning is about training without centralizing data. The handshake is about inference without persisting user state. They can be combined. A federated-trained model can be served through an entanglement-safe handshake.

**Trusted execution environments** (Intel SGX, AMD SEV, confidential computing) provide hardware-enforced isolation for sensitive computation. The handshake can be implemented inside a TEE to provide hardware-enforced verification that no persistent state is written. This strengthens the audit trail from a software guarantee to a hardware guarantee.

**Homomorphic encryption** and **secure multiparty computation** provide cryptographic guarantees on computation over encrypted data. These are more expensive and more powerful than the handshake. They are appropriate for a small subset of high-sensitivity applications. For the general case, the handshake plus DP plus TEEs provides a better cost-to-safety tradeoff.

**Self-sovereign identity** frameworks (W3C Verifiable Credentials, DIDs) provide the identity primitive the handshake requires but do not specify the inference protocol. The handshake composes naturally with SSI: the user's behavioral kernel can be an SSI-controlled credential that the user presents to the vendor at session start.

## 5. Open problems

The handshake is a sketch, not a proof. The following are the open problems that would need to be solved to make it a quantitative engineering artifact.

**OP1: Formal definition of "kernel reconstruction."** The threat model requires a precise definition of what it means to reconstruct a user's behavioral kernel. This is not obvious. Two kernels that differ in one drive score are arguably "the same" for most practical purposes. The reconstruction threshold needs a formal metric, likely a Wasserstein-style distance in the DECF drive space.

**OP2: Quantitative bounds on attribute inference given observable routing.** DP bounds exist for membership inference. The analogous bounds for attribute inference given observable routing decisions are less developed. This is a concrete research contribution a privacy-focused team could make.

**OP3: Side-channel analysis of routing decisions.** What information leaks through response latency, token count, model selection, cost? A systematic empirical study of observable routing side channels would be valuable.

**OP4: Formal verification toolkit.** The verified-absence-of-persistent-state property requires a toolkit. Extending static analysis tools like Opacus to cover behavioral routing is a concrete engineering target.

**OP5: Benchmarking against SOTA attacks.** Once a handshake implementation exists, it should be benchmarked against the full suite of membership inference, attribute inference, and embedding inversion attacks. This is the empirical analog of running a new hash function through the known cryptanalytic attacks.

## 6. Related work

The entanglement-safe handshake sits at the intersection of four existing research programs.

**ML privacy attacks.** Shokri et al. on membership inference, Carlini et al. on training data extraction, Fredrikson et al. on model inversion, Song et al. and Morris et al. on embedding inversion, Narayanan & Shmatikov on linkability. These provide the attack catalog.

**Differential privacy.** Dwork and collaborators provide the quantitative privacy framework. DP-SGD (Abadi et al. 2016) provides the training-time application. Opacus (Meta / PyTorch) and Tumult Labs provide practical libraries.

**Confidential computing.** Intel SGX, AMD SEV, and related TEE technologies provide the hardware substrate for verified non-capture.

**Self-sovereign identity.** W3C Verifiable Credentials, DIDs, and the broader decentralized-identity ecosystem provide the user-controlled identity primitive.

The handshake does not invent new cryptography. It specifies an architectural pattern that composes existing primitives to achieve a guarantee that any single primitive cannot achieve alone: no persistence of user behavioral state across sessions, subject to quantitative bounds on what can be inferred from the transcript and from the vendor's downstream model state.

## 7. Closing

A handshake is safe against future adversaries only when it has been documented in the way cryptographic primitives are documented: structural attacks, probabilistic attacks, partial attacks, complexity bounds, formal verification. Behavioral AI deployments currently have none of this. They have marketing claims about "we don't sell your data" and deletion forms that do not actually delete. This is not good enough for the verticals where behavioral AI is most valuable, which are also the verticals where users are most exposed.

This whitepaper proposes to fix the documentation gap by borrowing the genre wholesale from the cryptographic hash analysis literature. The attacks already exist in the ML privacy literature. They just have not been assembled into a threat model the way SHA-256 attacks have been assembled into a threat model. Assembling them is most of the work.

The next step is an implementation. A reference implementation of the handshake, sitting over a live LLM deployment with the five architectural properties of Section 3, benchmarked against the attack catalog of Section 2, with a formal-methods pass on the five open problems. That is the artifact that converts the threat model into engineering.

---

*Airlock Labs · airlocklabs.io · admin@airlocklabs.io*

## References (to be completed)

- Abadi, M., et al. (2016). Deep learning with differential privacy. *CCS.*
- Appel, A., et al. (2015). Verification of a cryptographic primitive: SHA-256. *TOPLAS.*
- Boyar, J., et al. (2021). Controlling half the output of SHA-256. *Information Processing Letters.*
- Carlini, N., et al. (2021). Extracting training data from large language models. *USENIX.*
- Carlini, N., et al. (2022). Membership inference attacks from first principles. *IEEE S&P.*
- Dwork, C. (2006). Differential privacy. *ICALP.*
- Fredrikson, M., et al. (2015). Model inversion attacks that exploit confidence information. *CCS.*
- Lerchner, A. (2026). The abstraction fallacy: why AI can simulate but not instantiate consciousness. Google DeepMind preprint, March 2026.
- Morris, J., et al. (2023). Text embeddings reveal (almost) as much as text. *EMNLP.*
- Narayanan, A., & Shmatikov, V. (2008). Robust de-anonymization of large sparse datasets. *IEEE S&P.*
- Shokri, R., et al. (2017). Membership inference attacks against machine learning models. *IEEE S&P.*
- Song, C., & Raghunathan, A. (2020). Information leakage in embedding models. *CCS.*
