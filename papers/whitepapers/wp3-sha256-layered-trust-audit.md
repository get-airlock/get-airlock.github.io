---
title: "SHA-256 as a Layered Trust Object"
subtitle: "Borrowing the structural-audit literature on cryptographic hash functions as a documentation genre for behavioral commitment functions"
author: "Zachary Holwerda"
affiliation: "Airlock Labs"
date: "2026-04-22 (working draft, research note)"
audience: "Cryptographers, protocol designers, ML privacy researchers"
paper_class: "Research note"
---

## Why this note exists

Two companion papers propose an architecture for behavioral AI deployments. The first (non-separability paper) argues that user-system interactions carry joint-state information that scalar methods cannot represent, and that consent frameworks predicated on scalar identifier deletion are therefore structurally insufficient. The second (entanglement-safe handshake whitepaper) proposes a protocol architecture for interactions that do not leave recoverable behavioral residue after disconnect.

Neither paper addresses the question of what a **behavioral commitment function** looks like. By analogy: if the behavioral handshake is a signature scheme, what is the hash function underneath it? What primitive lets a user "commit to" a behavioral state at session start in a way that the vendor can verify but not reconstruct?

This note is not the answer. It is the pointer to where the answer will come from. We argue that the body of structural-audit work on SHA-256 is the closest available template for how a behavioral commitment function should be documented, and we walk through the specific pieces of that literature that are most relevant.

## The SHA-256 structural audit literature

Most public discussion of SHA-256 treats it as a black box. You put bytes in, you get 256 bits out, the bits look random, the operation is one-way, move on. This is the useful simplification for application developers. It is not the serious research view.

The serious research view is that SHA-256 is a specific combinatorial object: a 64-round compression function operating on a message schedule that expands 16 input words into 64 round-words via a specific recurrence. The structure matters. Structural attacks exploit specific features of the structure. Structural defenses articulate why those features do not permit attacks below a specific complexity bound.

Four bodies of work are particularly relevant for the analogy we want to build.

**Round-reduced attacks.** Many of the most successful SHA-256 attacks are against reduced-round variants, breaking security claims on 24-round, 39-round, or 41-round SHA-256 when the full function is 64 rounds. These attacks inform the design margin: how many more rounds than the known-attacks' reach does the deployed function have? For SHA-256 the margin has stayed comfortable but is not infinite. This kind of analysis gives defenders a quantitative handle on how far the adversary has come.

**Structural second-preimage analysis.** The recent verified second-preimage work against the Bitcoin genesis block (Aumasson et al. and related writeups) demonstrates that for *specific* message inputs, constructions can be found that produce the same hash through exploiting message-schedule structure. This is a narrow result — it does not generalize to arbitrary preimages — but it is a non-trivial break and it illustrates what structural analysis can do when the input space is constrained.

**Controlling partial output (3XOR, Boyar et al. 2021).** Published in *Information Processing Letters*, this result shows that three SHA-256 inputs can be constructed such that the XOR of their outputs has a specific controlled pattern on the first half of the bits. The attack complexity is large but tractable. This is the most mathematically interesting recent result on SHA-256 because it demonstrates a three-way constraint structure in a function explicitly designed to make outputs statistically independent.

**Formal verification (Appel et al., Princeton).** The verification of the OpenSSL SHA-256 implementation against the spec is the gold standard for how primitives should be documented. The spec is separate from the implementation. The correspondence between them is machine-checked. Bugs in implementation cannot be silently introduced. For a behavioral commitment function deployed at scale, this is the verification posture we should eventually reach.

## What a "layered trust" frame adds

The concept of **layered trust** is worth isolating because it appears across cryptographic and non-cryptographic contexts and maps cleanly onto the problem we are trying to solve.

Layer 1 is the mathematical primitive itself: the specific function, the specific algebraic structure, the specific security properties claimed.

Layer 2 is the set of attacks against the primitive, as they are known at a given time: structural attacks, probabilistic attacks, partial attacks. Layer 2 changes over time as the literature matures.

Layer 3 is the protocol context in which the primitive is embedded. A hash function used for password storage has a different Layer 3 than a hash function used for message authentication or for proof-of-work. The same primitive, deployed in three contexts, has three different adversarial profiles.

Layer 4 is the implementation: what code runs when the primitive is invoked, what hardware it runs on, what side channels are present.

Layer 5 is the operational context: who uses the primitive, under what threat model, with what accountability mechanisms. This is where audit trails, access controls, regulatory frameworks live.

A trust claim about SHA-256 has to address all five layers. "SHA-256 is secure" is shorthand for "Layer 1 has the properties claimed in the FIPS spec, Layer 2 contains no attacks below the complexity bound, Layer 3 is a usage context compatible with those properties, Layer 4 is a verified implementation, and Layer 5 is an operational context with appropriate audits." Breaking any layer breaks the trust claim even if the other layers are intact.

## Mapping to behavioral commitment

The layered trust frame applied to a behavioral commitment function produces the following structure.

**Layer 1: the commitment primitive.** What is the function that takes a user's behavioral state and produces a commitment the vendor can verify but not invert? Candidate constructions include cryptographic commitments with behavioral-embedding inputs, zero-knowledge proofs of behavioral properties, homomorphic encryption of behavioral vectors, and differentially private one-way behavioral fingerprints. Each has different Layer 2 attack profiles.

**Layer 2: the attack catalog.** What does it mean to "break" a behavioral commitment? The closest analog to SHA-256 preimage attacks is behavioral-kernel reconstruction: given the commitment, recover the behavioral state. The closest analog to 3XOR attacks is partial-state inference: given the commitment, recover a specific projection (for example a drive-profile attribute) without needing the full state. The ML privacy literature on embedding inversion, model inversion, and attribute inference maps directly onto this layer.

**Layer 3: the protocol context.** What is the commitment being used for? A commitment used to authenticate a session has a different adversarial profile than a commitment used to establish a user's drive profile to a vendor's router. The non-separability paper and the handshake whitepaper specify the protocol contexts that matter for Airlock's architecture. Different deployments may require different Layer 3 analyses.

**Layer 4: the implementation.** A behavioral commitment function deployed in production must have an implementation that can be audited. The audit must cover not just correctness against the spec but also absence of side channels (timing, memory access patterns, logging) that could leak the committed state.

**Layer 5: the operational context.** Who operates the commitment function? Who verifies the commitments? What audit trails exist? What regulatory framework applies? The entanglement-safe handshake specifies the architectural answer to Layer 5; policy frameworks that cover behavioral-joint-state deletion are the regulatory answer.

## Why the 3XOR result specifically matters

Of the four SHA-256 literature pieces cited, the 3XOR result by Boyar et al. is the most directly analogous to something we care about. Here is why.

The 3XOR attack constructs three inputs $x_1, x_2, x_3$ such that the first 128 bits of SHA-256($x_1$) $\oplus$ SHA-256($x_2$) $\oplus$ SHA-256($x_3$) = 0. This is a three-way constraint on the outputs, and it works despite SHA-256's design goal of making outputs look statistically independent.

The analogy to behavioral commitment is direct. Suppose a vendor deploys a behavioral commitment function intended to produce commitments that look statistically independent across users. An attacker who could construct three user behavioral inputs such that a specific projection of their commitments correlates in a predictable way would have broken the commitment function's statistical-independence claim, even if they could not invert any individual commitment.

This is a three-way attack on a primitive designed for pairwise independence. It rhymes with the triadic NSI formalism flagged in the non-separability paper. It suggests that a behavioral commitment function's adversarial analysis should include three-way attacks as a first-class category, not as an afterthought.

It is also the cleanest empirical argument against designing behavioral commitment functions as straightforward hash-of-embedding constructions. The SHA-256 literature shows that even very well-designed hash functions have three-way structural properties that sophisticated adversaries can exploit. A behavioral commitment function deserves the same level of analysis before deployment.

## Connection to quantum-safe signature schemes

Post-quantum cryptography (NIST PQC finalists: Dilithium, Falcon, SPHINCS+) provides the outer layer that the behavioral handshake should eventually sit inside. The reasoning is as follows.

The threat model for the behavioral handshake is a future inference adversary substantially more powerful than the one the handshake was designed against. By analogy, the threat model for post-quantum signatures is a future quantum adversary substantially more powerful than today's classical adversary. The analogy is not about quantum mechanics. It is about designing primitives that remain safe under adversary scaling.

A mature behavioral commitment function will borrow at least three properties from PQC signature design.

**Conservative design margins.** Post-quantum signature schemes typically have very large parameter choices relative to their apparent attack complexity, because the attack surface against future quantum adversaries is not fully characterized. Behavioral commitment functions should follow the same design principle. Assume the adversary will get stronger. Build in margin.

**Standardization.** NIST's PQC process is slow, public, and adversarial. Candidate schemes are published, attacked by the community, improved, attacked again. A behavioral commitment function should go through an analogous process before deployment at scale, with published specs that the privacy-research community can attack.

**Hybrid deployment.** Real post-quantum deployments run classical and post-quantum schemes in parallel during the transition, so that a break in either does not compromise the system. Behavioral deployments should plan for the same: run both the new handshake and a fallback classical architecture, and retain the ability to switch if the new architecture is compromised.

## Research directions

This note identifies four directions for follow-up work.

**R1: Specify a candidate behavioral commitment function.** Propose a specific mathematical construction with stated security claims, stated attack surface, and stated deployment context. Publish the spec. Invite the privacy-research community to attack it. Expect revisions.

**R2: Empirical attack catalog.** Assemble the ML privacy literature into an attack catalog structured the way SHA-256 cryptanalysis literature is structured. This catalog does not yet exist in the form a protocol designer can use directly. Building it is a concrete research contribution.

**R3: Three-way attack analysis.** Inspired by the 3XOR result, systematically explore what three-way constraint attacks are possible against candidate behavioral commitment functions. This work would directly inform the triadic NSI formalism proposed in the non-separability paper.

**R4: Formal verification pipeline.** Extend the Appel et al. style machine-checked verification approach to behavioral commitment implementations. Start with a minimal implementation and verify non-leakage properties formally.

## Closing

SHA-256 is not the answer. SHA-256 is not even a good direct candidate for a behavioral commitment function, because its design goals (preimage resistance against unstructured inputs, collision resistance, statistical output independence) do not match what a behavioral commitment needs (resistance to partial-state inference, support for zero-knowledge verification, compatibility with differentially private projections).

The SHA-256 literature is, however, the right genre for how to write about a behavioral commitment function once it exists. Structural attacks, probabilistic attacks, partial attacks, complexity bounds, formal verification. Layered trust. Conservative design margins. Hybrid deployments.

If the entanglement-safe handshake whitepaper is about the signature, this note is about the hash. The hash does not exist yet. When it is designed, we will know by how it is documented whether it is serious. The template is SHA-256.

---

*Airlock Labs · airlocklabs.io · admin@airlocklabs.io*

## References (to be completed)

- Appel, A., et al. (2015). Verification of a cryptographic primitive: SHA-256. *ACM TOPLAS.*
- Aumasson, J.-P., et al. Second-preimage analyses on reduced-round SHA-256 (various).
- Boyar, J., et al. (2021). Controlling half the output of SHA-256. *Information Processing Letters.*
- NIST Post-Quantum Cryptography Standardization. (2024). Selected algorithms: Dilithium, Falcon, SPHINCS+.
- Shokri, R., et al. (2017). Membership inference attacks against machine learning models. *IEEE S&P.*
- Song, C., & Raghunathan, A. (2020). Information leakage in embedding models. *CCS.*
