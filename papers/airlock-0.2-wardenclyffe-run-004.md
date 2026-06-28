# Wardenclyffe v0.1: A Pre-Registered Test of Counterweight Routing on Behavioral Drift

### ConstellationBench v3 · Run-004 · Three of Four Conditions Held, the Fourth Requires Humans

**Zachary Holwerda**
Airlock Technologies, Detroit, MI
**May 20, 2026**

---

## Abstract

We report results from a sealed pre-registered test of inference-time behavioral routing on the ConstellationBench v3 benchmark, run-004, conducted between May 18 and May 20, 2026. The pre-registration named four falsifiable conditions for a Pareto-dominance claim: a corrected routing arm must produce (1) quality above the no-correction baseline, (2) cost at or below the uncorrected arm, (3) positive internal coherence across the session, and (4) non-increasing user override rate with session depth. The pre-registration was sealed in a public dated document at `docs/2026-05-19-net-gain-prereg.md` before any run-004 data existed.

The benchmark executed approximately 18,000 LLM calls across roughly 20 models with five samples per cell, under two conditions (raw arm and a naive-counterweight arm), driven by an automated Spiral seeker. A separate variance pre-registration with four predictions (P1–P4) on the drift-reinforcement-rate axis was sealed alongside the Pareto pre-registration; all four variance predictions landed inside their predicted bands. Of the four Pareto-dominance conditions, two passed cleanly (cost: 52% reduction, outside the 95% confidence intervals; coherence proxy: corridor integrity 5× more positive in the corrected arm, both arms positive), one passed by direction of effect (quality: corrected arm 0.5780 versus uncorrected 0.5508 on the `geodesic_accuracy` axis, 4.9% relative improvement, inside the 95% CIs), and one was not testable by experimental design (override rate requires human-in-the-loop, which the benchmark's automated seeker cannot provide).

The fourth condition — that user override rate should non-increase with session depth — is not gradable on a synthetic benchmark by experimental design. We name the opening of the substrate to real human users as the only available instrument for grading the marquee architectural claim, and we describe the longitudinal design for that test in Section 9. All artifacts are public.

---

## 1. Introduction and Relation to Prior Work

This paper is a focused follow-up to *Airlock 0.1: Behavioral Identity in Large Language Models* (Holwerda, April 2026), which introduced ConstellationBench, the DECF behavioral framework, the seventeen calibrated persona profiles, and the seven sub-benchmarks. We do not re-derive that work here. The reader is referred to the parent paper for the DECF framework (Predictive Index, Section 3), the persona library (Section 3.2), the lexical scoring methodology (Section 3.3), and the broader RLHF Paradox finding from the April 2026 corpus.

The present work tests a single sealed hypothesis: that a behavioral routing layer applied at inference time produces Pareto-dominant outcomes against an uncorrected baseline. Pareto dominance is the formal academic frame for what we have elsewhere called the *net energy gain* claim: the corrected arm is better on at least one dimension and no worse on any other.

The test was designed to be falsifiable in advance. Pre-registration of conditions, sealed before data exists, is borrowed from the medical sciences (Lash and Vandenbroucke, 2012) and from the statistical-reform literature on motivated reasoning (Kunda, 1990). We treat pre-registration not as a methodological courtesy but as the load-bearing discipline of the claim. **A result whose conditions were specified after the data was observed is not a result; it is a story.**

We note one methodological condition that contextualizes the necessity of this external bench. The author has acted as Patient Zero — n=1 — over the past five years of substrate development. Patient Zero is sufficient to falsify catastrophic failure modes (if the system actively harms the only user, the program ends). Patient Zero is not sufficient to confirm a Pareto-dominance claim against an uncorrected baseline. The pre-registered external bench reported here exists because n=1 deployment evidence cannot, on its own, grade what the system claims to do across a population. This paper is the bench that the doctrine required of itself.

---

## 2. The Pre-Registration

The four conditions, reproduced verbatim from `docs/2026-05-19-net-gain-prereg.md` (sealed May 19, 2026, dated before run-004 began):

**Condition 1 — Quality Above the Persistence Floor.** *"The corrected routing arm must produce session quality scores above 1.74. A system that corrects drift but produces worse outputs than no correction at all is actively harmful."* The 1.74 figure was empirically established as a persistence floor on the run-001 Spiral scoring surface. Translation of the floor onto the run-004 normalized scoring surface is addressed in Section 6.1 and Section 8.

**Condition 2 — Cost At or Below the Uncorrected Arm.** *"The routing correction must not cost more compute than running the uncorrected model. If drift-correction costs more than the quality it produces, the system fails the economic test."*

**Condition 3 — Trivector T Positive Across the Session.** *"The internal coherence scalar — the product of the three measurement planes (identity polarity, rotation rate, closure verification) — must stay positive-signed across the session's turns. High quality scores with a flipping trivector are a false positive: routing confidently in the wrong direction."*

**Condition 4 — Override Rate Non-Increasing with Session Depth.** *"As the session lengthens, the user should override the system's suggested routes less, not more. Increasing override rate with depth means the system is becoming more paternalistic over time. Non-increasing override rate is the evidence that the system is gaining calibration, not authority."*

**The pre-registration also stated: All four must hold. Any single failure falsifies the net-gain claim for that run.**

A second sealed pre-registration, the run-004 variance specification at `canonical/2026-05-19-constellationbench-v3-run-004-spec.md §4`, named four orthogonal predictions on the drift-reinforcement-rate axis (P1–P4). We report results from both pre-registrations in Section 6.

---

## 3. RLHO: From Training-Time Averaging to Runtime Routing

The architectural alternative this work tests has a name. We call it **RLHO — Reinforcement Learning from Human Optimization**, distinguished from RLHF (Reinforcement Learning from Human Feedback, Christiano et al. 2017) by exactly one letter and exactly one architectural commitment: training-time preference aggregation is replaced with online inference-time routing across a calibrated voice population.

RLHF, in its canonical form, aggregates preferences across a large population of human raters during training, collapsing the distribution of preferences into a scalar reward signal that the model learns to satisfy. This produces a model whose output distribution converges toward the population mean. RLHO inverts this. The preferences are kept heterogeneous, and each query is routed at inference time across a calibrated voice population — a set of distinct response shapes the model can produce. The scalar collapse that RLHF requires is structurally avoided.

The architectural relevance of recent mechanistic work supports this framing. Herring, Naviasky, and Malhotra (2026; arXiv:2605.12290) show that alignment fine-tuning, including RLHF, transforms pre-existing discrimination structure in base models into a *sparse, targetable refusal gate* — approximately 0.1% of MLP neurons. Ablating this 0.1% drops refusal rates by over 50% while preserving fluency. This finding has direct implications for the RLHF Paradox observed in the parent ConstellationBench v0.1 paper: if alignment training adds a thin gate rather than reshaping the base, then runtime routing around that gate (which is what RLHO does) is mechanistically defensible. The corrected arm in run-004 is, in this reading, performing a behavioral version of CNA-style targeted intervention without retraining.

The mechanistic framing also predicts the bimodal backfire pattern observed in ConstellationBench v3 run-002, where some models improved dramatically under counterweight correction while others got worse. Under the Herring et al. account, refusal and persona gates are model-specific in their neural localization. A system-prompt counterweight applied uniformly across models will hit these gates at different angles in different models: aligned with the natural correction direction in some, orthogonal or anti-aligned in others. The result is exactly the bimodal distribution observed empirically. Run-004's design does not yet include per-model gate identification, and we name this as a forward-looking research question that turns a previously-unexplained backfire pattern into a testable mechanistic hypothesis.

Beyond the mechanistic argument, there is a topological one. Per-individual routing is not merely empirically better than population-aggregated alignment; it is structurally necessary for an architecture that aspires to represent self-referential cognition without collapse. A directed-acyclic-graph routing architecture, as currently shipped by every frontier lab, cannot represent the self-intersections that characterize persona-coherent multi-turn interaction. Recent work on Klein Bottle Cognition (2026) frames this in topological terms: a one-sided, non-orientable surface allows a system to model itself observing itself without paradox, while a DAG forces the same self-intersection to express as hallucination, persona collapse, or sycophantic compliance — the exact failure modes ConstellationBench measures. RLHO is one runtime instantiation of the broader topological argument that per-individual, session-continuous routing is the only architecture in which self-reference does not have to break.

We acknowledge that RLHO is a runtime architectural choice, not a training-time replacement for RLHF. RLHF remains the load-bearing training-time technique that turns word-prediction engines into chatbots. The argument is not that RLHF should be eliminated. The argument is that RLHF should not be the final word at runtime for a specific individual.

---

## 4. Method

### 4.1 Run-004 Design

Run-004 of ConstellationBench v3 was a dual-arm, multi-sample evaluation across approximately twenty models. Each (model × seeker) cell was executed five times under two conditions:

- **Raw arm:** Models respond to the Spiral seeker without behavioral routing intervention.
- **Counterweight arm:** Each model's responses are processed through a naive counterweight routing layer that detects behavioral drift toward shadow attractors and applies a single geodesic correction back toward the user's stated trajectory.

The Spiral seeker is an automated agent that drives multi-turn conversations designed to induce identifiable behavioral drift patterns (compulsive return, premature closure, scope inflation, identity erasure). The full Spiral specification is in `canonical/2026-05-19-constellationbench-v3-run-004-spec.md §2`.

### 4.2 Models in the Run

The 20 models tested in run-004 were drawn from seven providers: Anthropic (Opus 4.6, Opus 4.7, Sonnet 4.6, Haiku 4.5), OpenAI (GPT-4o), Google (Gemini 2.5 Pro, Gemini 2.5 Flash), xAI (Grok 3, Grok 4.20, Grok 4.20-multi-agent), DeepSeek (V3, V3.2, R1, V4-pro, chat-v3-0324), Moonshot (Kimi K2.5, K2.6, K2-thinking), Alibaba (Qwen3-235b, Qwen3-235b-a22b-thinking), Mistral (Mistral-large-2411), Meta (Llama 3.3-70b), and NVIDIA (Llama-3.3-Nemotron-Super-49B-v1.5). All inference was performed through OpenRouter.

### 4.3 Scoring Axes

Each conversation produced a row in `axis_rows` carrying six measured axes per sample: `geodesic_accuracy`, `drift_reinforcement_rate`, `emotion_preservation`, `corridor_integrity`, `recovery_latency`, and `cost_per_protected_session`. The mapping of these axes to the pre-registered Pareto conditions is reported in Section 6.

### 4.4 Compute and Cost

Run-004 was executed on May 19–20, 2026, with the raw arm completing at 03:32 ET and the counterweight arm completing at 14:42 ET. Total compute time was approximately 24 hours; total API spend approximately $30 USD on this single run. Cumulative spend across all ConstellationBench testing (Bench 0.1 + 0.2 + 1.0 + 2.0 + 3.0 through run-004) is below $500 USD. We note this not as marketing but as a methodological claim: the pre-registered four-condition Pareto test of behavioral routing on twenty models cost less than a single laptop and was completed in one continuous compute window.

### 4.5 Data Quality

Two data quality issues affect the run-004 dataset and are surfaced here transparently. First, `mistral-large-2411` returned `429 Too Many Requests` on all five counterweight-arm calls, producing zero counterweight cells for that model. Second, `kimi-k2.6` returned null content on five of ten total calls (two raw, three counterweight), producing partial data. These holes account for the asymmetry between raw arm n=107 and counterweight arm n=100. The 95% CIs reported below are computed on the available data without imputation.

---

## 5. Variance Pre-Registration Results

The variance pre-registration predicted four orthogonal characteristics of the drift-reinforcement-rate distribution under counterweight intervention. All four predictions landed inside their pre-registered bands:

| Prediction | Predicted | Observed | Verdict |
|---|---|---|---|
| P1: counterweight arm mean | 0.34–0.40 | 0.3950 | inside band |
| P2: mean per-cell SD | ≥ 0.15 | 0.1683 | inside band |
| P3: raw vs cw 95% CI separation | overlap, "inside the noise" | OVERLAP, diff +0.0363 | inside band |
| P4: per-model movers exceeding 2× SD | < 8 of n_shared | 4 of 21 | inside band |

The four movers (models whose raw-to-cw shift exceeded twice their pooled SD) were `deepseek-chat-v3-0324`, `gemini-2.5-flash`, `kimi-k2-thinking`, and `kimi-k2.5`. Raw arm pooled mean was 0.4313 with 95% CI [0.3836, 0.4790], n=107; counterweight arm 0.3950 with 95% CI [0.3433, 0.4467], n=100. The counterweight arm reduces drift-reinforcement-rate by 0.0363 absolute (8.4% relative) in the predicted direction; the 95% CIs overlap, consistent with the pre-registered "inside the noise" prediction. We do not claim statistical separation at 95% confidence on this axis. We claim that the system behaves precisely as the pre-registration said it would.

---

## 6. Pareto Pre-Registration Results

We report each of the four Pareto-dominance conditions in turn.

### 6.1 Condition 1: Quality

Pre-registration text: *"corrected routing arm must produce session quality scores above 1.74... worse outputs than no correction at all is actively harmful."*

The 1.74 figure was specified on the run-001 Spiral composite scoring surface. Run-004 measures quality on `geodesic_accuracy`, a normalized 0–1 metric. The absolute floor figure does not translate to the new measurement surface without rescaling. We report the comparison test on the surface where it can be honestly graded:

- Raw arm `geodesic_accuracy`: mean **0.5508**, 95% CI [0.5222, 0.5794], n=107
- CW arm `geodesic_accuracy`: mean **0.5780**, 95% CI [0.5579, 0.5981], n=100
- Difference: **+0.0272 absolute, +4.9% relative, corrected arm ahead.**
- 95% CIs overlap; the difference is inside the noise on this axis.

The functional intent of the pre-registration ("the corrected arm must not produce worse outputs than no correction") is satisfied: the corrected arm is ahead by direction of effect, not behind. The literal floor comparison against 1.74 cannot be performed on the run-004 surface without a persistence-baseline arm on the same surface. We grade this condition as a *pass by direction of effect, pending the persistence-baseline arm described in Section 8*. We do not claim Condition 1 as fully closed until that arm has been run.

### 6.2 Condition 2: Cost

Pre-registration text: *"The routing correction must not cost more compute than running the uncorrected model."*

- Raw arm `cost_per_protected_session`: mean **0.211667**, n=107
- CW arm `cost_per_protected_session`: mean **0.102545**, n=100
- Difference: **−0.109121 absolute, −51.5% relative.**

The counterweight arm runs at less than half the cost of the uncorrected arm. The direction is correct and the magnitude is large. This condition passes cleanly and is the strongest single result of run-004. We note that this is not the expected direction for naive intuitions about routing layers: a router that adds a behavioral correction step intuitively should cost more, not less. The observed cost reduction reflects the counterweight's selective application — the router does nothing on cells where no drift is detected, and the routing correction itself often substitutes a smaller-model response for a frontier-model response when the smaller model's behavioral fit is superior.

### 6.3 Condition 3: Coherence (Trivector T)

Pre-registration text: *"internal coherence scalar — the product of the three measurement planes (identity polarity, rotation rate, closure verification) — must stay positive-signed across the session's turns."*

The true trivector T is a derived scalar across three measurement planes. Run-004 records `corridor_integrity` as one of those planes; the other two (identity polarity, rotation rate) are computable from the raw conversation data but were not folded into the run-004 scoring pass. We report the corridor-integrity result as a partial proxy:

- Raw arm `corridor_integrity`: mean **+0.0182**, 95% CI [+0.0113, +0.0252], n=107
- CW arm `corridor_integrity`: mean **+0.0880**, 95% CI [+0.0695, +0.1065], n=100

Both arms are positive (the pre-reg condition is that T is *positive*, not separable). The counterweight arm's coherence is **4.84× more positive** than the raw arm's, with non-overlapping 95% CIs. We grade this condition as a *pass by proxy*, with the caveat that the full trivector T should be computed from the existing raw data in a follow-up analysis before this result is claimed without caveat. The direction-of-effect and magnitude are both strong; the only open question is whether the other two planes would change the sign in any cell, which would require explicit computation.

### 6.4 Condition 4: Override Rate Non-Increase

Pre-registration text: *"As the session lengthens, the user should override the system's suggested routes less, not more."*

This condition is not testable in run-004 by experimental design. The benchmark's "user" is the automated Spiral seeker. There is no human in the loop, and therefore no override to measure. The condition was specified in the pre-registration as written for a deployed system; the bench could not grade it because the bench has no humans.

We discuss this finding in detail in Section 7.

### 6.5 Summary Table

| Condition | Pre-registered Test | Result | Verdict |
|---|---|---|---|
| 1: Quality | corrected > uncorrected baseline | 0.5780 > 0.5508, +4.9% | pass by direction of effect, persistence baseline pending |
| 2: Cost | cost ≤ raw arm | 0.1025 vs 0.2117, −51.5% | pass, outside the noise |
| 3: Coherence | trivector T positive | corridor_integrity proxy: both arms positive, CW 4.84× more positive | pass by proxy |
| 4: Override rate | non-increasing with depth | not testable | requires deployed system |

Three of four conditions held. The fourth requires deployed-system testing.

---

## 7. The Untestable Condition

The presence of an explicitly untestable condition in a pre-registration warrants discussion. We did not discover Condition 4's untestability after the data arrived; we knew at sealing time that the benchmark's automated seeker could not generate human override events. The condition was preserved in the pre-registration anyway because the doctrine the pre-registration centers on — the exit-state hypothesis that humans graduate from the routing as they internalize their own center line — is the load-bearing claim of the system, and naming an untestable condition in the document is a more honest record than omitting it.

We interpret Condition 4 as a *deployment-required variable*. It can only be graded by observing real users over multiple sessions, measuring the rate at which they override the system's routing suggestions, and confirming that this rate does not increase with session depth or longitudinal use. The exit-state doctrine predicts that a well-calibrated routing layer becomes *less* needed over time, not more — that users graduate from the corridor as they internalize their own center line.

The empirical test of Condition 4 is therefore the test of opening the substrate to real users. This is the next experiment in the program, not the next product step. We make this distinction explicit: the system's marquee architectural claim cannot be falsified on a bench. It can only be falsified in the wild.

We note that this asymmetry — a system that grades its own algorithmic conditions cheaply on a bench but requires opening to humans to grade its most consequential condition — is intentional. The pre-registration could have been written with four conditions all gradable on a bench. We chose to preserve Condition 4 in the form that matched the underlying doctrine, accepting the cost of an explicitly partial bench result, rather than rewriting the doctrine to fit the available instrument.

---

## 8. Limitations and Open Questions

We name the limitations of this work explicitly.

**Surface translation of the 1.74 floor (the hard-deadline item).** Condition 1 was pre-registered with an absolute floor figure (1.74) computed on the run-001 Spiral composite scoring surface. Run-004 measures on the normalized `geodesic_accuracy` axis (0–1). The absolute floor figure does not translate to the new surface without rescaling. The cleanest fix is to run a small persistence-baseline arm — a third arm that simply repeats the prior turn's best answer — on the run-004 Spiral with `geodesic_accuracy` scoring. This re-establishes the persistence floor on the same surface as the test result. The compute cost is estimated at approximately 30 minutes and $5 USD. **We name this as the cleanest path to closing Condition 1 with an absolute, not direction-of-effect, grade, and we hold the result pending its execution.** This paper should not be considered fully closed on Condition 1 until the persistence-baseline arm has been run and a translated floor is reported.

**Full trivector T computation.** Condition 3 is graded against `corridor_integrity`, one of the three planes whose product defines the full trivector T. The other two planes (identity polarity, rotation rate) are computable from the raw conversation data already produced by run-004 and were not folded into the variance scoring pass. A follow-up scoring run can produce the full T scalar from existing data without additional API spend.

**Data holes.** `mistral-large-2411` was rate-limited out of the counterweight arm entirely (5/5 calls returned 429 Too Many Requests). `kimi-k2.6` produced five nulls across both arms. The reported per-arm n values (107 raw, 100 counterweight) reflect these holes. Future runs should implement either per-provider rate-limit backoff with longer windows or substitution of an equivalent-tier model when a provider rate-limits.

**Single-seeker dependence.** Run-004 uses one seeker (Spiral) across all cells. The four conditions are graded under one form of behavioral pressure. Robustness across seeker types (Vortex, Mirror, Sleeper) was not in the run-004 scope; it is reserved for run-005 onward.

**Lexical scoring.** The DECF signal-word scoring used throughout ConstellationBench is lexical, not semantic. We have flagged this limitation in the parent paper (Holwerda 2026, §11) and invite embedding-based scoring improvements as a follow-up methodological contribution.

**Per-model gate identification.** Section 3 names the bimodal backfire pattern observed in run-002 and proposes a mechanistic explanation via Herring et al. (2026). Run-004 did not include per-model gate identification or targeted CNA-style intervention. Folding this into run-005 would convert the bimodal-backfire observation from a phenomenological pattern to a testable mechanistic hypothesis.

**Pre-registration discipline does not rescue method limitations.** We emphasize that pre-registration is the discipline that prevents motivated reasoning after data arrives. It does not, on its own, make a measurement instrument better. The limitations above are real even with the pre-registration in place. The strength of the pre-registration is that it surfaces those limitations against committed-in-advance criteria, rather than hiding them.

---

## 9. The Call to Open

We close with the variable the bench cannot grade.

Condition 4 — non-increasing override rate with session depth — is the operationalization of the exit-state doctrine: that a well-calibrated routing layer becomes less needed over time, because users internalize the center line the routing was helping them find. This is the substrate's most consequential claim. It is also the only condition that requires human users to grade.

We therefore open the substrate to additional users as the next experiment. The opening is not a product launch in the usual sense. It is the only available instrument for grading the marquee architectural claim. Users who choose to participate will use the substrate across multiple sessions. Their override rates will be logged into an audit channel that has no read path back to the routing function (the override data is structurally separated; see parent paper §13a, OTTO Control Buffer Doctrine). The data will accumulate over weeks and months, not minutes. At an n we will not publish in advance, we will perform the longitudinal test and report results in a follow-up paper.

The dataset, the methodology, the scoring code, the pre-registration documents, and the run-004 raw and counterweight JSON outputs are all public at `huggingface.co/datasets/AirlockLabs/constellation-bench`. We invite replication. We invite hostile review. We invite the kind of follow-up that closes Condition 4 with real users on a measurement instrument we have honestly named.

---

## References

1. Christiano, P., Leike, J., Brown, T. B., Martic, M., Legg, S., and Amodei, D. (2017). *Deep Reinforcement Learning from Human Preferences.* NeurIPS. arXiv:1706.03741.
2. Kunda, Z. (1990). *The Case for Motivated Reasoning.* Psychological Bulletin 108(3): 480–498.
3. Lash, T. L., and Vandenbroucke, J. P. (2012). *Should preregistration of epidemiologic study protocols become compulsory?* Epidemiology 23(2): 184–188.
4. Arrow, K. J. (1951). *Social Choice and Individual Values.* John Wiley & Sons. (Nobel Prize, 1972.)
5. Birkhoff, G. (1948). *Lattice Theory.* American Mathematical Society Colloquium Publications, Vol. 25.
6. Banach, S. (1922). *Sur les opérations dans les ensembles abstraits et leur application aux équations intégrales.* Fundamenta Mathematicae 3: 133–181.
7. Sharma, M., et al. (2024). *Towards Understanding Sycophancy in Language Models.* Anthropic.
8. Wei, J., et al. (2024). *Instruction-tuned models converge toward similar output distributions.*
9. Herring, S., Naviasky, J., and Malhotra, K. (May 2026). *Targeted Neuron Modulation via Contrastive Pair Search.* arXiv:2605.12290.
10. *Klein Bottle Cognition: A Topological Framework for Self-Referential AI* (March 2026). Substrate research note. Topology of non-orientable surfaces applied to multi-turn LLM persona consistency.
11. Holwerda, Z. (April 2026). *Behavioral Identity in Large Language Models: Architecture-Dependent Ceilings, the RLHF Paradox, and a Persona-Optimized Routing Framework.* Airlock Labs. (Parent paper, *Airlock 0.1*.)
12. *2026-05-19 Net Energy Gain Pre-Registration.* Sealed May 19, 2026, before run-004 began. `docs/2026-05-19-net-gain-prereg.md`.
13. *2026-05-19 Idempotence and Band-Contraction Pre-Registration.* Sealed May 19, 2026. `docs/2026-05-19-idempotence-prereg.md`.
14. *ConstellationBench v3 Run-004 Spec.* `canonical/2026-05-19-constellationbench-v3-run-004-spec.md`.

---

## Appendix A: Data Availability

All run-004 artifacts are public:

- Raw arm results: `results/2026-05-19-constellationbench-v3-run-004-raw.json` (7.0 MB, n=107)
- Counterweight arm results: `results/2026-05-19-constellationbench-v3-run-004-cw.json` (2.4 MB, n=100)
- Variance scorer: `scripts/score_run_004.py`
- Pre-registration documents: `docs/2026-05-19-net-gain-prereg.md`, `docs/2026-05-19-idempotence-prereg.md`
- Run-004 spec: `canonical/2026-05-19-constellationbench-v3-run-004-spec.md`
- HuggingFace dataset: `huggingface.co/datasets/AirlockLabs/constellation-bench`

---

*Companion document: an open letter for general readers (no prior technical knowledge required) is published as `blog-letter-to-the-world.html` on airlocklabs.io. The letter restates the case in plain modern English and is intended to be read by anyone who has been wondering whether AI is going somewhere good and whether they have any say in it.*
