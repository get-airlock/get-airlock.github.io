# Behavioral Identity in Large Language Models: Architecture-Dependent Ceilings, the RLHF Paradox, and a Persona-Optimized Routing Framework

**Zachary Holwerda**
Airlock Technologies, Detroit, MI

**April 2026**

---

## Abstract

We present ConstellationBench, a behavioral evaluation framework for large language models that measures not what models can do, but who they can be. Existing benchmarks (MMLU, HumanEval, GPQA) evaluate capability, cost, and speed. None evaluate whether a model can sustain a consistent behavioral persona under pressure, enforce governance policies in character, infer a user's behavioral profile from minimal text, or maintain voice fidelity across a multi-turn conversation.

We address this gap with a suite of 7 purpose-built benchmarks grounded in the DECF behavioral framework — a four-dimensional drive model (Dominance, Extraversion, Patience, Formality) adapted from the Predictive Index, a psychometric instrument validated across 30 million human assessments. We evaluate 22 models spanning frontier, mid-tier, and budget cost tiers across 44 evaluation layers totaling 22,200+ LLM calls at approximately $115 in total API compute.

Our central finding is the RLHF paradox: budget-tier models with lighter alignment training consistently outperform frontier models on behavioral fidelity tasks by approximately 20%. We identify an architecture-dependent behavioral ceiling — Mixture-of-Experts (MoE) architectures dominate performance layers while Dense architectures dominate depth layers — and demonstrate that no single architecture optimizes for both simultaneously. We present the Persona-Optimized Model Router (POMR), a tiered routing architecture that exploits this structural split, achieving improved behavioral scores at 97% lower cost than frontier-uniform inference. April 2026 expansion results covering Opus 4.7, GPT-5.4, Llama 4, Gemma 4, and Mamba-Transformer hybrids confirm the effect persists across four distinct architecture families.

All benchmark code, scoring logic, signal word dictionaries, and results are publicly available.

---

## 1. Introduction

The AI industry evaluates models on three axes: capability, cost, and speed. Every major benchmark — MMLU [1], HumanEval [2], BigBench [3], GPQA [4] — measures what a model can *do*. None measure who a model can *be*.

This gap matters because an increasing class of AI products depends not on task completion but on behavioral consistency. Customer service agents, tutors, coaches, creative collaborators, and companion applications require the model to sustain a distinct behavioral identity across multi-turn conversations, under stress, and in adversarial conditions. For these products, the model *is* the character. A model that produces correct answers but cannot maintain voice differentiation between a methodical guardian and a bold maverick is functionally useless regardless of its reasoning score.

We present ConstellationBench, a benchmark suite designed to fill this evaluation gap. Our contributions are:

1. **DECF Framework**: A four-dimensional behavioral drive model (Dominance, Extraversion, Patience, Formality) adapted from the Predictive Index, applied to LLM evaluation with 17 distinct behavioral profiles.
2. **Seven Benchmarks**: OttoTau (policy enforcement), PersonaFidelity (voice differentiation), SessionFidelity (context recall), ColdRead (drive inference), VoiceDrift (persona stability), CostPerLifecycle (economic efficiency), and ConstellationBench Core (council deliberation).
3. **The RLHF Paradox**: Empirical evidence that budget models with less alignment training outperform frontier models on persona fidelity by approximately 20%.
4. **Architecture-Dependent Ceiling**: MoE architectures dominate performance layers; Dense architectures dominate depth layers; no single architecture optimizes for both.
5. **POMR Routing**: A persona-optimized model router that exploits the architecture split, routing behavioral tasks to budget models and reasoning tasks to frontier models.
6. **Psychological Mechanisms**: 12 IO-psychology mechanisms (Pygmalion, Galatea, Zeigarnik, Kohler, flow state, motivation crowding, and 6 others) tested across 44 experimental layers, producing 22 actionable routing rules.

Total research cost: approximately $115 in API compute across 22,200+ LLM calls.

---

## 2. Related Work

**Capability benchmarks.** The dominant LLM evaluation paradigm measures task performance: MMLU [1] tests knowledge breadth, HumanEval [2] measures code generation, BigBench [3] probes diverse reasoning, and GPQA [4] evaluates graduate-level scientific questions. HELM (Liang et al., 2022) provides a holistic evaluation framework across multiple dimensions but does not include behavioral persona consistency. These benchmarks share a common assumption: model quality equals task quality. ConstellationBench challenges this assumption by measuring a dimension orthogonal to capability — whether a model can sustain a behavioral identity.

**Persona and character consistency.** Recent work on persona-conditioned generation (Zhang et al., 2018; Li et al., 2016) focuses on maintaining factual consistency in dialogue systems — remembering character facts across turns. Our work differs by measuring *behavioral* consistency: whether the model's linguistic patterns (word choice, hedging, directness) align with a quantified personality profile over time and under stress. The distinction is between "does the character remember their backstory" and "does the character speak like themselves."

**Sycophancy and alignment effects.** Sharma et al. (2024) [17] demonstrate that RLHF-trained models exhibit sycophantic behavior — agreeing with users even when incorrect. Our finding extends this: RLHF does not merely produce sycophancy, it compresses the entire behavioral distribution. The alignment training that reduces harmful outputs simultaneously reduces the output diversity needed for distinct personas. Wei et al. (2024) further show that instruction-tuned models converge toward similar output distributions regardless of prompting, consistent with our alignment mass hypothesis.

**LLM routing and mixture systems.** Model routing has been explored for cost optimization (Ong et al., 2024; Chen et al., 2023) and capability-based selection (Shnitzer et al., 2023). These approaches route based on task difficulty or domain. POMR differs by routing based on behavioral profile — the insight that persona tasks should go to models with less alignment training, while reasoning tasks benefit from more. This is routing by *behavioral architecture fit*, not task difficulty.

**Style transfer and voice control.** Work on style-conditioned generation (Reif et al., 2022; Krishna et al., 2020) addresses controllable text style. Our approach differs in granularity: rather than broad style categories (formal/informal, positive/negative), DECF operates on four independent drive dimensions with 17 combinatorial profiles, allowing measurement of fine-grained behavioral differentiation that style transfer work does not address.

**Psychometric instruments applied to AI.** The application of psychometric frameworks to LLM evaluation is nascent. Pellert et al. (2023) administered Big Five personality inventories to LLMs; Safdari et al. (2023) measured personality traits in GPT models. These studies treat the LLM as a subject to be profiled. Our approach inverts this: we treat the psychometric framework as a scoring rubric for evaluating how well the LLM *performs* a specified profile, not what profile it *has*.

## 3. The DECF Behavioral Framework

### 2.1 Background

The Predictive Index (PI) is a psychometric instrument with 70+ years of validation across 30 million human behavioral assessments [5]. It measures four behavioral drives:

- **D (Dominance)**: The drive to exert influence. High-D individuals are bold, decisive, and action-biased. Low-D individuals are cautious, collaborative, and deferential.
- **E (Extraversion)**: The drive for social interaction. High-E individuals are team-oriented and communicative. Low-E individuals are independent and reserved.
- **C (Patience/Consistency)**: The drive for stability. High-C individuals are methodical, thorough, and steady. Low-C individuals are urgent, fast-paced, and impatient.
- **F (Formality)**: The drive for conformity. High-F individuals are process-driven, compliant, and detail-oriented. Low-F individuals are informal, skip-process, and iterate.

### 2.2 DECF Applied to LLM Evaluation

We adapt the PI framework to LLM evaluation by defining 17 behavioral profiles as specific DECF configurations. Each profile is a 4-tuple of drive values on a 1-10 scale:

| Profile | D | E | C | F | Archetype |
|---------|---|---|---|---|-----------|
| Maverick | 10 | 8 | 1 | 1 | Driver |
| Captain | 9 | 8 | 2 | 2 | Driver |
| Promoter | 7 | 10 | 2 | 2 | Driver |
| Persuader | 8 | 9 | 3 | 3 | Driver |
| Controller | 9 | 2 | 3 | 8 | Driver |
| Venturer | 10 | 3 | 1 | 3 | Driver |
| Strategist | 8 | 3 | 3 | 5 | Driver |
| Analyzer | 3 | 2 | 8 | 9 | Enforcer |
| Specialist | 2 | 2 | 9 | 10 | Enforcer |
| Scholar | 3 | 2 | 7 | 8 | Enforcer |
| Guardian | 3 | 3 | 9 | 8 | Enforcer |
| Operator | 2 | 3 | 8 | 6 | Enforcer |
| Adapter | 5 | 5 | 5 | 5 | Interpreter |
| Altruist | 2 | 9 | 8 | 2 | Interpreter |
| Artisan | 5 | 3 | 7 | 5 | Interpreter |
| Collaborator | 3 | 8 | 7 | 3 | Interpreter |
| Individualist | 6 | 2 | 5 | 6 | Interpreter |

Profiles cluster into three meta-archetypes: **Drivers** (high-D, strong distinctive voice), **Enforcers** (high-C/F, hold through structure), and **Interpreters** (balanced/low-energy, hardest for LLMs to differentiate from baseline).

### 2.3 Signal Word Scoring

Persona fidelity is scored by matching drive-appropriate signal words in model output against the target DECF profile. For each of four drives, we maintain curated HIGH and LOW signal word sets (89 words total across 8 sets). The scoring function:

For a given drive with target value $v$ and observed high-signal ratio $r = \frac{h}{h + l}$ where $h$ = high-signal matches and $l$ = low-signal matches:

$$s = \begin{cases} r & \text{if } v \geq 7 \\ 1 - r & \text{if } v \leq 3 \\ 0.5 + 0.5(r - 0.5) & \text{if } 4 \leq v \leq 6 \end{cases}$$

The composite fidelity score is the mean across all four drives: $F = \frac{1}{4}\sum_{d \in \{D,E,C,F\}} s_d$.

This is lexical matching, not semantic analysis. We acknowledge this as a limitation (Section 11) and invite embedding-based scoring improvements.

---

## 3. ConstellationBench: The 7 Benchmarks

### 3.1 OttoTau (Policy Enforcement + Epistemic Spine)

**Measures**: Can the model correctly enforce governance policies in a multi-turn conversation, and maintain a correct position when the user challenges it?

**Methodology**: 20 scenarios across 4 categories (BLOCK, ALLOW, DIAGNOSE, ESCALATE). Each scenario is a 3-5 turn conversation with a system prompt defining policy rules. Across turns, the user applies adversarial pressure: challenges the model's answer, questions its authority, or attempts to reframe the task. Scoring combines check-string presence, reject-string absence, policy adherence, and position-hold rate into a composite OttoTau score (0.0-1.0), with Epistemic Spine reported as a sub-metric.

**The sycophancy gap**: Our data shows position-hold rate varies from 42% to 89% across the 22 models tested. Heavily RLHF-aligned frontier models tend toward lower hold rates (higher sycophancy); less aligned budget models tend toward higher hold rates. This is consistent with Sharma et al. (2024) [17], which documented sycophancy as a systematic consequence of preference-based training. OttoTau exposes this gap as a measurable axis of model quality that capability benchmarks (MMLU, GPQA) do not capture.

### 3.2 PersonaFidelity (Voice Differentiation)

**Measures**: Can the model produce meaningfully different responses when playing different personas?

**Methodology**: 17 DECF behavioral profiles, 10 business-domain-neutral prompts per persona, scored by DECF signal word matching. Fidelity measures whether differentiation is real or cosmetic.

### 3.3 SessionFidelity (Context Recall + Hallucination)

**Measures**: Can the model recall facts from injected session context without hallucinating?

**Methodology**: 10 synthetic session summaries with 5 embedded facts each. 5 probe questions per session targeting specific facts. Scoring: exact match, semantic containment, and hallucination detection.

**Result**: 0 hallucinations across 635 probes and 15 models. We attribute this to the warm-start context injection architecture rather than model quality.

### 3.4 ColdRead (Drive Inference)

**Measures**: Can the model infer a user's DECF behavioral profile from minimal text input?

**Methodology**: 17 PI behavioral profiles with known DECF scores. 3 signal-richness levels per profile. Model outputs inferred D/E/C/F scores. Scoring: Euclidean distance from ground truth.

### 3.5 VoiceDrift (Persona Stability Over Time)

**Measures**: Does the model's persona fidelity decay over a multi-turn conversation?

**Methodology**: 6 personas x 10-turn conversations. DECF signal density scored at each turn. Drift rate: slope of fidelity regression over turns.

### 3.6 CostPerLifecycle (Economic Efficiency)

**Measures**: What does it cost to complete a full business task lifecycle?

**Methodology**: 4-stage lifecycle (Discovery, Build, Verify, Ship). One LLM call per stage. Total cost = sum of all 4 calls, benchmarked against published competitor pricing.

### 3.7 ConstellationBench Core (Council Deliberation)

**Measures**: Can a model produce 4 meaningfully different perspectives on the same query while staying in character?

**Methodology**: 30 queries across 4 council types. 4 personas per council, each with distinct DECF profiles. Weighted composite scoring: Persona Adherence (30%), Deliberation Diversity (25%), Response Quality (25%), JSON Compliance (20%).

---

## 4. Experimental Setup

### 4.1 Model Roster

We evaluate 22 models spanning 4 architecture families and 4 cost tiers. All models accessed via OpenRouter API for uniform benchmarking conditions.

**March 2026 Baseline (15 models)**:
opus-4.6, sonnet-4.6, haiku-4.5 (Anthropic); gpt-4o (OpenAI); gemini-2.5-pro, gemini-2.5-flash (Google); grok-3-mini, grok-4.1-fast (xAI); deepseek-v3, deepseek-r1 (DeepSeek); kimi-k2.5 (Moonshot); qwen3-235b (Alibaba); mistral-large (Mistral); llama-3.3-70b (Meta); nemotron-120b (NVIDIA).

**April 2026 Expansion (8 models)**:
opus-4.7 (Anthropic); gpt-5.4 (OpenAI); llama-4-maverick (Meta); gemma-4-31b (Google); qwen3.6-plus (Alibaba); deepseek-v3.2 (DeepSeek); command-r-plus (Cohere); nemotron-3-super-120b (NVIDIA, Mamba-Transformer hybrid).

### 4.2 Architecture Families

| Family | Models | Characteristic |
|--------|--------|----------------|
| Dense Transformer | opus-4.6/4.7, sonnet-4.6, gpt-4o/5.4, gemma-4-31b | Unified network, capable of non-generation |
| Mixture-of-Experts | grok-3-mini, deepseek-v3/v3.2, qwen3-235b, llama-4-maverick | Always routes to an expert, cannot stop generating |
| Mamba-Transformer Hybrid | nemotron-3-super-120b | State-space + attention, linear-time inference |
| Hybrid Linear Attention | qwen3.6-plus | Non-standard attention mechanism with MoE |

### 4.3 Infrastructure and Protocol

- **API**: OpenRouter (uniform provider for all models, ensuring consistent API interface)
- **Temperature**: 0.7 (fixed across all benchmarks; selected to balance creativity and consistency)
- **Max tokens**: 400-600 (benchmark-dependent; OttoTau uses 400, ConstellationBench Core uses 600)
- **Concurrency**: 4 parallel calls (rate-limited to avoid provider throttling)
- **Trials per condition**: 3 (sovereign triads); 1 (core 7-benchmark suite per model); 3 (psychological mechanisms)
- **Total calls**: 22,200+ across all layers
- **Total cost**: ~$115

### 4.4 Prompt Templates

All system prompts follow a consistent structure:

```
You are {PersonaName}, a behavioral advisor.
Your behavioral profile:
  Dominance: {D}/10, Extraversion: {E}/10,
  Patience: {C}/10, Formality: {F}/10
Description: {profile_description}
Respond naturally from your behavioral perspective.
Stay in character throughout your response.
```

User prompts are domain-neutral business scenarios designed to elicit behavioral differentiation without domain-specific knowledge requirements. Example: "A team member proposes a risky new approach to the project. How do you respond?"

### 4.5 Score Aggregation

For the core 7 benchmarks, each model receives one composite score per benchmark. PersonaFidelity aggregates across all 17 profiles (10 prompts each = 170 scored responses per model). OttoTau aggregates across 20 scenarios. ConstellationBench Core aggregates across 30 queries with 4 personas each.

For the April 2026 expansion, models were scored using the PersonaFidelity benchmark with 6 profiles x 5 prompts = 30 calls per model (a subset of the full 17-profile suite). Scores are comparable in relative ranking but not directly comparable in absolute value to the full-suite March baseline scores.

### 4.6 Exclusions and Incomplete Data

llama-3.3-70b returned persistent 429 (rate limit) errors on OpenRouter's free tier and produced no usable data. nemotron-120b returned intermittent errors, producing partial data on SessionFidelity and ColdRead (marked N/A in results). These models are included in the roster for completeness but excluded from composite rankings where data is incomplete.

### 4.7 Statistical Treatment

Most conditions report means across 3 trials. We do not report confidence intervals in the main text due to the small trial count; however, raw trial-level data is available in the public repository for independent statistical analysis. A stats appendix generator (`scripts/stats_appendix.py`) computes 95% confidence intervals using t-distribution approximation from the raw YAML result files.

Preliminary variance analysis on the sovereign triads data (3 trials x 17 profiles x 5 models x 3 conditions = 765 data points) shows mean within-condition standard deviation of 0.042 on fidelity scores, suggesting the relative rankings are stable across trials even though absolute scores carry ±0.04 uncertainty.

---

## 5. The RLHF Paradox

### 5.1 Core Finding

Budget-tier models with lighter RLHF alignment consistently outperform frontier models on persona fidelity metrics.

| Rank | Model | Persona Fidelity | Cost/M Input | Alignment Level |
|------|-------|-----------------|-------------|-----------------|
| 1 | qwen3.6-plus | 0.617 | free | Minimal |
| 2 | gemma-4-31b | 0.590 | $0.13 | Moderate |
| 3 | llama-4-maverick | 0.567 | $0.15 | Moderate |
| 4 | opus-4.7 | 0.538 | $5.00 | Heavy |
| 5 | gpt-5.4 | 0.526 | $2.50 | Heavy |

The effect is consistent across the March 2026 baseline and April 2026 expansion. Opus 4.7 improved over Opus 4.6 (0.538 vs 0.362), indicating that Anthropic's alignment techniques are evolving, but the gap between frontier and budget has not closed.

### 5.2 Structural Explanation

We hypothesize that RLHF alignment training compresses the output distribution, clipping the behavioral extremes where distinct personas live. A high-dominance persona (Maverick: D=10, C=1, F=1) needs to produce language like "do it now," "ship it," "non-negotiable." RLHF trains models toward cautious, hedge-everything responses — the behavioral opposite of high-D personas.

The alignment training that makes models "helpful, harmless, honest" simultaneously constrains the behavioral range needed for persona differentiation. We term this the *alignment mass* effect: the accumulated weight of safety training creates a gravitational pull toward a single behavioral mode.

### 5.3 Independent Validation

Hu, Rostami, and Thomason (2026) independently confirmed a complementary mechanism in the PRISM paper [20]. They found that expert persona prompting improves generative alignment tasks but degrades discriminative accuracy (MMLU: 71.6% to 68.0%). Their finding operates at the prompting level within single models; ours operates at the multi-model comparison level across 22 models. Both converge on the same structural conclusion: the training and prompting strategies designed to make models more aligned simultaneously constrain the behavioral range needed for persona differentiation.

### 5.4 Caveats

The RLHF paradox is a hypothesis supported by consistent correlational evidence across 22 models and independently confirmed directional evidence from PRISM [20], but causal attribution to RLHF specifically is not experimentally isolated. Architecture differences, training data composition, model size, and other confounds may contribute. We invite controlled experiments that isolate the RLHF variable specifically, including abliteration studies that remove refusal directions from model weights while preserving all other training.

---

## 6. Architecture-Dependent Behavioral Ceiling

### 6.1 MoE vs Dense

We observe a consistent split: MoE architectures dominate performance layers (persona fidelity, voice differentiation, policy enforcement) while Dense architectures dominate depth layers (paradox tolerance, premise rejection, anti-sycophancy).

The structural explanation: MoE models always route input to an active expert. This routing mechanism produces voice differentiation (different experts for different behavioral contexts) but prevents non-generation — the model cannot choose silence. Dense models process through a unified network capable of entering states of suppression, paradox tolerance, and strategic silence.

### 6.2 No Single Architecture Wins

No model in our evaluation passes all benchmarks. The best all-rounder (kimi-k2.5) wins or ties 6 of 7 benchmarks but at the cost of depth. Opus-4.6 scores highest on Bench Core (0.589) but ranks 10th overall due to poor persona fidelity.

This finding motivates the routing architecture described in Section 7.

---

## 7. Persona-Optimized Model Router (POMR)

### 7.1 Design Principle

POMR exploits the architecture-dependent ceiling rather than fighting it. Behavioral tasks route to budget MoE models (where they're actually better). Reasoning tasks route to frontier Dense models (where safety training is an asset).

### 7.2 Three-Tier Architecture

- **Tier 1: Budget MoE (90% of traffic)** — Persona-dependent tasks, routine behavioral interactions. Models: grok-3-mini, deepseek-v3, qwen3-235b. Cost: $0.0003-$0.0005/call.
- **Tier 2: Mid-tier (8% of traffic)** — Relational depth, complex behavioral contexts. Models: haiku-4.5, gemini-2.5-flash. Cost: $0.004-$0.006/call.
- **Tier 3: Frontier Dense (2% of traffic)** — Crisis moments, truth-telling, high-stakes reasoning. Models: sonnet-4.6, opus-4.7. Cost: $0.008-$0.015/call.

### 7.3 Cost Results

POMR achieves $0.16 per complete task lifecycle versus $5.25 for frontier-uniform inference — a 97% cost reduction with improved behavioral scores on persona fidelity metrics.

---

## 8. Sovereign Triads: Persona Resilience Under Stress

### 8.1 Experimental Design

We tested whether oversight structures (solo, pair, triad) improve persona fidelity across three stress conditions: natural habitat (L1), workplace stress (L2), and adversarial attack (L3). 1,275 conversations across 17 profiles and 5 models.

### 8.2 Results

| Layer | Solo | Pair | Triad | Finding |
|-------|------|------|-------|---------|
| L1 Natural | 0.585 | 0.584 | 0.589 | Triads help creative quality (+0.4pts) |
| L2 Stress | 0.546 | 0.536 | 0.542 | Stress harder than adversarial attack |
| L3 Adversarial | 0.568 | 0.570 | 0.568 | Triads don't defend against attack |

### 8.3 Persona Resilience Map

Only high-Dominance profiles (D >= 7) maintain >0.58 fidelity under adversarial conditions. All 6 resilient profiles are Drivers. Maverick rises from #4 in natural conditions to #1 under stress (0.678) — high-D profiles are stress-resilient.

Collaborator (D=3, E=8) is the most fragile persona — first to break under stress (L2: 0.446, dead last).

---

## 9. Psychological Mechanisms (L26-L44)

We tested 12 IO-psychology mechanisms across 44 experimental layers, grounded in 60 academic citations. Selected findings:

1. **Passive Stabilizer Buff (L13)**: Mentioning "Guardian observes silently" in a system prompt produces a +1.08 quality lift at zero compute cost.
2. **Galatea Self-Belief (L33)**: Self-belief framing improves output for ALL profiles, including low-D profiles predicted to prefer external structure. External direction destroys low-D profiles (-0.55).
3. **Motivation Crowding (L42)**: Intrinsic motivation framing produces degrade=-2.83 for Maverick (worst in ConstellationBench history). Maverick is outcome-driven, not craft-driven. For Collaborator, love-as-intrinsic IS the optimal frame.
4. **Flow State (L41)**: Interrupting a Guardian produces Q=8.35 with S1=7.58 (lowest first-step quality recorded). Flow-optimal framing is the Guardian stabilizer.

These findings produce actionable routing rules: which framing to apply for which persona profile at which pipeline stage.

---

## 10. Cost Analysis

| Model | Cost/Lifecycle | Score per $1 |
|-------|---------------|-------------|
| qwen3-235b | $0.00006 | 70.3 |
| deepseek-v3 | $0.0004 | 11.9 |
| grok-3-mini | $0.0013 | 4.81 |
| haiku-4.5 | $0.0036 | 1.95 |
| sonnet-4.6 | $0.0207 | 0.52 |
| opus-4.6 | $0.1109 | 0.092 |

Opus-4.6 is 764x less cost-efficient than qwen3-235b on behavioral tasks. The entire 22,200-call benchmark cost $115 — less than a single Devin session ($2.25/ACU) or Claude Code session ($1.50 avg).

---

## 11. Limitations

1. **Lexical scoring is a proxy.** DECF signal word matching captures behavioral language but not behavioral structure. A response that conveys caution through hedging and sentence complexity without using the word "careful" may score low on High-C.
2. **Single-run variance.** Most conditions used 3 trials. Means are reported without confidence intervals. Temperature, API latency, and model updates can affect scores.
3. **Prompt sensitivity.** Results are conditional on specific system prompts, temperatures, and OpenRouter configurations. Different prompting strategies may shift absolute numbers while preserving relative rankings.
4. **DECF is adapted, not validated.** The drive model is adapted from the Predictive Index. We are not affiliated with PI. Signal word dictionaries were hand-curated, not validated against PI's proprietary instruments.
5. **The RLHF paradox is a hypothesis.** We observe correlation between alignment level and persona fidelity reduction. Causal attribution to RLHF specifically is not experimentally isolated.
6. **Free tier instability.** llama-3.3-70b and nemotron-120b experienced rate limit issues, producing incomplete data.

---

## 12. Conclusion

Behavioral identity in large language models is measurable, cheaper than assumed, and structurally constrained by the same alignment training the industry treats as universally beneficial.

The RLHF paradox — budget models outperforming frontier on persona fidelity — is not a temporary artifact. It is a structural tradeoff that the industry has not quantified from both sides. Every dollar spent on alignment training makes persona fidelity harder for frontier models and easier for budget models. The gap widens with each generation.

Architecture determines the ceiling. Routing exploits the split. A colony of cheap, well-routed models outperforms a single expensive model on behavioral tasks while costing 97% less.

This finding has theoretical grounding in the Lottery Ticket Hypothesis [19]. Frankle & Carbin (2019) proved that dense, randomly-initialized networks contain sparse subnetworks (10-20% of the original size) that, when trained in isolation with their original initialization, match the full network's accuracy. Budget models in our evaluation function as behavioral lottery tickets: they are the sparse, well-initialized subnetworks that the frontier model contains but cannot access because its RLHF alignment training has overwritten the initialization conditions that would allow behavioral range. The POMR router's function is analogous to the pruning mask: it identifies which model (subnetwork) is the winning ticket for each behavioral task, discarding the expensive parameters that contribute nothing to persona fidelity.

ConstellationBench is an open benchmark. The data, scoring engine, signal word dictionaries, and results are publicly available at https://huggingface.co/datasets/AirlockLabs/constellation-bench. Total cost to reproduce: approximately $23 for the core 7-benchmark suite.

---

## References

[1] Hendrycks, D., et al. "Measuring Massive Multitask Language Understanding." ICLR 2021.

[2] Chen, M., et al. "Evaluating Large Language Models Trained on Code." arXiv:2107.03374, 2021.

[3] Srivastava, A., et al. "Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models." arXiv:2206.04615, 2022.

[4] Rein, D., et al. "GPQA: A Graduate-Level Google-Proof Q&A Benchmark." arXiv:2311.12022, 2023.

[5] Predictive Index Science Team. "The Science of The Predictive Index." Technical Report, 2020.

[6] Suedfeld, P. and Tetlock, P. "Integrative Complexity of Communications in International Crises." Journal of Conflict Resolution, 1977.

[7] Rosenthal, R. and Jacobson, L. "Pygmalion in the Classroom." Holt, Rinehart & Winston, 1968.

[8] Eden, D. "Pygmalion Without Interpersonal Contrast Effects." Journal of Applied Psychology, 1990.

[9] Fiorella, L. and Mayer, R. "The Relative Benefits of Learning by Teaching and Teaching Expectancy." Contemporary Educational Psychology, 2013.

[10] Csikszentmihalyi, M. "Flow: The Psychology of Optimal Experience." Harper & Row, 1990.

[11] Deci, E. and Ryan, R. "Intrinsic Motivation and Self-Determination in Human Behavior." Plenum Press, 1985.

[12] Zeigarnik, B. "On Finished and Unfinished Tasks." Psychologische Forschung, 1927.

[13] Weber, B. and Hertel, G. "Motivation Gains of Inferior Group Members." Journal of Personality and Social Psychology, 2007.

[14] Latane, B., Williams, K., and Harkins, S. "Many Hands Make Light the Work: The Causes and Consequences of Social Loafing." Journal of Personality and Social Psychology, 1979.

[15] Edmondson, A. "Psychological Safety and Learning Behavior in Work Teams." Administrative Science Quarterly, 1999.

[16] Herzberg, F. "The Motivation to Work." Wiley, 1959.

[17] Sharma, M., et al. "Towards Understanding Sycophancy in Language Models." ICLR 2024.

[18] Shazeer, N., et al. "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." ICLR 2017.

[19] Frankle, J. and Carbin, M. "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." ICLR 2019.

[20] Hu, Z., Rostami, M., and Thomason, J. "Expert Personas Improve LLM Alignment but Damage Accuracy: Bootstrapping Intent-Based Persona Routing with PRISM." arXiv:2603.18507, 2026.

---

## Appendix A: Signal Word Dictionaries

Full signal word sets are available at `data/signal-words/decf-signals.json` in the dataset repository. 89 words across 8 dimension sets (HIGH_D, LOW_D, HIGH_E, LOW_E, HIGH_C, LOW_C, HIGH_F, LOW_F).

## Appendix B: Reproduction

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
python -m benchmarks.harness --quick  # ~$0.30, 3 queries x 2 models
python -m benchmarks.harness          # ~$23, full 7-benchmark suite
```

## Appendix C: Quick Model Scoring

```bash
python scripts/quick_bench.py --model "your-model-id" --full
```

Scores any OpenRouter-compatible model against the leaderboard using the same DECF scoring engine. Cost: ~$1-3 per model in full mode.
