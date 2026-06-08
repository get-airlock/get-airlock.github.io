# Airlock Labs — Fundraising Source of Truth
## Last Updated: 2026-04-20

### Company
- **Legal name**: Airlock Technologies (dba Airlock Labs)
- **Location**: Detroit, MI
- **Founded**: 2026 (incorporated TBD — verify with Zac)
- **Stage**: Pre-seed
- **Team**: Zachary Holwerda (solo founder, CEO)
- **One-liner**: "We build behavioral AI routing to help enterprises cut inference costs by 90% while getting better persona fidelity than frontier models."

### Product
- **ConstellationBench**: Open behavioral AI benchmark — 22 models, 22,200+ calls, 44 layers, 97% cheaper than frontier-uniform inference
- **POMR Router**: Persona-Optimized Model Router — routes behavioral tasks to cheap models, reasoning to frontier
- **Quick Bench Runner**: One-command model scoring tool
- **Pareto Frontier Bench**: Interactive routing challenge

### Key Finding
- RLHF paradox: budget models outperform frontier by ~20% on persona fidelity
- Holds across 4 architecture families (Dense, MoE, Mamba hybrid, Linear Attention)
- Free Qwen 3.6 (0.617) > Gemma 4 (0.590) > Opus 4.7 (0.538) > GPT-5.4 (0.526)

### Revenue Model
- **10% activation fee** on first month's savings (one-time)
- **20% of ongoing savings** (monthly, metered)
- **Referral program**: details TBD
- Example: Customer spending $20K/mo → we save $14K → we charge $1,400 activation + $2,800/mo ongoing → customer still saves $11,200/mo

### Unit Economics
- Pro tier ($9.99/mo, grok-3-mini): 91.2% gross margin at full usage, 97-99% at typical
- Budget model cost per council: $0.001-$0.005
- Frontier model cost per council: $0.03-$0.17
- Break-even on Pro at 5,676 councils (11x the allotment)

### Market
- Global AI inference market: ~$50B+ and growing (verify with current data)
- Every company using AI APIs is overpaying for behavioral tasks
- Target: enterprises spending $5K-$500K/mo on AI inference

### Traction (as of April 19, 2026)
- ConstellationBench published on HuggingFace (public dataset)
- 22 models benchmarked across 4 architecture families
- Website live at airlocklabs.io
- White paper drafted (arXiv submission pending, NeurIPS deadline May 4)
- Outreach: ZeroEval (Seb, warm contact), Create Music Group (Eduardo, former employer)
- Calculator live at airlocklabs.io/calculator
- 3 blog posts published, 2 in reserve
- Patents filed (behavioral routing, DECF scoring)

### IP
- Patents filed (exact count/status: verify with Zac)
- Open benchmark data (MIT license) — the data is open, the routing engine is proprietary
- 44 layers of experimental findings (22 routing rules)
- Signal word dictionaries, DECF scoring engine

### Raise
- **Instrument**: Post-money SAFE (standard per Carta 2025 data — 93% of pre-seed)
- **Target raise**: $5M
- **Valuation cap**: $25M post-money
- **Dilution**: 20%
- **Prior funding**: None (clean cap table, 100% founder-owned)
- **Use of funds**:
  - 40% — Train behavioral foundation model (the router becomes a language model)
  - 25% — Engineering team (3-4 hires: routing infra, SDK, ML)
  - 15% — Enterprise pilots + GTM
  - 10% — Research (NeurIPS, benchmark expansion to 100+ models)
  - 10% — Ops, legal, IP protection
- **Vision**: Not a routing SDK. A behavioral intelligence foundation model — the model that understands who the user is, not just what they asked.

### Comparable Companies / Adjacent Space
- Alice/ActiveFence: $100M+ raised, sells AI safety guardrails (our thesis: they make models worse)
- ZeroEval: AI evaluation platform (potential partner, not competitor)
- CrewAI: Multi-agent framework (91x more expensive per task than our routing)
- Devin: AI agent ($2.25/ACU, 1,711x more expensive than our approach)

### Missing / To Confirm with Zac
- [ ] Exact incorporation status and entity type
- [ ] Patent count and filing dates
- [ ] Matt's $10K — closed or still pending?
- [ ] Target raise amount
- [ ] Valuation cap expectation
- [ ] Specific use of funds breakdown
- [ ] Timeline to first enterprise customer
- [ ] Sarim meeting details (week of Apr 21)
