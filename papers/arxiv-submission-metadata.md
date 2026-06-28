# ConstellationBench — Arxiv Submission Metadata

> **Public-drop submission package for arxiv.org · attributed version (NOT anonymized — anonymized variant lives at `paper/neurips-anon` branch).** Substrate's strategy: arxiv preprint with attribution as the public-drop, then NeurIPS D&B submission as anonymized variant. This file is paste-ready for the arxiv submission web form OR for `arxiv-submit` CLI.

---

## Submission target

- **arxiv.org** primary subject: **`cs.CL`** (Computation and Language)
- Cross-list: **`cs.AI`** (Artificial Intelligence) · **`cs.LG`** (Machine Learning)
- Optional cross-list: **`cs.CY`** (Computers and Society) — given the RLHF Paradox finding has alignment-policy implications

## Title

> **ConstellationBench: The RLHF Paradox and Why Cheap Models Are Better at Being Someone**

(Voice-matched to the Great Accident lineage. Reviewers will quote this title.)

## Authors

- **Zachary Holwerda** (Airlock Labs · `admin@airlocklabs.io` · ORCID: TBD-IF-CAPTAIN-HAS-ONE)
- *(Co-authors to add iff UofM contact materializes via Jim Robinson before submission)*

## Affiliation block

```
Airlock Labs
admin@airlocklabs.io
https://airlocklabs.io
```

## Abstract (248 words — fits arxiv limit)

> Every major AI benchmark measures what a model can do. None measure who it can be.
>
> We built ConstellationBench because we needed a router — a system that sends behavioral tasks to cheap models and reasoning tasks to expensive ones. To build the router, we needed to measure something nobody was measuring: can a model sustain a consistent persona under pressure? Can it hold its position when a user pushes back? Can it sound like a Guardian in one conversation and a Maverick in the next, and can you tell the difference?
>
> We ran 22,200 LLM calls across 22 models for $115. What we found is that the industry's most expensive safety training — RLHF — systematically destroys the behavioral range models need to play distinct characters. Budget models with lighter alignment outperform frontier models on persona fidelity by ~20%. The alignment that makes models helpful also makes them all sound the same.
>
> No single architecture wins everything. MoE models dominate voice differentiation. Dense models dominate depth. A router that exploits this split achieves the same behavioral quality at 97% lower cost.
>
> We tested 4 architecture families including pure state-space (Mamba). The measurement discriminates between them at p < 10⁻⁶. Twelve IO-psychology mechanisms produce 22 actionable routing rules. The entire benchmark reproduces for $23.
>
> Everything is open — code, scoring engine, signal words, results for all 22 models — at huggingface.co/datasets/AirlockLabs/constellation-bench.
>
> The pattern was already there. We just measured it.

## Comments field (visible on arxiv)

```
22 models · 22,200+ LLM calls · $115 total compute · Reproduces for $23.
Public dataset: huggingface.co/datasets/AirlockLabs/constellation-bench
Public leaderboard: huggingface.co/spaces/AirlockLabs/constellation-bench-leaderboard
Code: github.com/AirlockLabs/constellation-bench
License: CC-BY-SA (data) · MIT (code)
Companion runtime tool (Apache-2.0): github.com/get-airlock/maverick-mcp
```

## License (arxiv)

- **arxiv-nonexclusive-distrib** (default · keeps copyright with author · arxiv hosts)
- *Avoid CC-BY at arxiv-license-tier — it's redundant with the dataset's CC-BY-SA. Arxiv-default is sufficient.*

## Categories — formal arxiv designators

| Tier | Category | Reason |
|---|---|---|
| Primary | `cs.CL` | Behavioral identity · persona fidelity · LLM evaluation are computational-linguistics-tier |
| Cross-list | `cs.AI` | Agent-relevant · routing architecture |
| Cross-list | `cs.LG` | Architecture-dependent ceiling finding · empirical ML measurement |
| Optional cross-list | `cs.CY` | RLHF Paradox has alignment-policy implications |

## Files to upload

| File | Purpose |
|---|---|
| `airlock-0.1-constellationbench.tex` (NeurIPS-template-converted) OR `airlock-0.1-constellationbench.pdf` | Main paper |
| `IEEEtran.cls` OR `neurips_2026.sty` | LaTeX template (whichever the final submission uses) |
| `fig-moe-vs-dense.png` | Figure 1 (architecture-ceiling viz) |
| `references.bib` | BibTeX bibliography (extracted from main .tex) |
| `supplemental.zip` *(optional)* | Eval YAML · signal-words JSON · personas profiles · methodology MD |

## Endorsement requirement check

- **First-time arxiv submitters in `cs.CL` need an endorsement** from a previously-published `cs.CL` author
- Captain status: **TBD — CHECK BEFORE SUBMISSION**
  - If never submitted to `cs.CL` before: need endorsement (lead time ~1 week)
  - If submitted to `cs.AI` previously: may auto-qualify for cross-list submission to `cs.CL`
  - **Workaround if endorsement is blocking**: submit primary as `cs.AI` (lower endorsement bar) and request cross-list to `cs.CL` post-acceptance

## Submission checklist (pre-flight)

- [ ] PDF compiles cleanly from .tex source (no missing references, all figures embedded)
- [ ] Title, authors, affiliations on first page match this metadata exactly
- [ ] Abstract on first page matches the 248-word version above (verbatim)
- [ ] All HuggingFace + GitHub URLs in the paper resolve to live pages
- [ ] License declarations consistent (CC-BY-SA data · MIT code · Apache-2.0 maverick-mcp · arxiv-default for the paper)
- [ ] Companion artifact mentions in Comments field point to real repos
- [ ] Bibliography compiled clean (no missing `[?]` references in PDF)
- [ ] `\maketitle` renders correctly with author + affiliation
- [ ] Endorsement status confirmed (or workaround chosen) before submission attempt
- [ ] Captain has arxiv account at arxiv.org/user (account creation: ~2 min)

## Submission flow (paste-ready)

```bash
# 1. Verify paper compiles
cd /Users/zacharyholwerda/Desktop/Airlock/airlock-labs-site/papers/latex
pdflatex airlock-0.1-constellationbench.tex
bibtex airlock-0.1-constellationbench
pdflatex airlock-0.1-constellationbench.tex
pdflatex airlock-0.1-constellationbench.tex

# 2. Bundle for upload
cd /Users/zacharyholwerda/Desktop/Airlock/airlock-labs-site/papers
zip -r arxiv-bundle.zip latex/airlock-0.1-constellationbench.tex \
                        latex/IEEEtran.cls \
                        latex/fig-moe-vs-dense.png \
                        latex/references.bib

# 3. Upload via arxiv.org/submit (web form) OR
#    arxiv-submit CLI if installed:
#    pip install arxiv-cli
#    arxiv-cli submit --metadata arxiv-submission-metadata.md --files arxiv-bundle.zip
```

## Post-submission cadence

| Step | When | Action |
|---|---|---|
| Submission accepted by arxiv moderators | ~12-24 hours after submit | arxiv ID assigned (e.g. `2605.XXXXX`) |
| Public on arxiv listing | Next announcement cycle (typically 8 PM UTC weekday) | DOI-equivalent URL: `arxiv.org/abs/2605.XXXXX` |
| Update airlocklabs.io | Same day arxiv goes live | Add arxiv badge to homepage · update papers/ index |
| Update HuggingFace dataset README | Same day | Add arxiv citation to dataset card |
| Cross-post announcement | Day 2 | Twitter/X · LinkedIn · Mastodon · `zac_the_builder` Threads handle · Hacker News (if you want to handle the heat) |
| Tag Angie Jones / Block / Goose ecosystem | Day 2-3 | Direct outreach with arxiv link · Maverick-MCP repo URL |
| NeurIPS D&B submission (anonymized variant) | May 4 abstract deadline · May 6 full paper | Submit via OpenReview · `paper/neurips-anon` branch's PDF |

## Anonymized variant — what to strip for NeurIPS submission

For the `paper/neurips-anon` branch (NeurIPS D&B double-blind):

- **Strip:** `Zachary Holwerda` · `Airlock Labs` · `admin@airlocklabs.io` · `airlocklabs.io` references in title/affiliation/acknowledgments
- **Strip:** `AirlockLabs/constellation-bench` HuggingFace org name in body (replace with placeholder `[anonymized-org]/constellation-bench` for review; restore for camera-ready)
- **Strip:** any link to `airlocklabs.io` or `doyoulikedags.xyz` in the paper body (NOT arxiv preprint references — those are external citations and allowed)
- **Keep:** ConstellationBench dataset CC-BY-SA license · MIT code license (license declarations don't break anonymity)
- **Keep:** the arxiv preprint reference itself (NeurIPS D&B explicitly permits arxiv preprints during review; cite as `[arxiv:2605.XXXXX]` once the ID is assigned)

## Substrate-canon composition

This submission composes with:
- **The Great Accident voice** — title and abstract match the v2 lineage (plain language · stakes-first · pattern-discovered-not-designed)
- **Sequoia AI Ascent 2026 framing** — abstract embodies "emerging sciences" positioning (thermodynamics-of-AI move) without naming it explicitly
- **Karpathy's "ghosts vs animals"** — abstract operationalizes "characterizing the jaggedness" empirically
- **Block-Goose lineage** — Comments field points at Maverick-MCP (Apache-2.0) which integrates with Block's Goose; ecosystem-recognition by attribution
- **Magnanimous-by-construction** — every artifact is OSS-licensed; reproduction cost ($23) is published; all 22 models' results are public
- **Composition is the moat** — paper cites the architectural-substrate finding as the key empirical claim; substrate's own runtime tooling (Maverick-MCP) gets attribution alongside the research

---

*Filed by Builder lane 1 · 2026-05-02 · paste-ready for Captain's arxiv submission once paper PR merges.*
