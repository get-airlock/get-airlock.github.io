# NeurIPS 2026 Paper — LaTeX Project

Scaffold for the merged ConstellationBench + NSI submission to the NeurIPS 2026 Evaluations & Datasets track. Structural spec in [`../../../constellation-bench-hf/docs/MERGED-PAPER-OUTLINE.md`](../../../constellation-bench-hf/docs/MERGED-PAPER-OUTLINE.md).

## Files

| File | Purpose |
|---|---|
| `paper.tex` | Main submission file. Structure and Christiano touchpoints pre-placed; section bodies are TODO stubs. |
| `references.bib` | Starter BibTeX. Will expand as citation-lineage memo lands. |
| `neurips_2026.sty` | NeurIPS 2026 style file (from official template ZIP, do not modify). |
| `neurips_2026.tex` | Template demo file (reference only, not submitted). |
| `checklist.tex` | NeurIPS Paper Checklist — required at submission. Fill before May 6. |
| `neurips_2026_template.zip` | Original template archive. |

## Submission configuration

`paper.tex` uses `\usepackage[eandd, nonanonymous]{neurips_2026}` — Evaluations & Datasets track, single-blind (dataset exemption; ConstellationBench is publicly released).

For acceptance: change to `[eandd, final]` for camera-ready.

## Deadlines

- **Abstract:** 2026-05-04 AoE
- **Full paper:** 2026-05-06 AoE

## Building locally

TeX Live basic install is missing packages NeurIPS style requires. One-time fix:

```bash
sudo tlmgr install environ trimspaces microtype nicefrac biblatex booktabs amsmath amsfonts
```

(Or install the `texlive-latex-extra` equivalent if available via Homebrew / MacTeX full.)

After that:

```bash
cd papers/latex-neurips
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Building on Overleaf

Overleaf's TeX Live image has all dependencies. To upload:

```bash
cd papers/latex-neurips
zip paper-submission.zip paper.tex references.bib neurips_2026.sty checklist.tex
```

Upload the ZIP to Overleaf (New Project → Upload Project).

## Christiano 2017 touchpoints

Four planned references to `\citet{christiano2017deep}`, all currently in comments in `paper.tex`:

1. §1 closing paragraph — lineage motivation
2. §2 RLHF limits subsection — direct ancestry
3. §5 RLHF paradox subsection — scalar-compression hypothesis citation
4. §8 Future Work closing — RLHO plant with v0.3 companion paper mention

Move from comment to body text during Day 2–6 drafting passes.

## Next milestones

- Day 2 (Apr 24): Abstract + §1 Introduction body
- Day 3 (Apr 25): §2 Related Work + §4 NSI definition bodies
- Day 4 (Apr 26): §3 Dataset + §5.1–5.2 empirical
- Day 5 (Apr 27): §5.3–5.6 empirical + §6 Reproducibility
- Day 6 (Apr 28): §7 Limitations + §8 Future Work + paper checklist
- Day 7–10 (Apr 29–May 2): polish, figures, internal review, integrate lineage memo citations
- Day 11 (May 3): final pass
- May 4: abstract submission
- May 5–6: buffer + full paper submission
