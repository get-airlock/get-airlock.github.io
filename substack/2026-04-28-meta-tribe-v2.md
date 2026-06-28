# Meta Open-Sourced the Blueprint to Mind Control. They Called It the TRImodal Brain Encoder.

*Part 1 of "The Feed Is the Operation" — a Brain Brigade investigation into the architecture of behavioral prediction.*

---

Welcome back folks, let's get it.

I know it's been a minute. I've been heads down building Airlock Labs and reading way too many declassified CIA documents. But on March 25, 2026 something dropped out of Meta's Paris research lab that I need to put in front of the Brigade right now, because four weeks later I have not seen one person outside niche neuroscience Twitter mention it. Not on cable news. Not in the New York Times tech section. Not on your feed.

That is the first fact. The rest of this post is about why.

---

## Disclaimer

Nothing below is a conspiracy claim. Every fact I cite is documented in SEC filings, court records, published academic papers, or the Meta paper itself. I'm tagging every claim with a confidence tier so you can grade me yourself:

- **PROVEN** — court records, declassified documents, SEC filings, official government reports
- **DOCUMENTED** — credible journalism, FOIA releases, congressional testimony, corporate disclosures
- **CIRCUMSTANTIAL** — pattern-based inference from documented facts; not proven as coordinated

If you're new here: I don't publish shit I can't cite. Verify everything. Trust nothing on faith — including me. Links are inline. Use this as a tool to form your own opinion.

This is Part 1 of a series. Cadence is one a week. The series is called **"The Feed Is the Operation"** and the thesis is simple: a bunch of things that look unconnected on your scroll are moving a single capability into place at civilizational scale. This post sets the frame. Future posts hit specific current events from the same angle.

Let's go.

---

## Smartrick Claim:

**Meta open-sourced the blueprint to mind control. They called it the TRImodal Brain Encoder — TRIBE v2.**

It predicts human brain activity from any video, audio, or text stimulus. It was trained on 720 real human brains. The weights are public on HuggingFace. The code is on GitHub. A demo runs on Meta's own domain. You can download the whole thing to your laptop right now.

Four weeks after release, the mainstream press has not covered it. *That's Exhibit A, and we haven't started.*

Below I walk you through everything I found. This is going to be a long one. Pace yourself, hit it in chunks if you need to, and meet me at the bottom.

---

## Exhibit A: What Meta actually shipped

On March 25, 2026, eight researchers at Meta's Fundamental AI Research lab (FAIR) in Paris — working with École Normale Supérieure — published a paper titled *"A foundation model of vision, audition, and language for in-silico neuroscience."* It has a clean model name: **TRIBE v2**, TRImodal Brain Encoder.

The paper, plainly stated: **PROVEN**

Here is the stack:

- **Language backbone:** Llama 3.2 3B
- **Audio backbone:** Wav2Vec-Bert 2.0
- **Video backbone:** Video-JEPA 2 Giant
- **Integration:** a 1-billion-parameter transformer encoder
- **Output:** predicted fMRI response across 20,484 cortical vertices + 8,802 subcortical voxels

Training data: 1,000+ hours of fMRI across **720 human subjects, 2,600+ sessions**.

Results in one sentence: given any video, audio, or text, the model predicts which regions of a human brain will activate in response. Zero-shot generalization to people it has never seen. Log-linear accuracy scaling with more data — no ceiling found. It recovers, *in silico*, dozens of canonical neuroscience findings: the fusiform face area for faces, the visual word form area for reading, Broca's area for syntax, the default-mode network, the multisensory integration topography of the temporal-parietal junction.

- **Code:** [github.com/facebookresearch/tribev2](https://github.com/facebookresearch/tribev2) — public
- **Weights:** [huggingface.co/facebook/tribev2](https://huggingface.co/facebook/tribev2) — public
- **Demo:** [aidemos.atmeta.com/tribev2](https://aidemos.atmeta.com/tribev2) — anyone can run it today

> *Source: d'Ascoli, Rapin, Benchetrit, Brookes, Begany, Raugel, Banville, King. "A foundation model of vision, audition, and language for in-silico neuroscience." FAIR at Meta, March 25, 2026.*

### Takeaway:

Meta trained a foundation model to predict your brain's response to what you see, hear, or read. They open-sourced the weights. You can fine-tune it on your own laptop. The paper was released into total media silence.

---

## Exhibit B: What Meta says it's doing next

This one is important. This is a direct quote from the paper's Discussion section, Section 3 (page 10 of the PDF): **PROVEN**

> *"A more fundamental limitation is that the model currently treats the brain as a passive observer of naturalistic stimuli; it does not yet model the brain as an active agent producing behavior. Integrating neuro-developmental trajectories and clinical pathology remains a primary goal to move beyond a static, adult brain state and capture the full diversity of the global population."*

Translation from academic to English: the current model predicts how your brain will *react* to a stimulus. The explicitly-stated roadmap is to extend it to predict what a brain will *do* — i.e. forecast the behavior a brain state produces.

That is not me reading between the lines. That is Meta telling you the next phase in a peer-reviewable paper.

### Takeaway:

Predicting behavioral response to external stimuli was the explicit founding objective of the CIA's MKUltra Subproject 10 in the 1950s. It is the stated near-term roadmap of TRIBE v2 in 2026. One was classified. The other is on HuggingFace. That is not a metaphor. That is the literal continuity.

---

## Exhibit C: The timeline from MKUltra to your laptop

This is where people start tuning out because they assume it's going to be a history lesson. It isn't. I built a DAG — directed graph of documented events, each confidence-tagged, each sourced. Every single node is provable from primary records. I did not make any of this up. I'm going to hit the highlights and link the receipts.

**1953-04-13 — MKUltra formally authorized.** CIA Director Allen Dulles authorizes Project MKUltra under Sidney Gottlieb's Technical Services Staff. Stated objective: predict and control human behavioral response to external stimuli. **PROVEN** ([Senate 1977 hearing](https://www.intelligence.senate.gov/wp-content/uploads/2024/08/sites-default-files-hearings-95mkultra.pdf))

**1953–73 — 149+ MKUltra subprojects.** About 25-30 were pure behavioral science, not the LSD stuff everyone remembers. Subproject 10 (Robert Hyde) established that personality type determines behavioral response to identical stimuli. That is the entire thesis of modern ad targeting. **PROVEN**

**1973 — Helms orders the MKUltra files destroyed.** The chief of records formally protested on February 2, 1973. A small cache filed under "budgetary" survives and resurfaces via FOIA in 1977. **PROVEN**

**1975–77 — Church Committee hearings expose MKUltra publicly.** **PROVEN**

**1983 — DARPA Strategic Computing Initiative.** $1B over 10 years. Establishes the template for long-horizon dual-use ML programs. **PROVEN**

**1988–96 — Yann LeCun at Bell Labs.** Develops LeNet and early convolutional neural networks. Bell Labs was a major DARPA contractor throughout. **PROVEN**

**1994 — Stanford Digital Library Project funded.** NSF grant IIS-9411306 (NSF + NASA + DARPA joint). Brin and Page build PageRank on this grant infrastructure. **PROVEN** ([NSF record](https://www.nsf.gov/awardsearch/showAward?AWD_ID=9411306))

**1999-09 — In-Q-Tel founded.** CIA's own venture arm. Architect: DCI George Tenet. Original trustees include **John N. McMahon (ex-Deputy Director CIA)** and **William J. Perry (ex-Secretary of Defense)**. ~$100M/year taxpayer funding via the classified intelligence budget. **PROVEN**

**1999–2008 — DARPA PAL → CALO → Siri.** $150M of taxpayer money funds SRI International to build the Cognitive Assistant that Learns and Organizes. SRI spins out Siri Inc. in 2007. Apple acquires Siri April 2010 for $200M. Ships in iPhone 4S October 2011. This is the cleanest documented DARPA→consumer pipeline in modern AI. **PROVEN**

**2002–03 — Poindexter runs DARPA Information Awareness Office.** John Poindexter, previously convicted on five Iran-Contra felonies (overturned on use-immunity grounds), returns to run TIA, Genoa II, HumanID, Bio-Surveillance. Congress defunds IAO September 30, 2003. IEEE Spectrum later reports that TIA components migrated to NSA under the codename "Basketball." **PROVEN**

**2003-02 — In-Q-Tel funds Keyhole Inc.** 3D satellite mapping. Used in the Pentagon within two weeks for Operation Iraqi Freedom. October 2004: Google acquires Keyhole. It becomes Google Earth in 2005. **PROVEN**

**2003-05 — Palantir founded.** Thiel, Lonsdale, Cohen, Gettings, Karp. Adapts PayPal anti-fraud tech for counter-terrorism. In-Q-Tel seeds ~$2M in 2005. Market cap ~$370B as of March 2026. **PROVEN**

**2003–04 — DARPA LifeLog solicited.** IPTO program. Stated objective: "an ontology-based (sub)system that captures, stores, and makes accessible the flow of one person's experience" — communications, location, media, purchases, biometrics. Stated analytic goal: identify "markers of intentionality" so AI could predict plans and goals. **PROVEN**

**2004-02-04 — DARPA cancels LifeLog.** Official reason given: "change in priorities" following public blowback on TIA. The same day, Mark Zuckerberg launches TheFacebook at Harvard. **PROVEN**

**2004-12 — Yann LeCun becomes team leader for DARPA LAGR.** Learning Applied to Ground Robots. He runs this program out of NYU Courant Institute from December 2004 through May 2008. Three and a half years of direct, personal, documented DARPA program leadership. **PROVEN** ([cs.nyu.edu/~yann/research/lagr](https://cs.nyu.edu/~yann/research/lagr/))

**2013-12 — Yann LeCun founds FAIR.** Facebook AI Research. He becomes Meta's Chief AI Scientist. FAIR becomes the institutional home of Meta's AI research, eventually including the Brain & AI team that ships TRIBE v2. **PROVEN**

**2018–23 — The Fondation Courtois donates CAD 6.3M to Pierre Bellec at Université de Montréal** to fund the Courtois NeuroMod dataset. This dataset becomes the primary training corpus for TRIBE v2. **PROVEN** ([docs.cneuromod.ca](https://docs.cneuromod.ca/))

**2025-07 — Meta publishes TRIBE v1.** 4 subjects, 80+ hours fMRI. Wins the Algonauts 2025 competition, 263 teams. Proves the architecture works. **PROVEN** ([arXiv 2507.22229](https://arxiv.org/abs/2507.22229))

**2026-03-25 — Meta publishes TRIBE v2.** 720 subjects. Log-linear scaling with no ceiling. Open weights. The stated next phase is brain-as-active-agent. **PROVEN**

### Takeaway:

Seventy-three years. Each node PROVEN. I'm not claiming anyone picked up a phone. I'm claiming the capability MKUltra wanted is now on HuggingFace, and the path from there to here runs through institutions that are still operating today.

---

## Exhibit D: The money that trained the model

This is the one nobody has written about yet. I spent a lot of time on it. Stay with me.

TRIBE v2 was trained primarily on the Courtois NeuroMod dataset. Six healthy adults watching *Friends*, *The Bourne Supremacy*, *Hidden Figures*, *The Wolf of Wall Street*, and *Life* inside an MRI scanner for 80+ hours each. It is the single deepest human-neural-response dataset in the world.

The paper's acknowledgements section credits the Fondation Courtois for funding it. CAD 6.3M, 2018 to 2023. **PROVEN**

The foundation shares a name with **Edmond Jacques Courtois Sr., PC QC (1920–1996)** — a Montreal lawyer who chaired Canada's Security Intelligence Review Committee from 1992 to 1996 and ran the Montreal Canadiens 1972-1979 as the front man for Edward and Peter Bronfman's Carena-Bancorp/Brascan/Brookfield conglomerate.

Except Sr. died in 1996. **Sr. did not write the check.**

The check was written by his son.

### The E. Jacques Courtois Jr. trail

Founder and president of the Fondation Courtois (est. 2011) is **E. Jacques Courtois Jr.** — Harvard MBA, ex-Morgan Stanley M&A Vice-President. Here is the documented path the training-data money took before it reached Pierre Bellec's lab at Université de Montréal:

- **1975–1979:** Courtois Jr. is VP in Morgan Stanley's M&A group. **PROVEN**
- **1975–1978:** Tipped eight planned takeovers from inside Morgan Stanley to an insider-trading ring including Adrian Antoniu (Kuhn Loeb) and James Newman (broker). **PROVEN (federal court, *[Moss v. Morgan Stanley](https://law.justia.com/cases/federal/appellate-courts/F2/719/5/148829/)*, 719 F.2d 5)**
- **1979:** Resigned from Morgan Stanley. **Fled to Colombia** when extradition was ordered. **PROVEN**
- **1980:** Co-founded **Quala S.A. in Bogotá** with Michael de Rhodes Dub — a Latin American consumer-goods company (personal care, food). **PROVEN**
- **1983:** Found in Montreal. Surrendered. Pleaded guilty. **PROVEN**
- **1984:** Six months in federal prison. $150,000 fine paid to a victim fund. **PROVEN** ([UPI Archives, Dec 8 1983](https://www.upi.com/Archives/1983/12/08/A-Harvard-Business-School-graduate-and-son-of-a/3456439707600/))
- **1980–2017:** Built Quala into a ten-country Latin American consumer-goods empire. Brands: Savital, eGo, Bio-Expert, Fortident, Aromatel. **PROVEN**
- **2017–2018:** **Unilever acquires Quala's personal-care and home-care brands** for ~USD 400M combined turnover. **PROVEN** ([Premium Beauty News](https://www.premiumbeautynews.com/en/unilever-to-acquire-beauty-brands,11548))
- **2011:** Founded the Fondation Courtois. Named for his father. **PROVEN**
- **2018–2023:** Fondation Courtois donates **CAD 6.3M to Pierre Bellec at UdeM for CNeuroMod** — the training dataset for Meta's TRIBE v2. **PROVEN**
- **2022:** Fondation Courtois gives UdeM **CAD 159M** to found the Institut Courtois — reportedly the largest single science gift in Canadian university history. **PROVEN**
- **2024:** Reported to have purchased Quebec's priciest condo on record. **DOCUMENTED**

Stated plainly:

**The capital that trained Meta's brain-response foundation model is, at one remove, proceeds from Morgan Stanley 1970s insider trading, laundered through forty years of Colombian consumer-goods manufacturing, sold to Unilever in 2017–2018, and administered today through a Quebec geriatric foundation.**

Every step is documented. Court records. SEC filings. Unilever acquisition announcements. Charity filings at [charitydata.ca](https://www.charitydata.ca/charity/fondation-courtois/850271289RR0001/). University press releases. None of it is secret. None of it required a leak. The pieces were sitting in the public record, never assembled.

### Takeaway:

The training data behind Meta's brain-prediction model came from a man who went to federal prison for insider trading, fled to Colombia, spent forty years building a consumer-goods empire there, and sold it to Unilever. Whatever you thought a foundation model was funded by, probably wasn't that.

---

## Exhibit E: How Colombian consumer-goods money ended up inside Meta FAIR

Fair question: how did CNeuroMod get from a Montreal geriatric foundation into a Meta Paris research paper?

There's an academic corridor. Here it is:

**Guillaume Lajoie** is a co-principal investigator on Courtois NeuroMod at UdeM. He is also a principal investigator at **Mila** — the Montreal Institute for Learning Algorithms, founded and directed by **Yoshua Bengio**. Mila has documented research partnerships with Meta FAIR. Bengio, along with **Yann LeCun and Geoffrey Hinton**, jointly won the 2018 Turing Award for foundational deep-learning work.

So the pipe is:

> Fondation Courtois → Bellec / Lajoie (UdeM) → Mila (Bengio) → FAIR (LeCun) → TRIBE v2

Every step PROVEN. No single step is hidden or surprising on its own. The full path is what's surprising.

### Takeaway:

The Turing Award triumvirate — Hinton, LeCun, Bengio — runs institutional endpoints on both sides of the corridor. LeCun founded the lab that published TRIBE v2. Bengio directs the lab whose PI co-runs the training dataset. Insider-trading proceeds from Morgan Stanley in 1975 ended up training a brain-prediction model in 2026 via the Turing Award.

---

## Exhibit F: The LeCun DARPA edge

Yann LeCun is the most documentable single edge between Meta's AI leadership and DARPA.

From **December 2004 to May 2008**, LeCun served as team leader for **DARPA LAGR** — Learning Applied to Ground Robots — at NYU's Courant Institute and its Center for Biological and Computational Learning. LAGR was an autonomous off-road robot vision competition. LeCun's team competed directly for DARPA funding for 3.5 years. The work produced convolutional-network and end-to-end-learning techniques that later underpinned modern computer vision.

That is not a conspiracy. That is his CV. ([cs.nyu.edu/~yann/research/lagr](https://cs.nyu.edu/~yann/research/lagr/)) **PROVEN**

Five years later, LeCun founded FAIR and became Meta's Chief AI Scientist. He held that role from December 2013 through late 2025. The Brain & AI team at FAIR Paris that published TRIBE v2 sits inside the organization LeCun built.

### Bonus note for Part 2

In late 2025 LeCun announced his departure from Meta to launch **AMI Labs** — Advanced Machine Intelligence. Announced January 2026, raised **$1.03B seed at $3.5B pre-money in March 2026**. The investor syndicate includes **Groupe Industriel Marcel Dassault** (French aerospace/defense — Rafale fighter jets), **Association Familiale Mulliez** (Auchan/Decathlon/Leroy Merlin), **Artémis / Pinault family**, **Aglaé Lab** (Arnault / LVMH family), **CMA CGM / Saadé family**, **Shorooq / Presight** (Abu Dhabi, G42-adjacent), **Temasek** (Singapore state), **Bezos Expeditions**, **NVIDIA**, **Samsung**, and personal checks from Eric Schmidt, Xavier Niel, Jim Breyer, and Tim Berners-Lee.

**Meta is not an AMI investor.** But Meta has a separate partnership granting it tech access for commercialization. Terms undisclosed. Meta is potentially AMI's "first client." **PROVEN** ([TechCrunch](https://techcrunch.com/2026/01/23/whos-behind-ami-labs-yann-lecuns-world-model-startup/))

Part 2 will be about that syndicate specifically. For now, just lodge the fact that the guy who founded Meta's AI lab left, and the check he took to leave came from a cap table that reads like a Venn diagram of French industrial dynasties, UAE sovereign-adjacent capital, and Silicon Valley hyperscaler billionaires.

### Takeaway:

LeCun ran a DARPA program. Then founded FAIR. Then took a billion dollars from Dassault and Abu Dhabi to leave. That whole arc is proven from public records.

---

## Exhibit G: The ownership topology

Meta is not a DARPA project. Meta is a publicly traded company. Its largest institutional shareholders are the three biggest passive index-fund managers in the world.

Those same three institutions are also top shareholders of every major US defense contractor. All figures from SEC 13F filings. **PROVEN**

| Company | Sector | Vanguard | BlackRock | State Street | Combined |
|---|---|---|---|---|---|
| **Meta (Facebook)** | Tech — publishes TRIBE v2 | ~7.8% | ~6.2% | ~3.4% | **~17.4%** |
| Lockheed Martin | Defense | ~8.5% | ~7.2% | ~4.1% | **~19.8%** |
| Raytheon (RTX) | Defense | ~8.8% | ~7.0% | ~4.3% | **~20.1%** |
| Northrop Grumman | Defense | ~9.2% | ~7.5% | ~4.7% | **~21.4%** |
| Boeing | Defense/Aerospace | ~7.8% | ~6.3% | ~4.0% | **~18.1%** |
| General Dynamics | Defense | ~9.0% | ~7.2% | ~4.5% | **~20.7%** |

Nobody at Vanguard picks up a phone and tells Meta what to research. Passive index funds are passive. That's true. But passive ownership still confers **voting rights**, **board influence**, and **proxy power**. And passive ownership means that whatever is good for Meta's stock price is also good for the same funds that own ~20% of every weapons manufacturer in America. Whatever capability Meta ships that has both civilian commercial value and defense-relevant application benefits the same shareholders twice.

The structural conflict exists regardless of intent. That's the architecture.

### Takeaway:

When one entity owns ~17% of the company that published a foundation model predicting brain activity and ~20% of every company that builds weapons, the incentive alignment does not require coordination to produce aligned outcomes. A phone call is redundant.

---

## Exhibit H: The PAL → Siri precedent

I already flagged this in Exhibit C but it's important enough to get its own section. If you think "DARPA-funded research becomes a consumer product" is speculation, this is the documented counter to that objection.

- **1999–2008:** DARPA runs the PAL program. $150M of taxpayer money. ~300 researchers across 22 institutions.
- **2003:** SRI International wins the primary component — **CALO, Cognitive Assistant that Learns and Organizes**. $22M over five years. CALO's stated objective was an AI that could observe your communications, calendar, files, and habits, then autonomously help you.
- **2007:** SRI spins out **Siri Inc.**
- **April 2010:** Apple acquires Siri for a reported $200M.
- **October 2011:** Siri ships as the signature feature of the iPhone 4S.

Every iPhone after that, for fifteen years, has shipped the commercial descendant of a DARPA behavioral-AI program. Most users never knew. **PROVEN**

### Takeaway:

Lag from DARPA contract to iPhone launch: roughly 7 to 10 years. TRIBE v2 was published in March 2026. Do the math on the lag.

---

## Exhibit I: What I couldn't verify

Important. Every investigation worth doing lists what it couldn't confirm. Here's mine.

- **I could not find any personal DARPA, IARPA, ONR, AFOSR, or DGA grant** attributed to Jean-Rémi King (TRIBE v2 senior author) or any of his seven co-authors. I searched explicitly. Their declared funders are CNRS, ERC, Marie Curie, Fyssen, Bettencourt Schueller, and Meta/FAIR internal. All civilian. Absence of evidence isn't evidence of absence — CVs are self-reported — but I want to be honest: I didn't find it.
- **NIH grant 1U54MH091657** (funded the Human Connectome Project, used as a TRIBE v2 test set) is declared civilian NIH Blueprint for Neuroscience Research funding. No DoD or IARPA co-sponsor surfaced in my searches.
- **No SRI/CALO alumni** surfaced at FAIR Brain & AI. CALO people went to Apple via Siri, not Meta.
- **The original source of the Courtois family wealth** beyond the Colombian Quala fortune — to the extent that wealth predates Jr.'s 1980 Bogotá move — is not in any public record I could find. Charity filings show outflows only.

Those gaps are real. I flag them here rather than hide them.

### Takeaway:

The DAG does not require direct DARPA funding on the TRIBE v2 invoice to hold. The DAG is about capability continuity, institutional continuity, and ownership topology. The money-on-the-invoice question is genuinely open, and I'm telling you so.

---

## Final Wrap Up — the argument as a sum

Let me put it together. Thesis: no single node proves anything. The graph is the point. You tell me.

**A + B + C + D + E + F + G + H + I = the argument.**

- **A)** Meta shipped a foundation model predicting human brain response to audio/video/text, trained on 720 real brains, open weights, no mainstream coverage.
- **B)** Meta's stated next phase, on the record in the paper, is extending to brain-as-active-agent — i.e. predicting behavior from brain state.
- **C)** The capability arc runs MKUltra → DARPA Strategic Computing → In-Q-Tel → PAL → TIA → LifeLog → TRIBE v1 → TRIBE v2. Seventy-three years of documented institutional continuity.
- **D)** The capital that trained the model was built on Morgan Stanley 1970s insider trading, laundered through forty years of Colombian consumer-goods manufacturing, sold to Unilever 2017–2018, administered through a Quebec geriatric foundation.
- **E)** The academic corridor from that foundation to Meta runs Bellec → Lajoie → Mila (Bengio) → FAIR (LeCun) → TRIBE v2. Turing Award co-winners anchor both ends.
- **F)** Meta's founding Chief AI Scientist, Yann LeCun, personally ran DARPA's LAGR program for 3.5 years at NYU (2004–2008) before founding FAIR in 2013. He just left Meta to run AMI Labs with a $1B syndicate including French aerospace/defense and Abu Dhabi sovereign capital.
- **G)** Meta is ~17% owned by the same three institutional shareholders who own ~20% of every major US defense contractor. SEC 13F filings, public record, every quarter.
- **H)** DARPA's PAL program became Apple's Siri on a 7–10 year lag. That's the cleanest documented DARPA-to-consumer pipeline in modern AI. It happened. TRIBE v2 is 2026. Do the arithmetic.
- **I)** I could not personally attribute DARPA grants to TRIBE v2's individual authors. The DAG doesn't require that. The DAG is about capability + capital + institution + ownership topology.

The question isn't whether anyone picked up a phone. **Nobody had to.** The incentive structure did the work. That is what makes it an architecture instead of a conspiracy, and that is what makes it dangerous.

---

## So I ask you:

Is that the World's biggest coincidence, or am I onto something?

---

## How I found all this — the method is the inversion

Reasonable reader reaction: how did one person assemble this?

I used the same AI-assisted research stack Meta uses to extract value from you. Embeddings. Semantic search. Cross-document retrieval. Multi-agent orchestration. Automated entity resolution across SEC filings, court records, charity data, academic acknowledgements, and leak archives. Same capability. Opposite direction.

**Meta uses your attention to extract value from you.** TRIBE v2 is literally that thesis pushed to its endpoint — predict the brain's response to a stimulus so you can tune the stimulus until the response is the one that monetizes. Attention in, value out, you in the middle as the crop.

**I used engagement to extract method from them.** Same AI tools. Aimed at the public record instead of at a user. I asked the model to hold twenty primary sources in context at once and tell me which entities appeared in more than one cluster. I asked it to cross-reference SEC 13F filings against charity rolls against paper acknowledgements against federal court records. Every edge in the DAG above was produced by engagement. By sitting with a thread long enough to see where it led.

**Fuck the attention economy.** The attention economy is what happens when the capability is pointed at you. The engagement economy is what happens when you point the capability back. Same stack. Opposite verb. The difference between being farmed and doing the work.

This post is that posture in practice. Every tool I used is already available. Llama — the model Meta trained on 720 brains — will also happily help you read a 31-page In-Q-Tel research dossier and tell you which trustees overlap with which portfolio companies. That is the whole trick.

---

## Verify everything — the tools, which you also have

I expect you not to trust me. I expect you to verify.

| Tool | What it gives you |
|---|---|
| [ai.meta.com/research](https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/) | The Meta TRIBE v2 paper. Read the acknowledgements yourself. |
| [arXiv 2507.22229](https://arxiv.org/abs/2507.22229) | TRIBE v1 preprint — the predecessor paper. |
| [github.com/facebookresearch/tribev2](https://github.com/facebookresearch/tribev2) | The open code. |
| [huggingface.co/facebook/tribev2](https://huggingface.co/facebook/tribev2) | The open weights. |
| [docs.cneuromod.ca](https://docs.cneuromod.ca/) | Courtois NeuroMod dataset documentation. |
| [charitydata.ca](https://www.charitydata.ca/charity/fondation-courtois/850271289RR0001/) | Fondation Courtois's Canadian charity filings. |
| [Moss v. Morgan Stanley, 719 F.2d 5](https://law.justia.com/cases/federal/appellate-courts/F2/719/5/148829/) | The federal appellate record of Courtois Jr.'s 1970s insider trading ring. |
| [ICIJ Offshore Leaks Database](https://offshoreleaks.icij.org/) | Panama Papers, Paradise Papers, Pandora Papers, Bahamas Leaks. Search any name from this post. |
| [SEC EDGAR](https://www.sec.gov/edgar.shtml) | Every 13F, 10-K, proxy statement. Verify the Big Three table in five minutes. |
| [NIH RePORTER](https://reporter.nih.gov/) | Full grant detail for NIH 1U54MH091657 and every other NIH grant touching TRIBE v2. |
| [FEC Individual Contributions](https://www.fec.gov/data/receipts/individual-contributions/) · [OpenSecrets](https://www.opensecrets.org/) | Every federal political donation by every person in this post. |
| [WikiLeaks](https://wikileaks.org/) · [Cryptome](https://cryptome.org/) | The leak archives. Search any name. The State Department cables, the Hacking Team leak, the Sony emails are all text-searchable. |
| [sirc-csars.gc.ca](http://www.sirc-csars.gc.ca/) | Canadian Security Intelligence Review Committee archive. |
| [cs.nyu.edu/~yann/research/lagr](https://cs.nyu.edu/~yann/research/lagr/) | Yann LeCun's own page documenting his DARPA LAGR program leadership. |
| [iqt.org/about](https://www.iqt.org/about) | In-Q-Tel's public "about" page — board of trustees, portfolio. |
| Any frontier LLM with long context | Feed it the above and ask *"which entities appear across more than one of these files and what connects them?"* That question wrote this post. |

None of this is secret. None of it was leaked. None of it required a whistleblower. The architecture was built in plain sight.

**The only thing that's new is that somebody finally sat with the public record long enough to write down every edge.**

That is engagement. The thing the attention economy is designed to prevent you from doing. Do it anyway.

---

## Bonus Info!

Three things the Brigade should watch for, because they're load-bearing for Part 2:

1. **Yann LeCun's AMI Labs cap table.** Part 2 will run that $1.03B syndicate end to end. Dassault, G42-adjacent Abu Dhabi money, Temasek, the Arnault family, Pinault, Mulliez, CMA CGM. Spoiler: Meta isn't on the cap table but has tech-access rights. What does that mean? Who's writing the structuring docs?

2. **The Courtois siblings.** Jacques Jr. funded the training data. His brother **Marc Courtois** chairs **Aireon LLC** — the space-based aircraft surveillance network running on the Iridium NEXT 66-satellite constellation, joint venture with NAV Canada, NATS (UK), ENAV (Italy), IAA (Ireland), Naviair (Denmark). Five-Eyes-adjacent infrastructure. His sister **Nicole Eaton** (ex-Harper-appointed Conservative senator) was named in the **ICIJ Bahamas Leaks** as a director of an undisclosed Bahamas offshore corporation for 12 years. Wild family.

3. **The Onion bought Infowars for $81K/month.** You probably saw the headline. You probably didn't see that **Everytown for Gun Safety** is the exclusive pre-locked multi-year launch advertiser. That isn't satire capturing a conspiracy outlet. That's a Bloomberg-funded gun-control apparatus capturing a grievance-radicalization distribution channel. Part 3 will run that whole architecture.

Stay engaged. Next one drops next Tuesday.

— Smartrick

---

*If you made it this far and you want the technical spine: the companion research paper **"The Lineage: From MKUltra to DECF"** lives at [airlocklabs.io/papers](https://airlocklabs.io). It's the academic receipts for everything above. Dense. Footnoted. Peer-review-grade.*

*Series is called "The Feed Is the Operation." Subscribe if you want the next one in your inbox. Share if you think someone should see it.*

*None of this proves a conspiracy. All of it is documented fact. Verify everything.*
