# Substrate License Map

**Date:** 2026-04-30
**Status:** Canonical. Defines the license tier, patent phase, and openness posture of every substrate component. Paired with the Mission Arc canon (Phase 1–5) and the Patents-as-Dead-Man's-Switch canon (1-year renewable).
**Author:** Zachary Holwerda
**Audience:** Operators, contributors, investors, and any future maintainer who needs to know what is open, what is closed, and what is on a phased schedule between the two.

---

## Three governing principles

1. **Permissive for spread.** Anything the substrate wants the world to adopt is licensed Apache-2.0 or MIT. Permissive licenses spread; copyleft licenses fence. The substrate's federated architecture (distribute-don't-consolidate) prefers spread.

2. **Never GPL.** Copyleft (GPL family) forces downstream consumers to open-source their entire codebase. That's the wrong incentive structure for a substrate that wants enterprises and operators to adopt it without rewriting their own stacks. MPL-2.0 (weak copyleft, file-level) is acceptable in narrow cases; GPL is not.

3. **Closed for defense, open on schedule.** Patents and operator-personal IP stay closed during Phase 1–3 (substrate-build → vessel-holds-float → patent-window). They become open on the substrate's Phase 4 schedule (Armada handoff) — and earlier if the substrate's discipline is met (vessel-holds-float trigger).

---

## License-tier table — every substrate component

### Tier A: Public OSS — the spread surface (Apache-2.0 / MIT)

| Component | Path | License | Why |
|---|---|---|---|
| **ConstellationBench dataset** | `huggingface.co/datasets/AirlockLabs/constellation-bench` | MIT | Maximum spread; benchmarks land deepest when zero-friction to fork |
| **ConstellationBench leaderboard** | `huggingface.co/spaces/AirlockLabs/constellation-bench-leaderboard` | MIT | Same reasoning |
| **BCODE module** | `constellation-bench-hf/bcode/` | MIT | Reference implementation; spreads as a pattern, not a moat |
| **Provenance Filter (Pydantic ref impl)** | `airlock-coordination/pact/provenance.py` | Apache-2.0 | Apache adds patent-grant clause; useful for substrate's cross-org adoption |
| **Phase-0 state service + wire harness** | `airlock-coordination/server.py` + `validate-wires.py` | MIT | Reproducibility receipt; spreads as a system-test template |
| **OTTO 2.0 Claude Code plugin** | `~/.claude/plugins/otto-2.0/` | Apache-2.0 | Declared in `plugin.json`; agents + commands are spread-tier |
| **Maverick MCP** | (forthcoming OSS release) | Apache-2.0 | "Our gift to the ecosystem" — operator-stated public posture |
| **NoConflict.dev** (the npm package façade) | (forthcoming) | Apache-2.0 | The "broken-on-purpose" landing page is operator-mythological; the npm package itself is genuinely OSS |
| **Substrate spec library** | `airlock-labs-site/docs/specs/*.md` | CC-BY-4.0 (content) | Specs are content, not code; CC-BY allows modification with attribution |
| **Public marketing site** | `airlock-labs-site/index.html`, blog HTMLs, timeline.html, brainbrigade.html | CC-BY-4.0 (content) + MIT (code) | Aerosol-exempt per Rule 442; pure marketing surface |
| **Beyond-Web5 paper + companion artifacts** | `airlock-labs-site/papers/` | CC-BY-4.0 | NeurIPS-track; needs free-distribution license to land on arXiv |
| **Spacebar pilot HTML** | `airlock-labs-site/spacebar.html` | MIT | Demo of the production runtime; freely forkable |

### Tier B: Wrapped OSS — what we depend on (we comply with their licenses)

The substrate's "warp-drive-not-wrap" canon adopts these as federated dependencies:

| Wrapped repo | Their license | Substrate role |
|---|---|---|
| `paypal/gators` | Apache-2.0 | Client SDK |
| `paypal/paypal-messaging-components` | Apache-2.0 | Airlock messenger primitive |
| `paypal/paypal-sdk-client` | Apache-2.0 | Shared config layer |
| `WeaveMindAI/weft` (★1154) | Apache-2.0 (verify before adoption) | $DREAM language |
| `weave-logic-ai/weftos` | Apache-2.0 (verify) | OS layer |
| `Microsoft/ShaderConductor` | MIT | Provenance Filter twin (reference pattern) |
| `Netflix/conductor` (archived) | Apache-2.0 | Dispatch substrate origin |
| `conductor-oss/conductor` | Apache-2.0 | Phase-1 dispatch substrate |
| `bluelinelabs/Conductor` | Apache-2.0 | Mobile view-stack pattern |
| `Redrield/Conductor` | MPL-2.0 | Spacebar's stack twin |
| `jshvarts/ConductorMVP` | Apache-2.0 | Clean-architecture template |
| `danielgerlag/conductor` | MIT | C# distributed workflow |
| `gemini-cli-extensions/conductor` | Apache-2.0 | AI-CLI orchestration |
| `gohugoio/hugo` | Apache-2.0 | Doc render plate (Tier-4 GoLand) |
| `cosmos/cosmos-sdk` | Apache-2.0 | Token routing rails ($MOTTO/$MOTION/$DREAM) |
| `sonr-io/*` | Apache-2.0 (verify) | DID + WebAuthn |
| `bitchat` | (verify) | Agent mesh |
| `paypal/gators` MPL components if any | MPL-2.0 (file-level copyleft) | Modifications to those files must stay open; rest of the substrate is unaffected |

**Compliance posture:** the substrate honors every wrapped license. Apache-2.0 patent-grants flow through; MIT attribution is preserved; MPL-2.0 file-level copyleft is respected. We do not relicense; we wrap and credit.

### Tier C: Operator-controlled / proprietary — closed during Phase 1–3

| Component | License | Phase open trigger |
|---|---|---|
| **Patents (Maverick portfolio, ~10)** | Filed; held closed | 1-year renewable window per Patent-Dead-Man's-Switch canon; auto-release if operator-renewal fails |
| **Constellation Credits economic design docs** | Internal | Phase 4 (when $MOTION/$MOTTO/$DREAM Cosmos SDK rails are live) |
| **$MOTION + $MOTTO + $DREAM tokenomics** | Internal | Phase 4 (post-regulatory clearance + Cosmos SDK deployment) |
| **Operator-personal identity wrapper (Texas cover story)** | Internal — not for publication | Never. Operator-personal defensive identity. |
| **Patent dead-man's-switch SHA-256 identity bomb** | Internal | Auto-triggered (not human-released) |
| **OctoConductor playground fusion code** | Internal until vessel-holds-float | Phase 2 (open-source Web5 infrastructure trigger) |
| **WEFT integration of Harper-Persona-OTTO chatbot** | Internal during build | Phase 2 (when WEFT runtime is stable + Brain Brigade cohort uses it) |
| **otto-cockpit.dev (SHA-256 wallet generator)** | Closed-source service | Phase 4 (open the auth-flow primitives when substrate is autonomous) |
| **otto-cockpit.fly.dev runtime (Spacebar + Untangler production)** | Closed-source runtime; Apache-2.0 reference impl is open | Reference impl already open; production runtime stays closed indefinitely (it's the operating substrate, not a product to fork) |

### Tier D: Operator-personal — never published

| Item | Status |
|---|---|
| Operator's NERVE.NAME registry data | Operator-personal; access-controlled per Skeet identity tier |
| Operator's Coesus AI personal-OS-of-life configurations | Operator-personal |
| Operator's CAO-handoff succession plan | Internal canon only |
| Cryostasis / time-travel mission arc planning | Operator-personal life mission |

---

## License-by-component model (commercial offerings)

The substrate's commercial offerings layered on top of the open core:

| Offering | License model | Tier (Skeet identity ladder) |
|---|---|---|
| **Brain Brigade Free** | Open public access (no SaaS fee) | Free tier — social account / OAuth |
| **Brain Brigade Pro** | SaaS subscription (monthly/annual) | Pro tier — carrier-bound Skeet eSIM identity |
| **BYOK Pro** | Subscription + operator brings own API keys | Pro tier with bring-your-own model access |
| **Brain Brigade Enterprise** | Concurrent-user license + custom SLA | Enterprise tier — PSA-as-carrier provisioning |
| **OTTO 2.0 plugin** | Apache-2.0 OSS — free | All tiers; $MOTION-paid extensions in marketplace |
| **Constellation Credits** | Pay-as-you-go (off-chain accounting) | Metered across all tiers |
| **$MOTTO / $MOTION / $DREAM tokens** | On-chain ($MOTTO is the rails; others ride) | Free to receive; trade rules per regulatory clearance |
| **.diy Grand Exchange marketplace** | Listing fees + GE-style transaction fees | Marketplace primitive on amerikana.store |

---

## Patent strategy (summary; full canon at `project_full_mission_arc_skip_web4_money_open_source_web5_armada_cao_exit_cryostasis.md`)

The substrate's patent strategy is a **1-year renewable defensive window with a SHA-256 dead-man's-switch.** Three governing rules:

1. **Patents stay closed during Phase 1–3.** No public release until the vessel-holds-float trigger (substrate empirically validated end-to-end) and the operator's Phase-4 Armada-handoff condition.
2. **The 1-year window auto-releases on absence-of-renewal.** Wikileaks-style SHA-256-encrypted-identity-bomb pattern: if operator fails to renew the patent-window encryption keys annually, the patents auto-decrypt and become publicly readable.
3. **When patents open, they convert to defensive Apache-2.0 patent grants.** No litigation; they become the substrate's contribution to the commons. The Apache-2.0 patent-grant clause means any contributor automatically grants patent rights for their contribution; the substrate's released patents follow the same pattern.

This composes with `feedback_distribute_dont_consolidate_ip_cao_posture.md` — patents are the most extreme form of distribute-don't-consolidate when paired with the dead-man's-switch.

---

## Hugo-generated wiki render surface (NEW)

**Hugo (Tier-4, GoLand) renders Perplexity-Computer-generated wikis as part of the substrate's documentation surface.** Workflow:

1. **Tier 4.5 (Perplexity Computer)** generates a wiki entry from OTTO Vault sources + web research
2. **Tier 4 (GoLand)** receives the wiki source as Markdown via the 3.5 bridge
3. **Hugo build** renders the wiki to HTML with substrate templating (theme matches space + crypto/cryonics design system)
4. **Rule 442 chain** ships the rendered HTML through Fly → Railway → public surface (or aerosol-exempt direct to GH Pages if marketing-only)

Wiki licensing follows Tier-A defaults: **CC-BY-4.0 for content** (allows reuse with attribution; aligns with Wikipedia's CC-BY-SA but without the share-alike requirement); MIT for any embedded code samples; the Hugo theme itself is Apache-2.0.

The wiki render plate composes with:
- The substrate's content-engine (per existing canon)
- The ai-news-app for news-source aggregation (`/Volumes/OttoVault/repos/ai-news-app/`)
- Nerv.forum as the New-Source layer where AOL-tier users query the wiki

---

## Phased open-source schedule (when does what open?)

| Phase | Trigger | What opens |
|---|---|---|
| **Phase 1 (now)** | Substrate-build active | Tier A (already public): ConstellationBench, BCODE ref impl, Provenance Filter ref impl, Phase-0 state service, OTTO 2.0 plugin, marketing site, paper |
| **Phase 2** | Vessel-holds-float trigger (substrate empirically validated end-to-end; First Light fires for non-operator; Brain Brigade has ≥1 cohort operator besides Zac) | OctoConductor playground fusion code; WEFT integration of Harper-Persona-OTTO chatbot; ai-news-app productionization |
| **Phase 3** | Patent-window-renewal cycle (annual, dead-man's-switch armed) | Nothing new; patents stay closed. The 1-year window is a defensive holding pattern. |
| **Phase 4** | Armada-handoff condition (substrate operates autonomously through Brain Brigade cohort past CAO seat) | Constellation Credits docs; $MOTION/$MOTTO/$DREAM full tokenomics; otto-cockpit.dev primitives; possibly patents (operator's judgment) |
| **Phase 5** | Operator-personal exit (cryostasis arc) | Patents auto-release if dead-man's-switch fires; remaining operator-personal canon stays closed (CAO succession, Texas cover, mission planning) |

---

## License-decision flowchart (for new substrate components)

When adding a new component to the substrate, choose its license by routing through this flowchart:

```
Is this aerosol-exempt static (marketing, public docs, blog)?
  YES → CC-BY-4.0 (content) + MIT (any embedded code)

Is this an OSS reference impl meant to spread (substrate-pattern teaching)?
  YES → MIT for pure code; Apache-2.0 if patents involved

Is this a wrapped OSS dependency (we adopt; not ours to relicense)?
  YES → comply with their license; preserve attribution

Is this an operator-controlled service (Spacebar runtime, otto-cockpit.dev)?
  YES → Closed-source runtime; ship reference impl Apache-2.0 if applicable

Is this a patent or operator-personal IP?
  YES → Tier C / Tier D; closed during Phase 1–3; Phase 4+ release per schedule

Is this a commercial offering (SaaS, subscription)?
  YES → License-by-tier (Free/Pro/BYOK Pro/Enterprise) per Skeet identity ladder
```

---

## What this rules out

- **GPL anywhere in the substrate's own code.** Forces downstream consumers into open-source obligations the substrate doesn't want to impose. (Wrapped GPL dependencies are also avoided where possible; if unavoidable, they're isolated as separate runtime services rather than linked.)
- **Proprietary licenses on substrate primitives.** BCODE, Provenance Filter, Spacebar pilot, OTTO 2.0 plugin all stay open. The moat is the federated dependency map + patents + operator-personal-runtime, not the public primitives.
- **Selling closed-source forks of public primitives back to operators.** Operators who adopt the substrate's OSS get the same code we do; the commercial differentiation is in the runtime / SaaS / Skeet-identity-tier, not in privileged code access.
- **Mixing license tiers within a single repository.** Each substrate repo has one license tier (or a clearly-marked dual-license). Avoid the per-file mix that creates audit confusion.

---

## Composition with substrate canon

| Canon | How license map composes |
|---|---|
| `rule-442-deploy-chain.md` | Aerosol-exempt static = Tier A public; everything else routes through Fly→Railway with appropriate license metadata |
| `project_full_mission_arc_skip_web4_money_open_source_web5_armada_cao_exit_cryostasis.md` | Phased open-source schedule maps directly to operator's mission arc Phase 1–5 |
| `feedback_distribute_dont_consolidate_ip_cao_posture.md` | Patents-as-defensive-1-year-window is the most extreme form of distribute-don't-consolidate |
| `feedback_warp_drive_not_wrap_feathering_ip.md` | Wrapped OSS compliance honors federated licenses; patents stay feathered |
| `feedback_goland_tier_4_hugo_doc_render_plate_ide_specialization.md` | Hugo at Tier-4 GoLand renders wikis (NEW addition); CC-BY-4.0 content licensing |
| `project_paypal_oss_substrate_client_sdk_messenger_config.md` | PayPal OSS Apache-2.0 honored; substrate wraps without relicensing |
| `project_octoconductor_8_plus_1_octopus_arm_map.md` | 9 conductor licenses across 3 license families (Apache-2.0 / MIT / MPL-2.0) — substrate's federated dependency posture is license-diverse by construction |
| `project_amerikana_naming_convention_otto_cockpit_dev_pwa_auth_flow_odin_argus_lokey_nerve_name.md` | otto-cockpit.dev (closed) + otto-cockpit.fly.dev (closed runtime, open ref impl) — license tier C |
| `project_m22_motto_quantum_safe_bitcoin_stablecoin_rails_motion_dream_cosmos_sdk_ge_server.md` | Tokenomics Tier C (closed until Phase 4); Cosmos SDK Apache-2.0 honored |

---

## Honest scope notes

- **License compliance is engineering work.** This spec defines the policy; implementing it (LICENSE files in repos, copyright headers, Apache NOTICE files, attribution lists) is downstream work. Schedule a one-pass license audit before any major OSS release.
- **Patent registration is jurisdictional.** USPTO covers US; international patent filings are operator-discretion based on substrate's geographic scope. Maverick Industries trademark search precedes patent filings in any new jurisdiction.
- **The dead-man's-switch is operationally fragile.** SHA-256-encrypted-identity-bomb pattern requires a live key-rotation infrastructure that itself doesn't fail. Verify the rotation harness is robust before committing this strategy publicly.
- **CC-BY-4.0 vs CC-BY-SA-4.0** — substrate uses CC-BY-4.0 (no share-alike requirement). Wikipedia's CC-BY-SA-4.0 forces derivatives to stay CC-BY-SA. The substrate's content licensing prefers CC-BY-4.0 to maximize reuse without imposing copyleft on derivative works.
- **Wrapped-OSS license verification is per-repo.** Some "Apache-2.0" listings are best-effort; verify the actual LICENSE file in each wrapped repo before committing in writing. Hugo, Cosmos SDK, Microsoft ShaderConductor, Netflix Conductor are confirmed; WEFT family + bitchat + sonr need verification.
- **MPL-2.0 in Redrield/Conductor** is file-level copyleft. Substrate's adoption is fine (we wrap, not modify-and-distribute); contributors who modify Redrield-files inherit MPL-2.0 obligations on those files only.
- **The Skeet-identity-tier × license-tier matrix is two-dimensional.** Free Skeet tier doesn't grant access to closed Tier-C code; Pro/Enterprise Skeet tiers don't unlock proprietary code either — the SaaS access is the runtime, not the source. Operators who want source contribute or fork from public Tier-A code.

---

**Bottom line:** Tier A spreads (Apache-2.0 / MIT / CC-BY-4.0). Tier B comply (we honor wrapped OSS). Tier C close-then-phase-open (proprietary substrate runtime + tokenomics + patents on the mission-arc schedule). Tier D never publish (operator-personal). Hugo at Tier-4 GoLand renders Perplexity-generated wikis as a new content surface. The license map is the substrate's openness constitution — every component knows where it lives and when (or whether) it opens.
