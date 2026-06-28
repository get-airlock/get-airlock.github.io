---
title: "The Expression-Substrate Business Model"
subtitle: "Why AI apps are not GPT wrappers and why the inversion matters for founders, operators, and investors"
author: "Zachary Holwerda"
affiliation: "Airlock Labs"
date: "2026-04-22 (working draft)"
audience: "Founders, operators, investors, regulators"
paper_class: "Whitepaper, plain language"
---

## The inversion

An expression-substrate business is not a specific kind of company. It is a concept. It is a role a business takes when it structures itself to carry user expression through a commercial substrate, rather than to extract user behavioral residue as a proprietary asset. Any business whose value lives in the data, service, or domain it provides — and whose customer relationship depends on the user trusting the business with something the user is not willing to have absorbed into a permanent behavioral dataset — is already a candidate. The question is not whether to become one. The question is whether to keep pretending to be a different kind of business while the regulations tighten and the customer trust erodes.

The AI industry has a default story about what an AI application is. A company takes a large language model, wraps an interface around it, routes user queries into the model, and sells the output. The model is the factory. The app is the wrapper. The user is the raw material. The valuable asset, the one venture capital is underwriting at forty-billion-dollar valuations, is the aggregated behavioral residue left behind as users pass through the factory.

This is a bad story. It is structurally upside down.

The better story, and the one this whitepaper argues is already quietly true for the businesses that are actually working, is this: **businesses are not wrappers around a model. Businesses are wrappers around user expression.** The user brings sovereignty, intent, and context. The business brings data, service, or domain. The model is a utility that runs underneath. The product is not the model's output. The product is the user's expression, made more fluent and more useful by passing through the business's substrate.

In this frame, TraderVerse is not a GPT wrapper for traders. TraderVerse is a substrate for trader expression, providing financial data and execution infrastructure that lets a trader's intent land on the market. A clinical decision-support tool is not a GPT wrapper for doctors. It is a substrate for clinical expression, providing medical literature and workflow infrastructure that lets a doctor's reasoning land on a patient. A legal research tool is not a GPT wrapper for lawyers. It is a substrate for legal expression.

The model is interchangeable. Next year's Qwen beats this year's Opus on the benchmarks that matter for most of these tasks, and the year after that a 30B local model will beat them both. The business's moat is not the model. The moat is the substrate. The substrate is what the user's expression flows through. The substrate is what makes the expression commercially valuable.

## Why the industry got this wrong

The industry got the story backwards because the first wave of AI companies to hit scale, OpenAI, Anthropic, Google, Meta, were companies whose entire value proposition was the model itself. For a foundation model lab, it is literally true that the business is a wrapper around the model. The model is the product. The aggregated user traffic is the moat.

Application layer companies then copied the business model without copying the underlying economics. They started treating their users the way foundation model labs treat their users: as raw material whose behavioral residue compounds into a proprietary dataset. This works for a foundation model lab because each user's behavior helps train better versions of the same model that every other user is interacting with. It does not work for an application layer company because the application's value was never the model. The value was the domain, the data, the workflow, the customer relationships, the regulatory certifications, the integrations with other systems. The behavioral residue the application captures is useful to precisely one entity, the application itself, and only so long as that application is the one customers keep using.

When an application layer company captures user behavioral residue as if it were a foundation model lab, three things happen. First, it takes on regulatory risk it never needed to take on, because GDPR and CCPA were written for companies that store user data, and behavioral residue is user data. Second, it creates an attack surface for model inversion and membership inference that it cannot easily defend. Third, and most damaging, it signals to its customers that the customer relationship is an extraction relationship rather than a service relationship, which corrodes trust and retention in exactly the verticals where trust and retention are the whole product: finance, healthcare, law, education.

The inversion fixes all three problems at once by moving the locus of sovereignty from the business to the user, and the locus of expertise from the model to the substrate.

## The three roles

A functional AI application in this frame has three distinct roles, and the boundaries between them are load-bearing.

**The user is sovereign.** The user brings wallet, persona, and identity. The user's behavioral kernel, their drive profile, their interaction history, their preferences, belongs to the user. It is reconstructed from session data the user controls every time they connect. When they disconnect, the reconstruction is torn down. The user's state never crosses into persistent infrastructure the business owns.

**The router is a utility.** The router is the technical layer that holds the user's behavioral kernel live during a session and uses it to select which model to call for which turn. The router is stateful within a session, stateless across sessions. It is a courier, not a curator. It preserves the user's expression in flight. When the session ends, its cache is purged.

**The business is a substrate.** The business provides the data, service, or domain that the user's expression needs in order to land. For TraderVerse, the substrate is financial data and execution. For a clinical tool, the substrate is medical literature and workflow. For a legal tool, the substrate is case law and document templates. The business's value is this substrate. The business's moat is the quality of this substrate, the exclusivity of the data feeds, the regulatory certifications, the customer relationships, the integrations. None of these require the business to capture user behavioral residue.

The three roles together form a triad. Each role contributes something the others cannot substitute for. The user cannot be the substrate because the user does not have the domain data. The substrate cannot be the user because the substrate does not have the intent. The router cannot be either because the router only makes sense as the layer that connects them.

## Which businesses benefit and which lose

This inversion is not neutral. It is commercially adversarial to a specific class of business and commercially favorable to another class. Being honest about the split is important.

**Businesses that benefit.** Any business whose actual value proposition is a data asset or service infrastructure, separate from aggregated user behavior, benefits from the inversion. These businesses were already sitting on the valuable thing. They just did not realize they could stop paying the extraction tax. This class includes regulated verticals like financial services, clinical software, legal tech, and education. It includes domain-heavy SaaS where the data flywheel is about the domain, not about users. It includes any business whose customers would happily pay more for the guarantee that their behavioral kernel stays their own, which in practice is most of the high-value verticals.

**Businesses that lose.** Businesses whose value proposition is aggregated user behavioral data lose under the inversion. This is the advertising-financed consumer internet, which is to say Meta, Google, TikTok, and their smaller peers. It is also the foundation model labs to the extent that their valuation depends on continued behavioral data capture. For these businesses, the entire moat is the data they extract from users. An architecture that prevents that extraction does not just fail to serve them. It is existentially adversarial to them.

This split matters because it determines who will adopt the architecture and who will fight it. The businesses that benefit can adopt the architecture unilaterally, and many of them are already halfway there. They are the target market. The businesses that lose will not adopt the architecture voluntarily. They must either be regulated into compliance or outcompeted by new entrants who ship the architecture natively.

## The web4 framing

Every few years someone proclaims the arrival of a new web era and most of the time it is marketing. The honest version of the claim we are making is this.

Web1 was read. Static pages, single direction. Web2 was read and write. Users generated content but the platforms owned what users created. Web3 was read, write, and own, where ownership applied to tokens and on-chain assets. The unit of ownership was a scarce digital object. The architecture centered on permanence and auditability.

The inversion proposed here implies a fourth layer, which we will call Web4 for the sake of naming it at all. Web4 is read, write, own, and express, where the unit that is owned is not a token but an expression. Expression is live. It is not a static object. It cannot be persisted on chain without losing what makes it expression. It requires an architecture that is stateful within a moment and discardable across moments. The commercial unit is not the token. The commercial unit is the session.

The distinction from Web3 matters. Web3 made tokens portable across services. Web4 makes expression portable across services. Token portability was solved by making the token an on-chain object. Expression portability cannot be solved that way because expression is not an object. It is a live state that exists only during interaction. It is solved instead by an entanglement-safe handshake protocol, which the companion whitepaper describes in technical detail.

What web4 changes commercially is what you are selling. In web2 you were selling access to other users. In web3 you were selling access to liquidity. In web4 you are selling access to a substrate that users flow through. Your moat is the substrate. Your customer relationship is with the user, not with the user's data.

## Worked example: TraderVerse

It is useful to walk through a concrete case so the framing has teeth.

TraderVerse is a financial services platform serving retail and semi-pro traders. Its data feeds include market data, options chains, fundamentals, news, and alternative data sources. Its service infrastructure includes execution, portfolio management, and performance analytics. Its customers are traders who want to make better decisions and execute them faster.

Under the industry default, TraderVerse would build an AI assistant that wraps a frontier model, routes trader queries into it, stores every query and response, and uses the aggregated behavioral data to fine-tune the assistant or sell the behavioral data to third parties. This is how most AI-in-fintech currently works. It is also why the regulatory burden on AI-in-fintech is rising sharply, because the aggregated behavioral data is material nonpublic information about the traders themselves.

Under the expression-substrate inversion, TraderVerse builds the same assistant but flips the architecture. The trader's behavioral kernel, their risk tolerance, their time horizon, their style, their domain specialization, is reconstructed per session from data the trader controls. TraderVerse's substrate, the data feeds and execution infrastructure, is the same as before. The router, which can be the Airlock Router or any equivalent, holds the kernel live during the session and selects models per turn. When the trader logs out, the kernel is gone from TraderVerse's infrastructure. TraderVerse never accumulates behavioral data about traders. It does not need to. The substrate is the moat.

The trader experience is better because the assistant behaves like a private advisor rather than like a surveilled user of a corporate tool. The regulatory posture is better because TraderVerse can truthfully claim it does not retain trader behavioral data. The competitive posture is better because the substrate, the financial data feeds and execution quality, are things competitors cannot easily replicate. The cost structure is better because TraderVerse is not paying to store and process behavioral data it does not need.

What TraderVerse gives up is the ability to sell aggregated trader behavioral data to third parties. In financial services, that was never going to end well anyway.

## Commercial implications

The inversion has specific consequences for how businesses that adopt it should structure their operations.

**Pricing.** Expression-substrate businesses price on the substrate, not on the model. The cost of the model is a wholesale utility cost, not a margin center. Customers pay for access to the substrate, with the AI assistant included as a mechanism for accessing the substrate more effectively. This matches how customers already think about what they are buying.

The structural claim behind this pricing model can be stated as a dynamic equilibrium. Define $V_u$ as the value the user contributes to the session (expression, attention, data, direct payment), $V_b$ as the value the user receives (service utility, substrate access, produced response), and $V_r$ as the router's overhead. Within a session, these quantities oscillate: the user asks, the system responds, the user refines, the system routes, and so on. Instantaneous balance is never zero; the exchange is a dynamic back-and-forth, not a frozen transaction. What sovereign balance requires is that the time-average over the session satisfies $\langle V_u - V_b - V_r \rangle \approx 0$. Business revenue comes from charging for substrate access, which is a real and measurable value the business provides. It does not come from the residual $\langle V_u - V_b \rangle$ surplus harvested when the exchange is engineered to be chronically asymmetric.

That residual harvesting is what the advertising-financed internet calls a business model. Platforms engineer the time-average $\langle V_u - V_b \rangle > 0$ by biasing user-contribution signals (infinite scroll, notification loops, engagement optimization) faster than user-benefit signals. The accumulated drift becomes the revenue. The attention economy is the name for this category, and it is structurally what the expression-substrate model refuses to build.

Under the expression-substrate architecture, the session's internal oscillations are allowed and expected. What is prohibited, architecturally rather than merely rhetorically, is the accumulation of time-averaged drift that would convert the user's expression into the business's commercial asset. Revenue is tied to substrate value, which the business provides independently, rather than to residual surplus extracted from user-side contribution.

**Licensing.** Model providers become wholesale utility providers. Router providers become wholesale infrastructure providers. Substrate businesses license the models and the routers on wholesale terms. This is a clean stack that mirrors the cloud infrastructure stack, and it is simpler to reason about than the current tangle of application-layer companies each trying to build their own moat out of the same foundation models.

**Defensibility.** The defensible moat is the substrate. For substrate businesses, the commercial question is therefore the same as it has always been: what data, service, domain, or relationship makes your business hard to replicate? AI does not change the answer. AI just stops obscuring the answer with a false moat made of aggregated user data.

**Regulation.** Expression-substrate businesses are in a structurally defensible regulatory position under any future extension of GDPR or CCPA to cover behavioral inference. They are not capturing the thing the regulations will eventually regulate. Aggregation-based businesses are not in a structurally defensible regulatory position, and should expect the regulatory climate to keep tightening against them.

**Fundraising.** For investors, the expression-substrate model offers a cleaner thesis than the application-layer AI consensus of the 2024 to 2026 period. The application layer had a thesis that was really a bet on data flywheels that would not materialize at the application layer. The expression-substrate thesis is a bet on underlying substrate quality, which is a bet investors have been making profitably for decades across enterprise SaaS.

## Closing

The industry story about what AI applications are has been a category error since 2023. AI applications are not wrappers around models. They are substrates through which user expression flows. The user is sovereign. The router is a utility. The business is a courier. The model is wholesale infrastructure.

Businesses that recognize this inversion early will find that they had the moat all along. They were just paying a behavioral-data tax that was not required.

Businesses that fight the inversion will find their moat dissolving as foundation models commoditize, regulation tightens, and customer trust erodes.

The point of naming the inversion now is that it is not a future state. It is already the architecture that the best AI deployments are converging toward. We are just giving it the vocabulary it needs.

Companion whitepapers describe the technical protocol that implements the inversion (the entanglement-safe handshake) and the structural audit literature that documents its adversarial properties (borrowing from cryptographic hash analysis). This whitepaper is the commercial frame. The architecture follows.

---

*Airlock Labs · airlocklabs.io · admin@airlocklabs.io*
