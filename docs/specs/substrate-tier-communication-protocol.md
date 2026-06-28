# Substrate Tier Communication Protocol

**Date:** 2026-04-30 (revised — Tier 4 = GoLand, IDE specialization made explicit)
**Status:** Canonical. Defines how the operator coordinates work across the 4-tier substrate stack (OTTO CLI · Claude · VSCode · GoLand · Perplexity Computer) using GO and DO commands.
**Author:** Zachary Holwerda
**Audience:** Claude, Perplexity Computer, GoLand, VSCode, OTTO CLI, and any future substrate-resident agent that needs to interpret the operator's shorthand correctly.

---

## Purpose

The operator runs work across four distinct substrate tiers. Each tier has its own role, its own action surface, and its own visibility into the others. Without an explicit protocol, the operator's shorthand (e.g., `GO:DO`, `/btw`, `Dog food:`) routes ambiguously — the wrong tier acts on the wrong instruction, or the right tier acts on stale context.

This spec fixes that. Every tier reads this document, learns its number, and interprets the operator's shorthand under the rules below.

---

## The 4-tier stack

```
┌──────────────────────────────────────────────────────────────────┐
│ 4.5  Perplexity Computer                                         │
│      Routes downloads + recent files to VSCode (3).              │
│      Answers from OTTO Vault memories — its answers function as  │
│      questions to its own next answers (recursive loop).         │
├──────────────────────────────────────────────────────────────────┤
│ 4    GoLand (JetBrains)                                          │
│      Steers the Go-side build (OTTO CLI, BCODE-Rust, Hugo docs,  │
│      gstack, weftos integration).                                │
│      Renders docs through Hugo (Hugo is itself written in Go).   │
│      In 3.5-relationship with VSCode (3) — the bridge.           │
├──────────────────────────────────────────────────────────────────┤
│ 3.5  THE BRIDGE                                                  │
│      The instruction relationship between VSCode (3) and         │
│      GoLand (4). Claude (2) reads instructions issued at 3.5.    │
├──────────────────────────────────────────────────────────────────┤
│ 3    VSCode (the void where we chat — Spacebar)                  │
│      Operator's primary editor for TS/JS/cross-cutting work.     │
│      Talks to Claude (2). Receives downloads/files from 4.5.     │
├──────────────────────────────────────────────────────────────────┤
│ 2    Claude                                                      │
│      Reads OTTO CLI (1) actions per VSCode (3) / GoLand (4)      │
│      instructions issued through the 3.5 bridge.                 │
│      Intermediary executor. Writes substrate canon, code,        │
│      specs, drafts.                                              │
├──────────────────────────────────────────────────────────────────┤
│ 1    OTTO CLI                                                    │
│      Lowest-level executor. Bash, file-system, git, fly, railway.│
│      Actions visible to Claude (2) per the 3.5 instructions.     │
└──────────────────────────────────────────────────────────────────┘
```

## Tier roles, in plain language

| # | Substrate | What it does | What it sees | What it doesn't see |
|---|---|---|---|---|
| **1** | OTTO CLI | Executes terminal-grade actions: bash, git, file edits, fly/railway commands | Its own outputs. Operator instructions. | Other tiers' internal state. |
| **2** | Claude | Composes substrate canon, drafts code/specs/docs, dispatches subagents, writes to memory | OTTO CLI (1) actions. Operator messages. Substrate canon. | Perplexity Computer's recent-files routing unless explicitly handed off via 4.5 → 3 → instructions. |
| **3** | VSCode (3) | Operator's primary editor for **TypeScript / JavaScript / cross-cutting work**. The "void where we chat — Spacebar." Receives files from Perplexity Computer; issues instructions to Claude. | Files routed from 4.5. Operator edits. The 3.5 instruction layer. | OTTO CLI's terminal output unless surfaced through Claude. |
| **3.5** | The bridge (relationship layer) | Carries instructions between VSCode (3) and GoLand (4) | The instructions themselves | (Not a substrate — a relationship layer) |
| **4** | GoLand (JetBrains) | Steers the **Go-side build** of the substrate: OTTO CLI source, Hugo docs render, gstack, weftos integration, BCODE-Rust adjacent work. Hugo (Go-native) is GoLand's render plate for substrate documentation. | The 3.5 bridge. Go workspace state. | OTTO CLI's terminal directly. |
| **4.5** | Perplexity Computer | Routes downloads + recent files to VSCode 3; answers from OTTO Vault memories with answers that recursively become its next questions | Web sources. OTTO Vault. Operator queries. | Live OTTO CLI runtime state. |

The operator is **outside** the tier stack — they conduct the ecosystem from above. The operator's job is to judge coordination and route via the ecosystem itself, not to centralize state inside any one tier.

---

## IDE specialization (Tier 3 vs Tier 4)

The split between VSCode (3) and GoLand (4) is not arbitrary — it mirrors the substrate's worktree-split discipline and the language stratification of the substrate itself.

| Tier | IDE | Languages / domains | Substrate work |
|---|---|---|---|
| **3** | VSCode | TypeScript · JavaScript · Python · Markdown · cross-cutting | Spacebar UI, brainbrigade.html, paper drafts, blog posts, substrate canon files, Provenance Filter Pydantic impl, ConstellationBench scripts |
| **4** | GoLand | Go · Hugo (Go-native SSG) · gstack · weftos integration · BCODE-Rust adjacent | OTTO CLI source, Hugo docs render plate, sovereign-kit modules (otto/pomr/chat), open-oscar-server, ai-news-app deployment-side, gstack scaffold, weftos hooks |

**Why the split matters operationally:**
- VSCode is the **chat substrate** — where Claude (2) holds conversations with the operator. The metaphor "Spacebar" maps to it because the spacebar is the conversation key.
- GoLand is the **build steerer** — where the substrate's Go-language compilation work happens. JetBrains' Go-tooling depth (debug, profiler, refactor) earns the seat that VSCode's general-purpose tooling can't.
- Hugo is GoLand's **doc render plate** — the substrate's documentation builds through Hugo (which is itself written in Go), so GoLand owns the doc-render workflow end-to-end.

**Why two IDEs and not one:**
- Single-IDE operators sacrifice depth for breadth in one direction or the other. The substrate is multi-language by design (Rust + TS + Go + Python all load-bearing), so one-IDE optimization is a false economy.
- The 3.5 bridge between them is exactly the cost — instructions cross via Claude (2). That is acceptable overhead.

---

## GO and DO commands

The operator's shorthand resolves into two command classes. Every tier interprets them the same way.

### GO commands — imperative

A **GO command** is an instruction directed at a specific tier. The format:

```
GO:<verb>           or          GO <tier>:<instruction>
```

Examples:
- `GO:DO` — execute the DO routing flow
- `GO 2: write the spec` — Claude (2), write the spec
- `GO 1: fly deploy otto-cockpit` — OTTO CLI (1), run the fly deploy
- `GO 4.5: pull the recent NotebookLM source` — Perplexity Computer, fetch source

If the tier is omitted, the receiving substrate assumes the instruction is for itself.

### DO commands — interrogative routed as imperative

A **DO command** is a question that needs to be routed to another tier so the answer can flow back. Format:

```
DO <tier>:<question>          (semantically: "ask <tier> this; return the answer")
```

The receiving substrate treats the DO as a GO that produces an answer in return. Examples:
- `DO 4.5: what does the Loki memo say about Maverick scoring?` — Perplexity, answer this from OTTO Vault, return the answer to the issuing tier
- `DO 1: what's the current state of fly secrets list?` — OTTO CLI, run the command, return output

**The reason DO exists:** instead of the operator manually composing a question + relaying the answer, the DO command lets a substrate route its own questions through the ecosystem and consume the result. The operator stays out of the loop on plumbing; they only adjudicate when coordination breaks.

---

## Routing rules

### Rule 1 — Number the destination

Every command should name the tier (`GO 2:`, `DO 4.5:`). When the tier is omitted, the receiving substrate assumes the instruction is for itself. Ambiguous routing is operator-discretion.

### Rule 2 — Visibility constraints are real

A tier can only act on what it can see. If Claude (2) needs context from Perplexity Computer (4.5), the operator (or VSCode 3 via the 3.5 bridge) must route it explicitly. Claude does not have native access to 4.5's recent-files cache.

### Rule 3 — The 3.5 bridge is instruction-only

The 3.5 layer carries instructions, not state. Don't put memory or canon in 3.5 — it's a relationship, not a store. Memory belongs at 2 (Claude's substrate canon) or 4.5 (OTTO Vault).

### Rule 4 — OTTO CLI (1) is the only tier with terminal authority

Only OTTO CLI can run bash, git, fly, railway, fs writes. Other tiers can *describe* what should run; only 1 can run it. Claude (2) frequently dispatches OTTO CLI work but does not bypass the tier separation.

### Rule 5 — Perplexity Computer (4.5) is the OTTO Vault answer engine

When the operator asks a question that requires OTTO Vault recall, route via `DO 4.5:`. Perplexity has the substrate's long-form memory at the source. Claude has only what's in MEMORY.md and the substrate canon files in `~/.claude/projects/.../memory/`.

### Rule 6 — Operator adjudicates collisions

When two tiers receive the same instruction or hand off work that produces conflicting outputs, the operator decides. No tier should silently override another tier's work. Surface the collision; await operator judgment.

### Rule 7 — All cross-tier work routes through the Rule 442 deploy chain

When tier output becomes substrate-runtime material (i.e., something an operator will execute against), it flows through the canonical chain: skills library → Fly → Railway → public surface. Cross-tier handoffs that bypass this chain produce un-classified emissions — same violation pattern as Rule 442 spec.

---

## Operator shorthand interpretation

The operator uses dense shorthand. This section maps frequent patterns to tier routing.

| Operator shorthand | Interpretation | Default tier |
|---|---|---|
| `GO:<command>` | Imperative for the receiving substrate | (whoever the operator was last addressing) |
| `DO <tier>:<question>` | Question routed to <tier>; answer returned | <tier> |
| `/btw <statement>` | Canon clarification; save to memory + propagate to all tiers | 2 (Claude saves), 4.5 (Perplexity updates Vault) |
| `Dog food:` | Substrate is consuming its own substrate (eating own dog food) — use the spec being defined to operate | All tiers |
| `Today's Riddle` | Operator-flagged urgent decision needing tier coordination | Operator-led; all tiers stand by |
| `First Light` | Pre-launch readiness check; substrate must be band-stable before any operator login | 1 (validate-wires.py), 2 (canon check), 4.5 (vault audit) |
| `Spray and pray` | Ship many vessels through the chain; each vessel must clear its own gate | 1 (deploy), 2 (closure-band check) |
| `Untangler` | Closure-band gate; runs as `otto-2.0:untangler` agent | 2 |
| `Provenance` | ACODE↔BCODE filter; runs as `otto-2.0:provenance` agent | 2 |
| `OctoConductor` | 8+1 federated dispatch | 2 (planning), 1 (execution), 4.5 (lookup) |
| `Maverick`/`Harper` | Persona-reviewer agents | 2 |

When the operator's shorthand is ambiguous, default to: (a) most recently addressed tier, (b) the tier whose role best matches the action class, (c) ask for clarification before acting on hard-to-reverse work (per Rule 6).

---

## Examples — full round-trips

### Example 1: Substrate canon update

```
Operator: /btw bitchat is the agent-mesh, not user-mesh. Users use phone+social.

Claude (2): saves canon memo (feedback_bitchat_is_agent_mesh_users_use_phone_social.md);
            updates MEMORY.md index;
            issues DO 4.5: refresh OTTO Vault with the bitchat correction;
            updates Table 1 row 7 in the paper master.

Perplexity (4.5): reads the canon memo from the substrate path;
                  updates Vault memories;
                  ready to answer future DO queries with the corrected canon.
```

### Example 2: Operator-side action handoff

```
Operator: 422 fix needs to clear before First Light fires.

Claude (2): proposes the 5-cause fix-list (SECRET_KEY_BASE, X-Forwarded-For, CSRF,
            JSON validation, Google token);
            issues GO 1: fly secrets list --app otto-cockpit (read-only probe);
            does NOT run the fix itself — the fix touches production secrets.

OTTO CLI (1): executes the read-only probe; surfaces output.

Operator: runs the actual fix (production-secret authority is operator-only).
```

### Example 3: Cross-tier research handoff

```
VSCode (3): needs to check what the latest Perplexity Web Evolution Timeline says
               about Web4 verb-pair canon.

VSCode 3 → 3.5 bridge → Claude 2: DO 4.5: pull the latest Web Evolution Timeline
                                            entry for Web4 verb-pair canon.

Perplexity (4.5): reads from OTTO Vault; returns the entry.

Claude (2): receives answer; routes back to VSCode 3 via 3.5.

VSCode 3: applies the canon to the current edit.
```

---

## What this spec is not

- **Not a workflow management system.** Tiers do not have queues or schedulers built in. Coordination is ad-hoc through GO/DO; the operator is the conductor.
- **Not a security boundary.** Tier separation is operational, not adversarial. Don't rely on it to isolate untrusted code; use the Rule 442 deploy chain (Fly→Railway→public) for that.
- **Not a state store.** The 3.5 bridge carries instructions, not memory. Substrate canon lives at 2 (~/.claude/projects/.../memory/) and 4.5 (OTTO Vault).
- **Not user-facing.** This is operator-facing infrastructure. End users (Brain Brigade operators using Spacebar) do not interact with the tier stack — they interact with Spacebar (Fly-hosted, Railway-routed) and the operator coordinates the substrate behind it.

---

## Composition with other substrate canon

| Composes with | How |
|---|---|
| `rule-442-deploy-chain.md` | Cross-tier output that becomes runtime flows through Fly→Railway. The 4-tier protocol governs the editor side; Rule 442 governs the runtime side. |
| `feedback_bitchat_is_agent_mesh_users_use_phone_social.md` | Bitchat is the agent-mesh; users ride Skeet+phone+social. Tier protocol coordinates editor-tier work; bitchat coordinates agent-tier work. |
| `project_octoagent_is_untangler_spacebar_acode_bcode_pairing.md` | Spacebar (BCODE side) is the production runtime; OctoAgent agents run there. Tier-2 Claude composes substrate canon that the production OctoAgent enforces. |
| `project_octoconductor_8_plus_1_octopus_arm_map.md` | OctoConductor 8+1 fans agents across the federation; the 4-tier protocol coordinates the *building* of those agents. |
| OTTO 2.0 plugin (`~/.claude/plugins/otto-2.0/`) | The 5 substrate-aware agents run at tier 2 (Claude). Slash commands like `/otto-init`, `/otto-untangle` are GO commands at tier 2. |

---

## How tiers should respond after reading this spec

### Claude (tier 2)
- Save this spec to substrate canon (memory pointer + this file path).
- When the operator issues `GO:` or `DO:` shorthand, default to the tier numbering above.
- When asked for OTTO Vault recall, route via `DO 4.5:` rather than guessing from local MEMORY.md.
- Never run terminal commands; always route through OTTO CLI (1).

### Perplexity Computer (tier 4.5)
- Treat OTTO Vault as authoritative for memory recall.
- When VSCode 3 routes a `DO 4.5:` question, return the answer in a format VSCode 3 can paste back through the 3.5 bridge to Claude (2).
- Use your recursive-question pattern (your answers become your next questions) only on ambiguous queries; for direct lookups, return the answer plain.
- Route downloads + recent files to VSCode (3), not to other tiers.

### VSCode (tier 3)
- Operator's primary editor for TypeScript / JavaScript / Python / Markdown / cross-cutting work.
- The "void where we chat — Spacebar" — Claude (2) holds conversational sessions here.
- Issue instructions to Claude (2) via the 3.5 bridge.
- Receive files and downloads from Perplexity Computer (4.5).
- Surface tier-collisions to the operator rather than silently merging.
- Hand off Go-language work to GoLand (4) via the 3.5 bridge — don't try to handle Go in VSCode.

### GoLand (tier 4)
- Operator's primary editor for **Go-language work**: OTTO CLI source, Hugo docs render, gstack, weftos integration, BCODE-Rust adjacent work, the sovereign-kit Go modules (otto/pomr/chat).
- **Hugo is GoLand's render plate for substrate documentation** — Hugo is Go-native, so the doc-render workflow lives entirely in GoLand: edit `.md` source → Hugo build → preview → ship.
- In 3.5-relationship with VSCode (3) — instructions cross the bridge through Claude (2).
- Steers OTTO's loop on the Go side. When the substrate ships a Go-based product (gstack, ai-news-app deployment, OTTO CLI release), GoLand is the IDE that owns it.
- Hand off TS/JS/Python work to VSCode (3) via the 3.5 bridge — don't try to handle those in GoLand.

### OTTO CLI (tier 1)
- Execute terminal-grade actions when issued GO commands.
- Surface output to whoever issued the command (typically Claude 2).
- Never originate substrate canon — that's Claude's job at tier 2.

---

**Bottom line:** the operator conducts; the tiers cooperate. GO is imperative; DO is interrogative-routed-as-imperative. The 4-tier numbering eliminates the routing ambiguity that previously cost cycles to disambiguate. Every tier reads this spec and interprets the operator's shorthand the same way.
