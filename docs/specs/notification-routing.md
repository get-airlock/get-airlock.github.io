# Notification Routing Spec

**Date:** 2026-04-30
**Status:** Canonical. Defines what fires Pushover (mobile push) vs Resend (email receipt) vs OTTO Vault (audit-only) for the substrate's orchestration events. Operator-facing visibility layer.
**Author:** Zachary Holwerda

---

## The three notification channels

| Channel | Purpose | Latency target | Operator action class | Where it lives |
|---|---|---|---|---|
| **Pushover** | Real-time mobile push for events the operator may need to step into | Seconds | Awareness · "do I need to act now?" | iOS/Android Pushover app |
| **Resend** | Email receipts for events that need a paper trail or operator review on a slower clock | Minutes–hours | Receipt · "did this happen as expected?" | Inbox at the operator's new Noble Mobile-bound address |
| **OTTO Vault (audit-only)** | Canonical record of every substrate event — silent, no notification | (none — pull-only) | Forensic · "what happened?" | OTTO Vault on Perplexity Computer (tier 4.5) |

**Routing principle:** the louder the channel, the higher the bar. Most events log silently to OTTO Vault. Some events email to Resend. A small number of events push to Pushover. The operator should never feel buzzed for things they don't need to act on.

---

## Event severity ladder

| Severity | Channel(s) | Definition | Examples |
|---|---|---|---|
| **S0 — Routine** | OTTO Vault only | Substrate operating in band; no operator attention required | Closure-band check passes; OctoConductor arm dispatches; PACT envelope signs cleanly |
| **S1 — Receipt** | OTTO Vault + Resend | Worth a paper trail; operator reviews on their own clock | Deploy completes; investor narrative pushed to airlocklabs.io; brainbrigade.html updated |
| **S2 — Awareness** | OTTO Vault + Resend + Pushover (silent) | Operator should know but action is not urgent | Untangler closure rejected (non-blocking); 422 fix passes a probe; new operator subscribes via Threema feed |
| **S3 — Step-in** | OTTO Vault + Resend + Pushover (sound) | Operator needs to act this hour | Apex-style payment anomaly; Joshua reply deadline approaching; Addie TG group message; First Light gate clear; LinkedIn DM from a named-canon contact (Harper, Eric, Sarim) |
| **S4 — Stop** | OTTO Vault + Resend + Pushover (priority emergency) | Operator must act NOW; ignored notification will repeat every 60s for 30 min | Substrate auth gate fails (otto-cockpit 422 hard error); Untangler rejects critical-chamber action; production runtime breach; deploy chain bypass detected (Rule 442 violation) |

---

## Default fire-table (per substrate event)

### Build & deploy events

| Event | Severity | Pushover sound |
|---|---|---|
| Local commit pushed to GitHub | S0 (silent) | — |
| GitHub Pages deploy succeeds | S1 (receipt) | — |
| Fly deploy starts | S1 | — |
| Fly deploy succeeds (otto-cockpit) | S2 | quiet |
| Fly deploy fails | S3 | classical |
| Railway emission succeeds | S1 | — |
| Railway picks up Fly emission (handoff verified) | S2 | quiet |
| Rule 442 bypass detected | S4 | siren (loop) |

### Orchestration events

| Event | Severity | Pushover sound |
|---|---|---|
| OctoConductor 8+1 dispatch fires | S0 | — |
| Untangler accepts closure | S0 | — |
| Untangler rejects closure (non-critical chamber) | S2 | quiet |
| Untangler rejects closure (Control chamber / critical) | S3 | classical |
| Provenance Filter denies payload | S2 | quiet |
| BCODE closure ledger writes | S0 | — |
| 422 gate at otto-cockpit fails | S4 | siren |
| First Light fires (first operator login) | S3 | bugle |

### Operator-attention events

| Event | Severity | Pushover sound |
|---|---|---|
| Named-canon contact messages (Addie, Harper, Eric, Sarim, Matt, Joshua, Cameron) | S3 | classical |
| Threema feed subscriber count crosses thresholds (10, 50, 100, 1k) | S2 | cosmic (small, no repeat) |
| Brain Brigade signup form (Formspree) submission | S2 | classical |
| NeurIPS deadline (May 4 abstract / May 6 paper) approaching < 24h | S3 | classical (every 6h) |
| TBE Demo Day reminder | S2 | classical |
| Miami after-party (luma) reminder | S2 | classical |
| Apex-style payment anomaly (duplicate charge, mismatch entity) | S3 | tugboat |

### Memory & canon events

| Event | Severity | Pushover sound |
|---|---|---|
| New substrate canon memo saved | S0 | — |
| Operator-corrected canon (`/btw …`) | S1 (receipt) | — |
| MEMORY.md size exceeds 24KB warning | S2 | quiet |
| ICM/LARQL forgets / drift detected | S2 | quiet |

---

## Pushover wiring (operator-side)

The keys are already in `/Volumes/OttoVault/repos/airlock-config/secrets/.env.constellation`:

```
PUSHOVER_USER_KEY=<operator's Pushover user key>
PUSHOVER_APP_TOKEN=<substrate app token>
```

Reference helper (Python — drop into `airlock-coordination/notify/pushover.py`):

```python
import os, requests
from typing import Literal

Severity = Literal["S0", "S1", "S2", "S3", "S4"]

# Pushover priority: -2 silent, -1 quiet, 0 default, 1 high, 2 emergency
SEV_PRIORITY = {"S0": -2, "S1": -1, "S2": 0, "S3": 1, "S4": 2}

def push(title: str, message: str, severity: Severity, sound: str | None = None,
         url: str | None = None, url_title: str | None = None) -> bool:
    """Send a Pushover notification. Returns True on success."""
    if severity == "S0":
        return True  # log to vault only; no push
    payload = {
        "token": os.environ["PUSHOVER_APP_TOKEN"],
        "user": os.environ["PUSHOVER_USER_KEY"],
        "title": title,
        "message": message,
        "priority": SEV_PRIORITY[severity],
    }
    if sound:
        payload["sound"] = sound
    if url:
        payload["url"] = url
        if url_title:
            payload["url_title"] = url_title
    if severity == "S4":
        # Emergency: retry every 60s, expire after 30 min
        payload.update({"retry": 60, "expire": 1800})
    r = requests.post("https://api.pushover.net/1/messages.json", data=payload, timeout=5)
    return r.status_code == 200
```

Usage from substrate code:

```python
from airlock_coordination.notify.pushover import push

# Untangler reject in Control chamber
push(
    title="Untangler · Closure rejected (Control)",
    message=f"||C||_F = {norm:.4f} > d = 0.40 in chamber=control. Action blocked.",
    severity="S3",
    sound="classical",
    url="https://otto-cockpit.fly.dev/audit/closures/<id>",
    url_title="Inspect closure",
)
```

---

## Resend wiring (email receipts)

```
RESEND_API_KEY=re_<operator's Resend key>
```

Reference helper (`airlock-coordination/notify/resend.py`):

```python
import os, resend

resend.api_key = os.environ["RESEND_API_KEY"]

def email_receipt(subject: str, html_body: str, to_address: str | None = None) -> bool:
    """Send an email receipt. Defaults to operator's notification address."""
    to = to_address or os.environ.get("OPERATOR_NOTIFY_EMAIL", "admin@airlocklabs.io")
    params = {
        "from": "substrate@airlocklabs.io",
        "to": [to],
        "subject": subject,
        "html": html_body,
    }
    r = resend.Emails.send(params)
    return bool(r.get("id"))
```

When the Noble Mobile eSIM is provisioned, the operator's new number can route to a number-bound email (e.g., `<number>@text.gateway`) and receive Resend emails as SMS. Until then, Resend goes to the operator's standing email address.

---

## Noble Mobile MVP path (operator-side, $10/3mo)

Operator's stated budget: $10 for 3 months → Noble Mobile MVP. This is operator-side provisioning, not substrate work. The substrate is ready to consume the new number in two ways:

1. **As a Skeet Pro identity binding:** the new Noble carrier-issued number gets bound to the operator's DID at the Pro tier. Substrate verifies via number-lookup that the line is carrier-issued, not VoIP.
2. **As a Resend SMS-via-email gateway:** if Noble (T-Mobile-backed) supports `<number>@tmomail.net` (or equivalent), Resend can fire emails that arrive as SMS on the operator's phone. Lower-cost than a dedicated SMS provider for MVP-tier volumes.

The substrate does not need to wait on Noble to wire Pushover. Pushover works today on any iOS/Android device with the app installed; it does not require a phone number. **Pushover should be wired this week regardless of Noble status.** Resend-as-SMS waits on Noble.

---

## What fires Pushover today (recommended starter set)

For the immediate operator-visibility need (orchestration awareness during NeurIPS push + 422 fix + Addie TG window), the minimum starter set:

1. **422 gate at otto-cockpit fails** — S4 siren
2. **Fly deploy succeeds (otto-cockpit)** — S2 quiet
3. **Railway emission picks up Fly handoff** — S2 quiet
4. **Untangler rejects closure (Control chamber)** — S3 classical
5. **Named-canon contact messages (Addie, Harper, Eric, Sarim)** — S3 classical
6. **Apex-style payment anomaly** — S3 tugboat
7. **NeurIPS deadline < 24h** — S3 classical (repeat every 6h)
8. **First Light fires (first operator login)** — S3 bugle

Eight events. Operator can adjust the severity ladder per their tolerance.

---

## What this spec is NOT

- **Not a notification implementation.** This documents the routing rules. The actual `push()` and `email_receipt()` helpers need to be wired into the substrate at relevant code paths (Fly deploy hooks, Untangler verdict path, OctoConductor dispatch, etc.).
- **Not a replacement for OTTO Vault.** Every event still logs to OTTO Vault for forensic recall. Pushover and Resend are *operator-awareness* channels; OTTO Vault is the *substrate audit ledger*.
- **Not a queue system.** No retry logic beyond Pushover's emergency-priority retry-every-60s. If finer-grained queue management is needed, that's a future task.

---

## Operator-side actions (in priority order)

1. **Today:** install Pushover app on phone if not already; verify `PUSHOVER_USER_KEY` and `PUSHOVER_APP_TOKEN` work via a manual `curl` test against `api.pushover.net`. **5 minutes.**
2. **Today:** decide Noble Mobile signup. $10/3mo trial unlocks Skeet Pro identity binding + Resend-as-SMS path. Operator-side; no AI dependency. **15 min if signup is smooth.**
3. **This week:** wire the 8 starter Pushover triggers into the substrate code paths (Fly hooks, Untangler verdict path, named-canon contact watchers).
4. **Next week:** extend Resend receipts to the S1 events (deploy completions, Threema subscriber milestones, etc.).
5. **Once Noble is provisioned:** add the operator's new number as Skeet Pro identity; wire Resend-as-SMS via T-Mobile email gateway.

---

## Composition with substrate canon

| Canon | Composition |
|---|---|
| `rule-442-deploy-chain.md` | Rule 442 violation (S4) is one of the loudest Pushover events. Bypass detection MUST notify. |
| `substrate-tier-communication-protocol.md` | Pushover is the operator-feedback channel ABOVE the 4-tier stack. The operator conducts; the tiers fire events; Pushover surfaces them. |
| `project_octoagent_is_untangler_spacebar_acode_bcode_pairing.md` | Untangler verdicts (S2/S3) are the highest-frequency S2+ events. Volume tuning matters. |
| `project_otto_cockpit_fly_dev_422_first_light_unlock.md` | 422 gate failures are S4. First Light fires S3. |
| `project_addie_ann_tbe_tobi_brent_miami_dealflow.md` | Addie TG message → S3 (operator should respond same-day). |
| `project_paypal_oss_substrate_client_sdk_messenger_config.md` | Apex-style payment anomalies → S3 (catches the bug-class that caught us last time). |

---

**Bottom line:** Pushover for awareness, Resend for receipts, OTTO Vault for audit. The severity ladder governs which channel fires. Start with the 8-event starter set; tune from there. Pushover wires today; Resend-as-SMS waits on Noble Mobile if operator pulls the trigger on the $10/3mo MVP.
