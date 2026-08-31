# AIM — a ledger-native planner

The premise: a transformer cannot be audited, because the artifact it produces is text,
and text is not a decision. The fix is to make **the plan the primary artifact** and text a
*rendering* of it. The transformer stays in the system, demoted from decision-maker to
proposal generator.

This repository holds the design, the build plan, and the schemas. Code follows the phases
in `docs/build-plan.md`.

| Document | What it is |
|---|---|
| `docs/architecture.md` | The system, stated precisely enough to build. Includes three amendments to the original spec. |
| `docs/build-plan.md` | Six phases, each shipping something usable, each with a numeric gate before the next begins. |
| `docs/risk-register.md` | The failure modes that kill this project, and what we do about each. |
| `schemas/` | Concrete JSON Schema for the three types the whole system rests on: record, action, decision trace. |

## The one-paragraph version

State is an append-only ledger of typed records, each with provenance and a confidence.
The model proposes typed actions from a closed catalog; anything it cannot express it must
refuse rather than approximate. A hybrid transition model — symbolic where rules are
written down, learned where they are not — is rolled forward under multiple hypotheses,
and branches are scored on cost and tail risk. Hard invariants are checked outside the
network on every branch; a violation prunes it rather than being argued with. The neural
component never has authority to commit. Output is a DAG of state → action → predicted
successor → evidence → checks passed, replayable offline and diffable across versions.

## What makes it auditable

Not the typing. Typing is necessary and not sufficient — a typed plan nobody can
independently re-derive is exactly as opaque as prose. Auditability comes from two
things the types make *possible*: the shield (invariants checked by code that is not the
model) and the replay harness (every historical decision re-executable, byte-identical, on
demand). Build those two or the rest is decoration.
