# Architecture

## 0. What the transformer is for

Three jobs, all non-authoritative:

1. **Extraction.** Turn unstructured input (documents, email, chat, PDFs) into candidate
   typed records. Every extraction carries a provenance pointer to the exact source span
   and a confidence. An extraction is a *proposal*, never a fact.
2. **Proposal.** Emit candidate actions from the closed catalog under constrained decoding.
   It suggests what to consider; it does not choose.
3. **Rendering.** Turn a committed decision trace back into prose for humans.

Rendering is strictly one-directional and regenerable. The prose is a *view*. If prose and
trace ever disagree, the trace wins by construction — because the prose is derived from the
trace on read and never stored as the source of truth.

The transformer is never in the commit path. Every place it touches the system, its output
passes through a validator that can reject it.

---

## 1. Typed state, not context

An append-only store of typed records. Not tokens in a window.

Every record carries:

- `id`, `type`, `occurred_at`, `recorded_at` (bitemporal — when it happened vs. when we
  learned it; you need both to replay a decision as it looked at the time)
- `provenance`: source system, source document, extraction span, extractor version
- `confidence`: `1.0` only for records with a symbolic origin (a settled transaction, a
  signed contract term). Anything model-extracted is `< 1.0` and is marked as such forever.

**Nothing enters state without provenance and confidence.** This is enforced at the write
boundary, not by convention.

**Corrections are records, not edits.** Append-only means a wrong record is superseded by a
correcting record that points at it. The original stays. Otherwise replay is a lie.

### Double-entry is an invariant engine, not a metaphor

The reason to be double-entry is not aesthetic. It makes a whole class of error
*unrepresentable* rather than detectable-after-the-fact: value cannot appear or vanish,
because every action compiles to a balanced set of entries and the balance identity is
checked as a total function over the ledger. An action that cannot be expressed as balanced
entries cannot be committed at all.

**Money is decimal, never float.** Fixed-point arithmetic with explicit rounding rules,
declared per-currency. A float in the ledger breaks both the balance identity and
determinism.

---

## 2. Closed action schema

The model emits typed actions with bounded arity and validated arguments, from a fixed
catalog: `transfer`, `hold`, `flag`, `request_document`, `escalate`, and one more that
matters more than all of them (see §6).

Anything unexpressible in the schema is **refused, not approximated.**

Enforcement is at three layers, because any one of them alone fails:

1. **Constrained decoding** — the grammar makes most malformed output unrepresentable.
2. **Schema validation** — types, ranges, referential integrity against the ledger.
3. **Precondition check** — the action must be *applicable* in this state, not merely
   well-typed. (A `transfer` referencing a closed account is valid JSON and an invalid act.)

---

## 3. Hybrid transition model

Symbolic where rules exist and are written down: accounting identities, regulatory limits,
contract terms, settlement mechanics. These are exact, and they are the majority of what
actually determines outcomes in this domain.

Learned where rules are not written down: customer behavior, drawdown, market response.
These carry an explicit error bar or they do not ship.

The split is a hard architectural boundary, not a blend:

- The symbolic half is a pure function. Same input, same output, no seed needed.
- The learned half predicts a **residual** against the symbolic baseline, not the raw
  quantity. This matters: it bounds what the learned component can be wrong about, keeps
  the accounting identities exact regardless of model error, and makes the error bar
  interpretable (it is the size of what the rules don't explain).

### Boundary drift is a real failure and needs a test

The learned half will, given enough data, implicitly re-learn rules that are also encoded
symbolically — and then quietly contradict them. Guard: a test suite of states where a
written rule fully determines the outcome; the learned residual must be ~0 there. A
drifting residual on a rule-determined state is a defect, and it is caught in CI.

---

## 4. Search over rollouts

Foresight comes from rolling the transition model forward under multiple hypotheses and
scoring outcomes, not from the language model's implicit sense of what comes next.

The proposer suggests branches. The search expands them. The scorer ranks them. The shield
prunes them. Only the proposer is stochastic, and it is downstream-filtered by three
deterministic components.

### Horizon is calibrated, not configured

Model error compounds. A fixed depth of 5 is a guess. Instead, two stopping rules, both
measured:

- **Error-budget stop.** Track per-step predictive error against realized history. Stop
  expanding when cumulative predicted variance crosses the threshold where the branch's
  score is no longer distinguishable from its siblings'.
- **Decision-relevance stop.** You do not need an accurate world model at step 5. You need
  the *ranking between the top actions* to be stable. Bootstrap-resample the learned
  component; if the top-1 action flips, you are past your real horizon. Stop there.

The second rule is the important one. It replaces "how far can we see" — unanswerable —
with "how far do we need to see for this to still be the same decision," which is
measurable per decision.

### Speculative depth is marked in the trace

The spec's own stated fear is a rollout that is fiction rendered in a convincing audit
trail. The mitigation is that **the trail renders its own uncertainty**: every node past the
calibrated horizon is flagged `speculative`, and a speculative node may not be cited as
*justification* for a committed action — only as context. A reader can see exactly where
fiction begins, because the artifact says so.

### The ledger is the calibration corpus

Append-only bitemporal state means every past decision point has a ground-truth successor
already recorded. Backtesting is not a separate data-collection project — it falls out of
the design. This is the single property that makes the whole thing tractable: **the audit
log is the training and calibration set.**

---

## 5. Shield before commit

Hard invariants checked outside the network, on every candidate branch, before commit.
A violation prunes the branch. It does not get argued with, re-prompted, or overridden.

Properties the shield must have:

- **Independent implementation.** Written against the spec, not derived from the model or
  from the scorer. If the shield shares code with the thing it checks, it is not a check.
- **Total.** Every invariant returns pass/fail on every state. No exceptions, no "not
  applicable" that silently passes.
- **Versioned and tested.** Invariants are code with their own test suite — including
  adversarial states that *must* be rejected. An invariant set with no tests is a prompt
  with better syntax highlighting; see `docs/risk-register.md` R-04.

---

## 6. Calibrated abstention (the sixth requirement)

Without this, every other guarantee is decoration.

Uncertainty is split into two kinds that must never be summed:

- **Aleatoric** — the world is noisy. From the learned model's predictive distribution
  (heteroscedastic head or quantile regression). Irreducible. Plan under it.
- **Epistemic** — we have not seen this before. From ensemble disagreement plus a
  density/coverage check on the state features. Reducible by data. **Route to a human.**

Calibration is verified by **conformal prediction** over the replay corpus, because it gives
distribution-free coverage guarantees. That is the auditable form of the claim: not "the
model is confident" but "this interval has covered the truth 90% of the time across N
replayed decisions," a sentence with a number in it that anyone can re-derive.

### `ABSTAIN` is a first-class action, always legal

The catalog always contains `ABSTAIN(reason, evidence)`. It is applicable in every state.
A proposed action must beat abstention **by a stated margin**, not by epsilon. This closes
the catalog under refusal and converts "no good option" from an edge case into a normal,
recordable outcome.

### Escalate on decision impact, not raw uncertainty

High uncertainty where every branch scores the same does not need a human — the decision is
insensitive to what we don't know. The escalation trigger is `epistemic uncertainty ×
spread in outcome across top branches`. This keeps the human queue small enough that humans
actually read it, which is the difference between a control and a formality.

---

## Three amendments to the original spec

### A1. Auditability comes from the shield and the replay harness, not from typing

Typing is necessary and not sufficient. A typed plan that no one can independently
re-derive is exactly as opaque as prose — it just looks rigorous. What makes a decision
auditable is that a second party can (a) re-run the invariants themselves and (b) replay the
decision offline and get the identical trace. Build those two first. They are the load-bearing
claims; the type system is the scaffolding that makes them expressible.

Determinism is an engineering discipline, not an aspiration. It breaks by default via
floating-point nondeterminism across hardware, hash iteration order, wall-clock reads, and
unpinned library versions. It requires: model artifacts pinned by content hash, decimal
arithmetic throughout the symbolic half, seed recorded in every trace, clock injected rather
than read, and a CI job that re-executes historical decisions and diffs traces byte-for-byte.
That CI job is the actual proof of the "precise and predictable" requirement. Without it the
claim is untested.

### A2. The objective function is a third weak point, and the least defended

The spec names two: compounding model error, and silent fallback to the nearest expressible
action. Both are real. There is a third.

Search scores outcomes "by cost and tail risk." In AlphaZero the reward is unambiguous —
you won or you lost. Here the objective is a **policy choice**, and encoding it wrong is
exactly as bad as a bad prompt, while looking far more rigorous. All the value judgments in
this system — how much liquidity risk is worth how much yield, whose loss counts — hide in
that function. A scalarized weighted sum is the worst form, because the weights are
tradeoffs asserted without argument and buried in a config file.

Mitigations:

- The objective is a **versioned, reviewed artifact with the same governance status as the
  invariant set.** Changing a weight is a reviewed change, not a tuning session.
- Prefer **hard constraints plus lexicographic preference** over scalarization wherever the
  domain permits. "Never breach the limit; among the survivors, minimize cost; among ties,
  prefer reversibility" is auditable. `0.3·risk + 0.7·cost` is not.
- Every decision report states **the marginal tradeoff it actually made** — what was given
  up and what was bought. That makes the objective visible per-decision rather than only in
  the config.

Related, and worth stating plainly: **tail risk from a learned model is the least trustworthy
number in the system**, because tails are precisely where the data is thinnest — and the
design leans on it hardest. Wherever possible, bound tails *symbolically* (worst case under
contract terms, position limits, regulatory caps) rather than estimating them statistically.
A symbolic bound that is loose beats a learned estimate that is wrong, because the loose
bound fails safe.

### A3. Cold start: the system must be useful before the learned model exists

There is no learned transition model until there is a ledger, and no ledger until the system
runs. The plan resolves this by making Phases 0–2 valuable with a **purely symbolic**
transition model and correspondingly wide abstention. The learned component is added in
Phase 3, once its calibration corpus exists. Autonomy narrows abstention as calibration
improves — it is earned, per action class, never granted globally at launch.
