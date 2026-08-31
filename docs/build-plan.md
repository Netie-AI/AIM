# Build plan

Six phases. Each ships something usable on its own. Each has a **numeric gate** that must be
met before the next begins — the gates are the point, because they are what stop the system
from acquiring authority it has not earned.

The organising principle: **autonomy is earned per action class, by demonstrated calibration
on replay.** It is never granted globally at launch.

---

## P0 — Ledger. Shadow mode, no actions.

Build the append-only bitemporal store, the record type system, the write boundary that
rejects any record lacking provenance or confidence, and the double-entry balance check.
Ingest real data. Take zero actions.

**Deliverables**
- Record schema + migration discipline (`schemas/record.schema.json`)
- Append-only store; corrections as superseding records, never edits
- Decimal money type with per-currency rounding rules; float banned at the type level
- Balance identity as a total function over the ledger, checked continuously
- Bitemporal query: "what did state look like as of time T, using only what we knew then"

**Gate to P1**
- Balance identity holds on 100% of ingested history
- As-of reconstruction matches the source system's own historical statements to the cent
- 0 records in the store lacking provenance or confidence
- Ingest lag and correction rate measured and stable

*Why this gate:* if you cannot reconstruct the past exactly, replay is fiction and every
downstream guarantee is unfounded.

---

## P1 — Action catalog + shield + abstention. Humans decide; the system checks.

Define the closed catalog. Build the shield as an independent implementation. Build
`ABSTAIN` as a first-class always-legal action. The system observes human decisions,
compiles them to typed actions, and runs the invariants — but humans still decide.

This phase is where the label corpus comes from, and it delivers real value immediately:
it catches invariant violations in decisions being made today.

**Deliverables**
- Action schema + preconditions (`schemas/action.schema.json`)
- Shield: total, versioned, independently implemented, with an adversarial test suite
- Compiler from action → balanced double-entry set (an action that will not compile cannot exist)
- Schema-gap telemetry: every human decision that cannot be expressed in the catalog is
  logged as a gap with its state context — see R-02

**Gate to P2**
- Shield is total: 100% of states return an explicit pass/fail on every invariant
- Adversarial suite: 100% of must-reject states rejected
- Catalog coverage ≥ 95% of observed human decisions expressible without distortion
- Gap log reviewed; the residual ≤5% is characterised, not unknown

*Why this gate:* below ~95% coverage the catalog is not closed, it is lossy, and the system
will start approximating. Coverage is measured against real decisions, not imagined ones.

---

## P2 — Symbolic transition model + 1-step lookahead. Recommendations.

Add the symbolic transition function and single-step search. The system now recommends,
with a full decision trace. Humans still commit. No learned component yet — abstention is
wide by design.

**Deliverables**
- Symbolic transition model as a pure function
- Decision trace DAG (`schemas/decision-trace.schema.json`): state, action, predicted
  successor, evidence, checks passed, objective tradeoff made
- Objective v1 as a versioned artifact: hard constraints + lexicographic preference
- Renderer: trace → prose, one-directional, regenerable
- **Replay harness**: re-execute any historical decision, diff traces byte-for-byte

**Gate to P3**
- Replay determinism: 1000 historical decisions re-execute byte-identical, twice, on two
  different machines
- Recommendation agreement with human decisions ≥ target on the covered subset
- Every disagreement is explainable — traced to a rule, a missing record, or a genuine
  human error. **Zero unexplained disagreements.**

*Why this gate:* replay determinism across machines is the single test that makes the
"precise and predictable" claim true rather than asserted. Do not advance without it.

---

## P3 — Learned residual + calibration harness. Still 1-step.

Add the learned half, predicting a residual against the symbolic baseline. Build the
calibration machinery before using its output for anything.

**Deliverables**
- Residual model with ensemble, trained on the ledger's own history
- Aleatoric head (quantile/heteroscedastic) and epistemic signal (ensemble disagreement +
  coverage check), kept separate and never summed
- Conformal prediction wrapper for distribution-free coverage guarantees
- Boundary-drift test: residual ≈ 0 on rule-determined states, enforced in CI
- Escalation rule: `epistemic × outcome-spread`, tuned so the human queue is small enough
  to actually be read

**Gate to P4**
- Conformal coverage within tolerance of nominal on held-out replay (e.g. 90% interval
  covers 88–92%)
- Boundary-drift test passes: no learned contradiction of a written rule
- Escalated volume within human review capacity, and reviewers agree the escalations were
  worth their time

*Why this gate:* an uncalibrated uncertainty estimate is worse than none, because it is
trusted. Calibration is verified before it is used, not after.

---

## P4 — Multi-step search with calibrated horizon.

Now, and only now, roll forward multiple steps.

**Deliverables**
- Rollout search under multiple hypotheses; shield prunes on every branch
- Error-budget stop and decision-relevance stop (bootstrap the residual; stop when top-1 flips)
- `speculative` flag on every node past the calibrated horizon; speculative nodes may not be
  cited as justification for a commit
- Tail risk bounded symbolically wherever contract/limit/regulatory structure permits;
  learned tail estimates used only where no symbolic bound exists, and marked

**Gate to P5**
- Measured calibrated horizon reported per action class (it will differ, and some classes
  will honestly be 1)
- Backtest: multi-step recommendations beat 1-step on realized outcomes, on held-out history
- No committed action in the backtest justified by a speculative node

*Why this gate:* if multi-step does not beat 1-step on realized history, the extra depth is
producing confident fiction. Ship 1-step and say so.

---

## P5 — Bounded autonomy.

Auto-commit inside a proven envelope: `(action class × magnitude × calibration status)`.
Everything outside the envelope routes to a human. The envelope widens only on evidence.

**Deliverables**
- Envelope definition as a versioned artifact
- Per-action-class autonomy grants, each traceable to the replay evidence that earned it
- Kill switch that reverts to P4 recommend-only, per class, in one action
- Continuous monitoring: calibration drift auto-narrows the envelope without waiting for a human

**Gate to widening**
- Per class: N committed decisions with realized outcomes inside predicted intervals at the
  stated coverage rate
- Zero shield violations reaching commit — ever, in any class. This one is not a threshold.

---

## Cross-cutting, built in every phase

- **Replay CI**: historical decisions re-executed on every commit; trace diffs fail the build
- **Gap telemetry**: rejected proposals clustered by state region, reviewed on a schedule
- **Objective changes**: reviewed like invariant changes, never tuned silently
- **Provenance**: never stripped, never summarised away, survives every transformation

## What is explicitly not in scope

- The transformer choosing anything. It extracts, proposes, renders. There is no phase where
  that changes.
- Prompt-based policy control. Policy lives in the invariant set, the action catalog, and the
  objective. If a behaviour can only be changed by editing a prompt, it is a defect.
