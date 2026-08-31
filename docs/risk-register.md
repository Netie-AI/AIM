# Risk register

Ordered by what actually kills the project.

---

## R-01 — Compounding model error over the rollout horizon

*Named in the original spec as the likely failure point. Correctly.*

**The failure:** by step five the rollout is fiction rendered in a very convincing audit
trail. A false plan with a clean audit log is worse than no plan, because the log buys
trust the plan has not earned.

**Mitigations**
1. **Decision-relevance horizon.** Stop when the top-1 ranking flips under bootstrap
   resampling of the learned component. Replaces the unanswerable "how far can we see" with
   the measurable "how far until this stops being the same decision."
2. **Error-budget horizon.** Stop when cumulative predicted variance makes sibling branches
   statistically indistinguishable.
3. **Residual formulation.** The learned half predicts a residual against the symbolic
   baseline, so accounting identities stay exact regardless of model error and the error bar
   means something interpretable.
4. **`speculative` marking.** Nodes past the calibrated horizon are flagged in the trace and
   may not be cited as justification for a commit. The audit trail renders its own
   uncertainty — which is the direct answer to the stated fear.
5. **Backtest gate (P4).** If multi-step does not beat 1-step on realized history, we ship
   1-step and say so.

**Residual risk:** the horizon is itself estimated from history, so a regime change
invalidates it silently. Partly covered by drift monitoring (R-06); not fully solved.
Accepted, and stated.

---

## R-02 — Silent fallback to the nearest expressible action

*Named in the original spec. This is the one that lies.*

**The failure:** if the action schema is wrong or incomplete, the system does not fail
loudly. It picks the nearest expressible action and produces a clean trace for it.

**Mitigations**
1. **`ABSTAIN` always legal, and a margin requirement.** A proposed action must beat
   abstention by a stated margin, not by epsilon. "No good option" becomes a normal,
   recordable outcome rather than an edge case that gets rounded away.
2. **Gap telemetry (highest value per unit of effort in the whole register).** Every
   proposal that fails validation is logged with its state context. A *cluster* of
   rejections in one region of state space is the signal that the catalog is incomplete.
   This converts a silent failure into a monitored one — which is most of the fix.
3. **Fit score, separate from margin.** Log how well the chosen action fits the situation,
   independently of how it scored. **Low fit + high margin is the exact signature of a
   nearest-expressible fallback**, and it is alertable.
4. **Adversarial catalog tests.** For each action, states where it must *not* be chosen.
5. **P1 coverage gate.** ≥95% of real human decisions expressible without distortion, with
   the residual characterised rather than unknown.

**Residual risk:** an action that is wrong in a way we have not imagined will not appear in
the gap log, because it validated fine. Detection then depends on outcome monitoring, which
is slow. Accepted.

---

## R-03 — The objective function encodes the wrong values, rigorously

*Not named in the original spec. Argued in `docs/architecture.md` A2.*

**The failure:** "score by cost and tail risk" hides every value judgment in the system
inside a function nobody reviews. A scalarized weighted sum asserts tradeoffs without
argument and buries them in config. It looks far more rigorous than a prompt while being
just as arbitrary — which makes it worse, because it is trusted more.

**Mitigations**
1. Objective is a **versioned, reviewed artifact** with the same governance status as the
   invariant set.
2. **Hard constraints + lexicographic preference** over scalarization wherever the domain
   permits.
3. Every decision report states **the marginal tradeoff it made** — what was given up, what
   was bought.
4. **Tail risk bounded symbolically** where contract terms, position limits, or regulatory
   caps permit. Tails are where learned data is thinnest and where the design leans hardest;
   a loose symbolic bound fails safe, a wrong learned estimate does not.

---

## R-04 — The invariant set becomes the new prompt-tweak

**The failure:** if changing policy means editing 400 invariants with no test coverage and
no review process, the fragility has been moved, not removed. The spec's controllability
claim — "policy changes are edits to the invariant set, not prompt tweaks you hope stick" —
is only true if invariant edits are *verifiable*. Otherwise they are prompt tweaks with a
type system.

**Mitigations**
1. Invariants are code, with their own test suite including must-reject adversarial states.
2. Every invariant change runs the replay harness: which historical decisions would have
   changed, and were those changes intended? This is the review artifact.
3. Invariants are named, dated, and attributed to a source authority (a regulation, a
   contract clause, a policy decision). An invariant with no cited source gets deleted.

---

## R-05 — Determinism breaks quietly

**The failure:** "identical input + identical seed = identical trace" is easy to assert and
breaks by default — floating-point nondeterminism across hardware, hash iteration order,
wall-clock reads, unpinned library versions, GPU kernel selection.

**Mitigations**
1. Model artifacts pinned by content hash and recorded in the trace.
2. Decimal arithmetic throughout the symbolic half; float banned at the type level for money.
3. Clock and randomness injected, never read ambiently. Seed recorded in the trace.
4. **Replay CI on every commit**, cross-machine, byte-for-byte diff. This is the P2 gate and
   the actual proof of the precision claim.

---

## R-06 — Symbolic/learned boundary drift

**The failure:** the learned half implicitly re-learns rules that are also encoded
symbolically, then quietly contradicts them. Nothing errors; the numbers just diverge.

**Mitigations**
1. Residual formulation bounds what the learned half can even disagree about.
2. Boundary-drift test in CI: on states where a written rule fully determines the outcome,
   the residual must be ≈0.
3. Drift monitoring on the residual distribution over time; a shifting residual on
   rule-determined states is a defect, not a retraining prompt.

---

## R-07 — Cold start

**The failure:** no learned model until there is a ledger; no ledger until the system runs.
A plan that only works at full build never gets built.

**Mitigation:** P0–P2 are useful with a purely symbolic transition model and wide
abstention. P1 alone catches invariant violations in decisions being made today, which is
real value with zero learned components. Autonomy narrows abstention only as calibration
earns it.

---

## R-08 — Escalation volume exceeds human capacity

**The failure:** calibrated abstention routes so much to humans that the queue is rubber-
stamped. The control becomes a formality, and the audit trail records human approval that
carries no information.

**Mitigations**
1. Escalate on `epistemic × outcome-spread`, not raw uncertainty — high uncertainty where
   all branches score alike does not need a human.
2. Escalation volume is a P3 gate, measured against actual capacity.
3. Monitor human review *time per item* as a health metric. Falling review time is the
   early signal that the control is decaying into a formality.

---

## R-09 — Provenance decay

**The failure:** provenance survives ingestion but gets stripped by an aggregation, a
summarisation, or a schema migration three phases later. Then a record's confidence is
uninterpretable and replay is unverifiable.

**Mitigation:** provenance is structurally required on every record — including derived
ones, which cite their inputs. Enforced at the write boundary. A derived record with no
parent citation is rejected at the type level, not in review.
