# Validation plan

The plan's exposure is not fabrication risk. It is that three algorithmic claims
are unverified and **all three are measurable on commodity hardware today**, for
roughly the cost of a few GPU-weeks, before anything is committed to silicon.

Order is by ratio of what-it-kills to what-it-costs. Each stage has a kill
criterion written before the experiment, not after.

---

## Done

### E1 — Are exact online gradients real? `experiments/e1_rtrl_equivalence.py`

**Kill criterion**: online gradients differ from BPTT, or online memory grows with T.

**Result: passed.** Worst-case relative error 3.18e-15 across widths 16–512 and
T to 1024. Live memory flat in T; 12,484x below BPTT at T = 1M. Truncated BPTT at
k=1 loses 47.7% of the gradient and at k=64 still loses 7.4%; the online path
loses nothing.

The temporal half of the architecture is settled. It was never the risk.

### E2 — How wide must the reward bus be? `experiments/e2_credit_bus.py`

**Kill criterion**: a scalar-broadcast rule scales as `O(n)` *and* extra lanes do
not help — in which case the interconnect saving is fake and the chip has no case.

**Result: v1's single wire killed; the bus survives.** One global scalar costs
119x an exact gradient at width 64 and 157x at 128, with a log-log slope of 1.99
against exact's 1.20. But the governing variable is fan-out per lane, not
parameter count: `slowdown ≈ (units per lane)^0.80`, from 5.3x at 1 unit/lane to
157x at 128. Lane count is an architectural free parameter (E4.3: 16 KB/step at
4096 lanes against 26.08 GB/step for an allreduce).

### E3 — How big is "O(1)"? `experiments/e3_state_capacity.py`

**Kill criterion**: state capacity sublinear in state size, or cost per fact worse
than a KV cache.

**Result: passed, better than expected.** Capacity linear in key dimension.
Cost per fact flat at ~54 B and independent of width, against `2(d_k+d_v)` B for a
KV cache which worsens as models widen — 3.4x to 12.1x in the recurrent state's
favour, at 90% recall rather than exact. The delta rule saturates at exactly 1.00
pairs per unit; plain accumulation reaches 2.4 by trading exactness for graceful
degradation.

### E4 — Where does the money go? `experiments/e4_system_budget.py`

Arithmetic over public specs, not measurement. Produced the memory table, the
1.7-million-fold interconnect ratio, the SSD tier analysis, and — the useful
output — the on-die trace budget that converts the chip concept into a product
definition (~10⁸ adapted parameters per reticle-scale die).

---

## Next, in order

### E5 — Do segmented options beat tokens? *(1 GPU, ~1 week)*

The only remaining v1 claim with **no** evidence behind it, which is why it is
next despite being the cheapest to run.

Measure gradient SNR per unit compute for a policy over syntactically segmented
spans against a token-level baseline, on a task with verifiable sparse reward
(arithmetic with a checkable answer, or unit-test-passing code).

**Kill criterion**: SNR-per-compute improvement below 3x. Below that, options are
complexity without payment and L4 should be deleted rather than defended. v1's
"10 to 50x" is not the bar; it was never derived.

### E6 — Does a bounded state hold real language? *(1–2 GPUs, ~2 weeks)*

E3 measured capacity on synthetic recall. Whether real tasks fit in 4–16 MB is
the open question, and it decides whether L0's hybrid window can stay short.

Train a ~1B hybrid (element-wise recurrence + 512-token exact window) and sweep
state size against a matched full-attention baseline on long-context retrieval
and multi-step reasoning.

**Kill criterion**: matching the baseline requires either a state above ~64 MB or
an exact window above ~8k tokens. Either outcome means the KV cache was
reintroduced under another name and the inference argument is gone.

### E7 — Does the lane law hold at scale? *(2–4 GPUs, ~3 weeks)*

E2's exponent comes from widths 16–128. Extrapolating it to 10⁶ units is the
single largest inference in the whole design, and it is the one the chip's
existence depends on.

Re-run the E2 sweep at widths 10³–10⁵ with the L1/L3 rule as actually specified —
local backprop within a block, lanes between blocks — rather than pure node
perturbation.

**Kill criterion**: units-per-lane exponent above 1.0, or ≤32 units per lane
proving insufficient. Either forces lane counts that reintroduce a gradient
fabric, which removes the chip's reason to exist.

### E8 — Full-stack integration on GPUs *(8 GPUs, ~2 months)*

Everything above, assembled at ~1B parameters: hybrid state, blocks with local
backprop, traces, multi-lane credit, segmented options, imagined rollouts,
NVMe replay. Emulate low precision and the bus in software.

**Kill criterion**: cannot reach a supervised baseline's quality within 10x its
sample budget on the target tasks. This is the composition risk — every stage
above can pass alone and the assembly still fail.

### E9 — Silicon feasibility, in parallel with E8

Endurance measurement on candidate NVM at the consolidation interval; SRAM array
area and leakage at the trace budget; bus fan-out timing closure. No tapeout
commitment until E7 and E8 both pass.

---

## What each stage is worth

| stage | cost | kills, if it fails |
|---|---|---|
| E5 | ~1 GPU-week | L4 only; the rest of the design stands |
| E6 | ~2 GPU-weeks | the inference argument (bounded state) |
| E7 | ~3 GPU-weeks | **the chip** — no bus, no reason to build silicon |
| E8 | ~16 GPU-weeks | the architecture as an integrated system |
| E9 | fab engagement | specific implementation choices, not the concept |

Roughly six GPU-weeks separate today from knowing whether the chip has a case.
That is the point of sequencing E7 before E8 despite E8 being the more
interesting experiment: **E7 is cheaper and kills more.**

## Standing rules

- No tapeout before E7 and E8 pass. The tapeout is the only irreversible
  expenditure here and every algorithmic risk is cheaper to retire first.
- Kill criteria are written before the run and not renegotiated after it.
- Every claim that reaches a slide cites a script in `experiments/`. The reason
  v1 needed this review is that its strongest item (A3) was stated wrongly and
  its headline hardware item (B2) was wrong, and neither had a number attached.
