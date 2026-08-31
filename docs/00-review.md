# Review of the original plan

Verdicts are one of **holds**, **holds, restated**, **overstated**, or **breaks**.
Every number cited is produced by a script in `experiments/`; nothing here is
asserted from memory that could have been measured instead.

---

## The framing error, first

> The core bet: kill backprop-through-time and the KV cache in the same move.

These are not two things that must be killed together as a gamble. They are one
property, and it is a theorem rather than a bet.

If the recurrent state update is element-wise — `h_t = φ(a ⊙ h_{t-1} + W x_t + b)`
with `a` a vector, not a matrix — then the recurrent Jacobian `∂h_t/∂h_{t-1}` is
diagonal. Unit `i`'s history never mixes into unit `j`'s. Two consequences fall
out of that single fact:

- **Inference**: the state is a fixed-size vector. Nothing accumulates with
  sequence length, so there is no KV cache.
- **Training**: real-time recurrent learning, which is `O(n³)` memory for a dense
  recurrence and therefore useless, collapses to `O(params)`. The sensitivity
  tensor `∂h_t/∂θ` has the same shape as `θ`.

E1 verifies the second point: online forward-only gradients match full BPTT to a
worst-case relative error of **3.18e-15** across widths from 16 to 512 and
sequence lengths to 1024. That is floating-point noise.

So the plan's riskiest-sounding item is the one already established. Eligibility
traces here are not a cheap approximation to BPTT — for this recurrence they
*are* the gradient. The per-cell trace register in the hardware proposal stores a
mathematically exact quantity. That is a stronger claim than the plan makes, and
it should be made.

The consequence is that the real risk sits elsewhere, and the plan does not
currently point at it. It is item 3's *other* half (§A3) and the reward bus (§B2).

---

## A. Architecture

### A1. Recurrent latent state, not attention — **holds, restated**

Correct in scaling. But `O(1)` is a statement about growth, not about size, and
the plan never names the constant. E3 measures it on associative recall:

- Capacity is **linear in state width** for both write rules tested. Outer-product
  (linear-attention) writes hold ~2.4 pairs per unit of key dimension; the delta
  rule holds almost exactly 1.00 per unit — it is solving least squares, so once
  keys stop being linearly independent, new writes *overwrite* old ones rather
  than adding crosstalk.
- Cost per fact in the recurrent state is **flat at ~54 B**, independent of key
  dimension, because size and capacity are both linear in it. A KV cache costs
  `2(d_k + d_v)` bytes per fact and gets *worse* as the model widens.

So the fixed state is not merely bounded, it is cheaper per remembered fact —
3.4x to 12.1x cheaper over the range tested. This is a better result than the plan
claims. The honest qualifier: that is at **90% recall, not exact recall**, and a
KV cache has no capacity cliff.

Restatement: *bounded state sized in megabytes for what must persist, plus a
short exact attention window for what must be recalled verbatim.* Do not claim
a pure recurrence; hybrid is what the recall evidence supports.

### A2. Options, not tokens — **overstated**

The options framework is real and long-established. Two problems with the claim
as written:

- The "10 to 50x" credit-horizon reduction is asserted, not derived, and horizon
  length is not what determines whether policy-gradient learning is tractable.
  Variance per unit compute is. Those are related but not the same quantity, and
  only the second is measurable.
- Option *discovery* is the open research problem, and the plan quietly assumes
  it away. Every part of the design that depends on options existing depends on
  an unsolved problem.

Evolution: do not discover options. **Segment** them at cheap syntactic
boundaries (sentence, clause, enumerated step, code statement) and bootstrap the
renderer from a pretrained decoder. This gives most of the variance reduction
with none of the discovery risk, and it is falsifiable now. Measure gradient SNR
per unit compute against a token-level baseline, and drop the horizon claim until
that measurement exists.

### A3. Eligibility traces instead of gradients — **breaks as written**

This item conflates two independent axes, and the plan is right about one and
wrong about the other.

Backpropagation runs through **time** and through **depth**. Traces eliminate the
temporal axis exactly (E1). They do nothing about depth: the classic trace update
still contains `∇_θ log π(a|s)`, a gradient of the policy with respect to its own
parameters *at the current step*, which is a backward pass through layers. "No
backward pass, no stored activations" is therefore false as stated.

This matters less than it sounds, and the reason is in E4.1. Once the temporal
axis is gone, per-step activation memory for a 7B-class model at batch 8 is
**256 KB**, against **16 GB** for the transformer — a 65,536x reduction. Depth-wise
backprop over one timestep is not a memory problem. It fits in on-die SRAM.

The alternative — eliminating depth backprop with a perturbation rule — is priced
in E2 and it is expensive. See §B2.

Evolution: **traces across time, exact local backprop within a shallow block,
scalars between blocks.** This keeps the memory win, which was never about depth,
and refuses a trade that E2 shows is bad.

### A4. Imagined rollouts for density — **holds**

Latent world models trained on imagined rollouts are established at the scale
claimed. The unlisted risk is model exploitation: the policy finds latent states
with high predicted reward that correspond to no real text. Standard mitigations
exist and should be written into the design rather than discovered later —
ensemble-disagreement penalties on the imagined reward, short imagination
horizons, periodic grounding against real transitions. No change to the claim,
but the risk belongs in the plan.

### A5. Intrinsic reward closes the loop — **overstated**

The framing (prediction as compression, compression gain as reward) is
defensible. The implied *mechanism* is not.

If you can compute `log p(x)`, you have an exact gradient of it for free.
Estimating that same quantity with a policy-gradient rule is a strictly
higher-variance estimator of something you could have had exactly. Nothing is
gained and a great deal of sample efficiency is lost. "Next-token prediction is
just RL where reward equals compression gain" is true as a statement about
objectives and false as a recipe for optimisers.

Evolution, which makes the slogan defensible instead of merely quotable:

- **Compression gain trains the world model**, with exact local gradients. It is
  a one-step prediction, so there is no BPTT and the cost stays low.
- **Policy gradients are reserved for genuinely non-differentiable reward.**

The honest version of "reinforcement is all you need" is *reinforcement where
differentiation is unavailable, and differentiation everywhere else.* That is a
narrower claim and a true one.

---

## B. The chip

### B1. Compute-in-memory tiles with in-place update — **holds, restated**

Compute-in-memory is real. The plan omits the constraint that decides whether
this specific use of it is buildable: **write endurance**. Non-volatile cells
tolerate on the order of 10⁶–10⁹ writes. A trace updated in place once per step,
at even 1 kHz, exhausts 10⁹ writes in under two weeks. In-place per-step update
of an NVM cell is not a design, it is a countdown.

Evolution — two-timescale memory, which is also what every mixed-precision
optimiser already does and what complementary-learning-systems accounts of
memory consolidation describe:

- **Fast**: traces and shadow accumulators in SRAM. Unlimited endurance.
- **Slow**: weights in NVM, consolidated every 10³–10⁵ updates.

E4.6 shows this constraint, not FLOPs, is what closes the design. See §C.

### B2. A single reward broadcast scalar — **breaks**

This is the headline of the chip and it is the part that does not survive. E2
measures the sample cost of a scalar-broadcast rule against an exact gradient on
the same task, charging the perturbation rules honestly for antithetic sampling
and giving every rule its own learning-rate sweep:

| hidden width | exact | 1 scalar | 4 lanes | 16 lanes |
|---|---|---|---|---|
| 16 | 30 | 980 | 600 | 160 |
| 32 | 70 | 3,760 | 880 | 220 |
| 64 | 230 | 27,400 | 1,740 | 500 |
| 128 | 320 | 50,280 | 5,760 | 1,760 |

*(forward passes to cut loss 20x; lower is better)*

A single global scalar costs **119x** an exact gradient at width 64 and **157x** at
width 128, and it scales worse than the baseline — log-log slope 1.99 against
exact's 1.20. Extrapolating that excess exponent to a real parameter count ends
the design.

The rescue is that the governing variable is not parameter count. Pooling all
configurations, the penalty tracks **fan-out per lane**:

```
slowdown ≈ (units per lane) ^ 0.80
```

with measured points from 5.3x at 1 unit/lane to 157x at 128 units/lane. Lane
count is an architectural parameter. Modularity is what buys lanes: a
block-diagonal readout makes each module's error an independently measurable
scalar at no extra forward cost.

So: **the broadcast bus survives; the single wire does not.** And the bus is still
overwhelmingly cheaper than a gradient fabric — E4.3 puts a 4096-lane bus at
16 KB/step against 26.08 GB/step for a data-parallel ring allreduce, a factor of
**1.7 million**. Widening from 1 lane to 4096 costs essentially nothing and is the
difference between a chip that learns and one that does not.

The "structurally like dopamine" analogy should go. Biological neuromodulation is
not one global scalar either; it is several systems with distinct targets, which
is the same conclusion this experiment reaches by another route.

### B3. Ternary/4-bit weights with stochastic rounding — **holds, but wins less than claimed**

Low-precision weights with stochastic rounding are established. Two corrections.

*It is not where the memory is.* E4.1: weights go 13.04 GB → 3.26 GB, a 4x on a
line item that is not the constraint. The activation row moves 65,536x. The plan
sells the small win and under-sells the large one.

*The noise argument is half right.* Noise in the **forward** pass is exploration —
correct, and a real advantage over supervised training. Noise in **update
accumulation** is a random walk that erodes learned structure. Stochastic
rounding survives this only because it is unbiased, and it still imposes a
precision floor that falls as `1/√N` in the number of updates. Low precision is
therefore for density and energy, not for capacity, and it does not remove the
need for higher-precision accumulators — which E4.1 shows become the **largest
single line item in the design** at 19.56 GB.

### B4. Masked-ROM embedding table — **drop from the critical path**

The arithmetic works: 128k vocabulary × 4096 dims × 4 bits ≈ 256 MB, and ROM is
denser than SRAM. It is still the wrong thing to freeze.

- Embeddings are among the parameters most worth adapting, and this is a design
  premised on continual adaptation.
- A vocabulary or domain change becomes a tapeout — months of latency and mask
  cost to alter a table.
- The win is small. E4.1 shows embeddings are not the bottleneck; §C shows the
  bottleneck is trace SRAM.

Evolution: embeddings in a small dedicated DRAM channel. Lookup is a sparse
gather of ~2 KB per token, which DRAM serves comfortably and which even NVMe can
serve at depth. Burn ROM only after a model is frozen for volume production,
where it is a cost optimisation rather than an architectural commitment.

### B5. Ring dataflow, no all-to-all — **holds, and is under-sold**

Correct, and stronger than stated. With no BPTT and no cross-device activations,
**batch parallelism becomes embarrassingly parallel**: N agents that share only a
credit bus. Data-parallel SGD must all-reduce the full parameter set every step;
this design exchanges kilobytes. That is a different scaling curve, not a
constant-factor saving, and it deserves to be item 1 of the hardware section
rather than item 5.

---

## C. What the plan does not budget for

The design deletes a large HBM requirement and creates a new one that is harder
to serve. **Traces are read-modify-written once per weight per step.** That is the
worst possible access pattern for every memory tier except the one the weights
already occupy. It cannot go on NAND — endurance. It should not go on HBM — that
is the row being deleted. It has to be on-die.

On-die trace SRAM, not arithmetic throughput, therefore sets how large a model a
die can adapt (E4.6, at 3 bytes per adapted weight — 1 B trace, 2 B accumulator):

| on-die SRAM | adapted params | array area (5nm-class, array only) |
|---|---|---|
| 64 MB | 22 M | 28 mm² |
| 256 MB | 89 M | 113 mm² |
| 1 GB | 358 M | 451 mm² |

A reticle-scale die holds traces for ~10⁸ adapted parameters, not 10¹⁰.

This is the most useful thing the arithmetic produces, because it converts a
vague chip concept into a product definition. **The RPU is not a from-scratch
pretraining chip. It is a sparse-adaptation engine**: a large frozen
low-precision backbone in cheap memory, plus a ~10⁸-parameter adapted surface
whose traces live on-die. Narrower than the original framing, and the only
version the numbers support.

Compute-in-memory is consequently not an optimisation here. It is load-bearing:
it is the only way to serve the access pattern the architecture creates.

---

## D. On "the chip is 90% HBM"

Overstated as an area claim, correct as an economic one. The logic die of a
current training GPU is on the order of 800 mm²; HBM stacks dominate *interposer*
area rather than total silicon, and advanced packaging capacity — not the logic
die — has been the binding supply constraint. The defensible version: **the memory
subsystem dominates cost and gates supply.** That is enough to carry the argument
and does not invite a correction that costs credibility.

---

## Summary

| # | Claim | Verdict |
|---|---|---|
| A1 | Recurrent latent state, O(1) memory | holds, restated — size the constant, keep a short exact window |
| A2 | Options over tokens | overstated — segment rather than discover; drop the 10–50x figure |
| A3 | Traces instead of gradients | breaks as written — kills time, not depth; keep local backprop |
| A4 | Imagined rollouts | holds — add exploitation mitigations to the plan |
| A5 | Intrinsic reward = compression gain | overstated — exact gradients for the world model, RL only where needed |
| B1 | CIM with in-place update | holds, restated — two-timescale, or endurance ends it |
| B2 | Single reward scalar | **breaks** — 119–157x penalty; needs lanes, which are nearly free |
| B3 | Ternary/4-bit + stochastic rounding | holds — 4x, on the wrong line item |
| B4 | Masked-ROM embeddings | drop — freezes the wrong parameters for a small win |
| B5 | Ring dataflow | holds, under-sold — promote it |
| C | Trace storage | **unbudgeted** — sets the product definition |
