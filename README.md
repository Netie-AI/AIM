# AIM — recurrent latent RL architecture, reviewed and evolved

A design review of a proposal to train language agents without
backpropagation-through-time and without a KV cache, and to build a chip around
the result. The review is quantitative: every claim that could be measured was
measured, and four of the ten did not survive contact with the numbers.

## The short version

**The core bet was misidentified.** "Kill BPTT and the KV cache in the same move"
treats as a gamble something that is a theorem. Make the recurrent state update
element-wise and the recurrent Jacobian becomes diagonal, which simultaneously
bounds the inference state and collapses real-time recurrent learning from
`O(n³)` to `O(params)`. E1 verifies that exact online gradients match full BPTT to
**3.18e-15**. The riskiest-sounding item was already settled.

**The real risk was the chip's headline feature.** A single global reward scalar —
"structurally like dopamine" — costs **119–157x** an exact gradient in E2, and
scales worse (log-log slope 1.99 vs 1.20). It does not survive. What rescues it is
that the governing variable is fan-out per lane, not parameter count:

```
slowdown ≈ (units per lane) ^ 1.0
```

E2 fitted 0.80 at widths 16–128; E7 shows the exponent drifts up to **1.01** by
width 65,536, which is the asymptote perturbation theory predicts. Small-scale
fits understate the penalty — design against 1.0. One wire across 65k units is
**55,000x** an exact gradient.

Lanes still rescue it. Holding slowdown near 10x needs ~10 units per lane, so
~10⁴ lanes for a 10⁵-unit adapted surface: 40 KB/step against 26.08 GB/step for
a data-parallel allreduce, still a ~650,000x saving. **The broadcast bus
survives a linear exponent; the single wire does not.**

**The plan created an unbudgeted constraint.** It deletes a large HBM requirement
and replaces it with a per-weight, per-step read-modify-write for traces — the
worst access pattern for every tier except on-die SRAM. That constraint, not
FLOPs, sets how large a model a die can adapt: ~10⁸ parameters at reticle scale,
not 10¹⁰.

**Which yields the product definition.** Not a from-scratch pretraining chip — a
**sparse-adaptation engine**: a large frozen low-precision backbone in cheap
memory, plus a ~10⁸-parameter adapted surface whose traces live on-die, learning
continually from sparse reward within a watt budget. Narrower than the original
framing, and the only version the arithmetic supports.

## On the SSD question

Directly, with arithmetic in E4:

| what | tier | verdict |
|---|---|---|
| experience / replay | **NVMe** | **yes** — 2.4 TB for 10M episodes, $149 of NAND vs $37,193 of HBM, 17x bandwidth headroom |
| training corpus | **NVMe** | **yes** — 7 GB/s exceeds any consumable token rate |
| frozen backbone | NVMe / DRAM | **conditionally** — break-even at 7,143 tokens/step |
| traces + accumulators | on-die SRAM | **never** — endurance forbids NAND, and HBM is the row being deleted |

And on where GPUs are genuinely mismatched: interconnect over-provisioned
1.7 million-fold, HBM capacity provisioned for an activation tape that no longer
exists, and tensor cores sized for an arithmetic intensity a recurrent step does
not have. The honest counter is that a 90%-idle GPU still beats a tapeout in the
datacenter; the case closes on-device, where per-unit power and cost dominate.

## Verdicts

| # | Claim | Verdict |
|---|---|---|
| A1 | Recurrent latent state, O(1) memory | holds, restated — ~54 B/fact, size the constant |
| A2 | Options over tokens | overstated — segment, don't discover |
| A3 | Traces instead of gradients | **breaks as written** — kills time, not depth |
| A4 | Imagined rollouts | holds — add exploitation guards |
| A5 | Intrinsic reward = compression gain | overstated — never estimate what you can compute |
| B1 | CIM with in-place update | holds, restated — two-timescale, or endurance ends it |
| B2 | Single reward scalar | **breaks** — 119–157x; needs lanes |
| B3 | Ternary/4-bit weights | holds — 4x, on the wrong line item |
| B4 | Masked-ROM embeddings | **drop** — freezes the wrong parameters |
| B5 | Ring dataflow | holds, under-sold — promote it |
| C | Trace storage | **unbudgeted** — and it sets the product definition |

## Layout

```
docs/00-review.md        claim-by-claim audit with verdicts
docs/01-architecture.md  the evolved architecture (v2)
docs/02-hardware.md      where GPUs are wrongly built; the revised RPU
docs/03-validation.md    staged experiments, kill criteria, costs
aim/diag_rnn.py          diagonal recurrence; BPTT and exact online gradients
aim/budget.py            memory/bandwidth/cost model, all constants in one place
experiments/             E1–E4, runnable; results/ holds their output
```

## Running it

Requires Python 3.11+ and numpy; no GPU, no framework.

```sh
pip install numpy
python3 experiments/e1_rtrl_equivalence.py   # gradient equivalence, memory scaling
python3 experiments/e2_credit_bus.py         # reward bus width  (~3 min)
python3 experiments/e3_state_capacity.py     # state capacity per fact
python3 experiments/e4_system_budget.py      # system budgets, SSD tiers, die sizing
```

Captured output is in `results/`. E2 accepts `E2_BUDGET` to trade runtime for
resolution at the widest setting.

## What is still unknown

Three claims remain unmeasured, and all three are testable on commodity hardware
for about three GPU-weeks before any silicon is committed — see
`docs/03-validation.md`:

- **E5**: do segmented options beat tokens on gradient SNR per unit compute?
- **E6**: does a 4–16 MB state hold real language, or does the window grow back?

E7, which was to decide whether the chip has a case, is **done**. Running it as
specified — training to convergence at width 10⁵ — would have cost ~110 GPU-days
for its single most important point, because the finding *is* that the
configuration is catastrophically slow. Replacing convergence time with
single-step gradient alignment (`1/cos²`, validated against E2's measured
slowdowns at r = 0.864, slope 1.03) gives the same law for ~7 orders of magnitude
less compute. It runs on a CPU in 90 seconds.
