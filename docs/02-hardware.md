# Hardware: where GPUs are wrongly built, and what replaces them

This is the part of the brief that asked for a direct answer: is there a regime
where current GPU resources are overkill or misallocated, and can reinforcement
learning run on SSD plus cheap compute? The answer is yes to both, but not in the
places v1 pointed at, and the honest version is narrower and more defensible.

---

## 1. Where a training GPU's money actually goes on this workload

E4.1, 7B-class model, sequence 8192, batch 8:

| line item | transformer | this design | ratio |
|---|---|---|---|
| weights | 13.04 GB | 3.26 GB | 4x |
| gradients | 13.04 GB | — | deleted |
| optimizer state | 78.23 GB | — | deleted |
| traces + accumulators | — | 19.56 GB | new |
| activations (backward) | 16.00 GB | 256 KB | **65,536x** |
| recurrent state | — | 64 MB | new |
| **total** | **120.31 GB** | **22.88 GB** | **5.3x** |

The aggregate 5.3x is the least interesting number in the table. The structure
matters more:

- The **parameter side** goes 104 GB → 23 GB, only 4.6x. Low-precision weights are
  not where the win is, despite being v1's item B3.
- The **activation side** goes 65,536x. That is the whole memory argument, and it
  comes entirely from deleting backprop through time — v1's item A3, which the
  review found is also the item stated incorrectly.

A GPU's HBM capacity exists to hold the activation tape and the KV cache. This
workload has neither. You would be renting 141 GB of HBM to use a fraction of it.

## 2. Three ways a GPU is mismatched, in descending order of severity

**Interconnect, 1.7 million-fold over-provisioned.** E4.3: data-parallel ring
allreduce moves 26.08 GB per step; a 4096-lane credit bus moves 16 KB. Link energy
per step, 0.448 J against 0.262 µJ. NVLink and the switch fabric are among the
most expensive things in a training system and this workload does not use them.

**Arithmetic intensity, and this is the subtle one.** A recurrent step is a small
number of dim-`d` matrix-vector products. Matrix-vector is the worst case for a
GPU: it is memory-bound, utilisation collapses, and tensor cores sized for
`O(n²·d)` attention idle. The problem is not that a GPU is too small for this
workload. It is that a GPU is built for arithmetic intensity this workload does
not have, and no amount of batching fixes the *latency* of the sequential
recurrence — only its throughput.

**HBM capacity, from §1.** Provisioned for a tape that no longer exists.

## 3. But the honest counter, which the pitch must survive

A 90%-idle GPU is frequently cheaper than a tapeout. Mask sets and NRE for a
modern node run to tens of millions of dollars before a single wafer, against a
GPU you can rent by the hour today with a toolchain that already works.

The economic case for custom silicon does **not** close on datacenter
pretraining. For dense supervision with a differentiable objective, backprop is
information-optimal and GPUs are correctly built for it. Nothing in this design
beats that regime, and claiming otherwise invites a correction that costs the
whole argument.

It closes where per-unit power and cost dominate and where the workload is
genuinely continual: **on-device adaptation.** An agent that learns during
deployment, from sparse reward, within a watt budget, with no room for an
optimizer state or a growing KV cache. There, the constraints are exactly the
ones this architecture removes, and there is no incumbent that fits.

That is also, not coincidentally, the setting where "SSD plus cheap compute" is
the actual constraint rather than a preference.

---

## 4. The SSD question, tier by tier

Answered directly, with the arithmetic in E4.4 and E4.5.

| what | tier | verdict |
|---|---|---|
| experience / replay buffer | **NVMe** | **yes** — the design's natural home |
| training corpus | **NVMe** | **yes** — 7 GB/s exceeds any consumable token rate |
| frozen backbone weights | NVMe or DRAM | **conditionally** — batch ≥ 7,143 tokens/step |
| traces + accumulators | on-die SRAM | **never** — this is the hard constraint |

**Replay: yes, decisively.** 2.4 TB for 10M episodes, $149 of NAND against $37,193
of HBM, needing 0.42 GB/s against 7 GB/s available. Append-mostly,
read-sequential, cold, huge — a textbook NAND workload, and 250x cheaper per byte
than HBM.

**Weights: only in batch.** E4.5 gives the break-even batch at which compute time
covers streaming the weights once:

| tier | bandwidth | break-even batch |
|---|---|---|
| HBM | 4,800 GB/s | 10 tokens/step |
| DRAM | 400 GB/s | 125 tokens/step |
| NVMe | 7 GB/s | 7,143 tokens/step |

Reachable for offline batch training. Not reachable for single-agent online
adaptation, which is the target application. So the weights want to be *in* the
compute — which is the real argument for compute-in-memory. Not density: the
deletion of this row.

**Traces: never, and this is what the plan missed.** Traces are read-modify-written
once per weight per step. That is the worst access pattern for every tier except
the one the weights already occupy. NAND endurance forbids it; HBM is the row
being deleted. It must be on-die, and that constraint closes the design (§6).

---

## 5. The RPU, revised

| # | v1 | v2 | why |
|---|---|---|---|
| 1 | CIM tiles, in-place update | **two-timescale**: SRAM traces, NVM weights consolidated every 10³–10⁵ updates | NVM endurance is 10⁶–10⁹ writes; per-step in-place update exhausts it in days |
| 2 | one global scalar wire | **multi-lane bus**, ≤32 units per lane | E2: one scalar costs 119–157x; lanes are nearly free |
| 3 | ternary/4-bit + stochastic rounding | kept, **plus int16 shadow accumulators** | unbiased rounding still leaves a `1/√N` precision floor; accumulation cannot be 4-bit |
| 4 | masked-ROM embeddings | **dedicated DRAM channel** | freezes the parameters most worth adapting; a vocab change becomes a tapeout |
| 5 | ring dataflow | kept, **promoted to item 1** | with no BPTT, batch parallelism needs only the bus — a different scaling curve, not a constant factor |

On item 2: the dopamine analogy should be dropped. Biological neuromodulation is
not a single global scalar either — it is several systems with distinct targets,
which is where E2 lands by a different route. The analogy argues against v1's
design, not for it.

On item 3, one correction that matters for correctness rather than marketing:
noise in the **forward** pass is exploration, and that is a genuine advantage over
supervised training. Noise in **update accumulation** is a random walk that erodes
learned structure. Stochastic rounding survives only because it is unbiased. The
two must not be conflated in a datasheet.

---

## 6. What the arithmetic decides

On-die trace SRAM, not FLOPs, sets how large a model a die can adapt. At 3 bytes
per adapted weight (1 B trace, 2 B accumulator), E4.6:

| on-die SRAM | adapted params | % of a 7B model | array area (5nm-class) |
|---|---|---|---|
| 64 MB | 22 M | 0.32% | 28 mm² |
| 256 MB | 89 M | 1.28% | 113 mm² |
| 1 GB | 358 M | 5.11% | 451 mm² |

*Array area only, 6T cell, excluding compute tiles, I/O and memory controllers.*

A reticle-scale die holds traces for ~10⁸ adapted parameters, not 10¹⁰.

**This is the product definition.** The RPU is not a from-scratch pretraining
chip. It is a **sparse-adaptation engine**: a large frozen low-precision backbone
in cheap memory — DRAM, or NVMe at sufficient batch — plus a ~10⁸-parameter
adapted surface whose traces live on-die, learning continually from sparse
reward within a watt budget.

Narrower than v1's framing. It is also the only version the numbers support, it
has no incumbent, and it inherits every one of the structural advantages the
review found held: bounded state, no activation tape, no gradient fabric, and an
experience store that costs $149 instead of $37,193.
