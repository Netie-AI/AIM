# Architecture v2

What survived review, restated as a buildable specification. Changes from v1 are
marked **[changed]**; everything else carried over intact.

The organising principle is narrower than v1's and follows from §A5 of the
review: **use exact gradients wherever they are available and cheap, and
reinforcement only where they are not.** v1 tried to replace differentiation
everywhere. That is neither necessary nor affordable, and E2 prices the attempt.

---

## L0 — State

Element-wise recurrent latent state, deliberately large, plus a short exact
attention window. **[changed: hybrid rather than pure recurrence]**

```
z_t = a ⊙ h_{t-1} + W x_t + b
h_t = φ(z_t)
```

`a` is a per-unit decay in `(0, 1)`. Diagonality is not a simplification for
tractability — it is the property the entire design rests on, and giving it up
costs both the bounded inference state and the exact online gradient at once.

- **State size**: 4–16 MB per agent, not kilobytes. E3 measures ~54 B per
  retrievable fact, flat in key dimension. Size the state for what must persist
  across the episode, then add margin — recall degrades gracefully, so
  under-sizing shows up as quality loss rather than failure, which is harder to
  notice.
- **Exact window**: a fixed 512–2048 token sliding attention window alongside the
  recurrent state. Its cache is bounded, so the KV-cache growth problem is still
  solved. E3 shows the recurrent state is cheaper per fact but only at ~90%
  recall; verbatim recall over recent context is exactly what the window is for,
  and refusing it buys nothing.
- **Write rule**: outer-product accumulation, not the delta rule. E3 shows delta
  saturates at exactly 1.0 pairs per unit of key dimension because it overwrites
  once keys stop being independent, while accumulation reaches ~2.4 and degrades
  gracefully instead of discarding.

## L1 — Local learning blocks **[changed: new layer, replaces "no backward pass"]**

The network is a stack of blocks of 2–6 layers. Within a block, gradients are
exact and computed by ordinary backpropagation over **one timestep**. Across
blocks, no gradient flows; blocks are coupled by the credit bus (L3).

This is the correction from §A3. Eliminating backprop through *time* is what the
memory argument needed. Eliminating it through *depth* was never necessary and
E2 prices the alternative at 119–157x. Per-step activation memory for a 7B-class
model at batch 8 is **256 KB** (E4.1) — it fits in SRAM, so there is nothing left
to save.

Block boundaries do double duty: they are the unit of local backprop and the unit
of credit-bus lane assignment, and E2 shows lane granularity is what determines
learning speed.

## L2 — Temporal credit: exact online sensitivities

Each recurrent parameter carries a trace, which is its RTRL sensitivity:

```
S_a ← φ'(z_t) ⊙ (h_{t-1} + a ⊙ S_a)
S_W ← φ'(z_t) ⊙ (x_t     + a ⊙ S_W)      # per weight, same shape as W
S_b ← φ'(z_t) ⊙ (1       + a ⊙ S_b)
```

Gradient at each step is `(∂L_t/∂h_t) ⊙ S`. E1 confirms this equals BPTT to
3.18e-15 and that live memory is flat in sequence length while BPTT's grows
linearly — 12,484x apart at T = 1M.

Two things worth stating plainly, because v1 undersold both:

- The decay constant of the eligibility trace is not a hyperparameter. It is `a`,
  the recurrence itself. The trace is derived, not postulated.
- Truncated BPTT, the usual way to bound this memory, discards gradient to do it:
  E1.3 measures 47.7% relative error at k=1 and 7.4% at k=64. The online path
  bounds memory and discards nothing.

## L3 — Structured credit: a multi-lane bus **[changed: was a single scalar]**

Per-block value heads produce **C independent scalars per step**, broadcast to all
tiles. Each block updates from its own lane.

E2 is unambiguous that this is the load-bearing parameter:

```
slowdown vs exact gradient ≈ (units per lane) ^ 1.0
```

E2 fitted 0.80 at widths 16–128; E7 measures **1.01** at widths 1k–65k, the
asymptote perturbation theory predicts. Design against 1.0 — small-scale fits
understate the penalty.

Target **≤ 10 units per lane**, giving ~10⁴ lanes for a 10⁵-unit adapted surface:
40 KB/step against 26.08 GB/step for a ring allreduce, a ~650,000x saving. The
fabric win is barely dented by widening the bus two orders of magnitude past
v1's, so there is no reason to be parsimonious with lanes and every reason not to
be.

## L4 — Action space: segmented options **[changed: segmented, not discovered]**

The policy acts over spans. Spans are **segmented** at cheap syntactic boundaries
— sentence, clause, enumerated step, code statement — not discovered by an
open research problem the plan cannot afford to depend on. A pretrained decoder,
frozen, renders a chosen option into tokens.

The benefit claimed is variance reduction, and it is to be measured as gradient
SNR per unit compute against a token-level baseline (E5). v1's "10 to 50x credit
horizon reduction" is withdrawn until that measurement exists.

Learned segmentation is a later upgrade, gated on segmented options working
first. It is not on the critical path.

## L5 — World model and imagination

Latent-space world model predicting its own next state, trained on **compression
gain with exact local gradients** — one-step prediction, no BPTT, so it is cheap.
The policy trains predominantly inside imagined rollouts; real reward corrects
the model.

Guards, promoted from unlisted risk to specification (§A4):

- Ensemble of 3–5 latent predictors; penalise imagined reward by ensemble
  disagreement, so the policy cannot profit from states the model is unsure of.
- Imagination horizon 8–16 steps. Short rollouts, frequent grounding.
- Fixed ratio of real to imagined transitions, monitored. Silent drift toward
  pure imagination is the failure mode to instrument for.

## L6 — Experience store

Replay of `(latent, option, reward, next-latent)` on NVMe. E4.4: 10M episodes ×
64 steps is 2.4 TB — **$149 of NAND against $37,193 of HBM** for the same bytes —
and feeding 100k replayed steps/s needs 0.42 GB/s against ~7 GB/s available, 17x
headroom. Append-mostly, read-sequential, cold, enormous. This is the one place
in the design where an SSD is not a compromise but the correct tier.

---

## The learning rule, whole

| axis | mechanism | why |
|---|---|---|
| across time | exact online sensitivities (traces) | E1: exact, O(params), flat in T |
| within a block | ordinary backprop, one timestep | E4.1: 256 KB, fits on-die |
| across blocks | multi-lane credit bus | E2: cheap; single scalar is not |
| world model | exact gradients on compression gain | §A5: never estimate what you can compute |
| policy | policy gradient × trace | the only genuinely non-differentiable part |

## What must be true

Ordered by how much rests on it and how early it can be checked.

1. **A bounded state suffices for the target tasks.** E3 gives the sizing law;
   whether real language tasks fit in 4–16 MB is open. Checked by E6.
2. ~~**≤32 units per lane is achievable at scale.**~~ **Settled by E7**, and the
   answer moved the target: the exponent is 1.0 at scale, not 0.80, so the design
   point is ≤10 units per lane and ~10⁴ lanes. Still ~650,000x below an allreduce.
3. **Segmented options beat tokens on gradient SNR per unit compute.** Entirely
   unmeasured. Checked by E5.
4. **Imagined rollouts stay grounded** under a disagreement penalty at language
   scale. Established elsewhere, unverified here.

Items 1–3 are all measurable on commodity hardware before any silicon exists.
That ordering is the point of `docs/03-validation.md`.
