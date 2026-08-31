"""E4 -- Where is a training GPU actually being spent, and what would replace it?

Arithmetic, not measurement. The point is to find which line items the proposed
architecture deletes, which it only shrinks, and which it does not touch --
because the plan claims a bigger win than the numbers support in some places
and a smaller one than they support in others.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aim.budget import *


def fmt(b):
    if b == 0:
        return f"{'-':>12}"
    if b >= GB:
        return f"{b / GB:>9,.2f} GB"
    if b >= 2 ** 20:
        return f"{b / 2 ** 20:>9,.2f} MB"
    return f"{b / 1024:>9,.2f} KB"


def row(label, val_a, val_b):
    if val_a and val_b:
        r = f"{val_a / val_b:>9,.0f}x" if val_a >= val_b else f"{val_a / val_b:>10.2f}x"
    else:
        r = f"{'--':>10}"
    print(f"  {label:<24} {fmt(val_a):>13} {fmt(val_b):>13} {r}")


def main():
    m = Model(params=7e9, layers=32, hidden=4096, seq=8192, batch=8)
    print("=" * 78)
    print("E4.1  training-time resident memory   (7B params, seq 8192, batch 8)")
    print("=" * 78)
    t = transformer_training_bytes(m)
    a = aim_training_bytes(m)
    print(f"  {'line item':<24} {'transformer':>12} {'this design':>12} {'ratio':>10}")
    row("weights", t["weights"], a["weights_int4"])
    row("gradients", t["grads"], 0)
    row("optimizer state", t["optimizer"], 0)
    row("traces + accumulators", 0, a["traces_int8"] + a["shadow_accum_int16"])
    row("activations (backward)", t["activations"], a["activations"])
    row("recurrent state", 0, a["state"])
    tt = sum(t.values())
    at = sum(a.values())
    print(f"  {'-' * 62}")
    row("TOTAL", tt, at)
    print(f"\n  Honest split of that {tt / at:.1f}x: the parameter side "
          f"({(t['weights'] + t['grads'] + t['optimizer']) / GB:,.0f} GB ->"
          f" {(a['weights_int4'] + a['traces_int8'] + a['shadow_accum_int16']) / GB:,.0f} GB)"
          f" is only "
          f"{(t['weights']+t['grads']+t['optimizer'])/(a['weights_int4']+a['traces_int8']+a['shadow_accum_int16']):.1f}x.")
    print(f"  The activation side is {t['activations'] / a['activations']:,.0f}x."
          "  Deleting backprop-through-time is")
    print("  where the memory win lives; low-precision weights are a rounding")
    print("  error by comparison. The plan under-sells one and over-sells the other.")

    print("\n" + "=" * 78)
    print("E4.2  inference-time state   (same model, per sequence)")
    print("=" * 78)
    m1 = Model(7e9, 32, 4096, 131072, 1)
    kv = transformer_kv_bytes(m1)
    st = aim_training_bytes(m1)["state"]
    print(f"  KV cache at 128k ctx:        {kv / GB:>10,.2f} GB   grows with context")
    print(f"  recurrent state:             {st / GB:>10,.4f} GB   flat in context")
    print(f"  ratio:                       {kv / st:>10,.0f}x")
    print(f"  HBM cost of that KV cache:   ${kv / GB * HBM_USD_PER_GB:>9,.0f} per concurrent sequence")

    print("\n" + "=" * 78)
    print("E4.3  per-step inter-device traffic")
    print("=" * 78)
    ic = interconnect_bytes_per_step(m)
    print(f"  data-parallel ring allreduce: {ic['ring_allreduce_bf16'] / GB:>10,.2f} GB/step")
    print(f"  4096-lane credit bus:         {ic['credit_bus'] / 1024:>10,.2f} KB/step")
    print(f"  ratio:                        {ic['ring_allreduce_bf16'] / ic['credit_bus']:>10,.0f}x")
    e_ar = ic["ring_allreduce_bf16"] * 8 * E_SERDES / 1e12
    e_cb = ic["credit_bus"] * 8 * E_SERDES / 1e12
    print(f"  link energy per step:         {e_ar:>10,.3f} J   vs {e_cb * 1e6:,.3f} uJ")
    print("\n  This is the largest ratio in the entire design and the one that")
    print("  justifies deleting the interconnect fabric. It is also the claim")
    print("  most exposed to E2: the bus only works if it has enough lanes.")

    print("\n" + "=" * 78)
    print("E4.4  does the experience store fit on an SSD?")
    print("=" * 78)
    latent = 4096                                   # 4096-dim int8 latent
    for eps in [100_000, 1_000_000, 10_000_000]:
        b = replay_buffer(eps, 64, latent)
        print(f"  {eps:>12,} episodes x 64 steps -> {b / TB:>7.2f} TB"
              f"   NAND ${b / GB * NAND_USD_PER_GB:>10,.0f}"
              f"   HBM ${b / GB * HBM_USD_PER_GB:>13,.0f}")
    steps_per_s = 100_000
    need = steps_per_s * (latent + 64) / 1e9
    print(f"\n  feeding {steps_per_s:,} replayed steps/s needs {need:.2f} GB/s;"
          f" one NVMe delivers {SSD_BW_GBPS:.0f} GB/s")
    print(f"  -> {SSD_BW_GBPS / need:.0f}x headroom. The replay path is genuinely")
    print("     SSD-shaped: append-mostly, read-sequential, cold, and huge.")
    print(f"  -> and {HBM_USD_PER_GB / NAND_USD_PER_GB:.0f}x cheaper per byte than HBM.")

    print("\n" + "=" * 78)
    print("E4.5  can the WEIGHTS live on the SSD too?")
    print("=" * 78)
    wb = 7e9 * 0.5                                   # int4
    flops_per_tok = 2 * 7e9
    for tier, bw in [("HBM", HBM_BW_GBPS), ("DRAM", DRAM_BW_GBPS), ("NVMe SSD", SSD_BW_GBPS)]:
        bmin = stream_weights_min_batch(wb, flops_per_tok, 200.0, bw)
        print(f"  {tier:<10} {bw:>7,.0f} GB/s -> break-even batch "
              f"{bmin:>12,.0f} tokens/step")
    print("\n  Weight streaming from NAND needs a batch three orders of magnitude")
    print("  larger than from HBM to stay compute-bound. For batched offline")
    print("  training that is reachable; for single-agent online adaptation it")
    print("  is not. So: experience on SSD yes, weights on SSD only in batch.")
    print("  The weights want to be IN the compute, which is the actual argument")
    print("  for compute-in-memory -- not density, but the deletion of this row.")

    print("\n" + "=" * 78)
    print("E4.6  the constraint the plan creates but does not budget for")
    print("=" * 78)
    print("  Traces are read-modify-written once per weight per step. That is the")
    print("  single worst access pattern for any tier except the one the weights")
    print("  already sit in. It cannot go on NAND (endurance) and should not go")
    print("  on HBM (that is the row we were deleting). It has to be on-die.")
    print("  So on-die trace SRAM, not FLOPs, sets how large a model a die can adapt.\n")
    print(f"  {'on-die SRAM':>12} {'adapted params':>16} {'as % of 7B':>12} {'array area*':>13}")
    for mb in [64, 256, 1024]:
        # 1 B trace (int8) + 2 B shadow accumulator per adapted weight
        adapted = mb * 2 ** 20 / 3.0
        # ~0.021 um^2 per SRAM bit at a 5nm-class node, ~2.5x for periphery/ECC
        area = mb * 2 ** 20 * 8 * 0.021e-6 * 2.5
        print(f"  {mb:>9,} MB {adapted / 1e6:>13,.0f} M {adapted / 7e9 * 100:>11.2f}% "
              f"{area:>10,.0f} mm2")
    print("\n  * array area only, at a 5nm-class 6T cell; excludes compute tiles,")
    print("    I/O and the DRAM/NVM controllers.")
    print("\n  This closes the design. A reticle-scale die can hold traces for")
    print("  roughly 10^8 adapted parameters, not 10^10. The RPU is therefore not")
    print("  a from-scratch pretraining chip -- it is a sparse-adaptation engine:")
    print("  a large frozen low-precision backbone held in cheap memory, plus a")
    print("  ~100M-parameter adapted surface whose traces live on-die. That is a")
    print("  narrower and much more defensible product than the original framing,")
    print("  and it is the only version the arithmetic actually supports.")


if __name__ == "__main__":
    main()
