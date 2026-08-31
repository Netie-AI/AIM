"""Memory, bandwidth and cost budgets for the two training regimes.

Every constant below is an order-of-magnitude planning figure, named and kept
in one place so the conclusions can be re-derived when a number is wrong.
Nothing here is a measurement; it is arithmetic over public specs.
"""

GB = 2 ** 30
TB = 2 ** 40

# --- media, ~2025 planning figures --------------------------------------
HBM_USD_PER_GB = 15.0        # HBM3/3e, supply-constrained
DRAM_USD_PER_GB = 4.0        # DDR5 module
NAND_USD_PER_GB = 0.06       # consumer TLC NVMe
HBM_BW_GBPS = 4800.0         # per-package, H200/B200 class
DRAM_BW_GBPS = 400.0         # multi-channel server DDR5
SSD_BW_GBPS = 7.0            # PCIe Gen4 x4 NVMe sequential read
SSD_RAND_IOPS = 1_000_000    # 4 KB random read, high queue depth

# --- energy, pJ per bit moved -------------------------------------------
E_SRAM = 0.2
E_HBM = 4.0
E_SERDES = 2.0               # off-package link


class Model:
    def __init__(self, params, layers, hidden, seq, batch,
                 kv_heads=8, head_dim=128):
        self.p, self.L, self.h = params, layers, hidden
        self.seq, self.batch = seq, batch
        self.kv_heads, self.head_dim = kv_heads, head_dim


def transformer_training_bytes(m):
    """Mixed-precision Adam: 16 B/param is the standard accounting."""
    weights = 2 * m.p                      # bf16
    grads = 2 * m.p                        # bf16
    optimizer = 12 * m.p                   # fp32 master + m + v
    # activations with recompute at layer boundaries
    acts = m.seq * m.batch * m.h * m.L * 2
    return {"weights": weights, "grads": grads, "optimizer": optimizer,
            "activations": acts}


def transformer_kv_bytes(m):
    return 2 * m.L * m.kv_heads * m.head_dim * m.seq * m.batch * 2


def aim_training_bytes(m, adapt_frac=1.0, state_bytes=8 * 2 ** 20,
                       block_depth=4):
    """Online-trace training. The two absent line items are the point.

    adapt_frac: fraction of weights carrying a trace + shadow accumulator.
    block_depth: layers inside one locally-backpropped block; activation
                 memory is one timestep deep, not one sequence deep.
    """
    adapted = m.p * adapt_frac
    return {
        "weights_int4": m.p * 0.5,
        "traces_int8": adapted * 1.0,
        "shadow_accum_int16": adapted * 2.0,
        "activations": m.batch * m.h * block_depth * 2,   # no seq factor
        "state": m.batch * state_bytes,
    }


def interconnect_bytes_per_step(m, lanes=4096, dtype=4):
    """Data-parallel allreduce vs a broadcast credit bus."""
    return {"ring_allreduce_bf16": 2 * (2 * m.p),        # 2x payload, ring
            "credit_bus": lanes * dtype}


def replay_buffer(episodes, steps_per_ep, latent_bytes, aux_bytes=64):
    return episodes * steps_per_ep * (latent_bytes + aux_bytes)


def stream_weights_min_batch(weight_bytes, flops_per_token, tflops,
                             link_gbps=SSD_BW_GBPS):
    """Batch at which compute time covers the time to stream weights once.

    Streaming weights from a tier only pays if arithmetic intensity is high
    enough to hide the transfer. This returns the break-even batch size.
    """
    stream_s = weight_bytes / (link_gbps * 1e9)
    per_token_s = flops_per_token / (tflops * 1e12)
    return stream_s / per_token_s
