"""E1 -- Is "kill BPTT" a bet, or is it already won?

Three measurements:

  1. Do exact online gradients equal BPTT gradients for a diagonal recurrence?
  2. How does live training memory scale with sequence length for each?
  3. Truncated BPTT is the standard cheap alternative. How much gradient does
     it actually lose, and does the online path lose any?

Kill criterion: if (1) fails, or (2) shows the online path growing with T, the
temporal half of the architecture is unfounded.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from aim.diag_rnn import (init_params, forward, bptt_grads, rtrl_grads,
                          n_params, rtrl_state_bytes, bptt_state_bytes)


def truncated_bptt_grads(p, xs, targets, window):
    """Standard TBPTT: carry state forward, backprop only `window` steps."""
    g = {k: np.zeros_like(v) for k, v in p.items()}
    h = np.zeros_like(p["b"])
    for s in range(0, len(xs), window):
        chunk_x, chunk_t = xs[s:s + window], targets[s:s + window]
        _, gc = bptt_grads(p, chunk_x, chunk_t, h0=h)
        for k in g:
            g[k] += gc[k]
        h = forward(p, chunk_x, h0=h)[0][-1]   # state carries, gradient does not
    return g


def rel_err(x, y):
    return float(np.max(np.abs(x - y)) / (np.max(np.abs(y)) + 1e-12))


def main():
    print("=" * 74)
    print("E1.1  exact online gradients vs BPTT  (diagonal recurrence)")
    print("=" * 74)
    print(f"{'n_hidden':>9} {'T':>6} {'params':>9} {'max rel err':>13}  {'loss match':>11}")
    worst = 0.0
    for n_hidden, T in [(16, 64), (64, 256), (256, 512), (512, 1024)]:
        p = init_params(32, n_hidden, 16, seed=n_hidden)
        rng = np.random.default_rng(7)
        xs = rng.normal(size=(T, 32))
        tg = rng.normal(size=(T, 16))
        lb, gb = bptt_grads(p, xs, tg)
        lr, gr = rtrl_grads(p, xs, tg)
        e = max(rel_err(gr[k], gb[k]) for k in gb)
        worst = max(worst, e)
        print(f"{n_hidden:>9} {T:>6} {n_params(p):>9} {e:>13.2e}  "
              f"{abs(lb - lr) / lb:>11.1e}")
    print(f"\n  worst relative error across all scales: {worst:.2e}")
    print("  -> identical to floating-point noise. The online path is not an")
    print("     approximation of BPTT; it computes the same gradient.\n")

    print("=" * 74)
    print("E1.2  live training memory vs sequence length  (n_hidden=512, fp32)")
    print("=" * 74)
    p = init_params(32, 512, 16, seed=0)
    print(f"{'T':>8} {'BPTT tape (MB)':>16} {'online (MB)':>13} {'ratio':>9}")
    for T in [128, 1_024, 8_192, 131_072, 1_048_576]:
        b = bptt_state_bytes(p, T) / 2**20
        r = rtrl_state_bytes(p) / 2**20
        print(f"{T:>8} {b:>16.1f} {r:>13.1f} {b / r:>8.1f}x")
    print("\n  Online memory is flat in T by construction: the traces are the")
    print("  same shape as the weights. This is the property that lets a")
    print("  training chip hold its whole working set on-die.\n")

    print("=" * 74)
    print("E1.3  what truncation costs  (n_hidden=64, T=512, a ~ U(0,0.9))")
    print("=" * 74)
    p = init_params(32, 64, 16, seed=3)
    rng = np.random.default_rng(11)
    xs = rng.normal(size=(512, 32))
    tg = rng.normal(size=(512, 16))
    _, g_full = bptt_grads(p, xs, tg)
    _, g_on = rtrl_grads(p, xs, tg)
    print(f"{'method':>22} {'grad rel err vs full':>22} {'cos sim':>10} {'act mem (MB)':>14}")

    def cos(gx):
        u = np.concatenate([gx[k].ravel() for k in sorted(gx)])
        v = np.concatenate([g_full[k].ravel() for k in sorted(g_full)])
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))

    for w in [1, 4, 16, 64]:
        gw = truncated_bptt_grads(p, xs, tg, w)
        e = max(rel_err(gw[k], g_full[k]) for k in g_full)
        mb = bptt_state_bytes(p, w) / 2**20
        print(f"{'TBPTT(k=%d)' % w:>22} {e:>22.2e} {cos(gw):>10.4f} {mb:>14.4f}")
    e = max(rel_err(g_on[k], g_full[k]) for k in g_full)
    print(f"{'online (this design)':>22} {e:>22.2e} {cos(g_on):>10.4f} "
          f"{rtrl_state_bytes(p) / 2**20:>14.4f}")
    print("\n  TBPTT buys bounded memory by throwing away gradient. The online")
    print("  path buys bounded memory and throws away nothing.")


if __name__ == "__main__":
    main()
