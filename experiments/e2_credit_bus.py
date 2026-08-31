"""E2 -- How wide does the reward bus have to be?

The chip proposal deletes the gradient interconnect and replaces it with a
single broadcast scalar. That is only sound if one scalar can carry enough
credit for every parameter it fans out to. The perturbation-learning literature
says the sample cost of a scalar-broadcast rule grows with the number of units
sharing that scalar. This measures the exponent, and measures whether splitting
the network into independently-credited modules buys it back.

Setup: teacher-student regression, y = V tanh(W x), V fixed, learn W.
  exact      -- true gradient (what a GPU backward pass delivers)
  1 scalar   -- node perturbation, ONE global scalar for all n hidden units
  C lanes    -- network split into C modules, each with its own scalar readout
                (block-diagonal V -- modularity is what buys independent lanes)

All rules see the same minibatch stream and are charged for forward passes, so
the 2x of antithetic sampling is billed honestly to the perturbation rules.
Every rule gets its own learning-rate sweep; we report each one's best.

Kill criterion: if `1 scalar` scales as O(n) and extra lanes do not improve on
it, the single-wire bus cannot work and the interconnect saving is fake.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

D_IN, D_OUT, BATCH = 16, 16, 8
MAX_FWD = int(os.environ.get("E2_BUDGET", 120_000))


def make_task(n_hidden, n_lanes, seed):
    rng = np.random.default_rng(seed)
    V = np.zeros((D_OUT, n_hidden))
    if n_lanes == 1:
        V = rng.normal(0, 1 / np.sqrt(n_hidden), size=(D_OUT, n_hidden))
    else:                                    # block-diagonal readout
        hs, os_ = n_hidden // n_lanes, D_OUT // n_lanes
        for b in range(n_lanes):
            V[b * os_:(b + 1) * os_, b * hs:(b + 1) * hs] = \
                rng.normal(0, 1 / np.sqrt(hs), size=(os_, hs))
    W_star = rng.normal(0, 1 / np.sqrt(D_IN), size=(n_hidden, D_IN))
    W0 = rng.normal(0, 1 / np.sqrt(D_IN), size=(n_hidden, D_IN))
    Xe = rng.normal(size=(512, D_IN))        # held-out probe for loss readout
    return V, W_star, W0, Xe


def full_loss(W, W_star, V, Xe):
    E = np.tanh(Xe @ W.T) @ V.T - np.tanh(Xe @ W_star.T) @ V.T
    return 0.5 * float((E ** 2).sum(1).mean())


def run(rule, n_hidden, n_lanes, lr, sigma, seed, target_frac):
    V, W_star, W0, Xe = make_task(n_hidden, n_lanes, seed)
    W = W0.copy()
    rng = np.random.default_rng(seed + 999)
    L0 = full_loss(W, W_star, V, Xe)
    target, fwd, check = target_frac * L0, 0, 0
    hs = n_hidden // n_lanes
    while fwd < MAX_FWD:
        X = rng.normal(size=(BATCH, D_IN))
        Y = np.tanh(X @ W_star.T) @ V.T
        if rule == "exact":
            H = np.tanh(X @ W.T)
            dZ = ((H @ V.T - Y) @ V) * (1 - H ** 2)
            W -= lr * (dZ.T @ X) / BATCH
            fwd += 1
        else:
            # node perturbation. Noise is per-sample and local to each unit;
            # the ONLY quantity crossing modules is the scalar (or C scalars).
            xi = rng.normal(0, sigma, size=(BATCH, n_hidden))
            Zc = X @ W.T
            Ep = np.tanh(Zc + xi) @ V.T - Y
            Em = np.tanh(Zc - xi) @ V.T - Y
            fwd += 2
            # per-lane reward difference (n_lanes==1 -> the global scalar)
            dp = 0.5 * (Ep ** 2).reshape(BATCH, n_lanes, -1).sum(2)
            dm = 0.5 * (Em ** 2).reshape(BATCH, n_lanes, -1).sum(2)
            d = np.repeat(dp - dm, hs, axis=1)          # (BATCH, n_hidden)
            ghat = d * xi / (2 * sigma ** 2)            # local: scalar x noise
            W -= lr * (ghat.T @ X) / BATCH              # local: x presynaptic
        check += 1
        if check % 10 == 0:
            L = full_loss(W, W_star, V, Xe)
            if not np.isfinite(L) or L > 50 * L0:
                return None
            if L <= target:
                return fwd
    return None


def best_cost(rule, n_hidden, n_lanes, seed=0, target_frac=0.05):
    best = None
    for lr in [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3]:
        c = run(rule, n_hidden, n_lanes, lr, 0.05, seed, target_frac)
        if c is not None and (best is None or c < best):
            best = c
    return best


def main():
    widths = [16, 32, 64, 128]
    lane_opts = [1, 4, 16]
    print("=" * 78)
    print("E2  forward passes to cut loss 20x vs width   (lower is better)")
    print(f"    minibatch={BATCH}, budget={MAX_FWD} fwd, best LR per cell")
    print("=" * 78)
    print(f"{'n_hidden':>9} {'params':>8} {'exact':>10} {'1 scalar':>10} "
          f"{'4 lanes':>10} {'16 lanes':>10}")
    exact, lanes = {}, {}
    for n in widths:
        exact[n] = best_cost("exact", n, 1)
        for c in lane_opts:
            lanes[(n, c)] = best_cost("perturb", n, c)
        f = lambda v: (f"{v:>10,}" if v else f"{'>budget':>10}")
        print(f"{n:>9} {n * D_IN:>8} {f(exact[n])} "
              + " ".join(f(lanes[(n, c)]) for c in lane_opts))

    print("\n  log-log slope of cost vs width  (exact is the baseline slope):")
    for label, series in [("exact gradient", [exact[n] for n in widths])] + \
            [(f"{c} lane(s)", [lanes[(n, c)] for n in widths]) for c in lane_opts]:
        if all(x is not None for x in series):
            s = float(np.polyfit(np.log(widths), np.log(series), 1)[0])
            print(f"    {label:>16}: {s:>6.2f}")
        else:
            miss = sum(x is None for x in series)
            print(f"    {label:>16}: over budget at {miss}/{len(widths)} widths")

    # The design variable is not total width and not lane count on its own --
    # it is how many units are forced to share one scalar.
    print("\n" + "=" * 78)
    print("  slowdown vs exact gradient, as a function of UNITS PER LANE")
    print("  (this, not parameter count, is what sets the required bus width)")
    print("=" * 78)
    print(f"{'units/lane':>11} {'slowdown vs exact':>19}   {'from (n_hidden, lanes)':>24}")
    rows = []
    for n in widths:
        for c in lane_opts:
            v = lanes[(n, c)]
            if v is not None and exact[n]:
                rows.append((n // c, v / exact[n], n, c))
    for upl, sd, n, c in sorted(rows):
        print(f"{upl:>11} {sd:>18.1f}x   {'(%d, %d)' % (n, c):>24}")
    if len(rows) > 2:
        s = float(np.polyfit(np.log([r[0] for r in rows]),
                             np.log([r[1] for r in rows]), 1)[0])
        print(f"\n  slowdown ~ (units per lane)^{s:.2f}")
        print("  -> a single global scalar is not viable at scale; the cost of")
        print("     credit is set by fan-out per lane, which is a bus-width")
        print("     parameter the architecture gets to choose.")


if __name__ == "__main__":
    main()
