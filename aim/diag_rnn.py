"""Diagonal (element-wise) recurrent core.

    z_t = a * h_{t-1} + W x_t + b
    h_t = tanh(z_t)
    y_t = V h_t

The recurrence is element-wise in `a`, so the recurrent Jacobian dh_t/dh_{t-1}
is diagonal. That single property is what the whole architecture rests on:

  * Inference: the state is a fixed-size vector. Nothing accumulates with
    sequence length, so there is no KV cache to grow.
  * Training: because unit i's history never mixes into unit j's, the RTRL
    sensitivity tensor dh_t/dtheta stays the same shape as theta itself.
    Exact online gradients cost O(params) memory and zero stored activations.

The second point is the one people get wrong. Eligibility traces are usually
introduced as a cheap *approximation* to backprop through time. For a diagonal
recurrence they are not an approximation -- `rtrl_grads` and `bptt_grads` below
return the same numbers to floating-point precision, and `experiments/e1` checks
it. The per-weight trace register in the hardware proposal is exactly this
sensitivity, so the chip is storing a mathematically exact quantity, not a
heuristic.
"""

import numpy as np


def init_params(n_in, n_hidden, n_out, seed=0, spectral=0.9):
    rng = np.random.default_rng(seed)
    return {
        # a is the diagonal transition. Kept inside (-1, 1) for stability; this
        # doubles as the natural per-unit trace decay rate.
        "a": rng.uniform(0.0, spectral, size=n_hidden),
        "W": rng.normal(0, 1.0 / np.sqrt(n_in), size=(n_hidden, n_in)),
        "b": np.zeros(n_hidden),
        "V": rng.normal(0, 1.0 / np.sqrt(n_hidden), size=(n_out, n_hidden)),
    }


def n_params(p):
    return sum(v.size for v in p.values())


def forward(p, xs, h0=None):
    """Plain rollout. Returns (states, outputs). Used by both gradient paths."""
    T = len(xs)
    h = np.zeros_like(p["b"]) if h0 is None else h0.copy()
    hs, ys = [], []
    for t in range(T):
        h = np.tanh(p["a"] * h + p["W"] @ xs[t] + p["b"])
        hs.append(h)
        ys.append(p["V"] @ h)
    return np.array(hs), np.array(ys)


def bptt_grads(p, xs, targets, h0=None):
    """Reference path: store every activation, walk backwards.

    Activation storage is O(T * n_hidden). This is the cost the design is
    trying to delete, and it is measured in experiments/e1.
    """
    T = len(xs)
    h = np.zeros_like(p["b"]) if h0 is None else h0.copy()
    hs = [h]
    for t in range(T):                       # forward, keeping the whole tape
        h = np.tanh(p["a"] * h + p["W"] @ xs[t] + p["b"])
        hs.append(h)

    g = {k: np.zeros_like(v) for k, v in p.items()}
    dh = np.zeros_like(p["b"])
    loss = 0.0
    for t in reversed(range(T)):
        err = p["V"] @ hs[t + 1] - targets[t]
        loss += 0.5 * float(err @ err)
        g["V"] += np.outer(err, hs[t + 1])
        dh = dh + p["V"].T @ err             # output path + future recurrence
        dz = dh * (1.0 - hs[t + 1] ** 2)
        g["a"] += dz * hs[t]
        g["W"] += np.outer(dz, xs[t])
        g["b"] += dz
        dh = p["a"] * dz                     # diagonal Jacobian: elementwise
    return loss, g


def rtrl_grads(p, xs, targets, h0=None):
    """Online path: forward only, no stored activations.

    Carries three sensitivity buffers -- one per recurrent parameter block,
    each the same shape as the parameter it tracks:

        S_a[i]    = dh_t[i] / da[i]
        S_W[i, j] = dh_t[i] / dW[i, j]
        S_b[i]    = dh_t[i] / db[i]

    Total live memory is O(params), independent of T. Each is a decaying trace
    with per-unit decay `a`, driven by the local presynaptic term -- the
    reward-modulated-Hebbian form the RPU tiles implement, derived rather than
    postulated.
    """
    T = len(xs)
    h = np.zeros_like(p["b"]) if h0 is None else h0.copy()
    S_a = np.zeros_like(p["a"])
    S_W = np.zeros_like(p["W"])
    S_b = np.zeros_like(p["b"])

    g = {k: np.zeros_like(v) for k, v in p.items()}
    loss = 0.0
    for t in range(T):
        h_prev = h
        h = np.tanh(p["a"] * h_prev + p["W"] @ xs[t] + p["b"])
        dphi = 1.0 - h ** 2                              # tanh'

        # advance sensitivities: decay by `a`, add this step's local drive
        S_a = dphi * (h_prev + p["a"] * S_a)
        S_W = dphi[:, None] * (xs[t][None, :] + p["a"][:, None] * S_W)
        S_b = dphi * (1.0 + p["a"] * S_b)

        # the only non-local quantity: error projected back to the state
        err = p["V"] @ h - targets[t]
        loss += 0.5 * float(err @ err)
        gh = p["V"].T @ err

        g["V"] += np.outer(err, h)
        g["a"] += gh * S_a
        g["W"] += gh[:, None] * S_W
        g["b"] += gh * S_b
    return loss, g


def rtrl_state_bytes(p, dtype_bytes=4):
    """Live training memory for the online path (traces + params), in bytes."""
    recurrent = p["a"].size + p["W"].size + p["b"].size
    return (n_params(p) + recurrent) * dtype_bytes


def bptt_state_bytes(p, T, dtype_bytes=4):
    """Live training memory for the reference path: params + the whole tape."""
    return (n_params(p) + (T + 1) * p["b"].size) * dtype_bytes
