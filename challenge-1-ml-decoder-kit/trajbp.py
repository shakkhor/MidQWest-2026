"""Batched sum-product BP that records every fault's marginal LLR at every iteration.

This is the BP front end of TCBP (trajectory-calibrated BP, NVIDIA's TCBP-JOBS paper):
plain flooding sum-product on the syndrome, with message clipping, for a fixed number of
iterations. Unlike a normal BP decoder it keeps the whole trajectory of marginals, which the
TCBP model turns into calibrated fault probabilities.
"""
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp


class Tanner(NamedTuple):
    """Edge lists of H, sorted by check and by variable, for vectorised message passing."""
    n_checks: int
    n_vars: int
    check_of_edge: np.ndarray   # check index of every edge (edges in check order)
    var_of_edge: np.ndarray     # variable index of every edge (edges in check order)
    check_starts: np.ndarray    # reduceat offsets of each check's edges
    var_order: np.ndarray       # permutation: check-ordered edges -> variable-ordered edges
    var_starts: np.ndarray      # reduceat offsets of each variable's edges (variable order)


def tanner(H):
    """Build the edge structure of a 0/1 sparse matrix H (checks x variables)."""
    Hr = sp.csr_matrix(H)
    Hr.sort_indices()
    check_of_edge = np.repeat(np.arange(Hr.shape[0]), np.diff(Hr.indptr))
    var_of_edge = Hr.indices.astype(np.int64)
    var_order = np.argsort(var_of_edge, kind="stable")
    counts = np.bincount(var_of_edge, minlength=Hr.shape[1])
    if (counts == 0).any():
        raise ValueError("every column of H needs at least one nonzero")
    var_starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return Tanner(Hr.shape[0], Hr.shape[1], check_of_edge, var_of_edge, Hr.indptr[:-1].copy(),
                  var_order, var_starts)


def bp_trajectories(T, syndromes, priors, iters=10, clip=15.0):
    """Run sum-product BP on a batch of syndromes.

    syndromes: (B, n_checks) 0/1. priors: (n_vars,) or (B, n_vars) fault probabilities.
    Returns (traj, converged, decision):
      traj      (B, iters, n_vars) float32 marginal LLRs after each iteration
      converged (B,) bool, BP's hard decision satisfied the syndrome at some iteration
      decision  (B, n_vars) uint8, the first syndrome-satisfying hard decision
                (the last iteration's decision where BP never converged)
    """
    syndromes = np.asarray(syndromes, np.uint8)
    B = len(syndromes)
    priors = np.broadcast_to(np.asarray(priors, np.float64), (B, T.n_vars))
    prior_llr = np.clip(np.log((1 - priors) / priors), -clip, clip)
    sign_s = 1.0 - 2.0 * syndromes[:, T.check_of_edge]          # (B, E): -1 where the check fired

    v2c = prior_llr[:, T.var_of_edge]                             # variable-to-check messages
    traj = np.empty((B, iters, T.n_vars), np.float32)
    converged = np.zeros(B, bool)
    decision = np.zeros((B, T.n_vars), np.uint8)
    for t in range(iters):
        c2v = _check_update(T, v2c, sign_s, clip)
        incoming = np.add.reduceat(c2v[:, T.var_order], T.var_starts, axis=1)   # (B, n_vars)
        marginal = prior_llr + incoming
        traj[:, t] = marginal
        hard = (marginal < 0).astype(np.uint8)
        ok = _satisfies(T, hard, syndromes) & ~converged
        decision[ok] = hard[ok]
        converged |= ok
        v2c = np.clip(marginal[:, T.var_of_edge] - c2v, -clip, clip)
    decision[~converged] = hard[~converged]
    return traj, converged, decision


def _check_update(T, v2c, sign_s, clip):
    """Check-to-variable messages by the tanh rule, computed in the log domain per check."""
    mag = np.abs(v2c)
    # log|tanh(x/2)|, guarded at 0 so a zero message does not give -inf
    log_t = np.log(np.maximum(np.tanh(np.maximum(mag, 1e-12) / 2), 1e-300))
    neg = (v2c < 0)
    sum_log = np.add.reduceat(log_t, T.check_starts, axis=1)[:, T.check_of_edge]
    n_neg = np.add.reduceat(neg.astype(np.int32), T.check_starts, axis=1)[:, T.check_of_edge]
    ext_log = sum_log - log_t                                   # product over the other edges
    ext_sign = np.where((n_neg - neg) % 2 == 1, -1.0, 1.0)
    prod = np.exp(np.minimum(ext_log, 0.0))
    msg = 2 * np.arctanh(np.minimum(prod, np.tanh(clip / 2)))
    return np.clip(sign_s * ext_sign * msg, -clip, clip)


def _satisfies(T, hard, syndromes):
    """Rows where H @ hard == syndrome (mod 2)."""
    parity = np.add.reduceat(hard[:, T.var_of_edge].astype(np.int32), T.check_starts, axis=1) % 2
    return (parity == syndromes).all(axis=1)


# ---------------------------------------------------------------- numba kernels (same maths)
import numba as nb  # noqa: E402


@nb.njit(cache=True, inline="always")
def _bp_iteration(check_starts, var_of_edge, var_order, var_starts, n_vars, prior_llr, sgn, v2c, c2v,
                  marginal, buf, clip, tclip, c_lo, c_hi, v_lo, v_hi):
    E = v2c.shape[0]
    for c in range(c_lo, c_hi):
        a = check_starts[c]
        b = check_starts[c + 1] if c + 1 < check_starts.shape[0] else E
        # forward/backward products of tanh(m/2), excluding each edge itself
        acc = 1.0
        for k in range(a, b):
            buf[k] = acc
            acc *= np.tanh(0.5 * v2c[k])
        acc = 1.0
        for k in range(b - 1, a - 1, -1):
            prod = buf[k] * acc
            acc *= np.tanh(0.5 * v2c[k])
            if prod > tclip:
                prod = tclip
            elif prod < -tclip:
                prod = -tclip
            m = sgn[c] * 2.0 * np.arctanh(prod)
            c2v[k] = m
    for v in range(v_lo, v_hi):
        a = var_starts[v]
        b = var_starts[v + 1] if v + 1 < n_vars else E
        tot = prior_llr[v]
        for q in range(a, b):
            tot += c2v[var_order[q]]
        marginal[v] = tot
        for q in range(a, b):
            k = var_order[q]
            x = tot - c2v[k]
            v2c[k] = clip if x > clip else (-clip if x < -clip else x)


@nb.njit(cache=True)
def _parity_ok(check_starts, var_of_edge, hard, syn):
    E = var_of_edge.shape[0]
    for c in range(check_starts.shape[0]):
        a = check_starts[c]
        b = check_starts[c + 1] if c + 1 < check_starts.shape[0] else E
        par = 0
        for k in range(a, b):
            par ^= hard[var_of_edge[k]]
        if par != syn[c]:
            return False
    return True


@nb.njit(cache=True, parallel=True)
def bp_batch_nb(check_starts, var_of_edge, var_order, var_starts, n_vars, syndromes, prior_llr, iters, clip):
    """Batched sum-product BP; threads over shots. prior_llr is (B, n_vars)."""
    B = syndromes.shape[0]
    E = var_of_edge.shape[0]
    traj = np.empty((B, iters, n_vars), np.float32)
    conv = np.zeros(B, np.bool_)
    dec = np.zeros((B, n_vars), np.uint8)
    tclip = np.tanh(clip / 2)
    for s in nb.prange(B):
        sgn = 1.0 - 2.0 * syndromes[s].astype(np.float64)
        v2c = np.empty(E)
        for k in range(E):
            v2c[k] = prior_llr[s, var_of_edge[k]]
        c2v = np.empty(E)
        buf = np.empty(E)
        marginal = np.empty(n_vars)
        hard = np.zeros(n_vars, np.uint8)
        for t in range(iters):
            _bp_iteration(check_starts, var_of_edge, var_order, var_starts, n_vars, prior_llr[s], sgn, v2c, c2v,
                          marginal, buf, clip, tclip, 0, check_starts.shape[0], 0, n_vars)
            for v in range(n_vars):
                traj[s, t, v] = marginal[v]
                hard[v] = 1 if marginal[v] < 0 else 0
            if not conv[s] and _parity_ok(check_starts, var_of_edge, hard, syndromes[s]):
                conv[s] = True
                dec[s] = hard
        if not conv[s]:
            dec[s] = hard
    return traj, conv, dec


@nb.njit(cache=True, parallel=True)
def bp_single_nb(check_starts, var_of_edge, var_order, var_starts, n_vars, syndrome, prior_llr, iters, clip, chunks):
    """One shot, threads over chunks of checks and of variables (for large graphs)."""
    E = var_of_edge.shape[0]
    n_checks = check_starts.shape[0]
    traj = np.empty((1, iters, n_vars), np.float32)
    sgn = 1.0 - 2.0 * syndrome.astype(np.float64)
    v2c = np.empty(E)
    for k in range(E):
        v2c[k] = prior_llr[var_of_edge[k]]
    c2v = np.empty(E)
    buf = np.empty(E)
    marginal = np.empty(n_vars)
    hard = np.zeros(n_vars, np.uint8)
    dec = np.zeros((1, n_vars), np.uint8)
    conv = np.zeros(1, np.bool_)
    tclip = np.tanh(clip / 2)
    for t in range(iters):
        for ch in nb.prange(chunks):
            c_lo = ch * n_checks // chunks
            c_hi = (ch + 1) * n_checks // chunks
            _bp_iteration(check_starts, var_of_edge, var_order, var_starts, n_vars, prior_llr, sgn, v2c, c2v,
                          marginal, buf, clip, tclip, c_lo, c_hi, 0, 0)
        for ch in nb.prange(chunks):
            v_lo = ch * n_vars // chunks
            v_hi = (ch + 1) * n_vars // chunks
            _bp_iteration(check_starts, var_of_edge, var_order, var_starts, n_vars, prior_llr, sgn, v2c, c2v,
                          marginal, buf, clip, tclip, 0, 0, v_lo, v_hi)
        for v in range(n_vars):
            traj[0, t, v] = marginal[v]
            hard[v] = 1 if marginal[v] < 0 else 0
        if not conv[0] and _parity_ok(check_starts, var_of_edge, hard, syndrome):
            conv[0] = True
            dec[0] = hard
    if not conv[0]:
        dec[0] = hard
    return traj, conv, dec


def bp_fast(T, syndromes, priors, iters=10, clip=15.0, chunks=12):
    """Numba version of bp_trajectories with identical outputs."""
    syndromes = np.ascontiguousarray(np.atleast_2d(syndromes), np.uint8)
    B = len(syndromes)
    pri = np.asarray(priors, np.float64)
    prior_llr = np.clip(np.log((1 - pri) / pri), -clip, clip)
    args = (T.check_starts.astype(np.int64), T.var_of_edge.astype(np.int64), T.var_order.astype(np.int64),
            T.var_starts.astype(np.int64), T.n_vars)
    if B == 1 and len(T.var_of_edge) > 100_000:
        llr = prior_llr if prior_llr.ndim == 1 else prior_llr[0]
        return bp_single_nb(*args, syndromes[0], np.ascontiguousarray(llr), iters, clip, chunks)
    llr = np.ascontiguousarray(np.broadcast_to(prior_llr, (B, T.n_vars)))
    return bp_batch_nb(*args, syndromes, llr, iters, clip)
