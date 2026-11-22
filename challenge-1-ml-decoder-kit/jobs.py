"""JOBS: jittered OSD beam search (after NVIDIA's TCBP-JOBS paper), re-implemented on CPU with numba.

For a shot where BP does not converge:
  1. Order the columns of H by the TCBP score. K* = shortest prefix whose span holds the syndrome.
  2. Make D orderings: draw 0 is the TCBP order, draws 1..D-1 add Gumbel noise (scale beta) to the
     scores. Each draw keeps its first M = ceil(alpha K*) columns (its window).
  3. In every draw: eliminate the window in order -> basis + OSD-0 correction, then beam search
     (width W, depth L, patience P) over syndrome-preserving moves: flip one free column and
     compensate on the basis.
  4. Return the cheapest correction over all draws, by the prior log-likelihood cost.
The search also records the cheapest cost seen in every logical class; the gap between the two
cheapest classes is a per-shot confidence ("complementary gap").

The draws run in parallel threads (numba prange).
"""
import math

import numba as nb
import numpy as np

from gf2 import _lowest_bit, osd0, pack_columns, pack_rows

U1 = np.uint64(1)
INF = np.inf


@nb.njit(cache=True, inline="always")
def _popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@nb.njit(cache=True)
def _bits(words, out):
    """Write the indices of the set bits of `words` into out; return how many."""
    n = 0
    for t in range(words.shape[0]):
        x = words[t]
        while x != 0:
            low = x & (~x + U1)
            out[n] = t * 64 + _popcount(low - U1)
            n += 1
            x ^= low
    return n


@nb.njit(cache=True, inline="always")
def _splitmix(x):
    z = x + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@nb.njit(cache=True)
def _visit(table, used, h):
    """Insert h into an open-addressing set. Returns False when it was already there."""
    size = np.int64(table.shape[0])
    k = np.int64(h >> np.uint64(40)) % size
    while used[k]:
        if table[k] == h:
            return False
        k = (k + np.int64(1)) % size
    used[k] = True
    table[k] = h
    return True


@nb.njit(cache=True)
def jobs_draw(cols, window, s, w, lcol, n_classes, beam_w, depth, patience, max_rank):
    """One draw: eliminate `window` in order, OSD-0, beam search.

    Returns (cost, correction (n,) uint8, class_best (n_classes,) float64). cost is inf when
    the window does not span the syndrome.
    """
    n, W = cols.shape
    CW = (max_rank + 63) // 64
    M = window.shape[0]
    class_best = np.full(n_classes, INF)
    e_out = np.zeros(n, np.uint8)

    # --- incremental elimination; every basis vector's pivot is its lowest set bit
    owner = -np.ones(W * 64, np.int64)
    vec = np.zeros((max_rank, W), np.uint64)
    combo = np.zeros((max_rank, CW), np.uint64)
    basis_col = np.zeros(max_rank, np.int64)
    free_col = np.zeros(M, np.int64)
    free_combo = np.zeros((M, CW), np.uint64)
    R = 0
    nf = 0
    v = np.zeros(W, np.uint64)
    c = np.zeros(CW, np.uint64)
    for k in range(M):
        j = window[k]
        v[:] = cols[j]
        c[:] = 0
        p = _lowest_bit(v)
        while p >= 0 and owner[p] >= 0:
            o = owner[p]
            for t in range(W):
                v[t] ^= vec[o, t]
            for t in range(CW):
                c[t] ^= combo[o, t]
            p = _lowest_bit(v)
        if p < 0:
            free_col[nf] = j
            free_combo[nf] = c
            nf += 1
        else:
            c[R >> 6] ^= U1 << np.uint64(R & 63)
            vec[R] = v
            combo[R] = c
            owner[p] = R
            basis_col[R] = j
            R += 1
    res = s.copy()
    rc = np.zeros(CW, np.uint64)
    p = _lowest_bit(res)
    while p >= 0:
        o = owner[p]
        if o < 0:
            return INF, e_out, class_best, np.zeros(1, np.uint64), np.zeros(1, np.int64), np.zeros(1), 0
        for t in range(W):
            res[t] ^= vec[o, t]
        for t in range(CW):
            rc[t] ^= combo[o, t]
        p = _lowest_bit(res)

    # --- free-column moves x_f (basis indices), and the inverse incidence
    tmp = np.zeros(CW * 64, np.int64)
    xptr = np.zeros(nf + 1, np.int64)
    for f in range(nf):
        cnt = 0
        for t in range(CW):
            cnt += _popcount(free_combo[f, t])
        xptr[f + 1] = xptr[f] + cnt
    xidx = np.zeros(xptr[nf], np.int64)
    for f in range(nf):
        _bits(free_combo[f], xidx[xptr[f]:xptr[f + 1]])
    deg = np.zeros(R + 1, np.int64)
    for q in range(xidx.shape[0]):
        deg[xidx[q] + 1] += 1
    iptr = np.cumsum(deg)
    fill = iptr[:-1].copy()
    iidx = np.zeros(xidx.shape[0], np.int64)
    for f in range(nf):
        for q in range(xptr[f], xptr[f + 1]):
            i = xidx[q]
            iidx[fill[i]] = f
            fill[i] += 1

    wb = np.empty(R)
    lb = np.empty(R, np.int64)
    for i in range(R):
        wb[i] = w[basis_col[i]]
        lb[i] = lcol[basis_col[i]]
    wf = np.empty(nf)
    of = np.empty(nf, np.int64)
    z = np.empty(nf, np.uint64)
    for f in range(nf):
        wf[f] = w[free_col[f]]
        o = lcol[free_col[f]]
        for q in range(xptr[f], xptr[f + 1]):
            o ^= lb[xidx[q]]
        of[f] = o
        z[f] = _splitmix(np.uint64(free_col[f]) * np.uint64(0x100000001B3) + np.uint64(7))
    # global hash of a correction (same in every draw): xor of one key per column
    zb = np.empty(R, np.uint64)
    for i in range(R):
        zb[i] = _splitmix(np.uint64(basis_col[i]) * np.uint64(0x9E3779B1) + np.uint64(13))
    hz = np.empty(nf, np.uint64)
    for f in range(nf):
        hh = _splitmix(np.uint64(free_col[f]) * np.uint64(0x9E3779B1) + np.uint64(13))
        for q in range(xptr[f], xptr[f + 1]):
            hh ^= zb[xidx[q]]
        hz[f] = hh

    # --- OSD-0 correction = beam state 0
    bx = np.zeros((beam_w, R), np.bool_)
    bF = np.zeros((beam_w, nf), np.bool_)
    bd = np.zeros((beam_w, nf))
    bc = np.zeros(beam_w)
    bh = np.zeros(beam_w, np.uint64)
    bo = np.zeros(beam_w, np.int64)
    bg = np.zeros(beam_w, np.uint64)
    m = _bits(rc, tmp)
    cost0 = 0.0
    obs0 = 0
    g0 = np.uint64(0)
    for q in range(m):
        bx[0, tmp[q]] = True
        cost0 += wb[tmp[q]]
        obs0 ^= lb[tmp[q]]
        g0 ^= zb[tmp[q]]
    bg[0] = g0
    cap = beam_w * depth + 1
    rec_h = np.zeros(cap, np.uint64)
    rec_c = np.zeros(cap, np.int64)
    rec_cost = np.zeros(cap)
    rec_h[0] = g0
    rec_c[0] = obs0
    rec_cost[0] = cost0
    nrec = 1
    for f in range(nf):
        d = wf[f]
        for q in range(xptr[f], xptr[f + 1]):
            i = xidx[q]
            d += -wb[i] if bx[0, i] else wb[i]
        bd[0, f] = d
    bc[0] = cost0
    bo[0] = obs0
    nbeam = 1
    class_best[obs0] = cost0
    best_cost = cost0
    best_x = bx[0].copy()
    best_F = bF[0].copy()

    size = 1
    while size < 4 * (beam_w * depth + 2):
        size *= 2
    table = np.zeros(size, np.uint64)
    used = np.zeros(size, np.bool_)
    _visit(table, used, np.uint64(0))

    nx = np.zeros((beam_w, R), np.bool_)
    nF = np.zeros((beam_w, nf), np.bool_)
    nd = np.zeros((beam_w, nf))
    nc = np.zeros(beam_w)
    nh = np.zeros(beam_w, np.uint64)
    no = np.zeros(beam_w, np.int64)
    ng = np.zeros(beam_w, np.uint64)
    stall = 0
    if nf > 0:
        for step in range(depth):
            ncand = nbeam * nf
            cc = np.empty(ncand)
            for b in range(nbeam):
                for f in range(nf):
                    val = bc[b] + bd[b, f]
                    cc[b * nf + f] = val
                    cls = bo[b] ^ of[f]
                    if val < class_best[cls]:
                        class_best[cls] = val
            K = min(4 * beam_w, ncand)
            thr = np.partition(cc.copy(), K - 1)[K - 1]
            sel = np.where(cc <= thr)[0]
            sel = sel[np.argsort(cc[sel])]
            acc = 0
            for rnd in range(2):
                if rnd == 1:
                    if acc >= beam_w or K >= ncand:
                        break
                    sel = np.argsort(cc)
                for q in range(sel.shape[0]):
                    if acc >= beam_w:
                        break
                    idx = sel[q]
                    b = idx // nf
                    f = idx - b * nf
                    h = bh[b] ^ z[f]
                    if not _visit(table, used, h):
                        continue
                    nx[acc] = bx[b]
                    nF[acc] = bF[b]
                    nd[acc] = bd[b]
                    if nF[acc, f]:
                        nd[acc, f] += 2 * wf[f]
                    else:
                        nd[acc, f] -= 2 * wf[f]
                    nF[acc, f] = not nF[acc, f]
                    for qq in range(xptr[f], xptr[f + 1]):
                        i = xidx[qq]
                        g_old = -wb[i] if nx[acc, i] else wb[i]
                        for r in range(iptr[i], iptr[i + 1]):
                            nd[acc, iidx[r]] -= 2 * g_old
                        nx[acc, i] = not nx[acc, i]
                    nc[acc] = bc[b] + bd[b, f]
                    nh[acc] = h
                    no[acc] = bo[b] ^ of[f]
                    ng[acc] = bg[b] ^ hz[f]
                    if nrec < cap:
                        rec_h[nrec] = ng[acc]
                        rec_c[nrec] = no[acc]
                        rec_cost[nrec] = nc[acc]
                        nrec += 1
                    acc += 1
            if acc == 0:
                break
            improved = False
            for a in range(acc):
                if nc[a] < best_cost - 1e-9:
                    best_cost = nc[a]
                    best_x[:] = nx[a]
                    best_F[:] = nF[a]
                    improved = True
            bx, nx = nx, bx
            bF, nF = nF, bF
            bd, nd = nd, bd
            bc, nc = nc, bc
            bh, nh = nh, bh
            bo, no = no, bo
            bg, ng = ng, bg
            nbeam = acc
            stall = 0 if improved else stall + 1
            if stall >= patience:
                break

    for i in range(R):
        if best_x[i]:
            e_out[basis_col[i]] = 1
    for f in range(nf):
        if best_F[f]:
            e_out[free_col[f]] = 1
    return best_cost, e_out, class_best, rec_h, rec_c, rec_cost, nrec


@nb.njit(cache=True, parallel=True)
def jobs_draws(cols, windows, s, w, lcol, n_classes, beam_w, depth, patience, max_rank):
    """All draws of one shot in parallel. windows (D, M)."""
    D = windows.shape[0]
    cap = beam_w * depth + 1
    costs = np.empty(D)
    es = np.zeros((D, cols.shape[0]), np.uint8)
    cbs = np.empty((D, n_classes))
    rh = np.zeros((D, cap), np.uint64)
    rc = np.zeros((D, cap), np.int64)
    rcost = np.full((D, cap), np.inf)
    for d in nb.prange(D):
        cst, e, cb, h, c, cs, nr = jobs_draw(cols, windows[d], s, w, lcol, n_classes, beam_w, depth, patience, max_rank)
        costs[d] = cst
        es[d] = e
        cbs[d] = cb
        rh[d, :nr] = h[:nr]
        rc[d, :nr] = c[:nr]
        rcost[d, :nr] = cs[:nr]
    return costs, es, cbs, rh, rc, rcost


def coset_decision(rh, rc, rcost):
    """Most likely logical class from the distinct corrections visited by all draws.

    P(class) is approximated by the sum of exp(-cost) over the distinct visited corrections of
    that class (deduplicated by their global hash). Returns (class, log-odds against the runner-up).
    """
    ok = np.isfinite(rcost)
    h, c, cost = rh[ok], rc[ok], rcost[ok]
    _, first = np.unique(h, return_index=True)
    c, cost = c[first], cost[first]
    classes, inv = np.unique(c, return_inverse=True)
    m = cost.min()
    weight = np.bincount(inv, weights=np.exp(-(cost - m)))
    order = np.argsort(-weight)
    margin = np.log(weight[order[0]] / weight[order[1]]) if len(order) > 1 else np.inf
    return int(classes[order[0]]), float(margin)


def logical_columns(L):
    """Observable flips of every column of L packed into one int per column."""
    Lc = L.tocsc()
    out = np.zeros(Lc.shape[1], np.int64)
    for j in range(Lc.shape[1]):
        for r in Lc.indices[Lc.indptr[j]:Lc.indptr[j + 1]]:
            out[j] |= 1 << int(r)
    return out


class JOBS:
    """JOBS over one decoding problem (H, L, priors). Search settings default to the paper's."""

    def __init__(self, H, L, priors, draws=16, beta=1.0, alpha=8.0, beam=32, depth=40, patience=8):
        self.H, self.L, self.priors = H, L, np.asarray(priors, np.float64)
        self.cols = pack_columns(H)
        self.w = np.log((1 - self.priors) / self.priors)
        self.lcol = logical_columns(L)
        self.n_classes = 1 << L.shape[0]
        self.draws, self.beta, self.alpha = draws, beta, alpha
        self.beam, self.depth, self.patience = beam, depth, patience

    def windows(self, score, syndrome_packed):
        """(D, M) column windows of one shot, and K*. Noise is seeded from the syndrome."""
        n = len(score)
        order0 = np.argsort(score, kind="stable")
        _, kstar = osd0(self.cols, order0, syndrome_packed, self.H.shape[0])
        if kstar <= 0:
            return None, kstar
        M = min(n, math.ceil(self.alpha * kstar))
        rng = np.random.default_rng(int.from_bytes(syndrome_packed.tobytes()[:8] + syndrome_packed.tobytes()[-8:], "little")
                                    ^ int(syndrome_packed.sum() % (1 << 63)))
        out = np.empty((self.draws, M), np.int64)
        out[0] = order0[:M]
        if self.draws > 1:
            noisy = score[None] + self.beta * rng.gumbel(size=(self.draws - 1, n))
            top = np.argpartition(noisy, M - 1, axis=1)[:, :M]
            top_scores = np.take_along_axis(noisy, top, axis=1)
            out[1:] = np.take_along_axis(top, np.argsort(top_scores, axis=1, kind="stable"), axis=1)
        return out, kstar

    def search(self, score, syndrome, w=None):
        """(correction, cost, gap, K*) for one non-converged shot. syndrome is 0/1 over checks.

        w overrides the per-fault costs log((1 - p) / p) for this shot (per-shot priors, e.g. heralds).
        """
        s = pack_rows(syndrome[None])[0]
        windows, kstar = self.windows(score, s)
        if windows is None:
            raise ValueError("syndrome is not in the span of H")
        w = self.w if w is None else np.asarray(w, np.float64)
        costs, es, cbs, rh, rc, rcost = jobs_draws(self.cols, windows, s, w, self.lcol, self.n_classes,
                                                   self.beam, self.depth, self.patience, self.H.shape[0])
        best = int(np.argmin(costs))
        class_best = cbs.min(axis=0)
        two = np.partition(class_best, 1)[:2]
        gap = float(two[1] - two[0]) if np.isfinite(two[1]) else np.inf
        self.last_coset = coset_decision(rh, rc, rcost)
        return es[best], float(costs[best]), gap, kstar

    def cost(self, e):
        return float(self.w[np.asarray(e, bool)].sum())
