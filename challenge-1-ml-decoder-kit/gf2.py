"""GF(2) linear algebra for OSD with an arbitrary column ordering (numba, bit-packed).

osd0 walks the columns of H in the given order, inserting each into an incremental echelon
basis, until the syndrome lies in the span. That prefix length is K* of the TCBP-JOBS paper.
The OSD-0 correction uses only the basis columns.
"""
import numba as nb
import numpy as np
import scipy.sparse as sp


def pack_columns(H):
    """(n_vars, words) uint64: column j of H as a bit vector over the checks."""
    Hc = sp.csc_matrix(H)
    words = (Hc.shape[0] + 63) // 64
    out = np.zeros((Hc.shape[1], words), np.uint64)
    for j in range(Hc.shape[1]):
        for r in Hc.indices[Hc.indptr[j]:Hc.indptr[j + 1]]:
            out[j, r >> 6] |= np.uint64(1) << np.uint64(r & 63)
    return out


def pack_rows(bits):
    """(B, n) 0/1 -> (B, words) uint64, bit i of a row in word i // 64."""
    bits = np.asarray(bits, np.uint8)
    words = (bits.shape[1] + 63) // 64
    padded = np.zeros((bits.shape[0], words * 64), np.uint8)
    padded[:, :bits.shape[1]] = bits
    # little bit order + little-endian uint64 view: bit i of word w is bits[64 w + i]
    return np.ascontiguousarray(np.packbits(padded, axis=1, bitorder="little")).view("<u8").astype(np.uint64)


@nb.njit(cache=True)
def _lowest_bit(v):
    for w in range(v.shape[0]):
        x = v[w]
        if x != 0:
            b = 0
            while (x >> np.uint64(b)) & np.uint64(1) == 0:
                b += 1
            return w * 64 + b
    return -1


@nb.njit(cache=True)
def _test(v, bit):
    return (v[bit >> 6] >> np.uint64(bit & 63)) & np.uint64(1)


@nb.njit(cache=True)
def osd0(cols, order, s, max_rank):
    """OSD-0 in the given column order.

    cols (n_vars, W) packed columns, order (n_vars,) column indices most-likely first,
    s (W,) packed syndrome. Returns (correction column indices, K*); K* = -1 when the
    syndrome is not in the span of all columns.
    Every basis vector's pivot is its lowest set bit, so reducing a new column only touches the
    basis vectors whose pivots it hits (not all of them).
    """
    W = cols.shape[1]
    CW = (max_rank + 63) // 64
    vec = np.zeros((max_rank, W), np.uint64)
    combo = np.zeros((max_rank, CW), np.uint64)
    owner = -np.ones(W * 64, np.int64)
    basis_col = np.zeros(max_rank, np.int64)
    res = s.copy()
    res_combo = np.zeros(CW, np.uint64)
    R = 0
    if _lowest_bit(res) < 0:
        return np.zeros(0, np.int64), 0
    v = np.zeros(W, np.uint64)
    c = np.zeros(CW, np.uint64)
    for k in range(order.shape[0]):
        j = order[k]
        v[:] = cols[j]
        c[:] = 0
        p = _lowest_bit(v)
        while p >= 0 and owner[p] >= 0:
            o = owner[p]
            for w in range(W):
                v[w] ^= vec[o, w]
            for w in range(CW):
                c[w] ^= combo[o, w]
            p = _lowest_bit(v)
        if p < 0:
            continue
        c[R >> 6] ^= np.uint64(1) << np.uint64(R & 63)
        vec[R] = v
        combo[R] = c
        owner[p] = R
        basis_col[R] = j
        R += 1
        # reduce the syndrome residual by the new basis vector chain
        q = _lowest_bit(res)
        while q >= 0 and owner[q] >= 0:
            o = owner[q]
            for w in range(W):
                res[w] ^= vec[o, w]
            for w in range(CW):
                res_combo[w] ^= combo[o, w]
            q = _lowest_bit(res)
        if q < 0:
            n = 0
            for i in range(R):
                if _test(res_combo, i):
                    n += 1
            out = np.empty(n, np.int64)
            n = 0
            for i in range(R):
                if _test(res_combo, i):
                    out[n] = basis_col[i]
                    n += 1
            return out, k + 1
    return np.zeros(0, np.int64), -1
