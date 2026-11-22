"""Min-sum belief propagation that records its beliefs after every iteration.

The relay-BP library only returns BP's final beliefs. NVIDIA's TCBP idea needs the whole
history ("trajectory") of every fault's belief, so this is a small numpy BP that keeps it.
It is checked against the library's own BP in test_bp_trajectory.py.

Beliefs are log-likelihood ratios (LLRs): log(P(no error) / P(error)).
  large positive -> this fault almost surely did not happen
  negative       -> BP thinks this fault happened
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class TrajectoryBP:
    """Min-sum BP on a check matrix H (detectors x faults), with optional memory (gamma).

    gamma=None is plain min-sum. With gamma, each iteration's marginal keeps a fraction gamma
    of the previous marginal instead of the prior ("memory BP", the first pass of relay-BP).
    """

    def __init__(self, H, priors, alpha=1.0, gamma=None):
        H = sp.csr_matrix(H, dtype=np.uint8)
        H.sort_indices()
        self.H = H
        self.n_checks, self.n_faults = H.shape
        self.edge_check = np.repeat(np.arange(self.n_checks), np.diff(H.indptr))  # edges grouped by check
        self.edge_fault = H.indices.copy()
        self.row_start = H.indptr[:-1]
        self.prior_llr = np.log((1 - priors) / priors)
        self.alpha = alpha
        self.gamma = gamma

    def _check_to_fault(self, q, syndrome):
        """Min-sum check update: message = (syndrome sign) * product of other signs * alpha * min of other magnitudes."""
        mag = np.abs(q)
        neg = q < 0
        # sign product per check (including the syndrome bit), via parity of negative messages
        parity = (np.add.reduceat(neg.astype(np.int64), self.row_start) + syndrome) % 2
        out_neg = (parity[self.edge_check] + neg) % 2  # remove this edge's own sign
        # smallest and second-smallest magnitude per check
        min1 = np.minimum.reduceat(mag, self.row_start)
        is_min = mag == min1[self.edge_check]
        first_min = np.zeros_like(is_min)
        idx = np.flatnonzero(is_min)
        _, keep = np.unique(self.edge_check[idx], return_index=True)
        first_min[idx[keep]] = True  # one edge per check holds the minimum
        mag2 = np.where(first_min, np.inf, mag)
        min2 = np.minimum.reduceat(mag2, self.row_start)
        other_min = np.where(first_min, min2[self.edge_check], min1[self.edge_check])
        return np.where(out_neg == 1, -1.0, 1.0) * self.alpha * other_min

    def run(self, syndrome, iterations, stop_when_solved=False):
        """Run BP. Returns (llr_history, converged_at).

        llr_history has shape (iterations, n_faults): the marginal LLR of every fault after each
        iteration. converged_at is the first iteration whose hard decision explains the syndrome,
        or -1. With stop_when_solved the history ends there.
        """
        syndrome = np.asarray(syndrome, np.int64)
        q = self.prior_llr[self.edge_fault].copy()  # fault-to-check messages start at the prior
        marginal = self.prior_llr.copy()
        history = np.empty((iterations, self.n_faults))
        converged_at = -1
        for t in range(iterations):
            r = self._check_to_fault(q, syndrome)
            incoming = np.bincount(self.edge_fault, weights=r, minlength=self.n_faults)
            if self.gamma is None:
                marginal = self.prior_llr + incoming
            else:
                marginal = (1 - self.gamma) * self.prior_llr + self.gamma * marginal + incoming
            q = marginal[self.edge_fault] - r
            history[t] = marginal
            if converged_at < 0:
                guess = (marginal < 0).astype(np.uint8)
                if np.array_equal((self.H @ guess) % 2, syndrome % 2):
                    converged_at = t
                    if stop_when_solved:
                        return history[: t + 1], converged_at
        return history, converged_at
