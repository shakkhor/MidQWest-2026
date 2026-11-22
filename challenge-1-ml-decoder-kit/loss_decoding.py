"""Loss-aware decoding problems and the loss-aware TCBP-JOBS decoder.

Heralds turn into per-shot priors over the check-only decoding matrix:
  - a fault caused by the erasure of a lost qubit (herald fired) gets its conditional
    probability given the loss, about 1/2 per basis component, so it is almost free;
  - a fault tied to a herald that did not fire is ruled out (prior ~ 0);
  - every other fault keeps its circuit prior.
Relay-BP gets the same information through herald detectors appended to H (loss-aware), or
none of it (loss-unaware, erasures marginalised into Pauli noise).

Train the loss-aware TCBP model:
  python loss_decoding.py --p 0.004 --p-loss 0.002 --shots 3000 --out tcbp_loss_model.npz
"""
import argparse
import time

import numpy as np
import scipy.sparse as sp

from jobs import JOBS
from loss import decoding_ids, dem_matrices_ids, detector_groups, loss_circuit
from tcbp import TCBP, features, fit
from trajbp import bp_fast, bp_trajectories, tanner

RULED_OUT = 1e-7
MAX_COND = 0.45


class LossProblem:
    """One memory basis of the loss circuit, as check-only H with herald-dependent priors."""

    def __init__(self, basis, p, p_loss):
        self.basis, self.p, self.p_loss = basis, p, p_loss
        self.circuit = loss_circuit(basis, p, p_loss)
        groups = detector_groups(self.circuit)
        self.ids_aware = decoding_ids(self.circuit, basis, aware=True)
        self.ids_checks = groups[basis]
        self.ids_heralds = groups["herald"]
        H_aw, L_aw, pr_aw = dem_matrices_ids(self.circuit, self.ids_aware)
        is_herald = np.isin(self.ids_aware, self.ids_heralds)
        Hc = H_aw.tocsr()[~is_herald].tocsc()
        Hh = H_aw.tocsr()[is_herald].tocsc()
        herald_count = np.diff(Hh.indptr)
        if herald_count.max() > 1:
            raise ValueError("a fault flips two heralds; the per-shot prior rule assumes one")
        herald_of = np.full(H_aw.shape[1], -1)
        cols_with = np.flatnonzero(herald_count == 1)
        herald_of[cols_with] = Hh.indices[Hh.indptr[cols_with]]
        # P(herald h fires) = sum of the priors of its faults (they are disjoint components)
        self.herald_prob = np.bincount(herald_of[herald_of >= 0], weights=pr_aw[herald_of >= 0],
                                       minlength=len(self.ids_heralds))
        keep = np.diff(Hc.indptr) > 0          # herald-only faults touch no check
        if np.asarray(L_aw.tocsc()[:, ~keep].sum()) > 0:
            raise ValueError("an undetectable fault flips an observable")
        self.keep = keep                        # aware-DEM columns -> check-only columns
        self.H = Hc[:, keep]
        self.L = L_aw.tocsc()[:, keep].tocsr()
        self.base_priors = pr_aw[keep]
        self.herald_of = herald_of[keep]
        # the relay-BP views of the same shots
        self.H_aware, self.L_aware, self.priors_aware = H_aw, L_aw, pr_aw

    def shot_priors(self, heralds):
        """(B, n) per-shot priors from (B, n_heralds) herald bits."""
        heralds = np.atleast_2d(heralds).astype(bool)
        B = len(heralds)
        pri = np.broadcast_to(self.base_priors, (B, len(self.base_priors))).copy()
        hc = np.flatnonzero(self.herald_of >= 0)
        h = self.herald_of[hc]
        cond = np.minimum(self.base_priors[hc] / self.herald_prob[h], MAX_COND)
        pri[:, hc] = np.where(heralds[:, h], cond[None], RULED_OUT)
        return pri

    def split(self, det):
        """(check syndrome, heralds) from full detector samples."""
        return det[:, self.ids_checks], det[:, self.ids_heralds]


class LossAwareTcbpJobs:
    """TCBP-JOBS with herald-dependent priors in BP, TCBP and the JOBS cost."""

    def __init__(self, problem, model_path="tcbp_loss_model.npz", iters=10, **jobs_settings):
        self.pb = problem
        self.tanner = tanner(problem.H)
        self.model = TCBP(model_path)
        self.iters = iters
        self.jobs = JOBS(problem.H, problem.L, problem.base_priors, **jobs_settings)

    def decode_full(self, syndrome, heralds):
        pri = self.pb.shot_priors(heralds)
        traj, conv, dec = bp_fast(self.tanner, np.asarray(syndrome, np.uint8)[None], pri, iters=self.iters)
        w = np.log((1 - pri[0]) / pri[0])
        if conv[0]:
            e = dec[0]
            return e, {"converged": True, "cost": float(w[e.astype(bool)].sum()), "gap": np.inf, "kstar": 0}
        score = self.model.scores(traj, pri)[0]
        e, cost, gap, kstar = self.jobs.search(score, np.asarray(syndrome, np.uint8), w=w)
        return e, {"converged": False, "cost": cost, "gap": gap, "kstar": kstar,
                   "coset": self.jobs.last_coset[0], "margin": self.jobs.last_coset[1]}


def loss_training_rows(pb, shots, rng, iters=10, neg_keep=0.05, batch=256):
    """Errors sampled straight from the loss problem: heralds, one component per lost qubit, Pauli faults."""
    T = tanner(pb.H)
    Hc = pb.H.tocsc()
    n = pb.H.shape[1]
    hc = np.flatnonzero(pb.herald_of >= 0)
    plain = np.flatnonzero(pb.herald_of < 0)
    Xs, ys, ws = [], [], []
    done = 0
    while done < shots:
        b = min(batch, shots - done)
        scale = np.exp(rng.uniform(np.log(0.5), np.log(1.5), (b, 1)))
        heralds = rng.random((b, len(pb.herald_prob))) < np.minimum(pb.herald_prob[None] * scale, 0.5)
        pri = pb.shot_priors(heralds)
        errors = np.zeros((b, n), np.uint8)
        # Pauli faults, jittered
        rates = np.clip(pri[:, plain] * scale * np.exp(rng.normal(0, 0.3, (b, len(plain)))), 1e-5, 0.1)
        errors[:, plain] = rng.random(rates.shape) < rates
        # erasures: each component fault of a fired herald with its conditional probability
        cond = pri[:, hc]
        fired = heralds[:, pb.herald_of[hc]]
        errors[:, hc] = fired & (rng.random(cond.shape) < cond)
        syndromes = (Hc @ errors.T).T % 2
        traj, conv, _ = bp_trajectories(T, syndromes, pri, iters=iters)
        keep = ~conv
        f = features(traj[keep], pri[keep]).reshape(-1, iters + 4)
        y = errors[keep].reshape(-1)
        sel = (y == 1) | (rng.random(len(y)) < neg_keep)
        Xs.append(f[sel]); ys.append(y[sel]); ws.append(np.where(y[sel] == 1, 1.0, 1 / neg_keep))
        done += b
    return np.concatenate(Xs), np.concatenate(ys).astype(np.float32), np.concatenate(ws).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--p-loss", type=float, default=0.002)
    parser.add_argument("--shots", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="tcbp_loss_model.npz")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    start = time.time()
    parts = []
    for basis in "XZ":
        pb = LossProblem(basis, args.p, args.p_loss)
        parts.append(loss_training_rows(pb, args.shots, rng))
        print(f"basis {basis}: H {pb.H.shape}, {len(parts[-1][0]):,} rows ({time.time() - start:.0f} s)")
    X, y, w = (np.concatenate(z) for z in zip(*parts))
    np.savez(args.out, **fit(X, y, w, seed=args.seed), iters=10)
    print(f"saved {args.out} ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
