"""Step 3: train the small network on BP histories of hard shots, and test its ranking.

  python step3_train.py                 # train on p = 0.003, 0.004, 0.005 together, then test each

Training data: fresh fault-sampled shots (seeds 1000+), kept only if relay-BP's first pass
would not solve them (hard shots), because that is the only place the model will be used.
Test data: different fresh shots (seeds 5000+), never seen in training.

The test asks: when the faults are ranked most-likely-first, how high do the TRUE faults land?
  - final belief   : what BP-OSD ranks by
  - lowest belief  : the simple hand-made trick from step 2
  - model          : our trained network
Writes results/tcbp_model.joblib and results/step3_ranking.png.
"""
import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np

from bp_trajectory import TrajectoryBP
import _kit  # noqa: F401,E402  (makes the kit's circuit/baseline/decoders importable)
from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit
from decoders import RELAY_SETTINGS
from step2_trajectories import ranks, sample_faults
from tcbp import TCBPModel, features

RESULTS = Path(__file__).parent / "results"
TOP = 1000  # "near the top": roughly how deep OSD's first solve reaches (H has rank ~ 936)
_problems = {}


def problem(p):
    """(H, L, priors, bp) for X memory at noise p, built once per worker process."""
    if p not in _problems:
        H, L, priors = dem_matrices(load_circuit(GROSS_CIRCUITS["X"], p), "X")
        bp = TrajectoryBP(H, priors, alpha=RELAY_SETTINGS["alpha"], gamma=RELAY_SETTINGS["gamma0"])
        _problems[p] = (H, L, priors, bp)
    return _problems[p]


def hard_shots(p, seed, n_hard):
    """Yield (history, true faults) for hard shots until n_hard are found."""
    H, L, priors, bp = problem(p)
    found, batch = 0, 0
    while found < n_hard:
        faults, syndromes, _ = sample_faults(H, L, priors, 200, seed * 1000 + batch)
        batch += 1
        for e, s in zip(faults, syndromes):
            history, converged_at = bp.run(s, RELAY_SETTINGS["pre_iter"], stop_when_solved=True)
            if converged_at < 0:
                yield history, np.flatnonzero(e)
                found += 1
                if found == n_hard:
                    return


def train_job(args):
    """Features of every true fault and of a random `neg_rate` share of the other faults."""
    p, seed, n_hard, neg_rate = args
    priors_llr = problem(p)[3].prior_llr
    rng = np.random.default_rng(seed)
    X, y = [], []
    for history, true in hard_shots(p, seed, n_hard):
        label = np.zeros(history.shape[1], bool)
        label[true] = True
        keep = label | (rng.random(len(label)) < neg_rate)
        X.append(features(history, priors_llr)[keep])
        y.append(label[keep])
    return np.concatenate(X), np.concatenate(y)


def test_job(args):
    """For each hard test shot and each ranking: share of true faults in the top, deepest true fault."""
    p, seed, n_hard, model = args
    priors_llr = problem(p)[3].prior_llr
    rows = []
    for history, true in hard_shots(p, seed, n_hard):
        scores = {
            "final belief (BP-OSD)": history[-1],
            "lowest belief (trick)": history.min(0),
            "model (ours)": -model.logit(features(history, priors_llr)),
        }
        rows.append({name: ranks(s, true) for name, s in scores.items()})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=float, nargs="+", default=[0.003, 0.004, 0.005])
    ap.add_argument("--train-hard", type=int, default=600, help="hard training shots per noise level")
    ap.add_argument("--test-hard", type=int, default=200, help="hard test shots per noise level")
    ap.add_argument("--neg-rate", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    args = ap.parse_args()
    W = args.workers

    with Pool(W) as pool:
        t0 = time.time()
        jobs = [(p, 1000 + 10 * i + k, args.train_hard // W, args.neg_rate) for i, p in enumerate(args.p) for k in range(W)]
        parts = pool.map(train_job, jobs)
        X = np.concatenate([x for x, _ in parts])
        y = np.concatenate([t for _, t in parts])
        print(f"Training data: {len(args.p) * (args.train_hard // W) * W} hard shots -> {len(y):,} fault examples "
              f"({y.sum():,} true faults, {(~y).sum():,} sampled others)  [{time.time() - t0:.0f}s]")

        t0 = time.time()
        model = TCBPModel().fit(X, y, args.neg_rate)
        print(f"Trained an MLP with {model.n_weights()} weights  [{time.time() - t0:.0f}s]")
        RESULTS.mkdir(exist_ok=True)
        joblib.dump(model, RESULTS / "tcbp_model.joblib")

        t0 = time.time()
        jobs = [(p, 5000 + 10 * i + k, args.test_hard // W, model) for i, p in enumerate(args.p) for k in range(W)]
        results = pool.map(test_job, jobs)
        print(f"Tested on fresh hard shots  [{time.time() - t0:.0f}s]")

    per_p = {}
    for (p, *_), rows in zip(jobs, results):
        per_p.setdefault(p, []).extend(rows)
    for p, rows in per_p.items():
        print(f"\n=== p = {p}: {len(rows)} hard test shots ===")
        print(f"  {'ranking by':24s} {'true faults in top 500':>23s} {'shots with ALL true faults in top ' + str(TOP):>40s} "
              f"{'deepest true fault (median)':>29s}")
        for name in rows[0]:
            r = [row[name] for row in rows]
            top500 = np.mean(np.concatenate(r) < 500)
            all_top = np.mean([x.max() < TOP for x in r])
            deepest = np.median([x.max() for x in r])
            print(f"  {name:24s} {top500:23.1%} {all_top:40.1%} {deepest:29.0f}")
    plot(per_p)


def plot(per_p):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(per_p), figsize=(5 * len(per_p), 4), sharey=True)
    colors = {"final belief (BP-OSD)": "#9ca3af", "lowest belief (trick)": "#d97706", "model (ours)": "#0f7b6c"}
    for ax, (p, rows) in zip(np.atleast_1d(axes), per_p.items()):
        for name, color in colors.items():
            deepest = np.sort([row[name].max() for row in rows])
            ax.plot(deepest, np.arange(1, len(deepest) + 1) / len(deepest), color=color, lw=2, label=name)
        ax.set_xscale("log")
        ax.axvline(TOP, color="grey", ls=":", lw=1)
        ax.set_title(f"p = {p}")
        ax.set_xlabel("rank of the deepest true fault (log scale)")
    np.atleast_1d(axes)[0].set_ylabel("share of hard shots")
    np.atleast_1d(axes)[0].legend(loc="lower right")
    fig.suptitle("Further left = every true fault is near the top of the list = easier for the solver")
    fig.tight_layout()
    out = RESULTS / "step3_ranking.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
