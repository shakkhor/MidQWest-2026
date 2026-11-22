"""Step 2: can we see BP's round-by-round guesses, and do they carry useful information?

  python step2_trajectories.py            # p = 0.004, X memory

1. Labelled shots. The kit's data only says which logical qubits flipped. To train a model on
   individual faults we draw the faults directly from the decoding problem (H, priors), as the
   NVIDIA paper does, so every shot comes with the true fault list. We check this gives the
   same detector statistics as the kit's circuit simulation.
2. Trajectories. On HARD shots (relay-BP's first pass does not solve them) we record every
   fault's belief after each of the 80 BP iterations.
3. Signal. OSD works down a ranked list of faults, so the true faults should be near the top.
   We compare where the true faults land when ranking by BP's final belief (what BP-OSD uses)
   versus simple summaries of the whole trajectory. If summaries already rank them higher,
   a learned model has something to learn.
Writes results/step2_trajectories.png.
"""
import argparse
from pathlib import Path

import numpy as np

from bp_trajectory import TrajectoryBP
import _kit  # noqa: F401,E402  (makes the kit's circuit/baseline/decoders importable)
from circuit import GROSS_CIRCUITS, dem_matrices, detector_ids, load_circuit, sample
from decoders import RELAY_SETTINGS

RESULTS = Path(__file__).parent / "results"


def sample_faults(H, L, priors, shots, seed):
    """Draw every fault independently with its prior. Returns (faults, syndromes, observables)."""
    rng = np.random.default_rng(seed)
    faults = (rng.random((shots, len(priors))) < priors).astype(np.uint8)
    syndromes = (H @ faults.T % 2).T.astype(np.uint8)
    observables = (L @ faults.T % 2).T.astype(np.uint8)
    return faults, syndromes, observables


def ranks(scores, true_faults):
    """Position of each true fault when faults are sorted most-likely-first (lowest score first)."""
    order = np.argsort(scores, kind="stable")
    position = np.empty_like(order)
    position[order] = np.arange(len(order))
    return position[true_faults]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=float, default=0.004)
    ap.add_argument("--hard-shots", type=int, default=200)
    args = ap.parse_args()

    circuit = load_circuit(GROSS_CIRCUITS["X"], args.p)
    H, L, priors = dem_matrices(circuit, "X")
    print(f"Decoding problem at p={args.p}: {H.shape[0]} detectors, {H.shape[1]} possible faults")

    # ------------------------------------------------------------------ 1. labelled shots
    print("\n1. Are fault-level samples equivalent to the kit's circuit samples?")
    det_c, obs_c = sample(circuit, 5000, seed=99)
    det_c = det_c[:, detector_ids(circuit, "X")]
    faults, det_f, obs_f = sample_faults(H, L, priors, 5000, seed=99)
    rate_c, rate_f = det_c.mean(0), det_f.mean(0)
    print(f"   detector firing rate     circuit {rate_c.mean():.4f}   fault-sampled {rate_f.mean():.4f}")
    print(f"   per-detector agreement   correlation {np.corrcoef(rate_c, rate_f)[0, 1]:.3f} (1.0 = identical pattern)")
    print(f"   detectors firing / shot  circuit {det_c.sum(1).mean():.1f}   fault-sampled {det_f.sum(1).mean():.1f}")
    print(f"   faults per shot          {faults.sum(1).mean():.1f} on average")

    # ------------------------------------------------------------------ 2. trajectories
    print(f"\n2. Recording 80-iteration trajectories until {args.hard_shots} hard shots are found...")
    bp = TrajectoryBP(H, priors, alpha=RELAY_SETTINGS["alpha"], gamma=RELAY_SETTINGS["gamma0"])
    iters = RELAY_SETTINGS["pre_iter"]
    hard, seen = [], 0
    for e, s in zip(faults, det_f):
        seen += 1
        history, converged_at = bp.run(s, iters)
        if converged_at < 0:
            hard.append((history, np.flatnonzero(e)))
        if len(hard) == args.hard_shots:
            break
    print(f"   {len(hard)} of {seen} shots were hard ({len(hard) / seen:.1%}); step 1 measured ~14% at p=0.004")
    print(f"   each hard shot gives a {iters} x {H.shape[1]} table of beliefs")

    # ------------------------------------------------------------------ 3. signal
    print("\n3. Where do the TRUE faults land in each ranking? (lower = nearer the top = easier for OSD)")
    summaries = {
        "final belief (what BP-OSD uses)": lambda h: h[-1],
        "lowest belief over all rounds": lambda h: h.min(0),
        "average belief over all rounds": lambda h: h.mean(0),
    }
    for name, summary in summaries.items():
        r = np.concatenate([ranks(summary(h), true) for h, true in hard])
        deepest = np.array([ranks(summary(h), true).max() for h, true in hard])
        print(f"   {name:34s} median rank {np.median(r):5.0f}   in top 100: {np.mean(r < 100):5.1%}   "
              f"in top 500: {np.mean(r < 500):5.1%}   deepest true fault, median: {np.median(deepest):5.0f}")
    print("   (rank 0 = the fault BP is most sure about. There are 8784 faults in total.)")

    plot(hard, priors)


def plot(hard, priors):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # the hard shot whose true faults are ranked worst by the final belief: the interesting case
    worst = max(hard, key=lambda hs: np.median(ranks(hs[0][-1], hs[1])))
    history, true = worst
    false = np.setdiff1d(np.argsort(history[-1])[:40], true)[:6]  # confident but wrong
    fig, ax = plt.subplots(figsize=(9, 5))
    for j in true[:8]:
        ax.plot(history[:, j], color="#0f7b6c", lw=1.4, label="true fault" if j == true[0] else None)
    for j in false:
        ax.plot(history[:, j], color="#c2410c", lw=1.2, ls="--", label="BP's wrong favourite" if j == false[0] else None)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("BP iteration")
    ax.set_ylabel("belief (LLR): below 0 = 'this fault happened'")
    ax.set_title("One hard shot: true faults vs. faults BP wrongly favours at the end")
    ax.legend()
    fig.tight_layout()
    out = RESULTS / "step2_trajectories.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
