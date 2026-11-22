"""Does TCBP give OSD a better column ordering? (Section 3.1 of the TCBP-JOBS paper, on the kit.)

Same shots, same BP (10 sum-product iterations), OSD-0 on the non-converged shots; only the
ordering changes: BP's final LLR, BP's minimum LLR over iterations, or TCBP.

  python eval_ordering.py --p 0.004 --shots 2000
"""
import argparse

import numpy as np

from circuit import GROSS_CIRCUITS, dem_matrices, detector_ids, load_circuit, sample
from gf2 import osd0, pack_columns, pack_rows
from tcbp import TCBP
from trajbp import bp_trajectories, tanner


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--model", default="tcbp_model.npz")
    args = parser.parse_args()
    model = TCBP(args.model)
    print(f"{args.shots} shots per basis at p={args.p}; OSD-0 failures [X Z] and median K*")
    results = {}
    for k, basis in enumerate("XZ"):
        circuit = load_circuit(GROSS_CIRCUITS[basis], args.p)
        H, L, priors = dem_matrices(circuit, basis)
        det, obs = sample(circuit, args.shots, args.seed + k)
        det = det[:, detector_ids(circuit, basis)]
        T, cols, S = tanner(H), pack_columns(H), pack_rows(det)
        for start in range(0, args.shots, 250):
            sl = slice(start, start + 250)
            traj, conv, dec = bp_trajectories(T, det[sl], priors)
            orderings = {"final LLR": traj[:, -1], "min LLR": traj.min(axis=1),
                         "TCBP": model.scores(traj, priors)}
            for name, score in orderings.items():
                fails, ks = results.setdefault((name, basis), ([], []))
                for b in range(len(traj)):
                    e = dec[b].copy()
                    if not conv[b]:
                        idx, kstar = osd0(cols, np.argsort(score[b], kind="stable"), S[sl][b], H.shape[0])
                        e[:] = 0
                        e[idx] = 1
                        ks.append(kstar)
                    fails.append(bool(((L @ e) % 2 != obs[sl][b]).any()))
    for name in ("final LLR", "min LLR", "TCBP"):
        fx, kx = results[(name, "X")]
        fz, kz = results[(name, "Z")]
        print(f"{name:10s} failures [{sum(fx)} {sum(fz)}]  pair fail {np.mean(np.array(fx) | np.array(fz)):.4f}"
              f"  median K* {np.median(kx + kz):.0f}")


if __name__ == "__main__":
    main()
