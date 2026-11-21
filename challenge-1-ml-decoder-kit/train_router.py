"""Train the router of fast_decoder.TieredDecoder.

  python train_router.py --p 0.003 0.004 0.005 --shots 2000 --out router.npz

For every training shot whose first relay leg does not converge, it runs relay-BP with
stop_nconv=1 and with stop_nconv=5. The label is 1 when the two predict different logical
observables: the first solution was not good enough. Training uses its own seed, so keep it
different from the test seed of bench_tiered.py.
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np

from circuit import GROSS_CIRCUITS, dem_matrices, detector_ids, load_circuit, sample
from fast_decoder import Router, _relay, features


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", type=float, nargs="+", default=[0.003, 0.004, 0.005])
    parser.add_argument("--shots", type=int, default=2000, help="training shots per p and basis")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--recall", type=float, default=0.95, help="fraction of label-1 shots the router must flag")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    parser.add_argument("--out", default="router.npz")
    return parser.parse_args()


def collect(job):
    """(features, labels) of the hard shots in one slice of one (p, basis) experiment."""
    p, basis, shots, seed, first, step = job
    circuit = load_circuit(GROSS_CIRCUITS[basis], p)
    det, _ = sample(circuit, shots, seed)
    det = det[first::step, detector_ids(circuit, basis)]
    H, L, priors = dem_matrices(circuit, basis)
    llr = np.log((1 - priors) / priors)
    first_leg = _relay(H, priors, num_sets=0, stop_nconv=1)
    one, five = _relay(H, priors, stop_nconv=1), _relay(H, priors)
    X, y = [], []
    for syndrome in det:
        if first_leg.decode_detailed(syndrome).success:
            continue
        result = one.decode_detailed(syndrome)
        quick = (L @ np.asarray(result.decoding, np.uint8)) % 2
        full = (L @ np.asarray(five.decode(syndrome), np.uint8)) % 2
        X.append(features(syndrome, result, llr))
        y.append(float((quick != full).any()))
    return X, y


def main():
    args = parse_args()
    jobs = [(p, basis, args.shots, args.seed + 10 * i + j, k, args.workers)
            for i, p in enumerate(args.p) for j, basis in enumerate("XZ") for k in range(args.workers)]
    with Pool(args.workers) as pool:
        parts = pool.map(collect, jobs)
    X = np.array([x for part in parts for x in part[0]])
    y = np.array([v for part in parts for v in part[1]])
    router = Router.fit(X, y, recall=args.recall)
    flagged = np.array([router.distrust(x) for x in X])
    print(f"{len(y)} hard shots, {int(y.sum())} where the first solution disagrees with relay-BP")
    print(f"router flags {flagged.mean():.1%} of hard shots, catches {flagged[y == 1].mean():.1%} of disagreements")
    router.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()