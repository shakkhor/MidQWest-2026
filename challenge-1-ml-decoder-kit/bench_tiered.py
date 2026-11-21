"""Compare relay-BP with the tiered decoder on the same shots as baseline.py.

  python bench_tiered.py --p 0.004 --shots 10000 --router router.npz

With the same --p, --shots, --seed and --workers as baseline.py, the shots are identical, so
the relay-bp line reproduces baseline.py and the tiered lines can be compared shot for shot.
Use --skip-relay to leave out the slow relay-BP run and compare with a saved baseline result.
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np

from baseline import memory_experiments, split_shots
from circuit import GROSS_CIRCUITS
from decoders import print_score, relay_decoder, score
from fast_decoder import Router, TieredDecoder


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    parser.add_argument("--router", help="router file from train_router.py; without it only tiered-fast runs")
    parser.add_argument("--skip-relay", action="store_true")
    return parser.parse_args()


def make_decoder(name, H, priors, router_path):
    if name == "relay-bp":
        return relay_decoder(H, priors)
    return TieredDecoder(H, priors, Router.load(router_path) if name == "tiered-ml" else None)


def score_slice(job):
    """failed, seconds (one column per basis) and the stage counts of one slice."""
    name, experiments, router_path = job
    failed, seconds, stages = [], [], np.zeros(3, int)
    for e in experiments:
        decoder = make_decoder(name, e.H, e.priors, router_path)
        f, s = score(decoder.decode, e.L, e.det, e.obs)
        failed.append(f), seconds.append(s)
        stages += getattr(decoder, "stages", 0)
    return np.column_stack(failed), np.column_stack(seconds), stages


def main():
    args = parse_args()
    experiments = memory_experiments([GROSS_CIRCUITS["X"], GROSS_CIRCUITS["Z"]], args.p, args.shots, args.seed)
    slices = split_shots(experiments, args.workers)
    names = ([] if args.skip_relay else ["relay-bp"]) + ["tiered-fast"] + (["tiered-ml"] if args.router else [])
    print(f"{args.shots} shots at p={args.p}, {args.cycles} cycles, failures listed as [X Z]")
    with Pool(args.workers) as pool:
        for name in names:
            results = pool.map(score_slice, [(name, s, args.router) for s in slices])
            failed, seconds, stages = zip(*results, strict=True)
            print_score(name, np.concatenate(failed), np.concatenate(seconds), args.cycles)
            stages = np.sum(stages, axis=0)
            if stages.sum():
                share = stages / stages.sum()
                print(f"{'':10s} decodes ended at stage 1/2/3: {share[0]:.1%} / {share[1]:.1%} / {share[2]:.1%}")


if __name__ == "__main__":
    main()