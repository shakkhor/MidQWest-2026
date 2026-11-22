"""Run the kit's baselines and save every shot's result, so we can analyse them later.

  python run_baselines.py --p 0.004 --shots 10000

Same shots, decoders and scoring as baseline.py (it reuses its functions). In addition it
writes results/baseline_p<p>.npz with, for each decoder:
  <name>_failed   (shots x 2) bool, did the X / Z shot fail
  <name>_seconds  (shots x 2) decode time of the X / Z shot
Shot k in the file is shot k of the sampled data, so other decoders can be compared shot by shot.
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import _kit  # noqa: F401,E402  (makes the kit's circuit/baseline/decoders importable)
from baseline import BASELINES, memory_experiments, score_slice, split_shots
from circuit import GROSS_CIRCUITS
from decoders import print_score


def unsplit(parts, shots, pieces):
    """Undo split_shots: slice k holds shots k, k + pieces, ...; put them back in order."""
    out = np.empty((shots, parts[0].shape[1]), parts[0].dtype)
    for k, part in enumerate(parts):
        out[k::pieces] = part
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=float, default=0.004)
    ap.add_argument("--shots", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--decoders", nargs="+", default=list(BASELINES))
    args = ap.parse_args()

    experiments = memory_experiments([GROSS_CIRCUITS["X"], GROSS_CIRCUITS["Z"]], args.p, args.shots, args.seed)
    slices = split_shots(experiments, args.workers)
    print(f"{args.shots} shots at p={args.p}, seed {args.seed}, failures listed as [X Z]", flush=True)

    saved = {}
    with Pool(args.workers) as pool:
        for name in args.decoders:
            results = pool.map(score_slice, [(name, s) for s in slices])
            failed = unsplit([r[0] for r in results], args.shots, args.workers)
            seconds = unsplit([r[1] for r in results], args.shots, args.workers)
            print_score(name, failed, seconds, 12)
            saved[f"{name}_failed"], saved[f"{name}_seconds"] = failed, seconds

    out = Path(__file__).parent / "results" / f"baseline_p{args.p}.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez(out, p=args.p, shots=args.shots, seed=args.seed, **saved)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
