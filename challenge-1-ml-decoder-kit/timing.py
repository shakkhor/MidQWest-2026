"""Latency benchmark on an otherwise idle machine: every decoder on the same shots, one after another.

  python timing.py --p 0.004 --shots 500                 # TCBP-JOBS uses all cores for its draws
  NUMBA_NUM_THREADS=1 python timing.py --p 0.004 --only tcbp-jobs --tag 1thread

Times one X shot plus one Z shot per pair (as baseline.py). Writes results/timing_{tag}_p{p}.txt.
"""
import argparse
import os
import time

import numpy as np

from baseline import memory_experiments
from circuit import GROSS_CIRCUITS
from decoders import bposd_decoder, relay_decoder


def time_decoder(name, experiments, model):
    seconds = np.zeros((len(experiments[0].det), 2))
    for k, exp in enumerate(experiments):
        if name == "tcbp-jobs":
            from decoder import TcbpJobsDecoder
            dec = TcbpJobsDecoder(exp.H, exp.L, exp.priors, model_path=model)
            dec.decode(exp.det[0])                                    # compile / warm up
        elif name == "two-stage":
            from decoder import TwoStageDecoder
            dec = TwoStageDecoder(exp.H, exp.L, exp.priors, model_path=model, tau=15.0)
            dec.decode(exp.det[0])
        else:
            dec = {"relay-bp": relay_decoder, "bposd": bposd_decoder}[name](exp.H, exp.priors)
        for i, syndrome in enumerate(exp.det):
            start = time.perf_counter()
            dec.decode(syndrome)
            seconds[i, k] = time.perf_counter() - start
    return seconds.sum(axis=1) * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--shots", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--only", nargs="*", default=["two-stage", "tcbp-jobs", "relay-bp", "bposd"])
    parser.add_argument("--tag", default="idle")
    parser.add_argument("--model", default="tcbp_model.npz")
    args = parser.parse_args()
    experiments = memory_experiments([GROSS_CIRCUITS["X"], GROSS_CIRCUITS["Z"]], args.p, args.shots, args.seed)
    threads = os.environ.get("NUMBA_NUM_THREADS", str(os.cpu_count()))
    lines = [f"{args.shots} pairs at p={args.p} (seed {args.seed}), one decoder at a time on an idle machine,"
             f" numba threads {threads}"]
    for name in args.only:
        ms = time_decoder(name, experiments, args.model)
        lines.append(f"{name:10s} mean {ms.mean():7.1f} ms  median {np.median(ms):7.1f} ms  p90 {np.percentile(ms, 90):7.1f} ms"
                     f"  p99 {np.percentile(ms, 99):7.0f} ms  max {ms.max():7.0f} ms")
        print(lines[-1], flush=True)
    os.makedirs("results", exist_ok=True)
    with open(f"results/timing_{args.tag}_p{args.p}.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
