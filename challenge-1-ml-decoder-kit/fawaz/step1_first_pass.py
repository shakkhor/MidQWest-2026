"""Step 1: how often does relay-BP's first quick pass already solve a shot?

Relay-BP first runs one pass of 80 BP iterations. Only if that pass does not find a correction
that explains the detectors ("does not converge") does it start up to 300 more attempts.
This script runs ONLY the first pass on the same shots as run_baselines.py (seed 1) and
compares with the full relay-BP results saved there.

  python step1_first_pass.py                    # p = 0.003, 0.004, 0.005, 10,000 shots each
  python step1_first_pass.py --p 0.004 --shots 1000

Writes results/first_pass_p<p>.npz with, per shot (shots x 2, columns X and Z):
  solved   first pass converged: its correction explains every detector
  failed   its correction gets a logical observable wrong
  seconds  decode time
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import _kit  # noqa: F401,E402  (makes the kit's circuit/baseline/decoders importable)
from baseline import memory_experiments, split_shots
from circuit import GROSS_CIRCUITS
from decoders import relay_decoder
from run_baselines import unsplit

RESULTS = Path(__file__).parent / "results"


def first_pass_slice(experiments):
    """Decode one slice of shots with relay-BP's first pass only (num_sets=0)."""
    import time
    cols = []
    for ex in experiments:
        decoder = relay_decoder(ex.H, ex.priors, num_sets=0)
        n = len(ex.det)
        solved, failed, seconds = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n)
        for k, (syndrome, truth) in enumerate(zip(ex.det, ex.obs)):
            t0 = time.perf_counter()
            result = decoder.decode_detailed(syndrome)
            seconds[k] = time.perf_counter() - t0
            error = np.asarray(result.decoding, np.uint8)
            solved[k] = result.success
            failed[k] = ((ex.L @ error) % 2 != truth).any()
        cols.append((solved, failed, seconds))
    return [np.column_stack(x) for x in zip(*cols)]


def run(p, shots, workers):
    experiments = memory_experiments([GROSS_CIRCUITS["X"], GROSS_CIRCUITS["Z"]], p, shots, seed=1)
    slices = split_shots(experiments, workers)
    with Pool(workers) as pool:
        parts = pool.map(first_pass_slice, slices)
    solved, failed, seconds = (unsplit([part[i] for part in parts], shots, workers) for i in range(3))
    np.savez(RESULTS / f"first_pass_p{p}.npz", p=p, shots=shots, solved=solved, failed=failed, seconds=seconds)
    return solved, failed, seconds


def report(p, solved, failed, seconds):
    """Compare the first pass with full relay-BP on the same shots (one row = one X or Z decode)."""
    base = np.load(RESULTS / f"baseline_p{p}.npz")
    n = len(solved)
    if n != int(base["shots"]):
        # Stim draws different shots for different shot counts, so only equal counts are the same shots.
        print(f"\np = {p}: first pass solves {solved.mean():.1%} of decodes, {failed[solved].sum()} wrong among those. "
              f"(No comparison with the baselines: they used {int(base['shots'])} shots, so the shots differ.)")
        return
    s, f, t = solved.ravel(), failed.ravel(), seconds.ravel() * 1e3
    full_f = base["relay-bp_failed"][:n].ravel()
    full_t = base["relay-bp_seconds"][:n].ravel() * 1e3
    bposd_f = base["bposd_failed"][:n].ravel()
    bposd_t = base["bposd_seconds"][:n].ravel() * 1e3
    easy, hard = s, ~s

    print(f"\n=== p = {p}: {s.size} decodes ({n} shots x X and Z) ===")
    print(f"First pass solves {easy.mean():.1%} of decodes (EASY); {hard.mean():.1%} need more work (HARD).")
    print(f"\nEASY decodes ({easy.sum()}):")
    print(f"  first pass:     {f[easy].sum():4d} wrong   mean {t[easy].mean():6.1f} ms")
    print(f"  full relay-BP:  {full_f[easy].sum():4d} wrong   mean {full_t[easy].mean():6.1f} ms")
    print(f"\nHARD decodes ({hard.sum()}):  this is where a new decoder has to win")
    print(f"  first pass alone:    {f[hard].sum():4d} wrong   mean {t[hard].mean():6.1f} ms  (its answer doesn't explain the detectors)")
    print(f"  full relay-BP:       {full_f[hard].sum():4d} wrong   mean {full_t[hard].mean():6.1f} ms   p99 {np.percentile(full_t[hard], 99):6.0f} ms")
    print(f"  BP-OSD:              {bposd_f[hard].sum():4d} wrong   mean {bposd_t[hard].mean():6.1f} ms   p99 {np.percentile(bposd_t[hard], 99):6.0f} ms")
    share = full_t[hard].sum() / full_t.sum()
    print(f"\nHard decodes are {hard.mean():.0%} of decodes but {share:.0%} of relay-BP's total time "
          f"and {full_f[hard].sum()}/{full_f.sum()} of its mistakes.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=float, nargs="+", default=[0.003, 0.004, 0.005])
    ap.add_argument("--shots", type=int, default=10_000)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    args = ap.parse_args()
    for p in args.p:
        report(p, *run(p, args.shots, args.workers))


if __name__ == "__main__":
    main()
