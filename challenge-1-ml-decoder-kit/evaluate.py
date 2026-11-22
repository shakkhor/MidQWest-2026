"""Decode the baseline's shots with TCBP-JOBS (or a baseline) and save per-shot records.

The shots are exactly those of baseline.py with the same --p, --shots and --seed: X memory is
sampled with seed, Z memory with seed + 1. Records are saved per shot so that decoders can be
compared shot by shot and combined offline (routing, joint decisions).

  python evaluate.py --decoder tcbp-jobs --p 0.004 --shots 10000 --out results/tcbpjobs_p0.004.npz
  python evaluate.py --decoder relay-bp  --p 0.004 --shots 10000 --workers 10 --out results/relay_p0.004.npz

Saved arrays have shape (shots, 2), one column per memory basis [X, Z]:
  failed, seconds, cost (prior log-likelihood cost of the correction), pred (observable flips as int)
TCBP-JOBS adds converged, gap (complementary gap) and kstar.
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from baseline import Experiment, memory_experiments
from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit, sample
from decoders import bposd_decoder, per_cycle_rate, relay_decoder
from jobs import logical_columns


def full_dem_experiments(paths, p, shots, seed):
    """Same shots as memory_experiments, decoded from all detectors with the full DEM."""
    out = []
    for k, path in enumerate(paths):
        circuit = load_circuit(path, p)
        det, obs = sample(circuit, shots, seed + k)
        out.append(Experiment(*dem_matrices(circuit), det, obs))
    return out


def _records(name, exp, first, step, settings):
    """Decode shots first, first + step, ... of one experiment. Returns a dict of arrays."""
    det, obs = exp.det[first::step], exp.obs[first::step]
    w = np.log((1 - exp.priors) / exp.priors)
    obs_weights = 1 << np.arange(exp.L.shape[0])
    if name == "tcbp-jobs":
        from decoder import TcbpJobsDecoder
        dec = TcbpJobsDecoder(exp.H, exp.L, exp.priors, **settings)
        decode = dec.decode_full
    elif name == "two-stage":
        from decoder import TwoStageDecoder
        dec = TwoStageDecoder(exp.H, exp.L, exp.priors, model_path=settings["model_path"],
                              stage1={"draws": settings["s1_draws"], "beam": settings["s1_beam"],
                                      "depth": settings["s1_depth"], "patience": 4})
        decode = dec.decode_full
    else:
        base = {"bposd": bposd_decoder, "relay-bp": relay_decoder}[name](exp.H, exp.priors)
        decode = lambda s: (base.decode(s), {})
    out = {k: [] for k in ("failed", "seconds", "cost", "pred", "converged", "gap", "kstar", "failed_coset", "margin",
                           "failed1", "gap1", "t1", "t2")}
    for syndrome, truth in zip(det, obs):
        start = time.perf_counter()
        e, info = decode(syndrome)
        seconds = time.perf_counter() - start
        e = np.asarray(e, np.uint8)
        pred = (exp.L @ e) % 2
        out["failed"].append(bool((pred != truth).any()))
        out["seconds"].append(seconds)
        out["cost"].append(float(w[e.astype(bool)].sum()))
        out["pred"].append(int(pred @ obs_weights))
        out["converged"].append(info.get("converged", False))
        out["gap"].append(info.get("gap", np.nan))
        out["kstar"].append(info.get("kstar", -1))
        # class-sum (coset) decision of JOBS; the min-cost class where there is none
        coset = info.get("coset")
        truth_int = int(truth @ obs_weights)
        out["failed_coset"].append(out["failed"][-1] if coset is None else coset != truth_int)
        out["margin"].append(info.get("margin", np.nan))
        # two-stage: stage 1 alone, its gap and both stage times (tau is tuned offline in analyze.py)
        e1 = info.get("e1")
        out["failed1"].append(out["failed"][-1] if e1 is None else bool(((exp.L @ np.asarray(e1, np.uint8)) % 2 != truth).any()))
        out["gap1"].append(info.get("gap1", np.nan))
        out["t1"].append(info.get("t1", np.nan))
        out["t2"].append(info.get("t2", np.nan))
    return {k: np.array(v) for k, v in out.items()}


def _job(args):
    return _records(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decoder", default="tcbp-jobs", choices=["tcbp-jobs", "two-stage", "relay-bp", "bposd"])
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, help="processes (tcbp-jobs parallelises draws itself)")
    parser.add_argument("--model", default="tcbp_model.npz")
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--beam", type=int, default=32)
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--s1-draws", type=int, default=3)
    parser.add_argument("--s1-beam", type=int, default=8)
    parser.add_argument("--s1-depth", type=int, default=15)
    parser.add_argument("--full-dem", action="store_true",
                        help="decode all 1728 detectors with the full DEM (X/Z correlations kept)")
    parser.add_argument("--limit", type=int, default=None,
                        help="decode only the first LIMIT of the --shots sampled shots (same shots as the sweep)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    settings = {"model_path": args.model, "draws": args.draws, "beam": args.beam, "depth": args.depth,
                "alpha": args.alpha} if args.decoder == "tcbp-jobs" else (
        {"model_path": args.model, "s1_draws": args.s1_draws, "s1_beam": args.s1_beam, "s1_depth": args.s1_depth}
        if args.decoder == "two-stage" else {})
    paths = [GROSS_CIRCUITS["X"], GROSS_CIRCUITS["Z"]]
    experiments = (full_dem_experiments(paths, args.p, args.shots, args.seed) if args.full_dem
                   else memory_experiments(paths, args.p, args.shots, args.seed))
    if args.limit:
        experiments = [e._replace(det=e.det[:args.limit], obs=e.obs[:args.limit]) for e in experiments]
        args.shots = args.limit
    start = time.time()
    columns = []
    for exp in experiments:
        if args.workers == 1:
            columns.append(_records(args.decoder, exp, 0, 1, settings))
        else:
            with Pool(args.workers) as pool:
                parts = pool.map(_job, [(args.decoder, exp, k, args.workers, settings) for k in range(args.workers)])
            merged = {}
            for key in parts[0]:
                arr = np.empty(args.shots, parts[0][key].dtype)
                for k, part in enumerate(parts):
                    arr[k::args.workers] = part[key]
                merged[key] = arr
            columns.append(merged)
    rec = {k: np.column_stack([c[k] for c in columns]) for k in columns[0]}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **rec, p=args.p, shots=args.shots, seed=args.seed, decoder=args.decoder)
    rate, err = per_cycle_rate(rec["failed"].any(axis=1), 12)
    ms = rec["seconds"].sum(axis=1) * 1e3
    print(f"{args.decoder:10s} p={args.p} failures {rec['failed'].sum(axis=0)} of {args.shots}  per cycle "
          f"{rate:.2e} +- {err:.1e}  mean {ms.mean():.1f} ms  p99 {np.percentile(ms, 99):.0f} ms  "
          f"(wall {time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
