"""Option A data: per-shot features and the outcome of every stage-3 variant, for router v2.

  python -m bench.collect_hard --p 0.004 --seed 1000 --shots 10000 --workers 10   # training
  python -m bench.collect_hard --p 0.004 --seed 1    --shots 10000 --workers 4    # test (timed)

For each shot: stage 1 (first relay leg). For each hard shot (stage 1 not converged):
  stage 2   relay stop_nconv=1, with the router features of its solution
  stage 3a  4 more solutions (stop_nconv=4), keep the most likely candidate
  stage 3b  1 more solution from a differently seeded relay run, keep the most likely
Each stage is timed separately, so any router and either stage-3 variant can be replayed
offline (bench.router_v2). Shots are dealt as in bench.run_decoder.
"""
import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from bench.make_shots import BASES, load_shots
from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
TAUS = (0.5, 1.0, 2.0, 4.0)


def rich_features(syndrome, r1, r2, llr):
    """Router v2 inputs from the failed first leg (r1) and the stage-2 solution (r2)."""
    error = np.asarray(r2.decoding, bool)
    weight = float(llr[error].sum())
    synd = float(np.count_nonzero(syndrome))
    post = np.abs(np.asarray(r2.posterior_ratios, float))
    post1 = np.abs(np.asarray(r1.posterior_ratios, float))
    smallest = np.sort(post)[:10]
    unsat1 = int(np.count_nonzero(np.asarray(r1.decoded_detectors, np.uint8) != syndrome))
    f = {"f_synd": synd, "f_log_iter": np.log1p(r2.iterations), "f_weight": weight,
         "f_weight_per_synd": weight / max(synd, 1.0), "f_nflip": float(error.sum()),
         "f_legs": max(0.0, (r2.iterations - 80) / 60), "f_unsat_after_leg1": unsat1,
         "f_post_min": float(post.min()), "f_post_small10": float(smallest.mean()),
         "f_post_flipped_min": float(post[error].min()) if error.any() else 0.0,
         "f_post_flipped_mean": float(post[error].mean()) if error.any() else 0.0,
         "f_leg1_post_small10": float(np.sort(post1)[:10].mean())}
    for tau in TAUS:
        f[f"f_post_below_{tau}"] = float((post < tau).sum())
    return f


def _timed(fn, *args):
    start = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - start


def worker(job):
    p, seed, shots, k, workers = job
    from fast_decoder import _relay
    rows = []
    for basis in BASES:
        H, L, priors = dem_matrices(load_circuit(GROSS_CIRCUITS[basis], p), basis)
        llr = np.log((1 - priors) / priors)
        det, obs = load_shots(p, basis, seed, shots)
        first_leg = _relay(H, priors, num_sets=0, stop_nconv=1)
        stage2 = _relay(H, priors, stop_nconv=1)
        more4 = _relay(H, priors, stop_nconv=4)
        more1 = _relay(H, priors, stop_nconv=1, seed=12345)
        wrong = lambda e, truth: bool(((L @ np.asarray(e, np.uint8)) % 2 != truth).any())
        best = lambda a, b: a if llr[a].sum() <= llr[b].sum() else b
        for shot in range(k, len(det), workers):
            syndrome, truth = det[shot], obs[shot]
            r1, t1 = _timed(first_leg.decode_detailed, syndrome)
            row = {"shot": shot, "basis": basis, "t1": t1, "hard": not r1.success, "fail1": wrong(r1.decoding, truth),
                   "t2": 0.0, "t3a": 0.0, "t3b": 0.0}
            if r1.success:
                row |= {"fail2": row["fail1"], "fail3a": row["fail1"], "fail3b": row["fail1"]}
            else:
                r2, t2 = _timed(stage2.decode_detailed, syndrome)
                feats, tf = _timed(rich_features, syndrome, r1, r2, llr)
                mine = np.asarray(r2.decoding, bool)
                a, t3a = _timed(more4.decode, syndrome)
                b, t3b = _timed(more1.decode, syndrome)
                row |= feats | {"t2": t2 + tf, "t3a": t3a, "t3b": t3b, "fail2": wrong(mine, truth),
                                "fail3a": wrong(best(mine, np.asarray(a, bool)), truth),
                                "fail3b": wrong(best(mine, np.asarray(b, bool)), truth)}
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", type=float, nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    LOG_DIR.mkdir(exist_ok=True)
    for p in args.p:
        with Pool(args.workers) as pool:
            parts = pool.map(worker, [(p, args.seed, args.shots, k, args.workers) for k in range(args.workers)])
        log = pd.DataFrame([r for part in parts for r in part]).sort_values(["basis", "shot"])
        log.insert(2, "p", p)
        out = LOG_DIR / f"hard_p{p}_seed{args.seed}_n{args.shots}_w{args.workers}.csv"
        log.to_csv(out, index=False)
        hard = log[log.hard]
        print(f"{out.name}: hard {len(hard)} of {len(log)}, stage-2 wrong {int(hard.fail2.sum())}, "
              f"3a fixes {int((hard.fail2 & ~hard.fail3a).sum())} breaks {int((~hard.fail2 & hard.fail3a).sum())}, "
              f"3b fixes {int((hard.fail2 & ~hard.fail3b).sum())} breaks {int((~hard.fail2 & hard.fail3b).sum())}, "
              f"mean t3a {hard.t3a.mean()*1e3:.0f} ms t3b {hard.t3b.mean()*1e3:.0f} ms")


if __name__ == "__main__":
    main()
