"""Step 4: the full decoder, run on exactly the same 10,000 shots as the baselines.

  python step4_pipeline.py                         # p = 0.003, 0.004, 0.005 (background: ~30 min)
  python step4_pipeline.py --p 0.004 --limit 200   # quick trial on the first 200 hard decodes

Every decode (one X or one Z shot) starts with relay-BP's first pass (80 BP iterations).
If that solves the shot ("easy", 64-97% of decodes), we stop. Otherwise ("hard") one of:

  no ML  | BP-OSD           : the kit's BP-OSD
  no ML  | relay-BP         : the rest of relay-BP (it restarts, so its first pass runs twice)
  ML     | 16 orderings     : our model ranks the faults; 16 OSD-0 solves, one on the model's
                              ranking and 15 on randomly shuffled ("jittered") copies of it;
                              keep the answer that is most likely under the original priors
  ML     | 4 + relay-BP     : 4 orderings; if all 4 answers agree on the logical result accept,
                              otherwise hand the shot to relay-BP
  ML     | 16 + relay-BP    : the same with 16 orderings (slower, more accurate)

Easy-shot results are reused from results/first_pass_p<p>.npz (same shots, same decoder).
Each fallback's time includes everything it runs, including re-running BP to record its history.
Writes results/pipeline_p<p>.npz; make_report.py turns them into the table and plots.
"""
import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np
from ldpc import BpOsdDecoder

import _kit  # noqa: F401,E402  (makes the kit's circuit/baseline/decoders importable)
from baseline import memory_experiments
from bp_trajectory import TrajectoryBP
from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit
from decoders import RELAY_SETTINGS, bposd_decoder, per_cycle_rate, relay_decoder
from tcbp import features

RESULTS = Path(__file__).parent / "results"
FALLBACKS = ["BP-OSD", "relay-BP", "ML 16 orderings", "ML 4 + relay-BP", "ML 16 + relay-BP"]
DRAWS = 16
_cache = {}


def tools(p, basis):
    """Decoders for one noise level and basis, built once per worker process."""
    key = (p, basis)
    if key not in _cache:
        H, L, priors = dem_matrices(load_circuit(GROSS_CIRCUITS[basis], p), basis)
        _cache[key] = dict(
            H=H, L=L, priors=priors, cost=np.log((1 - priors) / priors),
            bp=TrajectoryBP(H, priors, alpha=RELAY_SETTINGS["alpha"], gamma=RELAY_SETTINGS["gamma0"]),
            bposd=bposd_decoder(H, priors),
            relay=relay_decoder(H, priors),
            # BP with check messages scaled to ~0 leaves the posteriors equal to the probabilities
            # we hand in, so OSD sorts the faults exactly by our ranking.
            osd0=BpOsdDecoder(H, error_channel=list(priors), max_iter=1, bp_method="minimum_sum",
                              ms_scaling_factor=1e-9, osd_method="OSD_0"),
            model=joblib.load(RESULTS / "tcbp_model.joblib"),
        )
    return _cache[key]


def llr_to_prob(llr):
    return np.clip(1 / (1 + np.exp(np.clip(llr, -50, 50))), 1e-9, 1 - 1e-9)


def ml_orderings(t, syndrome, seed):
    """Run the model and DRAWS OSD-0 solves. Returns per draw: (logical class, cost), and the time
    after each draw (so 'stop after 4 draws' can be read off the same run)."""
    start = time.perf_counter()
    history, _ = t["bp"].run(syndrome, RELAY_SETTINGS["pre_iter"])
    score = -t["model"].logit(features(history, t["bp"].prior_llr))
    rng = np.random.default_rng(seed)
    classes, costs, elapsed = [], [], []
    for d in range(DRAWS):
        jitter = rng.gumbel(size=len(score)) if d else 0.0
        t["osd0"].update_channel_probs(llr_to_prob(score + jitter))
        error = np.asarray(t["osd0"].decode(syndrome), np.uint8)
        classes.append(tuple((t["L"] @ error) % 2))
        costs.append(t["cost"][error.astype(bool)].sum())
        elapsed.append(time.perf_counter() - start)
    return classes, np.array(costs), np.array(elapsed)


def decode_hard(job):
    """All fallbacks on a list of hard decodes. Returns (failed, seconds), each (n, len(FALLBACKS))."""
    p, basis, syndromes, truths, seeds = job
    t = tools(p, basis)
    n = len(syndromes)
    failed, seconds = np.zeros((n, len(FALLBACKS)), bool), np.zeros((n, len(FALLBACKS)))
    for i, (s, truth) in enumerate(zip(syndromes, truths)):
        wrong = lambda error: bool(((t["L"] @ np.asarray(error, np.uint8)) % 2 != truth).any())
        for k, name in ((0, "bposd"), (1, "relay")):
            start = time.perf_counter()
            failed[i, k] = wrong(t[name].decode(s))
            seconds[i, k] = time.perf_counter() - start
        relay_failed, relay_seconds = failed[i, 1], seconds[i, 1]

        classes, costs, elapsed = ml_orderings(t, s, seeds[i])
        for k, draws, route in ((2, DRAWS, False), (3, 4, True), (4, DRAWS, True)):
            best = classes[int(np.argmin(costs[:draws]))]
            agree = len(set(classes[:draws])) == 1
            if route and not agree:
                failed[i, k] = relay_failed
                seconds[i, k] = elapsed[draws - 1] + relay_seconds
            else:
                failed[i, k] = bool((np.array(best) != truth).any())
                seconds[i, k] = elapsed[draws - 1]
    return failed, seconds


def run(p, workers, limit=None):
    experiments = memory_experiments([GROSS_CIRCUITS["X"], GROSS_CIRCUITS["Z"]], p, 10_000, seed=1)
    first = np.load(RESULTS / f"first_pass_p{p}.npz")
    hard = np.argwhere(~first["solved"])  # (shot, basis column) of every hard decode
    if limit:
        hard = hard[:limit]
    jobs = []
    for chunk in np.array_split(np.arange(len(hard)), workers * 6):
        for col, basis in enumerate("XZ"):
            rows = hard[chunk][hard[chunk][:, 1] == col, 0]
            if len(rows):
                ex = experiments[col]
                jobs.append(((p, basis, ex.det[rows], ex.obs[rows], rows * 2 + col), rows, col))
    t0 = time.time()
    with Pool(workers) as pool:
        results = pool.map(decode_hard, [j[0] for j in jobs])
    print(f"p={p}: {len(hard)} hard decodes x {len(FALLBACKS)} fallbacks in {time.time() - t0:.0f}s", flush=True)

    shape = first["solved"].shape + (len(FALLBACKS),)
    hard_failed, hard_seconds = np.zeros(shape, bool), np.full(shape, np.nan)
    for (_, rows, col), (f, s) in zip(jobs, results):
        hard_failed[rows, col], hard_seconds[rows, col] = f, s
    out = RESULTS / f"pipeline_p{p}{'_trial' if limit else ''}.npz"
    np.savez(out, fallbacks=FALLBACKS, hard_failed=hard_failed, hard_seconds=hard_seconds)
    return out


def combine(p, path=None):
    """Per decoder: (name, failed, seconds) over all shots (shots x 2), easy and hard put together."""
    first = np.load(RESULTS / f"first_pass_p{p}.npz")
    base = np.load(RESULTS / f"baseline_p{p}.npz")
    pipe = np.load(path or RESULTS / f"pipeline_p{p}.npz")
    solved, fp_failed, fp_seconds = first["solved"], first["failed"], first["seconds"]
    rows = [("BP-OSD (baseline)", base["bposd_failed"], base["bposd_seconds"]),
            ("relay-BP (baseline)", base["relay-bp_failed"], base["relay-bp_seconds"])]
    for k, name in enumerate(pipe["fallbacks"]):
        failed = np.where(solved, fp_failed, pipe["hard_failed"][..., k])
        seconds = np.where(solved, fp_seconds, fp_seconds + pipe["hard_seconds"][..., k])
        rows.append((f"first pass + {name}", failed, seconds))
    done = solved | ~np.isnan(pipe["hard_seconds"][..., 0])
    return rows, done.all(axis=1), solved


def report(p, path=None):
    rows, pairs, solved = combine(p, path)
    print(f"\n=== p = {p}: {pairs.sum()} X+Z pairs ===")
    print(f"  {'decoder':32s} {'pair fails':>10s} {'per-cycle rate':>20s} {'mean ms':>8s} {'median':>7s} {'p99 ms':>7s}")
    for name, failed, seconds in rows:
        f_pair = failed[pairs].any(axis=1)
        ms = seconds[pairs].sum(axis=1) * 1e3
        rate, err = per_cycle_rate(f_pair, 12)
        print(f"  {name:32s} {f_pair.sum():10d} {rate:11.2e} +- {err:.1e} {ms.mean():8.1f} {np.median(ms):7.1f} "
              f"{np.percentile(ms, 99):7.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=float, nargs="+", default=[0.003, 0.004, 0.005])
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--limit", type=int, help="only the first N hard decodes (quick trial)")
    args = ap.parse_args()
    for p in args.p:
        report(p, run(p, args.workers, args.limit))


if __name__ == "__main__":
    main()
