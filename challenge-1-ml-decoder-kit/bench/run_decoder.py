"""Decode cached shots with one decoder config and write a per-shot CSV log.

  python -m bench.run_decoder --decoder relay --p 0.004 --shots 10000 --workers 5
  python -m bench.run_decoder --decoder relay --config stop_nconv=1,num_sets=50 --p 0.005 --shots 3000
  python -m bench.run_decoder --decoder tiered-ml --router router.npz --p 0.004

Shots are dealt to workers as baseline.py deals them (worker k gets shots k, k + W, ...;
it decodes its X shots, then its Z shots, with a fresh decoder per basis). Relay-BP carries
random state from shot to shot, so --workers 5 replays baseline.py's default run exactly.
Use --workers 4 for timed runs on the M4 (4 performance cores).

The time of a shot covers the decode() call only, single-threaded.
"""
import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from bench.make_shots import BASES, load_shots
from circuit import GROSS_CIRCUITS, dem_matrices, load_circuit
from decoders import bposd_decoder, relay_decoder

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
DECODERS = ["relay", "bposd", "tiered-fast", "tiered-ml"]


def parse_config(text):
    """'stop_nconv=1,num_sets=50' -> {'stop_nconv': 1, 'num_sets': 50}."""
    config = {}
    for item in filter(None, (text or "").split(",")):
        key, value = item.split("=")
        number = float(value)
        config[key.strip()] = int(number) if number.is_integer() and "." not in value else number
    return config


def make_decoder(name, H, priors, config, router):
    if name == "relay":
        return relay_decoder(H, priors, **config)
    if name == "bposd":
        return bposd_decoder(H, priors, **config)
    from fast_decoder import Router, TieredDecoder
    return TieredDecoder(H, priors, Router.load(router) if name == "tiered-ml" else None)


def decode_one(decoder, name, syndrome):
    """(error vector, seconds, stage, iterations) of one shot."""
    if name == "relay":
        start = time.perf_counter()
        result = decoder.decode_detailed(syndrome)
        seconds = time.perf_counter() - start
        return np.asarray(result.decoding, np.uint8), seconds, np.nan, result.iterations
    start = time.perf_counter()
    error = decoder.decode(syndrome)
    seconds = time.perf_counter() - start
    if name == "bposd":
        return np.asarray(error, np.uint8), seconds, np.nan, decoder.iter
    return np.asarray(error, np.uint8), seconds, decoder.last_stage, decoder.last_iterations


def run_worker(job):
    """Per-shot rows of worker k: its X shots, then its Z shots."""
    name, config, router, p, seed, shots, k, workers = job
    rows = []
    for basis in BASES:
        H, L, priors = dem_matrices(load_circuit(GROSS_CIRCUITS[basis], p), basis)
        llr = np.log((1 - priors) / priors)
        det, obs = load_shots(p, basis, seed, shots)
        decoder = make_decoder(name, H, priors, config, router)
        for shot in range(k, len(det), workers):
            error, seconds, stage, iterations = decode_one(decoder, name, det[shot])
            predicted = (L @ error) % 2
            rows.append({"shot": shot, "basis": basis, "time_s": seconds, "stage": stage, "iterations": iterations,
                         "syndrome_weight": int(det[shot].sum()), "solution_weight": float(llr[error == 1].sum()),
                         "pred_obs": "".join(map(str, predicted)), "true_obs": "".join(map(str, obs[shot])),
                         "fail": bool((predicted != obs[shot]).any())})
    return rows


def run(name, config_text, p, seed, shots, workers, router=None):
    config = parse_config(config_text)
    jobs = [(name, config, router, p, seed, shots, k, workers) for k in range(workers)]
    with Pool(workers) as pool:
        rows = [row for part in pool.map(run_worker, jobs) for row in part]
    log = pd.DataFrame(rows).sort_values(["basis", "shot"]).reset_index(drop=True)
    log.insert(2, "p", p)
    log.insert(3, "decoder", name)
    log.insert(4, "config", config_text or "default")
    return log


def log_path(name, config_text, p, seed, shots, workers):
    tag = (config_text or "default").replace(",", "_").replace("=", "")
    return LOG_DIR / f"{name}_{tag}_p{p}_seed{seed}_n{shots}_w{workers}.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decoder", choices=DECODERS, required=True)
    parser.add_argument("--config", default="", help="settings that replace the defaults, e.g. stop_nconv=1,num_sets=50")
    parser.add_argument("--router", help="router file, for tiered-ml")
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--shots", type=int, default=10_000, help="first N cached shots per basis")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", help="default: logs/<decoder>_<config>_p<p>_seed<seed>_n<shots>_w<workers>.csv")
    args = parser.parse_args()
    if args.decoder == "tiered-ml" and not args.router:
        parser.error("tiered-ml needs --router")
    log = run(args.decoder, args.config, args.p, args.seed, args.shots, args.workers, args.router)
    out = Path(args.out) if args.out else log_path(args.decoder, args.config, args.p, args.seed, args.shots, args.workers)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(out, index=False)
    from bench.stats import summarize
    s = summarize(log)
    print(f"{out.name}: failures [{s['fail_X']} {s['fail_Z']}] of {s['pairs']}  per cycle {s['rate']:.3g} "
          f"[{s['rate_lo']:.3g}, {s['rate_hi']:.3g}]  mean {s['mean_ms']:.1f} ms  p99 {s['p99_ms']:.0f} ms")


if __name__ == "__main__":
    main()
