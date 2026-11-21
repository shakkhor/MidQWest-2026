"""Statistics of per-shot logs: Wilson intervals, per-cycle rates, latency percentiles.

  python -m bench.stats logs/*.csv
"""
import sys

import numpy as np
import pandas as pd

CYCLES = 12


def wilson(k, n, z=1.96):
    """Wilson score interval (lo, hi) of a binomial proportion k / n."""
    if n == 0:
        return 0.0, 1.0
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def per_cycle(P, cycles=CYCLES):
    """Per-cycle logical error rate from the failure fraction P over `cycles` cycles, as in baseline.py."""
    return 1 - (1 - np.asarray(P, float)) ** (1 / cycles)


def pairs(log):
    """One row per shot index: X and Z fail and time, the pair fails if either fails."""
    wide = log.pivot_table(index="shot", columns="basis", values=["fail", "time_s"], aggfunc="first")
    out = pd.DataFrame({"fail_X": wide["fail"]["X"].astype(bool), "fail_Z": wide["fail"]["Z"].astype(bool),
                        "time_s": wide["time_s"]["X"] + wide["time_s"]["Z"]})
    out["fail"] = out.fail_X | out.fail_Z
    return out


def summarize(log, cycles=CYCLES):
    """Dict of the headline numbers of one decoder config on paired X and Z shots."""
    pr = pairs(log)
    n, k = len(pr), int(pr.fail.sum())
    lo, hi = wilson(k, n)
    ms = pr.time_s.to_numpy() * 1e3
    row = {"decoder": log.decoder.iloc[0], "config": log.config.iloc[0], "p": log.p.iloc[0], "pairs": n,
           "fail_X": int(pr.fail_X.sum()), "fail_Z": int(pr.fail_Z.sum()), "fail_pairs": k,
           "rate": float(per_cycle(k / n, cycles)), "rate_lo": float(per_cycle(lo, cycles)),
           "rate_hi": float(per_cycle(hi, cycles)),
           "mean_ms": ms.mean(), "median_ms": np.median(ms), "p99_ms": np.percentile(ms, 99),
           "p999_ms": np.percentile(ms, 99.9), "mean_us_per_cycle": ms.mean() * 1e3 / cycles}
    if "stage" in log and log.stage.notna().any():
        for s in (1, 2, 3):
            row[f"stage{s}"] = float((log.stage == s).mean())
    return row


def main(paths):
    rows = [summarize(pd.read_csv(path)) for path in paths]
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.4g}".format):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
