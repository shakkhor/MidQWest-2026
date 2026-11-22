"""Build the speed vs. accuracy table and plots from per-shot logs.

  python -m bench.pareto --p 0.004 --router router.npz

Relay-BP and BP-OSD logs are summarized as they are. A tiered-all log is replayed at the
router thresholds for --recall (and at an infinite threshold, which is tiered-fast).
Writes results/pareto_p{p}.csv and results/pareto_p{p}.png.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bench.run_decoder import LOG_DIR
from bench.stats import read_log, summarize

RESULTS = Path(__file__).resolve().parent.parent / "results"


def replay(log, threshold, name, config):
    """The tiered decoder at one router threshold, from a tiered-all log."""
    hard = log.stage >= 2
    escalate = hard & (log.score >= threshold)
    out = log.copy()
    out["time_s"] = log.t1 + hard * log.t2 + escalate * log.t3
    out["fail"] = np.where(~hard, log.fail1, np.where(escalate, log.fail3, log.fail2))
    out["stage"] = np.where(~hard, 1, np.where(escalate, 3, 2))
    out["decoder"], out["config"] = name, config
    return out


def table(p, router_path, recalls, seed=1, workers=4):
    from fast_decoder import Router
    rows = []
    for path in sorted(LOG_DIR.glob(f"*_p{p}_seed{seed}_n*_w{workers}.csv")):
        log = read_log(path)
        if log.decoder.iloc[0] != "tiered-all":
            rows.append(summarize(log))
            continue
        router = Router.load(router_path)
        rows.append(summarize(replay(log, np.inf, "tiered-fast", "no router")))
        for recall in recalls:
            rows.append(summarize(replay(log, router.threshold_for_recall(recall), "tiered-ml", f"recall={recall}")))
    return pd.DataFrame(rows)


def plot(df, p, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    styles = {"relay": ("o", "#4C78A8"), "bposd": ("s", "#9D9D9D"), "tiered-fast": ("^", "#F58518"),
              "tiered-ml": ("D", "#E45756")}
    for ax, col, label in ((axes[0], "mean_us_per_cycle", "mean"), (axes[1], "p99_ms", "p99")):
        for name, group in df.groupby("decoder"):
            marker, color = styles.get(name, ("x", "black"))
            x = group[col] if col == "mean_us_per_cycle" else group[col] * 1e3 / 12
            yerr = [group.rate - group.rate_lo, group.rate_hi - group.rate]
            ax.errorbar(x, group.rate, yerr=yerr, fmt=marker, color=color, label=name, capsize=3, ms=6)
            for xi, yi, cfg in zip(x, group.rate, group.config):
                ax.annotate(cfg, (xi, yi), fontsize=6, xytext=(4, 3), textcoords="offset points", color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{label} decode time per cycle (µs, X+Z pair / 12)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("logical error rate per cycle (95% Wilson)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Gross code, p = {p}: speed vs. accuracy on identical shots")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--router", default="router.npz")
    parser.add_argument("--workers", type=int, default=4, help="use only logs timed with this many workers")
    parser.add_argument("--recall", type=float, nargs="+", default=[0.8, 0.9, 0.95, 0.99])
    args = parser.parse_args()
    df = table(args.p, args.router, args.recall, workers=args.workers)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / f"pareto_p{args.p}.csv", index=False)
    plot(df, args.p, RESULTS / f"pareto_p{args.p}.png")
    cols = ["decoder", "config", "pairs", "fail_X", "fail_Z", "rate", "rate_lo", "rate_hi", "mean_ms", "median_ms",
            "p99_ms", "p999_ms", "stage1", "stage2", "stage3"]
    with pd.option_context("display.width", 220, "display.float_format", "{:.4g}".format):
        print(df[[c for c in cols if c in df]].to_string(index=False))


if __name__ == "__main__":
    main()
