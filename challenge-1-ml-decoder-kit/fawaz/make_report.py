"""Turn the saved results into the comparison table and the plots for the presentation.

  python make_report.py

Reads results/baseline_p*.npz, first_pass_p*.npz, pipeline_p*.npz (no decoding, runs in seconds).
Writes:
  results/summary.md               the full table (every decoder, every noise level)
  results/plot_error_rate.png      logical error rate per cycle vs noise
  results/plot_time.png            mean and p99 decode time vs noise (two panels, one scale each)
  results/plot_tradeoff_p0.004.png speed vs accuracy at p = 0.004
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import _kit  # noqa: F401,E402  (makes the kit's circuit/baseline/decoders importable)
from decoders import per_cycle_rate  # noqa: E402
from step4_pipeline import combine  # noqa: E402

RESULTS = Path(__file__).parent / "results"
NOISE = [0.003, 0.004, 0.005]
# Plotted decoders, in fixed categorical order (validated palette: blue, orange, aqua, yellow, magenta).
# Distinct markers carry identity too, so colour is never the only cue.
SHOWN = {
    "BP-OSD (baseline)": ("#2a78d6", "o", "BP-OSD"),
    "relay-BP (baseline)": ("#eb6834", "s", "relay-BP"),
    "first pass + relay-BP": ("#1baf7a", "^", "first pass + relay-BP (no ML)"),
    "first pass + ML 16 orderings": ("#eda100", "D", "ours: ML, 16 orderings"),
    "first pass + ML 4 + relay-BP": ("#e87ba4", "v", "ours: ML 4 + relay-BP"),
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"


def summarise():
    """{p: [(name, pair failures, pairs, rate, err, mean ms, median ms, p99 ms)]}"""
    out = {}
    for p in NOISE:
        rows, pairs, _ = combine(p)
        out[p] = []
        for name, failed, seconds in rows:
            f = failed[pairs].any(axis=1)
            ms = seconds[pairs].sum(axis=1) * 1e3
            rate, err = per_cycle_rate(f, 12)
            out[p].append((name, int(f.sum()), int(pairs.sum()), rate, err, ms.mean(), np.median(ms),
                           np.percentile(ms, 99)))
    return out


def write_table(summary):
    lines = ["# Results: Gross code, 10,000 X+Z shot pairs per noise level, same shots for every decoder", "",
             "Rate = logical error rate per syndrome cycle (lower is better). Times are per X+Z pair on a laptop CPU.", ""]
    for p, rows in summary.items():
        lines += [f"## p = {p}", "", "| decoder | failed pairs | rate per cycle | mean ms | median ms | p99 ms |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name, f, n, rate, err, mean, med, p99 in rows:
            lines.append(f"| {name} | {f} / {n} | {rate:.2e} ± {err:.1e} | {mean:.1f} | {med:.1f} | {p99:.0f} |")
        lines.append("")
    (RESULTS / "summary.md").write_text("\n".join(lines))


def style(ax):
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.set_xticks(NOISE)
    ax.set_xticklabels([f"{p:.3f}" for p in NOISE])


def series(summary, name, col):
    return np.array([next(r for r in summary[p] if r[0] == name)[col] for p in NOISE])


def plot_error_rate(summary):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for name, (color, marker, label) in SHOWN.items():
        y, e = series(summary, name, 3), series(summary, name, 4)
        ax.errorbar(NOISE, y, yerr=e, color=color, marker=marker, ms=8, lw=2, capsize=3, label=label,
                    mec="white", mew=1.5)
    ax.set_yscale("log")
    style(ax)
    ax.set_xlabel("physical error rate p", color=INK)
    ax.set_ylabel("logical error rate per cycle (log scale, lower is better)", color=INK)
    ax.set_title("Logical error rate, same 10,000 shot pairs for every decoder", color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "plot_error_rate.png", dpi=160)


def plot_time(summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, col, title in ((axes[0], 5, "Mean decode time per shot pair"),
                           (axes[1], 7, "Slowest 1% (p99) decode time per shot pair")):
        for name, (color, marker, label) in SHOWN.items():
            ax.plot(NOISE, series(summary, name, col), color=color, marker=marker, ms=8, lw=2, label=label,
                    mec="white", mew=1.5)
        ax.set_yscale("log")
        style(ax)
        ax.set_xlabel("physical error rate p", color=INK)
        ax.set_ylabel("milliseconds (log scale, lower is better)", color=INK)
        ax.set_title(title, color=INK, loc="left")
    axes[0].legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "plot_time.png", dpi=160)


def plot_tradeoff(summary, p=0.004):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for name, (color, marker, label) in SHOWN.items():
        r = next(r for r in summary[p] if r[0] == name)
        ax.errorbar(r[5], r[3], yerr=r[4], color=color, marker=marker, ms=10, capsize=3, mec="white", mew=1.5,
                    ls="none", label=label)
        ax.annotate(label, (r[5], r[3]), textcoords="offset points", xytext=(9, 6), fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("mean decode time per shot pair, ms (log scale)", color=INK)
    ax.set_ylabel("logical error rate per cycle (log scale)", color=INK)
    ax.set_title(f"Speed vs accuracy at p = {p}: bottom-left is best", color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(RESULTS / f"plot_tradeoff_p{p}.png", dpi=160)


def main():
    summary = summarise()
    write_table(summary)
    plot_error_rate(summary)
    plot_time(summary)
    plot_tradeoff(summary)
    print((RESULTS / "summary.md").read_text())


if __name__ == "__main__":
    main()
