"""Compare the per-shot records of evaluate.py and draw the plots.

  python analyze.py                      # every p with records in results/

For each p it prints BP-OSD (from baseline_p*.txt), relay-BP, TCBP-JOBS and the routed hybrid:
TCBP-JOBS answers every shot; shots whose complementary gap is below tau are also decoded by
relay-BP and the cheaper of the two corrections (prior log-likelihood cost) is returned.
"""
import glob
import os
import re

import numpy as np
from scipy.stats import binomtest

from decoders import per_cycle_rate

CYCLES = 12
OUT = "results"


def load(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def bposd_from_txt(p):
    """(failures [X, Z], rate, err, mean ms, p99 ms) of BP-OSD from the baseline sweep, or None."""
    path = os.path.join(OUT, f"baseline_p{p}.txt")
    if not os.path.exists(path):
        return None
    for line in open(path):
        m = re.match(r"bposd\s+failures \[\s*(\d+)\s+(\d+)\] of (\d+)\s+per cycle (\S+) \+- (\S+)\s+mean (\S+) ms\s+p99 (\S+) ms", line)
        if m:
            return (int(m[1]), int(m[2])), float(m[4]), float(m[5]), float(m[6]), float(m[7])
    return None


def summary(failed, seconds):
    pair = failed.any(axis=1)
    rate, err = per_cycle_rate(pair, CYCLES)
    ms = seconds.sum(axis=1) * 1e3
    return {"fx": int(failed[:, 0].sum()), "fz": int(failed[:, 1].sum()), "pairs": int(pair.sum()),
            "rate": rate, "err": err, "mean": ms.mean(), "p50": np.median(ms), "p99": np.percentile(ms, 99)}


def routed(tj, rl, tau):
    """Hybrid: per basis shot, route to relay-BP when TCBP-JOBS did not converge and gap < tau.

    Routed shots return the cheaper correction of the two (joint decision). Their time is the
    sum of both decoders' times, since relay-BP runs after TCBP-JOBS.
    """
    route = (~tj["converged"]) & (tj["gap"] < tau)
    pick_relay = route & (rl["cost"] < tj["cost"] - 1e-9)
    failed = np.where(pick_relay, rl["failed"], tj["failed"])
    seconds = tj["seconds"] + np.where(route, rl["seconds"], 0.0)
    return failed, seconds, route.mean()


def mcnemar(a, b):
    """Pairs failed only by a, only by b, and the exact two-sided p-value."""
    pa, pb = a.any(axis=1), b.any(axis=1)
    only_a, only_b = int((pa & ~pb).sum()), int((pb & ~pa).sum())
    pval = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
    return only_a, only_b, pval


def line(name, s):
    return (f"{name:28s} [{s['fx']:5d} {s['fz']:5d}] pairs {s['pairs']:5d}  {s['rate']:.2e} +- {s['err']:.1e}"
            f"  mean {s['mean']:7.1f} ms  p50 {s['p50']:6.1f}  p99 {s['p99']:7.0f} ms")


def main():
    rows = {}
    ps = sorted({re.search(r"_p([\d.]+)\.npz", f)[1] for f in glob.glob(os.path.join(OUT, "tcbpjobs_p*.npz"))})
    for p in ps:
        tj_path, rl_path = os.path.join(OUT, f"tcbpjobs_p{p}.npz"), os.path.join(OUT, f"relay_p{p}.npz")
        tj = load(tj_path)
        print(f"\n=== p = {p}, {int(tj['shots'])} shots per basis (failures [X Z], logical error rate per cycle)")
        b = bposd_from_txt(p)
        if b:
            print(f"{'BP-OSD (baseline sweep)':28s} [{b[0][0]:5d} {b[0][1]:5d}]              {b[1]:.2e} +- {b[2]:.1e}"
                  f"  mean {b[3]:7.1f} ms               p99 {b[4]:7.0f} ms")
        st = summary(tj["failed"], tj["seconds"])
        rows.setdefault(p, {})["TCBP-JOBS"] = st
        if b:
            rows[p]["BP-OSD"] = {"rate": b[1], "err": b[2], "mean": b[3], "p99": b[4]}
        if os.path.exists(rl_path):
            rl = load(rl_path)
            sr = summary(rl["failed"], rl["seconds"])
            rows[p]["relay-BP"] = sr
            print(line("relay-BP", sr))
            print(line("TCBP-JOBS (ours)", st))
            oa, ob, pv = mcnemar(tj["failed"], rl["failed"])
            print(f"  shot by shot: pairs only TCBP-JOBS fails {oa}, only relay-BP fails {ob}, McNemar p = {pv:.2g}")
            lower_cost = (rl["cost"] < tj["cost"] - 1e-9) & ~tj["converged"]
            print(f"  relay-BP found a cheaper correction on {lower_cost.sum()} of {int((~tj['converged']).sum())} "
                  f"non-converged basis shots")
            best = None
            for tau in [0.0, 0.5, 1, 2, 3, 4, 6, 8, np.inf]:
                f, s, q = routed(tj, rl, tau)
                sm = summary(f, s)
                print(line(f"  routed tau={tau:<4} q={q:.3f}", sm))
                if best is None or (sm["pairs"], sm["mean"]) < (best[1]["pairs"], best[1]["mean"]):
                    best = (tau, sm, q)
            rows[p]["routed"] = best[1]
            rows[p]["routed_tau"] = best[0]
            # how well does the gap flag failures?
            nc = ~tj["converged"]
            g, fl = tj["gap"][nc], tj["failed"][nc]
            if fl.any():
                print(f"  gap of failed non-converged shots: median {np.median(g[fl]):.2f}; "
                      f"of correct ones: median {np.median(g[~fl]):.2f}")
        else:
            print(line("TCBP-JOBS (ours)", st))
        if "failed_coset" in tj:
            fc = tj["failed_coset"]
            print(f"  class-sum (coset) decision: failures [{fc[:, 0].sum()} {fc[:, 1].sum()}] pairs {fc.any(axis=1).sum()}")
        rows[p]["post"] = postselection(tj)
        print("  postselection on the complementary gap (discard the least confident pairs): " + ", ".join(
            f"{d:.1%} -> {r:.2e}" for d, r, _ in rows[p]["post"]))
        kst = tj["kstar"][~tj["converged"]]
        print(f"  BP converged on {tj['converged'].mean():.1%} of basis shots; median K* {np.median(kst):.0f}")
    if rows:
        plot(rows)
    for path in sorted(glob.glob(os.path.join(OUT, "twostage_p*.npz"))):
        two_stage_table(path)
    loss_rows = loss_tables()
    if loss_rows:
        plot_loss(loss_rows)


def two_stage_table(path):
    """Accept stage 1 when its gap >= tau, else pay for stage 2 too. Swept offline over tau."""
    z = load(path)
    nc = ~z["converged"]
    print(f"\n=== two-stage decoder, p = {float(z['p'])}, {int(z['shots'])} shots per basis")
    print(f"  stage 1 alone everywhere: failures [{z['failed1'][:, 0].sum()} {z['failed1'][:, 1].sum()}]")
    for tau in [0, 5, 10, 15, 20, 30, np.inf]:
        stage2 = nc & (z["gap1"] < tau)
        failed = np.where(stage2, z["failed"], z["failed1"])
        seconds = z["t1"] + np.where(stage2, z["t2"], 0.0)
        print(line(f"  tau={tau:<4} stage2 on {stage2.mean():.1%}", summary(failed, np.nan_to_num(seconds))))


def postselection(tj, fractions=(0.0, 0.001, 0.005, 0.01, 0.02, 0.05)):
    """[(discarded fraction, rate per cycle of the kept pairs, kept)] using the pair's smaller gap."""
    conf = np.where(tj["converged"], np.inf, tj["gap"]).min(axis=1)
    order = np.argsort(conf)
    failed = tj["failed"].any(axis=1)
    out = []
    for frac in fractions:
        keep = np.ones(len(conf), bool)
        keep[order[:int(round(frac * len(conf)))]] = False
        out.append((frac, per_cycle_rate(failed[keep], CYCLES)[0], keep.sum()))
    return out


def loss_tables():
    """Print the loss experiments: relay-BP without and with heralds against loss-aware TCBP-JOBS."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(OUT, "loss_pl*_tcbp-jobs_p*.npz"))):
        m = re.search(r"loss_pl([\d.]+)_tcbp-jobs_p([\d.]+)\.npz", path)
        pl, p = m[1], m[2]
        recs = {"TCBP-JOBS, loss-aware (ours)": load(path)}
        for name, tag in (("relay-BP, loss-unaware", "relay-unaware"), ("relay-BP, loss-aware", "relay-aware"),
                          ("TCBP-JOBS as published (no loss info)", "tcbp-jobs-unaware")):
            q = os.path.join(OUT, f"loss_pl{pl}_{tag}_p{p}.npz")
            if os.path.exists(q):
                recs[name] = load(q)
        print(f"\n=== loss: p = {p}, p_loss = {pl} per data qubit per cycle, {int(recs['TCBP-JOBS, loss-aware (ours)']['shots'])} shots per basis")
        for name in ("relay-BP, loss-unaware", "relay-BP, loss-aware", "TCBP-JOBS as published (no loss info)",
                     "TCBP-JOBS, loss-aware (ours)"):
            if name in recs:
                sm = summary(recs[name]["failed"], recs[name]["seconds"])
                rows.setdefault(pl, {})[name] = sm
                print(line(name, sm))
        ours = recs["TCBP-JOBS, loss-aware (ours)"]
        for name in ("relay-BP, loss-unaware", "relay-BP, loss-aware", "TCBP-JOBS as published (no loss info)"):
            if name in recs:
                oa, ob, pv = mcnemar(ours["failed"], recs[name]["failed"])
                print(f"  vs {name}: pairs only ours fails {oa}, only theirs {ob}, McNemar p = {pv:.2g}")
    return rows


def plot_loss(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pls = sorted(rows, key=float)
    names = ["relay-BP, loss-unaware", "relay-BP, loss-aware", "TCBP-JOBS as published (no loss info)",
             "TCBP-JOBS, loss-aware (ours)"]
    colors = ["#9aa0a6", "#e8710a", "#8ab4f8", "#1a73e8"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.2
    for k, (name, color) in enumerate(zip(names, colors)):
        xs = [i + (k - 1.5) * width for i, pl in enumerate(pls) if name in rows[pl]]
        rs = [rows[pl][name] for pl in pls if name in rows[pl]]
        axes[0].bar(xs, [r["rate"] for r in rs], width, yerr=[r["err"] for r in rs], color=color, label=name, capsize=3)
        axes[1].bar(xs, [r["mean"] for r in rs], width, color=color, label=name)
    for ax in axes:
        ax.set_xticks(range(len(pls)))
        ax.set_xticklabels([f"p_loss = {pl}" for pl in pls])
        ax.set_yscale("log")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set(ylabel="logical error rate per cycle", title="Gross code, p = 0.004 Pauli + heralded loss")
    axes[1].set(ylabel="mean decode time per pair (ms)", title="Mean latency")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "loss.png"), dpi=150)
    print(f"saved {os.path.join(OUT, 'loss.png')}")


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ps = sorted(rows, key=float)
    x = [float(p) for p in ps]
    styles = {"BP-OSD": ("#9aa0a6", "s"), "relay-BP": ("#e8710a", "o"), "TCBP-JOBS": ("#1a73e8", "^")}
    labels = {"BP-OSD": "BP-OSD (OSD-CS 10)", "relay-BP": "Relay-BP", "TCBP-JOBS": "TCBP-JOBS (ours, CPU)",
              "routed": "TCBP-JOBS → Relay-BP routed (ours)"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, (color, marker) in styles.items():
        pts = [(xi, rows[p][name]) for xi, p in zip(x, ps) if name in rows[p]]
        if not pts:
            continue
        xs = [a for a, _ in pts]
        axes[0].errorbar(xs, [r["rate"] for _, r in pts], yerr=[r["err"] for _, r in pts], color=color,
                         marker=marker, capsize=3, label=labels[name])
        axes[1].plot(xs, [r["mean"] for _, r in pts], color=color, marker=marker, label=labels[name] + " mean")
        axes[1].plot(xs, [r["p99"] for _, r in pts], color=color, marker=marker, ls="--", alpha=0.6)
    axes[0].set(xscale="log", yscale="log", xlabel="physical error rate p", ylabel="logical error rate per cycle",
                title="Gross code [[144,12,12]], 12 cycles")
    axes[1].set(yscale="log", xlabel="physical error rate p", ylabel="decode time per X+Z pair (ms)",
                title="Latency: mean (solid), p99 (dashed)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x])
        ax.minorticks_off()
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ler_and_latency.png"), dpi=150)
    print(f"\nsaved {os.path.join(OUT, 'ler_and_latency.png')}")


if __name__ == "__main__":
    main()
