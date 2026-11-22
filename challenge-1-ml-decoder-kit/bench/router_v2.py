"""Option A: train and evaluate router v2 offline from bench.collect_hard logs.

  python -m bench.router_v2 --train-seed 1000 --val-seed 2000                 # model selection only
  python -m bench.router_v2 --train-seed 1000 --val-seed 2000 --test-seed 1   # plus held-out test curves

Training uses seed 1000, model choice and thresholds use seed 2000, and the test seed is used
once, for the final curves. Label: "stage 3 fixes this shot" (stage 2 wrong, stage 3 right).
Compared: base (5 features of v1) vs. rich features; logistic regression vs. gradient boosting;
stage 3a (4 more solutions) vs. 3b (1 more). Writes results/router_v2_*.csv and .png.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bench.stats import summarize

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
RESULTS = Path(__file__).resolve().parent.parent / "results"
BASE = ["f_synd", "f_log_iter", "f_weight", "f_weight_per_synd", "f_nflip"]
ESCALATE_FRACTIONS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]   # of hard shots, set on validation


def load(seed, workers=None, ps=None):
    paths = sorted(LOG_DIR.glob(f"hard_p*_seed{seed}_n*_w{workers or '*'}.csv"))
    logs = [pd.read_csv(path) for path in paths]
    log = pd.concat(logs, ignore_index=True)
    return log if ps is None else log[log.p.isin(ps)]


def models():
    return {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)),
        "gboost": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                         l2_regularization=1.0, class_weight="balanced", random_state=0),
    }


def label(log, variant):
    return (log.fail2 & ~log[f"fail3{variant}"]).astype(int).to_numpy()


def flagged_at_recall(y, score, recall=0.95):
    """Fraction of hard shots flagged when the threshold catches `recall` of the positives."""
    if y.sum() == 0:
        return np.nan
    threshold = np.quantile(score[y == 1], 1 - recall)
    return float((score >= threshold).mean())


def fit_all(train, val):
    """Every (variant, features, model): calibrated on validation. Returns rows and fitted models."""
    rich = [c for c in train.columns if c.startswith("f_")]
    th, vh = train[train.hard], val[val.hard]
    rows, fitted = [], {}
    for variant in "ab":
        ytr, yva = label(th, variant), label(vh, variant)
        for fname, cols in (("base", BASE), ("rich", rich)):
            for mname, make in models().items():
                model = make().fit(th[cols].to_numpy(), ytr)
                calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid").fit(vh[cols].to_numpy(), yva)
                score = calibrated.predict_proba(vh[cols].to_numpy())[:, 1]
                x1 = vh[cols].to_numpy()[:1]
                start = time.perf_counter()
                for _ in range(200):
                    calibrated.predict_proba(x1)
                infer_s = (time.perf_counter() - start) / 200
                key = (variant, fname, mname)
                fitted[key] = (calibrated, cols, infer_s, score)
                rows.append({"variant": variant, "features": fname, "model": mname, "train_pos": int(ytr.sum()),
                             "val_pos": int(yva.sum()), "val_hard": len(yva),
                             "val_avg_precision": average_precision_score(yva, score) if yva.any() else np.nan,
                             "val_flagged_at_95_recall": flagged_at_recall(yva, score),
                             "infer_us": infer_s * 1e6})
    return pd.DataFrame(rows), fitted


def replay(log, escalate, variant, infer_s, name, config):
    """Tiered decoder outcome when `escalate` (bool per hard row) chooses stage 3."""
    hard = log.hard.to_numpy()
    esc = np.zeros(len(log), bool)
    esc[hard] = escalate
    out = log.copy()
    out["time_s"] = log.t1 + hard * (log.t2 + infer_s) + esc * log[f"t3{variant}"]
    out["fail"] = np.where(~hard, log.fail1, np.where(esc, log[f"fail3{variant}"], log.fail2))
    out["stage"] = np.where(~hard, 1, np.where(esc, 3, 2))
    out["decoder"], out["config"] = name, config
    return out


def test_curves(test, fitted, keys, val):
    rows = []
    for p in sorted(test.p.unique()):
        tp, vp = test[test.p == p], val[val.p == p]
        th = tp[tp.hard]
        for key in keys:
            model, cols, infer_s, _ = fitted[key]
            vscore = model.predict_proba(vp[vp.hard][cols].to_numpy())[:, 1]
            tscore = model.predict_proba(th[cols].to_numpy())[:, 1]
            for frac in ESCALATE_FRACTIONS:
                threshold = np.quantile(vscore, 1 - frac)          # set on validation only
                s = summarize(replay(tp, tscore >= threshold, key[0], infer_s, "router-v2", f"{key}|{frac}"))
                rows.append(s | {"p": p, "variant": key[0], "features": key[1], "model": key[2], "val_escalate": frac})
        for variant in "ab":   # references: never and always escalate, and the oracle
            hard_y = (th.fail2 & ~th[f"fail3{variant}"]).to_numpy()
            for name, esc in (("never", np.zeros(len(th), bool)), ("always", np.ones(len(th), bool)), ("oracle", hard_y)):
                s = summarize(replay(tp, esc, variant, 0.0, name, variant))
                rows.append(s | {"p": p, "variant": variant, "features": "-", "model": name, "val_escalate": np.nan})
    return pd.DataFrame(rows)


def plot(curves, p, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    pareto = RESULTS / f"pareto_p{p}.csv"
    if pareto.exists():
        relay = pd.read_csv(pareto)
        relay = relay[relay.decoder == "relay"]
        ax.errorbar(relay.mean_us_per_cycle, relay.rate, yerr=[relay.rate - relay.rate_lo, relay.rate_hi - relay.rate],
                    fmt="o", color="#4C78A8", capsize=3, label="relay-BP (fixed settings)")
        for x, y, c in zip(relay.mean_us_per_cycle, relay.rate, relay.config):
            ax.annotate(c.replace("stop_nconv=", "n=").replace("num_sets=", "s="), (x, y), fontsize=6,
                        xytext=(4, 3), textcoords="offset points", color="#4C78A8")
    c = curves[curves.p == p]
    colors = {"a": "#E45756", "b": "#F58518"}
    for (variant, feats, model), g in c[c.features != "-"].groupby(["variant", "features", "model"]):
        g = g.sort_values("mean_us_per_cycle")
        ax.plot(g.mean_us_per_cycle, g.rate, "-" if feats == "rich" else ":", marker="D" if model == "gboost" else "s",
                ms=4, color=colors[variant], alpha=0.9, label=f"router v2: 3{variant}, {feats}, {model}")
    oracle = c[c.model == "oracle"]
    ax.scatter(oracle.mean_us_per_cycle, oracle.rate, marker="*", s=140, color="black", label="oracle router (ceiling)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean decode time per cycle (µs, X+Z pair / 12)")
    ax.set_ylabel("logical error rate per cycle")
    ax.set_title(f"Router v2 on held-out shots, p = {p}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-seed", type=int, default=1000)
    parser.add_argument("--val-seed", type=int, default=2000)
    parser.add_argument("--test-seed", type=int)
    args = parser.parse_args()
    train, val = load(args.train_seed), load(args.val_seed)
    table, fitted = fit_all(train, val)
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "router_v2_validation.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(table.to_string(index=False))
    if args.test_seed is None:
        return
    test = load(args.test_seed, workers=4)
    curves = test_curves(test, fitted, list(fitted), val)
    curves.to_csv(RESULTS / "router_v2_test_curves.csv", index=False)
    for p in sorted(test.p.unique()):
        plot(curves, p, RESULTS / f"router_v2_p{p}.png")
    cols = ["p", "variant", "features", "model", "val_escalate", "fail_pairs", "rate", "rate_lo", "rate_hi", "mean_ms",
            "p99_ms", "stage3"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format, "display.max_rows", 500):
        print(curves[[c for c in dict.fromkeys(cols) if c in curves]].to_string(index=False))


if __name__ == "__main__":
    main()
