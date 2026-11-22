"""Decode memory experiments with heralded qubit loss and save per-shot records.

Decoders, all on the same sampled shots (X memory seed, Z memory seed + 1):
  relay-unaware   relay-BP on the X (Z) checks; erasures folded into the Pauli priors
  relay-aware     relay-BP on the checks plus the herald detectors
  tcbp-jobs       loss-aware TCBP-JOBS (heralds -> per-shot priors)

  python evaluate_loss.py --decoder tcbp-jobs --p 0.004 --p-loss 0.002 --shots 5000
Records go to results/loss_pl{p_loss}_{decoder}_p{p}.npz with the fields of evaluate.py.
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np

from decoders import per_cycle_rate, relay_decoder
from loss import dem_matrices_ids
from loss_decoding import LossAwareTcbpJobs, LossProblem


def sample_shots(pb, shots, seed):
    det, obs = pb.circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return det.astype(np.uint8), obs.astype(np.uint8)


def _records(name, basis, p, p_loss, shots, seed, first, step, model):
    pb = LossProblem(basis, p, p_loss)
    det, obs = sample_shots(pb, shots, seed)
    det, obs = det[first::step], obs[first::step]
    syn, her = pb.split(det)
    weights = 1 << np.arange(obs.shape[1])
    if name == "tcbp-jobs":
        dec = LossAwareTcbpJobs(pb, model_path=model)
        run = lambda k: dec.decode_full(syn[k], her[k])
        L = pb.L
    elif name == "relay-aware":
        rd = relay_decoder(pb.H_aware, pb.priors_aware)
        full = det[:, pb.ids_aware]
        L = pb.L_aware
        run = lambda k: (rd.decode(full[k]), {})
    elif name == "tcbp-jobs-unaware":
        # NVIDIA's TCBP-JOBS as published: no loss information, erasures folded into the priors
        from decoder import TcbpJobsDecoder
        H, L, pr = dem_matrices_ids(pb.circuit, pb.ids_checks)
        dec = TcbpJobsDecoder(H, L, pr, model_path="tcbp_model.npz")
        run = lambda k: dec.decode_full(syn[k])
    elif name == "relay-unaware":
        H, L, pr = dem_matrices_ids(pb.circuit, pb.ids_checks)
        rd = relay_decoder(H, pr)
        run = lambda k: (rd.decode(syn[k]), {})
    else:
        raise ValueError(name)
    out = {k: [] for k in ("failed", "seconds", "cost", "pred", "converged", "gap", "kstar", "failed_coset", "margin")}
    for k in range(len(det)):
        start = time.perf_counter()
        e, info = run(k)
        seconds = time.perf_counter() - start
        e = np.asarray(e, np.uint8)
        pred = (L @ e) % 2
        # every correction is costed under the same loss-aware per-shot priors (check-only columns)
        w = np.log((1 - pb.shot_priors(her[k])[0]) / pb.shot_priors(her[k])[0])
        if name == "relay-aware":
            e_cost = e[pb.keep]
        elif name == "tcbp-jobs":
            e_cost = e
        else:
            e_cost = None
        out["failed"].append(bool((pred != obs[k]).any()))
        out["seconds"].append(seconds)
        out["cost"].append(float(w[e_cost.astype(bool)].sum()) if e_cost is not None else np.nan)
        out["pred"].append(int(pred @ weights))
        out["converged"].append(info.get("converged", False))
        out["gap"].append(info.get("gap", np.nan))
        out["kstar"].append(info.get("kstar", -1))
        coset = info.get("coset")
        out["failed_coset"].append(out["failed"][-1] if coset is None else coset != int(obs[k] @ weights))
        out["margin"].append(info.get("margin", np.nan))
    return {k: np.array(v) for k, v in out.items()}


def _job(a):
    return _records(*a)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--decoder", required=True, choices=["tcbp-jobs", "tcbp-jobs-unaware", "relay-aware", "relay-unaware"])
    parser.add_argument("--p", type=float, default=0.004)
    parser.add_argument("--p-loss", type=float, default=0.002)
    parser.add_argument("--shots", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model", default="tcbp_loss_model.npz")
    args = parser.parse_args()
    start = time.time()
    columns = []
    for k, basis in enumerate("XZ"):
        if args.workers == 1:
            columns.append(_records(args.decoder, basis, args.p, args.p_loss, args.shots, args.seed + k, 0, 1, args.model))
        else:
            with Pool(args.workers) as pool:
                parts = pool.map(_job, [(args.decoder, basis, args.p, args.p_loss, args.shots, args.seed + k, i,
                                         args.workers, args.model) for i in range(args.workers)])
            merged = {}
            for key in parts[0]:
                arr = np.empty(args.shots, parts[0][key].dtype)
                for i, part in enumerate(parts):
                    arr[i::args.workers] = part[key]
                merged[key] = arr
            columns.append(merged)
    rec = {key: np.column_stack([c[key] for c in columns]) for key in columns[0]}
    out = f"results/loss_pl{args.p_loss}_{args.decoder}_p{args.p}.npz"
    os.makedirs("results", exist_ok=True)
    np.savez(out, **rec, p=args.p, p_loss=args.p_loss, shots=args.shots, seed=args.seed, decoder=args.decoder)
    rate, err = per_cycle_rate(rec["failed"].any(axis=1), 12)
    ms = rec["seconds"].sum(axis=1) * 1e3
    print(f"{args.decoder:14s} p={args.p} p_loss={args.p_loss} failures {rec['failed'].sum(axis=0)} of {args.shots}  "
          f"per cycle {rate:.2e} +- {err:.1e}  mean {ms.mean():.1f} ms  p99 {np.percentile(ms, 99):.0f} ms  "
          f"(wall {time.time() - start:.0f} s) -> {out}")


if __name__ == "__main__":
    main()
