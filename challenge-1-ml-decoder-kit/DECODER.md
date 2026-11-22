# TCBP-JOBS on the Gross code: a learned BP front end + parallel OSD search, with loss

**Team decoder for Infleqtion Challenge 1 (ML-assisted QEC decoding).** We re-implemented NVIDIA's
TCBP-JOBS idea (Howe, Maurya et al., "TCBP-JOBS: GPU-Accelerated Quantum Error Correction Decoding and
Noise Learning with Near-Optimal Accuracy") from the paper's description, on CPU, for the kit's
`[[144,12,12]]` Gross code memory experiments, and extended it to **heralded qubit loss**.

**Headline (identical shots, 10,000 per basis, exact McNemar tests):** logical error rate per cycle

| p | BP-OSD | relay-BP | **TCBP-JOBS (ours)** |
|---|---|---|---|
| 0.003 | 4.9e-4 | 6.7e-5 | **8.3e-6** |
| 0.004 | 4.5e-3 | 8.6e-4 | **3.4e-4** |
| 0.005 | 2.4e-2 | 6.9e-3 | **3.0e-3** |

With heralded loss (p=0.004, p_loss 0.001-0.004) the loss-aware decoder makes 1.3-1.9x fewer errors than
TCBP-JOBS without loss information and 2.5-3.7x fewer than relay-BP. The two-stage confidence variant keeps the
accuracy at 2.3x less time. Full tables, timing and caveats: `README_ABEL.md` at the repository root and
`python analyze.py`.

## How it works

```
syndrome ──► BP, 10 sum-product iterations ──► converged? ── yes ──► BP's answer        (~9% of shots at p=0.004)
                  │ (whole trajectory of every fault's belief)
                  ▼ no
            TCBP: one 529-weight MLP, shared by all faults,
            trajectory (10 LLRs + 4 shape features) ─► calibrated P(fault fired)
                  ▼
            JOBS: 16 orderings (TCBP order + 15 Gumbel-jittered copies), in parallel threads
                  each: GF(2) elimination of a window of ⌈8 K*⌉ columns ─► OSD-0
                        ─► beam search (width 32, depth 40) over syndrome-preserving moves
                  ▼
            cheapest correction (prior log-likelihood) + complementary gap (confidence)
```

* **The ML part (TCBP).** BP's final belief often ranks the true faults badly exactly on the shots
  where OSD is needed: beliefs oscillate on trapping sets. A tiny MLP reads each fault's *whole*
  BP trajectory and outputs a calibrated probability. It is trained in ~70 s on CPU from faults
  sampled straight from the detector error model, so labels are free and no hardware data is needed.
  On the kit's shots it cuts OSD-0 failures 1.8× and shortens the prefix OSD needs (median K*
  103 → 67), with only the ordering changed (`eval_ordering.py`).
* **The search (JOBS).** OSD commits to one ordering. JOBS searches 16 plausible orderings and
  walks away from each OSD-0 solution with a beam search, always staying on valid corrections,
  and keeps the most likely correction overall. It also records the cheapest correction of every
  logical class, so the gap between the two best classes is a per-shot confidence at no extra cost.
* **Loss (our extension).** Neutral-atom loss is modelled as erasure conversion: at the end of each
  cycle each data qubit is lost with probability `p_loss`, detected (a herald bit), and replaced by a
  fresh atom in the maximally mixed state (`HERALDED_ERASE` in Stim, `loss.py`). Heralds become
  per-shot priors: faults of a lost qubit are nearly free (probability ≈ 1/2), faults tied to a
  herald that did not fire are ruled out, everything else keeps its circuit prior. The same priors
  drive BP, the TCBP features and the JOBS cost. A separate TCBP model is trained on lossy faults.

## Reproduce

Python 3.12 with the kit's setup (`pip install -r requirements.txt`, Rust for relay-BP).

```bash
python tcbp.py --shots 3000 --out tcbp_model.npz                       # train TCBP (~1-2 min, CPU)
python loss_decoding.py --p 0.004 --p-loss 0.002 --out tcbp_loss_model.npz
python baseline.py --p 0.004 --shots 10000 > results/baseline_p0.004.txt                 # BP-OSD + relay-BP totals
python evaluate.py --decoder tcbp-jobs --p 0.004 --shots 10000 --out results/tcbpjobs_p0.004.npz
python evaluate.py --decoder relay-bp  --p 0.004 --shots 10000 --workers 10 --out results/relay_p0.004.npz
python evaluate_loss.py --decoder tcbp-jobs     --p 0.004 --p-loss 0.002 --shots 5000
python evaluate_loss.py --decoder relay-aware   --p 0.004 --p-loss 0.002 --shots 5000 --workers 6
python evaluate_loss.py --decoder relay-unaware --p 0.004 --p-loss 0.002 --shots 5000 --workers 6
python timing.py --p 0.004 --shots 500                                  # latency, idle machine
python analyze.py                                                       # tables + results/*.png
python -m pytest test_kit.py test_ours.py
```

`evaluate.py` decodes exactly the shots of `baseline.py` (same `--p`, `--shots`, `--seed`), so every
comparison is on identical shots and is tested shot by shot (exact McNemar test).

| file | what it does |
|------|--------------|
| `trajbp.py` | sum-product BP that keeps every fault's belief trajectory (NumPy reference + numba kernels) |
| `tcbp.py` | TCBP features, the 529-weight MLP, training on faults sampled from the DEM |
| `gf2.py` | bit-packed GF(2) elimination and OSD-0 for any column ordering (numba) |
| `jobs.py` | JOBS: jittered windows, elimination, beam search, complementary gap, class sums (numba, parallel draws) |
| `decoder.py` | the full decoder: BP → TCBP → JOBS |
| `loss.py`, `loss_decoding.py` | heralded loss circuits, loss-aware priors, loss-aware TCBP training and decoder |
| `evaluate.py`, `evaluate_loss.py` | per-shot records for any decoder on the baseline's shots |
| `analyze.py`, `timing.py` | comparison tables, McNemar tests, routing/postselection curves, plots; idle-machine latency |
| `eval_ordering.py` | the TCBP ordering experiment (OSD-0 with final-LLR, min-LLR or TCBP order) |
| `test_ours.py` | tests: numba BP = NumPy BP, valid corrections, JOBS never worse than OSD-0, loss circuit exact |

## Honest caveats

* This is a **re-implementation from the paper's description**, not NVIDIA's code (CUDA-Q QEC needs
  an NVIDIA GPU; we ran on a 12-core Apple M-series CPU). Our numbers are on the kit's benchmark
  (uniform circuit noise, per-basis decoding), not the paper's (SI1000 noise, joint XYZ decoding),
  so they are not directly comparable to the paper's tables.
* Latency is CPU wall-clock per X+Z pair. JOBS's 16 draws run on parallel threads (the paper runs
  them on a GPU); the single-thread numbers are reported as the resource cost.
* The baselines are run at the relay-BP authors' published settings from the kit. Both decode one
  basis at a time, as do we unless stated (the full-DEM experiment uses all 1728 detectors).
* The loss model is erasure conversion with immediate replacement; it does not model a lost atom
  staying lost for several cycles.
