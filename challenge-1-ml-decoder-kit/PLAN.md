# PLAN v2: Learned per-shot compute allocation for relay-BP (Challenge 1)

This revises the plan in `TIERED_DECODER.md` using the papers suggested by the Infleqtion
mentors. Sections marked **[new]** or **[changed]** differ from v1.
Deadline: about 13:00 Sunday. Time now: Saturday about 16:45.

## 0. Where we stand

Results on the same seed-1 shots as `baseline.py`, 10k shots per basis:

| p | decoder | failures [X Z] | per cycle | mean / pair | p99 / pair |
|---|---|---|---|---|---|
| 0.003 | relay-BP | [3 2] | 4.2e-5 ± 1.9e-5 | 63 ms | 446 ms |
| 0.003 | tiered-fast | [6 4] | 8.3e-5 ± 2.6e-5 | 14 ms | 129 ms |
| 0.003 | tiered-ml | [5 3] | 6.7e-5 ± 2.4e-5 | 22 ms | 191 ms |
| 0.004 | relay-BP | [51 40] | 7.6e-4 ± 0.8e-4 | 356 ms | 4803 ms |
| 0.004 | tiered-fast | [64 61] | 1.05e-3 ± 0.09e-3 | 82 ms | 1492 ms |
| 0.004 | tiered-ml | [58 47] | 8.8e-4 ± 0.9e-4 | 194 ms | 4964 ms |
| 0.005 | all | _pending_ | | | |

The router flags 64% of hard shots at 95% recall, so it is not selective.

## 1. What the reference papers mean for us **[new]**

| work | what it is | what it means for us |
|---|---|---|
| **Cascade**: Gu, Bonilla Ataides, Lukin, Yelin, arXiv:2604.08358 (Harvard) | A single CNN decoder with convolutions on the BB-code torus. About 17× below Relay and about 4000× below BP+OSD on the Gross code at p=0.1%. About 40 µs/cycle single-shot on an H200. Under 100 H200 GPU-hours to train. Calibrated output probabilities, used for post-selection. No hierarchy, routing, or qubit loss. | The accuracy frontier, but it needs a GPU cluster to train. **We don't compete on accuracy.** We keep relay-BP, which has published FPGA implementations, and add a cheap learned policy. We borrow one idea: a calibrated confidence score is itself useful (see 3c). |
| **Hierarchical decoding survey**: Tang et al. (21 authors: AMD, HPE, Riverlane, Xanadu, IQM, IBM), arXiv:2611.16271 | A survey of decoder-level and code-level hierarchies. Our method is a **predecoding cascade with confidence gating** in their terms. | It sets how we must evaluate (Section 2). It also notes that relay-BP configurations searching about 100 candidates are beyond current FPGA real-time decoders. That is our motivation: give the big search only to shots that need it. |
| **Lazy decoder**: Delfosse, arXiv:2001.11427 | Accepts a correction only for easy syndromes and escalates the rest. | Our stage 1 is a lazy decoder for qLDPC. Cite as the origin of the idea. |
| **Confidence-gated neural decoding**: Chongder, arXiv:2607.05814 (cited by Tang et al.) | Confidence-gated neural decoder for the rotated surface code. | **The closest prior work.** Ours differs in code (qLDPC/Gross vs. surface), in what is gated (relay-BP's search budget, not a neural decoder's output), and in being CPU/FPGA-friendly. Read the abstract before the write-up. |
| **NVIDIA Ising-Decoding**: arXiv:2604.12841, `github.com/NVIDIA/ising-decoding` | A 3D-CNN predecoder in front of PyMatching or Chromobius. **Surface and color codes only.** About 2.75–3.06× end-to-end speedup over PyMatching at d=31, p=0.6% on a GB300. | **There is no NVIDIA baseline on the Gross code.** Its CNN assumes a local grid layout; BB checks are nonlocal. See Section 6. |
| relay-BP: arXiv:2506.01779 (IBM; package authors T. Alexander, T. Müller) | Baseline | |
| BP-OSD: Panteleev & Kalachev, Quantum 5, 585 (2021); Roffe et al., `ldpc` | Baseline | |
| Gross code: Bravyi et al., Nature 627, 778 (2024) | Code | |

## 2. The claim and how we evaluate it **[changed]**

> At **matched logical error rate**, a learned per-shot stopping policy needs less decoding
> compute than **any fixed relay-BP setting**, at both the mean and the tail.

Following Tang et al.:
1. **Matched accuracy.** Compare against relay-BP's whole trade-off curve (Phase 1), not just
   its default. Use the curve point whose logical error rate is within the confidence interval of ours.
2. **Report the rare-path load:** the share of shots reaching stages 2 and 3, and the extra
   work they cost.
3. **Latency as a distribution:** mean, median, p99, p99.9, and a CDF, not just the mean.
4. **Timing boundaries stated explicitly:** we time only `decode()`, on one CPU thread per
   shot. Readout, transport, and apply time are excluded. CPU timings are not hardware latency.
5. **Full problem specification:** code, p, noise model (uniform circuit-level depolarizing,
   no qubit loss), 12 rounds, basis.
6. **Throughput and backlog [new]:** replay the per-shot service times through a queue
   simulation (syndrome rounds arrive every T; the decoder serves them in order). Report the
   largest T at which the backlog stays bounded, for relay-BP vs. tiered. The logs from
   Phase 1 make this almost free.

## 3. Ground rules (v1 rules kept; these change) **[changed]**

1. **Cached shots.** `baseline.py` samples X with `seed` and Z with `seed + 1`. Stim output
   depends on the shot count, so always sample 10k shots and take prefixes for smaller runs.
2. **Relay-BP depends on decode order.** I tested this: one decoder instance carries random
   state from shot to shot. Re-running `baseline.py` exactly requires the same worker count
   (5) and the same dealing (`k, k+5, ...`). Acceptance for Phase 0: **exact** for BP-OSD,
   **within error bars** for relay-BP, plus one exact replay using baseline's order.
3. **Timing.** The machine is an Apple M4 with 4 performance and 6 efficiency cores, and
   macOS can't pin processes to cores. Use **4 workers for all timed runs**, run nothing else
   alongside, and record this in the README.
4. **Environment.** Keep the existing Python 3.14 `.venv`, since relay-bp is already built
   there (v1 said 3.12). Add `scikit-learn`, `pandas` and `matplotlib`, and record versions
   with `pip freeze`.
5. Ask before any run longer than 30 minutes. Commit at the end of each phase.

## 4. Phases **[changed]**

### Phase 0: Benchmark setup (16:45–18:00)
- `bench/make_shots.py`: cached shots, seeded as `baseline.py` seeds them.
- `bench/run_decoder.py`: per-shot CSV log with `shot, basis, p, decoder, config, time_s,
  stage, iterations, syndrome_weight, solution_weight, fail`.
  - `n_solutions` is not exposed by `relay_bp`, so it is dropped.
  - Legs used ≈ (iterations − 80) / 60, so legs are not a separate feature.
- `bench/stats.py`: Wilson interval, per-cycle conversion, percentiles, plus unit tests.
- **Checkpoint 0.**

### Phase 1: relay-BP's own trade-off curve (18:00–22:00) **[changed: smaller grid]**
- Relay-BP grid: `stop_nconv` ∈ {1, 2, 5} × `num_sets` ∈ {300, 50}, 6 configs.
- Plus tiered-fast, and tiered-ml at recall ∈ {0.8, 0.9, 0.95, 0.99}. Changing recall only
  changes the threshold, so all 4 can be scored from **one** run that logs the router score
  and both candidate outcomes. That is a large saving.
- p=0.004: 10k shots per basis. p=0.005: first 3k, then confirm the frontier points at 5k.
- Rough cost with 4 workers: about 1 h at p=0.004 and about 2–2.5 h at p=0.005. **Ask first.**
- Output: `results/pareto_p{p}.csv`, plots of logical error rate vs. mean and vs. p99 µs/cycle.
- **Checkpoint 1:** does tiered-ml lie beyond the curve? Answer yes, no, or within noise.

### Phase 2: Router v2 and tail cap (22:00–01:00)
- **Features:** add `posterior_ratios` statistics (min |LLR|, counts of bits with |LLR| < τ,
  mean of the 10 smallest) to the existing 5 features.
- **Models:** logistic regression vs. `HistGradientBoostingClassifier`, both calibrated on the
  validation seed (2000).
- **Labels:** main label is "disagrees with full relay-BP". Ablation label is "wrong vs. truth";
  it has few positives, so present it as indicative.
- **Tail cap:** run stage 3 with a smaller `num_sets` / `stop_nconv` budget. If it finds no new
  solution, keep the stage-2 candidate. Relay-BP can't be stopped mid-run. Sweep the cap.
- **3c [new, from Cascade]:** the router score as a **failure flag**. Plot logical error rate
  among accepted shots vs. the fraction of shots flagged. It is cheap, uses the same model,
  and shows the confidence score is useful beyond routing.
- **Checkpoint 2:** is the flagged fraction well below 64% at 95% recall? Have the points
  moved beyond the curve?

### Overnight (01:00–06:00, ask first)
Final held-out runs at p ∈ {0.003, 0.004, 0.005}, both bases, 10k shots: best tiered config,
default relay-BP, and the relay-BP point at matched accuracy.

### Phase 3: Figures (06:00–09:00)
1. Trade-off curve for each p (the headline).
2. Stage fractions (rare-path load) for each p.
3. Router reliability diagram, plus the 3c failure-flag curve.
4. Per-shot latency CDF.
5. **[new]** Backlog vs. syndrome period.

### Phase 4: Write-up and demo (09:00–12:00)
As in v1, with the positioning in Section 1 and the evaluation wording in Section 2.

## 5. Out of scope
Unchanged from v1: neural prior-steering, a qubit-loss model, a Cascade reproduction, and GPU work.

## 6. "Can we beat the NVIDIA baseline?" **[new]**

Not head to head, and we shouldn't claim to:
- **NVIDIA Ising-Decoding has no Gross-code result.** It supports only surface and color
  codes, and its 3D CNN relies on a local grid that BB codes don't have. Porting it means new
  circuits, a non-grid architecture, and multi-GPU training: days of CUDA work. This Mac has
  no CUDA.
- **Cascade** does cover the Gross code, and it beats relay-BP by about 17×. Matching that
  takes about 100 H200 GPU-hours. Not feasible here.

What we can honestly claim is the same kind of result NVIDIA reports, in our setting:

> NVIDIA's predecoder gives about a 3× end-to-end speedup over its global decoder (PyMatching)
> on surface codes. Our learned gate gives an N× speedup over relay-BP on the Gross code at
> matched accuracy, with no GPU and no retraining of the global decoder.

Current N at matched accuracy: about 1.8× at p=0.004 and about 2.9× at p=0.003 (within noise).
Phase 2's goal is to raise it to 3× or more and to fix the p99. The comparison is between
approaches, not numbers on the same benchmark, and the README must say so.
