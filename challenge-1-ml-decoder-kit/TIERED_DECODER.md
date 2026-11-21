# PLAN.md: Learned per-shot stopping for relay-BP (Challenge 1)

Read this whole file before writing code. Work **one phase at a time**. At each checkpoint,
**stop and report to Abdullah** before continuing.

## 0. Context

- Challenge: Infleqtion MidQWest 2026, Challenge 1: *ML-assisted QEC decoding*. Gross code
  [[144,12,12]], memory experiment, 12 cycles. Required sweep: p ∈ {0.003, 0.004, 0.005}.
  Required metrics: logical error rate, decoding latency, decoder resources.
- Deadline: about 13:00 Sunday (local). Today is Saturday; we start at about 16:30.
- Existing files (do not change how they behave): `baseline.py`, `decoders.py` (`decoders.score()`),
  `base-results.txt`.
- Existing tiered work (see `TIERED_DECODER.md`): `fast_decoder.py` (`TieredDecoder`, `Router`,
  `features`), `train_router.py`, `bench_tiered.py`, `router.npz`.
- Baseline at p=0.004, 10k shots per basis:
  - relay-BP: failures [51 40], 7.62e-4 per cycle, mean 355.8 ms, p99 4803 ms
  - BP-OSD: failures [266 267], 4.50e-3 per cycle, mean 228.2 ms, p99 685 ms
- Current tiered results at p=0.004:
  - tiered-fast: 1.05e-3 per cycle, 82.1 ms mean, p99 1492 ms
  - tiered-ml: 8.79e-4 per cycle, 194.1 ms mean, p99 4964 ms
  - Router flags 64% of hard shots at 95% recall, so it is not selective.

## 1. The claim we are testing

> A learned per-shot stopping policy gives a better speed vs. accuracy trade-off than
> **any fixed relay-BP setting**.

The claim holds only if tiered-ml points lie beyond relay-BP's own trade-off curve: lower
latency at the same logical error rate, within confidence intervals. "Faster than default
relay-BP" alone is **not** the claim. tiered-fast may just be relay-BP with `stop_nconv=1`.

## 2. Ground rules

1. **Same shots for every decoder.** Sample once, cache to disk, and decode the cached shots.
2. **No tuning on test data.** Training uses seed 1000, validation seed 2000, test seed 1
   (the same shots as `baseline.py`).
3. **Error bars on every logical error rate** (Wilson 95% interval on shot failures). Convert to
   per cycle using the same formula as `baseline.py`.
4. **Timing:** time each shot's `decode()` call with `time.perf_counter`. Report mean, median,
   p99 and p99.9 per shot, plus µs per cycle (per-shot time ÷ 12). State the CPU, the number
   of workers, and that it is single-threaded per shot.
5. **Report negative results.** If the claim fails, say so in the README.
6. **Don't change how `baseline.py` or `decoders.py` behave.** Put new code in new files.
7. **Ask before any run expected to take more than 30 minutes.** Give the estimated time.
8. **Check `relay_bp` parameter names** (e.g. `stop_nconv`, leg/iteration limits,
   `posterior_ratios`) against the installed package. Don't guess them.
9. Commit at the end of each phase with a clear message.

## 3. Phases

### Phase 0: Benchmark hygiene (target 16:30–17:30)

Build `bench/` with:
- `make_shots.py`: samples detector shots and true observables for each (p, basis, seed) and
  saves them to `shots/p{p}_{basis}_seed{seed}.npz`. **First check how `baseline.py` samples**
  (seed, workers) so that seed=1 reproduces its shots exactly.
- `run_decoder.py`: decodes a cached shot file with a named decoder config and writes a
  per-shot log (`logs/*.parquet` or `.csv`) with columns:
  `shot, basis, p, decoder, config, time_s, stage, iterations, n_solutions, syndrome_weight,
  pred_obs, true_obs, fail`.
- `stats.py`: Wilson interval, per-cycle conversion, and latency percentiles.

**Acceptance:** re-running relay-BP and BP-OSD on cached seed-1 shots at p=0.004 reproduces the
failure counts in `base-results.txt` exactly.

**Checkpoint 0:** report the reproduced numbers.

### Phase 1: relay-BP's own trade-off curve (target 17:30–21:00)

Sweep relay-BP settings on the **same** cached shots:
- `stop_nconv` ∈ {1, 2, 3, 5} × 2 leg budgets (the default and a smaller one): about 8
  configs.
- p=0.005, 5k shots per basis (more failures, so tighter error bars at lower cost).
- p=0.004, 10k shots per basis (headline).

Also run on the same shots: tiered-fast, and tiered-ml at several router thresholds
(`--recall` ∈ {0.8, 0.9, 0.95, 0.99}).

Estimate the runtime before launching (relay-BP mean is about 0.36 s/shot at p=0.004 and about
0.86 s/shot at p=0.005), then ask.

**Output:** `results/pareto_p{p}.csv` and a plot with mean µs/cycle (log scale) on x and
per-cycle logical error rate with intervals on y. Repeat with p99 on x.

**Checkpoint 1:** does any tiered-ml point lie beyond the relay-BP curve? Report yes, no, or
within noise.

### Phase 2: Router v2 and tail fix (target 21:00–01:00)

**2a. Richer features.** Add features from the stage-2 relay-BP run to `features`:
- statistics of `posterior_ratios`: min |log ratio|, count of bits with |log ratio| < τ for
  a few τ, and the mean of the smallest 10;
- iterations, number of legs used, solution weight, solution weight ÷ syndrome weight,
  syndrome weight, and number of predicted faults.

Keep the existing 5 features as a baseline model for comparison.

**2b. Models.**
- Logistic regression (the existing one) vs. `sklearn` `HistGradientBoostingClassifier`.
- Calibrate both (isotonic or Platt on the validation seed).
- Keep the label as in `TIERED_DECODER.md` (the first solution's logical prediction differs
  from full relay-BP's). As an ablation, also try the label "first solution is wrong vs. the
  true observable".

**2c. Tail fix.** Cap stage-3 work (a fixed leg/iteration budget; return the best candidate
found so far). Sweep the cap and report how p99 and the logical error rate change.

**Output:** a reliability diagram, the flagged fraction of hard shots at recall 0.95, and new
points on the Phase 1 trade-off plots.

**Checkpoint 2:** at 95% recall, has the flagged fraction dropped well below 64%? Have the new
points moved beyond the relay-BP curve?

### Overnight (01:00–06:00)

Final held-out runs, after asking: all p ∈ {0.003, 0.004, 0.005}, both bases, the best
router config and its cap, plus the relay-BP default and the best relay-BP sweep point at
matched accuracy. 10k shots per basis.

### Phase 3: Results (06:00–09:00)

Four figures, saved to `figures/`:
1. Trade-off curve for each p (the headline).
2. Where decodes stopped (stage 1/2/3 fractions) for each p.
3. Router reliability diagram.
4. Per-shot latency CDF: relay-BP vs. tiered-ml.

Plus a results table: decoder, config, failures [X Z], per-cycle rate ± interval, mean,
median, p99, µs/cycle.

### Phase 4: Write-up (09:00–12:00)

The README must include:
- **Problem and claim**, as in Section 1.
- **Method:** the three stages, router features and model, the tail cap.
- **Results:** the figures and table, with honest wording. If the claim fails or is within
  noise, say so.
- **Prior work and positioning:**
  - Gu, Bonilla Ataides, Lukin, Yelin. *Scalable Neural Decoders for Practical Fault-Tolerant
    Quantum Computation* (Cascade), arXiv:2604.08358. It is about 17× more accurate than relay-BP
    on the Gross code at p=0.1%, with about 40 µs/cycle single-shot on an H200, but training
    takes under 100 H200 GPU-hours. We do not compete with it. Our method keeps relay-BP
    (which has FPGA implementations) and uses ML only to decide how much compute each shot
    gets.
  - Tang et al. *Hierarchical Decoding for Quantum Error Correction*, arXiv:2611.16271. Our
    decoder is a confidence-gated cascade in their terms. Follow their guidance: compare at
    matched accuracy and report the rare-path load.
  - Lazy predecoder: use the reference the Tang et al. survey cites.
  - relay-BP (IBM, 2025): copy the exact citation from the `relay_bp` package/repo.
  - BP-OSD: Panteleev & Kalachev, Quantum 5, 585 (2021); Roffe et al., `ldpc` package.
  - Gross code: Bravyi et al., Nature (2024).
- **Limitations:** CPU timings are not hardware latency; no qubit-loss model; statistics.
- **Future work:** (1) feed qubit-loss/erasure flags in as stage-1 prior updates and as router
  features; (2) a small neural model that adjusts stage-1 BP priors (per-fault labels via
  Stim's error sampling); (3) a resumable stage 3.
- **Reproduce:** exact commands, seeds, and versions (`pip freeze`).

## 4. Out of scope (don't build)

- Neural prior-steering, a qubit-loss noise model, anything resembling Cascade, and GPU or
  CUDA-Q work (the machine is a Mac).

## 5. Repo hygiene

- `requirements.txt` updated (add `scikit-learn`, `pandas`, `matplotlib` if used).
- Python 3.12 venv. Fix all seeds.
- Small unit tests for `stats.py` (Wilson interval, per-cycle conversion) and for checking that
  cached shots reproduce the `baseline.py` shots.
- No hard-coded absolute paths.