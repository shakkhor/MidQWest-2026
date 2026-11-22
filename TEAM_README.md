# Team README: Challenge 1, ML-assisted decoding of the Gross code

Status as of Sunday morning. Branch: `develop/abdullah`. This is the place to catch up: what we
built, what we found, and what is left.

## Headline

**The team's two-stage TCBP-JOBS decoder (from `develop/Abel`) beats every relay-BP setting on
accuracy and speed, at both error rates we timed, on one CPU thread:**

| p | decoder | failed pairs | per cycle [95% CI] | mean / pair | p99 / pair |
|---|---|---|---|---|---|
| 0.004 | **TCBP-JOBS two-stage** | **42** | **3.5e-4** [2.6, 4.7] | **35 ms** | **0.45 s** |
| 0.004 | relay-BP default | 100 | 8.4e-4 [6.9, 10.2] | 212 ms | 3.6 s |
| 0.004 | relay-BP, fastest setting | 219 | 1.8e-3 | 39 ms | 0.58 s |
| 0.005 | **TCBP-JOBS two-stage** | **107** | **3.0e-3** [2.5, 3.6] | **144 ms** | 1.25 s |
| 0.005 | relay-BP, best setting (`stop_nconv=2`) | 244 | 7.0e-3 [6.2, 8.0] | 587 ms | 4.0 s |

- All decoders ran on identical shots: p=0.004 at 10,000 per basis, p=0.005 at 3,000.
- Same machine (Apple M4), 4 workers, one thread per shot, timing covers the `decode()` call only.
- NVIDIA's own CUDA-Q TCBP-JOBS, run by Abel on Kaggle on the same p=0.004 shots, gets the same
  41 failed pairs as our re-implementation.
- Details and caveats:
  [`results/optionB_analysis.md`](challenge-1-ml-decoder-kit/results/optionB_analysis.md).
  Plots: `results/pareto_p0.004.png`, `results/pareto_p0.005.png`.

## The two lines of work

### 1. TCBP-JOBS (Abel, `develop/Abel`): the headline decoder
- BP runs first.
- A 529-weight neural net reads each fault's BP belief history.
- A parallel OSD/beam search over 16 orderings returns the most likely correction and a
  confidence score.
- Two-stage version: accept a cheap search when it is confident, otherwise run the full search.
- Also supports heralded qubit loss: 1.3–1.9× fewer errors than the method without loss
  information.
- See `README_ABEL.md` on that branch.

### 2. Tiered relay-BP with a learned router (this branch): a supporting negative result

Idea: spend little compute on easy shots, and the full relay-BP search only where a learned
router says it is needed.

```
Stage 1: first relay-BP leg (80 BP iterations)   converged? → done   (86% of shots at p=0.004)
Stage 2: relay-BP, stop at the first solution    router trusts it? → done
Stage 3: more relay-BP solutions, keep the most likely candidate
```

What we found:
- **Phase 1:** relay-BP's own speed/accuracy curve at 6 settings. Our router lands on or behind
  it. ([`phase1_analysis.md`](challenge-1-ml-decoder-kit/results/phase1_analysis.md))
- **Option A:** router v2 (17 features, gradient boosting, label "stage 3 fixes this shot").
  It picks useful escalations 3–4× better than chance, but the gain over relay-BP
  `stop_nconv=2` is within noise. ([`optionA_analysis.md`](challenge-1-ml-decoder-kit/results/optionA_analysis.md))
- **Even a perfect router** (7.0e-4 at 65 ms) stays behind TCBP-JOBS two-stage.
- Findings worth keeping, no ML involved:
  - relay-BP `stop_nconv=2` matches the default's accuracy at about half the time;
  - cutting `num_sets` from 300 to 50 roughly doubles the error rate.

**Why this work still matters for the presentation:** relay-BP's trade-off curve is what lets us
say TCBP-JOBS beats *every* relay-BP setting, not just the default.

## Noise model

The kit's model: uniform circuit-level depolarizing noise with one strength p.
- Two-qubit depolarizing after every CX.
- A flip after each reset and before each measurement.
- Single-qubit depolarizing on idle data qubits.
- 12 noisy cycles, then a noiseless final readout.

Qubit loss is modelled in Abel's branch as heralded erasure (Stim `HERALDED_ERASE`).

## Positioning against the mentors' papers

- **Cascade** (Gu et al., arXiv:2604.08358): a neural decoder about 17× more accurate than
  relay-BP at p=0.1%, but about 100 H200 GPU-hours to train. We don't compete on accuracy.
- **NVIDIA Ising-Decoding** (arXiv:2604.12841): surface and color codes only, so there is no
  Gross-code number to beat.
- **NVIDIA TCBP-JOBS:** re-implemented on CPU, and validated against NVIDIA's code on our shots.
- **Hierarchical decoding survey** (Tang et al., arXiv:2611.16271): sets how we evaluate.
  Compare at matched accuracy, report how often the expensive path runs, give latency as a
  distribution, and state exactly what the timings cover.
- Closest prior work to the router idea: Chongder, arXiv:2607.05814. Lazy decoder: Delfosse,
  arXiv:2001.11427.

## Caveats to say out loud
- CPU timings of the `decode()` call, not hardware latency.
- NVIDIA's GPU timing on Kaggle ran with `bp_batch_size=1` (one shot at a time). It does not
  show what NVIDIA's decoder can do.
- p=0.003 is not on the timed trade-off plots. Abel's 100k-shot run there gives 1.9× fewer
  errors than relay-BP.

## Plan to the deadline (about 13:00 Sunday)

| step | status |
|---|---|
| Phase 0: benchmark harness, exact baseline reproduction | done |
| Phase 1: relay-BP trade-off curve vs. tiered decoder | done (negative result) |
| Option A: router v2 | done (within noise) |
| Option B: TCBP-JOBS in the shared harness | done (headline) |
| Phase 3: figures | trade-off plots done; latency CDF and stage-fraction plots optional |
| Phase 4: write-up and slides | next |
| Optional: 100k shots at p=0.003 on Babbage (`bench/accuracy_runs.sh`) | not run |

## How to run it

```bash
git clone -b develop/abdullah https://github.com/shakkhor/MidQWest-2026.git
cd MidQWest-2026
bash challenge-1-ml-decoder-kit/bench/setup_remote.sh   # venv, Rust, relay-bp, tests
cd challenge-1-ml-decoder-kit && source ../.venv/bin/activate

python -m bench.make_shots --p 0.004 --shots 10000 --seed 1          # cache shots
python -m bench.run_decoder --decoder relay --p 0.004 --shots 1000   # per-shot log in logs/
python -m bench.pareto --p 0.004                                     # table and plot in results/
```

To time Abel's decoders, check out `develop/Abel` at `07fb8d2`, set `ABEL_KIT` to its
`challenge-1-ml-decoder-kit` directory, and use `--decoder abel-two-stage` or
`--decoder abel-tcbp-jobs` (needs `numba` and `torch`). See `results/optionB_analysis.md`.

- **Timed runs:** one machine only, 4 workers, nothing else running.
- **Large accuracy runs:** a many-core machine (Babbage), via `bench/accuracy_runs.sh`.

## Files

| file | what |
|---|---|
| `challenge-1-ml-decoder-kit/bench/` | harness: `make_shots`, `run_decoder` (also times Abel's decoders), `stats`, `pareto`, `collect_hard`, `router_v2`, scripts, tests |
| `challenge-1-ml-decoder-kit/results/` | write-ups (`phase0_acceptance`, `phase1_analysis`, `optionA_analysis`, `optionB_analysis`), CSVs, plots |
| `challenge-1-ml-decoder-kit/fast_decoder.py`, `train_router.py` | tiered relay-BP decoder and router v1 |
| `challenge-1-ml-decoder-kit/PLAN.md` | plan v2 (`TIERED_DECODER.md`: plan v1) |
| `base-results.txt` | first baseline run (timing taken with other jobs running; failure counts are valid) |

## Where teammates can help
1. **Slides and demo:** the headline table and the two trade-off plots are final.
2. **Babbage:** `bash bench/accuracy_runs.sh`, to add a p=0.003 point at 100k shots.
3. **Literature check:** read Chongder (arXiv:2607.05814) and confirm how it differs.

Please work on your own branch (e.g. `develop/<name>`) and open a PR into `develop/abdullah`.
