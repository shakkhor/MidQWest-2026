# Fawaz's work: baselines, easy/hard analysis, ML-ordered OSD, and verification

Everything here runs on the kit's Gross-code benchmark (`../circuits`, `../baseline.py`,
`../decoders.py`, unchanged). Every comparison uses **the same shots as the kit's `baseline.py`**
(seed 1, 10,000 X+Z pairs per noise level), saved per shot in `results/` so any decoder can be
compared shot by shot.

## Setup

From the kit folder (`challenge-1-ml-decoder-kit/`), Python 3.12, Rust for relay-BP:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r fawaz/requirements.txt
cd fawaz
python -m pytest test_bp_trajectory.py -v      # 3 tests, ~3 s
python make_report.py                          # rebuilds table + plots from saved results, seconds
```

## The steps

| step | script | what it does | output |
|---|---|---|---|
| 0 | `run_baselines.py` | the kit's BP-OSD and relay-BP, saving every shot's pass/fail and time | `results/baseline_p*.npz` |
| 1 | `step1_first_pass.py` | relay-BP's first pass (80 BP iterations) alone: which shots are "easy" | `results/first_pass_p*.npz`, `first_pass.log` |
| 2 | `bp_trajectory.py`, `step2_trajectories.py` | our own BP that records every iteration (matches the library to 1e-12); fault-labelled shots; do the trajectories carry signal? | `results/step2_trajectories.png` |
| 3 | `tcbp.py`, `step3_train.py` | small network (593 weights) that reads each fault's BP history, NVIDIA TCBP-style | `results/tcbp_model.joblib`, `step3_ranking.png`, `step3.log` |
| 4 | `step4_pipeline.py` | first pass, then a fallback for the hard shots (BP-OSD / relay-BP / ML-ordered OSD / routing) | `results/pipeline_p*.npz`, `pipeline.log` |
| - | `make_report.py` | the comparison table and plots | `results/summary.md`, `results/plot_*.png` |

Rerunning from scratch: step 0 takes about 1 h on 10 cores, step 1 a few minutes, step 3 about 2 min,
step 4 about 25 min. Steps depend on earlier outputs (1 needs 0, 4 needs 0, 1 and 3).

## Findings

**1. Relay-BP's time and mistakes sit in a few "hard" shots** (`first_pass.log`). Its first pass
alone solves 97% / 86% / 64% of decodes at p = 0.003 / 0.004 / 0.005, in about 3 ms each and
almost never wrong. The remaining hard decodes take 30-89% of relay-BP's time and hold
almost all of its mistakes (100 of 103 at p = 0.004).

**2. Stopping after the first pass when it succeeds is a free speed-up** (no ML): relay-BP keeps
searching for 5 solutions even when the first pass already found one. "First pass + relay-BP"
is 2.5x faster at p = 0.003 and 1.4x at p = 0.004, with accuracy within error bars.

**3. BP's history ranks the true faults far better than its final belief** (step 2-3). On hard shots,
ranking by the final belief buries at least one true fault near the bottom (median deepest rank
~7,000 of 8,784); our network puts every true fault in the top 1,000 in 61% of hard shots at
p = 0.004, vs 2% (final belief) and 44% (lowest belief, a hand-made trick).

**4. ML-ordered OSD beats BP-OSD on accuracy and mean time, but not relay-BP** (`summary.md`).
At p = 0.004: 302 failed pairs vs 527 (BP-OSD) and 103 (relay-BP); mean 82 ms vs 190 / 182 ms.
The missing piece vs relay-BP is a search that improves each OSD answer (NVIDIA's beam search).

## Verification of the team's TCBP-JOBS decoder (teammate's implementation)

Recomputed from its per-shot files, on the same 10,000 shot pairs, against `results/baseline_p*.npz`:

| p | BP-OSD | relay-BP | TCBP-JOBS | fails only under relay-BP / only under TCBP-JOBS |
|---|---|---|---|---|
| 0.003 | 59 | 8 | **1** | 8 / 1 (McNemar p = 0.04) |
| 0.004 | 527 | 103 | **41** | 70 / 8 (p = 2e-13) |
| 0.005 | 2488 | 795 | **349** | 500 / 54 (p = 2e-91) |

Same shots (its baseline counts match ours exactly), training data independent of the test shots
(faults sampled from the decoding matrix), same scoring (pair fails if any of 12 observables is wrong
in X or Z), the decoder sees only the syndrome.

Timing at p = 0.004, 300 pairs, one decoder at a time on an idle 12-core laptop:

| decoder | cores | mean | median | p99 |
|---|---|---|---|---|
| relay-BP | 1 | 119 ms | 42 ms | 1392 ms |
| BP-OSD | 1 | 144 ms | 160 ms | 446 ms |
| TCBP-JOBS | 1 | 127 ms | 70 ms | 769 ms |
| TCBP-JOBS | 12 | 38 ms | 27 ms | 160 ms |

**Estimate, not yet a real run:** putting relay-BP's first pass in front of TCBP-JOBS (accept when
the first pass solves the shot) would give 1 / 47 / 362 failed pairs at mean 9 / 23 / 81 ms at
p = 0.003 / 0.004 / 0.005 (from combining the two per-shot files). Needs a real run to confirm.

## Caveats

- Relay-BP draws random memory strengths, so rerunning it on the same shot can change the answer
  (8 vs 6 failed pairs at p = 0.003 between two runs). Small count differences are noise.
- Times are wall-clock on a laptop CPU; always state the number of cores used.
- The step 3 model is trained on X-memory shots and also used for Z memory.
