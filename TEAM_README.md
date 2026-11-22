# Team README: Challenge 1, learned per-shot compute for relay-BP

Status as of Saturday evening. This is the place to catch up: what we built, what we
found, and what comes next. Branch: `develop/abdullah`.

## The idea in one paragraph

Relay-BP is the strongest baseline in the kit, and it is slow mainly because it spends the
same large effort on every shot: it keeps searching until it has 5 solutions. Most shots
are easy. On 86% of shots at p=0.004, the first 80 BP iterations already converge, and that
answer is essentially never wrong. **Our decoder spends little compute on easy shots and the
full relay-BP search only on hard ones. A small learned model (the router) decides which
shots are hard.** The claim we are testing is that this beats *every fixed relay-BP
setting* on speed at equal accuracy, not just the default one.

## The decoder (`challenge-1-ml-decoder-kit/fast_decoder.py`)

```
syndrome
  │
  ▼  Stage 1: first relay-BP leg only (80 iterations of BP)      ~5 ms
  │  converged? ── yes ──► done                                  (86% of shots at p=0.004)
  ▼  no
  Stage 2: relay-BP, stop at the first solution
  │  router trusts it? ── yes ──► done
  ▼  no
  Stage 3: 4 more relay solutions; keep the most likely of all candidates
```

**Router:** logistic regression on 5 features of the stage-2 solution:
- syndrome weight;
- iterations needed;
- solution weight (how unlikely the error pattern is);
- solution weight ÷ syndrome weight;
- number of predicted errors.

**Label:** 1 when the first solution disagrees with full relay-BP on the logical outcome.
Trained on separate seeds (1000+); the test seed is 1.

## Noise model

The kit's model, unchanged: uniform circuit-level depolarizing noise with one strength p.
- Two-qubit depolarizing after every CX.
- A flip after each reset and before each measurement.
- Single-qubit depolarizing on idle data qubits.
- 12 noisy cycles, then a noiseless final readout.

**No qubit loss is modelled yet.** The challenge asks for it "where available". This is a
known gap; see Next steps.

## What we have done

1. **Reproduced the baselines** at p=0.004, 10k shots per basis (`base-results.txt`):
   relay-BP 7.6e-4 per cycle, BP-OSD 4.5e-3 per cycle. Both match the kit's reference.
2. **Profiled relay-BP.** Most of its time on easy shots goes to collecting 5 solutions when
   the first is already right. BP-OSD as a fallback is 3–4× less accurate, so we dropped it.
3. **Built the tiered decoder and router** (`fast_decoder.py`, `train_router.py`).
4. **Read the mentors' suggested papers** and wrote `PLAN.md` (v2): positioning against
   Cascade, NVIDIA Ising-Decoding, and the hierarchical-decoding survey.
5. **Built a clean benchmark harness** (`bench/`, Phase 0):
   - shots are sampled once and cached, so every decoder sees identical shots;
   - per-shot CSV logs;
   - Wilson 95% intervals;
   - **Re-running both baselines reproduces `base-results.txt` exactly** ([266 267] and
     [51 40]).
6. **Phase 1 (running):** relay-BP's own speed/accuracy curve vs. our decoder.

## Results so far (p = 0.004, 10k shots per basis, identical shots, 4 workers on an M4)

| decoder | per cycle [95% interval] | mean / pair | p99 / pair |
|---|---|---|---|
| relay-BP default (`stop_nconv=5`) | 8.4e-4 [6.9e-4, 1.0e-3] | 212 ms | 3.6 s |
| relay-BP `stop_nconv=2` | 9.7e-4 [8.1e-4, 1.2e-3] | **103 ms** | 2.1 s |
| relay-BP `stop_nconv=1` | 1.14e-3 [9.6e-4, 1.35e-3] | 58 ms | 1.2 s |
| tiered-fast (no router) | 1.17e-3 [9.9e-4, 1.38e-3] | 59 ms | 1.0 s |
| tiered-ml, router at 95% recall | 9.7e-4 [8.1e-4, 1.2e-3] | 152 ms | 3.7 s |
| tiered-ml, router at 99% recall | 9.1e-4 [7.6e-4, 1.1e-3] | 160 ms | 3.7 s |
| BP-OSD | 4.5e-3 | 203 ms | 0.6 s |

**Honest reading: at p=0.004 the claim currently fails.**
- tiered-ml matches relay-BP `stop_nconv=2` on accuracy but is about 50% slower.
- tiered-fast is just relay-BP `stop_nconv=1` under another name.

The cause is clear: escalated shots rerun relay-BP from scratch and pay more than a plain
`stop_nconv=2` run. This is what Phase 2 fixes. p=0.005 results are still running.

An earlier figure of "4× faster" was wrong. It compared against a baseline timing taken
while other jobs were using the CPU. All numbers above come from clean, comparable runs.

## Plan (details in `challenge-1-ml-decoder-kit/PLAN.md`)

| phase | what | when |
|---|---|---|
| 0 | Benchmark harness, exact baseline reproduction | done |
| 1 | relay-BP trade-off curve vs. tiered decoder, p=0.004 and 0.005 | running, about 1 h left |
| 2 | **Router v2 and stage-3 fix:** cheaper escalation (1 extra solution instead of 4), posterior-based features, gradient-boosted model, calibration, tail cap | overnight |
| — | Final held-out runs at p = 0.003, 0.004, 0.005 (big accuracy runs on Babbage) | overnight |
| 3 | Figures: trade-off curves, stage fractions, router reliability, latency CDF, backlog | Sunday 06:00–09:00 |
| 4 | Write-up and demo | Sunday 09:00–12:00 |

How we report (from the Tang et al. survey):
- compare at matched accuracy;
- report how many shots take the expensive path;
- give latency as a distribution, not just the mean;
- state exactly what the timings cover (CPU, single thread, the `decode()` call only).

## Positioning against the mentors' papers

- **Cascade** (Gu et al., arXiv:2604.08358): a neural decoder about 17× more accurate than
  relay-BP on the Gross code, but it takes about 100 H200 GPU-hours to train. We don't compete
  on accuracy.
- **NVIDIA Ising-Decoding** (arXiv:2604.12841): a learned predecoder, about 3× end-to-end
  speedup, but **surface and color codes only**, so there is no Gross-code number to beat. Our
  story is the analogous result for qLDPC codes, on CPU.
- **Hierarchical decoding survey** (Tang et al., arXiv:2611.16271): in its terms our decoder
  is a "predecoding cascade with confidence gating". It defines how we evaluate.
- Closest prior work: Chongder, arXiv:2607.05814 (confidence-gated decoding, surface code).
  Lazy decoder: Delfosse, arXiv:2001.11427.

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

- **Timed runs:** Abdullah's Mac only, with 4 workers and nothing else running. Timings from
  different machines must not be mixed.
- **Large accuracy runs:** Babbage, via `bench/accuracy_runs.sh`. It uses seed 3 and all cores.

## Files

| file | what |
|---|---|
| `challenge-1-ml-decoder-kit/fast_decoder.py` | `TieredDecoder`, `Router`, `features` |
| `challenge-1-ml-decoder-kit/train_router.py` | trains `router.npz` |
| `challenge-1-ml-decoder-kit/bench/` | harness: `make_shots`, `run_decoder`, `stats`, `pareto`, `phase1.sh`, `accuracy_runs.sh`, `setup_remote.sh`, tests |
| `challenge-1-ml-decoder-kit/results/` | committed result summaries |
| `challenge-1-ml-decoder-kit/PLAN.md` | the current plan (v2) |
| `challenge-1-ml-decoder-kit/TIERED_DECODER.md` | the first plan (v1) |
| `base-results.txt` | our baseline run |

## Where teammates can help

1. **Qubit loss.** Add heralded loss as erasure: a known location whose error priors are raised
   for that shot. This is the biggest gap against the challenge text. It is self-contained:
   a new circuit/noise variant plus a prior update in stage 1.
2. **Babbage accuracy runs:** `bash bench/accuracy_runs.sh` (100k shots at p=0.003).
3. **Slides and demo skeleton** (Phase 4). The story and the positioning above are stable
   even though the numbers will change.
4. **Literature check:** read Chongder (arXiv:2607.05814) and confirm how it differs from ours.

Please work on your own branch (e.g. `develop/<name>`) and open a PR into `develop/abdullah`.
