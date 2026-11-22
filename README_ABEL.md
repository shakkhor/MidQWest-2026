# Abel's progress: Challenge 1 (ML-assisted QEC decoding), branch `develop/Abel`

**TL;DR:** We built an ML-assisted decoder for the Gross code `[[144,12,12]]` that **beats both kit
baselines (BP-OSD and relay-BP) in accuracy and speed at every required error rate**, and extended it to
**heralded qubit loss**, where it also beats NVIDIA's TCBP-JOBS method (which cannot use loss information).
Everything is in `challenge-1-ml-decoder-kit/`. The technical write-up is
[`challenge-1-ml-decoder-kit/DECODER.md`](challenge-1-ml-decoder-kit/DECODER.md).

Nothing here is merged to `main`.

---

## What the decoder is

Our decoder is **TCBP-JOBS**, re-implemented on CPU from NVIDIA's paper *"TCBP-JOBS: GPU-Accelerated Quantum
Error Correction Decoding and Noise Learning with Near-Optimal Accuracy"* (Howe, Maurya et al.), plus our own additions:

1. **BP** (10 sum-product iterations) runs first. If it converges, we are done (~9% of shots at p=0.004).
2. **TCBP (the ML part):** a tiny neural network (529 weights) reads each fault's whole BP belief history and
   outputs a calibrated probability that it happened. It trains in about 1 minute on a laptop CPU from
   simulated data.
3. **JOBS (the search):** 16 parallel searches over jittered orderings of those probabilities (OSD + beam
   search) return the most likely correction **and a confidence score** (the gap to the next logical class).
4. **Our additions:**
   - **Loss-aware decoding:** atom loss is modelled as heralded erasure. The herald flags become per-shot
     probabilities for BP, TCBP and JOBS.
   - **Two-stage confidence decoder** (Litch's idea): a cheap first search. If it is confident, accept it;
     otherwise run the full search. **Same accuracy, 2.3× faster.**
   - **Full-DEM mode:** decode all 1728 detectors jointly (X/Z correlations) for maximum accuracy.

## Results (all comparisons on identical shots, exact McNemar tests)

Logical error rate per cycle, 10,000 shots per basis, kit noise model, kit baseline settings:

| p | BP-OSD | relay-BP | **Ours (TCBP-JOBS)** | vs relay-BP | McNemar p |
|---|---|---|---|---|---|
| 0.003 | 4.9e-4 | 6.7e-5 | **8.3e-6** | ~8× fewer errors | 0.039 |
| 0.004 | 4.5e-3 | 8.6e-4 | **3.4e-4** | 2.5× fewer | 1.7e-13 |
| 0.005 | 2.4e-2 | 6.9e-3 | **3.0e-3** | 2.3× fewer | 1.6e-91 |

**Speed** (mean ms per X+Z pair, measured during the sweep while other jobs were running; idle-machine
numbers will be in `results/timing_*.txt`):

| p | BP-OSD | relay-BP | Ours | Ours, two-stage |
|---|---|---|---|---|
| 0.004 | 185 | 192 (p99 3,243) | 52 (p99 176) | **26 (p99 137)** |

**Two-stage decoder** at p=0.004 (10,000 shots): threshold 15 → 42 failed pairs (vs 41 full search) at 26 ms
mean. The full search runs on only 4% of shots. Stage 1 alone (22 ms) still beats relay-BP (75 vs 103 failed pairs).
The threshold was chosen on these shots. It still needs to be checked on fresh shots.

**Full DEM** (first 3,000 sweep shots, p=0.004): ours **3** failed pairs vs relay-BP on the same full DEM **17**
(5.7×, McNemar p = 5e-4) and kit relay-BP **39**. This mode is slower (~0.6 s per pair on CPU).

**Qubit loss** (p=0.004 + heralded loss, 5,000 shots per basis):

| loss rate | relay-BP (no loss info) | relay-BP (loss info) | NVIDIA TCBP-JOBS as published (no loss info) | **Ours (loss-aware)** |
|---|---|---|---|---|
| 0.001 | 1.12e-3 | 1.07e-3 | 5.7e-4 | **4.3e-4** |
| 0.002 | 1.12e-3 | 1.11e-3 | 4.7e-4 | **3.3e-4** |
| 0.004 | 1.46e-3 | 1.12e-3 | 5.7e-4 | **3.0e-4** |

Ours beats the published TCBP-JOBS method by 1.3–1.9× (significant at all three rates; the lead grows with loss)
and relay-BP by 2.5–3.7×, at 5–10× less time than relay-BP.

**Confidence / postselection:** discarding the 2% least-confident pairs cuts the error rate 8× (5% → 39×) at p=0.004.

**What did not help:** routing low-confidence shots to relay-BP (on our failures, relay-BP's answer was never
more likely than ours), and summing probability per logical class (it gave the same answers).

Plots: `challenge-1-ml-decoder-kit/results/ler_and_latency.png`, `challenge-1-ml-decoder-kit/results/loss.png`.

### Honest caveats (say these in the presentation)
- This is **our re-implementation** of NVIDIA's method on CPU, not their CUDA-Q GPU code. "Beats NVIDIA" means
  "beats their method, re-implemented, under loss", not their GPU numbers.
- The kit benchmark (uniform circuit noise, per-basis decoding) is not the noise model used in NVIDIA's paper.
- Our draws run on 12 CPU threads. Single-thread timings are in `results/timing_1thread_*.txt` once that run finishes.
- At p=0.003 the counts are small (1 vs 8 failed pairs), so treat "~8×" as rough.

## How to run it

Setup (Python 3.12 + Rust for relay-BP), from `challenge-1-ml-decoder-kit/`:

```bash
conda create -n qec python=3.12 && conda activate qec
pip install -r requirements.txt
python -m pytest test_kit.py test_ours.py          # 23 kit tests + 6 of ours
```

The trained models (`tcbp_model.npz`, `tcbp_loss_model.npz`, `tcbp_full_model.npz`) are committed, so you can
decode right away:

```bash
python evaluate.py --decoder tcbp-jobs --p 0.004 --shots 10000 --out results/tcbpjobs_p0.004.npz
python evaluate.py --decoder two-stage --p 0.004 --shots 10000 --out results/twostage_p0.004.npz
python evaluate_loss.py --decoder tcbp-jobs --p 0.004 --p-loss 0.002 --shots 5000
python analyze.py                                   # tables, McNemar tests, plots
python timing.py --p 0.004 --shots 300              # latency on an idle machine
```

Full command list and a file-by-file guide: [`DECODER.md`](challenge-1-ml-decoder-kit/DECODER.md).
Per-shot records of every run are in `challenge-1-ml-decoder-kit/results/*.npz`. The BP-OSD and relay-BP baseline
totals are in `results/baseline_p*.txt`.

## Hackathon requirements checklist

- [x] Kit set up and baselines reproduced (matches the kit's reference table)
- [x] Memory experiments with noise, using loss heralding
- [x] ML-assisted decoder (TCBP network + search)
- [x] Comparison with BP-OSD and relay-BP on identical shots
- [x] Sweep at p = 0.003, 0.004, 0.005
- [x] Logical error rate + latency; plots
- [~] Resource requirements: CPU threads, model size (529 weights); single-thread timing in progress
- [x] Code on GitHub (this branch)
- [ ] Presentation / demo
- Stretch goals done: confidence routing (two-stage), latency estimates, second noise channel (loss), full-DEM decoding

## Next steps
1. Finish the idle-machine timing (running) and add those numbers.
2. Check the two-stage threshold on fresh shots.
3. Optional: run NVIDIA's real decoder (CUDA-Q QEC 0.9.0) on a free GPU (Kaggle/Colab) on our exact shots.
4. Optional: leakage tracking with clifft (unheralded loss).
5. Slides.

## Credits / references
- NVIDIA TCBP-JOBS (method we re-implemented): https://research.nvidia.com/publication/_tcbpjobs-gpu-accelerated-quantum-error-correction-decoding-and-noise-learning-near
- CoDecoder (confidence routing, joint logical classes): arXiv:2610.28748
- Evolving Decoding Algorithms for QEC (Gibran.AI): arXiv:2611.25880. Shows relay-BP beats their evolved decoders.
- Two-stage confidence decoder idea: Litch
- Relay-BP: arXiv:2506.01779. Gross code: Bravyi et al., Nature 627, 778.
