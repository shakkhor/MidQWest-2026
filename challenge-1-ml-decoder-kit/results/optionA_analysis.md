# Option A (router v2): validation results

**Question:** can a better router find the hard shots where extra relay-BP search actually
helps, so that the tiered decoder beats relay-BP's own trade-off curve?

**Short answer: it helps, but not enough to make a clear win.** Router v2 picks useful
escalations 3–4× better than chance. But "will one more relay-BP search fix this shot?" is
mostly unpredictable from the first solution. The estimated operating points land on relay-BP's
`stop_nconv=1`–`2` frontier, or slightly beyond it, within statistical noise. Rich features and
gradient boosting did not beat the 5 v1 features with logistic regression.

## Setup
- **Data:** `bench/collect_hard.py`.
  - Training: seed 1000. Validation: seed 2000.
  - p=0.003 and 0.004 at 10k shots per basis, p=0.005 at 4k.
  - The test seed 1 was **not** used.
- **For every hard shot** (first BP leg not converged), we logged:
  - 17 features of the first leg and of the stage-2 solution (posterior statistics, unsatisfied
    checks after leg 1, legs used, solution weight, ...);
  - the outcome and time of two stage-3 variants: **3a**, 4 more relay solutions; **3b**, 1 more
    from a differently seeded relay run.
- **Label:** "stage 3 fixes this shot" (stage 2 wrong, stage 3 right). Training positives: 168
  for 3a, 112 for 3b. Stage 3 also *breaks* shots: 79 (3a) and 72 (3b) in the training data.
- **Models:** logistic regression vs. gradient boosting, each with the 5 v1 features ("base")
  and with all 17 ("rich"). Calibrated on validation (`bench/router_v2.py`).

## Model selection (validation, all p pooled; positives are about 2–3% of hard shots)

| stage 3 | features | model | avg. precision | hard shots flagged at 95% recall |
|---|---|---|---|---|
| 3a | base | logistic | **0.094** | 69% |
| 3a | rich | logistic | 0.093 | 68% |
| 3a | base | gboost | 0.066 | 78% |
| 3b | base | logistic | **0.101** | **51%** |
| 3b | rich | logistic | 0.081 | 55% |
| 3b | rich | gboost | 0.047 | 82% |

- **Rich features do not help.** The posterior statistics carry no extra signal about whether
  another random relay leg will do better.
- **Gradient boosting overfits** with only 110–170 positives.
- **Checkpoint 2 criterion** ("flagged fraction well below 64% at 95% recall"): only 3b-base
  gets there (51%). Not "well below".

## Router vs. random escalation (validation, base logistic)

Net fixes = shots fixed − shots broken, when escalating the top-scored fraction of hard shots.

| p, stage 3 | escalate 5% | 10% | 20% | 40% | 100% (all) |
|---|---|---|---|---|---|
| 0.004, 3a | 5 (random 1.6) | 14 (3.4) | 18 (6.6) | 29 (14.3) | 35 |
| 0.004, 3b | 4 (0.9) | 7 (1.6) | 7 (2.7) | 12 (5.9) | 15 |
| 0.005, 3a | 13 (3.4) | 12 (6.2) | 25 (13.5) | 43 (25.9) | 65 |
| 0.005, 3b | 16 (1.9) | 15 (3.8) | 14 (7.9) | 27 (16.0) | 39 |

Escalating the top 10% of hard shots recovers about 40% of the benefit of escalating all of them.
That is clearly better than chance, but it leaves most of the benefit on the table.

## Estimated position against relay-BP (p=0.004, 3a)

These estimates use the Phase 1 test timings. They are not a measured test result.

| operating point | per cycle | mean / pair |
|---|---|---|
| relay `stop_nconv=1` | 1.14e-3 | 58 ms |
| router v2, escalate 10% of hard shots | about 1.03e-3 | about 70 ms |
| router v2, escalate 40% of hard shots | about 0.89e-3 | about 102 ms |
| relay `stop_nconv=2` | 9.7e-4 [8.1e-4, 1.17e-3] | 103 ms |

Router v2 at 40% would be about 8% more accurate than `stop_nconv=2` at the same time. The 95%
interval at 10k shots is ±17%, so a timed test on 10k shots **cannot distinguish them**. It would
take about 100k shots per basis to settle this, which is a Babbage-sized run.

## Conclusions

1. **The main limit is randomness, not the router's features.** Whether one more relay-BP search
   finds a better solution behaves mostly like a coin flip, given anything visible in the first
   solution.
2. A learned per-shot stopping rule for relay-BP gives **at best a marginal gain** over simply
   choosing `stop_nconv=2`. That is a useful negative result for the write-up.
3. Escalation is not free of risk: stage 3 breaks about 40% as many shots as it fixes, because
   "lowest weight wins" is sometimes wrong.
4. **Recommendation:** stop Option A here. Move to Option B: use our harness, and relay-BP's
   curve, to give the team's TCBP-JOBS decoder fair, held-out numbers.

Reproduce:

    python -m bench.collect_hard --p 0.003 0.004 --seed 1000 --shots 10000 --workers 10
    python -m bench.collect_hard --p 0.005 --seed 1000 --shots 4000 --workers 10
    # the same with --seed 2000
    python -m bench.router_v2 --train-seed 1000 --val-seed 2000
