# Phase 0 acceptance: cached shots reproduce baseline.py

Cached seed-1 shots at p=0.004, 10,000 per basis, 5 workers in baseline.py's dealing order.

| decoder | baseline.py (`base-results.txt`) | bench.run_decoder | match |
|---|---|---|---|
| BP-OSD | [266 267], 4.50e-3 per cycle | [266 267], 4.50e-3 [4.13e-3, 4.90e-3] | exact |
| relay-BP | [51 40], 7.62e-4 per cycle | [51 40], 7.62e-4 [6.20e-4, 9.35e-4] | exact |

Timing differs: relay-BP mean 208.0 ms vs 355.8 ms, p99 3536 vs 4803 ms; BP-OSD mean
190.8 vs 228.2 ms. The base-results.txt run shared the CPU with other jobs, so its times
are inflated. All later comparisons use timings measured in this harness with nothing
else running.

Commands:

    python -m bench.make_shots --p 0.003 0.004 0.005 --shots 10000 --seed 1 1000 2000
    python -m bench.run_decoder --decoder bposd --p 0.004 --shots 10000 --workers 5
    python -m bench.run_decoder --decoder relay --p 0.004 --shots 10000 --workers 5
