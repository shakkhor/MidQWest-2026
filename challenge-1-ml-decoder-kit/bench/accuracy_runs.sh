#!/usr/bin/env bash
# Large accuracy runs for a many-core machine (e.g. Babbage). Failure counts only: the times
# in these logs come from a different CPU and must not be mixed with the Mac timing runs.
# From the kit directory with the venv active:
#   bash bench/accuracy_runs.sh                 # p=0.003, 100k shots
#   P="0.004 0.005" SHOTS=30000 bash bench/accuracy_runs.sh
# Uses seed 3, which no other run uses. Logs go to logs/, summaries to results/accuracy_seed3.txt.
set -euo pipefail

P=${P:-0.003}
SHOTS=${SHOTS:-100000}
SEED=3
WORKERS=${WORKERS:-$(python -c "import os; print(len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count())")}
echo "workers: $WORKERS, p: $P, shots per basis: $SHOTS, seed: $SEED"

for p in $P; do
  python -m bench.make_shots --p "$p" --shots "$SHOTS" --seed "$SEED"
  run() { python -m bench.run_decoder --workers "$WORKERS" --p "$p" --seed "$SEED" --shots "$SHOTS" "$@"; }
  run --decoder relay
  run --decoder relay --config stop_nconv=1,num_sets=300
  run --decoder tiered-all --router router.npz
done 2>&1 | grep -v RuntimeWarning | tee -a results/accuracy_seed3.txt
