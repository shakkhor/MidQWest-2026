#!/usr/bin/env bash
# Phase 1: relay-BP's own speed vs. accuracy curve, and the tiered decoder, on identical shots.
# Run from the kit directory with the venv active:  bash bench/phase1.sh
# Timed runs use 4 workers (the M4's performance cores); run nothing else meanwhile.
set -euo pipefail

# Router with training scores saved (seeds 1000+, never the test seed 1). Not timed.
python train_router.py --shots 1500 --workers 10 --out router.npz

run() { python -m bench.run_decoder --workers 4 "$@"; }

for spec in "0.004 10000" "0.005 3000"; do
  read -r p shots <<< "$spec"
  for nconv in 1 2 5; do
    for sets in 300 50; do
      run --decoder relay --config "stop_nconv=$nconv,num_sets=$sets" --p "$p" --shots "$shots"
    done
  done
  run --decoder tiered-all --router router.npz --p "$p" --shots "$shots"
  [ "$p" = "0.004" ] && run --decoder bposd --p "$p" --shots "$shots"
  python -m bench.pareto --p "$p" --router router.npz
done
