#!/usr/bin/env bash
# Reproduce every table and chart in the repo (~1 minute on a laptop).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q
python -m deephedge.experiments.run_sim  --config configs/sim.json  --out results/sim  --figures figures/sim
python -m deephedge.experiments.run_real --config configs/real.json --out results/real --figures figures/real --data data/raw/hk
