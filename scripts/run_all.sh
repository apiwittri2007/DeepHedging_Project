#!/usr/bin/env bash
# Reproduce every table and chart in the repo (~3.5 hours on 2 CPUs, almost all of it the 10-seed training-data study).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q
python -m deephedge.experiments.run_sim  --config configs/sim.json  --out results/sim  --figures figures/sim
python -m deephedge.experiments.run_real --config configs/real.json --out results/real --figures figures/real --data data/raw/hk
python -m deephedge.experiments.run_training_data --config configs/training_data.json --out results/training_data --figures figures/training_data --data data/raw/hk
