#!/bin/bash
#SBATCH -J fbo-dash-oct
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --mem=12G
#SBATCH -t 01:30:00
#SBATCH -o .slurm-logs/%x-%j.out

cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
# Same -132 deg target as tri-pin-hard, 8 spokes instead of 3: 45 deg spacing,
# so the isolated variable is how often a pushable spoke comes into reach.
uv run python scripts/run_fbo.py \
  --seed 0 \
  --target-angle -2.30383 \
  --num-spokes 8 \
  --out outputs/oct-pin-hard
