#!/bin/bash
#SBATCH -J fbo-dash
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --mem=12G
#SBATCH -t 01:30:00
#SBATCH -o .slurm-logs/%x-%j.out

cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
uv run python scripts/run_fbo.py --seed 0 --num-spokes 3 --out outputs/tri-pin-easy
