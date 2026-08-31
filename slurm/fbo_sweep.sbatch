#!/bin/bash
#SBATCH -J fbo-sweep
#SBATCH -a 0-7
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH -N 1
#SBATCH --mem=12G
#SBATCH -t 02:00:00
#SBATCH -o .slurm-logs/%x-%A_%a.out

cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
uv run python scripts/run_fbo.py \
  --seed "$SLURM_ARRAY_TASK_ID" \
  --num-spokes 3 \
  --out "outputs/tri-pin-easy-s$SLURM_ARRAY_TASK_ID"
