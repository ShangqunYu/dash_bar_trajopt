#!/bin/bash
#SBATCH --job-name=fbo-spoke-sweep
#SBATCH --account=pi_mehuber_umass_edu
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=.slurm-logs/fbo-sweep-%j.log

mkdir -p .slurm-logs
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"

# Run all 6 configurations in parallel, staggered by 30s to avoid render collisions
uv run scripts/run_fbo.py --num-spokes 1 --target-angle -0.5 &
sleep 30
uv run scripts/run_fbo.py --num-spokes 3 --target-angle -0.5 &
sleep 30
uv run scripts/run_fbo.py --num-spokes 6 --target-angle -0.5 &
sleep 30
uv run scripts/run_fbo.py --num-spokes 1 --target-angle 3.0 &
sleep 30
uv run scripts/run_fbo.py --num-spokes 3 --target-angle 3.0 &
sleep 30
uv run scripts/run_fbo.py --num-spokes 6 --target-angle 3.0 &

wait
echo "All jobs complete"
