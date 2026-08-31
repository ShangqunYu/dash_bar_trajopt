#!/bin/bash
#SBATCH --job-name=fbo-spoke-sweep-2arms
#SBATCH --account=pi_mehuber_umass_edu
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=.slurm-logs/fbo-sweep-2arms-%j.log

mkdir -p .slurm-logs
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"

# Same 6 configurations, both arms driven, staggered by 30s to avoid render collisions
uv run scripts/run_fbo.py --arms both --num-spokes 1 --target-angle -0.5 --out outputs/turn-1-spoke-to-61p352-2arms &
sleep 30
uv run scripts/run_fbo.py --arms both --num-spokes 3 --target-angle -0.5 --out outputs/turn-3-spoke-to-61p352-2arms &
sleep 30
uv run scripts/run_fbo.py --arms both --num-spokes 6 --target-angle -0.5 --out outputs/turn-6-spoke-to-61p352-2arms &
sleep 30
uv run scripts/run_fbo.py --arms both --num-spokes 1 --target-angle 3.0 --out outputs/turn-1-spoke-to-261p89-2arms &
sleep 30
uv run scripts/run_fbo.py --arms both --num-spokes 3 --target-angle 3.0 --out outputs/turn-3-spoke-to-261p89-2arms &
sleep 30
uv run scripts/run_fbo.py --arms both --num-spokes 6 --target-angle 3.0 --out outputs/turn-6-spoke-to-261p89-2arms &

wait
echo "All jobs complete"
