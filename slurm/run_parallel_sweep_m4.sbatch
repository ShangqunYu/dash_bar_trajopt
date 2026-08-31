#!/bin/bash
#SBATCH --job-name=fbo-sweep-m4
#SBATCH --account=pi_mehuber_umass_edu
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=.slurm-logs/fbo-sweep-m4-%j.log

# Same 128-eval budget as the m=8 sweep, spent on a shallower basis schedule:
# 16 initial samples and stages m=1,2,4, so the final basis gets 64 evals
# instead of 32. Usage: sbatch run_parallel_sweep_m4.sbatch <right|both>
set -u
ARMS="${1:?pass 'right' or 'both'}"
[ "$ARMS" = "both" ] && SUFFIX="-2arms" || SUFFIX=""

cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
mkdir -p .slurm-logs outputs-2

for cfg in "1 -0.5 61p352" "3 -0.5 61p352" "6 -0.5 61p352" \
           "1 3.0 261p89" "3 3.0 261p89" "6 3.0 261p89"; do
  read -r spokes target tag <<<"$cfg"
  uv run scripts/run_fbo.py \
    --arms "$ARMS" --n-init 16 --m 4 \
    --num-spokes "$spokes" --target-angle "$target" \
    --out "outputs-2/turn-${spokes}-spoke-to-${tag}${SUFFIX}" &
  sleep 30  # stagger so the running-best renders never collide
done

wait
echo "All jobs complete"
