#!/bin/bash
# Watch and regenerate dashboards every 1 minute for 6 hours

DIRS=(
  "outputs/turn-1-spoke-to-61p352"
  "outputs/turn-3-spoke-to-61p352"
  "outputs/turn-6-spoke-to-61p352"
  "outputs/turn-1-spoke-to-261p89"
  "outputs/turn-3-spoke-to-261p89"
  "outputs/turn-6-spoke-to-261p89"
)

END=$(($(date +%s) + 6 * 3600))  # 6 hours from now

while [ $(date +%s) -lt $END ]; do
  for dir in "${DIRS[@]}"; do
    if [ -f "$dir/log.jsonl" ]; then
      echo "[$(date '+%H:%M:%S')] Regenerating $dir"
      uv run scripts/make_fbo_dashboard.py "$dir" 2>/dev/null
    fi
  done
  sleep 60
done

echo "[$(date '+%H:%M:%S')] Watcher finished"
