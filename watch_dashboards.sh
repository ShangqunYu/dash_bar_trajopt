#!/bin/bash
# Rebuild the combined sweep dashboards every minute for 6 hours.
# Each page embeds every run in its directory, so this covers all configs.

DIRS=(outputs outputs-2)
END=$(($(date +%s) + 6 * 3600))

while [ $(date +%s) -lt $END ]; do
  for dir in "${DIRS[@]}"; do
    if compgen -G "$dir/*/log.jsonl" >/dev/null; then
      echo "[$(date '+%H:%M:%S')] Rebuilding $dir/fbo_sweep.html"
      uv run scripts/make_fbo_multi_dashboard.py "$dir" 2>/dev/null | tail -1
    fi
  done
  sleep 60
done

echo "[$(date '+%H:%M:%S')] Watcher finished"
