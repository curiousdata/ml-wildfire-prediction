#!/bin/bash
# FGDC weekly batch — catch-up-if-overdue wrapper for launchd. Fired DAILY (+ at load) by the LaunchAgent, but
# runs the heavy batch only if it's been >= DUE_DAYS since the last SUCCESS. This is sleep-robust: a fixed-time
# cron silently skips weeks when the laptop is asleep, whereas this no-ops cheaply until it's actually overdue,
# then runs on the next wake. Success is recorded by a heartbeat file (exit-code based) — the source of truth for
# "did the weekly refresh happen", independent of any log noise (e.g. MODIS/dask shutdown chatter).
#
# Idempotent + re-entrant: batch_job processes ALL new settled days since gold_last (so a missed week catches up
# in one run), exits "nothing to do" if already current, and the gold edge write is atomic (a crash leaves only
# fewer-day rows the date check self-heals). A lock prevents an overlapping run.
set -uo pipefail
REPO="/Users/vladimir/ml-wildfire-prediction"
PY="$REPO/.venv/bin/python"
STORE="$REPO/data/serving_store"
HEARTBEAT="$STORE/.batch_last_success"      # epoch seconds of the last successful run
LOG="$STORE/batch_cron.log"
LOCK="$STORE/.batch_lock"
DUE_DAYS=7
mkdir -p "$STORE"
cd "$REPO" || { echo "$(date -u +%FT%TZ) cannot cd $REPO" >>"$LOG"; exit 1; }
ts() { date -u +%FT%TZ; }

# single-run lock (mkdir is atomic); stale lock (>6h) is reclaimed
if ! mkdir "$LOCK" 2>/dev/null; then
  if [[ -d "$LOCK" ]] && [[ $(( $(date +%s) - $(stat -f %m "$LOCK") )) -gt 21600 ]]; then
    echo "$(ts) reclaiming stale lock" >>"$LOG"; rmdir "$LOCK" 2>/dev/null; mkdir "$LOCK" 2>/dev/null || exit 0
  else
    echo "$(ts) a run is already in progress — skip" >>"$LOG"; exit 0
  fi
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

now=$(date +%s)
if [[ -f "$HEARTBEAT" ]]; then
  age=$(( (now - $(cat "$HEARTBEAT")) / 86400 ))
  if (( age < DUE_DAYS )); then
    echo "$(ts) not due (last success ${age}d ago < ${DUE_DAYS}d) — skip" >>"$LOG"; exit 0
  fi
fi

echo "$(ts) === batch run starting ===" >>"$LOG"
if "$PY" scripts/batch_job.py >>"$LOG" 2>&1; then
  echo "$now" >"$HEARTBEAT"
  echo "$(ts) === batch run OK (heartbeat updated) ===" >>"$LOG"
else
  rc=$?
  echo "$(ts) === batch run FAILED (exit $rc) — heartbeat NOT updated ===" >>"$LOG"
  exit "$rc"
fi
