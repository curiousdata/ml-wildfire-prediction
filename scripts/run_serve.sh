#!/bin/bash
# FGDC live SERVE — 2×/day wrapper for launchd (Ship A). Predicts t+1 from the forecast edge and publishes to the
# HF Dataset, so the Space refreshes itself. Mirrors run_batch.sh's robustness (heartbeat + lock + catch-up on wake)
# but is EPHEMERAL: it writes NOTHING to the cube (daily_job --mode live → extend_cube.serve_edge in memory), so a
# crash can't corrupt anything — it just doesn't update the heartbeat and the previous prediction stays live.
#
# Cadence: fired at ~08:00 and ~20:00 local (= ~06/18 UTC) by the LaunchAgent, plus RunAtLoad. The two slots give
# the progressive-refinement pattern (see the fire-label-timing memory): morning = PRELIMINARY (today's afternoon
# SNPP pass not yet settled), evening = FINAL (re-predicts the same t+1 with today's fire complete). A short
# MIN_GAP_HRS guard stops RunAtLoad + a scheduled slot from double-firing; a missed slot just runs once on wake
# (past predictions are moot — serve always predicts the LATEST, never backfills).
set -uo pipefail
REPO="/Users/vladimir/ml-wildfire-prediction"
PY="$REPO/.venv/bin/python"
STORE="$REPO/data/serving_store"
HEARTBEAT="$STORE/.serve_last_success"      # epoch seconds of the last successful serve+push
LOG="$STORE/serve_cron.log"
LOCK="$STORE/.serve_lock"
MIN_GAP_HRS=4                               # skip if a serve succeeded < this ago (dedup RunAtLoad vs a slot)
mkdir -p "$STORE"
cd "$REPO" || { echo "$(date -u +%FT%TZ) cannot cd $REPO" >>"$LOG"; exit 1; }
ts() { date -u +%FT%TZ; }

# single-run lock (mkdir is atomic); stale lock (>1h — a serve is minutes) is reclaimed
if ! mkdir "$LOCK" 2>/dev/null; then
  if [[ -d "$LOCK" ]] && [[ $(( $(date +%s) - $(stat -f %m "$LOCK") )) -gt 3600 ]]; then
    echo "$(ts) reclaiming stale lock" >>"$LOG"; rmdir "$LOCK" 2>/dev/null; mkdir "$LOCK" 2>/dev/null || exit 0
  else
    echo "$(ts) a serve is already in progress — skip" >>"$LOG"; exit 0
  fi
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

now=$(date +%s)
if [[ -f "$HEARTBEAT" ]]; then
  gap=$(( (now - $(cat "$HEARTBEAT")) / 3600 ))
  if (( gap < MIN_GAP_HRS )); then
    echo "$(ts) last serve ${gap}h ago (< ${MIN_GAP_HRS}h) — skip (dedup)" >>"$LOG"; exit 0
  fi
fi

echo "$(ts) === serve starting ===" >>"$LOG"
if "$PY" scripts/daily_job.py --mode live >>"$LOG" 2>&1 && "$PY" scripts/push_serving.py >>"$LOG" 2>&1; then
  echo "$now" >"$HEARTBEAT"
  echo "$(ts) === serve OK (predicted + pushed; heartbeat updated) ===" >>"$LOG"
else
  rc=$?
  echo "$(ts) === serve FAILED (exit $rc) — heartbeat NOT updated; previous prediction stays live ===" >>"$LOG"
  exit "$rc"
fi
