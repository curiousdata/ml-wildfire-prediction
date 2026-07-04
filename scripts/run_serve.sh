#!/bin/bash
# FGDC live SERVE — 4×/day wrapper for launchd (Ship A). Predicts t+1 from the forecast edge and publishes to the
# HF Dataset, so the Space refreshes itself. Mirrors run_weekly.sh's robustness (heartbeat + lock + catch-up on wake)
# but is EPHEMERAL: it writes NOTHING to the cube (serve --mode live → serve_engine.serve_edge in memory), so a
# crash can't corrupt anything — it just doesn't update the heartbeat and the previous prediction stays live.
#
# Cadence: fired at ~06:15, 07:15, 18:15, 19:15 local by the LaunchAgent, plus RunAtLoad. Each slot lands ~30 min
# after one of the four VIIRS passes (S-NPP + NOAA-20, night + afternoon) settles in FIRMS NRT, so every run folds
# the freshest active-fire pass into today's dist_to_fire before predicting t+1 (progressive refinement — see the
# fire-label-timing memory): the first three are PRELIMINARY (today's afternoon SNPP pass not yet settled), the
# 19:15 run (>17 UTC) is FINAL with today's fire complete. A sub-hour MIN_GAP_MIN guard stops RunAtLoad + a
# scheduled slot from double-firing while still letting the 1 h-apart pairs both run; a missed slot just runs once
# on wake (past predictions are moot — serve always predicts the LATEST, never backfills).
set -uo pipefail
REPO="/Users/vladimir/ml-wildfire-prediction"
PY="$REPO/.venv/bin/python"
STORE="$REPO/data/serving_store"
HEARTBEAT="$STORE/.serve_last_success"      # epoch seconds of the last successful serve+push
LOG="$STORE/serve_cron.log"
LOCK="$STORE/.serve_lock"
MIN_GAP_MIN=45                              # skip if a serve succeeded < this many minutes ago (dedup RunAtLoad vs a
                                            # slot); sub-hour so the 1 h-apart pass-pair slots each still fire
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
  gap=$(( (now - $(cat "$HEARTBEAT")) / 60 ))
  if (( gap < MIN_GAP_MIN )); then
    echo "$(ts) last serve ${gap}min ago (< ${MIN_GAP_MIN}min) — skip (dedup)" >>"$LOG"; exit 0
  fi
fi

echo "$(ts) === serve starting ===" >>"$LOG"
if "$PY" scripts/serve.py --mode live >>"$LOG" 2>&1 && "$PY" scripts/push_predictions.py >>"$LOG" 2>&1; then
  echo "$now" >"$HEARTBEAT"
  echo "$(ts) === serve OK (predicted + pushed; heartbeat updated) ===" >>"$LOG"
else
  rc=$?
  echo "$(ts) === serve FAILED (exit $rc) — heartbeat NOT updated; previous prediction stays live ===" >>"$LOG"
  exit "$rc"
fi
