---
title: Fire Guard
emoji: 🛡️
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
license: mit
---

# Fire Guard — wildfire control center (Spain)

A next-day wildfire-risk monitor for Spain. The map shows **today's active fire** (FIRMS) over fresh
VIIRS true-color imagery, **tomorrow's predicted ignition/spread risk** (a calibrated point-wise
gradient-boosted model on the IberFire 4 km grid), the **regions most at risk**, and the **biggest
drivers** behind today's prediction.

## How it works (read-only by design)

This Space is a **renderer**, not a compute job:

- An **engine** (`scripts/daily_job.py --mode live` in the project repo, run on a schedule) fetches the
  live feeds (Open-Meteo weather, FIRMS fire), runs the model, and publishes a prediction grid.
- This app **reads the latest published grid** from `store/grids/*.npz` plus a small precomputed
  `display_assets.npz` (region grid + map georeferencing) and `gbt_coarse4.importance.json`. No model,
  no datacube, no GPU — so it loads fast and stays within the free tier.
- A background fragment auto-reruns when a newer grid lands, so the view stays current without a manual
  reload.

## Honesty about the feed

The model was trained on the IberFire datacube; the live feed currently refreshes weather + active fire,
while some slower features (vegetation, soil, fire-history) may be seasonally **warm-started** — the app
shows a **degraded-inputs banner** whenever that's the case, so risk is never presented as more current
than it is. Absolute risk tightens as more of the pipeline is fed live.

*Built with [Claude Code](https://claude.com/claude-code).*
