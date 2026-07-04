---
title: Fire Guard Control Center
emoji: 🛡️
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Fire Guard Control Center — next-day wildfire risk for Spain

A live, **map-first** control center for next-day wildfire risk across mainland Spain and the Balearic Islands.
On a dark schematic base it renders:

- **Tomorrow's per-cell risk** as a heat glow (dark-red → bright yellow; brighter = hotter = more dangerous),
  calibrated and day-comparable so quiet days stay dark.
- **Today's active fire** (NASA FIRMS VIIRS) in cyan.
- **Hover-able danger areas** — watershed-clustered zones of elevated risk, each labelled with the **aggregate
  chance of a fire** there tomorrow *and* the **plain-language reasons** (e.g. *"very dry ground · extreme heat"*),
  not raw feature names.

Predictions come from a calibrated **point-wise gradient-boosted model** on the 4 km **Fire Guard Datacube**.

## How it works (read-only by design)

This Space is a **renderer**, not a compute job:

- A local **engine** (`scripts/serve.py --mode live`, run on a schedule) fetches the live feeds (Open-Meteo
  weather, FIRMS fire), runs the model, and **publishes each prediction to a Hugging Face Dataset**
  (`curiousdata/fireguard-serving`).
- This app **reads the latest published prediction** from that Dataset (a tiny `latest.json` manifest → the grid),
  plus a precomputed `display_assets.npz` (region grid + map georeferencing). No datacube, no model, no GPU at
  runtime — it loads fast and stays within the free tier.
- A background fragment auto-refreshes when a newer prediction lands, so the view stays current without a reload.

## Reading the map

- **Heat glow** — brighter/yellower = higher next-day ignition/spread probability.
- **Cyan cells** — active fire detected today.
- **Circles** — danger areas; hover for the aggregate probability + drivers.
- **◐ PRELIMINARY / ● LIVE** — morning predictions are preliminary (before the afternoon satellite pass settles
  ~17 UTC) and refine in the evening.

*Part of the **FireGuard** ecosystem (datacube · pipeline · forecaster · control center). Built with
[Claude Code](https://claude.com/claude-code).*
