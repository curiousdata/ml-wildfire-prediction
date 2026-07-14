# FireGuard Datacube (FGDC v2) — Feature Reference

Description of the variables in the Fire Guard Datacube. This documents the **datacube**, which is broader than
any single model: §1–7 are the **135 features the production model uses** (the authoritative list is
[`src/data/features.py`](src/data/features.py), `FGDC_FEATURE_VARS` — if the two disagree, the code wins), and
[§8](#8-inactive-layers--in-the-cube-not-in-the-current-model) documents variables that are **materialized in
the cube but not currently fed to the model**. The active set is a curated *subset*, not the whole cube, and
is expected to change (see §8).

- **Task.** Per-cell next-day fire occurrence: features at day *t* → binary fire label at *t + 1*.
- **Grid.** 2 km × 2 km daily cells over peninsular Spain + Balearic Islands, EPSG:3035 (production grid;
  the pipeline also builds 4 km). Working history 2012 → present.
- **Feature count.** **135**, in a fixed, load-bearing order (the model is trained on exactly this order).
- **Label (not a feature).** `is_fire` — 1 if ≥ 1 VIIRS 375 m active-fire detection (confidence ≥ nominal)
  falls in the cell that UTC day. Max-pooled on coarsening to preserve rare positives.

## Reading this document

- **Type** — `static` (time-invariant) or `daily` (one grid per day). Two "daily" layers (`popdens`,
  `built_s`) are in fact slow-varying, linearly interpolated between multi-year census epochs.
- **Range** — the **1st–99th percentile over land**, sampled across the full record (a robust spread, not the
  hard min/max). Units in the stored cube attributes are sparse; units below are the physical units of the
  source variable.
- **Pooling** — on 1 km → working-grid coarsening, `*_max` features are max-pooled, `*_min` min-pooled, the
  label max-pooled, everything else mean-pooled.
- **Provenance tags** — **[raw]** ingested from a provider; **[engineered]** derived in
  [`scripts/build_features.py`](scripts/build_features.py) /
  [`src/data/feature_engineering.py`](src/data/feature_engineering.py); **[static/inherited]** time-invariant
  layers refined from the source dataset.

Features are grouped by family for readability; the **canonical order is alphabetical**, as in the code.

---

## 1. Weather — ERA5 reanalysis via Open-Meteo · [raw] · daily (21 features)

Daily aggregates of hourly ERA5 fields, regridded from the native ~25 km reanalysis to the working grid.
Wind is decomposed to u/v components at the hourly level *before* daily averaging (a circular-mean-safe form).

| feature | unit | range (p1–p99) | description |
|---|---|---|---|
| `t2m_mean` | °C | −0.2 – 30.5 | daily mean 2 m air temperature |
| `t2m_max` | °C | 3.9 – 38.4 | daily maximum 2 m temperature |
| `t2m_min` | °C | −4.4 – 23.4 | daily minimum 2 m temperature |
| `t2m_range` | °C | 2.4 – 19.2 | diurnal temperature range (max − min) |
| `RH_mean` | % | 25.9 – 92.9 | daily mean relative humidity |
| `RH_min` | % | 12.4 – 85.4 | daily minimum RH (afternoon dryness; a fire driver) |
| `RH_max` | % | 41.6 – 100 | daily maximum RH |
| `RH_range` | % | 10.4 – 67.7 | RH diurnal range |
| `wind_speed_mean` | m/s | 0.9 – 7.3 | daily mean 10 m wind speed |
| `wind_speed_max` | m/s | 1.7 – 9.7 | daily maximum 10 m wind speed |
| `wind_u_mean` | m/s | −4.3 – 5.5 | daily mean eastward wind component |
| `wind_v_mean` | m/s | −4.9 – 5.1 | daily mean northward wind component |
| `wind_u_atmaxspeed` | m/s | −6.0 – 7.6 | eastward component at the hour of peak wind |
| `wind_v_atmaxspeed` | m/s | −6.9 – 7.4 | northward component at the hour of peak wind |
| `surface_pressure_mean` | hPa | 855 – 1020 | daily mean surface pressure (elevation-dependent) |
| `surface_pressure_min` | hPa | 852 – 1010 | daily minimum surface pressure |
| `surface_pressure_max` | hPa | 858 – 1020 | daily maximum surface pressure |
| `surface_pressure_range` | hPa | 1.5 – 11.8 | surface-pressure diurnal range |
| `total_precipitation_mean` | mm | 0 – 0.92 | mean hourly precipitation (daily total ≈ this × 24) |
| `soil_moisture_mean` | m³/m³ | 0.07 – 0.46 | volumetric soil moisture, 7–28 cm layer |
| `soil_temperature_mean` | °C | 1.4 – 30.6 | soil temperature, 7–28 cm layer |

> **Note on `total_precipitation_mean`:** stored as an hourly mean (mm/h), following the source convention;
> the daily total is ≈ 24×. Downstream drought features scale it accordingly.

## 2. Vegetation & land surface — MODIS via Planetary Computer · [raw]/[engineered] · daily (8)

16-day MODIS composites interpolated to daily; anomalies and cover fraction derived from them.

| feature | source | unit | range | description |
|---|---|---|---|---|
| `NDVI` | raw | index | 0.14 – 0.83 | Normalized Difference Vegetation Index (greenness) |
| `EVI` | raw | index | 0.09 – 0.57 | Enhanced Vegetation Index |
| `LAI` | raw | m²/m² | 0.1 – 4.6 | Leaf Area Index |
| `FAPAR` | raw | fraction | 0.08 – 0.85 | fraction of absorbed photosynthetically active radiation |
| `LST` | raw | K | 276 – 322 | land-surface temperature |
| `fvc` | engineered | fraction | 0.01 – 0.92 | fractional vegetation cover from NDVI (Carlson & Ripley 1997) |
| `ndvi_anomaly` | engineered | z-score | −7.9 – 6.6 | NDVI vs prior-years day-of-year climatology (causal; NaN before 2 priors) |
| `lai_anomaly` | engineered | z-score | −5.8 – 7.4 | LAI seasonal anomaly (causal) |

## 3. Fire-weather & drought indices — [engineered] · daily (10)

Physically-grounded fire-danger indices computed from the weather + precipitation fields.

| feature | unit | range | description |
|---|---|---|---|
| `kbdi` | index (0–203) | 0 – 168 | Keetch–Byram Drought Index — recursive soil-moisture-deficit accumulator; builds on hot/dry days, drops after rain |
| `spi_90d` | z-score | −3.3 – 6.2 | standardized 90-day precipitation anomaly vs prior-years climatology (SPI proxy; causal) |
| `ffwi` | index | 4.9 – 40.3 | Fosberg Fire Weather Index (fuel moisture + wind); fast-reacting |
| `emc_peak` | % | 2.8 – 18.8 | equilibrium (1-hr dead-fuel) moisture from `t2m_max`/`RH_min`; lower = drier |
| `precip_sum_7d` | mm | 0 – 3.4 | trailing 7-day precipitation sum (short-term surface dryness) |
| `precip_sum_30d` | mm | 0.002 – 10.2 | trailing 30-day precipitation sum |
| `precip_sum_90d` | mm | 0.17 – 23.2 | trailing 90-day precipitation sum (seasonal dryness; drives `spi_90d`) |
| `precip_sum_180d` | mm | 0.69 – 41.1 | trailing 180-day precipitation sum |
| `precip_sum_365d` | mm | 0.70 – 69.2 | trailing 365-day precipitation sum (annual dryness) |

> Precip sums are in the same hourly-mean units as `total_precipitation_mean` (a trailing cumulative window).
> KBDI is unbounded-below-capped at 203.2 mm-equivalent; the range above reflects the working record.

## 4. Fire context — [engineered from the fire label] · daily (4)

Spatial/temporal fire-history features derived from the `is_fire` history. **Causal** (use only day ≤ *t*),
so they predict *t + 1* without leakage; these are among the model's strongest drivers.

| feature | unit | range | description |
|---|---|---|---|
| `dist_to_fire` | km | 9 – 1500 | distance to the nearest active-fire cell on day *t* (0 on fire cells; grid-diagonal on fireless days). Also defines the ignition/spread regime split at 6 km |
| `fire_upwind_exposure` | — | −1.3e-4 – 1.2e-4 | downwind-exposure scalar (W·d)/‖d‖² to the nearest fire; > 0 if the cell is downwind of a nearby fire, < 0 upwind |
| `time_since_last_fire` | days | 11 – 5250 | consecutive days since the cell last burned (0 on a fire day) |
| `burn_frequency_365d` | count | 0 – 1 | number of fire-days in the cell over the trailing 365 days |

## 5. Terrain — Copernicus DEM, [static/inherited] except where noted (20)

Elevation-derived terrain descriptors (static) plus an 8-sector aspect encoding.

| feature | type | unit | range | description |
|---|---|---|---|---|
| `elevation_mean` | static | m | 14 – 1800 | mean elevation |
| `elevation_stdev` | static | m | 1.3 – 110 | sub-cell elevation variability (ruggedness) |
| `slope_mean` | static | ° | 0.5 – 24.8 | mean slope |
| `slope_stdev` | static | ° | 0.4 – 9.0 | slope variability |
| `roughness_mean` | static | m | 0.9 – 38.1 | terrain roughness (local elevation relief) |
| `roughness_stdev` | static | m | 0.6 – 15.2 | roughness variability |
| `tpi` | static | m | −172 – 218 | Topographic Position Index (elevation − local 5×5 mean); ridge > 0, valley < 0 |
| `terrain_curvature` | static | — | −569 – 512 | curvature (Laplacian of elevation); convex > 0, concave < 0 |
| `hli` | static | 0–1 | 0.51 – 1.0 | McCune–Keon Heat Load Index (slope + aspect + latitude); terrain solar load |
| `aspect_1` … `aspect_8` | static | fraction | 0 – 1 | sub-cell aspect one-hots — proportion of the cell facing each 45° compass sector (1 = N, clockwise) |
| `aspect_NODATA` | static | fraction | 0 – 1 | proportion of the cell with undefined aspect (flat / no-data) |
| `aspect_eastness` | static | −1…1 | −0.39 – 0.40 | continuous eastness reconstructed from the aspect sectors (+1 = E, −1 = W) |
| `aspect_southness` | static | −1…1 | −0.49 – 0.51 | continuous southness (+1 = S, −1 = N) |

## 6. Human activity & infrastructure — GHSL / OpenStreetMap / CORINE / Natura 2000 (10)

Ignition-pressure proxies. Human features dominate the **new-ignition** regime in ablations.

| feature | source | type | unit | range | description |
|---|---|---|---|---|---|
| `popdens` | GHS-POP | daily* | persons/km² | 0 – 1530 | population density (interpolated between census epochs) |
| `built_s` | GHS-BUILT-S | daily* | m²/cell | 0 – 135000 | built-up surface area per cell (interpolated) |
| `dist_to_roads_mean` | OSM | static | km | 0.26 – 6.4 | mean distance to the nearest road |
| `dist_to_roads_stdev` | OSM | static | km | 0.15 – 0.29 | sub-cell variability of road distance (a strong ignition driver in ablations) |
| `dist_to_railways_mean` | OSM | static | km | 0.008 – 0.71 | mean distance to the nearest railway |
| `dist_to_railways_stdev` | OSM | static | km | 0.002 – 0.003 | railway-distance variability |
| `dist_to_waterways_mean` | OSM | static | km | 0.26 – 11 | mean distance to the nearest waterway |
| `dist_to_waterways_stdev` | OSM | static | km | 0.15 – 0.29 | waterway-distance variability |
| `dist_to_urban` | engineered | static | km | 2 – 136 | WUI proxy: distance to the nearest CORINE-artificial (> 0.5) cell |
| `is_natura2000` | Natura 2000 | static | 0/1 | 0 – 1 | inside a Natura 2000 protected area |

\* `popdens`/`built_s` carry a time dimension but change only between multi-year GHSL epochs (linearly interpolated).

## 7. Land cover — CORINE Land Cover 2018 · [static/inherited] (63)

Two representations of the same CLC 2018 map: 44 fine class proportions + 19 aggregate proportions. Only the
**2018** edition is used (the 2006/2012 editions are excluded as stale for a 2022→present serving window).

### 7a. CLC level-3 class proportions — `CLC_2018_1` … `CLC_2018_44` (44)

Each is the **proportion of the cell** covered by one CORINE level-3 class; the 44 sum to ≈ 1.0 per land cell
(verified). The numeric index follows the source dataset's encoding, which is the standard CORINE legend
ordered by class code:

| idx | CLC code | class | idx | CLC code | class |
|---|---|---|---|---|---|
| 1 | 111 | Continuous urban fabric | 23 | 311 | Broad-leaved forest |
| 2 | 112 | Discontinuous urban fabric | 24 | 312 | Coniferous forest |
| 3 | 121 | Industrial/commercial units | 25 | 313 | Mixed forest |
| 4 | 122 | Road & rail networks | 26 | 321 | Natural grasslands |
| 5 | 123 | Port areas | 27 | 322 | Moors & heathland |
| 6 | 124 | Airports | 28 | 323 | Sclerophyllous vegetation |
| 7 | 131 | Mineral extraction sites | 29 | 324 | Transitional woodland-shrub |
| 8 | 132 | Dump sites | 30 | 331 | Beaches, dunes, sands |
| 9 | 133 | Construction sites | 31 | 332 | Bare rocks |
| 10 | 141 | Green urban areas | 32 | 333 | Sparsely vegetated areas |
| 11 | 142 | Sport & leisure facilities | 33 | 334 | Burnt areas |
| 12 | 211 | Non-irrigated arable land | 34 | 335 | Glaciers & perpetual snow |
| 13 | 212 | Permanently irrigated land | 35 | 411 | Inland marshes |
| 14 | 213 | Rice fields | 36 | 412 | Peat bogs |
| 15 | 221 | Vineyards | 37 | 421 | Salt marshes |
| 16 | 222 | Fruit trees & berry plantations | 38 | 422 | Salines |
| 17 | 223 | Olive groves | 39 | 423 | Intertidal flats |
| 18 | 231 | Pastures | 40 | 511 | Water courses |
| 19 | 241 | Annual + permanent crops | 41 | 512 | Water bodies |
| 20 | 242 | Complex cultivation patterns | 42 | 521 | Coastal lagoons |
| 21 | 243 | Agriculture + natural vegetation | 43 | 522 | Estuaries |
| 22 | 244 | Agro-forestry areas | 44 | 523 | Sea & ocean |

> The index→class correspondence above is the standard CORINE level-3 legend by code. Treat it as the
> reference; for any single index, confirm against the source dataset's encoding before relying on it.

### 7b. CLC aggregate proportions — named groups (19)

Coarser thematic aggregates of the level-3 classes (redundant with 7a, retained for driver interpretability).
All are cell-cover fractions in [0, 1]; typical land means in parentheses.

`CLC_2018_agricultural_proportion` (0.48) · `CLC_2018_forest_and_semi_natural_proportion` (0.48) ·
`CLC_2018_arable_land_proportion` (0.25) · `CLC_2018_scrub_proportion` (0.25) ·
`CLC_2018_forest_proportion` (0.22) · `CLC_2018_heterogeneous_agriculture_proportion` (0.12) ·
`CLC_2018_permanent_crops_proportion` (0.09) · `CLC_2018_artificial_proportion` (0.02) ·
`CLC_2018_open_space_proportion` (0.02) · `CLC_2018_urban_fabric_proportion` (0.01) ·
`CLC_2018_waterbody_proportion` · `CLC_2018_industrial_proportion` · `CLC_2018_inland_waters_proportion` ·
`CLC_2018_marine_waters_proportion` · `CLC_2018_maritime_wetlands_proportion` · `CLC_2018_mine_proportion` ·
`CLC_2018_wetlands_proportion` · `CLC_2018_artificial_vegetation_proportion` ·
`CLC_2018_inland_wetlands_proportion`.

---

## 8. Inactive layers — in the cube, not in the current model

These variables are **materialized in the datacube but excluded from the production feature set**
(`FGDC_FEATURE_VARS`). They are documented here because this is a datacube reference, and because the active
set is not fixed: features can be re-measured and re-activated.

> **Why they may come back (a working hypothesis).** Feature importance was measured at the **4 km** grid,
> where several spatially-varying fire-weather couplings tested flat and were dropped. A plausible reason is
> that 4 km block-averaging *smooths away* the local extremes these indices encode (peak VPD, hot-dry-windy
> co-occurrence). At **finer resolution** (2 km → 1 km) that averaging is weaker, so importance should be
> **re-measured per resolution** — some of these may regain predictive power and be re-added. This does not
> apply to the aspatial calendar features (they are constant in space, so coarsening cannot erode them; they
> tested flat for a different reason).

### 8a. Fire-weather couplings (candidates for re-activation at fine resolution)

| feature | unit | range (p1–p99) | description | status |
|---|---|---|---|---|
| `vpd_peak` | kPa | 0.17 – 5.72 | Vapor Pressure Deficit from `t2m_max`/`RH_min`; the atmosphere's drying power (higher = drier) | inactive — ablated flat at 4 km |
| `hdw` | index | 0.57 – 28.9 | Hot-Dry-Windy Index (VPD × `wind_speed_max`); couples atmospheric dryness with wind | inactive — ablated flat at 4 km |

Both are physically standard fire-danger inputs (Srock et al. 2018 for HDW) and partially overlap the retained
`ffwi`/`emc_peak`/`kbdi`; the ablation is recorded in [`ABLATIONS.md`](ABLATIONS.md) (2026-06-23).

### 8b. Calendar & holidays (aspatial; dropped on merit, not resolution)

Materialized and ablated **flat-to-negative** — human-ignition timing is already proxied by `popdens`/roads,
and the fire label smears the weekly ignition signal across multi-day burns. Present in the cube for a possible
future calendar × forecast-weather interaction test, but not resolution-sensitive.

`doy_sin`, `doy_cos` (day-of-year cyclic encoding, target day *t+1*) · `dow_sin`, `dow_cos`, `dow_sin_tp1`,
`dow_cos_tp1` (day-of-week cyclic, feature and target day) · `is_holiday_national`, `is_holiday_national_tp1`
(Spain-wide public holidays) · `is_holiday_regional`, `is_holiday_regional_tp1` (autonomous-community holidays).

### 8c. Not features by construction

- **`is_fire`** — the fire grid. `is_fire[t+1]` is the prediction **target** and is never an input.
  `is_fire[t]` (today's fire) is **not a raw feature column** either — but its information *is* given to the
  model, encoded by the §4 fire-context features derived from it: `dist_to_fire` is exactly 0 on today's fire
  cells (so `dist_to_fire == 0 ⟺ is_fire[t] == 1` — today's fire is fully recoverable), plus
  `fire_upwind_exposure`, `time_since_last_fire`, and `burn_frequency_365d`. The raw binary mask is withheld;
  today's fire enters as distance/recency, which is the spatially-informative form for a point-wise model.
- **`is_spain` / `is_sea` / `is_waterbody`** — subsetting masks, not predictors.
- **`AutonomousCommunities`** — a region ID (reporting only; an ordinal code is meaningless as a feature).
- **`CLC_2006_*` / `CLC_2012_*`** — stale land-cover editions; only 2018 is kept for the serving window.

---

## 9. IberFire (v1) inputs not carried into FGDC — the serve-availability filter

FGDC's founding principle is **train = serve by construction**: every feature is collected from the same
operational feed at training time and at serving time. The predecessor dataset, **IberFire** (v1), was
assembled from the *publication's* sources, not from operationally re-collectable feeds — so several of its
inputs could not be reproduced same-day at serve time. FGDC either **replaced them with an operational
equivalent** or **dropped them**. This is not a quality judgment on IberFire (an excellent research dataset);
it is a deployment constraint.

The failure this prevents was real and measured: v1 trained its fire features on **EFFIS burned-area** but the
live app could only serve **FIRMS active-fire** same-day — a different physical quantity — and the
prediction-to-offline correlation **collapsed to 0.10** (see [`CHANGES.md`](CHANGES.md), the 0.10 fire-source
bug). FGDC exists to make that class of mismatch impossible.

| IberFire (v1) input | why not serveable as-is | FGDC resolution |
|---|---|---|
| **Fire label — EFFIS burned-area (> 5 ha, perimeters)** | published with lag + licensing; the open WFS backend was unreliable/down; not same-day, and a different quantity than what's available live | **VIIRS 375 m active fire (NASA FIRMS NRT)** — same source at train and serve; also the source for all fire-context features (§4) |
| **Soil Water Index (Copernicus SWI, 4 depths)** | CLMS SWI product not fetched from the same feed used live | **ERA5 `soil_moisture_mean` / `soil_temperature_mean`** via Open-Meteo (operational, identical train/serve) |
| **Fire Weather Index (Canadian FWI)** | carried pre-computed from the publication, not re-derived from an operational feed | **self-computed indices** (`kbdi`, `ffwi`, `emc_peak`) from operational ERA5 weather — full control of the train=serve path |
| **Land-surface temperature (stitched)** | v1 LST stitched ERA5 skin-temp → CLMS v1 → v2 with instrument breakpoints (a train/test inhomogeneity) | **single-source MODIS `LST`** via Planetary Computer (§2) — same product throughout |

Static layers (terrain, CORINE, OSM distances, population) *are* inherited from IberFire — they don't drift
between train and serve (a road's distance is the same today as at publication), so re-collecting them buys
nothing. Only the **dynamic** families, the ones that move day to day, had to be recollected from operational
providers. See the `iberfire-v1-reference` notes and [`CHANGES.md`](CHANGES.md) for the full lineage.

---

## Provenance & references

| Family | Source |
|---|---|
| Weather | ERA5 / ERA5-Land reanalysis (ECMWF), via Open-Meteo — Muñoz-Sabater et al. 2021, *ESSD* 13:4349 |
| Fire label | NASA FIRMS VIIRS 375 m active fire (S-NPP) — Schroeder et al. 2014, *RSE* 143:85 |
| Vegetation / LST | MODIS (NDVI/EVI, LAI/FAPAR, LST) via Microsoft Planetary Computer |
| Population / built-up | JRC Global Human Settlement Layer (GHS-POP, GHS-BUILT-S) |
| Land cover | CORINE Land Cover 2018 (Copernicus) |
| Terrain | Copernicus DEM (elevation → slope, aspect, roughness, TPI, curvature, HLI) |
| Roads / railways / waterways | OpenStreetMap via Geofabrik |
| Protected areas | Natura 2000 |
| Fire-danger indices | KBDI (Keetch & Byram 1968), FFWI (Fosberg 1978), EMC (Simard 1968), SPI (McKee et al. 1993), FVC (Carlson & Ripley 1997), HLI (McCune & Keon 2002), TPI (Weiss 2001) |

The static terrain / land-cover / infrastructure layers are inherited (refined) from the **IberFire** dataset
(Erzibengoa et al. 2025, arXiv:2505.00837; Zenodo 15225886); the dynamic families (weather, fire, vegetation,
population) are recollected from the operational feeds above so that training and serving draw from identical
sources.

---

*Generated from `data/gold/FireGuard_coarse2.zarr` and `src/data/features.py`. Ranges are 1st–99th percentile
over land across the full record. To regenerate the counts/ranges, sample the cube as in the commit that added
this file. Keep this document in sync with `FGDC_FEATURE_VARS` — the code list is authoritative.*
