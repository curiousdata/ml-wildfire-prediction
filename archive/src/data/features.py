"""Canonical IberFire feature set — the single source of truth.

Both training (``scripts/train.py``) and serving (``docker/monolith/app.py``)
import ``FEATURE_VARS`` from here, so the channel layout can never silently
drift between train and serve.

Channel ORDER is load-bearing: the shipped models (e.g. ``models/resnet34_v9.pth``)
and the normalization stats under ``stats/`` were trained with exactly this
order. Reordering or inserting a feature in the middle silently corrupts any
checkpoint loaded with ``strict=True``. Treat this list as append-only unless
you intend to retrain.

``is_near_fire`` was deliberately removed (suspected next-day leakage) and is
kept as a comment below so it is not re-added by accident.

The sub-groups are exposed as named constants so future work — e.g. collapsing
the 44 CLC one-hots into the ~19 proportions — is a one-line change here rather
than a hunt through two files.
"""

from typing import List

# Dynamic features (time-dependent: fire-weather, meteorology, vegetation indices)
DYNAMIC_VARS: List[str] = [
    "FAPAR",
    "FWI",
    "LAI",
    "LST",
    "NDVI",
    "RH_max",
    "RH_mean",
    "RH_min",
    "RH_range",
    "SWI_001",
    "SWI_005",
    "SWI_010",
    "SWI_020",
    "is_holiday",
    # "is_near_fire",  # removed: suspected next-day leakage — do not re-add
    "surface_pressure_max",
    "surface_pressure_mean",
    "surface_pressure_min",
    "surface_pressure_range",
    "t2m_max",
    "t2m_mean",
    "t2m_min",
    "t2m_range",
    "total_precipitation_mean",
    "wind_direction_at_max_speed",
    "wind_direction_mean",
    "wind_speed_max",
    "wind_speed_mean",
]

# CORINE Land Cover Level-3 one-hot classes (year-aware bases: 2006 / 2012 / 2018)
CLC_LEVEL3_VARS: List[str] = [f"CLC_{i}" for i in range(1, 45)]

# CORINE aggregated proportions (year-aware bases)
CLC_PROPORTION_VARS: List[str] = [
    "CLC_agricultural_proportion",
    "CLC_arable_land_proportion",
    "CLC_artificial_proportion",
    "CLC_artificial_vegetation_proportion",
    "CLC_forest_and_semi_natural_proportion",
    "CLC_forest_proportion",
    "CLC_heterogeneous_agriculture_proportion",
    "CLC_industrial_proportion",
    "CLC_inland_waters_proportion",
    "CLC_inland_wetlands_proportion",
    "CLC_marine_waters_proportion",
    "CLC_maritime_wetlands_proportion",
    "CLC_mine_proportion",
    "CLC_open_space_proportion",
    "CLC_permanent_crops_proportion",
    "CLC_scrub_proportion",
    "CLC_urban_fabric_proportion",
    "CLC_waterbody_proportion",
    "CLC_wetlands_proportion",
]

# Topographic aspect one-hot (8 compass sectors + NODATA)
ASPECT_VARS: List[str] = [
    "aspect_1",
    "aspect_2",
    "aspect_3",
    "aspect_4",
    "aspect_5",
    "aspect_6",
    "aspect_7",
    "aspect_8",
    "aspect_NODATA",
]

# Other static features (topography, human-activity distances, masks)
STATIC_VARS: List[str] = [
    "dist_to_railways_mean",
    "dist_to_railways_stdev",
    "dist_to_roads_mean",
    "dist_to_roads_stdev",
    "dist_to_waterways_mean",
    "dist_to_waterways_stdev",
    "elevation_mean",
    "elevation_stdev",
    "is_natura2000",
    "is_sea",
    "is_spain",
    "is_waterbody",
    "roughness_mean",
    "roughness_stdev",
    "slope_mean",
    "slope_stdev",
]

# Year-aware population density (resolved to popdens_YYYY family at read time)
POPDENS_VARS: List[str] = ["popdens"]

# The flat, ordered list every consumer should import.
FEATURE_VARS: List[str] = (
    DYNAMIC_VARS
    + CLC_LEVEL3_VARS
    + CLC_PROPORTION_VARS
    + ASPECT_VARS
    + STATIC_VARS
    + POPDENS_VARS
)

# Guard against accidental duplicates introduced during edits.
assert len(FEATURE_VARS) == len(set(FEATURE_VARS)), "Duplicate feature names in FEATURE_VARS"


# ---------------------------------------------------------------------------
# Segmentation-model feature set (the coarse4 rebuild — NOT the shipped resnet34_v9)
# ---------------------------------------------------------------------------
# Built from the cube's actual variables, applying the principled exclusions agreed for
# the rebuild (these are leakage/memorisation/redundancy calls, NOT importance pruning):
#   - identity/position (AutonomousCommunities; coordinates already dropped from the cube),
#   - mask redundancy (is_sea = inverse of the land mask we already apply),
#   - year-aware CLC/popdens collapse to BASE names (BaseIberFireDataset resolves the
#     calendar-year-appropriate map at read time) — feeding all years = stale data.
# Everything else with potential signal stays (per the no-prune principle), incl. is_fire(t)
# as a feature (current fire state) and the GBT-near-zero terrain/fuel-moisture features.

# Excluded by name (not signal judgments).
SEG_EXCLUDE: List[str] = ["is_sea", "AutonomousCommunities"]


def build_segmentation_features(data_vars) -> List[str]:
    """Return the segmentation model's feature list (base names) from a cube's data_vars.

    CLC_{2006,2012,2018}_X -> base 'CLC_X'; popdens_YYYY -> 'popdens'; everything else
    kept by exact name. Excludes SEG_EXCLUDE. Deterministic (sorted).
    """
    dv = set(data_vars)
    feats: List[str] = []
    seen_clc, seen_popdens = set(), False
    for v in sorted(dv):
        if v in SEG_EXCLUDE:
            continue
        if any(v.startswith(f"CLC_{yr}_") for yr in (2006, 2012, 2018)):
            base = "CLC_" + v.split("_", 2)[2]  # CLC_2018_1 -> CLC_1 ; CLC_2018_scrub_proportion -> CLC_scrub_proportion
            if base not in seen_clc:
                seen_clc.add(base)
                feats.append(base)
            continue
        if v.startswith("popdens_"):
            if not seen_popdens:
                seen_popdens = True
                feats.append("popdens")
            continue
        feats.append(v)
    return feats
