import zarr
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from typing import List, Dict, Literal, Optional
import logging

import xarray as xr

logger = logging.getLogger(__name__)

class BaseIberFireDataset(Dataset):
    """
    PyTorch Dataset for IberFire-style wildfire *segmentation* (heatmap output) focused on:
      - Full-image inputs (no tile-level classification)
      - Pixel-wise fire mask as target (for U-Net / FCN-style models)
      - Optional lead time (e.g., predict fire at t+1 from features at t)
      - Optional on-the-fly normalization

    Args:
        zarr_path: Path to IberFire.zarr directory
        time_start: Start date (e.g., "2018-01-01")
        time_end: End date (e.g., "2020-12-31")
        feature_vars: List of feature variable names
        label_var: Label variable name (e.g., "is_near_fire")
        lead_time: Predict label at t+lead_time (0 = same day, 1 = tomorrow, etc.)
        stats: Optional dict with precomputed normalization stats
               {var_name: {"mean": float, "std": float}}
        compute_stats: Whether to compute stats if not provided
        overwrite_stats: If True, recompute stats even when stats_path already
               exists. If False (default), an existing stats file is reused.
        stats_path: Optional path to load/save normalization stats JSON.

    Usage (typical for U-Net MVP):
        >>> dataset = BaseIberFireDataset(
        ...     zarr_path="data/processed/IberFire.zarr",
        ...     time_start="2018-01-01",
        ...     time_end="2020-12-31",
        ...     feature_vars=["wind_speed_mean", "t2m_mean", "RH_mean"],
        ...     label_var="is_fire",
        ...     lead_time=1,  # predict tomorrow's fire heatmap
        ...     compute_stats=True,
        ... )
    """

    def __init__(
        self,
        zarr_path: str,
        time_start: str,
        time_end: str,
        feature_vars: List[str],
        label_var: str,
        lead_time: int = 1,
        stats: Optional[Dict[str, Dict[str, float]]] = None,
        compute_stats: bool = False,
        overwrite_stats: bool = False,
        stats_path: Optional[str] = None,
        mode: str = "all",
        day_indices_path: Optional[str] = None,
        balanced_ratio: float = 1.0,
        seed: int = 42,
        nan_policy: Literal["mean", "zero", "error"] = "mean",
    ):
        self.zarr_path = Path(zarr_path)
        self.feature_vars = feature_vars
        self.label_var = label_var
        self.lead_time = lead_time
        self.mode = mode
        self.balanced_ratio = balanced_ratio
        self.seed = seed
        self.nan_policy = nan_policy
        self.stats_path: Optional[Path] = Path(stats_path) if stats_path is not None else None
        self.day_indices_path: Optional[Path] = (
            Path(day_indices_path) if day_indices_path is not None else None
        )

        logger.info(f" Opening Zarr dataset: {self.zarr_path}")
        self.ds = xr.open_zarr(
            self.zarr_path,
            consolidated=True,
            decode_times=True,
            chunks="auto",
        )
        # Also open raw Zarr root for fast array access in __getitem__
        self.root = zarr.open(str(self.zarr_path), mode="r")
        # Precompute calendar year for each global time index
        time_da = self.ds["time"]
        try:
            self._years_all = time_da.dt.year.values.astype(int)
        except Exception:
            time_vals = time_da.values
            self._years_all = np.array([int(str(v)[:4]) for v in time_vals], dtype=int)

        logger.info(f" Filtering time range: {time_start} to {time_end}")
        time = self.ds["time"].values  # this is datetime64 already
        mask = (time >= np.datetime64(time_start)) & (time <= np.datetime64(time_end))
        all_indices = np.where(mask)[0]

        # Ensure we can look ahead by lead_time
        if self.lead_time > 0:
            max_valid = len(time) - self.lead_time - 1
            all_indices = all_indices[all_indices <= max_valid]

        if len(all_indices) == 0:
            raise ValueError("No valid time indices found for the given range and lead_time.")

        self.time_indices = all_indices
        logger.info(f" Total usable time steps: {len(self.time_indices)}")
        # Optionally adjust the list of time indices based on fire/no-fire day information
        self._apply_day_mode()
        logger.info(f" Time steps after mode='{self.mode}': {len(self.time_indices)}")

        # Determine which variables are dynamic (have time dim),
        # which are simple static (no time dim),
        # and which are year-aware static bases (CLC_* or popdens).
        self.dynamic_vars: List[str] = []
        self.static_vars: List[str] = []
        self.clc_base_vars: List[str] = []
        self.clc_mapping: Dict[str, Dict[int, str]] = {}
        self.popdens_base_vars: List[str] = []
        self.popdens_mapping: Dict[str, Dict[int, str]] = {}

        data_var_names = set(self.ds.data_vars)

        for v in self.feature_vars:
            # Case 1: feature name directly matches a data variable
            if v in data_var_names:
                da = self.ds[v]
                if "time" in da.dims:
                    self.dynamic_vars.append(v)
                else:
                    self.static_vars.append(v)
                continue

            # Case 2: CLC base feature, e.g. "CLC_1" or "CLC_forest_proportion"
            if v.startswith("CLC_"):
                base_suffix = v[len("CLC_") :]
                year_map: Dict[int, str] = {}
                for year in (2006, 2012, 2018):
                    candidate = f"CLC_{year}_{base_suffix}"
                    if candidate in data_var_names:
                        year_map[year] = candidate
                if not year_map:
                    raise KeyError(
                        f" CLC base feature '{v}' could not be resolved "
                        f"to any of CLC_2006_{base_suffix}, CLC_2012_{base_suffix}, CLC_2018_{base_suffix}."
                    )
                self.clc_base_vars.append(v)
                self.clc_mapping[v] = year_map
                continue

            # Case 3: popdens base feature, e.g. "popdens" → popdens_2008..popdens_2020
            if v == "popdens":
                year_map: Dict[int, str] = {}
                for name in data_var_names:
                    if name.startswith("popdens_"):
                        parts = name.split("_")
                        if len(parts) != 2:
                            continue
                        try:
                            year = int(parts[1])
                        except ValueError:
                            continue
                        year_map[year] = name
                if not year_map:
                    raise KeyError(
                        " popdens base feature requested, "
                        "but no popdens_YYYY variables found in dataset."
                    )
                self.popdens_base_vars.append(v)
                self.popdens_mapping[v] = year_map
                continue

            # Case 4: unknown feature name
            raise KeyError(
                f" Feature '{v}' not found in dataset variables "
                f"and not recognized as CLC or popdens base."
            )

        logger.info(f" Dynamic vars (time-dependent): {self.dynamic_vars}")
        logger.info(f" Static vars (no time dimension, broadcast in time): {self.static_vars}")
        if self.clc_base_vars:
            logger.info(f" CLC base vars (year-aware static): {self.clc_base_vars}")
        if self.popdens_base_vars:
            logger.info(f" popdens base vars (year-aware static): {self.popdens_base_vars}")

        # Cache static variables in memory to avoid repeated disk reads
        self.static_cache: Dict[str, np.ndarray] = {}
        for v in self.static_vars:
            arr = self.root[v][:, :].astype("float32")
            self.static_cache[v] = arr
            logger.info(
                f" Cached static var '{v}' with shape {arr.shape} "
                f"and dtype {arr.dtype}"
            )

        # Cache CLC year-specific maps for each CLC base feature
        self.clc_cache: Dict[str, Dict[int, np.ndarray]] = {}
        for base_name, year_map in self.clc_mapping.items():
            year_cache: Dict[int, np.ndarray] = {}
            for year, varname in year_map.items():
                arr = self.root[varname][:, :].astype("float32")
                year_cache[year] = arr
                logger.info(
                    f" Cached CLC var '{varname}' for base '{base_name}' "
                    f"with shape {arr.shape} and dtype {arr.dtype}"
                )
            self.clc_cache[base_name] = year_cache

        # Cache popdens year-specific maps for each popdens base feature
        self.popdens_cache: Dict[str, Dict[int, np.ndarray]] = {}
        for base_name, year_map in self.popdens_mapping.items():
            year_cache: Dict[int, np.ndarray] = {}
            for year, varname in year_map.items():
                arr = self.root[varname][:, :].astype("float32")
                year_cache[year] = arr
                logger.info(
                    f" Cached popdens var '{varname}' for base '{base_name}' "
                    f"with shape {arr.shape} and dtype {arr.dtype}"
                )
            self.popdens_cache[base_name] = year_cache

        # Load or compute normalization stats
        if stats is not None:
            logger.info(" Using provided normalization stats.")
            self.stats = stats

        elif compute_stats:
            # Non-interactive: reuse an existing stats file unless explicitly told
            # to overwrite it (overwrite_stats=True). No prompts, so this is safe
            # under DataLoader workers and headless/CI runs.
            if (
                self.stats_path is not None
                and self.stats_path.exists()
                and not overwrite_stats
            ):
                logger.info(
                    f" Reusing existing stats from {self.stats_path} "
                    "(pass overwrite_stats=True to recompute)."
                )
                with open(self.stats_path) as f:
                    self.stats = json.load(f)
            else:
                logger.info(" Computing normalization stats from data...")
                self.stats = self._compute_stats()

                # If no explicit stats_path was provided, choose a sensible default:
                # <zarr_parent>/stats/simple_iberfire_stats.json
                if self.stats_path is None:
                    default_dir = self.zarr_path.parent / "stats"
                    self.stats_path = default_dir / "simple_iberfire_stats.json"

                self.save_stats(self.stats_path)

        elif self.stats_path is not None and self.stats_path.exists():
            logger.info(f" Loading normalization stats from: {self.stats_path}")
            with open(self.stats_path) as f:
                self.stats = json.load(f)

        else:
            logger.info(" No stats provided, using mean=0, std=1 for all vars.")
            self.stats = {v: {"mean": 0.0, "std": 1.0} for v in self.feature_vars}

        # Cache aligned stats arrays for faster __getitem__
        # Default missing-stat features to mean=0/std=1 (no-op normalization). This keeps
        # already-scaled features like a 0/1 `is_fire` feature channel passing through, and
        # is identical to before when every feature is present in stats.
        self._means = np.array(
            [self.stats.get(v, {}).get("mean", 0.0) for v in self.feature_vars], dtype="float32"
        )
        self._stds = np.array(
            [max(self.stats.get(v, {}).get("std", 1.0), 1e-6) for v in self.feature_vars],
            dtype="float32",
        )

    def _compute_stats(self) -> Dict[str, Dict[str, float]]:
        """Compute simple per-variable mean/std from a subset of time steps."""
        stats: Dict[str, Dict[str, float]] = {}
        # Sample up to 100 time steps for efficiency
        sample_indices = np.random.choice(
            self.time_indices,
            size=min(100, len(self.time_indices)),
            replace=False,
        )

        for v in self.feature_vars:
            data_list = []
            if v in self.dynamic_vars:
                # Time-varying variable: sample across time
                for idx in sample_indices:
                    arr = self.root[v][idx, :, :]
                    data_list.append(arr.ravel())
            elif v in self.static_vars:
                # Simple static variable: no time dimension, reuse cached array
                arr = self.static_cache[v]
                data_list.append(arr.ravel())
            elif v in getattr(self, "clc_base_vars", []):
                # CLC base variable: combine all available yearly maps
                for year, arr in self.clc_cache[v].items():
                    data_list.append(arr.ravel())
            elif v in getattr(self, "popdens_base_vars", []):
                # popdens base variable: combine all yearly population maps
                for year, arr in self.popdens_cache[v].items():
                    data_list.append(arr.ravel())
            else:
                raise KeyError(
                    f" Variable '{v}' not classified as dynamic, static, CLC base, or popdens base."
                )

            data = np.concatenate(data_list)

            mean = float(np.nanmean(data))
            std = float(np.nanstd(data))
            if std < 1e-6:
                std = 1.0

            stats[v] = {"mean": mean, "std": std}
            logger.info(f" {v}: mean={mean:.4f}, std={std:.4f}")

        return stats

    def _apply_day_mode(self) -> None:
        """
        Modify self.time_indices according to self.mode and an optional
        precomputed fire/no-fire day index file (indices are interpreted in
        label-day space, i.e. days where the label at time t has fire/no fire).

        Modes:
            - "all": keep all time indices (no change)
            - "fire_only": keep only days that have at least one fire pixel
            - "balanced_days": keep all fire days plus a random sample of no-fire days
        """
        # If no mode requested or no day-index file provided, do nothing.
        if self.mode == "all" or self.day_indices_path is None:
            return

        if not self.day_indices_path.exists():
            logger.info(
                f" Day index file not found at {self.day_indices_path}, "
                f"mode='{self.mode}' will be ignored (using all time steps)."
            )
            return

        import json

        with self.day_indices_path.open("r") as f:
            data = json.load(f)

        fire_days_global = np.array(data.get("fire_days", []), dtype=int)
        no_fire_days_global = np.array(data.get("no_fire_days", []), dtype=int)

        if fire_days_global.size == 0 and self.mode in ("fire_only", "balanced_days"):
            logger.info(
                " No fire_days found in the index file; "
                "mode will be ignored and all time steps will be used."
            )
            return

        base_indices = np.array(self.time_indices, dtype=int)

        # Map feature-day indices (t) to their corresponding label-day indices (t + lead_time)
        # The JSON file is assumed to store fire/no-fire indices in label-day space.
        lead = int(getattr(self, "lead_time", 0))
        label_indices_for_features = base_indices + lead

        # A feature day is considered "fire" if its label day (t + lead_time) is in fire_days_global.
        fire_mask = np.isin(label_indices_for_features, fire_days_global)
        no_fire_mask = np.isin(label_indices_for_features, no_fire_days_global)

        fire_in_range = base_indices[fire_mask]
        no_fire_in_range = base_indices[no_fire_mask]

        if self.mode == "fire_only":
            if fire_in_range.size == 0:
                raise ValueError(
                    "Mode 'fire_only' selected, but no fire days found in the given time range."
                )
            self.time_indices = fire_in_range.tolist()
            return

        if self.mode == "balanced_days":
            if fire_in_range.size == 0:
                raise ValueError(
                    "Mode 'balanced_days' selected, but no fire days found in the given time range."
                )

            n_fire = fire_in_range.size
            n_no_fire_target = int(self.balanced_ratio * n_fire)

            if no_fire_in_range.size == 0:
                # Nothing to balance with; fall back to fire-only
                chosen_no_fire = np.array([], dtype=int)
            else:
                rng = np.random.default_rng(self.seed)
                n_no_fire_target = min(n_no_fire_target, no_fire_in_range.size)
                chosen_no_fire = rng.choice(
                    no_fire_in_range, size=n_no_fire_target, replace=False
                )

            combined = np.concatenate([fire_in_range, chosen_no_fire])

            # Shuffle combined indices to mix fire and no-fire days
            rng = np.random.default_rng(self.seed)
            combined = rng.permutation(combined)

            self.time_indices = combined.tolist()
            return

        raise ValueError(
            f"Unknown mode '{self.mode}'. Expected 'all', 'fire_only', or 'balanced_days'."
        )

    def __len__(self) -> int:
        # One sample per time step
        return len(self.time_indices)

    def _raw_feature(self, v: str, t: int) -> np.ndarray:
        """Raw (H, W) array for feature `v` at time index `t` (pre-normalization).

        Handles the four feature kinds: dynamic (time,y,x), simple static (y,x), and the
        year-aware CLC / popdens bases (picks the calendar-year-appropriate map).
        Subclasses can override to add new feature kinds (e.g. time-only calendar vars).
        """
        if v in self.dynamic_vars:
            return self.root[v][t, :, :]
        if v in self.static_vars:
            return self.static_cache[v]
        if v in getattr(self, "clc_base_vars", []):
            year = int(self._years_all[t])
            chosen = 2006 if year <= 2011 else 2012 if year <= 2017 else 2018
            cache = self.clc_cache[v]
            if chosen not in cache:
                chosen = min(sorted(cache.keys()), key=lambda yy: abs(yy - year))
            return cache[chosen]
        if v in getattr(self, "popdens_base_vars", []):
            year = int(self._years_all[t])
            cache = self.popdens_cache[v]
            chosen = year if year in cache else min(sorted(cache.keys()), key=lambda yy: abs(yy - year))
            return cache[chosen]
        raise KeyError(
            f" Feature '{v}' not found among dynamic, static, CLC base, or popdens base variables."
        )

    def _build_X(self, t: int) -> np.ndarray:
        """Normalized feature stack [C, H, W] at time index `t`."""
        X_arrays = []
        for i, v in enumerate(self.feature_vars):
            arr = np.asarray(self._raw_feature(v, t), dtype="float32")
            mean = float(self._means[i])
            std = float(self._stds[i])
            if not np.isfinite(arr).all():
                if self.nan_policy == "error":
                    raise ValueError(f" Non-finite values found in feature '{v}' at t={t}.")
                fill_value = 0.0 if self.nan_policy == "zero" else mean
                arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
            X_arrays.append((arr - mean) / std)
        return np.stack(X_arrays, axis=0).astype("float32")

    def _build_y(self, t_label: int) -> np.ndarray:
        """Binary fire label [1, H, W] at time index `t_label`."""
        y = np.asarray(self.root[self.label_var][t_label, :, :], dtype="float32")
        if not np.isfinite(y).all():
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return (y > 0.5).astype("float32")[np.newaxis, ...]

    def __getitem__(self, idx: int):
        """
        Returns:
            X: Tensor of shape [C, H, W]  (features at time t)
            y: Tensor of shape [1, H, W]  (fire mask at time t + lead_time)
        """
        t = self.time_indices[idx]
        return torch.from_numpy(self._build_X(t)), torch.from_numpy(self._build_y(t + self.lead_time))

    def save_stats(self, path: str):
        """Save normalization stats to JSON file."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w") as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f" Saved normalization stats to: {path_obj}")

    def get_time_value(self, idx: int) -> str:
        """Return the datetime string for a given sample index (for debugging)."""
        t = self.time_indices[idx]
        time_value = self.ds["time"].values[t]
        return str(time_value)
    
class StackedIberFireDataset(Dataset):
    def __init__(self):
        # TODO: placeholder for a future temporal-stacking dataset (day-sequence input,
        # OR-conditioned multi-day targets). Not implemented; see RegimeIberFireDataset
        # for the current segmentation dataset.
        pass


class RegimeIberFireDataset(BaseIberFireDataset):
    """Segmentation dataset for next-day fire with regime labels + calendar broadcast.

    Extends BaseIberFireDataset — reuses its year-aware CLC/popdens resolution, static
    caching, stats-based normalization, NaN policy, and time/lead filtering. Adds:

      - **calendar (time,)-only features** (e.g. doy_sin/doy_cos/dow_sin/dow_cos) broadcast
        to the full grid at read time (base would mis-read them as (time,y,x));
      - a per-pixel **regime_code** ∈ {0 = sea/invalid, 1 = ignition (no fire within
        ~`regime_dist_cells` of t), 2 = spread (fire nearby at t)} from `dist_to_fire(t)`,
        consumed by the regime-aware loss;
      - `make_weighted_sampler()` — oversample days whose label day has fire.

    ``__getitem__`` returns ``(X[C,H,W], y[1,H,W], regime_code[1,H,W])``. Full-image
    batching is just ``DataLoader(batch_size=N)``.
    """

    def __init__(self, *args, regime_dist_cells: float = 1.5, **kwargs):
        super().__init__(*args, **kwargs)
        # base classified (time,)-only vars as dynamic; reclassify them as calendar.
        self.time_only_vars = [v for v in self.dynamic_vars if set(self.ds[v].dims) == {"time"}]
        self.dynamic_vars = [v for v in self.dynamic_vars if v not in self.time_only_vars]
        self.H = int(self.ds.sizes["y"])
        self.W = int(self.ds.sizes["x"])
        xc = self.ds["x"].values
        self.cell_km = abs(float(xc[1] - xc[0])) / 1000.0
        self.regime_dist_km = regime_dist_cells * self.cell_km
        if "dist_to_fire" not in self.ds.data_vars:
            raise KeyError("RegimeIberFireDataset requires 'dist_to_fire' in the cube.")
        # land mask: finite t2m_mean on the first usable day (matches the analysis land mask).
        t0 = int(self.time_indices[0])
        self.land_mask = np.isfinite(np.asarray(self.root["t2m_mean"][t0, :, :], dtype="float32"))
        logger.info(
            f" RegimeIberFireDataset: {len(self.time_only_vars)} calendar vars broadcast; "
            f"cell={self.cell_km:.0f}km, regime threshold={self.regime_dist_km:.0f}km, "
            f"land cells={int(self.land_mask.sum())}"
        )

    def _raw_feature(self, v: str, t: int) -> np.ndarray:
        if v in self.time_only_vars:  # calendar scalar -> broadcast across the grid
            return np.full((self.H, self.W), float(self.root[v][t]), dtype="float32")
        return super()._raw_feature(v, t)

    def __getitem__(self, idx: int):
        t = int(self.time_indices[idx])
        X = self._build_X(t)
        y = self._build_y(t + self.lead_time)
        dist = np.asarray(self.root["dist_to_fire"][t, :, :], dtype="float32")
        near = np.isfinite(dist) & (dist <= self.regime_dist_km)
        regime = np.where(self.land_mask, np.where(near, 2, 1), 0).astype("int64")[np.newaxis, ...]
        return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(regime)

    def make_weighted_sampler(self, fire_oversample: float = 10.0):
        """WeightedRandomSampler oversampling days whose LABEL day (t+lead) contains fire."""
        from torch.utils.data import WeightedRandomSampler

        lab = self.root[self.label_var]
        fire_day: Dict[int, bool] = {}
        weights = []
        for t in self.time_indices:
            tl = int(t) + self.lead_time
            if tl not in fire_day:
                fire_day[tl] = bool(np.asarray(lab[tl, :, :]).sum() > 0)
            weights.append(fire_oversample if fire_day[tl] else 1.0)
        return WeightedRandomSampler(weights, num_samples=len(self.time_indices), replacement=True)