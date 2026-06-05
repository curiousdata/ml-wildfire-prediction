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
        self._means = np.array([self.stats[v]["mean"] for v in self.feature_vars], dtype="float32")
        self._stds = np.array(
            [max(self.stats[v]["std"], 1e-6) for v in self.feature_vars],
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

    def __getitem__(self, idx: int):
        """
        Returns:
            X: Tensor of shape [C, H, W]  (features at time t)
            y: Tensor of shape [1, H, W]  (fire mask at time t + lead_time)
        """
        t = self.time_indices[idx]
        t_label = t + self.lead_time

        # Load and normalize features
        X_arrays = []
        for i, v in enumerate(self.feature_vars):
            if v in self.dynamic_vars:
                # Dynamic variable: read slice directly from coarsened Zarr
                arr = self.root[v][t, :, :]
            elif v in self.static_vars:
                # Simple static variable: reuse cached array
                arr = self.static_cache[v]
            elif v in getattr(self, "clc_base_vars", []):
                # CLC base variable: choose appropriate year's map based on calendar year
                year = int(self._years_all[t])
                if year <= 2011:
                    chosen_year = 2006
                elif year <= 2017:
                    chosen_year = 2012
                else:
                    chosen_year = 2018
                year_cache = self.clc_cache[v]
                if chosen_year not in year_cache:
                    # Fallback: pick the nearest available year
                    available_years = sorted(year_cache.keys())
                    chosen_year = min(available_years, key=lambda yy: abs(yy - year))
                arr = year_cache[chosen_year]
            elif v in getattr(self, "popdens_base_vars", []):
                # popdens base variable: choose yearly map closest to sample year
                year = int(self._years_all[t])
                year_cache = self.popdens_cache[v]
                available_years = sorted(year_cache.keys())
                if year in year_cache:
                    chosen_year = year
                else:
                    # Pick the nearest available year (e.g. 2021/2022 -> 2020)
                    chosen_year = min(available_years, key=lambda yy: abs(yy - year))
                arr = year_cache[chosen_year]
            else:
                raise KeyError(
                    f" Feature '{v}' not found among dynamic, static, CLC base, or popdens base variables."
                )

            mean = float(self._means[i])
            std = float(self._stds[i])

            # Ensure numeric dtype
            arr = np.asarray(arr, dtype="float32")

            # Handle NaNs / infs deterministically (critical for training stability)
            if not np.isfinite(arr).all():
                if self.nan_policy == "error":
                    raise ValueError(
                        f" Non-finite values found in feature '{v}' "
                        f"at t={t} (idx={idx})."
                    )
                fill_value = 0.0 if self.nan_policy == "zero" else mean
                arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)

            # Normalize (std already clamped in __init__)
            arr = (arr - mean) / std
            X_arrays.append(arr)

        X = np.stack(X_arrays, axis=0).astype("float32")  # [C, H, W]

        # Load label at t + lead_time directly from coarsened Zarr
        y = np.asarray(self.root[self.label_var][t_label, :, :], dtype="float32")
        if not np.isfinite(y).all():
            # For labels, safest fallback is treating non-finite as 0 (no-fire)
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y_bin = (y > 0.5).astype("float32")[np.newaxis, ...]  # [1, H, W]

        return torch.from_numpy(X), torch.from_numpy(y_bin)

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
        # TODO: this is a placeholder for the new dataset with improved functionality
        # that will inherit from BaseIberFireDataset and add new features: temporal stacking
        # of days before fire (day sequence input), OR-conditioned target masks for several days
        # (if there is a fire in any of the next N days) and more.
        pass