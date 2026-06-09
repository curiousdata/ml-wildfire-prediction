"""Canonical 1 km EPSG:3035 grid for the Fire Guard Datacube (FGDC) — the shared spatial contract every
ingester writes onto.

The grid is **aligned to IberFire v1** so that block-mean coarsening by 4 reproduces v1's
`IberFire_coarse4.zarr` grid bit-for-bit (enabling clean per-cell A/B between the FGDC model and v1).
Verified: v1 coarse4 is 230(y)×297(x) at 4 km; refined ×4 → 920×1188 at 1 km, and the 1 km block-means
recover v1's exact x/y cell centres (see `--verify`).

Coordinates are CELL CENTRES in metres (ETRS89-LAEA, EPSG:3035). The rasterio `transform()` returns the
pixel-corner affine (origin at the top-left corner = first cell centre shifted by half a pixel).

Masks (`is_spain`, `is_sea`, `is_waterbody`, `is_natura2000`): for P0 these are derived by refining v1's
4 km masks ×4 onto the 1 km grid (exact, since the grids align) — a provisional but faithful start. P3
re-derives them from CORINE + the DEM at native 1 km.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

EPSG = 3035
# 1 km cell-CENTRE origins + spacing + dims (derived from v1 coarse4, verified by --verify)
X0, DX, NX = 2674734.3466, 1000.0, 1188      # easting:  left → right (+)
Y0, DY, NY = 2492195.9911, -1000.0, 920      # northing: top → bottom (−, array/raster convention)

ROOT = Path(__file__).resolve().parents[3]
V1_CUBE = ROOT / "data" / "gold" / "IberFire_coarse4.zarr"
COARSEN_FACTOR = 4                            # 1 km → 4 km gold, matching v1


def x_coords() -> np.ndarray:
    return X0 + DX * np.arange(NX)


def y_coords() -> np.ndarray:
    return Y0 + DY * np.arange(NY)


def shape() -> tuple[int, int]:
    """(rows, cols) = (NY, NX) — array/y-x order."""
    return NY, NX


def transform():
    """rasterio Affine for the 1 km grid (origin at the top-left pixel CORNER)."""
    from rasterio.transform import Affine
    return Affine(DX, 0.0, X0 - DX / 2.0, 0.0, DY, Y0 - DY / 2.0)


def bounds() -> tuple[float, float, float, float]:
    """(west, south, east, north) pixel-edge bounds in EPSG:3035 metres."""
    w = X0 - DX / 2.0
    e = X0 + DX * (NX - 1) + DX / 2.0
    n = Y0 - DY / 2.0                 # DY<0 → top edge is above the first centre
    s = Y0 + DY * (NY - 1) + DY / 2.0
    return w, s, e, n


def meshgrid():
    """(LON-less) projected X, Y meshgrids [NY, NX] of cell centres, for regridding."""
    return np.meshgrid(x_coords(), y_coords())


def refine_to_1km(coarse_2d: np.ndarray, factor: int = COARSEN_FACTOR) -> np.ndarray:
    """Nearest-neighbour refine a v1 4 km field [230,297] to the 1 km grid [920,1188] (each coarse cell
    → factor×factor block). Used to seed provisional masks from v1."""
    return np.repeat(np.repeat(np.asarray(coarse_2d), factor, axis=0), factor, axis=1)


def load_masks_from_v1(cube=V1_CUBE):
    """Provisional 1 km masks refined from v1's 4 km masks (exact on the aligned grid). Returns a dict
    {name: bool/float grid[NY,NX]}. P3 will replace these with CORINE/DEM-native 1 km masks."""
    import xarray as xr
    z = xr.open_zarr(str(cube), consolidated=True)
    out = {}
    for m in ("is_spain", "is_sea", "is_waterbody", "is_natura2000"):
        if m in z:
            out[m] = refine_to_1km(np.nan_to_num(z[m].values)).astype(np.float32)
    return out


def _verify():
    """Confirm the 1 km grid block-means back to v1's exact coarse4 x/y centres."""
    import xarray as xr
    z = xr.open_zarr(str(V1_CUBE), consolidated=True)
    xv, yv = z["x"].values.astype(float), z["y"].values.astype(float)
    x1, y1 = x_coords(), y_coords()
    xc = x1.reshape(-1, COARSEN_FACTOR).mean(1)
    yc = y1.reshape(-1, COARSEN_FACTOR).mean(1)
    ok_dims = (len(x1) == COARSEN_FACTOR * len(xv)) and (len(y1) == COARSEN_FACTOR * len(yv))
    ok_x, ok_y = np.allclose(xc, xv), np.allclose(yc, yv)
    print(f"FGDC 1 km grid: {NY} x {NX}  (y,x) | bounds(WSEN)={tuple(round(b,1) for b in bounds())}")
    print(f"  dims ×4 == v1 coarse4: {ok_dims} | x block-mean == v1: {ok_x} | y block-mean == v1: {ok_y}")
    assert ok_dims and ok_x and ok_y, "FGDC grid does NOT align to v1 coarse4 — fix origins/dims."
    print("  ✓ aligned — coarsening FGDC ×4 reproduces v1's grid exactly.")


if __name__ == "__main__":
    _verify()
