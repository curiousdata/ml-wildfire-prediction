"""Push the latest serving prediction to the HF Dataset — Ship A (see the `hf-deploy-plan` memory).

The engine (`serve.py --mode {live,replay}`) writes date-partitioned files under `data/serving_store/`.
This step uploads only the SMALL, latest artifacts to the HF Dataset `curiousdata/fireguard-serving`:

  grids/<issue>.npz                       — the prob/regime/today_fire grid the Space renders (~30 KB)
  inference/issue_date=<issue>.parquet    — the per-region alert summary (~8 KB)
  latest.json                             — a tiny manifest pointing at the current grid (cheap to poll)

The Space reads the Dataset: it polls `latest.json` (tiny), and only when `pushed_at` changes does it pull
the referenced grid. So the map stays fresh without re-downloading on every fragment tick.

Upload order is grid → inference → **manifest last**, so a reader never sees a manifest pointing at a
not-yet-uploaded grid. Idempotent: re-pushing the same date overwrites in place.

Auth: needs only HF write access — `HF_TOKEN` in `.env` (or `hf auth login`). No FIRMS/Open-Meteo key here.

CLI:
  python scripts/push_predictions.py [--date YYYY-MM-DD] [--repo curiousdata/fireguard-serving] [--dry-run]
Run it right after `serve` in the scheduled job so each new prediction lands on the Space.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv; load_dotenv()          # HF_TOKEN lives in .env (gitignored)
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.data import metrics as M

STORE = M.project_root / "data" / "serving_store"
DEFAULT_REPO = "curiousdata/fireguard-serving"
log = logging.getLogger("push_predictions")


def _pick_grid(date_arg: str | None) -> Path:
    """Resolve the grid npz to push: the named --date, else the latest by date."""
    if date_arg:
        p = STORE / "grids" / f"{date_arg}.npz"
        if not p.exists():
            raise SystemExit(f"no grid for {date_arg} at {p}")
        return p
    grids = sorted((STORE / "grids").glob("*.npz"))
    if not grids:
        raise SystemExit(f"no grids under {STORE/'grids'} — run serve first")
    return grids[-1]


def _manifest(grid: Path) -> tuple[dict, Path | None]:
    """Read the grid's own metadata into a small manifest; locate its inference parquet (may be absent)."""
    d = np.load(grid, allow_pickle=True)
    issue = str(d["issue_date"]); target = str(d["target_date"])
    source = str(d["source"]) if "source" in d else "?"
    fetched = str(d["fetched_at"]) if "fetched_at" in d else None
    inf = STORE / "inference" / f"issue_date={issue}.parquet"
    man = dict(issue=issue, target=target, source=source, prelim=(source == "live-prelim"),
               fetched_at=fetched, pushed_at=datetime.now(timezone.utc).isoformat(),
               grid_path=f"grids/{issue}.npz",
               inference_path=(f"inference/issue_date={issue}.parquet" if inf.exists() else None))
    return man, (inf if inf.exists() else None)


def push(date_arg=None, repo=DEFAULT_REPO, dry_run=False):
    grid = _pick_grid(date_arg)
    man, inf = _manifest(grid)
    # write the manifest next to the store so the push is reproducible / inspectable
    man_path = STORE / "latest.json"
    man_path.write_text(json.dumps(man, indent=2))
    uploads = [(str(grid), man["grid_path"])]
    if inf is not None:
        uploads.append((str(inf), man["inference_path"]))
    uploads.append((str(man_path), "latest.json"))          # manifest LAST (pointer only valid once grid is up)

    log.info(f"grid {grid.name}: issue {man['issue']} → target {man['target']} "
             f"[{man['source']}{' PRELIM' if man['prelim'] else ''}]  →  dataset {repo}")
    for src, dst in uploads:
        log.info(f"  {'(dry) ' if dry_run else ''}upload {Path(src).name}  →  {repo}:{dst}")
    if dry_run:
        log.info("dry-run: nothing uploaded."); return man

    import os
    from huggingface_hub import HfApi
    api = HfApi(token=os.getenv("HF_TOKEN"))
    for src, dst in uploads:                                 # grid, [inference], manifest — in that order
        api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo, repo_type="dataset",
                        commit_message=f"serve {man['issue']}→{man['target']} [{man['source']}]")
    log.info(f"pushed {man['issue']} → {repo}  (Space will pick it up on its next poll)")
    return man


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ap = argparse.ArgumentParser(description="Push the latest serving prediction to the HF Dataset (Ship A).")
    ap.add_argument("--date", help="issue date YYYY-MM-DD (default: latest grid in the store)")
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"HF dataset repo (default {DEFAULT_REPO})")
    ap.add_argument("--dry-run", action="store_true", help="print the upload plan, upload nothing")
    args = ap.parse_args()
    push(args.date, args.repo, args.dry_run)


if __name__ == "__main__":
    main()
