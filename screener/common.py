"""Shared paths, IO helpers and date utilities.

Conventions used across the project:
* Every CSV writes missing values as the literal string UNAVAILABLE and reads it back as NaN.
  Nothing is ever estimated or carried forward.
* Dates are US/Eastern trading dates in ISO format (YYYY-MM-DD).
* Timestamps are UTC ISO 8601.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")
UNAVAILABLE = "UNAVAILABLE"
REPO_ROOT = Path(__file__).resolve().parent.parent
SESSION_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

log = logging.getLogger("screener")

DEFAULT_SETTINGS = {
    "max_expiry_days": 120,
    "top_contracts": 10,
    "baseline_window": 30,
    "min_history": 15,
    "z_threshold": 2.0,
    "ratio_threshold": 2.0,
    "opening_share_threshold": 0.3,
    "price_quiet_z": 1.0,
    "max_candidates": 10,
    "short_dated_days": 7,
    "sector_wide_share": 0.4,
    "market_wide_share": 0.3,
    "event_lookahead_days": 30,
    "oi_base_tolerance": 0.02,
    "oi_min_coverage": 0.5,
    "oi_unchanged_share_stale": 0.8,
    "eval_horizon_days": 10,
    "eval_hit_sigma": 2.0,
    "eval_sigma_window": 60,
    "request_pause_seconds": 0.35,
    "request_retries": 3,
}


@dataclass(frozen=True)
class Paths:
    root: Path = REPO_ROOT

    @property
    def config(self) -> Path:
        return self.root / "config"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def prices(self) -> Path:
        return self.data / "prices"

    @property
    def sessions(self) -> Path:
        return self.data / "sessions"

    @property
    def evaluation(self) -> Path:
        return self.data / "evaluation"

    def session(self, d: str) -> Path:
        return self.sessions / d


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime | None = None) -> str:
    dt = dt or utc_now()
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def et_date(dt: datetime) -> date:
    return dt.astimezone(ET).date()


def parse_date(s: str) -> date:
    return date.fromisoformat(str(s)[:10])


def load_settings(paths: Paths) -> dict:
    settings = dict(DEFAULT_SETTINGS)
    f = paths.config / "settings.json"
    if f.exists():
        settings.update(json.loads(f.read_text()))
    return settings


def load_universe(paths: Paths) -> pd.DataFrame:
    return pd.read_csv(paths.config / "universe.csv", keep_default_na=False, dtype=str)


def load_benchmarks(paths: Paths) -> pd.DataFrame:
    return pd.read_csv(paths.config / "benchmarks.csv", keep_default_na=False, dtype=str)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, na_rep=UNAVAILABLE)


def read_csv(path: Path) -> pd.DataFrame:
    """UNAVAILABLE becomes NaN, except in *_status columns where it is a status value.

    keep_default_na=False so a ticker like "NA" or an empty note is never turned into a missing value.
    """
    cols = pd.read_csv(path, nrows=0).columns
    na = {c: [UNAVAILABLE] for c in cols if not str(c).endswith("_status")}
    return pd.read_csv(path, na_values=na, keep_default_na=False)


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    return read_csv(path) if path.exists() else None


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_for_json(obj), indent=2, ensure_ascii=False, default=str) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def clean_for_json(obj):
    """Replace NaN/inf with the UNAVAILABLE marker so JSON output stays valid and explicit."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return UNAVAILABLE
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return clean_for_json(obj.item())
        except (ValueError, AttributeError):
            pass
    return obj


def is_missing(x) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def fnum(x, digits: int = 2, pct: bool = False, signed: bool = False) -> str:
    """Format a number for reports; missing values become UNAVAILABLE."""
    if is_missing(x):
        return UNAVAILABLE
    v = float(x) * (100 if pct else 1)
    s = f"{v:+,.{digits}f}" if signed else f"{v:,.{digits}f}"
    s = s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")  # German separators
    return s + (" %" if pct else "")


def pct(x, digits: int = 0) -> str:
    return fnum(x, digits, pct=True)


def list_sessions(paths: Paths, with_file: str | None = None) -> list[str]:
    if not paths.sessions.exists():
        return []
    out = []
    for p in sorted(paths.sessions.iterdir()):
        if p.is_dir() and SESSION_RE.match(p.name):
            if with_file is None or (p / with_file).exists():
                out.append(p.name)
    return out


def write_prices(paths: Paths, symbol: str, df: pd.DataFrame) -> None:
    write_csv(df, paths.prices / f"{symbol}.csv")


def read_prices(paths: Paths, symbol: str) -> pd.DataFrame | None:
    f = paths.prices / f"{symbol}.csv"
    if not f.exists():
        return None
    df = read_csv(f)
    df["Date"] = df["Date"].astype(str)
    return df.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)


def add_days(d: str, n: int) -> str:
    return (parse_date(d) + timedelta(days=n)).isoformat()
