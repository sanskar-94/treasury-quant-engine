"""Market-data acquisition.

Two public sources, neither of which requires an API key:

``Treasury.gov``
    The Daily Treasury Par Yield Curve Rates (the "CMT" curve).  One CSV per
    calendar year, tenors 1Mo..30Y.  Coverage is ragged: the 30Y was
    discontinued 2002-2006, the 20Y is missing 1987-1993, and 1/2/4-month bills
    only start in 2001/2018/2022.  The loader keeps the ragged shape rather than
    forward-filling, and downstream code decides how to handle gaps.

``FRED``
    Macro and market series via the public ``fredgraph.csv`` endpoint.

Everything is cached to Parquet on disk keyed by source and year, so a full
1990-present pull happens once and subsequent runs are instant and offline.
"""

from __future__ import annotations

import datetime as dt
import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..config import Config
from ..logging_utils import get_logger

log = get_logger("data.sources")

TREASURY_YIELD_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
    "?type={series}&field_tdr_date_value={year}&page&_format=csv"
)
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# FRED soft-throttles bursts by *hanging* the connection rather than returning
# 429, which surfaces as a read timeout.  Pacing requests avoids tripping it;
# the value is deliberately conservative because the whole bundle is cached.
FRED_MIN_INTERVAL_SEC = 4.0
_last_fred_call = 0.0

TREASURY_SERIES = {
    "yield_curve": "daily_treasury_yield_curve",     # par yields (CMT)
    "bill_rates": "daily_treasury_bill_rates",       # discount + coupon-equiv
    "real_yield": "daily_treasury_real_yield_curve",  # TIPS real curve
    "long_term": "daily_treasury_long_term_rate",
}

TENOR_YEARS: dict[str, float] = {
    "1 Mo": 1 / 12,
    # Treasury began publishing a 6-week ("1.5 Month") CMT point on 2025-02-18.
    "1.5 Month": 1.5 / 12,
    "2 Mo": 2 / 12,
    "3 Mo": 0.25,
    "4 Mo": 4 / 12,
    "6 Mo": 0.5,
    "1 Yr": 1.0,
    "2 Yr": 2.0,
    "3 Yr": 3.0,
    "5 Yr": 5.0,
    "7 Yr": 7.0,
    "10 Yr": 10.0,
    "20 Yr": 20.0,
    "30 Yr": 30.0,
}

USER_AGENT = "tqe/1.0 (Treasury Quant Engine; research use)"

# A single pooled session across the whole process.  Both providers sit behind a
# CDN that is markedly happier with keep-alive than with a fresh TLS handshake
# per request - pulling 37 years of curve data went from minutes to seconds.
_SESSION: requests.Session | None = None


def get_session() -> requests.Session:
    """Process-wide HTTP session with connection pooling."""
    global _SESSION
    if _SESSION is None:
        sess = requests.Session()
        sess.headers.update({"User-Agent": USER_AGENT, "Accept": "text/csv,*/*"})
        adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        _SESSION = sess
    return _SESSION


class DataFetchError(RuntimeError):
    """Raised when a source cannot be reached after all retries."""


@dataclass
class FetchStats:
    """What a pull actually did - surfaced by ``tqe data pull``."""

    downloaded: int = 0
    from_cache: int = 0
    failed: int = 0
    rows: int = 0

    def __str__(self) -> str:
        return (
            f"downloaded={self.downloaded} cached={self.from_cache} "
            f"failed={self.failed} rows={self.rows}"
        )


def _http_get(url: str, timeout: int, max_retries: int) -> str:
    """GET with exponential backoff.  Raises :class:`DataFetchError` on failure."""
    delay = 1.0
    last: Exception | None = None
    session = get_session()
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            if not resp.text.strip():
                raise DataFetchError(f"Empty body from {url}")
            return resp.text
        except Exception as exc:  # noqa: BLE001 - retried and re-raised below
            last = exc
            if attempt < max_retries:
                log.warning("fetch attempt %d/%d failed (%s); retrying in %.1fs", attempt, max_retries, exc, delay)
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
    raise DataFetchError(f"GET {url} failed after {max_retries} attempts: {last}") from last


# --------------------------------------------------------------------------- #
# Treasury par yield curve
# --------------------------------------------------------------------------- #
def fetch_treasury_year(
    year: int,
    series: str = "yield_curve",
    timeout: int = 45,
    max_retries: int = 5,
) -> pd.DataFrame:
    """Download one calendar year of a Treasury daily rate series.

    Returns a frame indexed by ``date`` (ascending) whose columns are the tenor
    labels present for that year, with values as **decimal** rates (4.25% ->
    0.0425).
    """
    if series not in TREASURY_SERIES:
        raise ValueError(f"Unknown treasury series {series!r}; choose from {sorted(TREASURY_SERIES)}")
    url = TREASURY_YIELD_URL.format(year=year, series=TREASURY_SERIES[series])
    text = _http_get(url, timeout, max_retries)
    df = pd.read_csv(io.StringIO(text))
    if df.empty or "Date" not in df.columns:
        raise DataFetchError(f"Unexpected Treasury payload for {year}: columns={list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    df.index.name = "date"

    # Percent -> decimal, coercing the odd blank/'N/A' cell to NaN.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0
    df = df.dropna(axis=1, how="all")
    return df


def fetch_treasury_curve(
    start_year: int,
    end_year: int,
    cache_dir: Path,
    series: str = "yield_curve",
    timeout: int = 45,
    max_retries: int = 5,
    force: bool = False,
    stats: FetchStats | None = None,
) -> pd.DataFrame:
    """Full multi-year par-yield history, cached one Parquet file per year.

    The current year is always re-downloaded (it grows daily); completed years
    are written once and read from disk forever after.
    """
    stats = stats or FetchStats()
    cache_dir.mkdir(parents=True, exist_ok=True)
    this_year = dt.date.today().year
    frames: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        path = cache_dir / f"treasury_{series}_{year}.parquet"
        stale = force or year >= this_year
        if path.exists() and not stale:
            frames.append(pd.read_parquet(path))
            stats.from_cache += 1
            continue
        try:
            df = fetch_treasury_year(year, series, timeout, max_retries)
        except DataFetchError as exc:
            if path.exists():
                log.warning("download failed for %d (%s); falling back to cache", year, exc)
                frames.append(pd.read_parquet(path))
                stats.from_cache += 1
            else:
                log.error("no data for %d and no cache: %s", year, exc)
                stats.failed += 1
            continue
        if df.empty:
            stats.failed += 1
            continue
        df.to_parquet(path)
        frames.append(df)
        stats.downloaded += 1

    if not frames:
        raise DataFetchError(f"No Treasury data retrieved for {start_year}-{end_year}")

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    # Preserve a stable tenor ordering (short -> long) rather than year-of-first-appearance.
    ordered = [c for c in TENOR_YEARS if c in out.columns]
    extra = [c for c in out.columns if c not in ordered]
    out = out[ordered + extra]
    stats.rows = len(out)
    return out


# --------------------------------------------------------------------------- #
# FRED
# --------------------------------------------------------------------------- #
def fetch_fred_series(
    series_id: str,
    timeout: int = 45,
    max_retries: int = 5,
) -> pd.Series:
    """Download one FRED series as a float ``Series`` indexed by date.

    FRED encodes missing observations as ``.`` which becomes NaN here.
    """
    global _last_fred_call
    wait = FRED_MIN_INTERVAL_SEC - (time.monotonic() - _last_fred_call)
    if wait > 0:
        time.sleep(wait)
    try:
        text = _http_get(FRED_CSV_URL.format(series_id=series_id), timeout, max_retries)
    finally:
        _last_fred_call = time.monotonic()
    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] < 2:
        raise DataFetchError(f"Unexpected FRED payload for {series_id}: {list(df.columns)}")
    date_col = df.columns[0]
    value_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col)
    s = pd.to_numeric(df[value_col], errors="coerce")
    s.index.name = "date"
    s.name = series_id
    return s.sort_index()


def fetch_fred_bundle(
    series_map: dict[str, str],
    cache_dir: Path,
    timeout: int = 45,
    max_retries: int = 5,
    force: bool = False,
    max_age_days: int = 1,
    stats: FetchStats | None = None,
) -> pd.DataFrame:
    """Download several FRED series and align them on a daily date index.

    ``series_map`` maps a friendly column name to the FRED series id.  Series are
    cached individually so adding one to the config does not re-download the rest.
    Monthly series (CPI, UNRATE) are **not** forward-filled here - that happens in
    the feature layer, where the publication lag is applied explicitly to avoid
    look-ahead.
    """
    stats = stats or FetchStats()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cols: dict[str, pd.Series] = {}

    for name, series_id in series_map.items():
        path = cache_dir / f"fred_{series_id}.parquet"
        fresh = path.exists() and not force
        if fresh:
            age = (time.time() - path.stat().st_mtime) / 86400.0
            fresh = age <= max_age_days
        if fresh:
            cols[name] = pd.read_parquet(path).iloc[:, 0]
            stats.from_cache += 1
            continue
        try:
            s = fetch_fred_series(series_id, timeout, max_retries)
        except DataFetchError as exc:
            if path.exists():
                log.warning("FRED %s failed (%s); using cache", series_id, exc)
                cols[name] = pd.read_parquet(path).iloc[:, 0]
                stats.from_cache += 1
            else:
                log.warning("FRED %s unavailable and uncached: %s", series_id, exc)
                stats.failed += 1
            continue
        s.to_frame(name=series_id).to_parquet(path)
        cols[name] = s
        stats.downloaded += 1

    if not cols:
        return pd.DataFrame()
    out = pd.DataFrame(cols).sort_index()
    out.index.name = "date"
    stats.rows = len(out)
    return out


# --------------------------------------------------------------------------- #
# Cleaning / validation
# --------------------------------------------------------------------------- #
def validate_curve(df: pd.DataFrame, max_daily_move_bp: float = 150.0) -> pd.DataFrame:
    """Flag implausible observations without silently deleting history.

    Returns a boolean frame of the same shape, ``True`` where the observation
    looks wrong: a level outside [-2%, 25%] or a one-day move larger than
    ``max_daily_move_bp``.  The largest genuine single-day 10y move on record is
    ~50bp, so 150bp is a wide net that only catches data errors.
    """
    level_bad = (df < -0.02) | (df > 0.25)
    move_bad = df.diff().abs() > (max_daily_move_bp * 1e-4)
    return (level_bad | move_bad).fillna(False)


def clean_curve(
    df: pd.DataFrame,
    max_daily_move_bp: float = 150.0,
    drop_flagged: bool = True,
) -> pd.DataFrame:
    """Apply :func:`validate_curve` and blank out anything flagged."""
    flags = validate_curve(df, max_daily_move_bp)
    n_bad = int(flags.to_numpy().sum())
    if n_bad:
        log.warning("curve validation flagged %d observations", n_bad)
    if not drop_flagged:
        return df
    out = df.copy()
    out[flags] = np.nan
    return out


def curve_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per-tenor coverage summary - first/last observation and missing count."""
    rows = []
    for col in df.columns:
        s = df[col].dropna()
        rows.append(
            {
                "tenor": col,
                "years": TENOR_YEARS.get(col, np.nan),
                "n_obs": int(len(s)),
                "first": s.index.min() if len(s) else pd.NaT,
                "last": s.index.max() if len(s) else pd.NaT,
                "pct_missing": float(1.0 - len(s) / max(len(df), 1)),
            }
        )
    return pd.DataFrame(rows).set_index("tenor")


# --------------------------------------------------------------------------- #
# Top-level convenience
# --------------------------------------------------------------------------- #
def load_market_data(
    cfg: Config,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull (or read from cache) the par-yield curve and the macro bundle.

    Returns ``(curve, macro)``, both indexed by date and restricted to the
    configured window.
    """
    start = pd.Timestamp(cfg.data.start_date)
    end = pd.Timestamp(cfg.data.end_date) if cfg.data.end_date else pd.Timestamp(dt.date.today())

    cstats = FetchStats()
    curve = fetch_treasury_curve(
        start_year=int(start.year),
        end_year=int(end.year),
        cache_dir=cfg.cache_dir,
        timeout=cfg.data.request_timeout,
        max_retries=cfg.data.max_retries,
        force=force,
        stats=cstats,
    )
    log.info("treasury curve: %s", cstats)
    curve = clean_curve(curve).loc[start:end]

    macro = pd.DataFrame()
    if cfg.features.include_macro and cfg.data.fred_series:
        mstats = FetchStats()
        macro = fetch_fred_bundle(
            cfg.data.fred_series,
            cache_dir=cfg.cache_dir,
            timeout=cfg.data.request_timeout,
            max_retries=cfg.data.max_retries,
            force=force,
            max_age_days=cfg.data.cache_days,
            stats=mstats,
        )
        log.info("fred bundle: %s", mstats)
        if not macro.empty:
            macro = macro.loc[: end]

    return curve, macro
