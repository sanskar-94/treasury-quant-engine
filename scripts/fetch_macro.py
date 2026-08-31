#!/usr/bin/env python
"""Patient, resumable FRED downloader.

The public ``fredgraph.csv`` endpoint aggressively rate-limits bursts, and it
does so by *hanging the connection* rather than returning HTTP 429 - so a naive
loop just collects read-timeouts.  This script therefore:

* fetches one series at a time with an adaptive delay (backs off on failure,
  eases off after a run of successes),
* caches each series to its own Parquet file the moment it arrives, so the job
  is fully resumable - re-running only fetches what is still missing,
* never gives up on a series until its own retry budget is exhausted.

Usage
-----
    python scripts/fetch_macro.py [--force] [--max-minutes 45]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqe.config import load_config  # noqa: E402
from tqe.logging_utils import get_logger, setup_logging  # noqa: E402

log = get_logger("scripts.fetch_macro")

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

MIN_DELAY = 8.0
MAX_DELAY = 150.0


def fetch_one(series_id: str, timeout: float = 30.0) -> pd.Series | None:
    """Single attempt.  Deliberately uses a fresh connection every time -
    a pooled ``requests.Session`` is markedly less reliable against this host.
    """
    try:
        resp = requests.get(FRED_URL.format(sid=series_id), timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        log.debug("%s failed: %s", series_id, exc)
        return None

    import io

    df = pd.read_csv(io.StringIO(resp.text))
    if df.shape[1] < 2 or df.empty:
        return None
    date_col, value_col = df.columns[0], df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col)
    s = pd.to_numeric(df[value_col], errors="coerce")
    s.index.name = "date"
    s.name = series_id
    return s.sort_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    ap.add_argument("--max-minutes", type=float, default=60.0, help="overall wall-clock budget")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    setup_logging("INFO")
    cfg = load_config(args.config)
    cache_dir = cfg.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    wanted = dict(cfg.data.fred_series)
    pending: list[tuple[str, str]] = []
    have: dict[str, pd.Series] = {}

    for name, sid in wanted.items():
        path = cache_dir / f"fred_{sid}.parquet"
        if path.exists() and not args.force:
            have[name] = pd.read_parquet(path).iloc[:, 0]
        else:
            pending.append((name, sid))

    log.info("macro pull: %d cached, %d to fetch", len(have), len(pending))

    deadline = time.monotonic() + args.max_minutes * 60.0
    delay = MIN_DELAY
    consecutive_ok = 0
    attempts: dict[str, int] = {}

    while pending and time.monotonic() < deadline:
        name, sid = pending.pop(0)
        attempts[sid] = attempts.get(sid, 0) + 1
        s = fetch_one(sid)
        if s is not None and len(s):
            path = cache_dir / f"fred_{sid}.parquet"
            s.to_frame(name=sid).to_parquet(path)
            have[name] = s
            consecutive_ok += 1
            log.info(
                "  ok  %-18s %-12s %6d obs  %s..%s  (delay %.0fs, %d left)",
                name, sid, len(s), s.index.min().date(), s.index.max().date(), delay, len(pending),
            )
            if consecutive_ok >= 3:
                delay = max(MIN_DELAY, delay * 0.7)
                consecutive_ok = 0
        else:
            consecutive_ok = 0
            delay = min(MAX_DELAY, delay * 1.8)
            if attempts[sid] < 8:
                pending.append((name, sid))  # back of the queue, try again later
                log.info("  retry %-18s %-12s attempt %d (delay now %.0fs)", name, sid, attempts[sid], delay)
            else:
                log.warning("  give up on %s (%s) after %d attempts", name, sid, attempts[sid])
        time.sleep(delay)

    if not have:
        log.error("no macro series retrieved")
        return 1

    out = pd.DataFrame(have).sort_index()
    out.index.name = "date"
    dest = Path(cfg.data.data_dir)
    if not dest.is_absolute():
        dest = cfg.root / dest
    dest = dest / "processed" / "macro.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest)

    log.info("wrote %s  shape=%s  %s..%s", dest, out.shape, out.index.min().date(), out.index.max().date())
    print(out.notna().sum().to_string())
    if pending:
        log.warning("still missing: %s", sorted({sid for _, sid in pending}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
