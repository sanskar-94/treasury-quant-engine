"""On-disk Parquet cache for anything expensive to recompute.

Two things this exists for.

*Speed.*  The constant-maturity total-return panel is ~1s to build and the NSS
history is minutes; neither changes unless the underlying curve does, and both
are needed by the CLI, the API and every notebook.

*Reproducibility.*  A backtest that silently rebuilt its inputs from a live
download would not be reproducible.  Caching to Parquet pins the exact inputs a
result was produced from, and ``max_age_days`` is the knob that decides when a
cached artefact is considered stale rather than when it happens to be convenient
to refresh.

Writes are **atomic**: the payload goes to a temporary file in the destination
directory and is then ``os.replace``d into place.  A process killed mid-write
therefore leaves the previous good cache intact rather than a truncated Parquet
file that fails to read on the next run.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from ..logging_utils import get_logger

__all__ = [
    "save_frame",
    "load_frame",
    "cache_path",
    "cached",
    "clear_cache",
    "cache_info",
]

log = get_logger("data.cache")

_SUFFIX = ".parquet"
_MANIFEST_SUFFIX = ".json"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(key: str) -> str:
    """Filesystem-safe cache key. ``"total_return 10 Yr" -> "total_return_10_Yr"``."""
    slug = _UNSAFE.sub("_", key.strip()).strip("._-")
    if not slug:
        raise ValueError(f"cache key {key!r} reduces to an empty filename")
    return slug


# --------------------------------------------------------------------------- #
# Single-frame IO
# --------------------------------------------------------------------------- #
def save_frame(df: pd.DataFrame | pd.Series, path: str | Path) -> Path:
    """Write a frame to Parquet atomically, preserving the index.

    Parameters
    ----------
    df:
        Frame (or Series, which is widened to a one-column frame) to persist.
        A ``DatetimeIndex`` and its name survive the round trip via pandas'
        Parquet metadata.
    path:
        Destination file.  Parent directories are created.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    ValueError
        If the frame has non-string column labels, which Parquet cannot store
        unambiguously.  Failing loudly here beats discovering on reload that
        column ``10`` came back as ``"10"``.
    """
    frame = df.to_frame() if isinstance(df, pd.Series) else df
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"save_frame expects a DataFrame or Series, got {type(df).__name__}")
    bad = [c for c in frame.columns if not isinstance(c, str)]
    if bad:
        raise ValueError(f"Parquet needs string column names; offending labels: {bad[:5]}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(tmp, engine="pyarrow", index=True)
        os.replace(tmp, path)  # atomic within a filesystem
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def load_frame(path: str | Path) -> pd.DataFrame:
    """Read a frame written by :func:`save_frame`.

    Raises
    ------
    FileNotFoundError
        If the cache entry does not exist.  Callers that want a silent miss
        should go through :func:`cached`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No cached frame at {path}")
    return pd.read_parquet(path, engine="pyarrow")


def cache_path(key: str, cache_dir: str | Path) -> Path:
    """Where :func:`cached` would store ``key``."""
    return Path(cache_dir) / f"{_slug(key)}{_SUFFIX}"


# --------------------------------------------------------------------------- #
# Memoisation
# --------------------------------------------------------------------------- #
def _manifest_path(key: str, cache_dir: Path) -> Path:
    return cache_dir / f"{_slug(key)}{_MANIFEST_SUFFIX}"


def _dir_path(key: str, cache_dir: Path) -> Path:
    return cache_dir / _slug(key)


def _age_days(manifest: Mapping[str, Any], fallback: Path) -> float:
    """Age of a cache entry in days, from the manifest if present, mtime if not."""
    stamp = manifest.get("created_utc")
    if isinstance(stamp, str):
        try:
            created = datetime.fromisoformat(stamp)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - created).total_seconds() / 86400.0
        except ValueError:
            pass
    return (time.time() - fallback.stat().st_mtime) / 86400.0


def cached(
    key: str,
    builder: Callable[[], Any],
    cache_dir: str | Path,
    max_age_days: float | None = None,
) -> Any:
    """Return ``key`` from the Parquet cache, building and storing it on a miss.

    Parameters
    ----------
    key:
        Logical name of the artefact.  Include everything that changes the
        result - ``"cmtr_core_v2"`` rather than ``"cmtr"`` - because the cache
        cannot tell that the code which produced an entry has changed.
    builder:
        Zero-argument callable producing a ``DataFrame``, a ``Series``, or a
        ``dict[str, DataFrame]`` (the shape
        :func:`tqe.data.universe.constant_maturity_total_return` returns).
    cache_dir:
        Directory to store entries in; created if absent.
    max_age_days:
        Treat an entry older than this as a miss.  ``None`` (default) means the
        entry never expires - correct for artefacts derived from a fixed
        historical file, wrong for anything that tracks a live feed.

    Returns
    -------
    Any
        Whatever ``builder`` returns, either loaded or freshly built.

    Notes
    -----
    A corrupt or unreadable entry is logged and treated as a miss rather than
    raised: a stale cache should never be able to take the daily run down, and
    the rebuild is deterministic anyway.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = _manifest_path(key, cache_dir)

    if manifest_file.exists():
        try:
            manifest = json.loads(manifest_file.read_text())
            age = _age_days(manifest, manifest_file)
            if max_age_days is not None and age > max_age_days:
                log.info("cache %r is %.1f days old (limit %.1f); rebuilding", key, age, max_age_days)
            else:
                payload = _load_payload(key, cache_dir, manifest)
                log.debug("cache hit %r (%s, %.1f days old)", key, manifest.get("kind"), age)
                return payload
        except Exception as exc:  # noqa: BLE001 - a bad cache must never be fatal
            log.warning("cache %r unreadable (%s); rebuilding", key, exc)

    value = builder()
    try:
        _store_payload(key, cache_dir, value)
    except Exception as exc:  # noqa: BLE001 - failing to cache is not failing to compute
        log.warning("could not cache %r (%s); returning the freshly built value", key, exc)
    return value


def _store_payload(key: str, cache_dir: Path, value: Any) -> None:
    """Persist ``value`` plus a manifest describing how to read it back."""
    manifest: dict[str, Any] = {
        "key": key,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    if isinstance(value, dict):
        if not all(isinstance(v, (pd.DataFrame, pd.Series)) for v in value.values()):
            raise TypeError("dict payloads must map str -> DataFrame/Series")
        target = _dir_path(key, cache_dir)
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        members = {}
        for name, frame in value.items():
            slug = _slug(str(name))
            save_frame(frame, target / f"{slug}{_SUFFIX}")
            members[slug] = str(name)  # slug -> original label, so " " survives
        manifest.update(kind="frames", members=members, n=len(members))
    elif isinstance(value, pd.Series):
        save_frame(value, cache_path(key, cache_dir))
        manifest.update(kind="series", name=value.name, rows=int(len(value)))
    elif isinstance(value, pd.DataFrame):
        save_frame(value, cache_path(key, cache_dir))
        manifest.update(kind="frame", rows=int(len(value)), columns=[str(c) for c in value.columns])
    else:
        raise TypeError(f"cached() can only persist DataFrame/Series/dict, got {type(value).__name__}")

    _manifest_path(key, cache_dir).write_text(json.dumps(manifest, indent=2, default=str))


def _load_payload(key: str, cache_dir: Path, manifest: Mapping[str, Any]) -> Any:
    kind = manifest.get("kind", "frame")
    if kind == "frames":
        target = _dir_path(key, cache_dir)
        members: Mapping[str, str] = manifest.get("members", {})
        return {label: load_frame(target / f"{slug}{_SUFFIX}") for slug, label in members.items()}
    frame = load_frame(cache_path(key, cache_dir))
    if kind == "series":
        series = frame.iloc[:, 0]
        series.name = manifest.get("name")
        return series
    return frame


# --------------------------------------------------------------------------- #
# Housekeeping
# --------------------------------------------------------------------------- #
def clear_cache(cache_dir: str | Path, pattern: str = "*") -> int:
    """Delete cache entries under ``cache_dir``; return how many were removed.

    ``pattern`` is a glob on the *slugged* key, so ``clear_cache(d, "cmtr_*")``
    drops every total-return artefact and leaves the raw downloads alone.
    Anything that is not a cache entry (``.gitkeep``, a user file) is left
    untouched.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return 0

    removed = 0
    for manifest_file in sorted(cache_dir.glob(f"{pattern}{_MANIFEST_SUFFIX}")):
        stem = manifest_file.stem
        for victim in (cache_dir / f"{stem}{_SUFFIX}", cache_dir / stem):
            if victim.is_dir():
                shutil.rmtree(victim)
            elif victim.exists():
                victim.unlink()
        manifest_file.unlink()
        removed += 1

    # Bare frames written by save_frame() without going through cached().
    for orphan in sorted(cache_dir.glob(f"{pattern}{_SUFFIX}")):
        if (cache_dir / f"{orphan.stem}{_MANIFEST_SUFFIX}").exists():
            continue
        orphan.unlink()
        removed += 1

    log.info("cleared %d cache entr%s from %s", removed, "y" if removed == 1 else "ies", cache_dir)
    return removed


def cache_info(cache_dir: str | Path) -> pd.DataFrame:
    """Inventory of the cache: key, kind, size in MB and age in days.

    Surfaced by ``tqe data status`` so it is obvious when a result is being
    served from a month-old artefact.
    """
    cache_dir = Path(cache_dir)
    rows: list[dict[str, Any]] = []
    if not cache_dir.exists():
        return pd.DataFrame(columns=["key", "kind", "size_mb", "age_days"])

    for manifest_file in sorted(cache_dir.glob(f"*{_MANIFEST_SUFFIX}")):
        try:
            manifest = json.loads(manifest_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        stem = manifest_file.stem
        payload = cache_dir / f"{stem}{_SUFFIX}"
        directory = cache_dir / stem
        size = payload.stat().st_size if payload.exists() else 0
        if directory.is_dir():
            size += sum(p.stat().st_size for p in directory.rglob(f"*{_SUFFIX}"))
        rows.append(
            {
                "key": manifest.get("key", stem),
                "kind": manifest.get("kind", "frame"),
                "size_mb": round(size / 1e6, 3),
                "age_days": round(_age_days(manifest, manifest_file), 3),
            }
        )
    return pd.DataFrame(rows, columns=["key", "kind", "size_mb", "age_days"])
