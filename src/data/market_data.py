"""OHLCV historical market data fetcher with local disk caching and symbol fallback resolution."""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _safe_filename(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", symbol.upper().strip())


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output to a clean tz-naive daily OHLCV DataFrame."""
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLS)

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] for c in out.columns]

    missing = [c for c in ["Open", "High", "Low", "Close"] if c not in out.columns]
    if missing:
        return pd.DataFrame(columns=OHLCV_COLS)

    if "Volume" not in out.columns:
        out["Volume"] = 0.0

    out = out[OHLCV_COLS].copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    # Normalize to midnight date so cross-market series (e.g. US + KRX) align cleanly
    out.index = out.index.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()

    for col in OHLCV_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Close"])
    out = out[out["Close"] > 0]
    out["Volume"] = out["Volume"].fillna(0.0).clip(lower=0.0)
    out["Open"] = out["Open"].fillna(out["Close"])
    out["High"] = out[["High", "Open", "Close"]].max(axis=1)
    out["Low"] = out[["Low", "Open", "Close"]].min(axis=1)
    return out


def build_aligned_close_matrix(
    ohlcv_map: dict[str, pd.DataFrame],
    symbols: list[str],
    min_obs: int = 25,
) -> pd.DataFrame:
    """Build a date-aligned Close price matrix across multiple symbols."""
    series_list: list[pd.Series] = []
    for sym in symbols:
        df = ohlcv_map.get(sym)
        if df is None or df.empty or "Close" not in df.columns or len(df) < min_obs:
            continue
        s = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(s) < min_obs:
            continue
        idx = pd.to_datetime(s.index, errors="coerce")
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        s = pd.Series(s.values, index=idx.normalize(), name=sym)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        series_list.append(s)

    if not series_list:
        return pd.DataFrame()

    mat = pd.concat(series_list, axis=1).sort_index()
    # Bridge short single-market holiday gaps before aligning overlapping history
    mat = mat.ffill(limit=3)
    mat = mat.dropna(how="any")
    if len(mat) < min_obs:
        # Relax to columns that share sufficient overlapping dates
        keep_cols = [c for c in mat.columns if mat[c].notna().sum() >= min_obs]
        if not keep_cols:
            mat = pd.concat(series_list, axis=1).sort_index().ffill(limit=3)
            # Drop the shortest-history column iteratively if needed
            while mat.shape[1] > 1 and len(mat.dropna(how="any")) < min_obs:
                worst = mat.isna().sum().idxmax()
                mat = mat.drop(columns=[worst])
            mat = mat.dropna(how="any")
    return mat


class MarketDataFetcher:
    """Downloads and caches daily OHLCV bars via yfinance with multi-ticker fallback support."""

    def __init__(
        self,
        data_dir: Path | str,
        history_years: int = 10,
        cache_ttl_hours: float = 6.0,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "ohlcv"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.history_years = max(1, int(history_years))
        self.cache_ttl_hours = float(cache_ttl_hours)

    def _cache_paths(self, symbol: str) -> tuple[Path, Path]:
        safe = _safe_filename(symbol)
        return self.cache_dir / f"{safe}.csv", self.cache_dir / f"{safe}.meta"

    def _load_cache(self, symbol: str, allow_stale: bool = False) -> tuple[pd.DataFrame, str] | None:
        csv_path, meta_path = self._cache_paths(symbol)
        if not csv_path.exists():
            return None
        age_hours = (time.time() - csv_path.stat().st_mtime) / 3600.0
        if not allow_stale and age_hours > self.cache_ttl_hours:
            return None
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df = _normalize_ohlcv(df)
            if df.empty:
                return None
            resolved = meta_path.read_text(encoding="utf-8").strip() if meta_path.exists() else symbol
            return df, resolved or symbol
        except Exception as exc:
            logger.debug("Cache read failed for %s: %s", symbol, exc)
            return None

    def _save_cache(self, symbol: str, df: pd.DataFrame, resolved: str) -> None:
        csv_path, meta_path = self._cache_paths(symbol)
        try:
            df.to_csv(csv_path)
            meta_path.write_text(resolved, encoding="utf-8")
        except Exception as exc:
            logger.debug("Cache write failed for %s: %s", symbol, exc)

    def fetch_one(
        self,
        symbol: str,
        fallbacks: list[str] | None = None,
        force_refresh: bool = False,
    ) -> tuple[pd.DataFrame, str]:
        """Fetch daily OHLCV history for one symbol, trying fallback Yahoo tickers if needed."""
        sym = symbol.upper().strip()
        if not force_refresh:
            cached = self._load_cache(sym, allow_stale=False)
            if cached is not None:
                return cached

        candidates: list[str] = [sym]
        for fb in fallbacks or []:
            fb_clean = str(fb).upper().strip()
            if fb_clean and fb_clean not in candidates:
                candidates.append(fb_clean)

        if yf is not None:
            period = f"{self.history_years}y"
            for cand in candidates:
                try:
                    hist = yf.Ticker(cand).history(period=period, auto_adjust=True, actions=False)
                    df = _normalize_ohlcv(hist)
                    if not df.empty and len(df) >= 5:
                        self._save_cache(sym, df, cand)
                        return df, cand
                except Exception as exc:
                    logger.debug("yfinance fetch failed for %s (candidate %s): %s", sym, cand, exc)

        # Fallback to stale disk cache if live download fails
        stale = self._load_cache(sym, allow_stale=True)
        if stale is not None:
            return stale

        return pd.DataFrame(columns=OHLCV_COLS), sym

    def fetch_universe(
        self,
        universe_cfg: dict[str, Any],
        extra_symbols: list[str] | None = None,
        force_refresh: bool = False,
        max_workers: int = 6,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
        """Fetch daily OHLCV bars for all benchmarks, universe equities, and extra symbols."""
        benchmarks = list((universe_cfg.get("benchmarks") or {}).keys())
        u_meta = universe_cfg.get("universe") or {}
        eq_syms = list(u_meta.keys())
        extras = [str(s).upper().strip() for s in (extra_symbols or []) if s]

        all_syms = list(dict.fromkeys(benchmarks + eq_syms + extras))
        ohlcv_map: dict[str, pd.DataFrame] = {}
        resolved_map: dict[str, str] = {}

        def _worker(s: str) -> tuple[str, pd.DataFrame, str]:
            fbs = list((u_meta.get(s) or {}).get("yahoo_fallbacks") or [])
            df, res = self.fetch_one(s, fallbacks=fbs, force_refresh=force_refresh)
            return s, df, res

        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(all_syms)))) as pool:
            futs = [pool.submit(_worker, s) for s in all_syms]
            for fut in as_completed(futs):
                try:
                    s, df, res = fut.result()
                    if not df.empty:
                        ohlcv_map[s] = df
                    resolved_map[s] = res
                except Exception as exc:
                    logger.warning("Universe fetch error: %s", exc)

        return ohlcv_map, resolved_map
