"""OHLCV historical market data fetcher with local disk caching and symbol fallback resolution."""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
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


def _safe_pos_float(val: Any) -> float | None:
    try:
        if val is None or pd.isna(val):
            return None
        f = float(val)
        return f if np.isfinite(f) and f > 0 else None
    except Exception:
        return None


def _safe_int(val: Any) -> int:
    try:
        if val is None or pd.isna(val):
            return 0
        return int(val)
    except Exception:
        return 0


def _pct_vs_ref(price: float | None, ref: float | None) -> float | None:
    if price is None or ref is None or ref <= 0:
        return None
    return round((price / ref - 1.0) * 100.0, 2)


def classify_sector_from_info(info: dict[str, Any] | None) -> str:
    """Map Yahoo Finance sector/industry metadata into a Front-Running supply-chain bucket."""
    meta = info or {}
    industry = str(meta.get("industry") or "").lower()
    sector = str(meta.get("sector") or "").lower()
    combined = f"{sector} {industry}"

    if any(k in combined for k in ("data storage", "memory", "computer hardware", "storage")):
        return "memory_storage"
    if any(k in combined for k in ("semiconductor", "communication equipment", "electronic components", "optical")):
        return "compute_optical"
    if any(k in combined for k in ("software", "internet", "cloud", "information technology")):
        return "cloud_ai"
    if any(k in combined for k in ("consumer defensive", "consumer cyclical", "utilities", "healthcare", "restaurant")):
        return "defensive"
    return "other"


def extract_extended_quote_from_info(
    symbol: str,
    info: dict[str, Any] | None,
    resolved_symbol: str | None = None,
) -> dict[str, Any]:
    """Parse regular, pre-market, and after-hours quote fields from a yfinance info dict."""
    meta = info or {}
    sym = symbol.upper().strip()
    resolved = (resolved_symbol or sym).upper().strip()

    raw_state = str(meta.get("marketState") or "UNKNOWN").upper().strip()
    reg_px = (
        _safe_pos_float(meta.get("regularMarketPrice"))
        or _safe_pos_float(meta.get("currentPrice"))
        or _safe_pos_float(meta.get("previousClose"))
    )
    prev_close = (
        _safe_pos_float(meta.get("regularMarketPreviousClose"))
        or _safe_pos_float(meta.get("previousClose"))
    )
    reg_chg_pct = _pct_vs_ref(reg_px, prev_close)

    pre_px = _safe_pos_float(meta.get("preMarketPrice"))
    post_px = _safe_pos_float(meta.get("postMarketPrice"))
    pre_ts = _safe_int(meta.get("preMarketTime"))
    post_ts = _safe_int(meta.get("postMarketTime"))

    # Anchor extended-hours % change against regular market price (or previous close)
    ext_ref = reg_px or prev_close
    pre_chg_pct = _pct_vs_ref(pre_px, ext_ref)
    post_chg_pct = _pct_vs_ref(post_px, ext_ref)

    # Determine which extended session is currently active or most recent
    ext_session: str | None = None
    ext_px: float | None = None
    ext_chg_pct: float | None = None

    if raw_state in ("PRE", "PREPRE"):
        if pre_px is not None:
            ext_session, ext_px, ext_chg_pct = "PRE", pre_px, pre_chg_pct
    elif raw_state in ("POST", "POSTPOST"):
        if post_px is not None:
            ext_session, ext_px, ext_chg_pct = "POST", post_px, post_chg_pct
    elif raw_state != "REGULAR":
        # Market closed / unknown: pick whichever extended quote has the newer timestamp
        if pre_px is not None and post_px is not None:
            if pre_ts >= post_ts and pre_ts > 0:
                ext_session, ext_px, ext_chg_pct = "PRE", pre_px, pre_chg_pct
            else:
                ext_session, ext_px, ext_chg_pct = "POST", post_px, post_chg_pct
        elif pre_px is not None:
            ext_session, ext_px, ext_chg_pct = "PRE", pre_px, pre_chg_pct
        elif post_px is not None:
            ext_session, ext_px, ext_chg_pct = "POST", post_px, post_chg_pct

    latest_px = ext_px if ext_px is not None else reg_px

    return {
        "symbol": sym,
        "resolved_symbol": resolved,
        "market_state": raw_state,
        "regular_price": round(reg_px, 2) if reg_px is not None else None,
        "regular_change_pct": reg_chg_pct,
        "pre_price": round(pre_px, 2) if pre_px is not None else None,
        "pre_change_pct": pre_chg_pct,
        "post_price": round(post_px, 2) if post_px is not None else None,
        "post_change_pct": post_chg_pct,
        "ext_session": ext_session,
        "ext_price": round(ext_px, 2) if ext_px is not None else None,
        "ext_change_pct": ext_chg_pct,
        "latest_price": round(latest_px, 2) if latest_px is not None else None,
        "fetched_at": datetime.now().strftime("%H:%M:%S"),
    }


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

    def fetch_live_quote(
        self,
        symbol: str,
        fallbacks: list[str] | None = None,
        include_intraday_fallback: bool = False,
    ) -> dict[str, Any]:
        """Fetch live regular + pre-market / after-hours quote without modifying daily OHLCV cache."""
        sym = symbol.upper().strip()
        candidates: list[str] = [sym]
        for fb in fallbacks or []:
            fb_clean = str(fb).upper().strip()
            if fb_clean and fb_clean not in candidates:
                candidates.append(fb_clean)

        if yf is None:
            return extract_extended_quote_from_info(sym, {}, sym)

        for cand in candidates:
            try:
                tk = yf.Ticker(cand)
                info = tk.info or {}
                quote = extract_extended_quote_from_info(sym, info, cand)
                if quote["regular_price"] is not None or quote["ext_price"] is not None:
                    # Optional prepost=True 1m fallback when Yahoo omits preMarketPrice in .info
                    if (
                        include_intraday_fallback
                        and quote["ext_price"] is None
                        and quote["market_state"] in ("PRE", "PREPRE", "POST", "POSTPOST")
                    ):
                        try:
                            intra = tk.history(period="2d", interval="1m", prepost=True, actions=False)
                            if intra is not None and not intra.empty and "Close" in intra.columns:
                                last_intra = _safe_pos_float(intra["Close"].iloc[-1])
                                if last_intra is not None:
                                    sess = "PRE" if quote["market_state"] in ("PRE", "PREPRE") else "POST"
                                    ref = quote["regular_price"]
                                    quote["ext_session"] = sess
                                    quote["ext_price"] = round(last_intra, 2)
                                    quote["ext_change_pct"] = _pct_vs_ref(last_intra, ref)
                                    quote["latest_price"] = round(last_intra, 2)
                                    if sess == "PRE":
                                        quote["pre_price"] = quote["ext_price"]
                                        quote["pre_change_pct"] = quote["ext_change_pct"]
                                    else:
                                        quote["post_price"] = quote["ext_price"]
                                        quote["post_change_pct"] = quote["ext_change_pct"]
                        except Exception:
                            pass
                    return quote
            except Exception as exc:
                logger.debug("Live quote fetch failed for %s (candidate %s): %s", sym, cand, exc)

        return extract_extended_quote_from_info(sym, {}, sym)

    def fetch_live_quotes_batch(
        self,
        universe_cfg: dict[str, Any],
        extra_symbols: list[str] | None = None,
        include_benchmarks: bool = True,
        max_workers: int = 8,
    ) -> dict[str, dict[str, Any]]:
        """Concurrently fetch live regular + pre-market/after-hours quotes for watchlist & benchmarks."""
        u_meta = universe_cfg.get("universe") or {}
        eq_syms = list(u_meta.keys())
        bench_syms = ["SPY", "QQQ", "SOXX"] if include_benchmarks else []
        extras = [str(s).upper().strip() for s in (extra_symbols or []) if s]

        all_syms = list(dict.fromkeys(bench_syms + eq_syms + extras))
        out: dict[str, dict[str, Any]] = {}

        def _worker(s: str) -> tuple[str, dict[str, Any]]:
            fbs = list((u_meta.get(s) or {}).get("yahoo_fallbacks") or [])
            return s, self.fetch_live_quote(s, fallbacks=fbs)

        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(all_syms)))) as pool:
            futs = [pool.submit(_worker, s) for s in all_syms]
            for fut in as_completed(futs):
                try:
                    s, q = fut.result()
                    out[s] = q
                except Exception as exc:
                    logger.debug("Batch live quote error: %s", exc)

        return out

    def fetch_ticker_metadata(self, symbol: str) -> dict[str, Any]:
        """Fetch display name, inferred sector bucket, and live quote for a newly added ticker."""
        sym = symbol.upper().strip()
        info: dict[str, Any] = {}
        if yf is not None:
            try:
                info = yf.Ticker(sym).info or {}
            except Exception as exc:
                logger.debug("Metadata fetch failed for %s: %s", sym, exc)
        name = str(info.get("shortName") or info.get("longName") or sym).strip()
        sector = classify_sector_from_info(info)
        quote = extract_extended_quote_from_info(sym, info, sym)
        return {
            "symbol": sym,
            "name": name,
            "sector": sector,
            "quote": quote,
        }
