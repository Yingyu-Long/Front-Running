"""Fundamental quality metrics, earnings calendar guard, and EPS surprise fetcher via yfinance."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None  # type: ignore[assignment]

from src.storage.db import TradingDB

logger = logging.getLogger(__name__)


def _safe_float(val: Any) -> float | None:
    try:
        if val is None or pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


def _compute_fundamental_score(
    rev_yoy: float | None,
    gross_margin: float | None,
    op_margin: float | None,
    fwd_pe: float | None,
    eps_surprise: float | None,
) -> float:
    """Compute a transparent 0-100 fundamental quality score."""
    score = 50.0
    if rev_yoy is not None:
        if rev_yoy >= 25.0:
            score += 18.0
        elif rev_yoy >= 12.0:
            score += 11.0
        elif rev_yoy >= 3.0:
            score += 4.0
        elif rev_yoy <= -10.0:
            score -= 14.0
        else:
            score -= 5.0

    if gross_margin is not None:
        if gross_margin >= 55.0:
            score += 10.0
        elif gross_margin >= 35.0:
            score += 5.0
        elif gross_margin < 20.0:
            score -= 6.0

    if op_margin is not None:
        if op_margin >= 25.0:
            score += 12.0
        elif op_margin >= 12.0:
            score += 6.0
        elif op_margin < 0.0:
            score -= 12.0

    if fwd_pe is not None and fwd_pe > 0:
        if 10.0 <= fwd_pe <= 35.0:
            score += 5.0
        elif fwd_pe > 65.0:
            score -= 6.0

    if eps_surprise is not None:
        if eps_surprise >= 5.0:
            score += 7.0
        elif eps_surprise > 0.0:
            score += 3.0
        elif eps_surprise <= -5.0:
            score -= 9.0

    return round(max(10.0, min(95.0, score)), 1)


def _format_earnings_flag_zh(days: int | None) -> str:
    if days is None:
        return "待确认"
    if days <= 10:
        return f"高危 ({days}天)"
    if days <= 21:
        return f"临近 ({days}天)"
    return f"安全 ({days}天)"


class FundamentalFetcher:
    """Fetches equity valuation, profitability, earnings dates, and EPS surprises with SQLite caching."""

    def __init__(self, db: TradingDB | None = None, cache_ttl_hours: float = 24.0) -> None:
        self.db = db
        self.cache_ttl_hours = float(cache_ttl_hours)

    def fetch_one(
        self,
        symbol: str,
        resolved_symbol: str | None = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        sym = symbol.upper().strip()
        cache_key = f"fund:{sym}"
        if self.db is not None and not force_refresh:
            cached = self.db.get_cache(cache_key, max_age_hours=self.cache_ttl_hours)
            if isinstance(cached, dict):
                return cached

        result: dict[str, Any] = {
            "symbol": sym,
            "market_cap_b": None,
            "forward_pe": None,
            "trailing_pe": None,
            "revenue_growth_yoy_pct": None,
            "gross_margin_pct": None,
            "operating_margin_pct": None,
            "last_eps_surprise_pct": None,
            "next_earnings_date": None,
            "days_to_earnings": None,
            "earnings_risk_flag": "待确认",
            "fundamental_score": 50.0,
        }

        if yf is None:
            return result

        target = (resolved_symbol or sym).upper().strip()
        try:
            tk = yf.Ticker(target)
            info = tk.info or {}

            mcap = _safe_float(info.get("marketCap"))
            fwd_pe = _safe_float(info.get("forwardPE"))
            trl_pe = _safe_float(info.get("trailingPE"))
            rev_g = _safe_float(info.get("revenueGrowth"))
            gm = _safe_float(info.get("grossMargins"))
            om = _safe_float(info.get("operatingMargins"))

            if mcap is not None and mcap > 0:
                result["market_cap_b"] = round(mcap / 1e9, 1)
            if fwd_pe is not None and fwd_pe > 0:
                result["forward_pe"] = round(fwd_pe, 1)
            if trl_pe is not None and trl_pe > 0:
                result["trailing_pe"] = round(trl_pe, 1)
            if rev_g is not None:
                result["revenue_growth_yoy_pct"] = round(rev_g * 100.0, 1)
            if gm is not None:
                result["gross_margin_pct"] = round(gm * 100.0, 1)
            if om is not None:
                result["operating_margin_pct"] = round(om * 100.0, 1)

            # Extract upcoming earnings date & most recent EPS surprise
            now_dt = datetime.now(timezone.utc).date()
            next_ed = None
            last_surp = None

            try:
                ed_df = tk.earnings_dates
                if ed_df is not None and not ed_df.empty:
                    idx = pd.to_datetime(ed_df.index, errors="coerce")
                    if getattr(idx, "tz", None) is not None:
                        idx = idx.tz_convert("UTC").tz_localize(None)
                    ed_dates = [d.date() for d in idx if not pd.isna(d)]
                    future_dates = sorted(d for d in ed_dates if d >= now_dt)
                    if future_dates:
                        next_ed = future_dates[0]

                    for col_name in ("Surprise(%)", "Surprise %"):
                        if col_name in ed_df.columns:
                            surp_series = pd.to_numeric(ed_df[col_name], errors="coerce").dropna()
                            if not surp_series.empty:
                                last_surp = round(float(surp_series.iloc[0]), 2)
                            break
            except Exception:
                pass

            if next_ed is None:
                try:
                    cal = tk.calendar
                    if isinstance(cal, dict):
                        ed_val = cal.get("Earnings Date")
                        if isinstance(ed_val, (list, tuple)) and ed_val:
                            ed_val = ed_val[0]
                        if ed_val is not None:
                            parsed_d = pd.to_datetime(ed_val, errors="coerce")
                            if not pd.isna(parsed_d):
                                cand_d = parsed_d.date()
                                if cand_d >= now_dt:
                                    next_ed = cand_d
                except Exception:
                    pass

            if next_ed is not None:
                days_left = int((next_ed - now_dt).days)
                result["next_earnings_date"] = next_ed.isoformat()
                result["days_to_earnings"] = days_left
                result["earnings_risk_flag"] = _format_earnings_flag_zh(days_left)

            if last_surp is not None:
                result["last_eps_surprise_pct"] = last_surp

            result["fundamental_score"] = _compute_fundamental_score(
                rev_yoy=result["revenue_growth_yoy_pct"],
                gross_margin=result["gross_margin_pct"],
                op_margin=result["operating_margin_pct"],
                fwd_pe=result["forward_pe"],
                eps_surprise=result["last_eps_surprise_pct"],
            )
        except Exception as exc:
            logger.debug("Fundamental fetch error for %s (%s): %s", sym, target, exc)

        if self.db is not None:
            self.db.set_cache(cache_key, result)
        return result

    def fetch_batch(
        self,
        symbols: list[str],
        resolved_map: dict[str, str] | None = None,
        force_refresh: bool = False,
        max_workers: int = 6,
    ) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        res_map = resolved_map or {}

        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(symbols)))) as pool:
            futs = {
                pool.submit(self.fetch_one, s, res_map.get(s), force_refresh): s
                for s in symbols
            }
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    out[s] = fut.result()
                except Exception as exc:
                    logger.debug("Batch fundamental error for %s: %s", s, exc)
                    out[s] = {"symbol": s, "fundamental_score": 50.0, "earnings_risk_flag": "待确认"}
        return out
