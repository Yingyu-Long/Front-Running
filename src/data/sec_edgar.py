"""Official SEC EDGAR company filings client (10-K, 10-Q, 8-K, Form 4, and red-flag screening)."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

from src.storage.db import TradingDB

logger = logging.getLogger(__name__)

# Sensitive 8-K item codes that commonly indicate governance, accounting, or impairment risks
SENSITIVE_8K_ITEMS: dict[str, str] = {
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "2.04": "Triggering Events / Acceleration of Financial Obligation",
    "2.05": "Exit or Disposal Activities / Restructuring",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Transfer Failure",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",
    "5.02": "Departure of Directors or Principal Officers",
}


class SecEdgarClient:
    """Lightweight SEC EDGAR API client with SQLite caching and rate-limit compliance."""

    def __init__(
        self,
        user_agent: str = "FrontRunningQuantResearch/1.0 (trader@local-research.dev)",
        db: TradingDB | None = None,
        cache_ttl_hours: float = 24.0,
    ) -> None:
        self.user_agent = user_agent
        self.db = db
        self.cache_ttl_hours = float(cache_ttl_hours)
        self._headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        self._cik_map: dict[str, str] | None = None

    def _load_ticker_cik_map(self) -> dict[str, str]:
        if self._cik_map is not None:
            return self._cik_map

        if self.db is not None:
            cached = self.db.get_cache("sec:ticker_cik_map", max_age_hours=168.0)
            if isinstance(cached, dict) and cached:
                self._cik_map = cached
                return cached

        mapping: dict[str, str] = {}
        if requests is not None:
            try:
                resp = requests.get(
                    "https://www.sec.gov/files/company_tickers.json",
                    headers=self._headers,
                    timeout=12.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    for entry in data.values():
                        tk = str(entry.get("ticker", "")).upper().strip()
                        cik_num = entry.get("cik_str")
                        if tk and cik_num is not None:
                            mapping[tk] = f"{int(cik_num):010d}"
                    if self.db is not None and mapping:
                        self.db.set_cache("sec:ticker_cik_map", mapping)
            except Exception as exc:
                logger.debug("Failed to load SEC company_tickers.json: %s", exc)

        self._cik_map = mapping
        return mapping

    def fetch_company_filings(
        self,
        symbol: str,
        force_refresh: bool = False,
        lookback_days: int = 90,
    ) -> dict[str, Any]:
        """Fetch recent SEC filings for one US equity and flag sensitive corporate events."""
        sym = symbol.upper().strip()
        cache_key = f"sec:{sym}"
        if self.db is not None and not force_refresh:
            cached = self.db.get_cache(cache_key, max_age_hours=self.cache_ttl_hours)
            if isinstance(cached, dict):
                return cached

        cik_map = self._load_ticker_cik_map()
        cik = cik_map.get(sym)
        if not cik or requests is None:
            empty_res = {
                "symbol": sym,
                "cik": cik,
                "recent_filings": [],
                "red_flag_filings": [],
                "sec_digest": "Non-US issuer or no SEC EDGAR CIK available.",
            }
            if self.db is not None:
                self.db.set_cache(cache_key, empty_res)
            return empty_res

        recent_filings: list[dict[str, Any]] = []
        red_flags: list[str] = []
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            resp = requests.get(url, headers=self._headers, timeout=12.0)
            if resp.status_code == 200:
                payload = resp.json()
                recent = (payload.get("filings") or {}).get("recent") or {}
                forms = recent.get("form") or []
                dates = recent.get("filingDate") or []
                items_list = recent.get("items") or []
                descs = recent.get("primaryDocDescription") or []

                form4_recent = 0
                counts: dict[str, int] = {}
                for i in range(min(len(forms), len(dates), 120)):
                    fdate = str(dates[i])
                    if fdate < cutoff:
                        continue
                    form = str(forms[i]).upper().strip()
                    raw_items = items_list[i] if i < len(items_list) else []
                    item_codes = raw_items if isinstance(raw_items, list) else [str(raw_items)]
                    desc = str(descs[i]) if i < len(descs) else ""

                    if form in ("10-K", "10-Q", "8-K", "S-3", "S-3ASR", "SC 13D", "SC 13G", "4", "6-K", "20-F"):
                        counts[form] = counts.get(form, 0) + 1
                        if len(recent_filings) < 12:
                            recent_filings.append(
                                {
                                    "form": form,
                                    "date": fdate,
                                    "items": item_codes,
                                    "description": desc,
                                }
                            )

                    if form == "4":
                        form4_recent += 1
                    elif form == "8-K":
                        for code in item_codes:
                            code_s = str(code).strip()
                            if code_s in SENSITIVE_8K_ITEMS:
                                red_flags.append(
                                    f"[{fdate}] 8-K Item {code_s}: {SENSITIVE_8K_ITEMS[code_s]}"
                                )
                    elif form in ("S-3", "S-3ASR"):
                        red_flags.append(f"[{fdate}] {form} Shelf Registration (potential dilution watch)")

                summary_parts = [f"{k} x{v}" for k, v in sorted(counts.items())]
                latest_str = (
                    f"Latest: {recent_filings[0]['form']} on {recent_filings[0]['date']}"
                    if recent_filings
                    else "No major filings in past 90d"
                )
                digest = f"CIK {cik} | Past {lookback_days}d: {', '.join(summary_parts) or 'None'} | {latest_str}"
                if form4_recent >= 15:
                    digest += f" | Elevated Form 4 insider activity ({form4_recent} filings)"
            else:
                digest = f"SEC EDGAR returned HTTP {resp.status_code} for CIK {cik}."
        except Exception as exc:
            logger.debug("SEC EDGAR fetch error for %s: %s", sym, exc)
            digest = f"SEC EDGAR fetch unavailable ({exc})."

        result = {
            "symbol": sym,
            "cik": cik,
            "recent_filings": recent_filings,
            "red_flag_filings": red_flags[:5],
            "sec_digest": digest,
        }
        if self.db is not None:
            self.db.set_cache(cache_key, result)
        return result

    def fetch_batch(
        self,
        symbols: list[str],
        force_refresh: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Fetch SEC filings sequentially with gentle pacing to honor SEC EDGAR rate limits."""
        out: dict[str, dict[str, Any]] = {}
        for sym in symbols:
            out[sym] = self.fetch_company_filings(sym, force_refresh=force_refresh)
            time.sleep(0.12)
        return out
