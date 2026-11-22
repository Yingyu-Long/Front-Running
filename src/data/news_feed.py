"""Financial news headline aggregator combining Yahoo Finance ticker feeds and Google News RSS."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus

try:
    import feedparser
except ImportError:  # pragma: no cover
    feedparser = None  # type: ignore[assignment]

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None  # type: ignore[assignment]

from src.storage.db import TradingDB

logger = logging.getLogger(__name__)


def _parse_yf_news_item(raw: dict[str, Any]) -> dict[str, str] | None:
    """Support both legacy yfinance news dicts and newer nested content dicts."""
    if not isinstance(raw, dict):
        return None
    content = raw.get("content") if isinstance(raw.get("content"), dict) else raw
    title = str(content.get("title") or "").strip()
    if not title:
        return None

    provider = content.get("provider")
    if isinstance(provider, dict):
        publisher = str(provider.get("displayName") or "").strip()
    else:
        publisher = str(content.get("publisher") or "").strip()

    canonical = content.get("canonicalUrl")
    click_url = content.get("clickThroughUrl")
    if isinstance(canonical, dict) and canonical.get("url"):
        link = str(canonical["url"])
    elif isinstance(click_url, dict) and click_url.get("url"):
        link = str(click_url["url"])
    else:
        link = str(content.get("link") or "").strip()

    pub_str = ""
    if content.get("pubDate"):
        pub_str = str(content["pubDate"])[:19]
    elif content.get("providerPublishTime"):
        try:
            ts = int(content["providerPublishTime"])
            pub_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        except Exception:
            pub_str = ""

    return {
        "title": title,
        "publisher": publisher or "Yahoo Finance",
        "link": link,
        "published": pub_str,
    }


class NewsAggregator:
    """Aggregates recent ticker-specific news headlines with SQLite caching."""

    def __init__(self, db: TradingDB | None = None, cache_ttl_hours: float = 6.0) -> None:
        self.db = db
        self.cache_ttl_hours = float(cache_ttl_hours)

    def fetch_one(
        self,
        symbol: str,
        resolved_symbol: str | None = None,
        max_items: int = 8,
        force_refresh: bool = False,
        company_name: str | None = None,
    ) -> list[dict[str, Any]]:
        sym = symbol.upper().strip()
        cache_key = f"news:{sym}"
        if self.db is not None and not force_refresh:
            cached = self.db.get_cache(cache_key, max_age_hours=self.cache_ttl_hours)
            if isinstance(cached, list):
                return cached[:max_items]

        items: list[dict[str, Any]] = []
        seen_titles: set[str] = set()

        def _append(entry: dict[str, str] | None) -> None:
            if not entry:
                return
            key = entry["title"].lower().strip()
            if key and key not in seen_titles:
                seen_titles.add(key)
                items.append(entry)

        # 1. Yahoo Finance ticker news
        if yf is not None:
            target = (resolved_symbol or sym).upper().strip()
            try:
                raw_news = yf.Ticker(target).news or []
                for raw in raw_news:
                    _append(_parse_yf_news_item(raw))
                    if len(items) >= max_items:
                        break
            except Exception as exc:
                logger.debug("Yahoo news fetch failed for %s: %s", target, exc)

        # 2. Google News RSS enrichment / fallback
        if len(items) < max_items and feedparser is not None:
            query = f"{sym} {company_name} stock" if company_name else f"{sym} stock"
            rss_url = (
                f"https://news.google.com/rss/search?q={quote_plus(query)}"
                "&hl=en-US&gl=US&ceid=US:en"
            )
            try:
                feed = feedparser.parse(rss_url)
                for ent in getattr(feed, "entries", []) or []:
                    title = str(getattr(ent, "title", "") or "").strip()
                    link = str(getattr(ent, "link", "") or "").strip()
                    pub = str(getattr(ent, "published", "") or "")[:16]
                    src = getattr(ent, "source", None)
                    publisher = str(getattr(src, "title", "") or "Google News") if src else "Google News"
                    _append(
                        {
                            "title": title,
                            "publisher": publisher,
                            "link": link,
                            "published": pub,
                        }
                    )
                    if len(items) >= max_items:
                        break
            except Exception as exc:
                logger.debug("Google News RSS failed for %s: %s", sym, exc)

        if self.db is not None:
            self.db.set_cache(cache_key, items)
        return items[:max_items]

    def fetch_batch(
        self,
        universe_cfg: dict[str, Any],
        symbols: list[str],
        resolved_map: dict[str, str] | None = None,
        max_items: int = 8,
        force_refresh: bool = False,
        max_workers: int = 6,
    ) -> dict[str, list[dict[str, Any]]]:
        u_meta = universe_cfg.get("universe") or {}
        res_map = resolved_map or {}
        out: dict[str, list[dict[str, Any]]] = {}

        def _worker(s: str) -> tuple[str, list[dict[str, Any]]]:
            raw_name = str((u_meta.get(s) or {}).get("name", "")).split("(")[0].strip()
            news = self.fetch_one(
                s,
                resolved_symbol=res_map.get(s),
                max_items=max_items,
                force_refresh=force_refresh,
                company_name=raw_name or None,
            )
            return s, news

        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(symbols)))) as pool:
            futs = [pool.submit(_worker, s) for s in symbols]
            for fut in as_completed(futs):
                try:
                    s, news = fut.result()
                    out[s] = news
                except Exception as exc:
                    logger.debug("Batch news error: %s", exc)

        for s in symbols:
            out.setdefault(s, [])
        return out
