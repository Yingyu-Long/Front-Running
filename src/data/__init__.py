"""Market data, fundamentals, SEC EDGAR filings, and financial news ingestion layer."""

from src.data.fundamentals import FundamentalFetcher
from src.data.market_data import MarketDataFetcher, build_aligned_close_matrix
from src.data.news_feed import NewsAggregator
from src.data.sec_edgar import SecEdgarClient

__all__ = [
    "MarketDataFetcher",
    "build_aligned_close_matrix",
    "FundamentalFetcher",
    "SecEdgarClient",
    "NewsAggregator",
]
