"""
Datenquellen-Layer
==================
Abstraktion über News-APIs. Einheitliches Article-Format.
Yahoo Finance als Basis (kein API-Key nötig).
Erweiterbar um NewsAPI, Finnhub, Benzinga etc.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import yfinance as yf

logger = logging.getLogger(__name__)


# ─── Einheitliches Artikel-Format ───────────────────────────────────────────
@dataclass
class Article:
    title: str
    summary: str
    source: str
    published: datetime
    url: str = ""
    ticker: str = ""
    relevance_score: float = 1.0
    metadata: Dict = field(default_factory=dict)


# ─── Abstrakte Basis ────────────────────────────────────────────────────────
class NewsSource(ABC):
    @abstractmethod
    def fetch(self, ticker: str, days_back: int = 5) -> List[Article]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ─── Yahoo Finance (immer verfügbar, kein Key) ─────────────────────────────
class YahooFinanceSource(NewsSource):
    @property
    def name(self) -> str:
        return "Yahoo Finance"

    def fetch(self, ticker: str, days_back: int = 5) -> List[Article]:
        cutoff = datetime.now() - timedelta(days=days_back)
        articles = []

        try:
            stock = yf.Ticker(ticker)
            raw_news = stock.news or []
        except Exception as e:
            logger.warning(f"[Yahoo] {ticker}: {e}")
            return articles

        for item in raw_news:
            content = item.get("content", {})
            title = content.get("title", "")
            pub_str = content.get("pubDate", "")
            summary = content.get("summary", "")
            provider = content.get("provider", {}).get("displayName", "Yahoo")
            url = content.get("canonicalUrl", {}).get("url", "")

            if not title or not pub_str:
                continue
            try:
                pub_date = datetime.strptime(pub_str[:10], "%Y-%m-%d")
            except ValueError:
                continue
            if pub_date < cutoff:
                continue

            articles.append(Article(
                title=title,
                summary=summary or title,
                source=provider,
                published=pub_date,
                url=url,
                ticker=ticker,
            ))

        return articles


# ─── Aggregator ─────────────────────────────────────────────────────────────
class NewsAggregator:
    """Sammelt aus allen aktiven Quellen, dedupliziert nach Titel."""

    def __init__(self, sources: List[NewsSource]):
        self.sources = sources
        logger.info(f"Aggregator: {[s.name for s in sources]}")

    def fetch_all(self, ticker: str, days_back: int = 5) -> List[Article]:
        all_articles = []
        for source in self.sources:
            try:
                all_articles.extend(source.fetch(ticker, days_back))
            except Exception as e:
                logger.error(f"{source.name} ausgefallen: {e}")

        # Deduplizierung
        seen = set()
        unique = []
        for a in all_articles:
            key = a.title.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(a)

        unique.sort(key=lambda a: a.published, reverse=True)
        return unique

    @property
    def source_names(self) -> List[str]:
        return [s.name for s in self.sources]
