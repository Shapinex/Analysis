"""
NLP Engine – Groq Edition
==========================
Nutzt Groq's LPU-Infrastruktur für ultraschnelle Sentiment-Analyse.
Modell: llama-3.3-70b-versatile (kostenlos, 30 req/min)

Vorteile vs. FinBERT API:
- 10-50x schneller (Millisekunden statt Sekunden)
- Versteht Kontext besser (70B Parameter vs. 110M)
- Kann Batch-Analyse: Mehrere Artikel in einem Call
- Event-Klassifikation direkt im selben Call
"""
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

from data.sources import Article
from config import NLP_MODEL, NLP_MAX_LENGTH, CONFIDENCE_THRESHOLD, MIN_ARTICLES_FOR_SIGNAL, INDEX

logger = logging.getLogger(__name__)

# ─── Event-Patterns (Fallback falls LLM keine Events liefert) ───────────────
EVENT_PATTERNS = {
    "Earnings": [
        r"earnings", r"revenue", r"profit", r"loss", r"quarterly results",
        r"financial results", r"beat estimates", r"missed expect",
        r"\bEPS\b", r"EBITDA", r"guidance", r"outlook",
    ],
    "M&A": [
        r"acqui", r"merger", r"takeover", r"buyout", r"bid for",
        r"deal to buy", r"joint venture",
    ],
    "Macro": [
        r"\bECB\b", r"interest rate", r"inflation", r"\bGDP\b", r"recession",
        r"central bank", r"monetary policy", r"tariff", r"trade war", r"sanctions",
    ],
    "Analyst": [
        r"upgrade", r"downgrade", r"price target", r"overweight",
        r"underweight", r"buy rating", r"sell rating", r"outperform",
    ],
    "Legal": [
        r"lawsuit", r"fine[sd]?\b", r"regulator", r"antitrust",
        r"investigation", r"penalty", r"EU probe",
    ],
}

_compiled = {
    cat: [re.compile(p, re.IGNORECASE) for p in pats]
    for cat, pats in EVENT_PATTERNS.items()
}

GROQ_MODEL = "llama-3.3-70b-versatile"

# System-Prompt für konsistente Sentiment-Analyse
SYSTEM_PROMPT = """You are a financial sentiment analysis engine. You analyze news headlines and summaries about stocks.

For each article, respond with ONLY valid JSON (no markdown, no explanation):
{
  "label": "positive" | "negative" | "neutral",
  "confidence": 0.0 to 1.0,
  "events": []
}

Rules:
- "positive": good news for the stock (earnings beat, upgrades, growth, deals)
- "negative": bad news (misses, downgrades, lawsuits, losses, layoffs)
- "neutral": mixed or irrelevant news
- "confidence": how certain you are (0.5 = uncertain, 0.95 = very certain)
- "events": classify into zero or more of: ["Earnings", "M&A", "Macro", "Analyst", "Legal"]
- Be precise. A headline like "Stock drops 5%" is clearly negative with high confidence.
- A headline like "Company announces restructuring" could be neutral or negative depending on context."""

BATCH_SYSTEM_PROMPT = """You are a financial sentiment analysis engine. You analyze batches of news articles about stocks.

You will receive multiple articles. For EACH article, provide sentiment analysis.
Respond with ONLY a valid JSON array (no markdown, no explanation):
[
  {"id": 0, "label": "positive"|"negative"|"neutral", "confidence": 0.0-1.0, "events": []},
  {"id": 1, "label": "positive"|"negative"|"neutral", "confidence": 0.0-1.0, "events": []},
  ...
]

Rules:
- "positive": good news for the stock (earnings beat, upgrades, growth, deals)
- "negative": bad news (misses, downgrades, lawsuits, losses, layoffs)  
- "neutral": mixed or irrelevant news
- "confidence": how certain (0.5=uncertain, 0.95=very certain)
- "events": classify into: ["Earnings", "M&A", "Macro", "Analyst", "Legal"]
- Return one object per article, matching the id numbers exactly."""


@dataclass
class SentimentResult:
    article: Article
    label: str
    score: float
    confidence: float
    events: List[str]


@dataclass
class TickerSentiment:
    ticker: str
    score: float
    confidence: float
    article_count: int
    is_reliable: bool
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    dominant_events: List[str] = field(default_factory=list)
    results: List[SentimentResult] = field(default_factory=list)
    sector: str = ""
    weight: float = 0.0


class SentimentEngine:
    """
    Groq-basierte Sentiment Engine.
    Analysiert Artikel in Batches (bis zu 10 pro Call) für maximale Geschwindigkeit.
    """
    BATCH_SIZE = 8  # Artikel pro Groq-Call

    def __init__(self, mode: str = "api"):
        from groq import Groq

        # Token aus Streamlit Secrets oder Umgebungsvariable
        self.api_key = os.getenv("GROQ_API_KEY", "")
        if not self.api_key:
            try:
                import streamlit as st
                self.api_key = st.secrets.get("GROQ_API_KEY", "")
            except Exception:
                pass

        if not self.api_key:
            logger.error("Kein GROQ_API_KEY gesetzt!")
            raise ValueError("GROQ_API_KEY fehlt. Bitte in Streamlit Secrets eintragen.")

        self.client = Groq(api_key=self.api_key)
        self.mode = "groq"
        logger.info(f"Groq Engine initialisiert (Modell: {GROQ_MODEL})")

    def _classify_events_regex(self, text: str) -> List[str]:
        """Regex-Fallback für Event-Klassifikation."""
        found = []
        for cat, patterns in _compiled.items():
            if any(p.search(text) for p in patterns):
                found.append(cat)
        return found

    def _call_groq(self, messages: list, retries: int = 3) -> Optional[str]:
        """Sendet Request an Groq mit Retry-Logik."""
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    temperature=0.1,  # Niedrig für konsistente Ergebnisse
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content
            except Exception as e:
                error_str = str(e)
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    wait = 5 * (attempt + 1)
                    logger.info(f"Rate limit, warte {wait}s...")
                    time.sleep(wait)
                    continue
                logger.warning(f"Groq Fehler (Versuch {attempt+1}): {e}")
                time.sleep(2)
        return None

    def _analyze_single(self, article: Article) -> SentimentResult:
        """Analysiert einen einzelnen Artikel (Fallback)."""
        text_parts = [f"Headline: {article.title}"]
        if article.summary and article.summary != article.title:
            text_parts.append(f"Summary: {article.summary[:300]}")

        user_msg = "\n".join(text_parts)

        raw = self._call_groq([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])

        if raw:
            try:
                data = json.loads(raw)
                label = data.get("label", "neutral").lower()
                conf = float(data.get("confidence", 0.5))
                events = data.get("events", [])

                if conf < CONFIDENCE_THRESHOLD:
                    label = "neutral"

                if label == "positive":
                    score = conf
                elif label == "negative":
                    score = -conf
                else:
                    score = 0.0

                return SentimentResult(
                    article=article, label=label,
                    score=score, confidence=conf, events=events,
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"JSON Parse Fehler: {e}")

        # Fallback: Regex-Events, neutral
        events = self._classify_events_regex(f"{article.title} {article.summary}")
        return SentimentResult(
            article=article, label="neutral",
            score=0.0, confidence=0.0, events=events,
        )

    def _analyze_batch(self, articles: List[Article]) -> List[SentimentResult]:
        """Analysiert mehrere Artikel in einem einzigen Groq-Call."""
        if not articles:
            return []

        # Batch-Prompt bauen
        article_texts = []
        for i, a in enumerate(articles):
            parts = [f"[{i}] Headline: {a.title}"]
            if a.summary and a.summary != a.title:
                parts.append(f"    Summary: {a.summary[:200]}")
            article_texts.append("\n".join(parts))

        user_msg = f"Analyze these {len(articles)} financial news articles:\n\n" + "\n\n".join(article_texts)

        raw = self._call_groq([
            {"role": "system", "content": BATCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])

        results = []

        if raw:
            try:
                parsed = json.loads(raw)
                # Manchmal kommt {"results": [...]} statt direkt [...]
                if isinstance(parsed, dict):
                    parsed = parsed.get("results", parsed.get("articles", []))

                if isinstance(parsed, list):
                    for item in parsed:
                        idx = int(item.get("id", -1))
                        if 0 <= idx < len(articles):
                            label = item.get("label", "neutral").lower()
                            conf = float(item.get("confidence", 0.5))
                            events = item.get("events", [])

                            if conf < CONFIDENCE_THRESHOLD:
                                label = "neutral"

                            if label == "positive":
                                score = conf
                            elif label == "negative":
                                score = -conf
                            else:
                                score = 0.0

                            results.append(SentimentResult(
                                article=articles[idx], label=label,
                                score=score, confidence=conf, events=events,
                            ))
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Batch JSON Parse Fehler: {e}")

        # Falls Batch fehlschlägt oder unvollständig: Fehlende einzeln analysieren
        analyzed_indices = {r.article.title for r in results}
        for a in articles:
            if a.title not in analyzed_indices:
                results.append(self._analyze_single(a))

        return results

    def analyze_ticker(self, ticker: str, articles: List[Article]) -> TickerSentiment:
        """Analysiert alle Artikel eines Tickers via Batch-Calls."""
        all_results = []

        # In Batches aufteilen
        for i in range(0, len(articles), self.BATCH_SIZE):
            batch = articles[i:i + self.BATCH_SIZE]
            batch_results = self._analyze_batch(batch)
            all_results.extend(batch_results)

        if not all_results:
            return TickerSentiment(
                ticker=ticker, score=0.0, confidence=0.0,
                article_count=0, is_reliable=False,
                sector=INDEX.get_sector(ticker),
                weight=INDEX.get_weight(ticker),
            )

        # Konfidenz-gewichteter Durchschnitt
        total_conf = sum(r.confidence for r in all_results)
        if total_conf > 0:
            avg_score = sum(r.score * r.confidence for r in all_results) / total_conf
        else:
            avg_score = sum(r.score for r in all_results) / len(all_results)

        avg_conf = total_conf / len(all_results)
        pos = sum(1 for r in all_results if r.label == "positive")
        neg = sum(1 for r in all_results if r.label == "negative")
        neu = sum(1 for r in all_results if r.label == "neutral")

        # Top Events
        ev_counts: Dict[str, int] = {}
        for r in all_results:
            for e in r.events:
                ev_counts[e] = ev_counts.get(e, 0) + 1
        dominant = sorted(ev_counts, key=ev_counts.get, reverse=True)[:3]

        return TickerSentiment(
            ticker=ticker, score=avg_score, confidence=avg_conf,
            article_count=len(all_results),
            is_reliable=len(all_results) >= MIN_ARTICLES_FOR_SIGNAL,
            positive_count=pos, negative_count=neg, neutral_count=neu,
            dominant_events=dominant, results=all_results,
            sector=INDEX.get_sector(ticker),
            weight=INDEX.get_weight(ticker),
        )
