"""
NLP Engine
==========
FinBERT-basierte Sentiment-Analyse mit:
- Titel + Summary Analyse (gewichtet 40/60)
- Konfidenz-Gate (unsichere Ergebnisse → neutral)
- Event-Klassifikation (Earnings, M&A, Macro, Analyst, etc.)
- Dual-Mode: HuggingFace InferenceClient (Cloud) oder lokal
"""
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

from data.sources import Article
from config import NLP_MODEL, NLP_MAX_LENGTH, CONFIDENCE_THRESHOLD, MIN_ARTICLES_FOR_SIGNAL, INDEX

logger = logging.getLogger(__name__)

# ─── Event-Patterns ─────────────────────────────────────────────────────────
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


@dataclass
class SentimentResult:
    article: Article
    label: str              # positive / negative / neutral
    score: float            # -1.0 bis +1.0
    confidence: float       # 0.0 bis 1.0
    events: List[str]       # ["Earnings", "Analyst"]


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
    Dual-Mode:
    - mode="api": Nutzt HuggingFace InferenceClient (empfohlen für Cloud)
    - mode="local": Lädt FinBERT lokal (braucht ~1GB RAM + PyTorch)
    """
    TITLE_W = 0.4
    SUMMARY_W = 0.6

    def __init__(self, mode: str = "api"):
        self.mode = mode
        self.pipe = None
        self.client = None

        # Token aus Streamlit Secrets oder Umgebungsvariable
        self.hf_token = os.getenv("HF_TOKEN", "")

        # Versuche auch Streamlit Secrets zu lesen
        if not self.hf_token:
            try:
                import streamlit as st
                self.hf_token = st.secrets.get("HF_TOKEN", "")
            except Exception:
                pass

        if mode == "api":
            self._init_api()
        else:
            self._init_local()

    def _init_api(self):
        """Initialisiert den HuggingFace InferenceClient."""
        from huggingface_hub import InferenceClient

        if not self.hf_token:
            logger.warning("Kein HF_TOKEN gesetzt! Bitte in Streamlit Secrets eintragen.")

        self.client = InferenceClient(
            model=NLP_MODEL,
            token=self.hf_token if self.hf_token else None,
        )
        self.mode = "api"
        logger.info(f"HuggingFace InferenceClient initialisiert ({'mit' if self.hf_token else 'ohne'} Token)")

    def _init_local(self):
        """Lädt FinBERT lokal mit transformers."""
        from transformers import pipeline as hf_pipeline
        logger.info(f"Lade {NLP_MODEL} lokal...")
        self.pipe = hf_pipeline(
            "sentiment-analysis", model=NLP_MODEL,
            truncation=True, max_length=NLP_MAX_LENGTH,
        )
        self.mode = "local"
        logger.info("Lokales Modell geladen.")

    def _query_api(self, text: str) -> List[dict]:
        """Sendet Text an HuggingFace Inference API via InferenceClient."""
        for attempt in range(3):
            try:
                result = self.client.text_classification(text[:NLP_MAX_LENGTH])
                # result ist eine Liste von ClassificationOutput Objekten
                return [{"label": r.label, "score": r.score} for r in result]
            except Exception as e:
                error_str = str(e)
                if "loading" in error_str.lower() or "503" in error_str:
                    logger.info(f"API: Modell wird geladen, warte 20s... (Versuch {attempt+1})")
                    time.sleep(20)
                    continue
                logger.warning(f"API Fehler (Versuch {attempt+1}): {e}")
                time.sleep(3)

        return [{"label": "neutral", "score": 0.5}]

    def _classify_events(self, text: str) -> List[str]:
        found = []
        for cat, patterns in _compiled.items():
            if any(p.search(text) for p in patterns):
                found.append(cat)
        return found

    def _analyze_text(self, text: str) -> Tuple[str, float, float]:
        """Returns (label, directed_score, confidence)."""
        if not text or len(text.strip()) < 10:
            return "neutral", 0.0, 0.0

        if self.mode == "local":
            res = self.pipe(text[:NLP_MAX_LENGTH])[0]
            label = res["label"]
            conf = res["score"]
        else:
            results = self._query_api(text)
            best = max(results, key=lambda x: x["score"])
            label = best["label"]
            conf = best["score"]

        if conf < CONFIDENCE_THRESHOLD:
            return "neutral", 0.0, conf

        if label == "positive":
            return label, conf, conf
        elif label == "negative":
            return label, -conf, conf
        return "neutral", 0.0, conf

    def analyze_article(self, article: Article) -> SentimentResult:
        t_label, t_score, t_conf = self._analyze_text(article.title)

        if article.summary and article.summary != article.title:
            s_label, s_score, s_conf = self._analyze_text(article.summary)
            combined_score = t_score * self.TITLE_W + s_score * self.SUMMARY_W
            combined_conf = t_conf * self.TITLE_W + s_conf * self.SUMMARY_W
        else:
            combined_score = t_score
            combined_conf = t_conf

        if combined_score > 0.1:
            final_label = "positive"
        elif combined_score < -0.1:
            final_label = "negative"
        else:
            final_label = "neutral"

        events = self._classify_events(f"{article.title} {article.summary}")

        return SentimentResult(
            article=article,
            label=final_label,
            score=combined_score,
            confidence=combined_conf,
            events=events,
        )

    def analyze_ticker(self, ticker: str, articles: List[Article]) -> TickerSentiment:
        results = [self.analyze_article(a) for a in articles]

        if not results:
            return TickerSentiment(
                ticker=ticker, score=0.0, confidence=0.0,
                article_count=0, is_reliable=False,
                sector=INDEX.get_sector(ticker),
                weight=INDEX.get_weight(ticker),
            )

        total_conf = sum(r.confidence for r in results)
        if total_conf > 0:
            avg_score = sum(r.score * r.confidence for r in results) / total_conf
        else:
            avg_score = sum(r.score for r in results) / len(results)

        avg_conf = total_conf / len(results)
        pos = sum(1 for r in results if r.label == "positive")
        neg = sum(1 for r in results if r.label == "negative")
        neu = sum(1 for r in results if r.label == "neutral")

        ev_counts: Dict[str, int] = {}
        for r in results:
            for e in r.events:
                ev_counts[e] = ev_counts.get(e, 0) + 1
        dominant = sorted(ev_counts, key=ev_counts.get, reverse=True)[:3]

        return TickerSentiment(
            ticker=ticker, score=avg_score, confidence=avg_conf,
            article_count=len(results),
            is_reliable=len(results) >= MIN_ARTICLES_FOR_SIGNAL,
            positive_count=pos, negative_count=neg, neutral_count=neu,
            dominant_events=dominant, results=results,
            sector=INDEX.get_sector(ticker),
            weight=INDEX.get_weight(ticker),
        )
