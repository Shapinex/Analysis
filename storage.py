"""
Persistenz-Layer (SQLite)
=========================
Speichert Scan-Ergebnisse für Trend-Analyse über Zeit.
"""
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from config import DB_PATH

logger = logging.getLogger(__name__)


class SentimentStore:

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_time TEXT NOT NULL,
                    index_score REAL NOT NULL,
                    total_articles INTEGER NOT NULL,
                    sources TEXT
                );
                CREATE TABLE IF NOT EXISTS ticker_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    article_count INTEGER NOT NULL,
                    is_reliable INTEGER NOT NULL,
                    positive_count INTEGER DEFAULT 0,
                    negative_count INTEGER DEFAULT 0,
                    neutral_count INTEGER DEFAULT 0,
                    dominant_events TEXT,
                    sector TEXT,
                    FOREIGN KEY (scan_id) REFERENCES scans(id)
                );
                CREATE TABLE IF NOT EXISTS articles_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id INTEGER NOT NULL,
                    ticker TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source TEXT,
                    published TEXT,
                    label TEXT,
                    score REAL,
                    confidence REAL,
                    events TEXT,
                    url TEXT,
                    FOREIGN KEY (scan_id) REFERENCES scans(id)
                );
                CREATE INDEX IF NOT EXISTS idx_scans_time ON scans(scan_time);
                CREATE INDEX IF NOT EXISTS idx_ticker_scores ON ticker_scores(scan_id, ticker);
            """)

    def save_scan(self, index_score, total_articles, ticker_sentiments, sources) -> int:
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO scans (scan_time, index_score, total_articles, sources) VALUES (?,?,?,?)",
                (now, index_score, total_articles, json.dumps(sources)),
            )
            scan_id = cur.lastrowid

            for ts in ticker_sentiments:
                conn.execute(
                    """INSERT INTO ticker_scores
                       (scan_id,ticker,score,confidence,article_count,is_reliable,
                        positive_count,negative_count,neutral_count,dominant_events,sector)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (scan_id, ts.ticker, ts.score, ts.confidence, ts.article_count,
                     int(ts.is_reliable), ts.positive_count, ts.negative_count,
                     ts.neutral_count, json.dumps(ts.dominant_events), ts.sector),
                )
                for r in ts.results:
                    conn.execute(
                        """INSERT INTO articles_log
                           (scan_id,ticker,title,source,published,label,score,confidence,events,url)
                           VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (scan_id, ts.ticker, r.article.title, r.article.source,
                         r.article.published.isoformat(), r.label, r.score,
                         r.confidence, json.dumps(r.events), r.article.url),
                    )
        logger.info(f"Scan #{scan_id} gespeichert.")
        return scan_id

    def get_index_history(self, days: int = 90) -> List[Dict]:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT scan_time, index_score, total_articles FROM scans WHERE scan_time>? ORDER BY scan_time",
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_ticker_history(self, ticker: str, days: int = 90) -> List[Dict]:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT s.scan_time, t.score, t.confidence, t.article_count
                   FROM ticker_scores t JOIN scans s ON t.scan_id=s.id
                   WHERE t.ticker=? AND s.scan_time>? ORDER BY s.scan_time""",
                (ticker, cutoff),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_scan_count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
