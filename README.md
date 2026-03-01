# 📊 EUSTX50 Sentiment Engine

Gewichtete Multi-Source NLP-Analyse des **Euro Stoxx 50** mit FinBERT.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![License](https://img.shields.io/badge/License-Private-gray)

---

## Was ist das?

Eine Echtzeit-Sentiment-Analyse aller 50 Aktien des Euro Stoxx 50. Das Tool:

- **Sammelt Nachrichten** von Yahoo Finance (erweiterbar um NewsAPI, Finnhub)
- **Analysiert Sentiment** mit FinBERT (Titel + Zusammenfassung, konfidenz-gewichtet)
- **Klassifiziert Events** (Earnings, M&A, Macro, Analyst-Ratings, Legal)
- **Berechnet einen gewichteten Index-Score** basierend auf Market-Cap-Gewichten
- **Erkennt Divergenzen** zwischen Kursverläufen und Stimmungslage
- **Speichert historische Daten** (SQLite) für Trend-Analyse

## Architektur

```
┌─────────────────────────────────────────────┐
│  Streamlit Dashboard (app.py)               │
│  5 Tabs: Übersicht│Einzelwerte│News│         │
│          Divergenz│Historie                  │
├──────────────┬──────────────┬───────────────┤
│ data/        │ core/        │ core/         │
│ sources.py   │ nlp_engine.py│ storage.py    │
│ Yahoo Finance│ FinBERT      │ SQLite DB     │
│ (NewsAPI)    │ Lokal / API  │ Zeitreihen    │
│ (Finnhub)    │ Event-Klassif│ Trend-Daten   │
├──────────────┴──────────────┴───────────────┤
│ config.py                                    │
│ Ticker, Gewichte, Sektoren, Entity-Map       │
└─────────────────────────────────────────────┘
```

## Verbesserungen vs. Prototyp

| Feature | Prototyp | Engine v2 |
|---------|----------|-----------|
| Analyse-Text | Nur Titel | Titel + Summary (60/40 gewichtet) |
| Konfidenz | Keine | Schwellen-Gate (< 60% → neutral) |
| Min. Artikel | Keine | Signal-Schwelle (< 3 → unzuverlässig) |
| Events | Keine | 5 Kategorien (Earnings, M&A, Macro...) |
| Gewichtung | Gleich | Market-Cap-basiert |
| Sektoren | Keine | 10 Sektoren mit Heatmap |
| Divergenz | Keine | Preis vs. Sentiment Radar |
| Persistenz | Keine | SQLite mit Zeitreihen |
| Datenquellen | Nur Yahoo | Multi-Source (erweiterbar) |
| Deployment | Lokal | Streamlit Cloud ready |

---

## Schnellstart

### Option A: Streamlit Cloud (empfohlen)

1. **GitHub Repo erstellen** (privat):
   ```bash
   git init
   git add .
   git commit -m "initial commit"
   git remote add origin https://github.com/DEIN-USER/eustx50-sentiment.git
   git push -u origin main
   ```

2. **Streamlit Cloud** → [share.streamlit.io](https://share.streamlit.io):
   - "New app" → Dein privates Repo auswählen
   - Main file: `app.py`
   - Deploy!

3. **Privat machen**: In Streamlit Cloud App Settings → Sharing → "This app is private"

4. **(Optional) HuggingFace Token**: In Streamlit Cloud → App Settings → Secrets:
   ```toml
   HF_TOKEN = "hf_dein_token_hier"
   ```

### Option B: Lokal

```bash
# Repo klonen
git clone https://github.com/DEIN-USER/eustx50-sentiment.git
cd eustx50-sentiment

# Abhängigkeiten (mit lokalem FinBERT)
pip install -r requirements.txt
pip install transformers torch

# Starten
streamlit run app.py
```

---

## NLP-Modi

| Modus | Vorteile | Nachteile |
|-------|----------|-----------|
| **☁️ API** (Standard) | Kein PyTorch nötig, ~100MB RAM | Langsamer, Rate-Limits |
| **💻 Lokal** | Schneller, offline-fähig | ~1GB RAM, PyTorch nötig |

Der API-Modus nutzt die kostenlose HuggingFace Inference API.
Für höhere Rate-Limits: Erstelle einen Token auf [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

---

## Konfiguration

Alle Einstellungen in `config.py`:

- **`CONFIDENCE_THRESHOLD`**: Unter diesem Wert wird ein Ergebnis als neutral behandelt (Standard: 0.60)
- **`MIN_ARTICLES_FOR_SIGNAL`**: Weniger Artikel → Score als "unzuverlässig" markiert (Standard: 3)
- **`NEWS_LOOKBACK_DAYS`**: Zeitfenster für Nachrichtensuche (Standard: 5)

Gewichte, Sektoren und Entity-Mapping sind in der `IndexConfig` Klasse.

---

## Erweiterung um zusätzliche Quellen

In `data/sources.py` eine neue Klasse erstellen:

```python
class MeineQuelleSource(NewsSource):
    @property
    def name(self) -> str:
        return "Meine Quelle"

    def fetch(self, ticker: str, days_back: int = 5) -> List[Article]:
        # Daten holen und als Article-Objekte zurückgeben
        ...
```

Dann in `app.py` dem Aggregator hinzufügen:
```python
sources = [YahooFinanceSource(), MeineQuelleSource()]
```

---

## Disclaimer

Dieses Tool dient ausschließlich zu Informations- und Bildungszwecken.
Es stellt keine Anlageberatung dar. Investitionsentscheidungen sollten
immer auf Basis eigener Recherche und professioneller Beratung getroffen werden.
