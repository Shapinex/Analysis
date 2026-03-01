"""
EUSTX50 Sentiment Engine – Zentrale Konfiguration
===================================================
Alle Einstellungen an einem Ort. Keine Magic Numbers im Code.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List

# ─── NLP ────────────────────────────────────────────────────────────────────
NLP_MODEL = "ProsusAI/finbert"
NLP_MAX_LENGTH = 512
CONFIDENCE_THRESHOLD = 0.60     # Unter 60% Konfidenz → wird als neutral behandelt
MIN_ARTICLES_FOR_SIGNAL = 3     # Weniger → "unzuverlässig" markiert

# ─── Zeitfenster ────────────────────────────────────────────────────────────
NEWS_LOOKBACK_DAYS = 5
HISTORY_CHART_PERIOD = "3mo"

# ─── Datenbank ──────────────────────────────────────────────────────────────
DB_PATH = "sentiment_history.db"


@dataclass
class IndexConfig:
    name: str = "EURO STOXX 50"
    default_weight: float = 0.013

    # Top-10 explizite Gewichte (approx. free-float market cap)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "ASML.AS": 0.095, "SAP.DE": 0.082, "MC.PA": 0.058,
        "TTE.PA": 0.048, "SIE.DE": 0.041, "OR.PA": 0.035,
        "SAN.PA": 0.032, "ALV.DE": 0.028, "SU.PA": 0.026,
        "IBE.MC": 0.023,
    })

    tickers: List[str] = field(default_factory=lambda: [
        "SAP.DE", "ASML.AS", "MC.PA", "OR.PA", "TTE.PA",
        "SIE.DE", "SAN.PA", "SU.PA", "ALV.DE", "IBE.MC",
        "AI.PA", "RMS.PA", "DTE.DE", "BNP.PA", "ITX.MC",
        "CS.PA", "MBG.DE", "SAF.PA", "EL.PA", "SAN.MC",
        "ABI.BR", "INGA.AS", "ENEL.MI", "MUV2.DE", "BAS.DE",
        "DG.PA", "ISP.MI", "BMW.DE", "BBVA.MC", "RI.PA",
        "ENI.MI", "VOW3.DE", "UCG.MI", "ADYEN.AS", "DHL.DE",
        "STMPA.PA", "IFX.DE", "HEIA.AS", "KONE.HE", "SGO.PA",
        "NOKIA.HE", "VONN.DE", "DB1.DE", "RWE.DE", "MT.AS",
        "ACA.PA", "ORA.PA", "GLE.PA", "ENGIE.PA", "PRX.AS",
    ])

    # Entity-Mapping: Firmenname (lowercase) → Ticker
    # Ermöglicht Zuordnung von News, die den Firmennamen statt Ticker nennen
    entity_map: Dict[str, str] = field(default_factory=lambda: {
        "asml": "ASML.AS", "sap": "SAP.DE", "lvmh": "MC.PA",
        "moët hennessy": "MC.PA", "louis vuitton": "MC.PA",
        "totalenergies": "TTE.PA", "total energies": "TTE.PA",
        "siemens": "SIE.DE", "l'oréal": "OR.PA", "loreal": "OR.PA",
        "sanofi": "SAN.PA", "allianz": "ALV.DE", "schneider electric": "SU.PA",
        "iberdrola": "IBE.MC", "air liquide": "AI.PA", "hermès": "RMS.PA",
        "hermes": "RMS.PA", "deutsche telekom": "DTE.DE", "bnp paribas": "BNP.PA",
        "inditex": "ITX.MC", "zara": "ITX.MC", "axa": "CS.PA",
        "mercedes-benz": "MBG.DE", "mercedes": "MBG.DE",
        "safran": "SAF.PA", "essilorluxottica": "EL.PA",
        "banco santander": "SAN.MC", "santander": "SAN.MC",
        "ab inbev": "ABI.BR", "anheuser-busch": "ABI.BR",
        "ing": "INGA.AS", "enel": "ENEL.MI",
        "munich re": "MUV2.DE", "münchener rück": "MUV2.DE",
        "basf": "BAS.DE", "vinci": "DG.PA",
        "intesa sanpaolo": "ISP.MI", "bmw": "BMW.DE",
        "bbva": "BBVA.MC", "pernod ricard": "RI.PA",
        "eni": "ENI.MI", "volkswagen": "VOW3.DE", "vw": "VOW3.DE",
        "unicredit": "UCG.MI", "adyen": "ADYEN.AS",
        "dhl": "DHL.DE", "deutsche post": "DHL.DE",
        "stmicroelectronics": "STMPA.PA", "infineon": "IFX.DE",
        "heineken": "HEIA.AS", "kone": "KONE.HE",
        "saint-gobain": "SGO.PA", "nokia": "NOKIA.HE",
        "vonovia": "VONN.DE", "deutsche börse": "DB1.DE",
        "rwe": "RWE.DE", "arcelormittal": "MT.AS",
        "crédit agricole": "ACA.PA", "credit agricole": "ACA.PA",
        "orange": "ORA.PA", "société générale": "GLE.PA",
        "societe generale": "GLE.PA", "engie": "ENGIE.PA",
        "prosus": "PRX.AS",
    })

    # Sektor-Zuordnung
    sectors: Dict[str, str] = field(default_factory=lambda: {
        "SAP.DE": "Technology", "ASML.AS": "Technology",
        "STMPA.PA": "Technology", "IFX.DE": "Technology",
        "ADYEN.AS": "Technology", "NOKIA.HE": "Technology",
        "MC.PA": "Consumer", "OR.PA": "Consumer", "RMS.PA": "Consumer",
        "ITX.MC": "Consumer", "EL.PA": "Consumer", "ABI.BR": "Consumer",
        "HEIA.AS": "Consumer", "RI.PA": "Consumer", "PRX.AS": "Consumer",
        "TTE.PA": "Energy", "IBE.MC": "Energy", "ENEL.MI": "Energy",
        "ENI.MI": "Energy", "RWE.DE": "Energy", "ENGIE.PA": "Energy",
        "SIE.DE": "Industrials", "SAF.PA": "Industrials",
        "DHL.DE": "Industrials", "KONE.HE": "Industrials",
        "SGO.PA": "Industrials", "MT.AS": "Industrials",
        "DG.PA": "Industrials", "AI.PA": "Industrials",
        "SU.PA": "Industrials",
        "SAN.PA": "Healthcare", "BAS.DE": "Chemicals",
        "ALV.DE": "Financials", "BNP.PA": "Financials",
        "CS.PA": "Financials", "INGA.AS": "Financials",
        "MUV2.DE": "Financials", "ISP.MI": "Financials",
        "BBVA.MC": "Financials", "UCG.MI": "Financials",
        "ACA.PA": "Financials", "GLE.PA": "Financials",
        "DB1.DE": "Financials", "SAN.MC": "Financials",
        "MBG.DE": "Automotive", "BMW.DE": "Automotive",
        "VOW3.DE": "Automotive",
        "DTE.DE": "Telecom", "ORA.PA": "Telecom",
        "VONN.DE": "Real Estate",
    })

    def get_weight(self, ticker: str) -> float:
        return self.weights.get(ticker, self.default_weight)

    def get_sector(self, ticker: str) -> str:
        return self.sectors.get(ticker, "Other")


INDEX = IndexConfig()
