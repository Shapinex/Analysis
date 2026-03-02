"""
EUSTX50 Sentiment Engine – Dashboard
=====================================
5 Tabs: Markt-Übersicht | Einzelwerte | News-Feed | Divergenz-Radar | Historie
Powered by Groq LPU + Llama 3.3 70B
"""
import warnings
warnings.filterwarnings("ignore")

import logging
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

from config import INDEX, NEWS_LOOKBACK_DAYS
from data.sources import YahooFinanceSource, NewsAggregator
from core.nlp_engine import SentimentEngine, TickerSentiment
from core.storage import SentimentStore

logging.basicConfig(level=logging.INFO)

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EUSTX50 Sentiment Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Einstellungen")

    st.subheader("📊 Parameter")
    lookback = st.slider("Tage zurückschauen", 1, 14, NEWS_LOOKBACK_DAYS)
    min_art = st.slider("Min. Artikel für Signal", 1, 10, 3)
    conf_thresh = st.slider("Konfidenz-Schwelle", 0.3, 0.9, 0.6, 0.05)

    st.divider()
    store = SentimentStore()
    scan_count = store.get_scan_count()
    st.caption(f"📦 {scan_count} Scans in der Datenbank")
    st.caption("⚡ Powered by Groq LPU + Llama 3.3 70B")
    st.caption("EUSTX50 Sentiment Engine v2.1")
    st.caption("⚠️ Keine Anlageberatung.")


# ─── Header ─────────────────────────────────────────────────────────────────
st.title("📊 EUSTX50 Sentiment Engine")
st.caption("Gewichtete Multi-Source NLP-Analyse des Euro Stoxx 50 · Groq LPU · Llama 3.3 70B")


# ─── Session State ──────────────────────────────────────────────────────────
if "scan_done" not in st.session_state:
    st.session_state.scan_done = False


# ═══════════════════════════════════════════════════════════════════════════
# SCAN-LOGIK
# ═══════════════════════════════════════════════════════════════════════════
def run_scan():
    sources = [YahooFinanceSource()]
    aggregator = NewsAggregator(sources)

    try:
        engine = SentimentEngine()
    except ValueError as e:
        st.error(f"❌ {e}")
        st.info("Bitte GROQ_API_KEY in den Streamlit Secrets eintragen (Settings → Secrets)")
        st.stop()

    progress = st.progress(0, text="Initialisiere...")
    ticker_sentiments = []
    total_articles = 0
    n = len(INDEX.tickers)

    for i, ticker in enumerate(INDEX.tickers):
        progress.progress((i + 1) / n, text=f"📡 {ticker} ({i+1}/{n})")
        articles = aggregator.fetch_all(ticker, days_back=lookback)
        ts = engine.analyze_ticker(ticker, articles)
        ticker_sentiments.append(ts)
        total_articles += ts.article_count

    # Gewichteter Index-Score
    w_sum, w_total = 0.0, 0.0
    for ts in ticker_sentiments:
        if ts.article_count > 0:
            w = INDEX.get_weight(ts.ticker)
            w_sum += ts.score * w
            w_total += w
    index_score = w_sum / w_total if w_total > 0 else 0.0

    # Speichern
    scan_id = store.save_scan(index_score, total_articles, ticker_sentiments, aggregator.source_names)

    # Session State
    st.session_state.scan_done = True
    st.session_state.index_score = index_score
    st.session_state.total_articles = total_articles
    st.session_state.ticker_sentiments = ticker_sentiments
    st.session_state.scan_time = datetime.now()
    st.session_state.source_names = aggregator.source_names
    st.session_state.scan_id = scan_id

    progress.empty()


# ─── Scan starten ───────────────────────────────────────────────────────────
if not st.session_state.scan_done:
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("🚀 Scan starten", type="primary", use_container_width=True):
            run_scan()
            st.rerun()
    with c2:
        st.info("⚡ Groq-powered Analyse aller 50 EUSTX50-Aktien. Dauer: ca. 1–2 Minuten.")

    # Zeige vergangene Scans als Vorschau
    history = store.get_index_history(days=30)
    if history:
        st.divider()
        st.subheader("📈 Letzte Scans")
        hdf = pd.DataFrame(history)
        hdf["scan_time"] = pd.to_datetime(hdf["scan_time"])
        fig_h = px.line(hdf, x="scan_time", y="index_score",
                        markers=True, labels={"index_score": "Score", "scan_time": "Zeit"})
        fig_h.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_h.update_layout(height=250, margin=dict(t=10, b=10))
        st.plotly_chart(fig_h, use_container_width=True)

    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
st.success(
    f"✅ Scan #{st.session_state.scan_id} · "
    f"{st.session_state.scan_time:%H:%M:%S} · "
    f"{st.session_state.total_articles} Artikel · "
    f"Quellen: {', '.join(st.session_state.source_names)}"
)

# DataFrame aufbauen
sentiments = st.session_state.ticker_sentiments
df = pd.DataFrame([{
    "Ticker": ts.ticker, "Score": ts.score, "Konfidenz": ts.confidence,
    "Artikel": ts.article_count, "Zuverlässig": ts.is_reliable,
    "Pos": ts.positive_count, "Neg": ts.negative_count, "Neutral": ts.neutral_count,
    "Events": ", ".join(ts.dominant_events) if ts.dominant_events else "–",
    "Sektor": ts.sector, "Gewicht": ts.weight,
} for ts in sentiments])
df_active = df[df["Artikel"] > 0].copy()

# Neuer Scan Button
if st.button("🔄 Neuen Scan starten"):
    st.session_state.scan_done = False
    st.rerun()


# ─── TABS ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Markt-Übersicht", "🏆 Einzelwerte", "📰 News-Feed",
    "⚡ Divergenz-Radar", "📈 Historie",
])


# ═══ TAB 1: MARKT-ÜBERSICHT ════════════════════════════════════════════════
with tab1:
    score = st.session_state.index_score

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        emoji = "🟢" if score > 0.05 else ("🔴" if score < -0.05 else "⚪")
        st.metric("Index-Score", f"{score:+.4f}", delta=emoji)
    with k2:
        st.metric("Artikel gesamt", st.session_state.total_articles)
    with k3:
        reliable = df_active["Zuverlässig"].sum()
        st.metric("Zuverlässige Signale", f"{reliable}/{len(df_active)}")
    with k4:
        coverage = len(df_active) / len(INDEX.tickers)
        st.metric("Abdeckung", f"{coverage:.0%}")

    st.divider()
    c_gauge, c_heat = st.columns(2)

    # Tacho
    with c_gauge:
        st.subheader("Stimmungs-Index")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 52}, "valueformat": "+.4f"},
            gauge={
                "axis": {"range": [-0.4, 0.4], "tickwidth": 1},
                "bar": {"color": "#1e293b"},
                "steps": [
                    {"range": [-0.4, -0.15], "color": "#ef4444"},
                    {"range": [-0.15, -0.05], "color": "#fb923c"},
                    {"range": [-0.05, 0.05], "color": "#e5e7eb"},
                    {"range": [0.05, 0.15], "color": "#86efac"},
                    {"range": [0.15, 0.4], "color": "#22c55e"},
                ],
            },
        ))
        fig_g.update_layout(height=280, margin=dict(t=20, b=10))
        st.plotly_chart(fig_g, use_container_width=True)

        if score > 0.15:
            st.success("**BULLISCH** – Breite positive Nachrichtenlage.")
        elif score > 0.05:
            st.info("**LEICHT BULLISCH** – Tendenz positiv.")
        elif score > -0.05:
            st.warning("**NEUTRAL** – Gemischte Signale.")
        elif score > -0.15:
            st.warning("**LEICHT BÄRISCH** – Tendenz negativ.")
        else:
            st.error("**BÄRISCH** – Überwiegend negative Nachrichtenlage.")

    # Sektor-Heatmap
    with c_heat:
        st.subheader("Sektor-Sentiment")
        if not df_active.empty:
            sec = df_active.groupby("Sektor").agg(
                Score=("Score", "mean"), Artikel=("Artikel", "sum"),
                Anzahl=("Ticker", "count"),
            ).reset_index().sort_values("Score", ascending=True)

            fig_s = px.bar(
                sec, y="Sektor", x="Score", orientation="h",
                color="Score", color_continuous_scale="RdYlGn",
                range_color=[-0.4, 0.4], text=sec["Score"].apply(lambda x: f"{x:+.3f}"),
                hover_data=["Artikel", "Anzahl"],
            )
            fig_s.update_layout(height=280, margin=dict(t=20, b=10, l=10),
                                showlegend=False, yaxis_title="")
            fig_s.update_traces(textposition="outside")
            st.plotly_chart(fig_s, use_container_width=True)

    # Sentiment-Verteilung
    st.subheader("Verteilung aller Aktien-Scores")
    if not df_active.empty:
        fig_dist = px.histogram(
            df_active, x="Score", nbins=20, color_discrete_sequence=["#6366f1"],
            labels={"Score": "Sentiment Score", "count": "Anzahl Aktien"},
        )
        fig_dist.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_dist.add_vline(x=score, line_dash="solid", line_color="red",
                           annotation_text=f"Index: {score:+.3f}")
        fig_dist.update_layout(height=250, margin=dict(t=30, b=10))
        st.plotly_chart(fig_dist, use_container_width=True)


# ═══ TAB 2: EINZELWERTE ════════════════════════════════════════════════════
with tab2:
    c_top, c_flop = st.columns(2)

    with c_top:
        st.subheader("🟢 Top 5")
        for _, row in df_active.nlargest(5, "Score").iterrows():
            tag = "✅" if row["Zuverlässig"] else "⚠️"
            ev = f" · {row['Events']}" if row["Events"] != "–" else ""
            st.success(
                f"**{row['Ticker']}** ({row['Sektor']}) · "
                f"Score: **{row['Score']:+.3f}** · {row['Konfidenz']:.0%} Konf. · "
                f"{row['Artikel']} Art. {tag}{ev}"
            )

    with c_flop:
        st.subheader("🔴 Flop 5")
        for _, row in df_active.nsmallest(5, "Score").iterrows():
            tag = "✅" if row["Zuverlässig"] else "⚠️"
            ev = f" · {row['Events']}" if row["Events"] != "–" else ""
            st.error(
                f"**{row['Ticker']}** ({row['Sektor']}) · "
                f"Score: **{row['Score']:+.3f}** · {row['Konfidenz']:.0%} Konf. · "
                f"{row['Artikel']} Art. {tag}{ev}"
            )

    st.divider()

    # Bubble Chart
    st.subheader("Bubble Chart: Score × Buzz × Konfidenz")
    if not df_active.empty:
        fig_b = px.scatter(
            df_active, x="Score", y="Artikel",
            size=df_active["Konfidenz"].clip(lower=0.1),
            color="Sektor", hover_name="Ticker",
            hover_data={"Events": True, "Zuverlässig": True, "Gewicht": ":.3f"},
            size_max=45,
        )
        fig_b.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_b.update_layout(
            height=450, margin=dict(t=10, b=10),
            xaxis_title="Sentiment Score", yaxis_title="Nachrichtenvolumen (Buzz)",
        )
        st.plotly_chart(fig_b, use_container_width=True)

    # Tabelle
    st.subheader("Alle Aktien (sortierbar)")
    st.dataframe(
        df_active.sort_values("Score", ascending=False)
        .style.background_gradient(subset=["Score"], cmap="RdYlGn", vmin=-0.5, vmax=0.5)
        .format({"Score": "{:+.3f}", "Konfidenz": "{:.0%}", "Gewicht": "{:.3f}"}),
        use_container_width=True, height=500,
    )


# ═══ TAB 3: NEWS-FEED ══════════════════════════════════════════════════════
with tab3:
    st.subheader("Einzelartikel mit Sentiment-Bewertung")

    ticker_options = ["Alle"] + df_active["Ticker"].tolist()
    selected = st.selectbox("Ticker filtern:", ticker_options)

    label_filter = st.multiselect(
        "Sentiment filtern:", ["positive", "negative", "neutral"],
        default=["positive", "negative", "neutral"],
    )

    all_results = []
    for ts in sentiments:
        if selected != "Alle" and ts.ticker != selected:
            continue
        for r in ts.results:
            if r.label in label_filter:
                all_results.append(r)

    all_results.sort(key=lambda r: r.article.published, reverse=True)
    st.caption(f"{len(all_results)} Artikel angezeigt")

    for r in all_results[:100]:
        color_map = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
        icon = color_map.get(r.label, "⚪")
        events_str = f" · 🏷️ {', '.join(r.events)}" if r.events else ""

        with st.container():
            st.markdown(
                f"{icon} **[{r.article.ticker}]** {r.article.title}  \n"
                f"Score: `{r.score:+.3f}` · Konf: `{r.confidence:.0%}` · "
                f"📰 {r.article.source} · 📅 {r.article.published:%Y-%m-%d}"
                f"{events_str}"
            )
            if r.article.url:
                st.caption(f"🔗 {r.article.url}")
            st.divider()


# ═══ TAB 4: DIVERGENZ-RADAR ════════════════════════════════════════════════
with tab4:
    st.subheader("Preis vs. Stimmung – Divergenz-Check")
    st.write(
        "Vergleicht den 1-Monats-Kurs-Trend mit dem aktuellen Sentiment. "
        "Divergenzen können Warnsignale oder Chancen sein."
    )

    @st.cache_data(ttl=3600, show_spinner="Lade Kursdaten...")
    def get_price_changes(tickers):
        changes = {}
        for t in tickers:
            try:
                hist = yf.Ticker(t).history(period="1mo")
                if len(hist) >= 2:
                    pct = (hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1) * 100
                    changes[t] = round(pct, 2)
            except Exception:
                pass
        return changes

    if not df_active.empty:
        price_changes = get_price_changes(df_active["Ticker"].tolist())

        df_div = df_active.copy()
        df_div["Kurs_1M_%"] = df_div["Ticker"].map(price_changes)
        df_div = df_div.dropna(subset=["Kurs_1M_%"])

        df_div["Divergenz"] = "Kein Signal"
        df_div.loc[
            (df_div["Score"] > 0.1) & (df_div["Kurs_1M_%"] < -3), "Divergenz"
        ] = "⚡ Positives Sentiment, fallender Kurs"
        df_div.loc[
            (df_div["Score"] < -0.1) & (df_div["Kurs_1M_%"] > 3), "Divergenz"
        ] = "⚡ Negatives Sentiment, steigender Kurs"
        df_div.loc[
            (df_div["Score"] > 0.1) & (df_div["Kurs_1M_%"] > 3), "Divergenz"
        ] = "✅ Bestätigt bullisch"
        df_div.loc[
            (df_div["Score"] < -0.1) & (df_div["Kurs_1M_%"] < -3), "Divergenz"
        ] = "✅ Bestätigt bärisch"

        fig_div = px.scatter(
            df_div, x="Score", y="Kurs_1M_%",
            color="Divergenz", hover_name="Ticker",
            size=df_div["Artikel"].clip(lower=2),
            hover_data={"Sektor": True, "Events": True},
            size_max=30,
            color_discrete_map={
                "⚡ Positives Sentiment, fallender Kurs": "#f59e0b",
                "⚡ Negatives Sentiment, steigender Kurs": "#f59e0b",
                "✅ Bestätigt bullisch": "#22c55e",
                "✅ Bestätigt bärisch": "#ef4444",
                "Kein Signal": "#94a3b8",
            },
        )
        fig_div.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
        fig_div.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.4)
        fig_div.update_layout(
            height=500, margin=dict(t=10),
            xaxis_title="Sentiment Score",
            yaxis_title="Kursveränderung 1M (%)",
        )
        st.plotly_chart(fig_div, use_container_width=True)

        divs = df_div[df_div["Divergenz"].str.startswith("⚡")]
        if not divs.empty:
            st.subheader("⚡ Divergenzen gefunden")
            for _, row in divs.iterrows():
                st.warning(
                    f"**{row['Ticker']}** ({row['Sektor']}): "
                    f"{row['Divergenz']} · "
                    f"Score: {row['Score']:+.3f} · Kurs: {row['Kurs_1M_%']:+.1f}%"
                )
        else:
            st.info("Keine starken Divergenzen erkannt.")

        # Deep Dive
        st.divider()
        st.subheader("🔍 Einzelaktie Deep Dive")
        pick = st.selectbox("Aktie auswählen:", df_active["Ticker"].tolist(), key="deep")
        if pick:
            ts_data = next((ts for ts in sentiments if ts.ticker == pick), None)
            hist = yf.Ticker(pick).history(period="3mo")

            c1, c2 = st.columns([2, 1])
            with c1:
                fig_c = go.Figure()
                fig_c.add_trace(go.Scatter(
                    x=hist.index, y=hist["Close"],
                    name="Kurs (€)", line=dict(color="#6366f1", width=2),
                    fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
                ))
                fig_c.update_layout(
                    height=300, margin=dict(t=10, b=10),
                    yaxis_title="Kurs (€)", xaxis_title="",
                )
                st.plotly_chart(fig_c, use_container_width=True)

            with c2:
                if ts_data:
                    st.metric("Score", f"{ts_data.score:+.3f}")
                    st.metric("Konfidenz", f"{ts_data.confidence:.0%}")
                    st.metric("Artikel", ts_data.article_count)
                    st.metric("Signal", "✅ Zuverlässig" if ts_data.is_reliable else "⚠️ Dünn")
                    if ts_data.dominant_events:
                        st.write(f"**Events:** {', '.join(ts_data.dominant_events)}")

                    if ts_data.score > 0.2:
                        st.success("Nachrichtenlage **positiv**.")
                    elif ts_data.score < -0.2:
                        st.error("Nachrichtenlage **negativ**.")
                    else:
                        st.warning("Nachrichtenlage **neutral**.")


# ═══ TAB 5: HISTORIE ═══════════════════════════════════════════════════════
with tab5:
    st.subheader("Sentiment-Verlauf über Zeit")
    st.write("Zeigt gespeicherte Scans. Je mehr Scans, desto aussagekräftiger der Trend.")

    idx_hist = store.get_index_history(days=90)

    if idx_hist:
        hdf = pd.DataFrame(idx_hist)
        hdf["scan_time"] = pd.to_datetime(hdf["scan_time"])

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hdf["scan_time"], y=hdf["index_score"],
            mode="lines+markers", name="Index Score",
            line=dict(color="#6366f1", width=2),
            marker=dict(size=8),
        ))
        fig_hist.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_hist.add_hrect(y0=-0.05, y1=0.05, fillcolor="gray", opacity=0.1,
                           annotation_text="Neutral-Zone")
        fig_hist.update_layout(
            height=350, margin=dict(t=20, b=10),
            xaxis_title="Scan-Zeitpunkt", yaxis_title="Index Score",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()
        st.subheader("Ticker-Historie")
        pick_t = st.selectbox("Ticker:", df_active["Ticker"].tolist(), key="hist_ticker")
        if pick_t:
            t_hist = store.get_ticker_history(pick_t, days=90)
            if t_hist:
                thdf = pd.DataFrame(t_hist)
                thdf["scan_time"] = pd.to_datetime(thdf["scan_time"])
                fig_th = px.line(thdf, x="scan_time", y="score", markers=True,
                                 labels={"score": "Score", "scan_time": "Zeit"})
                fig_th.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_th.update_layout(height=250, margin=dict(t=10, b=10))
                st.plotly_chart(fig_th, use_container_width=True)
            else:
                st.info(f"Noch keine historischen Daten für {pick_t}.")
    else:
        st.info(
            "Noch keine historischen Daten vorhanden. "
            "Nach dem ersten Scan erscheint hier der Trend."
        )
