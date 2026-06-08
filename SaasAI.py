"""
Obsługiwane API AI:
    - Groq      (darmowe konto na console.groq.com)
    - OpenAI    (platform.openai.com)
    - Gemini    (makersuite.google.com)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# KONFIGURACJA STRONY

st.set_page_config(
    page_title="SaaS-Insight AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# NIESTANDARDOWE CSS

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif; }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(145deg, #0d0f1a 0%, #131829 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 16px;
        padding: 22px 20px;
        text-align: center;
        transition: all 0.25s ease;
    }
    .kpi-card:hover {
        border-color: rgba(99,102,241,0.5);
        box-shadow: 0 0 24px rgba(99,102,241,0.15);
        transform: translateY(-3px);
    }
    .kpi-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: #818cf8;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 6px;
    }
    .kpi-positive { color: #34d399 !important; }
    .kpi-negative { color: #f87171 !important; }

    /* Section titles */
    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        color: #818cf8;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(99,102,241,0.18);
        margin-bottom: 18px;
    }

    /* AI badge */
    .ai-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: #fff;
        font-size: 0.65rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        vertical-align: middle;
        margin-left: 10px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        letter-spacing: 0.04em;
        padding: 10px 22px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        box-shadow: 0 8px 25px rgba(99,102,241,0.4);
        transform: translateY(-2px);
    }
    .stButton > button[kind="secondary"] {
        background: rgba(99,102,241,0.1) !important;
        border: 1px solid rgba(99,102,241,0.3) !important;
        color: #818cf8 !important;
    }

    /* Report container */
    .report-box {
        background: rgba(13,15,26,0.6);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 16px;
        padding: 28px 32px;
        margin-top: 16px;
        line-height: 1.75;
    }

    /* Example prompt button */
    .example-btn { width: 100%; text-align: left; }

    /* Info box styling */
    .locked-info {
        background: rgba(99,102,241,0.06);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 12px;
        padding: 20px 24px;
        margin: 16px 0;
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

#INICJALIZACJA

_DEFAULTS = {
    "df":              None,     # główny DataFrame z danymi sprzedażowymi
    "data_profile":    None,     # tekstowy profil statystyczny dla LLM
    "messages":        [],       # historia czatu (lista {role, content})
    "report_content":  None,     # ostatnio wygenerowany raport Markdown
    "queued_prompt":   None,     # prompt z przykładowych przycisków
    "demo_seed":       2025,     # default seed do generowania danych
    "using_csv":       False,    # czy użytkownik wgrał CSV
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# GENEROWANIE DANYCH SYNTETYCZNYCH

@st.cache_data(show_spinner=False)
def generate_synthetic_data(seed: int = 2025) -> pd.DataFrame:
    """
    Generuje realistyczny, 6-miesięczny zestaw danych e-commerce (650 transakcji).
    Dane mają wbudowany wzorzec sezonowości i różne rozkłady kategorii.
    """
    rng = np.random.default_rng(seed=seed)
    today = datetime.today()
    start_date = today - timedelta(days=183)

    products = {
        "Elektronika": [
            "Słuchawki Pro X",
            "Smartwatch Elite",
            "Klawiatura Mechaniczna",
            "Mysz Gamingowa RGB",
            "Hub USB-C 7w1",
        ],
        "Odzież": [
            "Kurtka Zimowa Parka",
            "Spodnie Slim Fit",
            "Bluza Oversize",
            "Sneakersy RunX",
            "Koszulka Polo Premium",
        ],
        "Dom & Ogród": [
            "Lampa LED Smart",
            "Organizer Biurkowy",
            "Dyfuzor Aromatyczny",
            "Zestaw Narzędzi 20el.",
            "Dywan Boho 180cm",
        ],
        "Sport & Fitness": [
            "Mata do Jogi 6mm",
            "Hantle Hex 5kg",
            "Butelka Sportowa 750ml",
            "Opaska Fitness HR",
            "Plecak Trekkingowy 30L",
        ],
    }
    base_prices = {
        "Elektronika": 280.0,
        "Odzież": 130.0,
        "Dom & Ogród": 90.0,
        "Sport & Fitness": 110.0,
    }
    countries   = ["Polska", "Niemcy", "Francja", "Czechy", "Holandia", "Włochy", "Szwecja"]
    channels    = ["Social Media", "Google", "Direct"]
    ch_weights  = [0.44, 0.34, 0.22]

    records = []
    n = 650
    day_offsets = rng.integers(0, 184, size=n)

    for i in range(n):
        date = start_date + timedelta(days=int(day_offsets[i]))

        # Sezonowość: więcej transakcji w weekendy
        dow = date.weekday()
        if dow >= 5 and rng.random() < 0.35:
            pass

        category = rng.choice(list(products.keys()), p=[0.35, 0.28, 0.20, 0.17])
        product  = rng.choice(products[category])
        price    = round(float(base_prices[category]) * float(rng.uniform(0.55, 2.4)), 2)
        qty      = int(rng.integers(1, 9))
        revenue  = round(price * qty, 2)

        records.append({
            "Data":              date,
            "Produkt":           product,
            "Kategoria":         category,
            "Liczba_sztuk":      qty,
            "Cena_jednostkowa":  price,
            "Przychód":          revenue,
            "Kraj_klienta":      rng.choice(countries),
            "Kanał_Pozyskania":  rng.choice(channels, p=ch_weights),
        })

    df = pd.DataFrame(records)
    df["Data"] = pd.to_datetime(df["Data"])
    df = df.sort_values("Data").reset_index(drop=True)
    return df


def compute_data_profile(df: pd.DataFrame) -> str:
    """
    Buduje tekstowy profil statystyczny DataFrame.
    Ten profil jest wstrzykiwany jako system prompt do LLM,
    dając modelowi kontekst do analizy danych bez rzeczywistego
    wysyłania pełnej tabeli.
    """
    total_rev   = df["Przychód"].sum()
    aov         = df["Przychód"].mean()
    top_cat     = df.groupby("Kategoria")["Przychód"].sum().idxmax()
    top_prod    = df.groupby("Produkt")["Przychód"].sum().idxmax()
    top_ch      = df.groupby("Kanał_Pozyskania")["Przychód"].sum().idxmax()
    top_country = df.groupby("Kraj_klienta")["Przychód"].sum().idxmax()

    df2 = df.copy()
    df2["Miesiąc"] = df2["Data"].dt.to_period("M")
    monthly = df2.groupby("Miesiąc")["Przychód"].sum()
    mom = ((monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2] * 100) if len(monthly) >= 2 else 0.0

    top5 = df.groupby("Produkt")["Przychód"].sum().nlargest(5)
    top5_str = "\n".join(f"  {j+1}. {p}: {r:,.0f} PLN" for j, (p, r) in enumerate(top5.items()))

    ch_rev = df.groupby("Kanał_Pozyskania")["Przychód"].sum()
    ch_str = "\n".join(f"  - {k}: {v:,.0f} PLN ({v/total_rev*100:.1f}%)" for k, v in ch_rev.items())

    cat_rev = df.groupby("Kategoria")["Przychód"].sum().sort_values(ascending=False)
    cat_str = "\n".join(f"  - {k}: {v:,.0f} PLN" for k, v in cat_rev.items())

    worst_prod = df.groupby("Produkt")["Przychód"].sum().nsmallest(3)
    worst_str  = ", ".join(worst_prod.index.tolist())

    return f"""
=== PROFIL DANYCH SPRZEDAŻOWYCH ===
Okres: {df["Data"].min().strftime("%d.%m.%Y")} — {df["Data"].max().strftime("%d.%m.%Y")}
Liczba transakcji: {len(df):,}

KLUCZOWE METRYKI:
  • Całkowity przychód:      {total_rev:,.0f} PLN
  • Śr. wartość zamówienia:  {aov:.2f} PLN
  • Najlepsza kategoria:     {top_cat}
  • Bestseller produktowy:   {top_prod}
  • Dom. kanał pozyskania:   {top_ch}
  • Kluczowy rynek:          {top_country}
  • Wzrost m/m (ostatni):    {mom:+.1f}%

TOP 5 PRODUKTÓW (przychód):
{top5_str}

PRZYCHÓD WG KANAŁÓW:
{ch_str}

PRZYCHÓD WG KATEGORII:
{cat_str}

PRODUKTY NAJSŁABSZE (uwaga):
  {worst_str}
==================================
"""

# KLIENT AI — obsługa Groq / OpenAI / Gemini

def call_ai(
    provider: str,
    api_key: str,
    messages: list[dict],
    system_prompt: str = "",
    max_tokens: int = 2048,
) -> str | None:
    """
    Ujednolicony klient AI obsługujący trzech dostawców modeli LLM.
    Args:
        provider    : Nazwa dostawcy: "Groq", "OpenAI" lub "Gemini".
        api_key     : Klucz API dostawcy.
        messages    : Historia wiadomości [{role: str, content: str}, ...].
        system_prompt: Systemowy kontekst (profil danych) wstrzykiwany przed rozmową.
        max_tokens  : Limit tokenów odpowiedzi modelu.

    Returns:
        Treść odpowiedzi modelu (str) lub None przy błędzie.
    """
    try:
        # Groq (llama-3.1-8b-instant)
        if provider == "Groq":
            from groq import Groq
            client = Groq(api_key=api_key)

            full_msgs = []
            if system_prompt:
                full_msgs.append({"role": "system", "content": system_prompt})
            full_msgs.extend(messages)

            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=full_msgs,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return resp.choices[0].message.content

        # OpenAI (gpt-4o-mini)
        elif provider == "OpenAI":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            full_msgs = []
            if system_prompt:
                full_msgs.append({"role": "system", "content": system_prompt})
            full_msgs.extend(messages)

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=full_msgs,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return resp.choices[0].message.content

        # Google Gemini (gemini-1.5-flash)
        elif provider == "Gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)

            model = genai.GenerativeModel(
                model_name="gemini-3.5-flash",
                system_instruction=system_prompt if system_prompt else None,
            )

            # Gemini używa ról "user" / "model" (nie "assistant")
            history = []
            for m in messages[:-1]:
                history.append({
                    "role": "user" if m["role"] == "user" else "model",
                    "parts": [m["content"]],
                })

            chat = model.start_chat(history=history)
            resp = chat.send_message(messages[-1]["content"])
            return resp.text

        else:
            st.error(f"Nieznany dostawca AI: {provider}")
            return None

    except ImportError as exc:
        # Brak wymaganej biblioteki — podpowiedź instalacji
        lib = str(exc).split("'")[1] if "'" in str(exc) else str(exc)
        st.error(
            f"❌ Brak biblioteki `{lib}`. Zainstaluj: "
            f"`pip install {lib.replace('.', '-').replace('google-generativeai', 'google-generativeai')}`"
        )
        return None

    except Exception as exc:
        err = str(exc)
        if "401" in err or "authentication" in err.lower() or "api_key" in err.lower():
            st.error("❌ Błąd autoryzacji — sprawdź poprawność klucza API.")
        elif "429" in err or "rate" in err.lower():
            st.error("⏳ Przekroczono limit zapytań. Poczekaj chwilę i spróbuj ponownie.")
        elif "timeout" in err.lower() or "connect" in err.lower():
            st.error("🌐 Błąd połączenia — sprawdź internet lub status serwisu API.")
        else:
            st.error(f"❌ Błąd API ({provider}): {exc}")
        return None

# SIDEBAR

with st.sidebar:
    st.markdown("## 📊 SaaS-Insight AI")
    st.markdown("<p style='color:#64748b;font-size:0.82rem;'>Twój asystent Business Intelligence</p>", unsafe_allow_html=True)
    st.divider()

    # Wybór dostawcy AI
    st.markdown("#### Dostawca AI")
    ai_provider = st.selectbox(
        "Dostawca:",
        ["Groq", "OpenAI", "Gemini"],
        index=0,
        label_visibility="collapsed",
        help="Groq oferuje darmowy dostęp do modeli Llama — idealny na start.",
    )

    model_labels = {
        "Groq":   "llama-3.1-8b-instant (szybki & darmowy)",
        "OpenAI": "gpt-4o-mini",
        "Gemini": "gemini-1.5-flash",
    }
    st.markdown(
        f"<p style='font-size:0.75rem;color:#475569;margin-top:-8px;'>Model: {model_labels[ai_provider]}</p>",
        unsafe_allow_html=True
    )

    api_key = st.text_input(
        "Klucz API:",
        type="password",
        placeholder=f"{ai_provider} API key...",
        label_visibility="visible",
    )

    api_links = {
        "Groq":   "https://console.groq.com/keys",
        "OpenAI": "https://platform.openai.com/api-keys",
        "Gemini": "https://makersuite.google.com/app/apikey",
    }
    st.markdown(f"👉 [Zdobądź darmowy klucz {ai_provider}]({api_links[ai_provider]})")

    st.divider()

    # Upload CSV
    st.markdown("#### 📁 Dane sprzedażowe")
    uploaded = st.file_uploader(
        "Wgraj CSV (opcjonalnie):",
        type=["csv"],
        help="Wymagane kolumny: Data, Produkt, Kategoria, Liczba_sztuk, Cena_jednostkowa, Przychód, Kraj_klienta, Kanał_Pozyskania",
    )

    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            # Próba konwersji kolumny Data
            if "Data" in df_up.columns:
                df_up["Data"] = pd.to_datetime(df_up["Data"], errors="coerce")
            st.session_state.df = df_up
            st.session_state.data_profile = compute_data_profile(df_up)
            st.session_state.using_csv = True
            st.success(f"✅ Wgrano {len(df_up):,} rekordów z pliku CSV.")
        except Exception as e:
            st.error(f"Błąd wczytywania CSV: {e}")
    else:
        st.session_state.using_csv = False
        if st.session_state.df is None:
            with st.spinner("Generowanie danych demo..."):
                st.session_state.df = generate_synthetic_data(st.session_state.demo_seed)
            st.session_state.data_profile = compute_data_profile(st.session_state.df)
            st.info("💡 Używam syntetycznych danych demonstracyjnych (650 transakcji, 6 miesięcy).")
        st.markdown(
            "<p style='font-size:0.78rem;color:#475569;margin-bottom:6px;'>"
            "Dane demonstracyjne (seed: <code style='color:#818cf8'>"
            f"{st.session_state.demo_seed}</code>)</p>",
            unsafe_allow_html=True,
        )
        if st.button("🎲 Generuj nowe dane demo", use_container_width=True, type="secondary"):
            new_seed = random.randint(1, 99999)
            st.session_state.demo_seed = new_seed
            with st.spinner("Generowanie nowego zestawu danych..."):
                new_df = generate_synthetic_data(new_seed)
            st.session_state.df = new_df
            st.session_state.data_profile = compute_data_profile(new_df)
            st.session_state.messages = []
            st.session_state.report_content = None
            st.success(f"✅ Wygenerowano nowy zestaw danych (seed: {new_seed}).")
            st.rerun()

    # Aktualizuj profil danych gdy DataFrame istnieje
    if st.session_state.df is not None and st.session_state.data_profile is None:
        st.session_state.data_profile = compute_data_profile(st.session_state.df)

    st.divider()
    st.markdown(
        "<p style='font-size:0.72rem;color:#334155;text-align:center;'>"
        "SaaSAI<br>Dane przetwarzane lokalnie</p>",
        unsafe_allow_html=True
    )

# GŁÓWNE ZAKŁADKI

df = st.session_state.df

tab1, tab2, tab3 = st.tabs([
    "📊  Panel Analityczny",
    "💬  Chat z Danymi",
    "🤖  Generator Raportów AI",
])

# TAB 1 — PANEL ANALITYCZNY

with tab1:
    st.markdown("## 📊 Panel Analityczny")

    if df is None:
        st.warning("Brak danych — wgraj plik CSV lub odśwież stronę.")
        st.stop()

    # Obliczenia KPI
    total_rev = df["Przychód"].sum()
    aov       = df["Przychód"].mean()
    top_cat   = df.groupby("Kategoria")["Przychód"].sum().idxmax()

    df_m = df.copy()
    df_m["Miesiąc"] = df_m["Data"].dt.to_period("M")
    monthly = df_m.groupby("Miesiąc")["Przychód"].sum()
    mom = ((monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2] * 100) if len(monthly) >= 2 else 0.0
    mom_color  = "kpi-positive" if mom >= 0 else "kpi-negative"
    mom_prefix = "+" if mom >= 0 else ""

    st.markdown(
        f"<p style='color:#475569;font-size:0.88rem;margin-bottom:20px;'>"
        f"Analizujesz <b>{len(df):,}</b> transakcji • "
        f"{df['Data'].min().strftime('%d.%m.%Y')} — {df['Data'].max().strftime('%d.%m.%Y')}"
        f"</p>",
        unsafe_allow_html=True
    )

    # KPI
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">💰 {total_rev/1000:.1f}k</div>
            <div class="kpi-label">Całkowity przychód (PLN)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">🧾 {aov:.0f}</div>
            <div class="kpi-label">Śr. wartość zamówienia (PLN)</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">🏆 {top_cat}</div>
            <div class="kpi-label">Najlepsza kategoria</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value {mom_color}">{mom_prefix}{mom:.1f}%</div>
            <div class="kpi-label">Dynamika wzrostu m/m</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Wykres 1: Trend przychodów
    st.markdown('<p class="section-title">📈 Trend przychodów w czasie (tygodniowo)</p>', unsafe_allow_html=True)

    rev_weekly = (
        df.set_index("Data")["Przychód"]
        .resample("W")
        .sum()
        .reset_index()
        .rename(columns={"Data": "Tydzień", "Przychód": "Przychód"})
    )

    fig_trend = px.area(
        rev_weekly,
        x="Tydzień",
        y="Przychód",
        color_discrete_sequence=["#6366f1"],
        labels={"Przychód": "Przychód tygodniowy (PLN)", "Tydzień": ""},
    )
    fig_trend.update_traces(line_width=2.5, fillcolor="rgba(99,102,241,0.12)")
    fig_trend.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#64748b", family="IBM Plex Sans"),
        hovermode="x unified",
        margin=dict(l=0, r=0, t=8, b=8),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", color="#475569"),
        xaxis=dict(showgrid=False, color="#475569"),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Wykresy 2 & 3
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<p class="section-title">📡 Kanały pozyskania klientów</p>', unsafe_allow_html=True)

        ch_data = (
            df.groupby("Kanał_Pozyskania")["Przychód"]
            .sum()
            .reset_index()
            .sort_values("Przychód", ascending=False)
        )
        ch_data.columns = ["Kanał", "Przychód"]

        fig_funnel = go.Figure(go.Funnel(
            y=ch_data["Kanał"],
            x=ch_data["Przychód"],
            textposition="inside",
            textinfo="label+percent initial",
            marker=dict(
                color=["#6366f1", "#818cf8", "#a5b4fc"],
                line=dict(width=0),
            ),
            connector=dict(line=dict(color="rgba(0,0,0,0)")),
        ))
        fig_funnel.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="IBM Plex Sans", size=13),
            margin=dict(l=0, r=0, t=8, b=8),
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

    with col_r:
        st.markdown('<p class="section-title">🏅 Top 5 produktów — przychód</p>', unsafe_allow_html=True)

        top5 = (
            df.groupby("Produkt")["Przychód"]
            .sum()
            .nlargest(5)
            .reset_index()
            .sort_values("Przychód")
        )
        top5.columns = ["Produkt", "Przychód"]

        fig_bar = px.bar(
            top5,
            x="Przychód",
            y="Produkt",
            orientation="h",
            color="Przychód",
            color_continuous_scale=["#1e1b4b", "#818cf8"],
            text=top5["Przychód"].apply(lambda v: f"{v:,.0f} PLN"),
            labels={"Przychód": "Przychód (PLN)", "Produkt": ""},
        )
        fig_bar.update_traces(textposition="outside", textfont_color="#94a3b8")
        fig_bar.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#64748b", family="IBM Plex Sans"),
            coloraxis_showscale=False,
            margin=dict(l=0, r=60, t=8, b=8),
            yaxis=dict(showgrid=False),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Tabela podglądu
    with st.expander("🔍 Podgląd surowych danych (pierwsze 25 wierszy)"):
        display = df.drop(columns=["Miesiąc"], errors="ignore").head(25).copy()
        display["Data"] = display["Data"].dt.strftime("%d.%m.%Y")
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Cena_jednostkowa": st.column_config.NumberColumn(format="%.2f PLN"),
                "Przychód":         st.column_config.NumberColumn(format="%.2f PLN"),
            },
        )

# CHAT Z DANYMI

with tab2:
    st.markdown(
        "## 💬 Chat z Danymi"
        "<span class='ai-badge'>AI Powered</span>",
        unsafe_allow_html=True
    )
    st.markdown("Zadaj dowolne pytanie o swoje dane sprzedażowe. AI przeanalizuje je i odpowie po polsku.")

    if not api_key:
        st.markdown("""
        <div class="locked-info">
            🔒 <strong>Konsultant AI jest zablokowany</strong><br><br>
            Aby odblokować czat z danymi, wprowadź swój klucz API w panelu bocznym.<br>
            Obsługujemy <strong>Groq</strong> (darmowy limit), <strong>OpenAI</strong> oraz <strong>Google Gemini</strong>.
        </div>
        """, unsafe_allow_html=True)
    else:
        # System prompt dla chatu — profil danych wstrzykiwany jako kontekst
        CHAT_SYSTEM = f"""Jesteś zaawansowanym asystentem Business Intelligence dla platformy e-commerce.
Posiadasz dostęp do rzeczywistego profilu statystycznego danych sprzedażowych firmy klienta.
Odpowiadaj zawsze po polsku. Bądź konkretny i operuj na liczbach z profilu danych.
Gdy tworzysz treści marketingowe (posty, maile, reklamy) — formatuj je czytelnie i profesjonalnie.
Jeśli pytanie wymaga danych spoza profilu, powiedz o tym wprost.

{st.session_state.data_profile or "Brak profilu danych — poproś użytkownika o wgranie danych."}
"""

        # Przykładowe pytania (quickstart)
        st.markdown("**💡 Szybki start — przykładowe pytania:**")
        examples = [
            "Który produkt powinienem promować w przyszłym miesiącu i dlaczego?",
            "Napisz profesjonalny post na LinkedIn o naszym bestsellerze",
            "Jakie są największe ryzyka w obecnej strategii sprzedaży?",
            "Zaproponuj kampanię rabatową na najgorzej sprzedające się produkty",
            "Który rynek zagraniczny ma największy potencjał wzrostu?",
            "Wygeneruj temat i preheader do newslettera na ten miesiąc",
        ]

        ex_cols = st.columns(3)
        for i, ex in enumerate(examples):
            with ex_cols[i % 3]:
                if st.button(
                    f"_{ex[:42]}..._" if len(ex) > 42 else f"_{ex}_",
                    key=f"ex_{i}",
                    use_container_width=True,
                ):
                    st.session_state.queued_prompt = ex

        st.divider()

        # Historia czatu
        for msg in st.session_state.messages:
            avatar = "👤" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        # Obsługa inputu (chat_input LUB przykładowy przycisk)
        user_text = st.chat_input("Wpisz pytanie do konsultanta AI...", key="chat_input")

        # Jeśli nie wpisano tekstu, sprawdź czy jest kolejkowany prompt z przycisku
        if user_text is None and st.session_state.queued_prompt:
            user_text = st.session_state.queued_prompt
            st.session_state.queued_prompt = None

        if user_text:
            # Dodaj wiadomość użytkownika
            st.session_state.messages.append({"role": "user", "content": user_text})

            with st.chat_message("user", avatar="👤"):
                st.markdown(user_text)

            # Odpowiedź AI
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner(f"🧠 {ai_provider} analizuje Twoje dane..."):
                    ai_response = call_ai(
                        provider=ai_provider,
                        api_key=api_key,
                        messages=st.session_state.messages,
                        system_prompt=CHAT_SYSTEM,
                        max_tokens=1800,
                    )

                if ai_response:
                    st.markdown(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})

        # Akcje czatu
        if st.session_state.messages:
            col_clr, col_exp = st.columns([1, 4])
            with col_clr:
                if st.button("🗑️ Wyczyść czat", type="secondary"):
                    st.session_state.messages = []
                    st.rerun()
            with col_exp:
                chat_export = "\n\n".join(
                    f"**{m['role'].upper()}:** {m['content']}"
                    for m in st.session_state.messages
                )
                st.download_button(
                    "📥 Eksportuj rozmowę (.md)",
                    data=chat_export,
                    file_name=f"chat_eksport_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    type="secondary",
                )

# TAB 3 — GENERATOR RAPORTÓW AI

with tab3:
    st.markdown(
        "## 🤖 Generator Raportów i Strategii AI"
        "<span class='ai-badge'>AI Powered</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        "Skonfiguruj parametry raportu i wygeneruj pełny **Audyt Biznesowy** "
        "wraz z analizą SWOT, planem działania i propozycją kampanii mailingowej."
    )

    if not api_key:
        st.markdown("""
        <div class="locked-info">
            🔒 <strong>Generator Raportów AI jest zablokowany</strong><br><br>
            Aby odblokować automatyczne generowanie raportów, wprowadź swój klucz API w panelu bocznym.<br>
            Obsługujemy <strong>Groq</strong> (darmowy limit), <strong>OpenAI</strong> oraz <strong>Google Gemini</strong>.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Formularz konfiguracji raportu
        st.markdown('<p class="section-title">⚙️ Konfiguracja Audytu Biznesowego</p>', unsafe_allow_html=True)

        with st.container():
            cfg_col1, cfg_col2 = st.columns(2)

            with cfg_col1:
                business_goal = st.selectbox(
                    "🎯 Cel biznesowy:",
                    [
                        "Maksymalizacja przychodów",
                        "Czyszczenie magazynu (wyprzedaż wolnorotujących)",
                        "Ekspansja zagraniczna",
                        "Budowanie lojalności i retencji klientów",
                        "Optymalizacja mix-u kanałów pozyskania",
                        "Podniesienie marżowości sprzedaży",
                    ],
                    help="Cel wpływa na rekomendacje i priorytety w planie działania.",
                )

            with cfg_col2:
                report_tone = st.selectbox(
                    "🗣️ Ton i styl raportu:",
                    [
                        "Profesjonalny i analityczny (CEO/CFO)",
                        "Agresywny marketingowo (Growth Hacking)",
                        "Sformalizowany korporacyjny (Board Report)",
                        "Przejrzysty i operacyjny (Menedżer sprzedaży)",
                    ],
                )

            planning_horizon = st.select_slider(
                "📅 Horyzont planowania:",
                options=["1 miesiąc", "3 miesiące", "6 miesięcy", "12 miesięcy"],
                value="3 miesiące",
            )

            st.markdown("<br>", unsafe_allow_html=True)

            gen_btn = st.button("🚀 Wygeneruj Audyt Biznesowy AI", use_container_width=True)

        # ── Generowanie raportu ──────────────
        if gen_btn:
            REPORT_PROMPT = f"""Jesteś doświadczonym Senior Business Analyst i Chief Strategy Officer.
Na podstawie poniższego profilu danych sprzedażowych stwórz kompletny, profesjonalny Audyt Biznesowy po polsku.
Raport musi być oparty na konkretnych liczbach z profilu danych — bez ogólników.

{st.session_state.data_profile or "Brak danych."}

PARAMETRY RAPORTU:
  • Cel biznesowy:      {business_goal}
  • Styl i ton:         {report_tone}
  • Horyzont działania: {planning_horizon}

Wygeneruj raport w dokładnie następującej strukturze Markdown:

---

# 📋 Audyt Biznesowy — SaaSAI
**Data wygenerowania:** {datetime.now().strftime("%d.%m.%Y, %H:%M")}
**Cel biznesowy:** {business_goal} | **Horyzont:** {planning_horizon}

---

## 1. 🔍 Analiza SWOT — na podstawie danych sprzedażowych

### 💪 Mocne strony (Strengths)
[3–4 punkty oparte na faktycznych danych — co działa dobrze]

### ⚠️ Słabe strony (Weaknesses)
[3–4 punkty — konkretne słabości widoczne w danych]

### 🚀 Szanse (Opportunities)
[3–4 punkty — potencjalne obszary wzrostu]

### 🔴 Zagrożenia (Threats)
[2–3 punkty — ryzyka rynkowe lub wewnętrzne]

---

## 2. 🗺️ 3-Etapowy Plan Działania (Action Plan — {planning_horizon})

Każdy etap zawiera: Nazwę, Cel, Konkretne działania, Miernik sukcesu (KPI) i Szacowany wpływ.

### Etap 1: [Nazwa etapu]
**Cel:** ...
**Działania:** ...
**KPI:** ...
**Szacowany wpływ:** ...

### Etap 2: [Nazwa etapu]
**Cel:** ...
**Działania:** ...
**KPI:** ...
**Szacowany wpływ:** ...

### Etap 3: [Nazwa etapu]
**Cel:** ...
**Działania:** ...
**KPI:** ...
**Szacowany wpływ:** ...

---

## 3. ✉️ Propozycja Kampanii Mailingowej

*(Dopasowana do dominującego kanału pozyskania i celów biznesowych)*

**Temat maila:** [angażujący temat z personalizacją]
**Preheader:** [uzupełniający temat, max 80 znaków]

**Treść wiadomości:**
---
[Pełna treść maila z: pozdrowieniem, głównym przekazem, ofertą/korzyścią, CTA — przyciski, podpisem]
---

---

## 4. 💡 Kluczowe Rekomendacje Strategiczne

1. [Konkretna rekomendacja oparta na danych]
2. [...]
3. [...]
4. [...]
5. [...]

---
*Raport wygenerowany przez SaaSAI na podstawie {len(df) if df is not None else "N/A"} transakcji.*
"""

            with st.spinner(f"🔬 {ai_provider} generuje Twój audyt biznesowy... Może to potrwać do 30 sekund."):
                report = call_ai(
                    provider=ai_provider,
                    api_key=api_key,
                    messages=[{"role": "user", "content": REPORT_PROMPT}],
                    system_prompt="",
                    max_tokens=3500,
                )

            if report:
                st.session_state.report_content = report
                st.success("✅ Audyt wygenerowany pomyślnie!")

        # Wyświetl raport (zachowany w session_state)
        if st.session_state.report_content:
            st.divider()

            with st.container():
                st.markdown('<div class="report-box">', unsafe_allow_html=True)
                st.markdown(st.session_state.report_content)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Przyciski eksportu
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    label="📥 Pobierz raport (.md)",
                    data=st.session_state.report_content,
                    file_name=f"audyt_biznesowy_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown",
                    type="secondary",
                    use_container_width=True,
                )
            with dl_col2:
                if st.button("🔄 Wyczyść i wygeneruj nowy", type="secondary", use_container_width=True):
                    st.session_state.report_content = None
                    st.rerun()