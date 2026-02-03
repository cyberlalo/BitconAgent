import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime

import plotly.graph_objects as go

from bitcoin_agent import BitcoinAnalysisAgent
from memory import init_db

# =========================================================
# CONFIGURAÇÃO BÁSICA
# =========================================================

st.set_page_config(
    page_title="Bitcoin Autonomous Agent Monitor",
    page_icon="₿",
    layout="wide"
)

st.title("₿ Bitcoin Autonomous Agent")
st.caption("Monitoring dashboard for an autonomous mathematical analysis agent")

# =========================================================
# INICIALIZAÇÃO DO BANCO (OBRIGATÓRIO)
# =========================================================

if "db_initialized" not in st.session_state:
    init_db()
    st.session_state.db_initialized = True

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("Analysis Settings")

    days = st.slider(
        "Historical window (days)",
        min_value=30,
        max_value=365,
        value=180
    )

    run_analysis = st.button("Run Manual Analysis")

    st.markdown("---")
    st.markdown("**Agent Mode**")
    st.info(
        "This dashboard does not control the autonomous agent.\n\n"
        "It only visualizes data, memory and on-demand analysis."
    )

# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

@st.cache_data(ttl=300)
def load_agent_memory(limit=100):
    conn = sqlite3.connect("memory.db", check_same_thread=False)
    df = pd.read_sql(
        f"""
        SELECT timestamp, model, predicted_price, confidence
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT {limit}
        """,
        conn
    )
    conn.close()
    return df


def run_on_demand_analysis(days):
    agent = BitcoinAnalysisAgent()
    data = agent.fetch_bitcoin_data(days=days)
    results = agent.find_approximations()
    return agent, data, results

# =========================================================
# PAINEL 1 — MEMÓRIA DO AGENTE
# =========================================================

st.subheader("Agent Memory (Predictions History)")

memory_df = load_agent_memory()

if memory_df.empty:
    st.info("No predictions stored yet.")
else:
    st.dataframe(memory_df, use_container_width=True)

# =========================================================
# PAINEL 2 — ANÁLISE MANUAL (ON-DEMAND)
# =========================================================

st.subheader("Manual Analysis (On-Demand)")

if run_analysis:
    with st.spinner("Running analysis..."):
        agent, data, results = run_on_demand_analysis(days)

    # -------------------------
    # MÉTRICAS
    # -------------------------

    col1, col2, col3 = st.columns(3)

    current_price = float(data["price"].iloc[-1])
    initial_price = float(data["price"].iloc[0])
    delta = (current_price / initial_price - 1) * 100

    with col1:
        st.metric("Current Price", f"${current_price:,.2f}", f"{delta:+.2f}%")

    with col2:
        vol = results["volatility"]["daily_volatility"]
        st.metric("Daily Volatility", f"{vol:.2f}%")

    with col3:
        slope = results["linear"]["params"]["slope"]
        st.metric("Linear Trend", f"${slope:.2f}/day")

    # -------------------------
    # GRÁFICO DE PREÇO
    # -------------------------

    st.subheader("Bitcoin Price & Models")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data["date"],
        y=data["price"],
        mode="lines",
        name="Bitcoin Price",
        line=dict(width=2)
    ))

    x = np.arange(len(data))

    for key, style in [
        ("linear", "dash"),
        ("polynomial", "dot"),
        ("exponential", "dashdot")
    ]:
        if key in results:
            y_pred = results[key]["prediction"](x)
            fig.add_trace(go.Scatter(
                x=data["date"],
                y=y_pred,
                mode="lines",
                name=f"{key.capitalize()} (R²={results[key].get('r_squared', 0):.3f})",
                line=dict(dash=style)
            ))

    fig.update_layout(
        height=500,
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # CONSELHO AUTOMÁTICO
    # -------------------------

    st.subheader("🤖 Agent Recommendation")

    advice = agent.investment_advice()

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("Recommendation", advice["recommendation"])

    with colB:
        st.metric("Risk Level", advice["risk_level"])

    with colC:
        st.metric("Confidence", f"{int(advice['confidence'] * 100)}%")

    with st.expander("Detailed Reasoning", expanded=True):
        for point in advice["key_points"]:
            st.write(f"• {point}")

        st.info(advice["summary"])

else:
    st.info("Click **Run Manual Analysis** to perform an on-demand analysis.")

# =========================================================
# RODAPÉ
# =========================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
     "Bitcoin Agent by Eduardo Araujo © 2026<br>"
    "<strong>Bitcoin Autonomous Agent</strong><br>"
    "<small>Mathematical modeling • Memory • Autonomy</small>"
    "</div>",
    unsafe_allow_html=True
)