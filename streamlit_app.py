import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from bitcoin_agent import BitcoinAnalysisAgent

st.set_page_config(
    page_title="Bitcoin Quant Agent",
    layout="wide"
)

st.title("📊 Bitcoin Quantitative Analysis Agent")
st.caption("Linear • Exponential • Polynomial + Sine • Moving Average • Stochastic Oscillator")

# =========================
# SIDEBAR
# =========================

st.sidebar.header("Configuração")

days = st.sidebar.slider(
    "Janela histórica (dias)",
    min_value=60,
    max_value=365,
    value=180,
    step=30
)

show_linear = st.sidebar.checkbox("Linear", True)
show_exponential = st.sidebar.checkbox("Exponencial", True)
show_poly_sine = st.sidebar.checkbox("Polinomial + Seno", True)
show_ma = st.sidebar.checkbox("Média móvel (30d)", True)

# =========================
# EXECUÇÃO
# =========================

agent = BitcoinAnalysisAgent()

with st.spinner("Coletando dados do CoinGecko..."):
    agent.fetch_bitcoin_data(days=days)
    results = agent.find_approximations()
    advice = agent.investment_advice()

prices = agent.prices
x = np.arange(len(prices))

# =========================
# GRÁFICO DE PREÇO
# =========================

st.subheader("Preço do Bitcoin + Curvas de Aproximação")

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(prices, label="Preço BTC", linewidth=2, color="black")

if show_linear:
    ax.plot(
        results["linear"]["prediction"](x),
        linestyle="--",
        label=f"Linear (R²={results['linear']['r2']:.2f})"
    )

if show_exponential:
    ax.plot(
        results["exponential"]["prediction"](x),
        linestyle="--",
        label=f"Exponencial (R²={results['exponential']['r2']:.2f})"
    )

if show_poly_sine:
    ax.plot(
        results["poly_sine"]["prediction"](x),
        linestyle="--",
        label=f"Polinomial + Seno (R²={results['poly_sine']['r2']:.2f})"
    )

if show_ma:
    ma = results["moving_average"]["values"]
    ax.plot(ma, label="Média móvel (30d)", linewidth=2)

ax.set_ylabel("USD")
ax.set_xlabel("Dias")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)

# =========================
# OSCILADOR ESTOCÁSTICO
# =========================

st.subheader("Oscilador Estocástico")

stoch = results["stochastic"]

fig2, ax2 = plt.subplots(figsize=(14, 4))

ax2.plot(stoch["k"], label="%K", linewidth=2)
ax2.plot(stoch["d"], label="%D", linewidth=2)

ax2.axhline(80, linestyle="--", alpha=0.5)
ax2.axhline(20, linestyle="--", alpha=0.5)

ax2.set_ylim(0, 100)
ax2.set_ylabel("Stochastic")
ax2.set_xlabel("Tempo")
ax2.legend()
ax2.grid(alpha=0.3)

st.pyplot(fig2)

# =========================
# RECOMENDAÇÃO
# =========================

st.subheader("📌 Recomendação do Agente")

col1, col2, col3 = st.columns(3)

col1.metric("Stochastic %K", f"{advice['stochastic_k']:.1f}")
col2.metric("Recomendação", advice["recommendation"])

best_model = max(
    ((k, v) for k, v in results.items() if "r2" in v),
    key=lambda x: x[1]["r2"]
)

col3.metric("Modelo dominante", best_model[0])

# =========================
# EXPLICAÇÃO
# =========================

with st.expander("🧠 Como o agente decide"):
    st.markdown("""
- **Curvas de preço** são usadas para **entender tendência**
- O **melhor modelo (R²)** é usado para previsão
- O **oscilador estocástico** decide *timing*
    - %K < 20 → **Accumulate**
    - %K > 80 → **Sell**
    - Caso contrário → **Hold**
- Nenhum modelo “manda sozinho”
    - tendência ≠ ponto de entrada
    - ciclo ≠ direção
    """)

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