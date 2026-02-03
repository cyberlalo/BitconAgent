import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from bitcoin_agent import BitcoinAnalysisAgent

# =========================
# CONFIG STREAMLIT
# =========================

st.set_page_config(
    page_title="Bitcoin Quant Agent",
    layout="wide"
)

st.title("Agente de Bitcoin")
st.caption("Modelo Linear , Polinomial + Seno , Média Movel + Oscilador Estocástico")

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
show_poly_sine = st.sidebar.checkbox("Polinomial + Seno", True)
show_ma = st.sidebar.checkbox("Média móvel (30d)", True)

FORECAST_DAYS = 7

# =========================
# EXECUÇÃO DO AGENTE
# =========================

agent = BitcoinAnalysisAgent()

with st.spinner("Coletando dados do CoinGecko..."):
    agent.fetch_bitcoin_data(days=days)
    results = agent.find_approximations()
    advice = agent.investment_advice()

prices = agent.prices
x = np.arange(len(prices))
current_price = prices[-1]

# =========================
# PREVISÃO DO AGENTE (D+7)
# =========================

best_model_name, best_model = max(
    ((k, v) for k, v in results.items() if "r2" in v and "prediction" in v),
    key=lambda item: item[1]["r2"]
)

future_x = len(prices) + FORECAST_DAYS
forecast_price = float(best_model["prediction"](future_x))

forecast_change_pct = (
    (forecast_price - current_price) / current_price
) * 100

# =========================
# MÉTRICAS DE PREVISÃO
# =========================

st.subheader("Previsão do Agente")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Preço atual (BTC)",
    f"${current_price:,.0f}"
)

c2.metric(
    f"Previsão D+{FORECAST_DAYS}",
    f"${forecast_price:,.0f}",
    delta=f"{forecast_change_pct:+.2f}%"
)

c3.metric(
    "Modelo dominante",
    best_model_name
)

c4.metric(
    "Confiança (R²)",
    f"{best_model['r2']:.2f}"
)

st.caption(
    f"O agente estima que, mantendo a estrutura atual do mercado, "
    f"o preço do Bitcoin em {FORECAST_DAYS} dias será "
    f"{forecast_change_pct:+.2f}% em relação ao preço atual, "
    f"com base no modelo **{best_model_name}**."
)

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

st.subheader("Recomendação do Agente")

col1, col2, col3 = st.columns(3)

col1.metric("Estocástico %K", f"{advice['stochastic_k']:.1f}")
col2.metric("Recomendação", advice["recommendation"])
col3.metric("Modelo dominante", best_model_name)

# =========================
# EXPLICAÇÃO
# =========================

with st.expander("Como o agente decide"):
    st.markdown("""
- **Curvas de preço** identificam a **tendência estrutural**
- O **modelo com maior R²** é usado para projeção
- A previsão é sempre **D + N dias**
- O **oscilador estocástico** define o *timing*:
    - %K < 20 → **Acumular**
    - %K > 80 → **Vender**
    - Caso contrário → **Manter**
- Tendência ≠ ponto de entrada
- O agente é **determinístico e reavaliado a cada ciclo**
    """)

# =========================
# RODAPÉ
# =========================

st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
    "Agente de Bitcoin criado por Eduardo Araujo © 2026<br>"
    "</div>",
    unsafe_allow_html=True
)
