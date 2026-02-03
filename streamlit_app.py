import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from bitcoin_agent import BitcoinAnalysisAgent

st.set_page_config(
    page_title="Agente de Análise de Bitcoin",
    layout="wide"
)

st.title("Agente de Análise Quantitativa de Bitcoin")
st.caption("Linear • Polinomial + Seno • Média Móvel • Oscilador Estocástico")

# =========================
# BARRA LATERAL
# =========================

st.sidebar.header("Configuração")

dias = st.sidebar.slider(
    "Janela histórica (dias)",
    min_value=60,
    max_value=365,
    value=180,
    step=30
)

mostrar_linear = st.sidebar.checkbox("Linear", True)
mostrar_poli_seno = st.sidebar.checkbox("Polinomial + Seno", True)
mostrar_ma = st.sidebar.checkbox("Média móvel (30d)", True)

# =========================
# EXECUÇÃO
# =========================

agente = BitcoinAnalysisAgent()

with st.spinner("Coletando dados do CoinGecko..."):
    agente.fetch_bitcoin_data(days=dias)
    resultados = agente.find_approximations()
    recomendacao = agente.investment_advice()

precos = agente.prices
x = np.arange(len(precos))

# =========================
# GRÁFICO DE PREÇO
# =========================

st.subheader("Preço do Bitcoin + Curvas de Aproximação")

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(precos, label="Preço BTC", linewidth=2, color="black")

if mostrar_linear:
    ax.plot(
        resultados["linear"]["prediction"](x),
        linestyle="--",
        label=f"Linear (R²={resultados['linear']['r2']:.2f})"
    )

if mostrar_poli_seno:
    ax.plot(
        resultados["poly_sine"]["prediction"](x),
        linestyle="--",
        label=f"Polinomial + Seno (R²={resultados['poly_sine']['r2']:.2f})"
    )

if mostrar_ma:
    ma = resultados["moving_average"]["values"]
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

estocastico = resultados["stochastic"]

fig2, ax2 = plt.subplots(figsize=(14, 4))

ax2.plot(estocastico["k"], label="%K", linewidth=2)
ax2.plot(estocastico["d"], label="%D", linewidth=2)

ax2.axhline(80, linestyle="--", alpha=0.5)
ax2.axhline(20, linestyle="--", alpha=0.5)

ax2.set_ylim(0, 100)
ax2.set_ylabel("Estocástico")
ax2.set_xlabel("Dias")
ax2.legend()
ax2.grid(alpha=0.3)

st.pyplot(fig2)

# =========================
# RECOMENDAÇÃO
# =========================

st.subheader("Recomendação do Agente")

col1, col2, col3 = st.columns(3)

col1.metric("Estocástico %K", f"{recomendacao['stochastic_k']:.1f}")
col2.metric("Recomendação", recomendacao["recommendation"])

melhor_modelo = max(
    ((k, v) for k, v in resultados.items() if "r2" in v),
    key=lambda x: x[1]["r2"]
)

col3.metric("Modelo dominante", melhor_modelo[0])

# =========================
# EXPLICAÇÃO
# =========================

with st.expander("Como o agente decide"):
    st.markdown("""
- **Curvas de preço** são usadas para **entender tendência**
- O **melhor modelo (R²)** indica a forma dominante do mercado
- O **oscilador estocástico** decide *timing*
    - %K < 20 → **Acumular**
    - %K > 80 → **Vender**
    - Caso contrário → **Manter**
- Nenhum modelo manda sozinho:
    - tendência ≠ ponto de entrada
    - ciclo ≠ direção
    """)

# =========================================================
# RODAPÉ
# =========================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
    "Agente Bitcoin por Eduardo Araujo © 2026<br>"
    "<strong>Agente Autônomo Bitcoin</strong><br>"
    "</div>",
    unsafe_allow_html=True
)