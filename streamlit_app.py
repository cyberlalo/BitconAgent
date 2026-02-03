import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta

from bitcoin_agent import BitcoinAnalysisAgent
from memory import (
    init_db,
    get_prediction_accuracy,
    get_recent_predictions,
    get_model_performance,
    get_recommendation_accuracy
)

# =========================
# CONFIGURAÇÃO STREAMLIT
# =========================

st.set_page_config(
    page_title="Agente Quantitativo Bitcoin",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

st.title("Agente de Análise Quantitativa de Bitcoin")
st.caption("Análise Multi-Modelo com Machine Learning • Gestão de Risco • Backtesting")

# Inicializar DB
init_db()

# =========================
# BARRA LATERAL
# =========================

st.sidebar.header("⚙️ Configuração")

days = st.sidebar.slider(
    "Janela histórica (dias)",
    min_value=60,
    max_value=365,
    value=180,
    step=30
)

FORECAST_DAYS = st.sidebar.slider(
    "Horizonte de previsão (dias)",
    min_value=1,
    max_value=30,
    value=7,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Modelos")

show_linear = st.sidebar.checkbox("Linear", True)
show_poly = st.sidebar.checkbox("Polinomial", True)
show_poly_sine = st.sidebar.checkbox("Polinomial + Seno", True)
show_ma = st.sidebar.checkbox("Média Móvel (30d)", True)
show_ema = st.sidebar.checkbox("Média Móvel Exponencial", False)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Indicadores")

show_stochastic = st.sidebar.checkbox("Oscilador Estocástico", True)
show_rsi = st.sidebar.checkbox("RSI", True)

# =========================
# EXECUÇÃO DO AGENTE
# =========================

agent = BitcoinAnalysisAgent()

with st.spinner("🔄 Coletando dados do CoinGecko..."):
    try:
        agent.fetch_bitcoin_data(days=days)
        results = agent.find_approximations()
        risk_metrics = agent.calculate_risk_metrics()
        forecast = agent.forecast_with_confidence(forecast_days=FORECAST_DAYS)
        advice = agent.investment_advice()
        
        data_loaded = True
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
        st.stop()

prices = agent.prices
dates = agent.dates
x = np.arange(len(prices))
current_price = prices[-1]
forecast_price = forecast["forecast"]
forecast_change_pct = ((forecast_price - current_price) / current_price) * 100

# =========================
# MÉTRICAS PRINCIPAIS
# =========================

st.markdown("## 📊 Visão Geral")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Preço Atual",
        f"${current_price:,.0f}",
        delta=f"{risk_metrics['daily_return_mean']*100:.2f}% diário"
    )

with col2:
    delta_color = "normal" if forecast_change_pct > 0 else "inverse"
    st.metric(
        f"Previsão D+{FORECAST_DAYS}",
        f"${forecast_price:,.0f}",
        delta=f"{forecast_change_pct:+.2f}%"
    )

with col3:
    st.metric(
        "Volatilidade",
        f"{risk_metrics['volatility']:.1%}",
        delta="Anualizada"
    )

with col4:
    st.metric(
        "Índice Sharpe",
        f"{risk_metrics['sharpe_ratio']:.2f}",
        delta="Retorno ajustado"
    )

with col5:
    st.metric(
        "Drawdown Máximo",
        f"{risk_metrics['max_drawdown']:.1%}",
        delta="Risco máximo"
    )

# =========================
# RECOMENDAÇÃO
# =========================

st.markdown("## 💡 Recomendação de Investimento")

rec_col1, rec_col2, rec_col3, rec_col4 = st.columns(4)

with rec_col1:
    # Cor baseada na recomendação
    rec = advice['recommendation']
    if "Acumular" in rec or "Comprar" in rec:
        color = "green"
        emoji = "🟢"
    elif "Vender" in rec or "Reduzir" in rec:
        color = "red"
        emoji = "🔴"
    else:
        color = "gray"
        emoji = "⚪"
    
    st.markdown(f"### {emoji} {rec}")

with rec_col2:
    st.metric("Confiança", advice['confidence'].title())

with rec_col3:
    st.metric("Pontuação", f"{advice['score']:.1f}")

with rec_col4:
    st.metric("Modelo", forecast['model'])

# Razões da recomendação
if advice['reasons']:
    with st.expander("📋 Análise Detalhada"):
        for i, reason in enumerate(advice['reasons'], 1):
            st.write(f"{i}. {reason}")

# =========================
# INTERVALO DE CONFIANÇA
# =========================

st.markdown("## 🎯 Intervalo de Confiança (95%)")

conf_col1, conf_col2, conf_col3 = st.columns(3)

with conf_col1:
    st.metric(
        "Limite Inferior",
        f"${forecast['lower_bound']:,.0f}",
        delta=f"{((forecast['lower_bound'] - current_price) / current_price * 100):+.2f}%"
    )

with conf_col2:
    st.metric(
        "Previsão Central",
        f"${forecast_price:,.0f}",
        delta=f"{forecast_change_pct:+.2f}%"
    )

with conf_col3:
    st.metric(
        "Limite Superior",
        f"${forecast['upper_bound']:,.0f}",
        delta=f"{((forecast['upper_bound'] - current_price) / current_price * 100):+.2f}%"
    )

st.caption(
    f"Com 95% de confiança, o preço do Bitcoin em {FORECAST_DAYS} dias "
    f"estará entre ${forecast['lower_bound']:,.0f} e ${forecast['upper_bound']:,.0f}, "
    f"usando o modelo **{forecast['model']}** (R²={forecast['r2']:.4f})."
)

# =========================
# GRÁFICO INTERATIVO DE PREÇO
# =========================

st.markdown("## 📈 Análise de Preço e Modelos")

fig = go.Figure()

# Preço real
fig.add_trace(go.Scatter(
    x=dates,
    y=prices,
    mode='lines',
    name='Preço BTC',
    line=dict(color='black', width=2)
))

# Modelos
if show_linear and "linear" in results:
    fig.add_trace(go.Scatter(
        x=dates,
        y=results["linear"]["prediction"](x),
        mode='lines',
        name=f"Linear (R²={results['linear']['r2']:.3f})",
        line=dict(dash='dash')
    ))

if show_poly and "polynomial" in results:
    fig.add_trace(go.Scatter(
        x=dates,
        y=results["polynomial"]["prediction"](x),
        mode='lines',
        name=f"Polinomial (R²={results['polynomial']['r2']:.3f})",
        line=dict(dash='dash')
    ))

if show_poly_sine and "poly_sine" in results:
    fig.add_trace(go.Scatter(
        x=dates,
        y=results["poly_sine"]["prediction"](x),
        mode='lines',
        name=f"Poli+Seno (R²={results['poly_sine']['r2']:.3f})",
        line=dict(dash='dot')
    ))

if show_ma and "moving_average" in results:
    ma_dates = dates[29:]  # Ajustar para janela de 30 dias
    ma_values = results["moving_average"]["values"][29:]
    fig.add_trace(go.Scatter(
        x=ma_dates,
        y=ma_values,
        mode='lines',
        name='Média Móvel (30d)',
        line=dict(width=2)
    ))

if show_ema and "exponential_ma" in results:
    fig.add_trace(go.Scatter(
        x=dates,
        y=results["exponential_ma"]["values"],
        mode='lines',
        name='Média Móvel Exponencial',
        line=dict(width=2)
    ))

# Previsão futura
future_dates = [dates[-1] + timedelta(days=i) for i in range(1, FORECAST_DAYS + 1)]
future_x = np.arange(len(prices), len(prices) + FORECAST_DAYS)
future_forecast = forecast["forecast_array"]

fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_forecast[1:],
    mode='lines+markers',
    name=f'Previsão D+{FORECAST_DAYS}',
    line=dict(color='red', width=3, dash='dash'),
    marker=dict(size=8)
))

fig.update_layout(
    title="Preço do Bitcoin + Modelos de Aproximação",
    xaxis_title="Data",
    yaxis_title="Preço (USD)",
    hovermode='x unified',
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# INDICADORES TÉCNICOS
# =========================

st.markdown("## 📊 Indicadores Técnicos")

indicator_tabs = st.tabs(["Oscilador Estocástico", "RSI", "Métricas de Risco"])

with indicator_tabs[0]:
    if show_stochastic and "stochastic" in results:
        stoch = results["stochastic"]
        
        fig_stoch = go.Figure()
        
        stoch_dates = dates[14:]  # Ajustar para período de 14 dias
        
        fig_stoch.add_trace(go.Scatter(
            x=stoch_dates,
            y=stoch["k"],
            mode='lines',
            name='%K',
            line=dict(color='blue', width=2)
        ))
        
        fig_stoch.add_trace(go.Scatter(
            x=stoch_dates,
            y=stoch["d"],
            mode='lines',
            name='%D',
            line=dict(color='orange', width=2)
        ))
        
        # Linhas de referência
        fig_stoch.add_hline(y=80, line_dash="dash", line_color="red", 
                           annotation_text="Sobrecompra (80)")
        fig_stoch.add_hline(y=20, line_dash="dash", line_color="green", 
                           annotation_text="Sobrevenda (20)")
        
        fig_stoch.update_layout(
            title="Oscilador Estocástico",
            xaxis_title="Data",
            yaxis_title="Valor (%)",
            yaxis=dict(range=[0, 150]),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_stoch, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Estocástico %K Atual", f"{stoch['current_k']:.1f}")
        with col2:
            st.metric("Estocástico %D Atual", f"{stoch['current_d']:.1f}")

with indicator_tabs[1]:
    if show_rsi and "rsi" in results:
        rsi_data = results["rsi"]
        
        fig_rsi = go.Figure()
        
        rsi_dates = dates[15:]  # Ajustar para período do RSI
        
        fig_rsi.add_trace(go.Scatter(
            x=rsi_dates,
            y=rsi_data["values"][15:],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2),
            fill='tozeroy'
        ))
        
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                         annotation_text="Sobrecompra (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", 
                         annotation_text="Sobrevenda (30)")
        
        fig_rsi.update_layout(
            title="Índice de Força Relativa (RSI)",
            xaxis_title="Data",
            yaxis_title="RSI",
            yaxis=dict(range=[0, 100]),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_rsi, use_container_width=True)
        
        st.metric("RSI Atual", f"{rsi_data['current']:.1f}")

with indicator_tabs[2]:
    st.markdown("### Métricas de Risco Detalhadas")
    
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        st.metric("Volatilidade Anualizada", f"{risk_metrics['volatility']:.2%}")
        st.metric("Retorno Médio Diário", f"{risk_metrics['daily_return_mean']:.4%}")
        st.metric("VaR 95% (1 dia)", f"{risk_metrics['var_95']:.4%}")
    
    with risk_col2:
        st.metric("Desvio Padrão Diário", f"{risk_metrics['daily_return_std']:.4%}")
        st.metric("Drawdown Máximo", f"{risk_metrics['max_drawdown']:.2%}")
        st.metric("Índice Sharpe", f"{risk_metrics['sharpe_ratio']:.2f}")

# =========================
# PERFORMANCE HISTÓRICA
# =========================

st.markdown("## 📈 Performance Histórica do Agente")

perf_tabs = st.tabs(["Acurácia Geral", "Por Modelo", "Previsões Recentes"])

with perf_tabs[0]:
    accuracy = get_prediction_accuracy(days_back=90)
    
    if accuracy and accuracy['prediction_count'] > 0:
        acc_col1, acc_col2, acc_col3, acc_col4 = st.columns(4)
        
        with acc_col1:
            st.metric("Erro Médio", f"{accuracy['mean_error_pct']:.2f}%")
        with acc_col2:
            st.metric("Erro Mediano", f"{accuracy['median_error_pct']:.2f}%")
        with acc_col3:
            st.metric("Acurácia <5%", f"{accuracy['accuracy_95']:.1f}%")
        with acc_col4:
            st.metric("Total de Previsões", accuracy['prediction_count'])
        
        st.info(
            f"📊 Das {accuracy['prediction_count']} previsões analisadas, "
            f"{accuracy['accuracy_95']:.1f}% tiveram erro menor que 5% e "
            f"{accuracy['accuracy_90']:.1f}% tiveram erro menor que 10%."
        )
    else:
        st.info("📊 Ainda não há dados históricos suficientes. Execute o agente regularmente para acumular histórico.")

with perf_tabs[1]:
    model_perf = get_model_performance()
    
    if not model_perf.empty:
        st.dataframe(
            model_perf.style.format({
                'avg_confidence': '{:.4f}',
                'avg_error': '{:.2f}%',
                'best_error': '{:.2f}%',
                'worst_error': '{:.2f}%'
            }),
            use_container_width=True
        )
    else:
        st.info("📊 Nenhuma previsão validada ainda.")

with perf_tabs[2]:
    recent = get_recent_predictions(limit=10)
    
    if not recent.empty:
        # Formatar DataFrame
        recent['timestamp'] = pd.to_datetime(recent['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        recent['predicted_price'] = recent['predicted_price'].apply(lambda x: f"${x:,.0f}")
        recent['current_price'] = recent['current_price'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
        recent['actual_price'] = recent['actual_price'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "Pendente")
        recent['confidence'] = recent['confidence'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        recent['error_pct'] = recent['error_pct'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        
        st.dataframe(
            recent[['timestamp', 'model', 'predicted_price', 'actual_price', 'error_pct', 'recommendation']],
            use_container_width=True
        )
    else:
        st.info("📊 Nenhuma previsão registrada ainda.")

# =========================
# DOCUMENTAÇÃO
# =========================

with st.expander("📚 Como o Agente Funciona"):
    st.markdown("""
    ### Metodologia
    
    1. **Coleta de Dados**: Preços históricos do Bitcoin via API CoinGecko
    
    2. **Modelos Preditivos**:
       - **Linear**: Tendência simples de longo prazo
       - **Polinomial**: Captura curvaturas e mudanças de direção
       - **Polinomial + Seno**: Modela ciclos de mercado
       - **Média Móvel**: Suavização de volatilidade
       - **Média Móvel Exponencial**: Pesos maiores para dados recentes
    
    3. **Seleção de Modelo**: O modelo com maior R² é usado para previsão
    
    4. **Intervalo de Confiança**: Calculado baseado no erro histórico (95%)
    
    5. **Indicadores Técnicos**:
       - **Estocástico**: Momentum de curto prazo
         - %K < 20 → Sobrevenda (sinal de compra)
         - %K > 80 → Sobrecompra (sinal de venda)
       - **RSI**: Força relativa
         - RSI < 30 → Sobrevenda
         - RSI > 70 → Sobrecompra
    
    6. **Gestão de Risco**:
       - **Volatilidade**: Medida de incerteza do ativo
       - **Índice Sharpe**: Retorno ajustado ao risco
       - **Drawdown Máximo**: Maior perda histórica
       - **VaR 95%**: Valor em Risco
    
    7. **Recomendação**: Baseada em múltiplos fatores:
       - Tendência (inclinação linear)
       - Osciladores (Estocástico, RSI)
       - Volatilidade
       - Pontuação agregada determina: Acumular, Comprar, Manter, Reduzir ou Vender
    
    ### Limitações
    
    - ⚠️ Modelos baseados em dados históricos (o passado não garante o futuro)
    - ⚠️ Não considera eventos externos (regulação, hacks, notícias)
    - ⚠️ A volatilidade do Bitcoin pode invalidar previsões rapidamente
    - ⚠️ Recomendações são opinativas, não garantias
    
    ### Uso Recomendado
    
    - Use como **ferramenta de suporte**, não como único critério
    - Combine com análise fundamentalista
    - Considere seu perfil de risco
    - Diversifique seus investimentos
    - Nunca invista mais do que pode perder
    """)

# =========================
# RODAPÉ
# =========================

st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
    "₿ Agente Quantitativo Bitcoin<br>"
    "Desenvolvido por Eduardo Araujo © 2026<br>"
    "<small>Dados: API CoinGecko • Framework: Streamlit • Modelos: SciPy + NumPy</small>"
    "</div>",
    unsafe_allow_html=True
)
