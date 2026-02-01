import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from bitcoin_agent import BitcoinAnalysisAgent

st.set_page_config(
    page_title="Bitcoin Analysis Agent",
    page_icon="₿",
    layout="wide"
)

st.title("Bitcoin Analysis Agent")
st.markdown("Real-time analysis with mathematical approximation models")

with st.sidebar:
    st.header("Settings")
    
    dias = st.slider(
        "Analysis period (days)",
        min_value=30,
        max_value=365,
        value=180,
        help="Historical days for analysis"
    )
    
    modelos = st.multiselect(
        "Models for analysis",
        ["Linear", "Polynomial", "Polynomial+Sine", "Exponential", "Moving Average", "Stochastic Oscillator"],
        default=["Linear", "Polynomial", "Exponential", "Stochastic Oscillator"]
    )
    
    atualizar = st.button("Update Analysis", type="primary")
    
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("""
    This agent uses:
    - CoinGecko API data
    - Mathematical regression
    - Trend analysis
    - Stochastic Oscillator
    - Cycle detection
    """)

# Inicializar agente
if 'agent' not in st.session_state:
    st.session_state.agent = BitcoinAnalysisAgent()

agent = st.session_state.agent

main_container = st.container()

with main_container:
    if atualizar or 'data' not in st.session_state:
        with st.spinner("Fetching Bitcoin data..."):
            data = agent.fetch_bitcoin_data(days=dias)
            
            if data is not None:
                st.session_state.data = data
                
                with st.spinner("Analyzing patterns..."):
                    try:
                        results = agent.find_approximations()
                        st.session_state.results = results
                        st.session_state.analysis_done = True
                        st.success("Analysis completed successfully!")
                    except Exception as e:
                        st.error(f"Error in analysis: {str(e)}")
                        st.session_state.analysis_done = False
    
    if 'data' in st.session_state and st.session_state.get('analysis_done', False):
        data = st.session_state.data
        results = st.session_state.results
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            try:
                current_price = float(data['price'].iloc[-1])
                initial_price = float(data['price'].iloc[0])
                delta = ((current_price / initial_price - 1) * 100)
                st.metric(
                    "Current Price",
                    f"${current_price:,.2f}",
                    f"{delta:+.2f}%"
                )
            except:
                st.metric("Current Price", "N/A")
        
        with col2:
            if results and 'volatility' in results:
                vol = results['volatility']['daily_volatility']
                st.metric("Daily Volatility", f"{vol:.2f}%")
            else:
                st.metric("Daily Volatility", "N/A")
        
        with col3:
            if results and 'linear' in results:
                slope = results['linear']['params']['slope']
                st.metric("Daily Trend", f"${slope:.2f}")
            else:
                st.metric("Daily Trend", "N/A")
        
        with col4:
            if results and 'stochastic' in results:
                k = results['stochastic']['current_k']
                status = "Oversold" if k < 20 else "Overbought" if k > 80 else "Neutral"
                st.metric("Stochastic %K", f"{k:.1f} ({status})")
            else:
                st.metric("Stochastic %K", "N/A")
        
        # Gráfico 1: Preço vs Modelos
        if any(model in modelos for model in ["Linear", "Polynomial", "Polynomial+Sine", "Exponential", "Moving Average"]):
            st.subheader("Price Analysis")
            
            fig1 = go.Figure()
            
            # Preço real
            fig1.add_trace(go.Scatter(
                x=data['date'],
                y=data['price'],
                mode='lines',
                name='Bitcoin Price',
                line=dict(color='blue', width=2),
                opacity=0.7
            ))
            
            colors = ['red', 'green', 'purple', 'orange', 'cyan']
            color_idx = 0
            
            if 'Linear' in modelos and results and 'linear' in results:
                pred = results['linear']['prediction'](np.arange(len(data)))
                fig1.add_trace(go.Scatter(
                    x=data['date'],
                    y=pred,
                    mode='lines',
                    name=f'Linear (R²={results["linear"]["r_squared"]:.3f})',
                    line=dict(color=colors[color_idx], dash='dash', width=2)
                ))
                color_idx += 1
            
            if 'Polynomial' in modelos and results and 'polynomial' in results:
                pred = results['polynomial']['prediction'](np.arange(len(data)))
                fig1.add_trace(go.Scatter(
                    x=data['date'],
                    y=pred,
                    mode='lines',
                    name=f'Polynomial (R²={results["polynomial"]["r_squared"]:.3f})',
                    line=dict(color=colors[color_idx], dash='dot', width=2)
                ))
                color_idx += 1
            
            if 'Exponential' in modelos and results and 'exponential' in results:
                pred = results['exponential']['prediction'](np.arange(len(data)))
                fig1.add_trace(go.Scatter(
                    x=data['date'],
                    y=pred,
                    mode='lines',
                    name=f'Exponential (R²={results["exponential"]["r_squared"]:.3f})',
                    line=dict(color=colors[color_idx], dash='dashdot', width=2)
                ))
                color_idx += 1
            
            if 'Polynomial+Sine' in modelos and results and 'poly_sine' in results:
                pred = results['poly_sine']['prediction'](np.arange(len(data)))
                fig1.add_trace(go.Scatter(
                    x=data['date'],
                    y=pred,
                    mode='lines',
                    name=f'Poly+Sine (R²={results["poly_sine"]["r2"]:.3f})',
                    line=dict(color=colors[color_idx], dash='longdash', width=2)
                ))
                color_idx += 1
            
            if 'Moving Average' in modelos and results and 'moving_average' in results:
                ma_values = results['moving_average']['values']
                window = results['moving_average']['window']
                # Filter out NaN values
                valid_mask = ~np.isnan(ma_values)
                if np.any(valid_mask):
                    fig1.add_trace(go.Scatter(
                        x=data['date'][valid_mask],
                        y=ma_values[valid_mask],
                        mode='lines',
                        name=f'MA({window} days)',
                        line=dict(color=colors[color_idx], width=2)
                    ))
            
            fig1.update_layout(
                title="Bitcoin: Price vs Approximation Models",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                hovermode='x unified',
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig1, width='stretch')
        
        # Gráfico 2: Oscilador Estocástico
        if 'Stochastic Oscillator' in modelos and results and 'stochastic' in results:
            st.subheader("Stochastic Oscillator Analysis")
            
            stoch = results['stochastic']
            stoch_dates = stoch.get('dates', [])
            stoch_k = stoch.get('k', [])
            stoch_d = stoch.get('d', [])
            
            if stoch_dates and stoch_k and stoch_d and len(stoch_dates) == len(stoch_k) == len(stoch_d):
                fig2 = go.Figure()
                
                # %K line (blue)
                fig2.add_trace(go.Scatter(
                    x=stoch_dates,
                    y=stoch_k,
                    mode='lines',
                    name='%K (Fast)',
                    line=dict(color='blue', width=2)
                ))
                
                # %D line (red)
                fig2.add_trace(go.Scatter(
                    x=stoch_dates,
                    y=stoch_d,
                    mode='lines',
                    name='%D (Slow)',
                    line=dict(color='red', width=2)
                ))
                
                # Overbought/Oversold zones
                fig2.add_hrect(y0=80, y1=100, 
                             fillcolor="rgba(255,0,0,0.1)", 
                             line_width=0)
                
                fig2.add_hrect(y0=0, y1=20, 
                             fillcolor="rgba(0,255,0,0.1)", 
                             line_width=0)
                
                # Overbought line
                fig2.add_hline(y=80, line_dash="dash", 
                             line_color="gray")
                
                # Oversold line
                fig2.add_hline(y=20, line_dash="dash", 
                             line_color="gray")
                
                # Current values markers
                current_k = stoch.get('current_k', 0)
                current_d = stoch.get('current_d', 0)
                last_date = stoch_dates[-1] if stoch_dates else data['date'].iloc[-1]
                
                fig2.add_trace(go.Scatter(
                    x=[last_date],
                    y=[current_k],
                    mode='markers',
                    name=f'Current %K: {current_k:.1f}',
                    marker=dict(color='blue', size=10, symbol='circle')
                ))
                
                fig2.add_trace(go.Scatter(
                    x=[last_date],
                    y=[current_d],
                    mode='markers',
                    name=f'Current %D: {current_d:.1f}',
                    marker=dict(color='red', size=10, symbol='circle')
                ))
                
                # Market condition annotation
                condition = stoch.get('condition', 'neutral')
                signal = stoch.get('signal', 'no_crossover')
                
                annotations = []
                
                if condition == 'overbought':
                    annotations.append(dict(
                        x=0.02, y=0.95,
                        xref="paper", yref="paper",
                        text="<b>OVERBOUGHT</b>",
                        showarrow=False,
                        font=dict(size=14, color="red"),
                        bgcolor="white",
                        bordercolor="red",
                        borderwidth=1,
                        borderpad=4
                    ))
                elif condition == 'oversold':
                    annotations.append(dict(
                        x=0.02, y=0.95,
                        xref="paper", yref="paper",
                        text="<b>OVERSOLD</b>",
                        showarrow=False,
                        font=dict(size=14, color="green"),
                        bgcolor="white",
                        bordercolor="green",
                        borderwidth=1,
                        borderpad=4
                    ))
                
                if signal == 'bullish_crossover':
                    annotations.append(dict(
                        x=0.02, y=0.85,
                        xref="paper", yref="paper",
                        text="<b>BULLISH CROSSOVER</b>",
                        showarrow=False,
                        font=dict(size=12, color="green"),
                        bgcolor="white",
                        bordercolor="green",
                        borderwidth=1,
                        borderpad=4
                    ))
                elif signal == 'bearish_crossover':
                    annotations.append(dict(
                        x=0.02, y=0.85,
                        xref="paper", yref="paper",
                        text="<b>BEARISH CROSSOVER</b>",
                        showarrow=False,
                        font=dict(size=12, color="red"),
                        bgcolor="white",
                        bordercolor="red",
                        borderwidth=1,
                        borderpad=4
                    ))
                
                fig2.update_layout(
                    title=f"Stochastic Oscillator ({stoch.get('period', 14)}-day period)",
                    xaxis_title="Date",
                    yaxis_title="Stochastic Value",
                    yaxis_range=[-5, 105],
                    hovermode='x unified',
                    template='plotly_white',
                    height=500,
                    annotations=annotations
                )
                
                st.plotly_chart(fig2, width='stretch')
                
                # Stochastic insights
                stoch_col1, stoch_col2, stoch_col3 = st.columns(3)
                
                with stoch_col1:
                    st.info("**Current Values**")
                    st.metric("%K (Fast)", f"{current_k:.1f}")
                    st.metric("%D (Slow)", f"{current_d:.1f}")
                
                with stoch_col2:
                    st.info("**Market Condition**")
                    if condition == 'overbought':
                        st.error("**OVERBOUGHT**")
                        st.write("Consider taking profits")
                    elif condition == 'oversold':
                        st.success("**OVERSOLD**")
                        st.write("Potential buying opportunity")
                    else:
                        st.info("**NEUTRAL**")
                        st.write("Market in normal range")
                
                with stoch_col3:
                    st.info("**Signal Analysis**")
                    if signal == 'bullish_crossover':
                        st.success("**BULLISH CROSSOVER**")
                        st.write("%K crossed above %D")
                    elif signal == 'bearish_crossover':
                        st.warning("**BEARISH CROSSOVER**")
                        st.write("%K crossed below %D")
                    else:
                        st.info("**NO CROSSOVER**")
                        st.write("Wait for clearer signal")
            else:
                st.warning("Stochastic oscillator data not available or incomplete")
        
        st.subheader("Model Results")
        
        resultados_df = []
        if results:
            for key, value in results.items():
                if key in ['linear', 'polynomial', 'exponential', 'poly_sine']:
                    r2 = value.get('r_squared', value.get('r2', 0))
                    resultados_df.append({
                        'Model': key.upper(),
                        'Function': str(value.get('function', 'N/A'))[:80] + '...' if len(str(value.get('function', ''))) > 80 else str(value.get('function', 'N/A')),
                        'R²': f"{r2:.4f}",
                        'Fit': 'Excellent' if r2 > 0.8 
                              else 'Good' if r2 > 0.6 
                              else 'Moderate' if r2 > 0.4 
                              else 'Low'
                    })
        
        if resultados_df:
            st.dataframe(pd.DataFrame(resultados_df), width='stretch')
        else:
            st.info("No model results available.")
        
        st.subheader("Insights and Recommendations")
        
        col_ins1, col_ins2 = st.columns(2)
        
        with col_ins1:
            st.info("**Technical Analysis**")
            
            if results and 'stochastic' in results:
                stoch = results['stochastic']
                condition = stoch.get('condition', 'neutral')
                
                if condition == 'overbought':
                    st.warning("**Overbought**: Consider taking profits")
                elif condition == 'oversold':
                    st.success("**Oversold**: Buying opportunity")
                else:
                    st.info("**Neutral**: Wait for clearer signals")
                
                # Crossover analysis
                signal = stoch.get('signal', 'no_crossover')
                if signal == 'bullish_crossover':
                    st.success("**Bullish Signal**: %K crossed above %D")
                elif signal == 'bearish_crossover':
                    st.warning("**Bearish Signal**: %K crossed below %D")
            else:
                st.info("Stochastic data not available")
        
        with col_ins2:
            st.info("**Risk and Volatility**")
            
            if results and 'volatility' in results:
                vol = results['volatility']['daily_volatility']
                if vol > 5:
                    st.error(f"High risk - Volatility: {vol:.1f}% daily")
                elif vol > 2:
                    st.warning(f"Moderate risk - Volatility: {vol:.1f}% daily")
                else:
                    st.success(f"Low risk - Volatility: {vol:.1f}% daily")
            else:
                st.info("Volatility data not available")
        
        st.subheader("Investment Advice")
        
        # Botão para gerar conselho
        if st.button("Generate Investment Advice", type="secondary"):
            try:
                # O agente já tem os dados e resultados
                advice = agent.investment_advice()
                
                if advice:
                    # Mostrar recomendação principal
                    col_rec1, col_rec2, col_rec3 = st.columns(3)
                    
                    with col_rec1:
                        st.metric(
                            "Recommendation",
                            advice['recommendation'],
                            help="Suggested action based on analysis"
                        )
                    
                    with col_rec2:
                        risk_color = {
                            'Very High': 'red',
                            'High': 'orange',
                            'Medium': 'yellow',
                            'Low': 'green'
                        }.get(advice.get('risk_level', 'Medium'), 'gray')
                        
                        st.markdown(f"""
                        <div style='padding: 10px; border-radius: 5px; border: 1px solid {risk_color};'>
                        <strong>Risk Level:</strong> {advice.get('risk_level', 'N/A')}<br>
                        <strong>Confidence:</strong> {int(advice.get('confidence', 0)*100)}%
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_rec3:
                        st.metric(
                            "Time Horizon",
                            advice.get('time_horizon', 'N/A'),
                            help="Suggested investment timeframe"
                        )
                    
                    # Explicação detalhada
                    with st.expander("Detailed Analysis", expanded=True):
                        st.write("**Key Factors Considered:**")
                        key_points = advice.get('key_points', [])
                        if key_points:
                            for point in key_points:
                                st.write(f"• {point}")
                        else:
                            st.write("No key points available")
                        
                        st.write("\n**Final Assessment:**")
                        st.info(advice.get('summary', 'No summary available'))
                        
                        # Adicionar disclaimer
                        st.warning("""
                        **Disclaimer:** This is automated analysis, not financial advice. 
                        Cryptocurrency investments are high risk. Always do your own research 
                        and consider consulting with a financial advisor.
                        """)
                else:
                    st.error("Could not generate investment advice. Please run analysis again.")
                    
            except Exception as e:
                st.error(f"Error generating investment advice: {str(e)}")
                st.info("Make sure you have run the analysis first by clicking 'Update Analysis'")
        
        st.download_button(
            label="Export Data (CSV)",
            data=data.to_csv(index=False).encode('utf-8'),
            file_name=f"bitcoin_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    elif 'data' in st.session_state:
        st.warning("Analysis not completed. Please click 'Update Analysis' again.")
        st.session_state.analysis_done = False
        
    else:
        st.info("Click 'Update Analysis' button to start the analysis")
        st.markdown("""
        This application will:
        1. Fetch Bitcoin price data from CoinGecko API
        2. Apply mathematical models to find patterns
        3. Analyze Stochastic Oscillator
        4. Generate insights and predictions
        5. Provide investment advice
        6. Provide downloadable results
        """)

# Rodapé
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>"
    "Bitcoin Agent by Eduardo Araujo © 2026<br>"
    "<small>AI-powered Bitcoin analysis and prediction</small>"
    "</div>",
    unsafe_allow_html=True
)