import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
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
        ["Linear", "Polynomial", "Polynomial+Sine", "Exponential", "Moving Average"],
        default=["Linear", "Polynomial", "Exponential"]
    )
    
    atualizar = st.button("Update Analysis", type="primary")
    
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("""
    This agent uses:
    - CoinGecko API data
    - Mathematical regression
    - Trend analysis
    - Cycle detection
    """)

@st.cache_resource
def get_agent():
    return BitcoinAnalysisAgent()

agent = get_agent()

main_container = st.container()

with main_container:
    if atualizar or 'data' not in st.session_state:
        with st.spinner("Fetching Bitcoin data..."):
            data = agent.fetch_bitcoin_data(days=dias)
            
            if data is not None:
                st.session_state.data = data
                st.session_state.agent = agent
                
                with st.spinner("Analyzing patterns..."):
                    results = agent.find_approximations()
                    st.session_state.results = results
    
    if 'data' in st.session_state:
        data = st.session_state.data
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = float(data['price'].iloc[-1])
            initial_price = float(data['price'].iloc[0])
            delta = ((current_price / initial_price - 1) * 100)
            st.metric(
                "Current Price",
                f"${current_price:,.2f}",
                f"{delta:+.2f}%"
            )
        
        with col2:
            if 'volatility' in st.session_state.get('results', {}):
                vol = st.session_state.results['volatility']['daily_volatility']
                st.metric("Daily Volatility", f"{vol:.2f}%")
            else:
                st.metric("Daily Volatility", "N/A")
        
        with col3:
            if 'linear' in st.session_state.get('results', {}):
                slope = st.session_state.results['linear']['params']['slope']
                st.metric("Daily Trend", f"${slope:.2f}")
            else:
                st.metric("Daily Trend", "N/A")
        
        with col4:
            if 'stochastic' in st.session_state.get('results', {}):
                k = st.session_state.results['stochastic']['current_k']
                status = "Oversold" if k < 20 else "Overbought" if k > 80 else "Neutral"
                st.metric("Stochastic %K", f"{k:.1f} ({status})")
            else:
                st.metric("Stochastic %K", "N/A")
        
        st.subheader("Graphical Analysis")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data['price'],
            mode='lines',
            name='Real Price',
            line=dict(color='blue', width=1),
            opacity=0.7
        ))
        
        colors = ['red', 'green', 'purple', 'orange', 'cyan']
        color_idx = 0
        
        if 'Linear' in modelos and 'linear' in st.session_state.get('results', {}):
            pred = st.session_state.results['linear']['prediction'](np.arange(len(data)))
            fig.add_trace(go.Scatter(
                x=data['date'],
                y=pred,
                mode='lines',
                name=f'Linear (R²={st.session_state.results["linear"]["r_squared"]:.3f})',
                line=dict(color=colors[color_idx], dash='dash')
            ))
            color_idx += 1
        
        if 'Polynomial' in modelos and 'polynomial' in st.session_state.get('results', {}):
            pred = st.session_state.results['polynomial']['prediction'](np.arange(len(data)))
            fig.add_trace(go.Scatter(
                x=data['date'],
                y=pred,
                mode='lines',
                name=f'Polynomial (R²={st.session_state.results["polynomial"]["r_squared"]:.3f})',
                line=dict(color=colors[color_idx], dash='dot')
            ))
            color_idx += 1
        
        if 'Exponential' in modelos and 'exponential' in st.session_state.get('results', {}):
            pred = st.session_state.results['exponential']['prediction'](np.arange(len(data)))
            fig.add_trace(go.Scatter(
                x=data['date'],
                y=pred,
                mode='lines',
                name=f'Exponential (R²={st.session_state.results["exponential"]["r_squared"]:.3f})',
                line=dict(color=colors[color_idx], dash='dashdot')
            ))
            color_idx += 1
        
        if 'Polynomial+Sine' in modelos and 'poly_sine' in st.session_state.get('results', {}):
            pred = st.session_state.results['poly_sine']['prediction'](np.arange(len(data)))
            fig.add_trace(go.Scatter(
                x=data['date'],
                y=pred,
                mode='lines',
                name=f'Poly+Sine (R²={st.session_state.results["poly_sine"]["r2"]:.3f})',
                line=dict(color=colors[color_idx], dash='longdash')
            ))
            color_idx += 1
        
        if 'Moving Average' in modelos and 'moving_average' in st.session_state.get('results', {}):
            ma_values = st.session_state.results['moving_average']['values']
            window = st.session_state.results['moving_average']['window']
            fig.add_trace(go.Scatter(
                x=data['date'],
                y=ma_values,
                mode='lines',
                name=f'MA({window} days)',
                line=dict(color=colors[color_idx], width=2)
            ))
        
        fig.update_layout(
            title="Bitcoin: Price vs Approximation Models",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Model Results")
        
        resultados_df = []
        if 'results' in st.session_state:
            for key, value in st.session_state.results.items():
                if key in ['linear', 'polynomial', 'exponential', 'poly_sine']:
                    r2 = value.get('r_squared', value.get('r2', 0))
                    resultados_df.append({
                        'Model': key.upper(),
                        'Function': value.get('function', 'N/A')[:80] + '...' if len(value.get('function', '')) > 80 else value.get('function', 'N/A'),
                        'R²': f"{r2:.4f}",
                        'Fit': 'Excellent' if r2 > 0.8 
                              else 'Good' if r2 > 0.6 
                              else 'Moderate' if r2 > 0.4 
                              else 'Low'
                    })
        
        if resultados_df:
            st.dataframe(pd.DataFrame(resultados_df), use_container_width=True)
        else:
            st.info("No model results available. Click 'Update Analysis'.")
        
        st.subheader("Insights and Recommendations")
        
        col_ins1, col_ins2 = st.columns(2)
        
        with col_ins1:
            st.info("**Technical Analysis**")
            
            if 'stochastic' in st.session_state.get('results', {}):
                k = st.session_state.results['stochastic']['current_k']
                if k > 80:
                    st.warning("**Overbought**: Consider taking profits")
                elif k < 20:
                    st.success("**Oversold**: Buying opportunity")
                else:
                    st.info("**Neutral**: Wait for clearer signals")
            else:
                st.info("Stochastic data not available")
        
        with col_ins2:
            st.info("**Risk and Volatility**")
            
            if 'volatility' in st.session_state.get('results', {}):
                vol = st.session_state.results['volatility']['daily_volatility']
                if vol > 5:
                    st.error(f"High risk - Volatility: {vol:.1f}% daily")
                elif vol > 2:
                    st.warning(f"Moderate risk - Volatility: {vol:.1f}% daily")
                else:
                    st.success(f"Low risk - Volatility: {vol:.1f}% daily")
            else:
                st.info("Volatility data not available")
        
        st.download_button(
            label="Export Data (CSV)",
            data=data.to_csv(index=False).encode('utf-8'),
            file_name=f"bitcoin_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    else:
        st.info("Click 'Update Analysis' button to start the analysis")
        st.markdown("""
        This application will:
        1. Fetch Bitcoin price data from CoinGecko API
        2. Apply mathematical models to find patterns
        3. Generate insights and predictions
        4. Provide downloadable results
        """)
        st.markdown("---")
    
    footer_col1, footer_col2, footer_col3 = st.columns([1, 2, 1])
    
    with footer_col1:
        st.markdown("**Version:** 1.0.0")
    
    with footer_col2:
        st.markdown(
            "<div style='text-align: center;'>"
            "**Bitcoin Agent by Eduardo Araujo © 2026**<br>"
            "<small>AI-powered Bitcoin analysis and prediction</small>"
            "</div>",
            unsafe_allow_html=True
        )
    
    with footer_col3:
        st.markdown("**Updated:** " + datetime.now().strftime("%Y-%m-%d"))