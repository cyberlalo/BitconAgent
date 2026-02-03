# agent_runner.py
"""
Autonomous Bitcoin Analysis Agent
Runs once per execution (cron-friendly),
fetches data, generates forecast,
computes % variation, and stores memory.
"""

from datetime import datetime
import numpy as np

from bitcoin_agent import BitcoinAnalysisAgent
from memory import save_prediction

# =========================
# CONFIGURAÇÃO
# =========================

ANALYSIS_DAYS = 180     # janela histórica
FORECAST_DAYS = 7       # horizonte da previsão (D+7)

# =========================
# EXECUÇÃO ÚNICA DO AGENTE
# =========================

def run_agent_once():
    now = datetime.now().isoformat()
    print("=" * 60)
    print(f"Agent run started: {now}")
    print("=" * 60)

    agent = BitcoinAnalysisAgent()

    # 1️⃣ Coleta de dados
    agent.fetch_bitcoin_data(days=ANALYSIS_DAYS)

    # 2️⃣ Análise
    results = agent.find_approximations()

    # 3️⃣ Seleção do melhor modelo (maior R²)
    best_model_name, best_model = max(
        ((k, v) for k, v in results.items() if "r2" in v),
        key=lambda x: x[1]["r2"]
    )

    # 4️⃣ Forecast
    current_price = agent.prices[-1]
    x_start = len(agent.prices)
    x_future = np.arange(x_start, x_start + FORECAST_DAYS)

    forecast_price = float(best_model["prediction"](x_future)[-1])

    variation_pct = (
        (forecast_price - current_price) / current_price
    ) * 100

    # 5️⃣ Persistência em memória
    save_prediction(
        model=best_model_name,
        price=forecast_price,
        confidence=best_model["r2"]
    )

    # 6️⃣ Recomendação (timing)
    advice = agent.investment_advice()

    # 7️⃣ Log claro
    print(f"Model: {best_model_name}")
    print(f"Current price: {current_price:,.2f} USD")
    print(f"Forecast (D+{FORECAST_DAYS}): {forecast_price:,.2f} USD")
    print(f"Δ Forecast: {variation_pct:+.2f}%")
    print(f"Advice: {advice['recommendation']} (Stochastic %K = {advice['stochastic_k']:.1f})")

    print("=" * 60)
    print("Agent run completed")
    print("=" * 60)


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    run_agent_once()
