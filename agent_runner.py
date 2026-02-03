"""
Autonomous Bitcoin Analysis Agent
Runs periodically, fetches data, generates predictions,
stores them in memory, and logs decisions.
"""

import time
from datetime import datetime

from bitcoin_agent import BitcoinAnalysisAgent
from memory import save_prediction

# =========================
# CONFIGURAÇÃO DO AGENTE
# =========================

ANALYSIS_DAYS = 180          # janela histórica
FORECAST_DAYS = 7            # previsão futura (D+7)
INTERVAL_SECONDS = 60 * 60   # 1 hora entre ciclos

# =========================
# LOOP AUTÔNOMO
# =========================

def run_agent():
    print("=" * 60)
    print("Starting Autonomous Bitcoin Agent")
    print("=" * 60)

    # verbose=False → logs limpos
    agent = BitcoinAnalysisAgent(verbose=False)

    while True:
        cycle_start = datetime.now()
        print(f"\n[{cycle_start.isoformat()}] New cycle started")

        try:
            # 1. Coleta de dados
            data = agent.fetch_bitcoin_data(days=ANALYSIS_DAYS)
            if data is None:
                print("Failed to fetch data. Skipping cycle.")
                time.sleep(INTERVAL_SECONDS)
                continue

            # 2. Análise
            agent.find_approximations()

            # 3. Previsão futura (AGORA PELO AGENTE)
            forecast = agent.forecast(days_ahead=FORECAST_DAYS)

            if forecast is None:
                print("No valid forecast generated.")
            else:
                # 4. Persistência (memória)
                save_prediction(
                    model=forecast["model"],
                    price=forecast["predicted_price"],
                    confidence=forecast["confidence"]
                )

                print(
                    f"Forecast saved | "
                    f"Model: {forecast['model']} | "
                    f"Price (D+{FORECAST_DAYS}): "
                    f"${forecast['predicted_price']:,.2f} | "
                    f"Confidence: {forecast['confidence']:.2f}"
                )

            # 5. Log de status
            advice = agent.investment_advice()
            if advice:
                print(f"Agent recommendation: {advice['recommendation']}")

        except Exception as e:
            print("Agent error:", e)

        finally:
            minutes = max(INTERVAL_SECONDS // 60, 1)
            print(f"Cycle completed. Sleeping for {minutes} minutes.")
            time.sleep(INTERVAL_SECONDS)

# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    run_agent()
