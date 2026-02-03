import time
import numpy as np
from datetime import datetime
from bitcoin_agent import BitcoinAnalysisAgent
from memory import save_prediction

INTERVAL = 3600  # 1 hora em segundos
FORECAST_DAYS = 7

def run_agent():
    agent = BitcoinAnalysisAgent()

    while True:
        print("\nNovo ciclo:", datetime.now().isoformat())
        agent.fetch_bitcoin_data()
        results = agent.find_approximations()

        # escolhe melhor modelo por R²
        best = max(
            ((k, v) for k, v in results.items() if "r2" in v),
            key=lambda x: x[1]["r2"]
        )

        x_future = np.arange(len(agent.prices), len(agent.prices) + FORECAST_DAYS)
        price = float(best[1]["prediction"](x_future)[-1])

        save_prediction(best[0], price, best[1]["r2"])

        advice = agent.investment_advice()
        print("Modelo:", best[0])
        print("Previsão:", price)
        print("Recomendação:", advice["recommendation"])

        time.sleep(INTERVAL)

if __name__ == "__main__":
    run_agent()