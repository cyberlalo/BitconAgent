import requests
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit
from scipy import stats

class BitcoinAnalysisAgent:
    def __init__(self):
        self.prices = None
        self.dates = None
        self.results = {}

    # =========================
    # DATA
    # =========================

    def fetch_bitcoin_data(self, days=180):
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        params = {
            "vs_currency": "usd",
            "days": days,
            "interval": "daily"
        }

        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()["prices"]

        self.prices = np.array([p[1] for p in data])
        self.dates = [datetime.fromtimestamp(p[0] / 1000) for p in data]

        return self.prices

    # =========================
    # MODELOS
    # =========================

    def find_approximations(self):
        x = np.arange(len(self.prices))
        y = self.prices
        results = {}

        # 1️⃣ Linear
        slope, intercept, r, _, _ = stats.linregress(x, y)
        results["linear"] = {
            "prediction": lambda t, a=slope, b=intercept: a * t + b,
            "r2": r ** 2
        }

        # 2️⃣ Polinomial + seno
        period = 45

        def poly_sine(x, a0, a1, a2, a3, A, phi):
            poly = a0 + a1 * x + a2 * x**2 + a3 * x**3
            return poly + A * np.sin(2 * np.pi * x / period + phi)

        p0 = [y.mean(), 0, 0, 0, 1000, 0]
        popt, _ = curve_fit(poly_sine, x, y, p0=p0, maxfev=20000)

        y_pred = poly_sine(x, *popt)
        r2_poly_sine = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

        results["poly_sine"] = {
            "prediction": lambda t, p=popt: poly_sine(np.array(t), *p),
            "r2": r2_poly_sine
        }

        # 3️⃣ Média móvel
        window = 30
        ma = pd.Series(y).rolling(window).mean().values

        results["moving_average"] = {
            "values": ma,
            "window": window
        }

        # =========================
        # OSCILADOR ESTOCÁSTICO
        # =========================

        k_values = []
        period_stoch = 14

        for i in range(period_stoch, len(y)):
            low = y[i - period_stoch : i].min()
            high = y[i - period_stoch : i].max()

            if high != low:
                k = 100 * (y[i] - low) / (high - low)
            else:
                k = 50

            k_values.append(k)

        d_values = pd.Series(k_values).rolling(3).mean().values

        results["stochastic"] = {
            "k": k_values,
            "d": d_values,
            "current_k": float(k_values[-1]),
            "current_d": float(d_values[-1])
        }

        self.results = results
        return results

    # =========================
    # RECOMENDAÇÃO
    # =========================

    def investment_advice(self):
        k = self.results["stochastic"]["current_k"]

        if k < 20:
            rec = "Acumular"
        elif k > 80:
            rec = "Vender"
        else:
            rec = "Manter"

        return {
            "recommendation": rec,
            "stochastic_k": k
        }
