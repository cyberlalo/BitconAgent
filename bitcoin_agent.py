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
        params = {"vs_currency": "usd", "days": days, "interval": "daily"}
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()["prices"]

        self.prices = np.array([p[1] for p in data])
        self.dates = [datetime.fromtimestamp(p[0]/1000) for p in data]
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
            "prediction": lambda t: slope * t + intercept,
            "r2": r**2
        }

        # 2️⃣ Exponencial
        def exp(x, a, b):
            return a * np.exp(b * x)

        y_shift = y - y.min() + 1
        popt, _ = curve_fit(exp, x, y_shift, maxfev=10000)
        results["exponential"] = {
            "prediction": lambda t: exp(t, *popt) + y.min() - 1,
            "r2": 1 - np.sum((y - (exp(x, *popt)+y.min()-1))**2)/np.sum((y-np.mean(y))**2)
        }

        # 3️⃣ Polinomial + seno
        period = 45
        def poly_sine(x, a0, a1, a2, a3, A, phi):
            poly = a0 + a1*x + a2*x**2 + a3*x**3
            return poly + A*np.sin(2*np.pi*x/period + phi)

        p0 = [y.mean(), 0, 0, 0, 1000, 0]
        popt, _ = curve_fit(poly_sine, x, y, p0=p0, maxfev=20000)

        results["poly_sine"] = {
            "prediction": lambda t: poly_sine(t, *popt),
            "r2": 1 - np.sum((y - poly_sine(x,*popt))**2)/np.sum((y-np.mean(y))**2)
        }

        # 4️⃣ Média móvel
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
        period = 14

        for i in range(period, len(y)):
            low = y[i-period:i].min()
            high = y[i-period:i].max()
            k = 100 * (y[i] - low) / (high - low) if high != low else 50
            k_values.append(k)

        d_values = pd.Series(k_values).rolling(3).mean().values

        results["stochastic"] = {
            "k": k_values,
            "d": d_values,
            "current_k": k_values[-1],
            "current_d": d_values[-1]
        }

        self.results = results
        return results

    # =========================
    # RECOMENDAÇÃO
    # =========================

    def investment_advice(self):
        k = self.results["stochastic"]["current_k"]

        if k < 20:
            rec = "Accumulate"
        elif k > 80:
            rec = "Sell"
        else:
            rec = "Hold"

        return {
            "recommendation": rec,
            "stochastic_k": k
        }
