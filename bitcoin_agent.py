"""
Bitcoin Analysis Agent
Historical data analysis and mathematical approximation functions
"""

import requests
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class BitcoinAnalysisAgent:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.data = None
        self.prices = None
        self.dates = None
        self.results = None

        if self.verbose:
            print("Bitcoin Analysis Agent initialized")

    # --------------------------------------------------
    # DATA FETCHING
    # --------------------------------------------------
    def fetch_bitcoin_data(self, days=365):
        if self.verbose:
            print(f"Fetching data for the last {days} days...")

        try:
            url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            params = {
                "vs_currency": "usd",
                "days": days,
                "interval": "daily"
            }

            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            prices = [p[1] for p in data["prices"]]
            timestamps = [p[0] for p in data["prices"]]
            dates = [datetime.fromtimestamp(ts / 1000) for ts in timestamps]

            self.data = pd.DataFrame({"date": dates, "price": prices})
            self.prices = np.array(prices)
            self.dates = dates

            return self.data

        except Exception as e:
            if self.verbose:
                print(f"Error fetching data: {e}")
            return None

    # --------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------
    def find_approximations(self):
        if self.prices is None:
            return None

        x = np.arange(len(self.prices))
        y = self.prices
        results = {}

        # Linear regression
        slope, intercept, r, _, _ = stats.linregress(x, y)
        linear_func = lambda t: slope * t + intercept
        results["linear"] = {
            "prediction": linear_func,
            "r_squared": r**2,
            "params": {"slope": slope, "intercept": intercept}
        }

        # Polynomial (degree 2)
        coeffs = np.polyfit(x, y, 2)
        poly = np.poly1d(coeffs)
        y_pred = poly(x)
        r2_poly = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        results["polynomial"] = {
            "prediction": poly,
            "r_squared": r2_poly,
            "params": {"coefficients": coeffs}
        }

        # Exponential
        def exp_func(t, a, b):
            return a * np.exp(b * t)

        y_adj = y - np.min(y) + 1
        popt, _ = curve_fit(exp_func, x, y_adj, p0=[y_adj[0], 0.001], maxfev=10000)
        exp_pred = lambda t: exp_func(t, *popt) + np.min(y) - 1
        y_exp = exp_pred(x)
        r2_exp = 1 - np.sum((y - y_exp)**2) / np.sum((y - np.mean(y))**2)

        results["exponential"] = {
            "prediction": exp_pred,
            "r_squared": r2_exp,
            "params": {"a": popt[0], "b": popt[1]}
        }

        # Volatility
        returns = np.diff(y) / y[:-1] * 100
        results["volatility"] = {
            "daily_volatility": float(np.std(returns)),
            "annual_volatility": float(np.std(returns) * np.sqrt(365))
        }

        self.results = results
        return results

    # --------------------------------------------------
    # FORECAST (NOVO — IMPORTANTE)
    # --------------------------------------------------
    def forecast(self, days_ahead=7):
        """
        Predict future price using the best available model
        """
        if self.results is None or self.prices is None:
            return None

        best_model = None
        best_r2 = -1

        for name, model in self.results.items():
            r2 = model.get("r_squared", -1)
            if r2 > best_r2 and "prediction" in model:
                best_model = name
                best_r2 = r2

        if best_model is None:
            return None

        n = len(self.prices)
        x_future = np.arange(n, n + days_ahead)
        prediction_func = self.results[best_model]["prediction"]
        future_prices = prediction_func(x_future)

        return {
            "model": best_model,
            "predicted_price": float(future_prices[-1]),
            "confidence": max(min(best_r2, 0.95), 0.3)
        }

    # --------------------------------------------------
    # INVESTMENT ADVICE (INALTERADO NA LÓGICA)
    # --------------------------------------------------
    def investment_advice(self):
        if self.results is None or self.prices is None:
            return None

        advice = {
            "recommendation": "Hold",
            "risk_level": "Medium",
            "confidence": 0.5,
            "key_points": []
        }

        slope = self.results["linear"]["params"]["slope"]
        if slope > 50:
            advice["recommendation"] = "Consider Buying"
            advice["confidence"] = 0.7
            advice["key_points"].append(f"Upward trend: +${slope:.2f}/day")
        elif slope < -50:
            advice["recommendation"] = "Consider Selling"
            advice["confidence"] = 0.6
            advice["key_points"].append(f"Downward trend: ${slope:.2f}/day")

        vol = self.results["volatility"]["daily_volatility"]
        if vol > 6:
            advice["risk_level"] = "High"
        elif vol < 2:
            advice["risk_level"] = "Low"

        return advice
