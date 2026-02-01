"""Agente de IA para análise do Bitcoin
Busca dados históricos e encontra funções de aproximação
"""

import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json

class BitcoinAnalysisAgent:
    def __init__(self):
        self.data = None
        self.prices = None
        self.dates = None
        print("Bitcoin Analysis Agent initialized!")

    def fetch_bitcoin_data(self, days=365):
        """Busca dados históricos do Bitcoin usando API pública"""
        print(f"\nFetching data for the last {days} days...")

        try:
            url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            prices = [item[1] for item in data['prices']]
            timestamps = [item[0] for item in data['prices']]
            dates = [datetime.fromtimestamp(ts/1000) for ts in timestamps]

            self.data = pd.DataFrame({
                'date': dates,
                'price': prices
            })

            self.prices = np.array(prices)
            self.dates = dates

            print(f"Data obtained! {len(prices)} price points")
            print(f"Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
            print(f"Initial price: ${prices[0]:,.2f}")
            print(f"Final price: ${prices[-1]:,.2f}")
            print(f"Change: {((prices[-1]/prices[0] - 1) * 100):+.2f}%")

            return self.data

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def find_approximations(self):
        """Encontra diferentes funções que aproximam a flutuação"""
        print("\nAnalyzing patterns and finding approximations...\n")

        if self.prices is None:
            print("No data available. Run fetch_bitcoin_data() first.")
            return

        x = np.arange(len(self.prices))
        y = self.prices

        results = {}

        # 1. Regressão Linear
        print("LINEAR TREND")
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            linear_func = lambda x: slope * x + intercept
            results['linear'] = {
                'function': f'f(x) = {slope:.2f}x + {intercept:.2f}',
                'r_squared': r_value**2,
                'prediction': linear_func,
                'params': {'slope': slope, 'intercept': intercept}
            }
            print(f"Function: f(x) = {slope:.2f}x + {intercept:.2f}")
            print(f"R² = {r_value**2:.4f}")
            print(f"Trend: {'Up' if slope > 0 else 'Down'} (${slope:.2f}/day)")
        except Exception as e:
            print(f"Linear regression error: {e}")

        # 2. Regressão Polinomial (grau 2)
        print("\nPOLYNOMIAL TREND (degree 2)")
        try:
            coeffs = np.polyfit(x, y, 2)
            poly_func = np.poly1d(coeffs)
            y_pred = poly_func(x)
            r2_poly = 1 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))
            results['polynomial'] = {
                'function': f'f(x) = {coeffs[0]:.4f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}',
                'r_squared': r2_poly,
                'prediction': poly_func,
                'params': {'coefficients': coeffs}
            }
            print(f"Function: f(x) = {coeffs[0]:.4f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}")
            print(f"R² = {r2_poly:.4f}")
        except Exception as e:
            print(f"Polynomial regression error: {e}")

        # 3. Regressão Polinomial + Senoidal
        print("\nPOLYNOMIAL + SEASONAL CYCLES")
        try:
            best_r2 = -np.inf
            best_result = None
            
            # Configuração simplificada
            cycle_period = 45
            poly_degree = 3
            
            def poly_sine_func(x, *params):
                poly_params = params[:poly_degree+1]
                A, phi = params[poly_degree+1], params[poly_degree+2]
                
                poly_val = 0
                for i, p in enumerate(poly_params):
                    poly_val += p * (x ** i)
                
                sine_val = A * np.sin(2 * np.pi * x / cycle_period + phi)
                return poly_val + sine_val
            
            # Valores iniciais
            p0_poly = np.polyfit(x, y, poly_degree)
            p0 = list(p0_poly)[::-1] + [1000.0, 0.0]
            
            # Ajustar
            popt, pcov = curve_fit(poly_sine_func, x, y, p0=p0, maxfev=5000)
            
            # Calcular R²
            y_pred = poly_sine_func(x, *popt)
            r2 = 1 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))
            
            # Construir string da função
            poly_terms = []
            for i, coef in enumerate(popt[:poly_degree+1]):
                if i == 0:
                    poly_terms.append(f"{coef:.2f}")
                elif i == 1:
                    poly_terms.append(f"{coef:+.2f}x")
                else:
                    poly_terms.append(f"{coef:+.4f}x^{i}")
            
            A = popt[poly_degree+1]
            phi = popt[poly_degree+2]
            
            func_str = " + ".join(poly_terms)
            func_str += f" {A:+.2f}×sin(2πx/{cycle_period} {phi:+.2f})"
            
            results['poly_sine'] = {
                'function': func_str,
                'r2': r2,
                'prediction': lambda t, p=popt: poly_sine_func(t, *p),
                'params': popt,
                'cycle_period': cycle_period,
                'amplitude': abs(A)
            }
            
            print(f"Polynomial degree: {poly_degree}")
            print(f"Cycle period: {cycle_period} days")
            print(f"Cycle amplitude: ${abs(A):,.2f}")
            print(f"R² = {r2:.4f}")
            
        except Exception as e:
            print(f"Polynomial+sine adjustment error: {e}")
            # Fallback para polinomial de grau 4
            try:
                coeffs_high = np.polyfit(x, y, 4)
                poly_high_func = np.poly1d(coeffs_high)
                y_pred_high = poly_high_func(x)
                r2_high = 1 - (np.sum((y - y_pred_high)**2) / np.sum((y - np.mean(y))**2))
                
                func_str = f"{coeffs_high[0]:.6f}x⁴ {coeffs_high[1]:+.4f}x³ {coeffs_high[2]:+.4f}x² {coeffs_high[3]:+.2f}x {coeffs_high[4]:+.2f}"
                
                results['poly_sine'] = {
                    'function': func_str + " (High-degree polynomial)",
                    'r2': r2_high,
                    'prediction': poly_high_func,
                    'params': coeffs_high
                }
                print(f"Using polynomial degree 4. R² = {r2_high:.4f}")
            except:
                print("Could not fit polynomial+sine model")

        # 4. Regressão Exponencial
        print("\nEXPONENTIAL GROWTH")
        try:
            def exponential(x, a, b):
                return a * np.exp(b * x)

            y_positive = y - np.min(y) + 1
            popt, pcov = curve_fit(exponential, x, y_positive, p0=[y_positive[0], 0.001], maxfev=10000)
            exp_func = lambda x: exponential(x, *popt) + np.min(y) - 1
            y_pred_exp = exp_func(x)
            r2_exp = 1 - (np.sum((y - y_pred_exp)**2) / np.sum((y - np.mean(y))**2))

            results['exponential'] = {
                'function': f'f(x) = {popt[0]:.2f} × e^({popt[1]:.6f}x) + {np.min(y)-1:.2f}',
                'r_squared': r2_exp,
                'prediction': exp_func,
                'params': {'a': popt[0], 'b': popt[1], 'offset': np.min(y)-1}
            }
            print(f"Function: f(x) = {popt[0]:.2f} × e^({popt[1]:.6f}x) + {np.min(y)-1:.2f}")
            print(f"R² = {r2_exp:.4f}")
            print(f"Growth rate: {(popt[1]*100):.4f}%/day")
        except Exception as e:
            print(f"Exponential regression error: {e}")

        # 5. Média Móvel
        print("\nMOVING AVERAGE (30 days)")
        window = min(30, len(y)//3)
        ma = pd.Series(y).rolling(window=window).mean()
        print(f"Smoothing with {window} days window")
        results['moving_average'] = {
            'function': f'MA({window} days)',
            'values': ma.values,
            'window': window
        }

        # 6. Volatilidade
        print("\nVOLATILITY ANALYSIS")
        returns = np.diff(y) / y[:-1] * 100
        volatility = np.std(returns)
        results['volatility'] = {
            'daily_volatility': volatility,
            'annual_volatility': volatility * np.sqrt(365)
        }
        print(f"Daily volatility: {volatility:.2f}%")
        print(f"Annualized volatility: {volatility * np.sqrt(365):.2f}%")
        print(f"Average daily return: {np.mean(returns):.3f}%")

        # 7. Oscilador Estocástico
        print("\nSTOCHASTIC OSCILLATOR")
        period = 14
        stoch_k = []
        stoch_d = []

        for i in range(period - 1, len(y)):
            lowest_low = np.min(y[i - period + 1:i + 1])
            highest_high = np.max(y[i - period + 1:i + 1])
            current_close = y[i]

            if highest_high - lowest_low != 0:
                k = 100 * (current_close - lowest_low) / (highest_high - lowest_low)
            else:
                k = 50
            stoch_k.append(k)

        for i in range(len(stoch_k)):
            if i < 2:
                stoch_d.append(stoch_k[i])
            else:
                d = np.mean(stoch_k[i-2:i+1])
                stoch_d.append(d)

        results['stochastic'] = {
            'k': stoch_k,
            'd': stoch_d,
            'period': period,
            'current_k': stoch_k[-1] if stoch_k else 0,
            'current_d': stoch_d[-1] if stoch_d else 0
        }

        current_k = stoch_k[-1] if stoch_k else 0
        current_d = stoch_d[-1] if stoch_d else 0

        print(f"Period: {period} days")
        print(f"Current %K: {current_k:.2f}")
        print(f"Current %D: {current_d:.2f}")

        if current_k > 80:
            print(f"Signal: OVERBOUGHT (K > 80)")
        elif current_k < 20:
            print(f"Signal: OVERSOLD (K < 20)")
        else:
            print(f"Signal: NEUTRAL (20 < K < 80)")

        self.results = results
        return results

    def visualize(self, save_path='bitcoin_analysis.png'):
        """Cria visualização das aproximações"""
        print(f"\nGenerating visualization...")

        if self.prices is None or not hasattr(self, 'results'):
            print("Run fetch_bitcoin_data() and find_approximations() first.")
            return

        return "Visualization would be saved to: " + save_path

    def generate_report(self):
        """Gera relatório completo da análise"""
        print("\n" + "="*60)
        print("FINAL REPORT - BITCOIN ANALYSIS")
        print("="*60)

        if not hasattr(self, 'results'):
            print("Run analysis first.")
            return

        print("\nBEST APPROXIMATION:")
        models_with_r2 = [(k, v) for k, v in self.results.items() if 'r_squared' in v or 'r2' in v]
        if models_with_r2:
            best_model = max(
                models_with_r2,
                key=lambda x: x[1].get('r_squared', x[1].get('r2', 0))
            )
            print(f"Model: {best_model[0].upper()}")
            print(f"Function: {best_model[1].get('function', 'N/A')}")
            r2 = best_model[1].get('r_squared', best_model[1].get('r2', 0))
            print(f"R² = {r2:.4f}")
        else:
            print("No regression models available")

        print("\nINSIGHTS:")
        if 'linear' in self.results:
            slope = self.results['linear']['params']['slope']
            if slope > 0:
                print(f"Upward trend: +${slope:.2f} per day on average")
            else:
                print(f"Downward trend: ${slope:.2f} per day on average")

        if 'volatility' in self.results:
            vol = self.results['volatility']['daily_volatility']
            if vol > 5:
                print(f"High volatility ({vol:.2f}% daily) - Risky asset")
            elif vol > 2:
                print(f"Moderate volatility ({vol:.2f}% daily)")
            else:
                print(f"Low volatility ({vol:.2f}% daily) - Stable period")

        print("\n" + "="*60)