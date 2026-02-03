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
        self.risk_metrics = {}

    # =========================
    # DATA
    # =========================

    def fetch_bitcoin_data(self, days=180):
        """Fetch Bitcoin price data with error handling"""
        try:
            url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
            params = {
                "vs_currency": "usd",
                "days": days,
                "interval": "daily"
            }

            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()

            if "prices" not in data or len(data["prices"]) == 0:
                raise ValueError("Dados vazios retornados da API")

            self.prices = np.array([p[1] for p in data["prices"]])
            self.dates = [datetime.fromtimestamp(p[0] / 1000) for p in data]

            return self.prices

        except requests.exceptions.Timeout:
            raise Exception("Timeout ao conectar com CoinGecko API")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Erro ao buscar dados: {e}")
        except (KeyError, ValueError) as e:
            raise Exception(f"Erro ao processar dados: {e}")

    # =========================
    # RISK METRICS
    # =========================

    def calculate_risk_metrics(self):
        """Calculate volatility, maximum drawdown, and Sharpe ratio"""
        if self.prices is None or len(self.prices) < 2:
            return None

        returns = np.diff(self.prices) / self.prices[:-1]

        # Volatilidade anualizada
        volatility = np.std(returns) * np.sqrt(365)

        # Maximum Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Sharpe Ratio (assumindo risk-free rate = 0)
        if np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(365)
        else:
            sharpe = 0

        # Value at Risk (VaR) 95%
        var_95 = np.percentile(returns, 5)

        self.risk_metrics = {
            "volatility": float(volatility),
            "max_drawdown": float(max_drawdown),
            "sharpe_ratio": float(sharpe),
            "daily_return_mean": float(np.mean(returns)),
            "daily_return_std": float(np.std(returns)),
            "var_95": float(var_95)
        }

        return self.risk_metrics

    # =========================
    # MODELOS
    # =========================

    def find_approximations(self):
        """Find best approximation models with validation"""
        if self.prices is None:
            raise ValueError("Dados não carregados. Execute fetch_bitcoin_data() primeiro.")

        x = np.arange(len(self.prices))
        y = self.prices
        results = {}

        # 1️⃣ Linear
        try:
            slope, intercept, r, _, _ = stats.linregress(x, y)
            results["linear"] = {
                "prediction": lambda t, a=slope, b=intercept: a * t + b,
                "r2": r ** 2,
                "slope": slope,
                "intercept": intercept
            }
        except Exception as e:
            print(f"Erro no modelo linear: {e}")

        # 2️⃣ Polinomial grau 2 (mais estável que grau 3)
        try:
            poly_coef = np.polyfit(x, y, 2)
            poly_pred = np.poly1d(poly_coef)
            y_pred_poly = poly_pred(x)
            r2_poly = 1 - np.sum((y - y_pred_poly) ** 2) / np.sum((y - np.mean(y)) ** 2)

            results["polynomial"] = {
                "prediction": lambda t: np.poly1d(poly_coef)(t),
                "r2": r2_poly,
                "coefficients": poly_coef
            }
        except Exception as e:
            print(f"Erro no modelo polinomial: {e}")

        # 3️⃣ Polinomial + seno
        try:
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
                "r2": r2_poly_sine,
                "parameters": popt
            }
        except Exception as e:
            print(f"Erro no modelo poly_sine: {e}")

        # 4️⃣ Exponencial móvel
        try:
            alpha = 0.1
            ema = [y[0]]
            for i in range(1, len(y)):
                ema.append(alpha * y[i] + (1 - alpha) * ema[-1])

            results["exponential_ma"] = {
                "values": np.array(ema),
                "alpha": alpha,
                "last_value": ema[-1]
            }
        except Exception as e:
            print(f"Erro no EMA: {e}")

        # 5️⃣ Média móvel
        try:
            window = 30
            ma = pd.Series(y).rolling(window).mean().values

            results["moving_average"] = {
                "values": ma,
                "window": window
            }
        except Exception as e:
            print(f"Erro na média móvel: {e}")

        # =========================
        # OSCILADOR ESTOCÁSTICO
        # =========================

        try:
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
                "current_k": float(k_values[-1]) if len(k_values) > 0 else 50,
                "current_d": float(d_values[-1]) if len(d_values) > 0 else 50
            }
        except Exception as e:
            print(f"Erro no oscilador estocástico: {e}")
            results["stochastic"] = {
                "k": [],
                "d": [],
                "current_k": 50,
                "current_d": 50
            }

        # =========================
        # RSI (Relative Strength Index)
        # =========================

        try:
            period_rsi = 14
            deltas = np.diff(y)
            
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = pd.Series(gains).rolling(period_rsi).mean().values
            avg_loss = pd.Series(losses).rolling(period_rsi).mean().values
            
            rs = avg_gain / (avg_loss + 1e-10)  # Evitar divisão por zero
            rsi = 100 - (100 / (1 + rs))
            
            results["rsi"] = {
                "values": rsi,
                "current": float(rsi[-1]) if len(rsi) > 0 else 50
            }
        except Exception as e:
            print(f"Erro no RSI: {e}")
            results["rsi"] = {"values": [], "current": 50}

        self.results = results
        return results

    # =========================
    # FORECAST WITH CONFIDENCE
    # =========================

    def forecast_with_confidence(self, forecast_days=7):
        """Generate forecast with confidence intervals"""
        if not self.results:
            raise ValueError("Execute find_approximations() primeiro")

        # Selecionar melhor modelo
        valid_models = {k: v for k, v in self.results.items() 
                       if "r2" in v and "prediction" in v}

        if not valid_models:
            raise ValueError("Nenhum modelo válido disponível")

        best_model_name, best_model = max(
            valid_models.items(),
            key=lambda x: x[1]["r2"]
        )

        x_start = len(self.prices)
        x_future = np.arange(x_start, x_start + forecast_days + 1)

        forecast = best_model["prediction"](x_future)

        # Calcular erro histórico
        x_hist = np.arange(len(self.prices))
        y_pred = best_model["prediction"](x_hist)
        residuals = self.prices - y_pred
        std_error = np.std(residuals)

        # Intervalo de confiança (95%)
        confidence_interval = 1.96 * std_error

        return {
            "model": best_model_name,
            "forecast": float(forecast[-1]),
            "lower_bound": float(forecast[-1] - confidence_interval),
            "upper_bound": float(forecast[-1] + confidence_interval),
            "r2": best_model["r2"],
            "std_error": float(std_error),
            "forecast_array": forecast
        }

    # =========================
    # RECOMENDAÇÃO MELHORADA
    # =========================

    def investment_advice(self):
        """Multi-indicator investment recommendation"""
        if not self.results:
            raise ValueError("Execute find_approximations() primeiro")

        k = self.results["stochastic"]["current_k"]
        rsi_current = self.results.get("rsi", {}).get("current", 50)

        # Análise de tendência (linear)
        if "linear" in self.results:
            trend_slope = self.results["linear"].get("slope", 0)
        else:
            trend_slope = 0

        score = 0
        reasons = []
        confidence = "média"

        # Estocástico (peso 2)
        if k < 20:
            score += 2
            reasons.append("Estocástico em zona de sobrevenda (<20)")
            confidence = "alta"
        elif k > 80:
            score -= 2
            reasons.append("Estocástico em zona de sobrecompra (>80)")
            confidence = "alta"
        elif k < 30:
            score += 1
            reasons.append("Estocástico próximo à sobrevenda")
        elif k > 70:
            score -= 1
            reasons.append("Estocástico próximo à sobrecompra")

        # RSI (peso 1)
        if rsi_current < 30:
            score += 1
            reasons.append("RSI indica sobrevenda (<30)")
        elif rsi_current > 70:
            score -= 1
            reasons.append("RSI indica sobrecompra (>70)")

        # Tendência (peso 1)
        if trend_slope > 100:  # Subida significativa
            score += 1
            reasons.append("Tendência de alta forte")
        elif trend_slope > 0:
            score += 0.5
            reasons.append("Tendência de alta moderada")
        elif trend_slope < -100:
            score -= 1
            reasons.append("Tendência de baixa forte")
        elif trend_slope < 0:
            score -= 0.5
            reasons.append("Tendência de baixa moderada")

        # Volatilidade
        if self.risk_metrics and self.risk_metrics.get("volatility", 0) > 1.0:
            score -= 0.5
            reasons.append(f"Alta volatilidade ({self.risk_metrics['volatility']:.2f})")
            if confidence == "alta":
                confidence = "média"

        # Decisão final
        if score >= 2:
            rec = "Acumular"
            action_color = "green"
        elif score >= 1:
            rec = "Comprar moderadamente"
            action_color = "lightgreen"
        elif score <= -2:
            rec = "Vender"
            action_color = "red"
        elif score <= -1:
            rec = "Reduzir posição"
            action_color = "orange"
        else:
            rec = "Manter"
            action_color = "gray"

        return {
            "recommendation": rec,
            "score": float(score),
            "confidence": confidence,
            "reasons": reasons,
            "stochastic_k": k,
            "rsi": rsi_current,
            "trend_slope": float(trend_slope),
            "action_color": action_color
        }

    # =========================
    # MODEL VALIDATION
    # =========================

    def validate_models(self, train_ratio=0.8):
        """Cross-validation of models"""
        if self.prices is None or len(self.prices) < 10:
            return None

        split_idx = int(len(self.prices) * train_ratio)

        test_x = np.arange(split_idx, len(self.prices))
        test_y = self.prices[split_idx:]

        validation_scores = {}

        for model_name, model_data in self.results.items():
            if "prediction" in model_data:
                try:
                    pred = model_data["prediction"](test_x)
                    
                    if isinstance(pred, (int, float)):
                        pred = np.array([pred])
                    
                    mse = np.mean((test_y - pred) ** 2)
                    mae = np.mean(np.abs(test_y - pred))
                    
                    validation_scores[model_name] = {
                        "mse": float(mse),
                        "rmse": float(np.sqrt(mse)),
                        "mae": float(mae)
                    }
                except Exception as e:
                    print(f"Erro ao validar modelo {model_name}: {e}")

        return validation_scores
