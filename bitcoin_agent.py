"""
Bitcoin Analysis Agent
Historical data analysis and mathematical approximation functions
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
        print("Bitcoin Analysis Agent initialized")

    def fetch_bitcoin_data(self, days=365):
        """Fetch historical Bitcoin data using public API"""
        print(f"Fetching data for the last {days} days...")

        try:
            url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
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

            print(f"Data obtained: {len(prices)} price points")
            print(f"Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
            print(f"Initial price: ${prices[0]:,.2f}")
            print(f"Final price: ${prices[-1]:,.2f}")
            print(f"Change: {((prices[-1]/prices[0] - 1) * 100):+.2f}%")

            return self.data

        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def find_approximations(self):
        """Find different functions that approximate price fluctuations"""
        print("Analyzing patterns and finding approximations...")

        if self.prices is None:
            print("No data available. Run fetch_bitcoin_data() first.")
            return

        x = np.arange(len(self.prices))
        y = self.prices

        results = {}

        # 1. Linear Regression
        print("\nLINEAR TREND ANALYSIS")
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
            print(f"Trend direction: {'Up' if slope > 0 else 'Down'}")
            print(f"Daily change: ${slope:.2f}")
        except Exception as e:
            print(f"Linear regression error: {e}")

        # 2. Polynomial Regression (degree 2)
        print("\nPOLYNOMIAL TREND ANALYSIS (degree 2)")
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

        # 3. Polynomial + Sinusoidal Regression
        print("\nPOLYNOMIAL + SEASONAL CYCLES ANALYSIS")
        try:
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
            
            p0_poly = np.polyfit(x, y, poly_degree)
            p0 = list(p0_poly)[::-1] + [1000.0, 0.0]
            
            popt, pcov = curve_fit(poly_sine_func, x, y, p0=p0, maxfev=5000)
            
            y_pred = poly_sine_func(x, *popt)
            r2 = 1 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))
            
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

        # 4. Exponential Regression
        print("\nEXPONENTIAL GROWTH ANALYSIS")
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
            print(f"Daily growth rate: {(popt[1]*100):.4f}%")
        except Exception as e:
            print(f"Exponential regression error: {e}")

        # 5. Moving Average
        print("\nMOVING AVERAGE ANALYSIS")
        window = min(30, len(y)//3)
        ma = pd.Series(y).rolling(window=window).mean()
        print(f"Moving average window: {window} days")
        results['moving_average'] = {
            'function': f'MA({window} days)',
            'values': ma.values,
            'window': window
        }

        # 6. Volatility Analysis
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

        # 7. Stochastic Oscillator
        print("\nSTOCHASTIC OSCILLATOR ANALYSIS")
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

        # Datas correspondentes aos valores do oscilador
        stoch_dates = self.dates[period-1:] if len(self.dates) > period else self.dates

        results['stochastic'] = {
            'k': stoch_k,  # %K values
            'd': stoch_d,  # %D values (signal line)
            'period': period,
            'current_k': stoch_k[-1] if stoch_k else 0,
            'current_d': stoch_d[-1] if stoch_d else 0,
            'dates': stoch_dates,  # Datas correspondentes
            'k_values': stoch_k,  # Alias para compatibilidade
            'd_values': stoch_d   # Alias para compatibilidade
        }

        current_k = stoch_k[-1] if stoch_k else 0
        current_d = stoch_d[-1] if stoch_d else 0

        print(f"Analysis period: {period} days")
        print(f"Current %K value: {current_k:.2f}")
        print(f"Current %D value: {current_d:.2f}")
        
        # Análise de crossover
        if len(stoch_k) > 1 and len(stoch_d) > 1:
            if stoch_k[-1] > stoch_d[-1] and stoch_k[-2] <= stoch_d[-2]:
                print(f"Signal: BULLISH CROSSOVER (%K crossed above %D)")
                results['stochastic']['signal'] = 'bullish_crossover'
            elif stoch_k[-1] < stoch_d[-1] and stoch_k[-2] >= stoch_d[-2]:
                print(f"Signal: BEARISH CROSSOVER (%K crossed below %D)")
                results['stochastic']['signal'] = 'bearish_crossover'
            else:
                results['stochastic']['signal'] = 'no_crossover'

        if current_k > 80:
            print(f"Market condition: OVERBOUGHT (K > 80)")
            results['stochastic']['condition'] = 'overbought'
        elif current_k < 20:
            print(f"Market condition: OVERSOLD (K < 20)")
            results['stochastic']['condition'] = 'oversold'
        else:
            print(f"Market condition: NEUTRAL (20 < K < 80)")
            results['stochastic']['condition'] = 'neutral'

        self.results = results
        return results

    def investment_advice(self):
        """Generate investment advice based on analysis"""
        print("\n" + "="*60)
        print("INVESTMENT ADVICE")
        print("="*60)
        
        if not hasattr(self, 'results') or self.prices is None:
            print("No analysis available. Run find_approximations() first.")
            return None
        
        advice = {
            'summary': '',
            'risk_level': 'Medium',
            'recommendation': 'Hold',
            'confidence': 0.5,
            'time_horizon': 'Medium-term',
            'key_points': []
        }
        
        # Trend analysis
        if 'linear' in self.results:
            slope = self.results['linear']['params']['slope']
            if slope > 100:
                advice['key_points'].append(f"Strong upward trend: +${slope:.2f}/day")
                advice['recommendation'] = 'Consider Buying'
                advice['confidence'] = 0.7
            elif slope > 20:
                advice['key_points'].append(f"Upward trend: +${slope:.2f}/day")
                advice['recommendation'] = 'Hold/Buy Dips'
                advice['confidence'] = 0.6
            elif slope < -100:
                advice['key_points'].append(f"Strong downward trend: ${slope:.2f}/day")
                advice['recommendation'] = 'Consider Selling'
                advice['confidence'] = 0.6
            elif slope < -20:
                advice['key_points'].append(f"Downward trend: ${slope:.2f}/day")
                advice['recommendation'] = 'Wait/Caution'
                advice['confidence'] = 0.5
            else:
                advice['key_points'].append("Sideways movement - no clear trend")
                advice['recommendation'] = 'Hold'
                advice['confidence'] = 0.4
        
        # Volatility analysis
        if 'volatility' in self.results:
            vol = self.results['volatility']['daily_volatility']
            if vol > 8:
                advice['risk_level'] = 'Very High'
                advice['key_points'].append(f"Very high volatility: {vol:.1f}% daily")
                advice['time_horizon'] = 'Short-term only'
            elif vol > 5:
                advice['risk_level'] = 'High'
                advice['key_points'].append(f"High volatility: {vol:.1f}% daily")
                advice['time_horizon'] = 'Short to Medium-term'
            elif vol > 2:
                advice['risk_level'] = 'Medium'
                advice['key_points'].append(f"Moderate volatility: {vol:.1f}% daily")
                advice['time_horizon'] = 'Medium-term'
            else:
                advice['risk_level'] = 'Low'
                advice['key_points'].append(f"Low volatility: {vol:.1f}% daily")
                advice['time_horizon'] = 'Long-term'
        
        # Stochastic analysis
        if 'stochastic' in self.results:
            k = self.results['stochastic']['current_k']
            if k > 90:
                advice['key_points'].append(f"Extremely overbought (Stochastic: {k:.1f})")
                advice['recommendation'] = 'Consider Taking Profits'
                if advice['confidence'] < 0.6:
                    advice['confidence'] = 0.65
            elif k > 80:
                advice['key_points'].append(f"Overbought (Stochastic: {k:.1f})")
                if advice['recommendation'] == 'Consider Buying':
                    advice['recommendation'] = 'Wait for Pullback'
            elif k < 10:
                advice['key_points'].append(f"Extremely oversold (Stochastic: {k:.1f})")
                advice['recommendation'] = 'Consider Accumulating'
                if advice['confidence'] < 0.6:
                    advice['confidence'] = 0.65
            elif k < 20:
                advice['key_points'].append(f"Oversold (Stochastic: {k:.1f})")
                if advice['recommendation'] == 'Consider Selling':
                    advice['recommendation'] = 'Hold/Accumulate'
        
        # Model accuracy analysis
        best_r2 = 0
        for key, value in self.results.items():
            r2 = value.get('r_squared', value.get('r2', 0))
            if r2 > best_r2:
                best_r2 = r2
        
        if best_r2 > 0.8:
            advice['key_points'].append(f"High model accuracy (R²={best_r2:.2f})")
            advice['confidence'] = min(advice['confidence'] + 0.15, 0.9)
        elif best_r2 > 0.6:
            advice['key_points'].append(f"Moderate model accuracy (R²={best_r2:.2f})")
            advice['confidence'] = min(advice['confidence'] + 0.1, 0.8)
        else:
            advice['key_points'].append(f"Low model accuracy (R²={best_r2:.2f})")
            advice['confidence'] = max(advice['confidence'] - 0.1, 0.3)
        
        # Current price vs averages
        current_price = self.prices[-1]
        if len(self.prices) > 30:
            ma_30 = np.mean(self.prices[-30:])
            if current_price > ma_30 * 1.2:
                advice['key_points'].append("Price >20% above 30-day average")
                if advice['recommendation'] == 'Consider Buying':
                    advice['recommendation'] = 'Wait for Correction'
            elif current_price < ma_30 * 0.8:
                advice['key_points'].append("Price <20% below 30-day average - potential opportunity")
                if advice['recommendation'] in ['Hold', 'Wait']:
                    advice['recommendation'] = 'Consider Dollar-Cost Averaging'
        
        # Generate final summary
        confidence_percent = int(advice['confidence'] * 100)
        
        advice['summary'] = (
            f"Recommendation: {advice['recommendation']}\n"
            f"Risk Level: {advice['risk_level']}\n"
            f"Time Horizon: {advice['time_horizon']}\n"
            f"Confidence: {confidence_percent}%\n"
            f"\nKey Factors:\n" + "\n".join(f"• {point}" for point in advice['key_points'])
        )
        
        # Print to console
        print(f"\nFINAL RECOMMENDATION: {advice['recommendation']}")
        print(f"Risk Level: {advice['risk_level']}")
        print(f"Time Horizon: {advice['time_horizon']}")
        print(f"Confidence: {confidence_percent}%")
        print(f"\nKey Analysis Points:")
        for point in advice['key_points']:
            print(f"  • {point}")
        
        if advice['recommendation'] in ['Consider Buying', 'Consider Accumulating', 'Hold/Buy Dips']:
            print(f"\nACTION: Potential buying opportunity")
            print("   Consider dollar-cost averaging strategy")
        elif advice['recommendation'] in ['Consider Selling', 'Consider Taking Profits']:
            print(f"\nACTION: Consider profit-taking")
            print("   Set stop-loss orders if maintaining position")
        else:
            print(f"\nACTION: Maintain current position")
            print("   Monitor market conditions for changes")
        
        print("="*60)
        
        return advice

    def visualize(self, save_path='bitcoin_analysis.png'):
        """Create visualization of approximations with Stochastic Oscillator"""
        print(f"Generating visualization...")

        if self.prices is None or not hasattr(self, 'results'):
            print("Run fetch_bitcoin_data() and find_approximations() first.")
            return

        # Create figure with 3 subplots (2 for price, 1 for stochastic)
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Bitcoin Analysis - Mathematical Approximations & Stochastic Oscillator', 
                    fontsize=16, fontweight='bold')

        x = np.arange(len(self.prices))
        dates = self.dates

        # Plot 1: Price with models
        ax1 = axes[0]
        ax1.plot(dates, self.prices, 'b-', alpha=0.5, label='Bitcoin Price', linewidth=2)
        
        if 'linear' in self.results:
            y_linear = self.results['linear']['prediction'](x)
            ax1.plot(dates, y_linear, 'r--', label='Linear Trend', linewidth=2)
        
        if 'polynomial' in self.results:
            y_poly = self.results['polynomial']['prediction'](x)
            ax1.plot(dates, y_poly, 'g--', label='Polynomial Trend', linewidth=2)
        
        ax1.set_title('Bitcoin Price with Trend Models')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (USD)')
        ax1.legend(fontsize=9, loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

        # Plot 2: Other models
        ax2 = axes[1]
        ax2.plot(dates, self.prices, 'b-', alpha=0.3, label='Bitcoin Price', linewidth=1)
        
        if 'poly_sine' in self.results:
            y_poly_sine = self.results['poly_sine']['prediction'](x)
            ax2.plot(dates, y_poly_sine, 'purple', linestyle='--', label='Polynomial+Sine', linewidth=2)
        
        if 'exponential' in self.results:
            y_exp = self.results['exponential']['prediction'](x)
            ax2.plot(dates, y_exp, 'orange', linestyle='--', label='Exponential', linewidth=2)
        
        if 'moving_average' in self.results:
            ma_values = self.results['moving_average']['values']
            ax2.plot(dates, ma_values, 'cyan', linestyle='-', label='Moving Average', linewidth=2)
        
        ax2.set_title('Advanced Models & Moving Average')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price (USD)')
        ax2.legend(fontsize=9, loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

        # Plot 3: Stochastic Oscillator
        ax3 = axes[2]
        
        if 'stochastic' in self.results:
            stochastic = self.results['stochastic']
            stoch_dates = stochastic.get('dates', dates[len(dates)-len(stochastic['k']):])
            stoch_k = stochastic['k']
            stoch_d = stochastic['d']
            
            # Plot %K and %D lines
            ax3.plot(stoch_dates, stoch_k, 'b-', label='%K (Fast)', linewidth=2)
            ax3.plot(stoch_dates, stoch_d, 'r-', label='%D (Slow)', linewidth=2)
            
            # Add overbought/oversold lines
            ax3.axhline(y=80, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            ax3.axhline(y=20, color='gray', linestyle='--', alpha=0.7, linewidth=1)
            
            # Fill overbought and oversold areas
            ax3.fill_between(stoch_dates, 80, 100, alpha=0.1, color='red', label='Overbought Zone')
            ax3.fill_between(stoch_dates, 0, 20, alpha=0.1, color='green', label='Oversold Zone')
            
            # Add current value markers
            current_k = stochastic['current_k']
            current_d = stochastic['current_d']
            
            # Find the last date for the marker
            last_date = stoch_dates[-1] if len(stoch_dates) > 0 else dates[-1]
            
            # Add markers for current values
            ax3.plot(last_date, current_k, 'bo', markersize=8, label=f'Current %K: {current_k:.1f}')
            ax3.plot(last_date, current_d, 'ro', markersize=8, label=f'Current %D: {current_d:.1f}')
            
            # Add crossover markers if any
            condition = stochastic.get('condition', 'neutral')
            if condition == 'overbought':
                ax3.text(0.02, 0.95, 'OVERBOUGHT', transform=ax3.transAxes, 
                        fontsize=12, fontweight='bold', color='red',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            elif condition == 'oversold':
                ax3.text(0.02, 0.95, 'OVERSOLD', transform=ax3.transAxes, 
                        fontsize=12, fontweight='bold', color='green',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            signal = stochastic.get('signal', 'no_crossover')
            if signal == 'bullish_crossover':
                ax3.text(0.02, 0.85, 'BULLISH CROSSOVER', transform=ax3.transAxes, 
                        fontsize=10, fontweight='bold', color='green',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            elif signal == 'bearish_crossover':
                ax3.text(0.02, 0.85, 'BEARISH CROSSOVER', transform=ax3.transAxes, 
                        fontsize=10, fontweight='bold', color='red',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax3.set_title('Stochastic Oscillator (14-day period)')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Stochastic Value')
        ax3.set_ylim(-5, 105)
        ax3.legend(fontsize=9, loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
        ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to: {save_path}")
        
        plt.show()
        
        return save_path

    def generate_report(self):
        """Generate complete analysis report"""
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

        print("\nKEY INSIGHTS:")
        if 'linear' in self.results:
            slope = self.results['linear']['params']['slope']
            if slope > 0:
                print(f"Upward trend: +${slope:.2f} per day on average")
            else:
                print(f"Downward trend: ${slope:.2f} per day on average")

        if 'volatility' in self.results:
            vol = self.results['volatility']['daily_volatility']
            if vol > 5:
                print(f"High volatility ({vol:.2f}% daily) - High risk asset")
            elif vol > 2:
                print(f"Moderate volatility ({vol:.2f}% daily)")
            else:
                print(f"Low volatility ({vol:.2f}% daily) - Stable period")

        if 'stochastic' in self.results:
            stoch = self.results['stochastic']
            print(f"\nSTOCHASTIC OSCILLATOR:")
            print(f"Current %K: {stoch['current_k']:.2f}")
            print(f"Current %D: {stoch['current_d']:.2f}")
            print(f"Market condition: {stoch.get('condition', 'N/A').upper()}")
            print(f"Signal: {stoch.get('signal', 'no_crossover').replace('_', ' ').upper()}")

        print("\n" + "="*60)

def main():
    """Main function - execute the agent"""
    print("Starting Bitcoin Analysis Agent")

    # Create agent
    agent = BitcoinAnalysisAgent()

    # Fetch data
    data = agent.fetch_bitcoin_data(days=180)
    if data is None:
        print("Using simulated data for demonstration.")
        return

    # Find approximations
    agent.find_approximations()

    # Generate investment advice
    agent.investment_advice()

    # Generate visualization with stochastic oscillator
    agent.visualize()

    # Final report
    agent.generate_report()

    print("Analysis complete!")


if __name__ == "__main__":
    main()