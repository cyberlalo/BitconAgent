# agent_runner.py
"""
Autonomous Bitcoin Analysis Agent
Runs once per execution (cron-friendly),
fetches data, generates forecast,
computes % variation, and stores memory.
"""

import sys
from datetime import datetime
import numpy as np

from bitcoin_agent import BitcoinAnalysisAgent
from memory import (
    init_db, 
    save_prediction, 
    get_prediction_accuracy,
    check_pending_predictions,
    update_prediction_actual,
    save_daily_metrics
)

# =========================
# CONFIGURAÇÃO
# =========================

ANALYSIS_DAYS = 180     # janela histórica
FORECAST_DAYS = 7       # horizonte da previsão (D+7)

# =========================
# EXECUÇÃO ÚNICA DO AGENTE
# =========================

def run_agent_once():
    """Execute agent analysis cycle"""
    now = datetime.now().isoformat()
    print("=" * 80)
    print(f"🤖 Bitcoin Analysis Agent - Execution started: {now}")
    print("=" * 80)

    try:
        # Inicializar banco de dados
        init_db()

        # Criar agente
        agent = BitcoinAnalysisAgent()

        # 1️⃣ Coleta de dados
        print("\n📊 Fetching Bitcoin data...")
        try:
            agent.fetch_bitcoin_data(days=ANALYSIS_DAYS)
            print(f"✅ Successfully fetched {len(agent.prices)} data points")
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return

        # 2️⃣ Análise de modelos
        print("\n🔬 Running model analysis...")
        try:
            results = agent.find_approximations()
            print(f"✅ {len(results)} models analyzed")
        except Exception as e:
            print(f"❌ Error in model analysis: {e}")
            return

        # 3️⃣ Cálculo de métricas de risco
        print("\n⚠️  Calculating risk metrics...")
        try:
            risk_metrics = agent.calculate_risk_metrics()
            if risk_metrics:
                print(f"   • Volatility: {risk_metrics['volatility']:.2%}")
                print(f"   • Max Drawdown: {risk_metrics['max_drawdown']:.2%}")
                print(f"   • Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
        except Exception as e:
            print(f"⚠️  Warning: Could not calculate risk metrics: {e}")
            risk_metrics = None

        # 4️⃣ Forecast com intervalo de confiança
        print(f"\n🔮 Generating {FORECAST_DAYS}-day forecast...")
        try:
            forecast = agent.forecast_with_confidence(forecast_days=FORECAST_DAYS)
            
            current_price = agent.prices[-1]
            forecast_price = forecast["forecast"]
            variation_pct = ((forecast_price - current_price) / current_price) * 100

            print(f"   • Model: {forecast['model']}")
            print(f"   • Current Price: ${current_price:,.2f}")
            print(f"   • Forecast (D+{FORECAST_DAYS}): ${forecast_price:,.2f}")
            print(f"   • Variation: {variation_pct:+.2f}%")
            print(f"   • Confidence Interval: ${forecast['lower_bound']:,.2f} - ${forecast['upper_bound']:,.2f}")
            print(f"   • R²: {forecast['r2']:.4f}")

        except Exception as e:
            print(f"❌ Error generating forecast: {e}")
            return

        # 5️⃣ Recomendação de investimento
        print("\n💡 Investment recommendation...")
        try:
            advice = agent.investment_advice()
            
            print(f"   • Recommendation: {advice['recommendation']}")
            print(f"   • Confidence: {advice['confidence']}")
            print(f"   • Score: {advice['score']:.1f}")
            print(f"   • Stochastic %K: {advice['stochastic_k']:.1f}")
            print(f"   • RSI: {advice['rsi']:.1f}")
            
            if advice['reasons']:
                print("   • Reasons:")
                for reason in advice['reasons']:
                    print(f"      - {reason}")

        except Exception as e:
            print(f"⚠️  Warning: Could not generate advice: {e}")
            advice = {"recommendation": "N/A", "stochastic_k": None, "rsi": None}

        # 6️⃣ Persistência em memória
        print("\n💾 Saving prediction to database...")
        try:
            prediction_id = save_prediction(
                model=forecast["model"],
                price=forecast_price,
                confidence=forecast["r2"],
                current_price=current_price,
                forecast_days=FORECAST_DAYS,
                risk_metrics=risk_metrics,
                recommendation=advice["recommendation"],
                stochastic_k=advice.get("stochastic_k"),
                rsi=advice.get("rsi")
            )
            print(f"✅ Prediction saved with ID: {prediction_id}")

            # Salvar métricas diárias
            save_daily_metrics(
                date=datetime.now().date().isoformat(),
                price=current_price,
                volatility=risk_metrics.get("volatility") if risk_metrics else None,
                rsi=advice.get("rsi"),
                stochastic_k=advice.get("stochastic_k")
            )

        except Exception as e:
            print(f"⚠️  Warning: Could not save prediction: {e}")

        # 7️⃣ Verificar previsões anteriores
        print("\n🔍 Checking previous predictions...")
        try:
            pending = check_pending_predictions()
            if len(pending) > 0:
                print(f"   • Found {len(pending)} predictions ready for validation")
                
                # Atualizar com preço atual (simplificado)
                for _, pred in pending.iterrows():
                    update_prediction_actual(pred["id"], current_price)
                    print(f"   • Updated prediction #{pred['id']}")
            else:
                print("   • No pending predictions to check")

        except Exception as e:
            print(f"⚠️  Warning: Could not check predictions: {e}")

        # 8️⃣ Mostrar acurácia histórica
        print("\n📈 Historical accuracy...")
        try:
            accuracy = get_prediction_accuracy(days_back=30)
            if accuracy:
                print(f"   • Mean Error: {accuracy['mean_error_pct']:.2f}%")
                print(f"   • Median Error: {accuracy['median_error_pct']:.2f}%")
                print(f"   • Predictions within 5% error: {accuracy['accuracy_95']:.1f}%")
                print(f"   • Predictions within 10% error: {accuracy['accuracy_90']:.1f}%")
                print(f"   • Total predictions analyzed: {accuracy['prediction_count']}")
            else:
                print("   • No historical data available yet")

        except Exception as e:
            print(f"⚠️  Warning: Could not calculate accuracy: {e}")

        # 9️⃣ Resumo final
        print("\n" + "=" * 80)
        print("📊 EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Current Price: ${current_price:,.2f}")
        print(f"Forecast (D+{FORECAST_DAYS}): ${forecast_price:,.2f} ({variation_pct:+.2f}%)")
        print(f"Recommendation: {advice['recommendation']}")
        print(f"Best Model: {forecast['model']} (R²={forecast['r2']:.4f})")
        
        if risk_metrics:
            print(f"Volatility: {risk_metrics['volatility']:.2%}")
            print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
        
        print("=" * 80)
        print("✅ Agent execution completed successfully")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ FATAL ERROR: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    try:
        run_agent_once()
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
