# ₿ Bitcoin Quantitative Analysis Agent

Um agente autônomo de análise quantitativa para Bitcoin usando múltiplos modelos matemáticos, indicadores técnicos e gestão de risco.

## 🚀 Características

### Modelos Preditivos
- **Linear**: Tendência de longo prazo
- **Polinomial**: Captura mudanças de direção
- **Polinomial + Seno**: Modela ciclos de mercado
- **Média Móvel**: Suavização de volatilidade
- **Exponential MA**: Pesos para dados recentes

### Indicadores Técnicos
- **Oscilador Estocástico**: Momentum de curto prazo
- **RSI** (Relative Strength Index): Força relativa
- **Intervalos de Confiança**: Estimativas probabilísticas

### Gestão de Risco
- **Volatilidade Anualizada**: Medida de incerteza
- **Sharpe Ratio**: Retorno ajustado ao risco
- **Maximum Drawdown**: Maior perda histórica
- **VaR 95%**: Value at Risk

### Sistema de Memória
- Persistência de previsões
- Análise de acurácia histórica
- Validação automática de previsões passadas
- Métricas por modelo

## 📋 Requisitos

```txt
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
plotly>=5.14.0
scipy>=1.10.0
requests>=2.31.0
```

## 🔧 Instalação

```bash
# Clone o repositório
git clone <seu-repo>
cd bitcoin-agent

# Instale as dependências
pip install -r requirements.txt

# Inicialize o banco de dados
python init_db.py
```

## 💻 Uso

### 1. Interface Web (Streamlit)

```bash
streamlit run streamlit_app.py
```

Acesse: `http://localhost:8501`

### 2. Execução Autônoma (Cron)

```bash
python agent_runner.py
```

**Configurar cron para execução diária:**

```bash
# Editar crontab
crontab -e

# Adicionar linha (executa todo dia às 9h)
0 9 * * * /usr/bin/python3 /path/to/agent_runner.py >> /path/to/logs/agent.log 2>&1
```

### 3. Uso Programático

```python
from bitcoin_agent import BitcoinAnalysisAgent

# Criar agente
agent = BitcoinAnalysisAgent()

# Coletar dados
agent.fetch_bitcoin_data(days=180)

# Análise
results = agent.find_approximations()
risk_metrics = agent.calculate_risk_metrics()
forecast = agent.forecast_with_confidence(forecast_days=7)
advice = agent.investment_advice()

print(f"Previsão: ${forecast['forecast']:,.2f}")
print(f"Recomendação: {advice['recommendation']}")
```

## 📊 Estrutura do Projeto

```
bitcoin-agent/
│
├── bitcoin_agent.py      # Classe principal do agente
├── agent_runner.py       # Script de execução autônoma
├── memory.py            # Sistema de persistência
├── init_db.py           # Inicialização do banco
├── streamlit_app.py     # Interface web
├── memory.db            # Banco SQLite (gerado)
├── requirements.txt     # Dependências
└── README.md           # Documentação
```

## 🎯 Funcionalidades Detalhadas

### Previsão com Intervalo de Confiança

```python
forecast = agent.forecast_with_confidence(forecast_days=7)

# Retorna:
{
    "model": "poly_sine",
    "forecast": 95000.0,
    "lower_bound": 90000.0,
    "upper_bound": 100000.0,
    "r2": 0.9234,
    "std_error": 2500.0
}
```

### Recomendação Multi-Indicador

```python
advice = agent.investment_advice()

# Retorna:
{
    "recommendation": "Acumular",
    "score": 2.5,
    "confidence": "alta",
    "reasons": [
        "Estocástico em zona de sobrevenda (<20)",
        "RSI indica sobrevenda (<30)",
        "Tendência de alta moderada"
    ],
    "stochastic_k": 18.5,
    "rsi": 28.3,
    "trend_slope": 150.2
}
```

### Análise de Acurácia

```python
from memory import get_prediction_accuracy

accuracy = get_prediction_accuracy(days_back=30)

# Retorna:
{
    "mean_error_pct": 3.2,
    "median_error_pct": 2.8,
    "accuracy_95": 75.0,  # 75% com erro < 5%
    "accuracy_90": 85.0,  # 85% com erro < 10%
    "prediction_count": 20
}
```

## 📈 Exemplo de Saída do Agent Runner

```
================================================================================
🤖 Bitcoin Analysis Agent - Execution started: 2026-02-03T10:00:00
================================================================================

📊 Fetching Bitcoin data...
✅ Successfully fetched 180 data points

🔬 Running model analysis...
✅ 6 models analyzed

⚠️  Calculating risk metrics...
   • Volatility: 65.30%
   • Max Drawdown: -23.45%
   • Sharpe Ratio: 1.42

🔮 Generating 7-day forecast...
   • Model: poly_sine
   • Current Price: $92,450.00
   • Forecast (D+7): $95,200.00
   • Variation: +2.97%
   • Confidence Interval: $90,100.00 - $100,300.00
   • R²: 0.9234

💡 Investment recommendation...
   • Recommendation: Comprar moderadamente
   • Confidence: média
   • Score: 1.5
   • Stochastic %K: 35.2
   • RSI: 52.8
   • Reasons:
      - Tendência de alta moderada
      - Volatilidade moderada

💾 Saving prediction to database...
✅ Prediction saved with ID: 42

📈 Historical accuracy...
   • Mean Error: 3.21%
   • Median Error: 2.85%
   • Predictions within 5% error: 75.0%
   • Predictions within 10% error: 85.0%
   • Total predictions analyzed: 20

================================================================================
📊 EXECUTION SUMMARY
================================================================================
Current Price: $92,450.00
Forecast (D+7): $95,200.00 (+2.97%)
Recommendation: Comprar moderadamente
Best Model: poly_sine (R²=0.9234)
Volatility: 65.30%
Sharpe Ratio: 1.42
================================================================================
✅ Agent execution completed successfully
================================================================================
```

## 🔍 Validação de Modelos

O agente inclui validação cruzada:

```python
validation = agent.validate_models(train_ratio=0.8)

# Retorna erro em dados não vistos
{
    "linear": {"mse": 1250000, "rmse": 1118, "mae": 892},
    "polynomial": {"mse": 980000, "rmse": 990, "mae": 756},
    "poly_sine": {"mse": 750000, "rmse": 866, "mae": 680}
}
```

## ⚠️ Avisos Importantes

### Limitações

1. **Dados Históricos**: Modelos baseados no passado não garantem resultados futuros
2. **Eventos Externos**: Não considera notícias, regulação, hacks
3. **Volatilidade**: Bitcoin pode invalidar previsões rapidamente
4. **API Limits**: CoinGecko tem rate limits (50 chamadas/min)

### Melhores Práticas

- ✅ Use como **ferramenta de suporte**, não como única fonte
- ✅ Combine com análise fundamental
- ✅ Considere seu perfil de risco
- ✅ Diversifique investimentos
- ✅ Nunca invista mais do que pode perder
- ✅ Valide previsões regularmente

## 🔄 Roadmap

### Próximas Versões

- [ ] Integração com exchanges (Binance, Coinbase)
- [ ] Machine Learning (LSTM, XGBoost)
- [ ] Análise de sentimento (Twitter, Reddit)
- [ ] Métricas on-chain (hash rate, volume)
- [ ] Sistema de alertas (email, Telegram)
- [ ] Backtesting automatizado
- [ ] Suporte a múltiplas criptomoedas
- [ ] API REST para integração

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto é licenciado sob a MIT License.

## 👤 Autor

**Eduardo Araujo**

- GitHub: [@eduardoaraujo](https://github.com/cyberlalo)
- Email: laloarauxo@gmail.com

## 🙏 Agradecimentos

- CoinGecko API por dados gratuitos
- Comunidade SciPy/NumPy
- Streamlit pelo framework incrível

---

**Disclaimer**: Este software é fornecido "como está", sem garantias. O autor não se responsabiliza por perdas financeiras. Use por sua conta e risco.
