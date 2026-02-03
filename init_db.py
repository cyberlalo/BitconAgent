import sqlite3
import os
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "memory.db")

def init_db():
    """Inicializa o banco e garante que todas as tabelas e colunas existam"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # =========================
    # TABELA DE PREVISÕES
    # =========================
    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        model TEXT NOT NULL,
        predicted_price REAL NOT NULL,
        confidence REAL,
        actual_price REAL,
        current_price REAL,
        forecast_days INTEGER DEFAULT 7,
        volatility REAL,
        max_drawdown REAL,
        sharpe_ratio REAL,
        recommendation TEXT,
        stochastic_k REAL,
        rsi REAL,
        error_pct REAL,
        checked_at TEXT
    )
    """)

    # =========================
    # TABELA DE MÉTRICAS DIÁRIAS
    # =========================
    c.execute("""
    CREATE TABLE IF NOT EXISTS daily_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        price REAL NOT NULL,
        volatility REAL,
        rsi REAL,
        stochastic_k REAL,
        volume REAL,
        UNIQUE(date)
    )
    """)

    # =========================
    # Índices
    # =========================
    c.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_daily_metrics_date ON daily_metrics(date)")

    # =========================
    # Garantir colunas extras (para atualizar banco antigo)
    # =========================
    existing_columns = [row[1] for row in c.execute("PRAGMA table_info(predictions)")]
    required_columns = ["actual_price", "error_pct", "checked_at"]

    for col in required_columns:
        if col not in existing_columns:
            c.execute(f"ALTER TABLE predictions ADD COLUMN {col} REAL")
            print(f"✅ Coluna {col} adicionada ao banco")

    conn.commit()
    conn.close()
    print("✅ Database initialized at:", DB_PATH)


def safe_read_sql(query):
    """
    Executa uma query no banco de forma segura, retornando DataFrame vazio se algo falhar
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print("⚠️ Falha ao executar query:", e)
        return pd.DataFrame()


def get_prediction_accuracy(days_back=90):
    """Exemplo seguro de leitura de acurácia"""
    query = f"""
        SELECT 
            model,
            predicted_price,
            actual_price,
            confidence,
            error_pct,
            timestamp,
            recommendation
        FROM predictions
        WHERE actual_price IS NOT NULL
        AND timestamp >= datetime('now','-{days_back} days')
        ORDER BY timestamp DESC
    """
    df = safe_read_sql(query)
    if df.empty:
        return None

    # Aqui você pode calcular métricas como mean_error_pct, accuracy_95 etc.
    mean_error_pct = df['error_pct'].mean() if 'error_pct' in df else None
    return {
        "prediction_count": len(df),
        "mean_error_pct": mean_error_pct,
        # outros campos...
    }
