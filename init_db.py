import sqlite3

def init_database():
    """Initialize database with enhanced schema"""
    conn = sqlite3.connect("memory.db")
    c = conn.cursor()

    # Tabela de previsões
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

    # Tabela de métricas diárias
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

    # Índices para performance
    c.execute("""
    CREATE INDEX IF NOT EXISTS idx_predictions_timestamp 
    ON predictions(timestamp)
    """)

    c.execute("""
    CREATE INDEX IF NOT EXISTS idx_predictions_model 
    ON predictions(model)
    """)

    c.execute("""
    CREATE INDEX IF NOT EXISTS idx_daily_metrics_date 
    ON daily_metrics(date)
    """)

    conn.commit()
    conn.close()
    
    print("✅ Database initialized successfully")

if __name__ == "__main__":
    init_database()
