import sqlite3
import os

# Caminho seguro para o banco (mesma pasta do script)
DB_PATH = os.path.join(os.path.dirname(__file__), "memory.db")

def init_db():
    """
    Inicializa o banco de dados SQLite com tabelas e índices necessários
    para armazenar previsões, métricas diárias e performance do agente.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # =====================
        # TABELA DE PREVISÕES
        # =====================
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

        # =====================
        # TABELA DE MÉTRICAS DIÁRIAS
        # =====================
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

        # =====================
        # ÍNDICES
        # =====================
        c.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_daily_metrics_date ON daily_metrics(date)")

        # Salvar alterações e fechar conexão
        conn.commit()
        conn.close()

        print("✅ Database initialized successfully at:", DB_PATH)

    except sqlite3.Error as e:
        print("❌ Erro ao inicializar o banco de dados:", e)
        if conn:
            conn.close()
