# memory.py
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

DB_PATH = "memory.db"


def get_connection():
    """Get database connection"""
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    """Initialize database with enhanced schema"""
    conn = get_connection()
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
    CREATE INDEX IF NOT EXISTS idx_daily_metrics_date 
    ON daily_metrics(date)
    """)

    conn.commit()
    conn.close()


def save_prediction(model, price, confidence, current_price=None, 
                   forecast_days=7, risk_metrics=None, recommendation=None,
                   stochastic_k=None, rsi=None):
    """Save prediction with full context"""
    conn = get_connection()
    c = conn.cursor()

    volatility = None
    max_drawdown = None
    sharpe_ratio = None

    if risk_metrics:
        volatility = risk_metrics.get("volatility")
        max_drawdown = risk_metrics.get("max_drawdown")
        sharpe_ratio = risk_metrics.get("sharpe_ratio")

    c.execute("""
    INSERT INTO predictions (
        timestamp, model, predicted_price, confidence,
        current_price, forecast_days, volatility, max_drawdown,
        sharpe_ratio, recommendation, stochastic_k, rsi
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        model,
        price,
        confidence,
        current_price,
        forecast_days,
        volatility,
        max_drawdown,
        sharpe_ratio,
        recommendation,
        stochastic_k,
        rsi
    ))

    conn.commit()
    prediction_id = c.lastrowid
    conn.close()

    return prediction_id


def update_prediction_actual(prediction_id, actual_price):
    """Update prediction with actual price and calculate error"""
    conn = get_connection()
    c = conn.cursor()

    # Buscar previsão
    c.execute("""
    SELECT predicted_price FROM predictions WHERE id = ?
    """, (prediction_id,))

    result = c.fetchone()
    if not result:
        conn.close()
        return False

    predicted_price = result[0]
    error_pct = abs((actual_price - predicted_price) / actual_price) * 100

    c.execute("""
    UPDATE predictions 
    SET actual_price = ?, error_pct = ?, checked_at = ?
    WHERE id = ?
    """, (actual_price, error_pct, datetime.now().isoformat(), prediction_id))

    conn.commit()
    conn.close()

    return True


def get_prediction_accuracy(days_back=30):
    """Analyze prediction accuracy for completed forecasts"""
    conn = get_connection()
    
    query = """
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
    AND timestamp >= datetime('now', '-{} days')
    ORDER BY timestamp DESC
    """.format(days_back)
    
    df = pd.read_sql_query(query, conn)
    conn.close()

    if len(df) == 0:
        return None

    # Calcular métricas gerais
    accuracy = {
        "mean_error_pct": df["error_pct"].mean(),
        "median_error_pct": df["error_pct"].median(),
        "std_error_pct": df["error_pct"].std(),
        "min_error_pct": df["error_pct"].min(),
        "max_error_pct": df["error_pct"].max(),
        "prediction_count": len(df),
        "accuracy_90": (df["error_pct"] < 10).sum() / len(df) * 100,  # % com erro < 10%
        "accuracy_95": (df["error_pct"] < 5).sum() / len(df) * 100,   # % com erro < 5%
    }

    # Métricas por modelo
    model_accuracy = df.groupby("model").agg({
        "error_pct": ["mean", "median", "count"]
    }).round(2)

    accuracy["by_model"] = model_accuracy.to_dict()

    return accuracy


def get_recent_predictions(limit=10):
    """Get recent predictions"""
    conn = get_connection()
    
    query = """
    SELECT 
        id,
        timestamp,
        model,
        predicted_price,
        current_price,
        actual_price,
        confidence,
        recommendation,
        error_pct,
        forecast_days
    FROM predictions 
    ORDER BY timestamp DESC
    LIMIT ?
    """
    
    df = pd.read_sql_query(query, conn, params=(limit,))
    conn.close()

    return df


def get_model_performance():
    """Get overall performance statistics by model"""
    conn = get_connection()
    
    query = """
    SELECT 
        model,
        COUNT(*) as total_predictions,
        AVG(confidence) as avg_confidence,
        AVG(error_pct) as avg_error,
        MIN(error_pct) as best_error,
        MAX(error_pct) as worst_error
    FROM predictions
    WHERE actual_price IS NOT NULL
    GROUP BY model
    ORDER BY avg_error ASC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def save_daily_metrics(date, price, volatility=None, rsi=None, 
                       stochastic_k=None, volume=None):
    """Save daily market metrics"""
    conn = get_connection()
    c = conn.cursor()

    try:
        c.execute("""
        INSERT INTO daily_metrics (date, price, volatility, rsi, stochastic_k, volume)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            price = excluded.price,
            volatility = excluded.volatility,
            rsi = excluded.rsi,
            stochastic_k = excluded.stochastic_k,
            volume = excluded.volume
        """, (date, price, volatility, rsi, stochastic_k, volume))

        conn.commit()
    except Exception as e:
        print(f"Erro ao salvar métricas diárias: {e}")
    finally:
        conn.close()


def get_historical_metrics(days=30):
    """Get historical daily metrics"""
    conn = get_connection()
    
    query = """
    SELECT * FROM daily_metrics
    WHERE date >= date('now', '-{} days')
    ORDER BY date DESC
    """.format(days)
    
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def check_pending_predictions():
    """
    Check predictions that should have results by now
    Returns list of predictions ready to be validated
    """
    conn = get_connection()
    
    query = """
    SELECT 
        id,
        timestamp,
        forecast_days,
        model,
        predicted_price
    FROM predictions
    WHERE actual_price IS NULL
    AND datetime(timestamp, '+' || forecast_days || ' days') <= datetime('now')
    ORDER BY timestamp ASC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def get_recommendation_accuracy():
    """Analyze accuracy of buy/sell/hold recommendations"""
    conn = get_connection()
    
    query = """
    SELECT 
        recommendation,
        COUNT(*) as count,
        AVG(CASE 
            WHEN actual_price > current_price THEN 1 
            ELSE 0 
        END) * 100 as pct_price_increased,
        AVG((actual_price - current_price) / current_price * 100) as avg_return_pct
    FROM predictions
    WHERE actual_price IS NOT NULL 
    AND current_price IS NOT NULL
    AND recommendation IS NOT NULL
    GROUP BY recommendation
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def cleanup_old_predictions(days_to_keep=90):
    """Remove old predictions to keep database size manageable"""
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
    DELETE FROM predictions
    WHERE timestamp < datetime('now', '-{} days')
    """.format(days_to_keep))

    deleted = c.rowcount
    conn.commit()
    conn.close()

    return deleted
