# memory.py
import sqlite3
from datetime import datetime

DB_PATH = "memory.db"


def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        timestamp TEXT,
        model TEXT,
        predicted_price REAL,
        confidence REAL
    )
    """)

    conn.commit()
    conn.close()


def save_prediction(model, price, confidence):
    conn = get_connection()
    c = conn.cursor()

    c.execute(
        """
        INSERT INTO predictions (timestamp, model, predicted_price, confidence)
        VALUES (?, ?, ?, ?)
        """,
        (datetime.now().isoformat(), model, price, confidence)
    )

    conn.commit()
    conn.close()
