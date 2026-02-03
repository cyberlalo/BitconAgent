import sqlite3

conn = sqlite3.connect("memory.db")
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
