import sqlite3
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
from config import config

db = sqlite3.connect('historical_data.db')
cursor = db.cursor()

cursor.execute("""DROP TABLE historical_prices""")
cursor.execute("""
    CREATE TABLE historical_prices(
        id SERIAL PRIMARY KEY,
        ticker_id TEXT NOT NULL,
        date VARCHAR(64) NOT NULL,
        high FLOAT(32) NOT NULL,
        low FLOAT(32) NOT NULL,
        open FLOAT(32) NOT NULL,
        close FLOAT(32) NOT NULL,
        volumeto FLOAT(32),
        volumefor FLOAT(32)
    )
""")

db.commit()

