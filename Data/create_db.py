import psycopg2
from config import config

params = config()

conn = psycopg2.connect(**params)
cur = conn.cursor()

#---------- DELETED TABLE ------------------------
# The table 'tickers' contains all the current tickers from binance
# The list is found in ticker_list.txt, which are trading pairs to USDT
# cur.execute("""
#     CREATE TABLE IF NOT EXISTS tickers (
#         id INTEGER PRIMARY KEY,
#         symbol TEXT NOT NULL UNIQUE,
#         name TEXT NOT NULL UNIQUE
#     )""")
#
#cur.execute("""DROP TABLE tickers CASCADE""")
#-------------------------------------------------

# The table historical prices lists all currently found 
# past prices for tickers in the 'tickers' table
cur.execute("""DROP TABLE historical_prices""")
cur.execute("""
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

conn.commit()