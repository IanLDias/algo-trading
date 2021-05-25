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
cur.execute("""
    CREATE TABLE IF NOT EXISTS historical_prices(
        id INTEGER PRIMARY KEY,
        ticker_id TEXT NOT NULL,
        date VARCHAR(64) NOT NULL,
        high FLOAT(10) NOT NULL,
        low FLOAT(10) NOT NULL,
        open FLOAT(10) NOT NULL,
        close FLOAT(10) NOT NULL,
        volumeto INT NOT NULL,
        volumefor INT NOT NULL,
        FOREIGN KEY (ticker_id) REFERENCES tickers (symbol)
    )
""")

conn.commit()