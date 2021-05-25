# Run this every day to get current prices
import config
import requests
import psycopg2
from datetime import datetime
import logging

logging.basicConfig(filename='Data/sql_input.log', level=logging.INFO,
                    format='%(levelname)s:%(message)s')
params = config.config()
conn = psycopg2.connect(**params)
cur = conn.cursor()

cur.execute("""SELECT DISTINCT ticker_id FROM historical_prices""")
symbols_in_db = cur.fetchall()
symbols_in_db = [symbols[0] for symbols in symbols_in_db]

cur.execute("""SELECT DISTINCT date FROM historical_prices""")
current_dates = cur.fetchall()
current_dates = [datetime.utcfromtimestamp(int(date[0])) for date in current_dates]

for i, symbol in enumerate(symbols_in_db):
    url = f'{config.BASE_URL_COMP}data/v2/histoday?fsym={symbol}&tsym=USD&limit=1&api_key={config.API_KEY_COMP}'
    resp = requests.get(url)
    day = resp.json()['Data']['Data'][-1]
    temp_date = datetime.utcfromtimestamp(day['time'])
    if temp_date in current_dates:
        logging.info(f"The date: {temp_date} already exists in the database")
        break
    else:
        if i == 0:
            logging.info(f"The date: {temp_date} has been added to the database")
        cur.execute("""SELECT DISTINCT date FROM historical_prices""")
        cur.execute(f'''INSERT INTO historical_prices (ticker_id, date, high, low, open, close, volumeto, volumefor)
                VALUES ('{symbol}','{day['time']}','{day['high']}','{day['low']}','
                {day['open']}','{day['close']}','{day['volumeto']}','{day['volumefrom']}')''')
        conn.commit()

cur.close()