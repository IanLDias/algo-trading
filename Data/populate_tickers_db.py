import psycopg2
import config
import asyncio
import asyncio
import logging
import aiohttp
from aiohttp import ClientSession
import sys
import pandas as pd

df = pd.read_csv('ticker_list')
print(df)
params = config.config()
conn = psycopg2.connect(**params)
cur = conn.cursor()

cur.execute("""SELECT * FROM tickers""")
rows = cur.fetchall()

#test for already existing symbols
current_symbols = [row['symbol'] for row in rows]

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.DEBUG,
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

logger = logging.getLogger("areq")
logging.getLogger("chardet.charsetprober").disabled = True
url = f'{config.BASE_URL_COMP}data/v2/histoday?fsym={symbol}&tsym=USD&limit=2000&api_key={config.API_KEY_COMP}'
print(url)
# async def request_pull(symbol, session:ClientSession):
#     'Makes a url based on a symbol and does a get request'
#     url = f'{config.BASE_URL_COMP}data/v2/histoday?fsym={symbol}&tsym=USD&limit=2000&api_key={config.API_KEY_COMP}'
#     resp = await session.request(method="GET", url=url)
#     json_data = await resp.json()
#     #waits for the request to finish. If isn't finished moves onto the next task
#     try:
#         data = json_data['Data']['Data']
#         return data
#     except:
#         pass

# async def writing_sql(data, symbol):
#     'Writes the json request data into sqlite'
#     for day in data:
#         print(symbol)
#         cur.execute(f'''INSERT INTO crypto_OHLCV (crypto_id, date, high, low, open, close, volumeto, volumefor)
#         VALUES ('{symbol}','{day['time']}','{day['high']}','{day['low']}','
#         {day['open']}','{day['close']}','{day['volumeto']}','{day['volumefrom']}')''')
        

# async def chain(symbol):
#     "Runs two processe, p1 and p2, asynchronously. Waits for one to finish before moving on"
#     try:
#         async with ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False)) as session:
#             p1 = await request_pull(symbol, session=session)
#             if p1:
#                 p2 = await writing_sql(data=p1, symbol=symbol)
#     except:
#         pass

# async def main():
#     await asyncio.gather(*(chain(symbol) for symbol in current_symbols))
    

# asyncio.run(main())
#conn.commit()