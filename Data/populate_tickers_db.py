import psycopg2
import config
import asyncio
import asyncio
import aiohttp
from aiohttp import ClientSession
import logging


params = config.config()
conn = psycopg2.connect(**params)
cur = conn.cursor()


#Adding new symbols by importing ticker_list.txt
f = open('Data/ticker_list.txt', 'r')
current_symbols, _ = zip(*[x.split('\n') for x in f.readlines()])

cur.execute("""SELECT DISTINCT ticker_id FROM historical_prices""")
symbols_in_db = cur.fetchall()
new_symbols = []
for i in current_symbols:
    if i not in symbols_in_db:
        new_symbols.append(i)


async def request_pull(symbol, session:ClientSession):
    'Makes a url based on a symbol and does a get request'
    url = f'{config.BASE_URL_COMP}data/v2/histoday?fsym={symbol}&tsym=USD&limit=2000&api_key={config.API_KEY_COMP}'
    resp = await session.request(method="GET", url=url)
    json_data = await resp.json()
    #waits for the request to finish. If isn't finished moves onto the next task
    try:
        data = json_data['Data']['Data']
        return data
    except:
        pass


async def writing_sql(data, symbol):
    'Writes the json request data into sqlite'
    print(f"symbol: {symbol}, length: {len(data)}")
    for day in data:
        cur.execute(f'''INSERT INTO historical_prices (ticker_id, date, high, low, open, close, volumeto, volumefor)
        VALUES ('{symbol}','{day['time']}','{day['high']}','{day['low']}','
        {day['open']}','{day['close']}','{day['volumeto']}','{day['volumefrom']}')''')
    #un-comment to add new data
    conn.commit()


async def chain(symbol):
    "Runs two processe, p1 and p2, asynchronously. Waits for one to finish before moving on"
    try:
        async with ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            p1 = await request_pull(symbol, session=session)
            if p1:
                p2 = await writing_sql(data=p1, symbol=symbol)
    except:
        pass

async def main():
    await asyncio.gather(*(chain(symbol) for symbol in new_symbols))

if __name__ == '__main__':
    print(f'Adding these new cryptocurrencies: {new_symbols}')
    asyncio.run(main())
    conn.commit()
    cur.close()