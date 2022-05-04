import pandas as pd
import math
import os.path
import time
import logging
import sqlite3
from binance.client import Client
from datetime import timedelta, datetime
from dateutil import parser
from tqdm import tqdm_notebook #(Optional, used for progress-bars)

logging.basicConfig(filename='data_logging.txt', level=logging.INFO)
BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY_STAGING')   
BINANCE_API_SECRET = os.environ.get('BINANCE_SECRET_KEY_STAGING')

BINSIZES = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
BATCHSIZE = 750

binance_client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

def minutes_of_new_data(df, symbol, currency, kline_size):
    """ How much new data has to be added """
    if not df.empty:
        old = pd.to_datetime(df['close_time'].iloc[-1], unit='ms')
    else:
        old = datetime.strptime('1 Jan 2019', '%d %b %Y')
    # Gets current time
    new = pd.to_datetime(binance_client.get_klines(symbol=symbol+currency, interval=kline_size)[-1][0], unit='ms')
    return old, new

def get_all_binance(symbol, kline_size, df=None):
    """ Takes in the current data if it exists"""
    filename = '%s-%s-data.csv' % (symbol, kline_size)
    if not df.empty: 
        last_row = df['close_time'].iloc[-1]
        data_df = df
    else: 
        data_df = pd.DataFrame()
    oldest_point, newest_point = minutes_of_new_data(df, symbol, currency, kline_size)
    # How many minutes there are between the oldest and newest point
    delta_min = (newest_point - oldest_point).total_seconds()/60
    # How much data there is to collect
    available_data = math.ceil(delta_min/BINSIZES[kline_size])
    if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'): 
        logging.info('Downloading all available %s data for %s.' % (kline_size, symbol))
    else:  
        logging.info('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (delta_min, symbol, available_data, kline_size))
    klines = binance_client.get_historical_klines(symbol+currency, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S"))
    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    if len(data_df) > 0:
        temp_df = pd.DataFrame(data)
        data_df = pd.concat([data_df, temp_df])
    else: 
        data_df = data
    data_df.set_index('timestamp', inplace=True)
    return data_df

def save_to_db(table_name, symbol, interval):
    db = sqlite3.connect('historical_data.db')
    conn = sqlite3.connect('historical_data.db')
    # Check if the table already exists
    try:
        df = pd.read_sql(f"select * from {table_name}", conn)
        # append only
        # Get data only from oldest point of the previous data
        new_df = get_all_binance(symbol, interval, df)
        new_df.to_sql(table_name, conn, if_exists='replace', index=False)
        return
    except pd.io.sql.DatabaseError:
        df = get_all_binance(symbol, interval)
        df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()

if __name__ == '__main__':
    # Do this for BTCUSDT, ETHUSDT
    symbols = ['BTC', 'ETH']
    currency = 'USDT'
    interval = "5m"
    for symbol in symbols:
        table_name = symbol+interval+currency
        save_to_db(table_name, symbol, interval)