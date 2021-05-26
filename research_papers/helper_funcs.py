import psycopg2
from pathlib import Path as pathl
import sys
from datetime import datetime

#append relevant file paths
new_path = pathl('.')
parent = new_path.resolve().parent
sys.path.append(str(parent))

from Data.config import config

def get_data(symbols):
    "Given a list of ticker_ids, returns historical data. sep_data returns multiple "
    if isinstance(symbols, list):
        str_list = ""
        for index, symbol in enumerate(symbols):
            if index != 0:
                str_list += ", " + '\'' + symbol + '\'' 
            else:
                str_list += '\'' + symbol + '\'' 
    else:
        str_list = '\'' + symbols + '\'' 
    
    "Get historical data given a list of symbols"
    params = config()
    conn = psycopg2.connect(**params)
    cur = conn.cursor()
    print(f'Finding data for: {str_list}')
    cur.execute(f"""SELECT * FROM historical_prices
    WHERE ticker_id IN ({str_list})
    """)
    rows = cur.fetchall()
    return rows

def convert_unix_to_datetime(date_col):
    'Converts the unix dates into YYYY-MM-DD'
    int_list = list((map(int,date_col)))
    date_list = list(map(datetime.utcfromtimestamp, int_list))
    converted_dates = [date_list[i].strftime('%Y-%m-%d') for i in range(len(date_list))]
    return converted_dates

def separate_symbols(df):
    "Returns an individual df for each symbol. Sorted by alphabetical values of the symbols"
    list_dfs = []
    for symbol in sorted(df['symbol'].unique()):
        current_df = df[df['symbol'] == symbol]
        current_df = current_df.sort_values(by='date')
        list_dfs.append(current_df)
    return list_dfs

def plot_crypto(symbol):
    'Plot the graph given a symbol name'
    df_symbol = df[df['symbol'] == f'{symbol}']
    fig = go.Figure(data=go.Ohlc(x=df_symbol['date'],
                        open=df_symbol['open'],
                        high=df_symbol['high'],
                        low=df_symbol['low'],
                        close=df_symbol['close']))
    fig.update_layout(
    title=f'{symbol} currency')
    return fig.show()


if __name__ == '__main__':
    rows = get_data('BTC')
    rows = get_data(['BTC'])
    rows = get_data(['BTC', 'ETH'])