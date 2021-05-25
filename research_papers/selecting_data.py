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
    "Given a list of ticker_ids, returns historical data"
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

if __name__ == '__main__':
    rows = get_data('BTC')
    rows = get_data(['BTC'])
    rows = get_data(['BTC', 'ETH'])