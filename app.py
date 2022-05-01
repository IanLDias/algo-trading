import csv
import os
import sqlite3
DB_PATH = os.environ.get("DB_PATH")

# Adds symbols to the tickers table

def insert_data(DB_FILE, tickers):
    conn = sqlite3.connect(DB_PATH)
    
    for ticker in tickers:
        insert_ticker_data(conn, ticker)
    conn.commit()
    conn.close()

def insert_ticker_data(conn, ticker):
    insert_query = """INSERT INTO tickers 
                    (symbol)
                    values (?);"""
    data = (ticker)
    cur = conn.cursor()
    cur.execute(insert_query, (data,))
    conn.commit()
    print(f"Added {ticker} to tickers table")

if __name__ == '__main__':
    with open('Data/ticker_list.csv') as f:
        reader = csv.reader(f)
        tickers = list(reader)
    insert_data(DB_PATH, tickers[0])
    


