import pandas as pd
import pandas_datareader as pdr

tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = tickers['Symbol'].to_list()
tickers = [i.replace('.', '-') for i in tickers]

# Errors
tickers.pop(474)
tickers.pop(489)

df = pdr.DataReader(tickers, 'yahoo')

df.to_hdf('notebooks/stocks/Reinforcement_learning/data/data.h5', key='yfinance')