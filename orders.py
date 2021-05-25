from os import access
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceOrderException
from config_var import access_key, secret_key

client = Client(access_key, secret_key)

#demo url
#REMOVE FOR PROD
client.API_URL = 'https://testnet.binance.vision/api'

# --- Sell an order ---
# order = client.create_order(symbol='BTCUSDT', side='SELL',type='MARKET', quantity=1)
# print(order)
client = Client(access_key, secret_key)
prices = client.get_all_tickers()
