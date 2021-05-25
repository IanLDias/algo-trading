from os import access
from binance.client import Client
from binance.exceptions import BinanceAPIException
from config_var import access_key, secret_key

client = Client(access_key, secret_key)
prices = client.get_all_tickers()
#demo url
#REMOVE FOR PROD
client.API_URL = 'https://testnet.binance.vision/api'
#print(client.get_account())
#print(client.get_asset_balance(asset='BTC'))
print(client.get_symbol_ticker(symbol='BTCUSDT'))

#Looking at USDtether to crypto exchange.
#Change to BTC as project develops
# if __name__ == '__main__':
#     with open('Data/ticker_list.txt', 'w') as f:
#         for coin in prices:
#             if 'USDT' in coin['symbol']:
#                 f.write(coin['symbol'][:-4] + "\n")
