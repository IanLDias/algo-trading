from binance.client import Client
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor
from config_var import access_key, secret_key

client = Client(access_key, secret_key)
btc_price = {'error': False}

def btc_trade_history(msg):
    ''' define how to process incoming WebSocket messages '''
    if msg['e'] != 'error':
        print(msg['c'])
        btc_price['last'] = msg['c']
        btc_price['bid'] = msg['b']
        btc_price['last'] = msg['a']
    else:
        btc_price['error'] = True

if __name__ == '__main__':
    bsm = BinanceSocketManager(client)
    conn_key = bsm.start_symbol_ticker_socket('BTCUSDT', btc_trade_history)
    bsm.start()
    # bsm.stop_socket(conn_key)
    # reactor.stop()

