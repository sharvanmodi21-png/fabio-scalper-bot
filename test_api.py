import json, ccxt, sys
cfg = json.load(open('config.json'))
exchange = ccxt.binance({'apiKey': cfg['apiKey'], 'secret': cfg['secret'], 'enableRateLimit': True})
try:
    data = exchange.fetch_ohlcv('BTC/USDT', '1m', limit=1)
    print('Success', data)
except Exception as e:
    print('Error', e)
    sys.exit(1)
