from collections import defaultdict
from time import sleep
from datetime import datetime
import json
from typing import Dict
import logging

from websocket import create_connection

while True:
    # warning: this is not a production code, it is intended to collect data once
    last_max_ts = 0
    fb = open(f'csv/bw.{datetime.utcnow().isoformat()}.books.csv', 'w+')
    ft = open(f'csv/bw.{datetime.utcnow().isoformat()}.trades.csv', 'w+')

    try:
        print('connecting')
        ws = create_connection("wss://kline.BW.com/websocket")
        #for number, currency in [(281, "BTC_USDT")]:
        for number, currency in [(281, "BTC_USDT"), (280, "ETH_USDT"), (291, "EOS_USDT"), (307, "EOS_BTC"), (3229, "BSV_BTC")]:
            ws.send('{"dataType":"%s_ENTRUST_ADD_%s","dataSize":50,"action":"ADD"}' % (number, currency))
            ws.send('{"dataType":"%s_TRADE_%s","dataSize":50,"action":"ADD"}' % (number, currency))
        bid_books: Dict[str, Dict[str, str]] = defaultdict(dict)
        ask_books: Dict[str, Dict[str, str]] = defaultdict(dict)

        while True:
            data = json.loads(ws.recv())
            if isinstance(data[0], list) and data[0][0] == 'AE':
                # [['AE', '281', 'BTC_USDT', '1550691335', {'asks': [['18000', '0.001'], ...
                symbol = data[0][2]
                bid_books[symbol] = {r[0]: r[1] for r in data[0][5]['bids'] if float(r[1])}
                ask_books[symbol] = {r[0]: r[1] for r in data[0][4]['asks'] if float(r[1])}
            elif data[0] == 'E':
                # ['E', '281', '1550688421', 'BTC_USDT', 'BID', '3946.4638', '0']
                # ['E', '281', '1550688420', 'BTC_USDT', 'BID', '3945.4911', '0.9784']
                # ['E', '281', '1550688420', 'BTC_USDT', 'ASK', '3964.0789', '0']
                symbol = data[3]
                side = 1 if data[4] == 'BID' else 2
                updated_book = bid_books[symbol] if data[4] == 'BID' else ask_books[symbol]
                price = data[5]
                size = data[6]
                if float(size) == 0:
                    if price in updated_book:
                        del updated_book[price]
                else:
                    updated_book[price] = size
                # filter zeroes
                len_was = len(bid_books[symbol])
                was_b = bid_books[symbol]
                bid_books[symbol] = {k: v for k, v in bid_books[symbol].items() if float(v)}
                ask_books[symbol] = {k: v for k, v in ask_books[symbol].items() if float(v)}
                if len(bid_books[symbol]) != len_was:
                    print(0)
                top_bids = sorted(bid_books[symbol].items(), key=lambda x: float(x[0]), reverse=True)[:5]
                top_asks = sorted(ask_books[symbol].items(), key=lambda x: float(x[0]), reverse=False)[:5]
                s = f'{symbol};{datetime.utcnow().isoformat()};{json.dumps(top_bids)};{json.dumps(top_asks)}\n'
                #print('bw', symbol, top_bids[:1], top_asks[:1])
                fb.write(s)
            elif data[0] == 'T':
                # ['T', '281', '1550688422', 'BTC_USDT', 'ask', '3949.2194', '0.2841']
                # ['T', '281', '1550688421', 'BTC_USDT', 'bid', '3949.3572', '0.0036']
                t_dt = datetime.utcfromtimestamp(int(data[2]))
                symbol = data[3]
                side = 1 if data[4] == 'bid' else 2
                s = f'{symbol};;{t_dt};{datetime.utcnow().isoformat()};{data[5]};{data[6]};{side};\n'
                print('bw', s)
                ft.write(s)
            else:
                print(data)

    except Exception as e:
        logging.exception(e)
    finally:
        fb.close()
        ft.close()
        sleep(5)
