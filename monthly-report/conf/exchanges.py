HEADERS = {
    "bitmart": [
        ["symbol", "local_time", "bids", "asks"],  # book
        ["symbol", "_", "exch_time", "local_time", "price", "size", "side", "amount"]  # trade
    ],
    "bitz": [
        ["symbol", "local_time", "bids", "asks", 'exch_time'],  # book
        ["symbol", "trade_id", "exch_time", "local_time", "price", "size", "side", "send_time"]  # trade
    ],
    "bw": [
        ["symbol", "local_time", "bids", "asks"],  # book
        ["symbol", "_", "exch_time", "local_time", "price", "size", "side", "-"]  # trade
    ],
    "coinbene": [
        ["symbol", "local_time", "bids", "asks", "exch_time"],  # book
        ["symbol", "batch_ts", "exch_time", "local_time", "price", "size", "side", "_"]  # trade
    ],
    "cointiger": [
        ["symbol", "local_time", "bids", "asks", "exch_time"],  # book
        ["symbol", "batch_ts", "exch_time", "local_time", "price", "size", "side", "amount", "_"]  # trade
    ],
    "hitbtc": [
        ["symbol", "local_time", "bids", "asks", "exch_time", "sequence"],  # book
        ["symbol", "trade_id", "exch_time", "local_time", "price", "size", "side", "_"]  # trade
    ],
    "huobi": [
        ["symbol", "local_time", "bids", "asks", "exch_time", "msg_ts"],   # book
        ["symbol", "sequence", "exch_time", "local_time", "price", "size", "side", "order_id", "tick_ts"]  # trade
    ],
    "idax": [
        ["symbol", "local_time", "bids", "asks", "exch_time"],  # book
        ["symbol", "trade_id", "exch_time", "local_time", "price", "size", "side", "_"]  # trade
    ],
    "lbank": [
        ["symbol", "local_time", "bids", "asks", "exch_time", "_"],  # book
        ["symbol", "_", "exch_time", "local_time", "price", "size", "side", "-", "send_time"]  # trade
    ],
    "livecoin": [
        ["symbol", "local_time", "bids", "asks"],  # book
        ["symbol", "trade_id", "exch_time", "local_time", "price", "size", "side", "order_buy_id", "order_sell_id"]  # trade
    ],
    "okex": [
        ["symbol", "local_time", "bids", "asks", "exch_time"],  # book
        ["symbol", "trade_id", "exch_time", "local_time", "price", "size", "side", "_"]  # trade
    ],
    "zb": [
        ["symbol", "local_time", "bids", "asks", "exch_time"],  # book
        ["symbol", "trade_id", "exch_time", "local_time", "price", "size", "side", "_"]  # trade
    ],
}
