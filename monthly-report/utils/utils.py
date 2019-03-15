import os
import gzip
import json
import logging
import pandas as pd
pd.options.display.max_columns = 100
import datetime as dt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

import seaborn as sns
import matplotlib.pyplot as plt

from ipywidgets import interact

import warnings

warnings.filterwarnings('ignore')

plotly.offline.init_notebook_mode(connected=True)

import numpy as np

logging.basicConfig(format='%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s',
                        level=logging.DEBUG)


class ExchangeDataLoader:
    
    def __init__(self, stock, path, books_columns, trades_columns, 
                 start_date=dt.datetime(2019, 1, 1),
                 end_date=None,
                 books_compression=None, 
                 trades_compression=None,
                 time_column='exch_time'):
        self.path = path
        self.files = os.listdir(path)
        self.stock = stock
        self.start_date = start_date - dt.timedelta(1)
        self.end_date = end_date if end_date is not None else dt.detetime.now()
        self.books_columns = books_columns
        self.trades_columns = trades_columns
        self.books_compression = books_compression
        self.trades_compression = trades_compression
        self.time_column = time_column
        
    def load_books(self):
        files = [x for x in self.files if 'books' in x]
        files = [x for x in files if (pd.to_datetime(x.split('.')[1]) > self.start_date)
                                    and (pd.to_datetime(x.split('.')[1]) < (self.end_date))]
        df = pd.DataFrame()
        
        for file in files:
            try:
                df_file = pd.read_csv(os.path.join(self.path, file), sep=';', header=None, 
                                      compression=self.books_compression)
                df = pd.concat([df, df_file])
                print(file, 'Ok')
            except Exception as e:
                print(file, e)
        
        df.columns = self.books_columns
        df.loc[:, self.time_column] = pd.to_datetime(df[self.time_column])
        df = df[(df[self.time_column] >= self.start_date) & (df[self.time_column] < self.end_date)]
        df.loc[:, 'bids'] = df['bids'].apply(json.loads)
        df.loc[:, 'asks'] = df['asks'].apply(json.loads)
        df.loc[:, 'bids'] = df['bids'].apply(lambda rows: [(float(row[0]), float(row[1])) for row in rows])
        df.loc[:, 'asks'] = df['asks'].apply(lambda rows: [(float(row[0]), float(row[1])) for row in rows])
        return df.sort_values(by=self.time_column).reset_index(drop=True)
        #return df.sort_values(by=self.time_column).set_index([self.time_column], drop=True)
    
    def load_trades(self):
        files = [x for x in self.files if 'trades' in x]
        files = [x for x in files if (pd.to_datetime(x.split('.')[1]) > self.start_date)
                                    and (pd.to_datetime(x.split('.')[1]) < (self.end_date))]
        df = pd.DataFrame()
        for file in files:
            try:
                df_file = pd.read_csv(os.path.join(self.path, file), sep=';', header=None, 
                                      compression=self.trades_compression)
                df = pd.concat([df, df_file])
                print(file, 'Ok')
            except Exception as e:
                print(file, e)
        df.columns = self.trades_columns
        df.loc[:, self.time_column] = pd.to_datetime(df[self.time_column])
        df.loc[:, 'side'] = df['side'].map({1: 2, 2: 1})
        df = df[(df[self.time_column] >= self.start_date) & (df[self.time_column] < self.end_date)]
        return df.sort_values(by=self.time_column).reset_index(drop=True)
        #return df.sort_values(by=self.time_column).set_index([self.time_column], drop=True)

    
# Calculate best bid&ask
def get_best(x):
    
    size = 0
    i = 0
#     bs = float(x[0][0])
    while size == 0:
        if i == len(x):
            return np.nan
        bs = float(x[i][0])
        size = float(x[i][1])
        i += 1
    return bs
    

def order_book_percent_liquidity(book, perc=3):
    book = json.loads(book)
    price_0 = book[0][0]
    size_cum = 0
    for (price, size) in book:
        if price > price_0 * (1 + perc / 100) or price < price_0 * (1 - perc / 100):
            break
        else:
            size_cum += size
    
    return size_cum


def handy_liquidity(bids, asks, perc):
    
    liq = 0
    
    invsizeask = 1 / asks[0][1]
    invsizebid = 1 / bids[0][1]
    
    midprice = (bids[0][0] * invsizebid + asks[0][0] * invsizeask ) / (invsizeask + invsizebid)
    bid_thr = midprice * (1 - perc / 100)
    ask_thr = midprice / (1 - perc / 100)
    
    for level in bids:
        price = level[0]
        size = level[1]
        if price >= bid_thr:
            liq += size
        else:
            break
    
    for level in asks:
        price = level[0]
        size = level[1]
        if price <= ask_thr:
            liq += size
        else:
            break
    
    return liq


def parse_book(book):
    prices, volumes = zip(*book)
    return prices, volumes


def price_by_volume(book, volume):
    volume = abs(volume)
    prices, volumes = parse_book(book)

    cumvol = 0
    wprice = 0
    i = 0

    while (cumvol < volume) and (i < len(book)):
        cumvol_old = cumvol
        add = min(volumes[i], volume - cumvol)
        cumvol += add
        wprice = (wprice * cumvol_old + prices[i] * add) / cumvol
        i += 1

    return wprice


def spread_by_volume(bids, asks, volume):
    bid = price_by_volume(bids, volume)
    ask = price_by_volume(asks, volume)
    return (ask - bid) / (ask + bid) * 2


def vpin(trades, packet_volume, n_packets, time_column='exch_time'):

    trades['block'] = np.floor(trades['size'].cumsum() / packet_volume).astype(int)
    
    buy_bal = trades[trades['side'] == '2'].groupby('block')[[time_column, 'size']].agg({time_column: 'last',
                                                                                      'size': 'sum'})
    sell_bal = trades[trades['side'] == '1'].groupby('block')[[time_column, 'size']].agg({time_column: 'last',
                                                                                        'size': 'sum'})
    full_idx = list(set(list(buy_bal.index) + list(sell_bal.index)))
    sell_bal = sell_bal.loc[full_idx]
    sell_bal.loc[buy_bal.index, time_column] = buy_bal[time_column]
    sell_bal.loc[:, 'size'] = sell_bal['size'].fillna(0)
    buy_bal = buy_bal.loc[full_idx]
    buy_bal.loc[sell_bal.index, time_column] = sell_bal[time_column]
    buy_bal.loc[:, 'size'] = buy_bal['size'].fillna(0)

    disbalance = ((buy_bal['size'] - sell_bal['size']) / (buy_bal['size'] + sell_bal['size']))
    disb_mean = disbalance.rolling(n_packets).mean()
    index = sell_bal[time_column].values
    return pd.Series(data=disb_mean.values, index=index).sort_index()


def plotly_trades(trades_sample, best_bid, best_ask, symbol, stock, time_column='exch_time'):

        trace0 = go.Scatter(
            x = trades_sample[trades_sample['side'] == 2][time_column],
            y = trades_sample[trades_sample['side'] == 2]['price'],
            name = 'Market BUY',
            mode = 'markers',
            marker = dict(
                size = 10,
                color = '#3F4B7F',
                symbol = 'triangle-up',
                line = dict(
                    width = 0,
                    color = '#3F4B7F'
                )
            )
        )

        trace1 = go.Scatter(
            x = trades_sample[trades_sample['side'] == 1][time_column],
            y = trades_sample[trades_sample['side'] == 1]['price'],
            name = 'Market SELL',
            mode = 'markers',
            marker = dict(
                size = 10,
                color = 'rgb(76, 182, 170)',
                symbol = 'triangle-down',
                line = dict(
                    width = 0,
                )
            )
        )

        trace2 = go.Scatter(
            x = best_bid.index,
            y = best_bid,
            name = 'Best BID',
            mode = 'lines',
            marker = dict(
                size = 5,
                color = '#737CB4',
                symbol = 'line-ew',
                opacity= 1,
                line = dict(
                    width = 1,
                    color = '#737CB4'
                )
            ),
            line = dict(
                    shape = 'hv',
                )
        )

        trace3 = go.Scatter(
            x = best_ask.index,
            y = best_ask,
            name = 'Best ASK',
            mode = 'lines',
            marker = dict(
                size = 5,
                color = '#6AD2C5',
                symbol = 'line-ew',
                opacity= 1,
                line = dict(
                    width = 1,
                    color = '#6AD2C5'
                )
            ),
            line = dict(
                    shape = 'hv',
                )

        )

        data = [trace2, trace3, trace0, trace1]

        layout = dict(title = f'{symbol.upper()} on {stock.upper()}',
                      yaxis = dict(zeroline = False),
                      xaxis = dict(zeroline = False)
                     )

        fig = dict(data=data, layout=layout)
        plotly.offline.iplot(fig, filename='trades')
        
        
def get_stats(stock, books, trades, price_btc, time_column):

    symbols = trades['symbol'].unique()
    # Only for Exchanges with snapshots instead of order updates
    if stock.lower() == 'huobi':
        latency2 = 2
    elif stock.lower() == 'hitbtc':
        latency2 = 0.2
    else:
        latency2 = 0

    stats = dict()

    for symbol in symbols:

        if '/' in symbol:
            base, quot = symbol.split('/')
        elif '_' in symbol:
            base, quot = symbol.split('_')
        else:
            if symbol[-3:].lower() in ['btc', 'usd', 'eur', 'eth', 'gbp', 'rub']:
                base, quot = symbol[:-3], symbol[-3:]
            elif symbol[-4:].lower() in ['usdt', 'eurt', 'usdc']:
                base, quot = symbol[:-4], symbol[-4:]
            else:
                logging.warning('Unable to define quot, base currencies!')
                base, quot = 'NA', 'NA'

        stats[symbol] = dict() 
    #     display(trades[[time_column, 'price', 'size', 'side']])
        trades_symbol = trades[[time_column, 'price', 'size', 'side']][trades['symbol'] == 
                                                                       symbol].set_index(time_column, drop=True)
        books_symbol = books[[time_column, 'asks', 'bids']][books['symbol'] == 
                                                            symbol].set_index(time_column, drop=True)
        books_symbol2 = books_symbol.copy()
        books_symbol2.index = [x - dt.timedelta(seconds=latency2) for x in books_symbol2.index]
        books_symbol2.columns = [x + '_2' for x in books_symbol2.columns]
        df = pd.concat([trades_symbol, books_symbol, 
                        books_symbol2, 
                       ]).sort_index()
        df = df.ffill().loc[trades_symbol.index].dropna().drop_duplicates(subset=[x for x in df.columns 
                                                                                if x not in ['bids', 'asks',
                                                                                             'bids_2', 'asks_2']])
        df['bid0'] = df['bids'].apply(get_best)
        df['ask0'] = df['asks'].apply(get_best)             
        df['bid0_2'] = df['bids_2'].apply(get_best)
        df['ask0_2'] = df['asks_2'].apply(get_best)
        df['spread'] = df['ask0'] - df['bid0']
        print(f'{STOCK.upper()}. {symbol.upper()}. Ask <= bid: {(df["spread"] <= 0).sum()}')
        df['midprice'] = (df['ask0'] + df['bid0']) / 2
        df['spread_share'] = df['spread'] / df['midprice']

        df['in-spread'] = ((df["price"] < df["ask0"]) & (df["price"] > df["bid0"])
                           & (df["price"] < df["ask0_2"]) & (df["price"] > df["bid0_2"])
                          )
        df['out-of-spread'] = ((df["price"] > df[["ask0", "ask0_2"]].max(axis=1)) | 
                               (df["price"] < df[["bid0", "bid0_2"]].min(axis=1)) )
              
        f = lambda x: handy_liquidity(x['bids'], x['asks'], 0.5)
        df['handy_liquidity'] = df[['bids', 'asks']].apply(f, axis=1) * price_btc[base.upper()]

        print(f'Handy liquidity for {symbol.upper()}: {df["handy_liquidity"].mean():.2f} BTC')

        plt.subplots(figsize=(12,6 ))
        sns.distplot(df['handy_liquidity'])
        plt.title(f'{STOCK.upper()}. {symbol.upper()}. Handy Liquidity Distribution')
        plt.xlabel('BTC')
        plt.show()

        f = lambda x: spread_by_volume(x['bids'], x['asks'], 10 / price_btc[base.upper()])
        df['spread_by_volume'] = df[['bids', 'asks']].apply(f, axis=1)

        n_in = df['in-spread'].sum()
        vol_in_perc = df['size'][df['in-spread']].sum() / df['size'].sum()
        n_out = df['out-of-spread'].sum()
        vol_out_perc = df['size'][df['out-of-spread']].sum() / df['size'].sum()

    #     stats[symbol]['Total Trades'] = df.shape[0]
    #     stats[symbol]['In-spread Trades, N'] = n_in
        stats[symbol]['Fake In-spread Trades, %'] = n_in / df.shape[0] * 100
        stats[symbol]['Fake In-spread Volume, %'] = vol_in_perc * 100
    #     stats[symbol]['Out-of-spread Trades, N'] = n_out
        stats[symbol]['Fake Out-of-spread Trades, %'] = n_out / df.shape[0] * 100
        stats[symbol]['Fake Out-of-spread Volume, %'] = vol_out_perc * 100
        stats[symbol]['Fake Total Volume, %'] = (vol_in_perc + vol_out_perc) * 100
        stats[symbol]['Fake Total Trades, %'] = (n_in + n_out) / df.shape[0] * 100

        stats[symbol]['Spread Bid-Ask, min, %'] = df['spread_share'].min() * 100
        stats[symbol]['Spread Bid-Ask, max, %'] = df['spread_share'].max() * 100
        stats[symbol]['Spread Bid-Ask, mean, %'] = df['spread_share'].mean() * 100

        stats[symbol][f'Handy Liquidity Mean, BTC'] = df['handy_liquidity'].mean()
        stats[symbol][f'Handy Liquidity Min, BTC'] = df['handy_liquidity'].min()
        stats[symbol][f'Handy Liquidity Max, BTC'] = df['handy_liquidity'].max()

        stats[symbol][f'Spread 10 BTC, mean, %'] = df['spread_by_volume'].mean() * 100
        stats[symbol][f'Spread 10 BTC, min, %'] = df['spread_by_volume'].min() * 100
        stats[symbol][f'Spread 10 BTC, max, %'] = df['spread_by_volume'].max() * 100

        # Life time of the best orders
        books_symbol['bid0'] = books_symbol['bids'].apply(get_best)
        books_symbol['ask0'] = books_symbol['asks'].apply(get_best)
        bid0 = books_symbol['bid0']
        bid0_int = pd.Series(bid0[bid0 != bid0.shift()].index).diff().dropna()
        bid0_int = ((bid0_int.dt.seconds * 1e6 + bid0_int.dt.microseconds) / 1e6)
        plt.subplots(figsize=(12,6))
        sns.distplot(bid0_int[bid0_int < 10], norm_hist=False)
        plt.title(f'{STOCK.upper()}. {symbol.upper()}. Best bid lifetime distribution')
        plt.xlabel('Lifetime, s')
    #     plt.ylabel('Relative frequency')
        plt.show();
        ask0 = books_symbol['ask0']
        ask0_int = pd.Series(ask0[ask0 != ask0.shift()].index).diff().dropna()
        ask0_int = ((ask0_int.dt.seconds * 1e6 + ask0_int.dt.microseconds) / 1e6)
        plt.subplots(figsize=(12,6))
        sns.distplot(ask0_int[ask0_int < 10], norm_hist=False)
        plt.title(f'{STOCK.upper()}. {symbol.upper()}. Best ask lifetime distribution')
        plt.xlabel('Lifetime, s')
    #     plt.ylabel('Relative frequency')
        plt.show();

        stats[symbol]['Lifetime of the best bid, mean, s'] = bid0_int.mean()
        stats[symbol]['Lifetime of the best bid, median, s'] = bid0_int.median()
        stats[symbol]['Lifetime of the best bid, min, s'] = bid0_int.min()
        stats[symbol]['Lifetime of the best bid, max, s'] = bid0_int.max()

        stats[symbol]['Lifetime of the best ask, mean, s'] =ask0_int.mean()
        stats[symbol]['Lifetime of the best ask, median, s'] = ask0_int.median()
        stats[symbol]['Lifetime of the best ask, min, s'] = ask0_int.min()
        stats[symbol]['Lifetime of the best ask, max, s'] = ask0_int.max()
    
    return pd.DataFrame(stats)
