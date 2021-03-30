import pandas as pd
import numpy as np
import random
import yfinance as yf
from IPython.display import clear_output

df = pd.read_csv('../data/tickers.csv')

def gen_symbol_dict():
    """Using Nasdaq's csv 
    returns dict of form {Ticker: {name: '', management: [], sector: '', Industry: ''}"""
    symbols = list(df['Symbol'])
    tickers = {}
    for t_i, t in enumerate(symbols):
        tickers[t] = {}
        tickers[t]['name'], tickers[t]['sector'], tickers[t]['industry'] = df.iloc[t_i]['Name'], df.iloc[t_i]['Sector'],                                                                                    df.iloc[t_i]['Industry']
    return tickers

def extract_tickers(dir_='../data/tickers.csv', sector = None, industry = None):
    """ return NASDAQ tickers; optionally by sector or industry
    probably specific to the nasdaq CSV"""
    tickers = pd.read_csv(dir_)
    tickers = tickers.dropna(axis=0, how='any')
    if sector:
        return list(tickers[tickers['Sector'] == sector]['Symbol'])
    elif industry:
        return list(tickers[tickers['Industry'] == industry]['Symbol'])
    else:
        return tickers
    
def data_by_category(num,kwargs,time='Close',dates=('2018-10-01','2020-9-30')):
    '''num: number of stocks
    kwargs: for extract_tickers e.g. {'sector':'Technology'}'''
    tickers = extract_tickers(**kwargs)
    raster = {}
    i=0
    t_cache = []
    while i < num:
        t = random.randint(1, len(tickers)) - 1
        if t not in t_cache:
            t_cache.append(t)
        
            stock = yf.download(tickers[t], dates[0],dates[1])
            if stock.shape == (0, 6) or len(stock['Close']) != 503: # no data found in range
                continue
            else: 
                raster[tickers[t]] = list(stock[time])
                i+=1

    clear_output()
    return raster