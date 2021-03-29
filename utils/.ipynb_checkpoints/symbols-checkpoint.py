import pandas as pd
import numpy as np

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