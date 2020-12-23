import pandas

df = pandas.read_csv('../data/tickers.csv')

def gen_symbol_dict():
    """Using Nasdaq's csv 
    returns dict of form {Ticker: {name: '', management: [], sector: '', Industry: ''}"""
    symbols = list(df['Symbol'])
    tickers = {}
    for t_i, t in enumerate(symbols):
        tickers[t] = {}
        tickers[t]['name'], tickers[t]['sector'], tickers[t]['industry'] = df.iloc[t_i]['Name'], df.iloc[t_i]['Sector'], df.iloc[t_i]['Industry']
    return tickers