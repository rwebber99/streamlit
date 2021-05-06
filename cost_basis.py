# Transactions

## setup
import os
import pandas as pd
import numpy as np
import re
from binance.client import Client
import ccxt
from twisted.internet import reactor
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import datetime
import time
import datatable as dt
import pytz



ColDict = {
    'kucoin': {'pair': 'symbol',
              'trade_time': 'tradeCreatedAt',
              'quantity_purchased': 'size',
              'quantity_fee': 'fee',
              'total_cost_quote': 'funds',
              'unit_cost_quote': 'price', # cost per unit of base coin in quote coin units
              'type': 'side',
              'order_id': 'orderId'},

    'kraken': {'pair': 'pair',
              'trade_time': 'time',
              'quantity_purchased': 'vol',
              'quantity_fee': 'fee',
              'total_cost_quote': 'cost',
              'unit_cost_quote': 'price', # cost per unit of base coin in quote coin units
              'type': 'type',
              'order_id': 'ordertxid'}
}

tzDict = {
    #'America/Los_Angeles'
    #'Asia/Singapore'
    'kucoin': {'tz_orig': 'Asia/Singapore'},
    'kraken': {'tz_orig': 'UTC'}
}

## access base usd symbol
def usd_symbol(api):
    """
    api -- 'binance' or 'ccxt_' + exchange_name
    """
    
    usd_dict = {'binance': 'USDT',
               'ccxt_kucoin': 'USDT',
               'ccxt_kraken': 'USD'}
    
    return usd_dict[api]



def symbol_lookup(df, base, quote):
    """Get exchange-specific symbol based on base and quote coins 
    df -- exchange.load_markets()
    """
    coin = df[(df['base']==base) & (df['quote']==quote)]['symbol']
    if (coin.shape[0] > 0):
        try:
            coin = coin.unique()[0]
        except:
            pass
    else:
        coin = None
    return coin




## back and forth between common date formats
def TimeConvertCustom(t, floor_min = True, tz_orig = 'UTC', tz_new = 'UTC'):
    """Convert long integer time (POSIX/UNIX) to human time, or vice versa
    t -- 
    floor_min -- 
    tz_orig -- 
    tz_new -- 
    """
    
    ## if t can be represented as integer, do so
    try: 
        t = int(t)
    except:
        pass
    
    ## convert 13 digit UNIX date representation to human form. potentially round down to the minute
    if (type(t)==int):
        if (len(str(t)) == 13):
            t = pd.to_datetime(float(t), unit='ms', origin='unix')
            if (floor_min):
                t = t.floor('min')
            t = t.strftime("%Y/%m/%d %H:%M:%S")
    
    else:
        
        ## if t is Timestamp obj, potentially round down to the minute
        if (type(t) == pd._libs.tslibs.timestamps.Timestamp):
            if (floor_min):
                t = t.floor('min')
            t = pd.to_datetime(t)
            t = t.strftime("%Y/%m/%d %H:%M:%S")    
        
        ## potentially round down to the minute manually
        else:
            tz = pytz.timezone(tz_orig) # 'Asia/Singapore'
            t = pd.Series(1,
                    index=pd.DatetimeIndex([t])).tz_localize(tz).tz_convert(tz_new).index
            t = t.strftime("%Y/%m/%d %H:%M:%S")[0]
            t = pd.to_datetime(t)
            t = t.strftime("%Y/%m/%d %H:%M:%S")   
            if (floor_min):
                t = t[:-2]+'00'
        
        t = datetime.datetime.strptime(t, "%Y/%m/%d %H:%M:%S")
                
        t = str(int(time.mktime(t.timetuple())))
        z = '{:<013}'
        t = z.format(t)
        t = float(t)
        
    return t


def tx_hist_prep_other(df):
    """
    """
    dfOther = df.copy()
    dfOther.drop(columns='cost_raw', inplace=True)
    dfOther['trade_time'] = pd.to_datetime(dfOther['trade_time'], format="%Y/%m/%d %H:%M:%S")
    dfOther['order_id'] = dfOther['base'] + dfOther['quote'] + dfOther['trade_time'].dt.strftime("%Y%m%d") + dfOther['type']
    return dfOther



def tx_hist_prep(df, col_dict, floor_min = True):
    """Convert exchange's transactions history to standard format
    df -- pd df of transaction history
    col_dict -- dictionary mapping tx hist to following cols: 
                pair, trade_time, quantity_purchased, quantity_fee, total_cost_quote, type
                (total_cost_quote is in the units of the quote currency; separate from fee col)
    """
    
    discontinued_coins = {'NPXS': 'PUNDIX'}
    dc_conversion = {'NPXS': 1/1000}
    
    filtered = {k: v for k, v in col_dict.items() if v is not None}
    col_dict.clear()
    col_dict.update(filtered)
    dict_flipped = {value:key for key, value in col_dict.items()}
    dfNew = df.copy()
    dfNew = dfNew[list(dict_flipped.keys())]
    dfNew = dfNew.rename(columns=dict_flipped)
    dfNew = dfNew[list(col_dict.keys())]
    dfNew['trade_time'] = pd.to_datetime(dfNew['trade_time'])
    if (floor_min):
        dfNew['trade_time'] = dfNew['trade_time'].dt.floor('min')
    
    for dc in discontinued_coins.keys():
        dfnFilter = dfNew['pair'].str.contains(dc)
        dfNew.loc[dfnFilter, 'quantity_purchased'] = dfNew.loc[dfnFilter, 'quantity_purchased'] * dc_conversion[dc]
        dfNew['pair'] = dfNew['pair'].str.replace(dc, discontinued_coins[dc])
    
    return dfNew


## binance version of cost_basis_calc
def cbc_binance(pair, trade_time, quantity_purchased, quantity_fee, client):
    """Calculates the cost basis in USD for trade history of any pair given transaction datetime. 
    Looks up the open price at the minute of the transaction (sets seconds to '00').
    datetime assumed to be in 'America/Los_Angeles' timezone.
    returns total cost basis
    
    pair -- ex 'LTC/USDT'
    trade_time -- ex '2021/03/20 11:41:00'
    quantity_purchased -- number of coins of the first symbol in the pair
    quantity_fee -- fee in units of the second symbol in the pair
    client -- login info
    """
    
#     pair = pair.str.split('-') ## for series
    pair = re.split('/|-', pair)
    coin_buy = pair[0]+'USDT'
    coin_base = pair[1]+'USDT'
    
    trade_time = pd.to_datetime(trade_time).floor('min')
    trade_time = pd.Series(1,
        index=pd.DatetimeIndex([trade_time])).tz_localize('America/Los_Angeles').tz_convert('UTC').index
    trade_time = trade_time.strftime("%Y/%m/%d %H:%M:%S")
    
    ## cost of shares purchased
    try:
        klines = client.get_historical_klines(coin_buy, 
                                              Client.KLINE_INTERVAL_1MINUTE, 
                                              trade_time[0], trade_time[0])
        basis_per_unit = float(klines[0][1])
    except:
        if ('USD' in coin_buy):
            basis_per_unit = 1
        else:
            basis_per_unit = 0
    
    ## cost of fees
    try:
        klines = client.get_historical_klines(coin_base, 
                                              Client.KLINE_INTERVAL_1MINUTE, 
                                              trade_time[0], trade_time[0])
        fee_per_unit = float(klines[0][1])
    except:
        if ('USD' in coin_buy):
            fee_per_unit = 1
        else:
            fee_per_unit = 0 #### if can't calc fees, set to zero to be conservative
    
    total_cost_basis = basis_per_unit * quantity_purchased + fee_per_unit * quantity_fee
    
    return total_cost_basis


def quote_hist_kraken_csv(filedir, trade_time, quote, usd_symbol = usd_symbol('ccxt_kraken'), OHLC = 'C'):
    """Get historical quote from klines
    filepath -- kraken csv
    base -- left coin in symbol
    quote -- right coin in symbol
    trade_time -- in UNIX time format
    OHLC -- Open, High, Low, Close
    """
    
    ohlc_pos = 'OHLC'.find(OHLC)
    quoteK = 'XBT' if quote == 'BTC' else quote
    lkup_file = quoteK + usd_symbol + '_1.csv'

    try:
        dtBak = dt.fread(filedir + lkup_file)
        unit_cost = dtBak[dt.f.C0==trade_time/1000, 'C' + str(ohlc_pos)][0,0]
        
    except:
        raise NameError('Error in Kraken CSV lookup. Check filedir & whether trade_time has a match')

    final_list = {'usd_symbol': usd_symbol,
                 'quote': quoteK,
                 'unit_cost': unit_cost}
    return final_list


def quote_hist(exchange, trade_time, df, base, quote, OHLC = 'C'):
    """Get historical quote from klines
    df -- from exchange.load_markets()
    base -- left coin in symbol
    quote -- right coin in symbol
    trade_time -- in UNIX time format
    exchange
    OHLC -- Open, High, Low, Close
    """
    
    ohlc_pos = 'OHLC'.find(OHLC)+1
    dfE = df.copy()
    coin = dfE[(dfE['base']==base) & (dfE['quote']==quote)]['symbol']
    try:
        coin = coin.unique()[0]
    except:
        pass

    try:
        klines = exchange.fetchOHLCV(symbol = coin, 
                                     timeframe = '1m', 
                                     since = trade_time, 
                                     limit = 1)
        unit_cost = float(klines[0][ohlc_pos])
        has_usd_pair = True
        trade_time_check = float(klines[0][0])
        
    except:
        if ('USD' in base):
            unit_cost = 1
            has_usd_pair = False
            trade_time_check = trade_time.copy()
        else:
            raise NameError('Error for ' + base + '/' + quote)

    if (not trade_time_check == trade_time):
        raise NameError('Kline date mismatch (CCXT). Requested: ' + str(trade_time) + 
                        ', Returned: ' + str(trade_time_check))

    final_list = {'coin': coin,
                 'base': base,
                 'quote': quote,
                 'unit_cost': unit_cost,
                 'has_usd_pair': has_usd_pair}
    return final_list
                

    
    
    
    
    
def quote_hist_binance(trade_time, quote, usd_symbol = usd_symbol('binance'), client=None, OHLC = 'C'):
    """Get historical quote from klines
    df -- from exchange.load_markets()
    base -- left coin in symbol
    quote -- right coin in symbol
    trade_time -- in UNIX time format
    client -- Binance API
    OHLC -- Open, High, Low, Close
    """
    
    ohlc_pos = 'OHLC'.find(OHLC)+1
    coin = quote + usd_symbol
    
    api_key = os.environ.get('binance_api')
    api_secret = os.environ.get('binance_secret')
    client = Client(api_key, api_secret)

    klines = client.get_historical_klines(coin, 
                                          Client.KLINE_INTERVAL_1MINUTE, 
                                          TimeConvertCustom(trade_time), TimeConvertCustom(trade_time))
    unit_cost = float(klines[0][ohlc_pos])
    has_usd_pair = True
    trade_time_check = float(klines[0][0])
        
    if (not trade_time_check == trade_time):
        raise NameError('Kline date mismatch (Binance). Requested: ' + str(trade_time) + 
                        ', Returned: ' + str(trade_time_check))

    final_list = {'quote': quote,
                 'usd_symbol': usd_symbol,
                 'unit_cost': unit_cost,
                 'has_usd_pair': has_usd_pair}
    return final_list        
        
        
        
        
        
## ccxt version of cost_basis_calc
def cbc_ccxt(pair, trade_time, quantity_purchased, quantity_fee, usd_symbol, exchange, 
             total_cost_quote=None, unit_cost_quote=None):
    """Calculates the cost basis in USD for trade history of any pair given transaction datetime. 
    Looks up the open price at the minute of the transaction (sets seconds to '00').
    datetime assumed to be in 'America/Los_Angeles' timezone.
    returns total cost basis
    
    pair -- ex 'LTC/USDT'
    trade_time -- ex '2021/03/20 11:41:53'
    quantity_purchased -- number of coins of the first symbol in the pair
    quantity_fee -- fee in units of the second symbol in the pair
    usd_symbol -- ex: 'USDT'
    exchange -- CCXT set up for specific exchange
    total_cost_quote -- cost given in tx history in quote coin units
    unit_cost_quote -- cost per unit of base coin in quote coin units
    """
    
    ## separate coins
    elm = exchange.load_markets()
    elm  ## load markets in this env
    
    try:
        mkt = exchange.markets_by_id[pair] 
        symbol = mkt['symbol']
        base = mkt['base']
        quote = mkt['quote']
    except:
        symbol = pair
        pair = re.split('/|-', pair)
        base = pair[0]
        quote = pair[1]
    dfELM = pd.DataFrame(elm).T
    
    ## round down (back) to the minute for matching, format dates
    trade_time = TimeConvertCustom(trade_time, floor_min = True)
    
    ## cost of quote coin
    if (quote == usd_symbol):
        if (not total_cost_quote==None):
            unit_cost = 1
            total_cost_basis = total_cost_quote + quantity_fee
            unit_coin = None
            quote_unit_count = total_cost_basis / unit_cost
        else:
            qh = quote_hist(exchange, trade_time, df=dfELM, base=base, quote=usd_symbol)
            unit_cost = qh['unit_cost']
            total_cost_basis = unit_cost * (quantity_purchased + quantity_fee)
            unit_coin = usd_symbol
            quote_unit_count = total_cost_basis / unit_cost
    else:
#         try:
            try:
                ## quote in terms of usd
                qh = quote_hist(exchange, trade_time, df=dfELM, base=quote, quote=usd_symbol)
                unit_cost_q = qh['unit_cost']
                
            except:
                try:
                    qh = quote_hist_kraken_csv('../kraken_ohlcvt/', trade_time, quote)
                    unit_cost_q = qh['unit_cost']
                except:
                    qh = quote_hist_binance(trade_time, quote)
                    unit_cost_q = qh['unit_cost']

            if (not unit_cost_quote==None):
                unit_cost_b = unit_cost_quote * unit_cost_q
                
            else:
                try:
                    ## base in terms of quote
                    qh = quote_hist(exchange, trade_time, df=dfELM, base=base, quote=quote)
                    unit_cost_b = qh['unit_cost']
                    
                except:
                    unit_cost_b = 0


            unit_cost = unit_cost_b
            total_cost_basis = unit_cost * (quantity_purchased + quantity_fee)
            unit_coin = quote
            quote_unit_count = total_cost_basis / unit_cost_q

    final_list = {'total_cost_basis': total_cost_basis,
                 'base': base,
                 'quote': quote,
                 'unit_cost': unit_cost,
                 'unit_coin': unit_coin,
                 'quote_unit_count': quote_unit_count}
    
    return final_list






def Transactions(df, col_dict, api, exchange):
    """Prep transaction history, add cost basis column
    df -- transaction history df
    col_dict -- mapping of df's column names to expected column names
    api -- 'binance' or 'ccxt_' + exchange_name
    exchange -- 
    """
    
    import cost_basis as cbf
    if ('ccxt_' in api):
        cbc_api = 'ccxt'
    else:
        cbc_api = api
    cbc = getattr(cbf, 'cbc_' + cbc_api)
    dfNew = cbf.tx_hist_prep(df, col_dict)

    tcbu = pd.Series(dtype=np.float64)
    tcbu = []
    
    if len(dfNew.index)>0:
        for i in dfNew.index:
            tcq = dfNew['total_cost_quote'][i] if col_dict['total_cost_quote'] is not None else None
            ucq = dfNew['unit_cost_quote'][i] if col_dict['unit_cost_quote'] is not None else None
            cbc_list = cbc(pair=dfNew['pair'][i], 
                          trade_time=dfNew['trade_time'][i], 
                          quantity_purchased=dfNew['quantity_purchased'][i], 
                          quantity_fee=dfNew['quantity_fee'][i],
                          total_cost_quote = tcq, 
                          unit_cost_quote = ucq,
                          usd_symbol = cbf.usd_symbol(api),
                          exchange = exchange)
            tcbu.append(cbc_list)
        dfTCBU = pd.DataFrame(tcbu, columns=cbc_list.keys())
        dfNew = dfNew.merge(dfTCBU, left_index=True, right_index=True)
        dfNew = dfNew.rename(columns={'total_cost_basis':'total_cost_basis_usd'})
    else:
        pass
    
    final_list = {'df': dfNew,
                 'tcbu': tcbu}
    return final_list


def CombineTransactions(*args):
    """
    args -- df(s) from Transactions
    """
    df = args[0].copy()
    if (len(args)>1):
        for ar in args[1:]:
            df = df.append(ar).reset_index(drop=True)
    df.sort_values(by=['trade_time'], ignore_index=True, inplace=True)
    return df




def CostBasis(df):
    """
    df -- df from cost_basis.CombineTransactions
    """
    
    dfTX = df.copy()
    colsNew = ['order_id', 'trade_time', 'coin', 'quantity', 'total_cost_basis_usd', 'unit_cost']
    
    ## base coin
    dfBuy_b = dfTX[dfTX['type']=='buy'][['order_id', 'trade_time', 'base', 'quantity_purchased', 'total_cost_basis_usd']]
    dfBuy_b['unit_cost'] = pd.Series(dtype=np.float64)
    dfBuy_b.columns=colsNew
    dfBuy_s = dfTX[dfTX['type']=='sell'][['order_id', 'trade_time', 'quote', 'quote_unit_count', 'total_cost_basis_usd']]
    dfBuy_s['unit_cost'] = pd.Series(dtype=np.float64)
    dfBuy_s.columns=colsNew
    dfBuy = dfBuy_b.append(dfBuy_s).reset_index(drop=True)
    dfBuy['unit_cost'] = dfBuy['total_cost_basis_usd'] / dfBuy['quantity']
    
    ## quote coin
    dfSell_b = dfTX[dfTX['type']=='sell'][['order_id', 'trade_time', 'base', 'quantity_purchased', 'total_cost_basis_usd']]
    dfSell_b['unit_cost'] = pd.Series(dtype=np.float64)
    dfSell_b.columns=colsNew
    dfSell_s = dfTX[dfTX['type']=='buy'][['order_id', 'trade_time', 'quote', 'quote_unit_count', 'total_cost_basis_usd']]
    dfSell_s['unit_cost'] = pd.Series(dtype=np.float64)
    dfSell_s.columns=colsNew
    dfSell = dfSell_b.append(dfSell_s).reset_index(drop=True)
#     dfSell['unit_cost'] = dfSell['total_cost_basis_usd'] / dfSell['quantity']
    dfSell['quantity'] = -dfSell['quantity']
    
    ## combine
    dfCostBasis = dfBuy.append(dfSell)
    dfCostBasis.sort_values(by=['trade_time'], ignore_index=True, inplace=True)
    
    dfCostBasis['trade_date'] = pd.to_datetime(dfCostBasis['trade_time']).dt.date
    dfCostBasis['unit_cost'] = dfCostBasis['total_cost_basis_usd'] / abs(dfCostBasis['quantity'])
    dfCostBasis = dfCostBasis[['order_id', 'trade_date', 'coin', 'quantity', 'total_cost_basis_usd']].groupby(['order_id', 'trade_date', 'coin'], as_index=False).sum(['quantity', 'total_cost_basis_usd'])

    return dfCostBasis




def CurrentHoldings(df):
    """
    df -- df from cost_basis.CostBasis
    """
    dfCostBasis = df.copy()
    return dfCostBasis[['coin', 'quantity']].groupby('coin').sum()




def PriceLookup(df, exchange_names):
    """
    df -- df CurrentHoldings
    exchange_names -- ccxt exchange_id
    bases -- list of coins owned
    """
    
    dfCH = df.copy()
    bases = list(set(dfCH.index)) ## gets unique items
    
    us = []
    for b in [i for i in bases if 'USD' in i]:
        us.append([b, 'USD', 1])
        bases.remove(b)
    dfPrices = pd.DataFrame(us, columns=['coin', 'symbol', 'price'])

    for en in exchange_names:
        exchange_class = getattr(ccxt, en)
        exchange = exchange_class({'timeout': 30000, 'enableRateLimit': True,})

        # separate coins
        elm = exchange.load_markets()
        elm  ## load markets in this env
        dfELM = pd.DataFrame(elm).T

        symbols = []
#         prices = []
        for b in [i for i in bases if i not in list(dfPrices.coin)]:
            sym = symbol_lookup(dfELM, b, usd_symbol('ccxt_' + en))
            if (not sym == None):
                p = pd.DataFrame(exchange.fetch_tickers([sym])).T['last'][0]
                symbols.append([b, sym, p])
            else:
                ## no pair with usd, look up existing pairs and convert price to usd
                try:
                    q = dfELM[dfELM['baseId']==b]['quoteId'][0]
                    sym = symbol_lookup(dfELM, b, q)
                    if (not sym == None):
                        p1 = pd.DataFrame(exchange.fetch_tickers([sym])).T['last'][0]
                        sym2 = symbol_lookup(dfELM, q, usd_symbol('ccxt_' + en))
                        p2 = pd.DataFrame(exchange.fetch_tickers([sym2])).T['last'][0]
                        p = p1 * p2
                        symbols.append([b, sym, p])
                    else:
                        pass # will wait for next exchange
                except:
                    pass # will wait for next exchange
        dfTemp = pd.DataFrame(symbols, columns=['coin', 'symbol', 'price'])
        dfPrices = dfPrices.append(dfTemp, ignore_index=True)
    
#NEEDED #### insert missed coins as 0 ####
    dfPrices.set_index('coin', drop=True, inplace=True)
    dfPortfolio = dfCH.merge(dfPrices, on='coin', how='outer')
    dfPortfolio.drop(columns='symbol', inplace=True)
    dfPortfolio.index.names=['Coin']
    dfPortfolio = dfPortfolio.rename({'quantity':'Quantity', 'price':'Current Price'}, axis=1)
    
    return dfPortfolio





def TransactionsToPortfolio(df, exchange_names):
    """
    """
    dfTX = df.copy()
    dfCB = CostBasis(dfTX)
    dfCH = CurrentHoldings(dfCB)
    
    final_list = {'CostBasis': dfCB,
                 'CurrentHoldings': dfCH}
    return final_list



def removePriorTx(df, dfALL_TRANSACTIONS):
    """Remove previously saved transactions to save time and get accurate results
    df -- saved processed transaction history 
    """
    dfor = df.copy()
    dfTX = dfor[~dfor['order_id'].isin(dfALL_TRANSACTIONS.order_id)]
    return dfTX


def ProcessCCXT(filepath, exchange_id, dfALL_TRANSACTIONS = None):
    """Go from downloaded csv to Transactions df to be combined
    filepath -- downloaded transaction history csv
    exchange_id -- for example, 'kucoin' or 'kraken' (https://github.com/ccxt/ccxt/wiki/Manual)
    """
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'timeout': 30000, 'enableRateLimit': True,})
    
    dfRaw = pd.read_csv(filepath)
    if (dfALL_TRANSACTIONS is not None):
        dfTemp = dfRaw.rename({ColDict[exchange_id]['order_id']: 'order_id'}, axis=1)
        dfRaw = removePriorTx(dfTemp, dfALL_TRANSACTIONS)
        dfRaw = dfRaw.rename({'order_id': ColDict[exchange_id]['order_id']}, axis=1)
    dftx = Transactions(dfRaw, ColDict[exchange_id], 'ccxt_'+exchange_id, exchange)
    dfFinal = dftx['df']

    final_list = {'dfTX': dfFinal,
                  'dictTX': dftx,
                  'dfRawHist': dfRaw}
    return final_list


def ProcessOther(filepath, dfALL_TRANSACTIONS = None):
    """Go from manually curated csv to Transactions df to be combined
    filepath -- manually curated transaction history csv
    dfALL_TRANSACTIONS -- df of all past processed transactions
    """
    dfRaw = pd.read_csv(filepath, infer_datetime_format=True)
    if (dfALL_TRANSACTIONS is not None):
        dfRaw = tx_hist_prep_other(dfRaw)
    dfFinal = removePriorTx(dfRaw, dfALL_TRANSACTIONS)

    final_list = {'dfTX': dfFinal,
                  'dfRawHist': dfRaw}
    return final_list