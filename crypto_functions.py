# Transactions

## setup
import os
import pandas as pd
import numpy as np
from binance.client import Client
from binance.websockets import BinanceSocketManager
from twisted.internet import reactor
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter



def crypto_portfolio(t_csv, client):
    """Create a bunch of stuff from csv of transactions.
    
    Arguments:
    t_csv -- filename & location of csv file
    client -- credentials
    """
    
    df_ctr = crypto_trans_read(t_csv)
    dict_cp = crypto_prices(df_ctr, client)
    dict_ctd = crypto_trans_detail(df_ctr, dict_cp)
    
    final_list = {'trans_read': df_ctr,
                 'prices': dict_cp,
                 'trans_detail': dict_ctd}
    return final_list




def crypto_trans_detail(crypto_trans_read, crypto_prices):
    """Combine current market prices with transaction data.
    
    Arguments:
    crypto_trans_read -- output of crypto_trans
    crypto_prices -- output of crypto_prices
    """
    
    df_crypto_trans = crypto_trans_read.merge(crypto_prices['df_mkt_prices'], on='Coin')
    df_crypto_trans = df_crypto_trans.sort_values(by=['Date'])
    df_crypto_trans['Current Price'] = df_crypto_trans['Current Price'].astype(float)

    df_crypto_trans['Difference'] = df_crypto_trans['Current Price'] / df_crypto_trans['Effective Price'] - 1

    format_dict = {'USD Amount':'${0:,.2f}', 
                   'Effective Price':'${0:,.2f}', 
                   'Current Price':'${0:,.2f}', 
                   'Difference': '{:.1%}',
                   'Date': lambda t: t.strftime("%Y-%m-%d")}
    pretty_crypto_trans = df_crypto_trans.style.format(format_dict).hide_index()

    Purchase = sum(df_crypto_trans['Effective Price'] * df_crypto_trans['Quantity'])
    Holdings = sum(df_crypto_trans['Current Price'] * df_crypto_trans['Quantity'])
    NetChange = round(Holdings - Purchase, 2)
    
    final_list = {'df_crypto_trans':df_crypto_trans,
                  'pretty_crypto_trans':pretty_crypto_trans,
                  'dict_crypto_prices':crypto_prices,
                  'net_change':NetChange}
    
    return final_list






def crypto_prices(crypto_trans_read, client):
    """Pull current market prices for all crypto types in transaction data.
    
    Arguments:
    crypto_trans_read -- output of crypto_trans_read
    client -- credentials
    """
    
    mkt_prices={}

    for c in crypto_trans_read['Coin'].unique():
        try:
            p = client.get_symbol_ticker(symbol=c+"USDT")['price']
        except:
            p = float('nan')
        if (c == 'USDT'):
            p = 1
        mkt_prices[c] = p

    df_mkt = pd.DataFrame.from_dict(mkt_prices, orient='index', columns=['Current Price'])
    df_mkt.index.name = 'Coin'
    
    final_list = {'df_mkt_prices': df_mkt,
                  'dict_mkt_prices': mkt_prices}
    return final_list







def crypto_trans_read(t_csv):
    """Create a Pandas DataFrame from csv of transactions.
    
    Arguments:
    t_csv -- filename & location of csv file
    """
    
    df_purch = pd.read_csv(t_csv, thousands=',', parse_dates=['Date'], infer_datetime_format=True)
#     df_purch.replace(to_replace='DOT1', value='DOT', inplace=True)
    df_purch['USD Amount'] = df_purch['USD Amount'].str.replace('\$|,', '')
    df_purch['USD Amount'] = df_purch['USD Amount'].astype(float)
    df_purch['Effective Price'] = df_purch['USD Amount'] / df_purch['Quantity']
#     df_purch['Effective Price'] = pd.to_numeric(df_purch['Effective Price'], errors='coerce')
    df_purch.drop(['Current Price', 'Difference'], axis = 1, inplace=True)
    
    return df_purch








def KCH_printer(kclist):
    for kl in kclist:
        print (kl)

def KuCoin_Helper(usd, buy_symbol, base_symbol, client, output_format = '.5E'):
    """Help translate from Human order to KuCoin order.
    
    Arguments:
    usd -- amount to trade, numeric
    buy_symbol -- coin symbol to buy
    base_symbol -- trade denomination coin symbol
    output_format -- non-USD numeric formatting
    """
    
    cbuy = buy_symbol
    cbase = base_symbol
    usd_equiv = usd
    sciformat = output_format
    buy_digits = 1
    base_digits = 10


    cbuy_usd = float(client.get_symbol_ticker(symbol=cbuy+"USDT")['price'])
    cbase_usd = float(client.get_symbol_ticker(symbol=cbase+"USDT")['price'])

    num_limit = cbuy_usd / cbase_usd
    num_shares = usd_equiv / cbuy_usd
    num_vol = round(num_shares, buy_digits) * round(num_limit, base_digits)
    num_vol_usd = round(num_vol, base_digits) * cbase_usd

    return_list = ['Total USD:        $' + str(f"{num_vol_usd:.2f}"),
                   ' ',
                   cbase + '/' + cbuy + ' Limit:    ' + str(f"{num_limit:{sciformat}}") + ' (round up)',
                   cbuy + ' Shares:       ' + str(f"{num_shares:{sciformat}}"),
                   cbase + ' volume:        ' + str(f"{num_vol:{sciformat}}"),
                   ' ',
                   cbuy + ' market price: $' + str(cbuy_usd),
                   cbase + ' market price:  $' + str(cbase_usd)]
    return KCH_printer(return_list)





def PP_port_donut(df_port):
    
    excl = list(df_port[df_port['Current Value'] < 5].index)
    if not 'USDT' in excl:
        excl.append('USDT')
    excl_df = df_port.index.isin(excl)
    plt.figure(figsize=[8, 6])
    plt.pie(df_port[~excl_df]['Current Value'], labels=df_port[~excl_df].index)
    
    ### add a circle at the center to transform it in a donut chart
    my_circle=plt.Circle( (0,0), 0.55, color='white')
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    port_donut = plt.show()
    return(port_donut)

def PP_dollar_by_coin(df_port):
    
    excl = list(df_port[df_port['Current Value'] < 5].index)
    if not 'USDT' in excl:
        excl.append('USDT')
    excl_df = df_port.index.isin(excl)
    plt.figure(figsize=[12, 4.8])
#    dollar_by_coin = plt.bar(df_port[~excl_df].index, df_port[~excl_df]['Current Value'])
    plt.bar(df_port[~excl_df].index, df_port[~excl_df]['Current Value'])
    return(plt)

def PP_change_by_coin(df_port):
    c = ['C3' if v else 'C0' for v in df_port['Price Movement'] < 0]
    plt.bar(df_port.index, df_port['Price Movement'], color=c)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    change_by_coin = plt#.show()
    return(change_by_coin)


# def Portfolio_Plots(df_port):
    
#     plot_list = {'port_donut':PP_port_donut(df_port),
#                 'dollar_by_coin':PP_dollar_by_coin(df_port),
#                 'change_by_coin':PP_change_by_coin(df_port)}
#     return plot_list







def negative_color_red(s):
    '''
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    '''
    neg = s < 0
    return ['color: red' if v else 'color: black' for v in neg]








def crypto_snapshot(df_or_csv, purchase_sum=None, fiat_out=0, missing_prices=None, client=None):
    '''
    Takes portfolio snapshot, total purchase $, any missing prices and creates summaries.
    
    csv -- csv containing Coin & Quantity cols
    purchase_sum -- numeric, total USD invested
    missing_prices -- dictionary to fill any prices not supported by client {'COIN':0.00}
    client -- API connection
    '''
    
    if (not type(df_or_csv)==pd.DataFrame):
        # read csv, collect market prices
        df_snap = pd.read_csv(df_or_csv, thousands=',')
        df_mkt = crypto_prices(df_snap, client)
        df_mkt = df_mkt['df_mkt_prices']

        # condense to Quantity per Coin
        df_snap_summ = df_snap[['Coin', 'Quantity']].groupby(['Coin']).sum()
        df_snap_summ = df_snap_summ.merge(df_mkt, on='Coin')
        df_snap_summ['Current Price'] = df_snap_summ['Current Price'].astype(float)

        # for any prices manually provided, update table if nan
        for mp in missing_prices:
            if (pd.isnull(df_snap_summ.loc[mp, 'Current Price'])):
                df_snap_summ.loc[mp, 'Current Price'] = missing_prices[mp]    

    else:
        df_snap_summ = df_or_csv.copy()
    
    # calc value, sort
    df_snap_summ['Current Value'] = df_snap_summ['Current Price'] * df_snap_summ['Quantity']
    df_snap_summ = df_snap_summ.sort_values('Current Value', ascending=False)
    df_snap_summ = df_snap_summ[df_snap_summ['Quantity'] > 0]
        
    # high level summary
    df_summ = pd.DataFrame({
        'Purchase': purchase_sum,
        'Fiat Out': fiat_out,
        'Holdings': df_snap_summ['Current Value'].sum()
    },
    index = ['USD Amount'])
    df_summ['$ Gain/(Loss)'] = df_summ['Holdings'] + df_summ['Fiat Out'] - df_summ['Purchase']
    df_summ['% Gain/(Loss)'] = df_summ['$ Gain/(Loss)'] / df_summ['Purchase']
    format_dict = {'Purchase':'${0:,.0f}', 
                   'Fiat Out':'${0:,.0f}',
                   'Holdings':'${0:,.0f}', 
                   '$ Gain/(Loss)':'${0:,.0f}', 
                   '% Gain/(Loss)': '{:.2%}'}
    pretty_summ = df_summ.style.format(format_dict).\
        apply(negative_color_red, subset=['$ Gain/(Loss)', '% Gain/(Loss)'])

    final_list = {'df_snap_summ':df_snap_summ,
                 'pretty_summ':pretty_summ}
    
    return final_list




def PortfolioPerformanceDetail(cs_df_snap_summ, TTP_CostBasis, sort_by='Current Value', ascending=False):
    """Return a detailed look at historical investment performance
    cs_df_snap_summ -- 'df_snap_summ' from crypto_functions.crypto_snapshot
    TTP_CostBasis -- 'CostBasis' df from cost_basis.TransactionsToPortfolio
    sort_by -- Sorts DF by any visible column name. Default is 'Current Value'
    ascending -- Sort ascending? Default False (highest values listed first)
    """
    cdss = cs_df_snap_summ.copy()
    dfttp = TTP_CostBasis.copy()
    dfttp = dfttp.rename({'coin':'Coin', 'total_cost_basis_usd':'Investment'}, axis=1)
    dfInv = dfttp[['Investment', 'Coin']][dfttp['quantity']>0].groupby('Coin').sum()
    dfSale = dfttp[['Investment', 'Coin']][dfttp['quantity']<0].groupby('Coin').sum()
    dfSale.rename(columns={'Investment':'Sales'}, inplace=True)

    csdss = cdss.merge(dfInv, how='outer', left_index=True, right_index=True).merge(
        dfSale, how='outer', left_index=True, right_index=True)
    csdss.fillna(0, inplace=True)
    csdss['Value + Sales'] = csdss['Current Value'] + csdss['Sales']
    csdss = csdss[['Quantity', 'Current Price', 'Investment', 'Value + Sales', 'Sales', 'Current Value']]
    csdss['Gain/(Loss)'] = csdss['Value + Sales'] / csdss['Investment'] - 1
    format_dict = {'Current Value':'${0:,.0f}', 
                   'Investment':'${0:,.0f}', 
                   'Sales':'${0:,.0f}',
                   'Value + Sales':'${0:,.0f}',
                   'Gain/(Loss)': '{:.1%}',}
    csdss = csdss.sort_values(sort_by, ascending=ascending)
    pretty_detail = csdss.style.format(format_dict)
    
    final_list = {'pretty_detail': pretty_detail,
                 'df_detail': csdss}
    return final_list