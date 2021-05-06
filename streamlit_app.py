from collections import namedtuple
import altair as alt
import math
import streamlit as st

import os
import re
import pandas as pd
import numpy as np
import ccxt
import datetime
import datatable as dt
from importlib import reload
import cost_basis
import crypto_functions as crypto_fn
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# secrets: https://share.streamlit.io/

url = st.secrets["txAll_url"]
path = url.replace('view.aspx', 'download.aspx')
##
exchange_names = ['kucoin', 'kraken']

def read_data(path):

    df = pd.read_csv(path) #, nrows=2

    ##
#     dfTX = pd.read_csv('../trade_history/ALL_TRANSACTIONS.csv')
    TTP = cost_basis.TransactionsToPortfolio(df, exchange_names)
    
    return(TTP)

def update_pricing(TTP):
    return(cost_basis.PriceLookup(TTP['CurrentHoldings'], exchange_names))

def finish_steps(TTP):
    ##
    dfttp = TTP['CostBasis'].copy()
    dfttp = dfttp.rename({'coin':'Coin', 'quantity':'Quantity', 'unit_cost':'Effective Price', 'total_cost_basis_usd':'USD Amount', 'trade_date':'Date'}, axis=1)
    dfttpsum = dfttp[['order_id', 'Date', 'Coin', 'Quantity', 'USD Amount']].groupby(['order_id', 'Date', 'Coin']).sum()
    dfttpsum['Effective Price'] = abs(dfttpsum['USD Amount'] / dfttpsum['Quantity'])
    dfttpsum = dfttpsum[dfttpsum['Quantity']>0].merge(TTP['Portfolio'][['Current Price']], how='outer', left_index=True, right_index=True)
    dfttpsum.sort_index(axis=0, level='Date')
    dfttpsum['Difference'] = dfttpsum['Current Price'] / dfttpsum['Effective Price'] - 1
    format_dict = {'USD Amount':'${0:,.2f}', 
                   'Effective Price':'${0:,.2f}', 
                   'Current Price':'${0:,.2f}', 
                   'Difference': '{:.1%}',}
    pretty_crypto_trans = dfttpsum.copy().sort_values('Date')
    pretty_crypto_trans = pretty_crypto_trans.reset_index(level=['Date','Coin'])
    pretty_crypto_trans = pretty_crypto_trans.style.format(format_dict).hide_index()
    # pretty_crypto_trans

    ##
    dfplot = dfttpsum.copy().sort_values('Date')
    dfplot = dfplot.reset_index(level=['Date','Coin'])
    excl = list(dfplot[dfplot['Coin'] == 'PUNDIX'].index)
    if not 'USDT' in excl:
        excl.append('USDT')
    excl_df = dfplot.index.isin(excl)
#     pltdata = pd.DataFrame({'Investments':dfplot[~excl_df].index, 'Gain/(Loss)':dfplot[~excl_df]['Difference']})

    ##
    mask = [len(i)!=19 for i in TTP['CostBasis'].order_id]
    dfSansKraken = TTP['CostBasis'][mask]
    purchase_sum = dfSansKraken[dfSansKraken['coin']=='USD']['total_cost_basis_usd'].sum()
    fiat_out = 0

    cs = crypto_fn.crypto_snapshot(TTP['Portfolio'], purchase_sum = purchase_sum, fiat_out = fiat_out)

    ##
    PPD = crypto_fn.PortfolioPerformanceDetail(cs['df_snap_summ'], TTP['CostBasis'], sort_by='Current Value', ascending=False)
    
    final_list = {'cs':cs, 'PPD':PPD, 'pretty_crypto_trans':pretty_crypto_trans, 'pltdata':dfplot[~excl_df]}
    return(final_list)

TTP = read_data(path)

def update_pricing_button(TTP):
#     if x:
        text_price_update = 'prices have been updated'
        TTP['Portfolio'] = update_pricing(TTP)
        f = finish_steps(TTP)
#     else:
#         text_price_update = 'prices have been updated'
#         dfPortfolio = TTP['CurrentHoldings'].copy()
#         dfPortfolio['price'] = 1
#         dfPortfolio.index.names=['Coin']
#         dfPortfolio = dfPortfolio.rename({'quantity':'Quantity', 'price':'Current Price'}, axis=1)
#         TTP['Portfolio'] = dfPortfolio.copy()
#         f = finish_steps(TTP)
        return(f)

# f = update_pricing_button(TTP)

# st.set_option('deprecation.showPyplotGlobalUse', False)
# st.dataframe(f['cs']['pretty_summ'])
# st.dataframe(f['PPD']['pretty_detail'])

# plt.figure(figsize=[12, 4.8])
# plt.bar(f['pltdata'].index, f['pltdata']['Difference'])
# plt.xticks([])
# plt.ylabel('Gain/(Loss)')
# plt.xlabel('Investments')
# plt.gca().axes.get_yaxis().set_major_formatter(PercentFormatter(xmax=1))
# st.pyplot(plt)
# st.pyplot(crypto_fn.PP_port_donut(f['cs']['df_snap_summ']))

# genre = st.sidebar.radio("Reports",
#                          ('Riches', 'Details', 'Donut', 'Bar'))
    
if (st.button('Update Pricing') | True):
    f = update_pricing_button(TTP)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    
#     if (genre=='Riches'):
    st.dataframe(f['cs']['pretty_summ'])

#     elif (genre=='Details'):
    st.dataframe(f['PPD']['pretty_detail'])

#     elif (genre=='Donut'):
    plt.figure(figsize=[12, 4.8])
    plt.bar(f['pltdata'].index, f['pltdata']['Difference'])
    plt.xticks([])
    plt.ylabel('Gain/(Loss)')
    plt.xlabel('Investments')
    plt.gca().axes.get_yaxis().set_major_formatter(PercentFormatter(xmax=1))
    st.pyplot(plt)

#     elif (genre=='Bar'):
    st.pyplot(crypto_fn.PP_port_donut(f['cs']['df_snap_summ']))

    
    
    
# st.pyplot(crypto_fn.PP_dollar_by_coin(f['cs']['df_snap_summ']))
##
# cs['pretty_summ']
# PPD['pretty_detail']

# pretty_crypto_trans



