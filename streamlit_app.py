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
# streamlit run .\streamlit_app.py

url = st.secrets["txAll_url"] ## must choose embed from onedrive to get URL
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
    dfttp = dfttp.rename({'coin':'Coin', 
                            'quantity':'Quantity', 
                            'unit_cost':'Effective Price', 
                            'total_cost_basis_usd':'USD Amount', 
                            'trade_date':'Date'}, 
                            axis=1)
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

    ##
    dfplot = dfttpsum.copy().sort_values('Date')
    dfplot = dfplot.reset_index(level=['Date','Coin'])
    dfp = dfplot.copy()
    ##
    purchase_sum = crypto_fn.PurchaseSum(TTP['CostBasis'])
    fiat_out = 0

    cs = crypto_fn.crypto_snapshot(TTP['Portfolio'], purchase_sum = purchase_sum, fiat_out = fiat_out)

    ##
    PPD = crypto_fn.PortfolioPerformanceDetail(cs['df_snap_summ'], TTP['CostBasis'], sort_by='Current Value', ascending=False)
    
    final_list = {'cs':cs, 'PPD':PPD, 'pretty_crypto_trans':pretty_crypto_trans, 'pltdata':dfp}
    return(final_list)

TTP = read_data(path)

def update_pricing_button(TTP):
    text_price_update = 'prices have been updated'
    TTP['Portfolio'] = update_pricing(TTP)
    f = finish_steps(TTP)
    return(f)

if (st.button('Update Pricing') | True):
    f = update_pricing_button(TTP)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    st.text('Gain/(Loss)')
    st.text('${0:,.0f}'.format(f['cs']['pretty_summ'].data.iloc[0,3]))
    st.text('{:.2%}'.format(f['cs']['pretty_summ'].data.iloc[0,4]))
    st.dataframe(f['cs']['pretty_summ'])

    st.dataframe(f['PPD']['pretty_detail'])

    plt.figure(figsize=[12, 4.8])
    plt.bar(f['pltdata'].index, f['pltdata']['Difference'])
    plt.xticks([])
    plt.ylabel('Gain/(Loss)')
    plt.xlabel('Investments')
    plt.gca().axes.get_yaxis().set_major_formatter(PercentFormatter(xmax=1))
    st.pyplot(plt)

    st.pyplot(crypto_fn.PP_port_donut(f['cs']['df_snap_summ']))




