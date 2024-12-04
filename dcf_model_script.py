#!/usr/bin/env python
# coding: utf-8

# # DCF model with web scraping for data

# We will import all the python packages, scrape yahoo finance for financial information on a company, then input that data into a monte carlo DCF model that will compute a normal distribution of the fair value price of thew company.

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection
import sklearn.linear_model
from scipy.stats import t
import yfinance as yf
from lxml import html
import requests
import json
import argparse
from collections import OrderedDict
get_ipython().run_line_magic('matplotlib', 'inline')


# **Note:** Can use Yahoo finance library yfinance for additional company data, but not for financials as the python package is not functioning properly and returns an empty array.

# In[18]:


def get_page(url):
    # Set up the request headers that we're going to use, to simulate
    # a request by the Chrome browser. Simulating a request from a browser
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'max-age=0',
        'Pragma': 'no-cache',
        'Referrer': 'https://google.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36'
    }

    return requests.get(url, headers=headers)

def parse_rows(table_rows):
    parsed_rows = []

    for table_row in table_rows:
        parsed_row = []
        el = table_row.xpath("./div")

        none_count = 0

        for rs in el:
            try:
                (text,) = rs.xpath('.//span/text()[1]')
                parsed_row.append(text)
            except ValueError:
                parsed_row.append(np.NaN)
                none_count += 1

        if (none_count < 4):
            parsed_rows.append(parsed_row)
            
    return pd.DataFrame(parsed_rows)

def clean_data(df):
    df = df.set_index(0) # Set the index to the first column: 'Period Ending'.
    df = df.transpose() # Transpose the DataFrame, so that our header contains the account names
    
    # Rename the "Breakdown" column to "Date"
    cols = list(df.columns)
    cols[0] = 'Date'
    df = df.set_axis(cols, axis='columns', inplace=False)
    
    numeric_columns = list(df.columns)[1::] # Take all columns, except the first (which is the 'Date' column)

    for column_index in range(1, len(df.columns)): # Take all columns, except the first (which is the 'Date' column)
        df.iloc[:,column_index] = df.iloc[:,column_index].str.replace(',', '') # Remove the thousands separator
        df.iloc[:,column_index] = df.iloc[:,column_index].astype(np.float64) # Convert the column to float64
        
    return df

def scrape_table(url):
    # Fetch the page that we're going to parse
    page = get_page(url);

    # Parse the page with LXML, so that we can start doing some XPATH queries
    # to extract the data that we want
    tree = html.fromstring(page.content)

    # Fetch all div elements which have class 'D(tbr)'
    table_rows = tree.xpath("//div[contains(@class, 'D(tbr)')]")
    
    # Ensure that some table rows are found; if none are found, then it's possible
    # that Yahoo Finance has changed their page layout, or have detected
    # that you're scraping the page.
    assert len(table_rows) > 0
    
    df = parse_rows(table_rows)
    df = clean_data(df)
        
    return df


# In[19]:


# choose a ticker symbol
symbol = 'MSFT'


# In[20]:


#scrape balance sheet
bal=scrape_table('https://finance.yahoo.com/quote/' + symbol + '/balance-sheet?p=' + symbol)
print(bal)


# In[21]:


#scrape the financial statements
fin=scrape_table('https://finance.yahoo.com/quote/' + symbol + '/financials?p=' + symbol)
print(fin)


# In[22]:


#scrape cash flow
cash=scrape_table('https://finance.yahoo.com/quote/' + symbol + '/cash-flow?p=' + symbol)
print(cash)


# **Comment out box if would like to export data into excel**

# In[ ]:


# export to excel
# date = datetime.today().strftime('%Y-%m-%d')
# writer = pd.ExcelWriter('Yahoo-Finance-Scrape-' + date + symbol + '.xlsx')
# df_combined.to_excel(writer)
# writer.save()


# In[ ]:


capex=cash.capex[0]
EBIT=fin.EBIT[0]
sales=fin.Total Revenue[0]
prior_sales=(sales[1]+sales[2])/2
sales_pct=(sales[0]/prior_sales)*100
Taxes=
Amort=
DEP=fin.Reconciled Depreciation[0]
non_cwc=0
fcf=cash.loc('Free Cash Flow')
fcf=fcf[0]
UFCF=EBIT-Taxes+DEP+Amort-Capex-non_cwc


# In[26]:


# Key inputs from DCF model
years = 5
starting_sales = 80.0
capex_percent = depr_percent = 0.032
sales_growth = 0.1
ebitda_margin = 0.14
nwc_percent = 0.24
tax_rate = 0.21
# DCF assumptions
r = 0.12
g = 0.02
# For MCS model
iterations = 1000
sales_std_dev = 0.01
ebitda_std_dev = 0.02
nwc_std_dev = 0.01


# In[27]:


def run_mcs():
    
    # Generate probability distributions
    sales_growth_dist = np.random.normal(loc=sales_growth, scale=sales_std_dev, size=(years, iterations))
    
    ebitda_margin_dist = np.random.normal(loc=ebitda_margin, scale=ebitda_std_dev, size=(years, iterations))
    
    nwc_percent_dist = np.random.normal(loc=nwc_percent, scale=nwc_std_dev, size=(years, iterations))
    
    # Calculate free cash flow
    sales_growth_dist += 1
    
    for i in range(1, len(sales_growth_dist)):
        sales_growth_dist[i] *= sales_growth_dist[i-1]
        
    sales = sales_growth_dist * starting_sales
    ebitda = sales * ebitda_margin_dist
    ebit = ebitda - (sales * depr_percent)
    tax = -(ebit * tax_rate)
    np.clip(tax, a_min=None, a_max=0)
    nwc = nwc_percent_dist * sales
    starting_nwc = starting_sales * nwc_percent
    prev_year_nwc = np.roll(nwc, 1, axis=0)
    prev_year_nwc[0] = starting_nwc
    delta_nwc = prev_year_nwc - nwc
    capex = -(sales * capex_percent)
    free_cash_flow = ebitda + tax + delta_nwc + capex
    
    # Discount cash flows to get DCF value
    terminal_value = free_cash_flow[-1] * (1 + g) / (r - g)
    discount_rates = [(1 / (1 + r)) ** i for i in range (1,6)]
    dcf_value = sum((free_cash_flow.T * discount_rates).T) 
    dcf_value += terminal_value * discount_rates[-1]
        
    return dcf_value


# In[28]:


get_ipython().run_line_magic('time', 'plt.hist(run_mcs(), bins=20, density=True, color="r")')
plt.show()


# In[ ]:




