import datetime as dt
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime
import yfinance as yf
import re
import matplotlib.pyplot as plt
import os
import socket

def get_first_business_day_ofmonth(start_date = '2015-01-01', end_date = '2021-12-31'):

    """
    Return dataframe on first business day of month.

    :param start_date: Starting date range
    :param end_date: Ending date range
    :return: returns dataframe on first business day of month.
    """    
    
    if not( (type(start_date) is str) & (type(end_date) is str) ):
        raise ValueError('start and end date must be string of YYYY-MM-DD')    
    
    cal = USFederalHolidayCalendar()
        
    def get_business_day(date):
        while date.isoweekday() > 5 or date in cal.holidays():
            date += dt.timedelta(days=1)
        return date
    
    first_bday_of_month = [get_business_day(d).date() for d in pd.date_range(start_date, end_date, freq='BMS')]
    
    first_business_day_ofmonth = pd.DataFrame (first_bday_of_month, columns=['first_business_day'])
    first_business_day_ofmonth['first_business_day_indicator'] = 1
    first_business_day_ofmonth.index = first_business_day_ofmonth['first_business_day']

    return first_business_day_ofmonth
    
    
def get_adj_open_close(tickers = ['AAPL', 'FB'], start_date = '2015-01-01', end_date = '2021-12-31', api = 'yfinance'):
            
    if api == 'yfinance':
    
        prices_df = yf.download(tickers, start = start_date, end = end_date, adjusted = True)
        prices_df = prices_df.ffill(axis = 'rows')
        
        open_df = prices_df['Open'] * (prices_df['Adj Close']/ prices_df['Close'])
        open_df = open_df.add_suffix('_adj_open_price')
        
        close_df = prices_df['Adj Close']
        close_df = close_df.add_suffix('_adj_close_price')
        
        price_df = pd.merge(close_df, open_df, how ='left', left_index = True, right_index = True)
    
    return price_df


def get_sharpe(return_stream):
    
    sharpe_ratio = (return_stream.mean() / return_stream.std()) * np.sqrt(252)
    
    return sharpe_ratio


def get_sortino(return_stream):
    
    downside_returns = return_stream[return_stream < 0]
    
    sortino_ratio = (return_stream.mean() / downside_returns.std()) * np.sqrt(252)
    
    return sortino_ratio


#get_max_drawdown(a['portfolio_vol_return'].iloc[1:])
def get_max_drawdown(return_stream):
    
    # Cumulative product of portfolio returns
    cumprod_ret = (return_stream + 1).cumprod()*100
        
    # Convert the index in datetime format
    cumprod_ret.index = pd.to_datetime(cumprod_ret.index)
    
    # Define a variable trough_index to store the index of lowest value before new high
    trough_index = (np.maximum.accumulate(cumprod_ret) - cumprod_ret).idxmax()
    
    # Define a variable peak_index to store the index of maximum value before largest drop
    peak_index = cumprod_ret.loc[:trough_index].idxmax()
    
    # Calculate the maximum drawdown using the given formula
    maximum_drawdown = 100 * \
        (cumprod_ret[trough_index] - cumprod_ret[peak_index]) / \
        cumprod_ret[peak_index]    
    
    return maximum_drawdown

def get_annual_returns(return_stream):
    
    # Total number of trading days in a year is 252
    trading_days = 252
    
    # Calculate the average daily returns
    average_daily_returns = return_stream.mean()    
    
    annual_returns = ((1 + average_daily_returns)**(trading_days) - 1) * 100
    
    return annual_returns

def get_compound_returns(return_stream):
    
    # Total number of trading days in a year is 252
    trading_days=return_stream.shape[0]        
    daily_ret = ((1 + return_stream).cumprod().iloc[-1]) ** (1/trading_days) - 1
    
    annual_returns = ((1 + daily_ret)**(252) - 1) * 100
    
    return annual_returns

def get_skewness(return_stream):
    return(return_stream.skew())

def get_kurtosis(return_stream):
    return(return_stream.kurtosis())
    