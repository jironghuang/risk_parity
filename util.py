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
    
    
def realised_equity_curve(strategy, dir_path, start_date, end_date, account):

    #Create sequence of dates
    datelist = pd.bdate_range(start = start_date, end = end_date, freq = 'B')
    df = pd.DataFrame({'dates': datelist})
    df['nlv'] = 0.0
    df['assets'] = 0.0
    
    for i in range(df.shape[0]):
        
        try:              
            nlv_path = dir_path + re.sub(" 00:00:00|-", "", str(df['dates'][i])) + '/acc_snapshot.csv'
            nlv = pd.read_csv(nlv_path)
            df.loc[:,'nlv'][i] = nlv['nlv'][0]            
            df.loc[:,'assets'][i] = nlv['value'][0]
                 
        except:
            print('no data for day')
            
    #Subset out days with 0 nlv
    df = df[df['nlv']>0]             
            
    df['leverage'] = df['assets']/df['nlv']           
    
    #Merge in cashflow
    cashflow = pd.read_csv('deposit_withdrawal.csv')
    cashflow['Date'] = pd.to_datetime(cashflow['Date'], format = '%d-%m-%Y')
    cashflow = cashflow[cashflow.account == account] 
    
    straddle_flow = pd.read_csv('straddle/straddle_profits.csv')
    straddle_flow['Date'] = pd.to_datetime(straddle_flow['Date'], format = '%d-%m-%Y')    
    straddle_flow = straddle_flow[straddle_flow.account == account]     
    
    if straddle_flow.shape[0] > 0:
        cashflow = pd.concat([straddle_flow, cashflow])
        cashflow.sort_values(by='Date',inplace = True)        
    
    #Left join
    eqcurve = pd.merge(df, cashflow, left_on = 'dates', right_on = 'Date', how = 'left')
    eqcurve['pct_change'] = eqcurve['nlv'].pct_change(1)
    eqcurve['nlv_lag'] = eqcurve['nlv'].shift(periods=1)
    eqcurve['pct_change_if_cf']  = (eqcurve['nlv'] - (eqcurve['nlv_lag'] + eqcurve['usd']))/(eqcurve['nlv_lag'] + eqcurve['usd'])
    eqcurve['pct_change_adj'] = np.where(np.isnan(eqcurve['usd']), eqcurve['pct_change'], eqcurve['pct_change_if_cf'])
    eqcurve['equity_curve'] = (1 + eqcurve['pct_change_adj']).cumprod()
    eqcurve['equity_curve'][0] = 1    
    
    #Read in USD exchange rate
    price_series = get_adj_open_close(tickers = ['USDSGD=X', 'VOO'], start_date = '2016-12-31', end_date = '2021-12-31', api = 'yfinance')                
    price_series = price_series[['USDSGD=X_adj_close_price']]
    price_series.columns = ['exch_rate']
    price_series['exch_rate'] = price_series['exch_rate'].ffill() 
    eqcurve = pd.merge(eqcurve, price_series, left_on = 'dates', right_on = 'Date', how = 'left')
        
    #Convert to sgd
    eqcurve['nlv_sgd'] = eqcurve['nlv'] * eqcurve['exch_rate']
    eqcurve['assets_sgd'] = eqcurve['assets'] * eqcurve['exch_rate']    
    eqcurve['pct_change_sgd'] = eqcurve['nlv_sgd'].pct_change(1)    
    eqcurve['nlv_lag_sgd'] = eqcurve['nlv_sgd'].shift(periods=1)
    eqcurve['pct_change_if_cf_sgd']  = (eqcurve['nlv_sgd'] - (eqcurve['nlv_lag_sgd'] + eqcurve['sgd']))/(eqcurve['nlv_lag_sgd'] + eqcurve['sgd'])
    eqcurve['pct_change_adj_sgd'] = np.where(np.isnan(eqcurve['sgd']), eqcurve['pct_change_sgd'], eqcurve['pct_change_if_cf_sgd'])
    eqcurve['equity_curve_sgd'] = (1 + eqcurve['pct_change_adj_sgd']).cumprod()
    eqcurve['equity_curve_sgd'][0] = 1 
    

    #Plot rolling standard deviation
    eqcurve['rolling_sd_usd'] = 100 * (252 ** 0.5)* eqcurve['pct_change_adj'].rolling(21).std() 
    eqcurve['rolling_sd_sgd'] = 100 * (252 ** 0.5)* eqcurve['pct_change_adj_sgd'].rolling(21).std() 
          
    #Plotting curve               
    fig, ax1 = plt.subplots()    
    ax2 = ax1.twinx()
    ax1.plot(eqcurve['dates'], eqcurve['equity_curve'], label = 'equity_curve_usd', color = 'b')
    ax1.plot(eqcurve['dates'], eqcurve['equity_curve_sgd'], label = 'equity_curve_sgd', color = 'r')    
    ax2.plot(eqcurve['dates'], eqcurve['rolling_sd_usd'], label = 'rolling_sd_usd', color = 'c')
    ax2.plot(eqcurve['dates'], eqcurve['rolling_sd_sgd'], label = 'rolling_sd_sgd', color = 'y')    
    ax1.set_xlabel('Dates')
    ax1.set_ylabel('Equity_curve')
    ax2.set_ylabel('Rolling annualized volatility (%)')
    ax1.legend(loc = 'lower left')
    ax2.legend(loc = 'lower right')
    ax1.tick_params(labelrotation=90, axis = 'x', labelsize=3)
    ax2.tick_params(labelrotation=90, axis = 'x', labelsize=3)      
        
    try:     
        os.remove(strategy + '_equity_curve.png')    
    except:
        print('Missing equity curve')
    
    #plt.show()
    fig.savefig(strategy + '_equity_curve', dpi = 600)
    plt.close()
    
    #Drawdown feature
    #Initiate drawdown data-frame
    eqcurve['hwm'] = eqcurve['equity_curve'].expanding().max()    
    eqcurve['current_drawdown'] = (eqcurve['equity_curve'] - eqcurve['hwm'])/eqcurve['hwm']
    eqcurve['drawdown_index'] = -1
           
    drawdown_period = 0    
    for r in range(1, eqcurve.shape[0]):

        #In new drawdown
        if((eqcurve['current_drawdown'][r] != 0) & (eqcurve['current_drawdown'][r-1] == 0)):
            drawdown_period += 1
            eqcurve['drawdown_index'][r] = drawdown_period

        #In current drawdown
        elif((eqcurve['current_drawdown'][r] != 0) & (eqcurve['current_drawdown'][r-1] != 0)):
            eqcurve['drawdown_index'][r] = drawdown_period    
            

    #Create drawdown table
    start_date = None
    end_date = None
    index_series = []
    start_date_series = []
    end_date_series = []                
    drawdown_series = []
    is_ongoing_series = []    
    
    for r in range(1, eqcurve.shape[0]):
        
        if((eqcurve['drawdown_index'][r] != -1) & (eqcurve['drawdown_index'][r-1] == -1)):           
            start_date = eqcurve['dates'][r]        
        
        if((eqcurve['drawdown_index'][r] == -1) & (eqcurve['drawdown_index'][r-1] != -1)):        
            end_date = eqcurve['dates'][r-1]
            start_date_series.append(start_date)
            end_date_series.append(end_date)
            index_series.append(eqcurve['drawdown_index'][r-1])
            is_ongoing_series.append(0)
            
        if((eqcurve['drawdown_index'][r] != -1) & (r == (eqcurve.shape[0]-1) )):
            end_date = eqcurve['dates'][r]            
            start_date_series.append(start_date)
            end_date_series.append(end_date)
            index_series.append(eqcurve['drawdown_index'][r])            
            is_ongoing_series.append(1)           
    
    df = pd.DataFrame({'drawdown_index':pd.Series(index_series), 
                      'start_date':pd.Series(start_date_series),
                      'end_date':pd.Series(end_date_series),
                      'is_ongoing':pd.Series(is_ongoing_series),                       
                      }
                     )      
    
    drawdown_table = eqcurve[['drawdown_index','current_drawdown']].groupby('drawdown_index').min()   
    drawdown_table = drawdown_table * 100
    drawdown_table['drawdown_index'] = drawdown_table.index        
    drawdown_table = drawdown_table[['drawdown_index', 'current_drawdown']]
    drawdown_table.index.name = 'index'

    drawdown_table = pd.merge(drawdown_table, df, left_on = 'drawdown_index', right_on = 'drawdown_index', how = 'left')    
       
    drawdown_table = drawdown_table[drawdown_table.drawdown_index != - 1]
    
    drawdown_table['drawdown_days_duration'] = drawdown_table['end_date'] - drawdown_table['start_date']
    drawdown_table['drawdown_days_duration'] = drawdown_table['drawdown_days_duration'] + dt.timedelta(days=1)
    
    drawdown_table['drawdown_bdays_duration'] = np.busday_count(drawdown_table['start_date'].values.astype('datetime64[D]'), drawdown_table['end_date'].values.astype('datetime64[D]'))    
    drawdown_table['drawdown_bdays_duration'] = drawdown_table['drawdown_bdays_duration'] + 1
    drawdown_table.rename(columns = {"current_drawdown": "drawdown_perc"}, inplace = True)
    
    #Write out tables
    eqcurve = eqcurve[['dates','nlv','nlv_sgd','usd','sgd','assets','assets_sgd','leverage','pct_change_adj','pct_change_adj_sgd','equity_curve','equity_curve_sgd', 'rolling_sd_usd', 'rolling_sd_sgd']]    
    eqcurve.to_csv(strategy + '_equity_curve.csv', index = False)
    
    drawdown_table.to_csv(strategy + '_drawdown.csv', index = False)
    
    return eqcurve


def combined_equity_curve(strategies):
        
    for i in range(len(strategies)):
    
        if i == 0:
        
            eq_curve_path = strategies[i] + '_equity_curve.csv'
            eq_curve = pd.read_csv(eq_curve_path)    
            eq_curve.columns = eq_curve.columns + '_' + strategies[i]
            eq_curve.rename(columns={ eq_curve.columns[0]: "dates" }, inplace = True)

        else:

            eq_curve_path = strategies[i] + '_equity_curve.csv'
            eq_curve_ind = pd.read_csv(eq_curve_path)    
            eq_curve_ind.columns = eq_curve_ind.columns + '_' + strategies[i]
            eq_curve_ind.rename(columns={ eq_curve_ind.columns[0]: "dates" }, inplace = True)            
            
            #Left join all the dataframe
            eqcurve = pd.merge(eq_curve, eq_curve_ind, left_on = 'dates', right_on = 'dates', how = 'left')   

    #Equity curve of usd
    eqcurve['nlv'] = eqcurve[['nlv_' + s for s in strategies]].sum(axis=1)
    eqcurve['assets'] = eqcurve[['assets_' + s for s in strategies]].sum(axis=1)      
    eqcurve['pct_change'] = eqcurve['nlv'].pct_change(1)
    eqcurve['nlv_lag'] = eqcurve['nlv'].shift(periods=1)
    eqcurve['usd'] = eqcurve[['usd_' + s for s in strategies]].sum(axis=1)    
    eqcurve['pct_change_if_cf']  = (eqcurve['nlv'] - (eqcurve['nlv_lag'] + eqcurve['usd']))/(eqcurve['nlv_lag'] + eqcurve['usd'])
    eqcurve['pct_change_adj'] = np.where(np.isnan(eqcurve['usd']), eqcurve['pct_change'], eqcurve['pct_change_if_cf'])
    eqcurve['equity_curve'] = (1 + eqcurve['pct_change_adj']).cumprod()
    eqcurve['equity_curve'][0] = 1    
           
    #Convert to sgd
    eqcurve['nlv_sgd'] = eqcurve[['nlv_sgd_' + s for s in strategies]].sum(axis=1)
    eqcurve['assets_sgd'] = eqcurve[['assets_sgd_' + s for s in strategies]].sum(axis=1)   
    eqcurve['pct_change_sgd'] = eqcurve['nlv_sgd'].pct_change(1)    
    eqcurve['nlv_lag_sgd'] = eqcurve['nlv_sgd'].shift(periods=1)
    eqcurve['sgd'] = eqcurve[['sgd_' + s for s in strategies]].sum(axis=1)     
    eqcurve['pct_change_if_cf_sgd']  = (eqcurve['nlv_sgd'] - (eqcurve['nlv_lag_sgd'] + eqcurve['sgd']))/(eqcurve['nlv_lag_sgd'] + eqcurve['sgd'])
    eqcurve['pct_change_adj_sgd'] = np.where(np.isnan(eqcurve['sgd']), eqcurve['pct_change_sgd'], eqcurve['pct_change_if_cf_sgd'])
    eqcurve['equity_curve_sgd'] = (1 + eqcurve['pct_change_adj_sgd']).cumprod()
    eqcurve['equity_curve_sgd'][0] = 1                      
            
    #Plot rolling standard deviation
    eqcurve['rolling_sd_usd'] = 100 * (252 ** 0.5)* eqcurve['pct_change_adj'].rolling(21).std() 
    eqcurve['rolling_sd_sgd'] = 100 * (252 ** 0.5)* eqcurve['pct_change_adj_sgd'].rolling(21).std()     
        
    #Plotting curve               
    fig, ax1 = plt.subplots()    
    ax2 = ax1.twinx()
    ax1.plot(eqcurve['dates'], eqcurve['equity_curve'], label = 'equity_curve_usd', color = 'b')
    ax1.plot(eqcurve['dates'], eqcurve['equity_curve_sgd'], label = 'equity_curve_sgd', color = 'r')    
    ax2.plot(eqcurve['dates'], eqcurve['rolling_sd_usd'], label = 'rolling_sd_usd', color = 'c')
    ax2.plot(eqcurve['dates'], eqcurve['rolling_sd_sgd'], label = 'rolling_sd_sgd', color = 'y')    
    ax1.set_xlabel('Dates', fontsize = 10)
    ax1.set_ylabel('Equity_curve', fontsize = 10)
    ax2.set_ylabel('Rolling annualized volatility (%)')
    ax1.legend(loc = 'lower left')
    ax2.legend(loc = 'lower right')
    ax1.tick_params(labelrotation=90, axis = 'x', labelsize=3)
    ax2.tick_params(labelrotation=90, axis = 'x', labelsize=3)    
        
    try:     
        os.remove('combined_equity_curve.png')    
    except:
        print('Missing equity curve')
    
    #plt.show()
    fig.savefig('combined_equity_curve', dpi = 600)
    plt.close()    
    
    #Drawdown feature
    #Initiate drawdown data-frame
    eqcurve['hwm'] = eqcurve['equity_curve'].expanding().max()    
    eqcurve['current_drawdown'] = (eqcurve['equity_curve'] - eqcurve['hwm'])/eqcurve['hwm']
    eqcurve['drawdown_index'] = -1
           
    drawdown_period = 0    
    for r in range(1, eqcurve.shape[0]):

        #In new drawdown
        if((eqcurve['current_drawdown'][r] != 0) & (eqcurve['current_drawdown'][r-1] == 0)):
            drawdown_period += 1
            eqcurve['drawdown_index'][r] = drawdown_period

        #In current drawdown
        elif((eqcurve['current_drawdown'][r] != 0) & (eqcurve['current_drawdown'][r-1] != 0)):
            eqcurve['drawdown_index'][r] = drawdown_period    
            

    #Create drawdown table
    start_date = None
    end_date = None
    index_series = []
    start_date_series = []
    end_date_series = []                
    drawdown_series = []
    is_ongoing_series = []    
    
    for r in range(1, eqcurve.shape[0]):
        
        if((eqcurve['drawdown_index'][r] != -1) & (eqcurve['drawdown_index'][r-1] == -1)):           
            start_date = eqcurve['dates'][r]        
        
        if((eqcurve['drawdown_index'][r] == -1) & (eqcurve['drawdown_index'][r-1] != -1)):        
            end_date = eqcurve['dates'][r-1]
            start_date_series.append(start_date)
            end_date_series.append(end_date)
            index_series.append(eqcurve['drawdown_index'][r-1])
            is_ongoing_series.append(0)
            
        if((eqcurve['drawdown_index'][r] != -1) & (r == (eqcurve.shape[0]-1) )):
            end_date = eqcurve['dates'][r]            
            start_date_series.append(start_date)
            end_date_series.append(end_date)
            index_series.append(eqcurve['drawdown_index'][r])            
            is_ongoing_series.append(1)           
    
    df = pd.DataFrame({'drawdown_index':pd.Series(index_series), 
                      'start_date':pd.Series(start_date_series),
                      'end_date':pd.Series(end_date_series),
                      'is_ongoing':pd.Series(is_ongoing_series),                       
                      }
                     )      
    
    drawdown_table = eqcurve[['drawdown_index','current_drawdown']].groupby('drawdown_index').min()   
    drawdown_table = drawdown_table * 100
    drawdown_table['drawdown_index'] = drawdown_table.index        
    drawdown_table = drawdown_table[['drawdown_index', 'current_drawdown']]
    drawdown_table.index.name = 'index'

    drawdown_table = pd.merge(drawdown_table, df, left_on = 'drawdown_index', right_on = 'drawdown_index', how = 'left')    
       
    drawdown_table = drawdown_table[drawdown_table.drawdown_index != - 1]
    
    drawdown_table['end_date'] = pd.to_datetime(drawdown_table['end_date'])
    drawdown_table['start_date'] = pd.to_datetime(drawdown_table['start_date'])

    drawdown_table['drawdown_days_duration'] = drawdown_table['end_date']- drawdown_table['start_date']
    drawdown_table['drawdown_days_duration'] = drawdown_table['drawdown_days_duration'] + dt.timedelta(days=1)
    
    drawdown_table['drawdown_bdays_duration'] = np.busday_count(drawdown_table['start_date'].values.astype('datetime64[D]'), drawdown_table['end_date'].values.astype('datetime64[D]'))    
    drawdown_table['drawdown_bdays_duration'] = drawdown_table['drawdown_bdays_duration'] + 1
    drawdown_table.rename(columns = {"current_drawdown": "drawdown_perc"}, inplace = True)
    
    #Write out tables
    eqcurve.to_csv('combined_equity_curve.csv', index = False)    
    drawdown_table.to_csv('combined_drawdown.csv', index = False)    
                
    eqcurve[['assets', 'assets_sgd', 'nlv', 'nlv_sgd', 'rolling_sd_usd', 'rolling_sd_sgd', 'current_drawdown']][-1:].to_csv('acc_snapshot.csv', index = False) 
    
    pass


#Performance attribution of equity curves   
def performance_attribution(strategies, currency = 'usd'):
    
    #Read in combined equity curve            
    combined_equity_curves = pd.read_csv('combined_equity_curve.csv')    
        
    #Extract relevant columns (nlv, equity_curve)
    if currency == 'usd':  
        
        nlv_cols = ['nlv_' + s for s in strategies]
        #ec_cols = ['equity_curve_' + s for s in strategies]
        
        ret_cols = ['pct_change_adj_' + s for s in strategies]
        ret = np.array(combined_equity_curves[ret_cols])
        
        cols = ['usd_' + s for s in strategies]         
        flow = np.array(combined_equity_curves[cols])
        flow[np.isnan(flow)] = 0
    
    else:

        nlv_cols = ['nlv_' + currency + '_' + s for s in strategies]
        #ec_cols = ['equity_curve_' + currency + '_' + s for s in strategies]
        
        ret_cols = ['pct_change_adj_' + currency + '_' + s for s in strategies]
        ret = np.array(combined_equity_curves[ret_cols])
        
        cols = [currency + '_' + s for s in strategies]         
        flow = np.array(combined_equity_curves[cols])
        flow[np.isnan(flow)] = 0
        
    
    #Derive weights
    nlv_lag_adj = np.array(combined_equity_curves[nlv_cols].shift(periods=1)) +  flow
    ret_weights = nlv_lag_adj / nlv_lag_adj.sum(axis=1, keepdims=True)
    
    #Weighted returns
    weighted_ret = ret * ret_weights
    
    #Check diff after decomposition
    weighted_ret.sum(axis = 1) - np.array(combined_equity_curves.pct_change_adj)
    
    #Convert to dataframe
    weighted_ret_df = pd.DataFrame(weighted_ret)
    weighted_ret_df.columns = ['weighted_ret_' + s for s in strategies]
    weighted_ret_df.index = combined_equity_curves['dates']
    
    if currency == 'usd':     
        weighted_ret_df['equity_curve'] = np.array(combined_equity_curves['equity_curve'])
    else:
        weighted_ret_df['equity_curve'] = np.array(combined_equity_curves['equity_curve_' + currency])        
    
    #Weighted equity curve
    for s in strategies:
        weighted_ret_df['weighted_equity_curve_' + s] = (1+weighted_ret_df['weighted_ret_' + s]).cumprod()
        weighted_ret_df['weighted_equity_curve_' + s] = weighted_ret_df['weighted_equity_curve_' + s] - 1        
    
    weighted_ret_df['equity_curve'] = weighted_ret_df['equity_curve'] - 1    
    
    eq_curves = weighted_ret_df[['weighted_equity_curve_' + s for s in strategies] + ['equity_curve']]
    eq_curves.plot()
                
    try:     
        os.remove('weighted_equity_curve_' + currency + '.png')    
    except:
        print('Missing equity curve')   
            
    plt.savefig('weighted_equity_curve_' + currency, dpi = 600)
    plt.close()        

    weighted_ret_df.to_csv('weighted_equity_curve_' + currency + '.csv', index = True) 
    
    pass   


if __name__=="__main__":
            
    if socket.gethostname() == 'jirong-UX430UNR':
    
        strategy = 'risk_parity'
        dir_path = '/run/user/1000/gvfs/smb-share:server=jirong-fitlet2,share=household_wealth/risk_parity_workspace/' 
        start_date ='2020-09-08'
        end_date ='2020-12-11'
        account = 'xxx'
        eq_curve_risk_parity = realised_equity_curve(strategy, dir_path, start_date, end_date, account)
            
        strategy = 'vol_risk_prem'
        dir_path = '/run/user/1000/gvfs/smb-share:server=jirong-fitlet2,share=household_wealth/vol_risk_prem_workspace/' 
        start_date ='2020-09-08'
        end_date ='2020-12-11'
        account = 'xxx'
        eq_curve_vrp = realised_equity_curve(strategy, dir_path, start_date, end_date, account)
        
        strategies = ['risk_parity', 'vol_risk_prem']

    else:
        
        strategy = 'risk_parity'
        dir_path = './risk_parity_workspace/' 
        start_date ='2020-09-08'
        end_date ='2020-10-05'
        account = 'xxx'
        realised_equity_curve(strategy, dir_path, start_date, end_date, account)
            
        strategy = 'vol_risk_prem'
        dir_path = './vol_risk_prem_workspace/' 
        start_date ='2020-09-08'
        end_date ='2020-10-05'
        account = 'xxx'
        realised_equity_curve(strategy, dir_path, start_date, end_date, account)    
        
        #Try cbinding options inflow to inflow_outflow
        cashflow = pd.read_csv('deposit_withdrawal.csv')
        cashflow['Date'] = pd.to_datetime(cashflow['Date'], format = '%d-%m-%Y')
        
        straddle_flow = pd.read_csv('straddle/straddle_profits.csv')
        straddle_flow['Date'] = pd.to_datetime(straddle_flow['Date'], format = '%d-%m-%Y')
        
        cashflow = pd.concat([straddle_flow, cashflow])
        cashflow.sort_values(by='Date',inplace = True)