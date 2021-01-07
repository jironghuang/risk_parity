#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:35:14 2021

@author: jirong
"""

import riskparityportfolio as rp
import numpy as np
import pandas as pd
import yfinance as yf
import functools
import re
import pyfolio
import sys
import util as ut #User-defined function

def risk_parity_weights(cov_matrix, concatenate_weights = True):
    """
    Return risk parity weights.

    :param cov_matrix: covariance matrix
    :param contenate_weights: cocnatenate weights into string
    :return: returns tuple of risk parity weights and risk contribution
    """
    if not(type(cov_matrix) is np.ndarray):
        raise ValueError('covariance matrix must be a numpy array')
    
    try:    
        risk_contribution = np.array([1/cov_matrix.shape[1]] * cov_matrix.shape[1])
        risk_weights = rp.vanilla.design(cov_matrix,risk_contribution)
        
        if concatenate_weights == True:            
            delimiter = ','
            concat_risk_contribution, concat_risk_weights  = (delimiter.join(np.ndarray.tolist(np.char.mod('%f', risk_contribution))),\
                                                              delimiter.join(np.ndarray.tolist(np.char.mod('%f', risk_weights))))
            
            return concat_risk_contribution, concat_risk_weights
        else:            
            return risk_contribution, risk_weights
    
    except:
        print('Error with computation of risk parity weights')

def risk_parity_weights_single_chunk (time_series_input):
    """
    Return risk parity based on single time slice.

    :param cov_matrix: time_series_input
    :return: returns concatenated risk parity weights
    """
    if not(type(time_series_input) is pd.core.frame.DataFrame or type(time_series_input) is pd.Series):
        raise ValueError('Time series input matrix must be of type pd.core.frame.DataFrame or pd.Series')

    try:    
        #Create covariance matrix
        cov_matrix = time_series_input.cov()
        
        #apply risk_parity_weights function
        weights = risk_parity_weights(np.array(cov_matrix), True)[1]
    
        return weights

    except:
        print('Error in applying risk parity weights to single chunk of time series')
       
def rolling_risk_parity_weights(time_series_input, window):
    
    """
    Return rolling risk parity weights

    :param cov_matrix: time_series_input
    :param window: window length
    :return: returns concatenated risk parity weights
    """    
    
    if not(type(time_series_input) is pd.core.frame.DataFrame or type(time_series_input) is pd.Series):
        raise ValueError('Time series input matrix must be of type pd.core.frame.DataFrame or pd.Series')
        
    try:
    
        rolled_weights = pd.concat([(pd.Series(risk_parity_weights_single_chunk(time_series_input.iloc[i:i+window]), \
                               index=[time_series_input.index[i+window]])) for i in range(len(time_series_input)-window) ])
        
        rolled_weights = rolled_weights.to_frame()
        rolled_weights.rename(columns={ rolled_weights.columns[0]: "concat_risk_weights" }, inplace = True)
            
        rolled_weights[time_series_input.columns.values] = rolled_weights['concat_risk_weights'].str.split(",",expand=True,)
        
    except:
        print('Error in applying risk parity weights to time series')        

    return rolled_weights

def rolling_risk_parity_returns (tickers = ['TLT', 'IEF', 'GLD', 'SPY'], 
                                 start_date = '2015-01-01',
                                 end_date = '2021-12-31', 
                                 api = 'yfinance',
                                 freq = 'D', window = 252, #freq = 'W'
                                 starting_amount = 1000000):
    
    """
    Return rolling risk parity weights

    :param tickers: list of tickers for risk parity
    :param start_date: start_date
    :param end_date: end_date
    :param api: api used
    :param freq: daily (D) or resample to weekly (W)    
    :param window: window length
    :param starting_amount: starting_amount of capital
    :return: returns concatenated risk parity weights
    """     
        
    #Get price 
    price_input = ut.get_adj_open_close(tickers = tickers, start_date = start_date, end_date = end_date, api = api)
    
    #Resample if week
    if freq == 'W':
        price_input = price_input.resample('W-FRI').last().ffill()
    
    #Compute returns from price    
    price_returns = price_input.pct_change()
    price_returns = price_returns.add_suffix('_returns')
    
    #Compute returns on actual day
    adj_open_price_cols = [col for col in price_input.columns if 'adj_open_price' in col]
    adj_close_price_cols = [col for col in price_input.columns if 'adj_close_price' in col]
    price_day_returns = pd.DataFrame( (np.array(price_input[adj_close_price_cols]) - np.array(price_input[adj_open_price_cols]))/ 
                                     np.array(price_input[adj_open_price_cols]), index = price_input.index, columns = adj_close_price_cols)
    price_day_returns = price_day_returns.add_suffix('_day_returns')
    
    #Cbind with price series
    price_input_returns = pd.merge(price_input, price_returns, how='left', left_index=True, right_index=True)
    price_input_returns = pd.merge(price_input_returns, price_day_returns, how='left', left_index=True, right_index=True)

    #Define adjusted close price returns columns        
    adj_close_returns_cols = [col for col in price_returns.columns if 'adj_close_price_returns' in col]
    
    #compute and left_join weights
    risk_parity_weights = rolling_risk_parity_weights(price_returns[adj_close_returns_cols], window)
    risk_parity_weights = risk_parity_weights.add_suffix('_weights')
    price_weights = pd.merge(price_input_returns, risk_parity_weights, how = 'left', left_index=True, right_index=True)

    #Drop NA for weights
    price_weights = price_weights[price_weights['concat_risk_weights_weights'].notna()]
    
    #Initial Position Values and Quantities
    initial_PosV_wt_columns =  [ticker + '_adj_close_price_returns_weights' for ticker in tickers]
    cur_date = price_weights.index[0]        
    initial_PosV = [starting_amount * float(price_weights.loc[cur_date,initial_PosV_wt_columns][i]) for i in range(len(price_weights.loc[cur_date,initial_PosV_wt_columns]))]  
        
    price_columns = [ticker + '_adj_close_price' for ticker in tickers]    
    initial_PosQ = (starting_amount * np.array(price_weights.loc[cur_date,initial_PosV_wt_columns]).astype(float))/\
                    np.array(price_weights.loc[cur_date,price_columns]).astype(float)
    
                          
    #Create additional columns for tracking position, values for various securities    
    ticker_pos = [ticker + '_pos' for ticker in tickers]        #Quantities
    ticker_val = [ticker + '_val' for ticker in tickers]        #Values
    new_cols = ticker_pos + ticker_val    
    price_weights = pd.concat([price_weights, pd.DataFrame(columns=new_cols)], axis=1)

    #Allocation at beginning                 
    price_weights.loc[price_weights.index[0],ticker_val] = initial_PosV
    price_weights.loc[price_weights.index[0],ticker_pos] = initial_PosQ  

    #Run strategy    
    for day in range(1, len(price_weights)):
        cur_date                         = price_weights.index[day]
        prev_date                        = price_weights.index[day-1]
        price_weights.loc[cur_date,ticker_pos] = price_weights.loc[prev_date,ticker_pos]
        price_weights.loc[cur_date,ticker_val] = (price_weights.loc[cur_date,ticker_pos].values) * (price_weights.loc[cur_date,price_columns].values)

        #Rebalance
        #if data.loc[cur_date,"rebalance"] == True:
        target_weights               = list(np.array(price_weights.loc[cur_date,initial_PosV_wt_columns]).astype(float))
        price_weights.loc[cur_date,ticker_val] = [sum(price_weights.loc[cur_date,ticker_val])*wt for wt in target_weights]
        price_weights.loc[cur_date,ticker_pos] = (price_weights.loc[cur_date,ticker_val].values) / (price_weights.loc[cur_date,price_columns].values)
    
    #Sum up portfolio value and obtain retunrs    
    price_weights['portfolio_value'] = price_weights[ticker_val].sum(axis=1)   
    price_weights['portfolio_ret'] = price_weights['portfolio_value'].pct_change()
    
    return price_weights

#Event driven backtesting; simulating commission fees and financing fees
def rolling_risk_parity_asset_class_returns (asset_class_dict, 
                                             start_date = '2015-01-01',
                                             end_date = '2021-12-31', 
                                             api = 'yfinance',
                                             freq = 'D', #freq = 'W' or 'D'
                                             window = 252, 
                                             window_week=52, 
                                             volatility_targeting = 0.1,
                                             volatility_targeting_lookback = 36,
                                             max_leverage = 2.0,         
                                             perc_diff_before_rebalancing = 0.15,  
                                             comm_fee = 2,
                                             financing_fee = 0.015,
                                             leverage = 'partial',    #leverage: full or partial
                                             starting_amount = 1000000):
    
    """
    Return rolling risk parity weights

    :param asset_class_dict: asset_class_dict containing list of tickers for in each of asset class
    :param start_date: start_date
    :param end_date: end_date
    :param api: api used
    :param freq: daily (D) or resample to weekly (W)    
    :param window: window length
    :param window_week: window length used if sampled weekly
    :param volatility_targeting: Exponential realized volatility target cap (e.g. 0.1)
    :param volatility_targeting_lookback: Lookback used for risk parity (e.g. 36)
    :param max_leverage: Max leverage used (e.g. 2.0)         
    :param perc_diff_before_rebalancing: Percentage different from optimal position before rebalancing (e.g. 0.15)  
    :param comm_fee: Commission fee per trade (e.g. 2)
    :param financing_fee: Annual financing rate (e.g. 0.015)
    :param leverage: Financing fee on entire asset or leveraged portion (e.g. 'partial' or 'full')    
    :param starting_amount: starting_amount of capital (e.g. 1000000)
    :return: returns dataframe with full parameters and returns data
    """    
        
    ##################################################
    #Obtain all risk parity returns within asset class
    #Daily and weekly returns
    ##################################################
    time_series = [rolling_risk_parity_returns(asset_class_dict[i], start_date = start_date, end_date = end_date) for i in asset_class_dict]   
    time_series_week = [rolling_risk_parity_returns(asset_class_dict[i], start_date = start_date, end_date = end_date, freq='W', window=52) for i in asset_class_dict]   
                      
    #############################################################################
    #Function to extract portfolio value and returns from each of the asset class
    #############################################################################
    def extract_portfolio_weights_value_returns(time_series):
       
       #Extract rolling weights
       weights_cols = [col for col in time_series.columns if 'returns_weights' in col]
       weights = time_series[weights_cols]

       #Extract portfolio value
       portval_cols = [col for col in time_series.columns if 'portfolio_value' in col]
       portfolio_value = time_series[portval_cols]

       #Extract relevant columns
       relevant_cols = [col for col in time_series.columns if 'adj_close_price' in col or 'adj_open_price' in col or 'returns_weights' in col or 'day_returns' in col]
       relevant_data = time_series[relevant_cols]
       
       #Extract portfolio returns
       portret_cols = [col for col in time_series.columns if 'portfolio_ret' in col]
       portfolio_ret = time_series[portret_cols]
       
       #Change column name to key of dictionary
       for i in asset_class_dict:
           if re.sub("_adj_close_price_returns_weights", "", weights_cols[0]) in asset_class_dict[i]: 
               portfolio_ret = portfolio_ret.rename(columns = {'portfolio_return': i +'_portfolio_return'})
                      
       portfolio_dict = {
               'portfolio_ret': portfolio_ret,
               'portfolio_value': portfolio_value,
               'weights': weights,
               'relevant_cols': relevant_data
               }

       return portfolio_dict
   
    ###########################################################################################################
    #Extract weekly returns if freq = 'W'. Have to put this section ahead of Daily section if not theres a bug
    #If weekly weights, delete asset class weights and nested weights, merge in and front fill
    #Create weekly df
    ############################################################################################################
    if freq == 'W':
        returns_list_week = [extract_portfolio_weights_value_returns(i)['portfolio_ret'] for i in time_series_week]   
        relevant_data_list_week = [extract_portfolio_weights_value_returns(i)['relevant_cols'] for i in time_series_week]   
        returns_portfolio_week = functools.reduce(lambda left,right: pd.merge(left,right,how = 'left', left_index=True, right_index=True), returns_list_week)     
        risk_parity_weights_week = rolling_risk_parity_weights(returns_portfolio_week, window=window_week)
        risk_parity_weights_week = risk_parity_weights_week.drop([risk_parity_weights_week.columns[0]], axis=1)
        risk_parity_weights_week.columns = list(asset_class_dict.keys())
        risk_parity_weights_week = risk_parity_weights_week.add_suffix('_weights')     
        data_portfolio_week = functools.reduce(lambda left,right: pd.merge(left,right,how = 'left', left_index=True, right_index=True), relevant_data_list_week)  
        sel_col_names =  [col for col in data_portfolio_week.columns if '_adj_close_price_returns_weights' in col]   
        data_portfolio_week = data_portfolio_week[sel_col_names]
        #Merge in asset class risk parity weights
        price_weights_week = pd.merge(data_portfolio_week, risk_parity_weights_week, how='left', left_index=True, right_index=True)     
        price_weights_week = price_weights_week[price_weights_week[list(asset_class_dict.keys())[0] + '_weights'].notna()]
        
        for i in asset_class_dict:             
            within_asset_class_weights_cols = [ticker + '_adj_close_price_returns_weights' for ticker in asset_class_dict[i]] 
            ticker_val = [ticker + '_nested_weights' for ticker in asset_class_dict[i]] 
           
            nested_weights_week = \
            pd.DataFrame(
            np.array(np.array(price_weights_week[i + '_weights']).astype(float).reshape((price_weights_week.shape[0], 1))) * \
            np.array(price_weights_week[within_asset_class_weights_cols].astype(float)), columns=ticker_val, index = price_weights_week.index
            )
           
            price_weights_week = pd.concat([price_weights_week, nested_weights_week], axis = 1)    
            
        #Select and delete from daily data-frame
        weights_names = [col for col in price_weights_week.columns if '_weights' in col]
        price_weights_week = price_weights_week[weights_names]         
        price_weights_week['week'] = price_weights_week.index.weekofyear
        price_weights_week['year'] = price_weights_week.index.year
    
    #Create daily df
    #Extract portfolio returns and change column names
    returns_list = [extract_portfolio_weights_value_returns(i)['portfolio_ret'] for i in time_series]   
    relevant_data_list = [extract_portfolio_weights_value_returns(i)['relevant_cols'] for i in time_series]
   
    #Construct risk parity weights of asset classes
    returns_portfolio = functools.reduce(lambda left,right: pd.merge(left,right,how = 'left', left_index=True, right_index=True), returns_list)     
    risk_parity_weights = rolling_risk_parity_weights(returns_portfolio, window)
    risk_parity_weights = risk_parity_weights.drop([risk_parity_weights.columns[0]], axis=1)
    risk_parity_weights.columns = list(asset_class_dict.keys())
    risk_parity_weights = risk_parity_weights.add_suffix('_weights')     
    data_portfolio = functools.reduce(lambda left,right: pd.merge(left,right,how = 'left', left_index=True, right_index=True), relevant_data_list)  
           
    #Merge in asset class risk parity weights
    price_weights = pd.merge(data_portfolio, risk_parity_weights, how='left', left_index=True, right_index=True)     
    price_weights = price_weights[price_weights[list(asset_class_dict.keys())[0] + '_weights'].notna()]

    #Multiply asset class weights with tickers within asset class to obtain nested weights       
    for i in asset_class_dict:             
        within_asset_class_weights_cols = [ticker + '_adj_close_price_returns_weights' for ticker in asset_class_dict[i]] 
        ticker_val = [ticker + '_nested_weights' for ticker in asset_class_dict[i]] 
       
        nested_weights = \
        pd.DataFrame(
        np.array(np.array(price_weights[i + '_weights']).astype(float).reshape((price_weights.shape[0], 1))) * \
        np.array(price_weights[within_asset_class_weights_cols].astype(float)), columns=ticker_val, index = price_weights.index
        )
       
        price_weights = pd.concat([price_weights, nested_weights], axis = 1)
      
    #Store date, week, year into price_weights
    price_weights['date'] = price_weights.index          
    price_weights['week'] = price_weights.index.weekofyear
    price_weights['year'] = price_weights.index.year

    #Merging weekly data into daily data
    if freq == 'W':
        price_weights.drop([col for col in price_weights.columns if "_weights" in col], axis=1, inplace=True)         
        price_weights = pd.merge(price_weights,price_weights_week,how = 'left', left_on=['week','year'], right_on = ['week','year'])  
        price_weights.loc[:,weights_names] = price_weights[weights_names].ffill()
        price_weights.loc[:,weights_names] = price_weights[weights_names].bfill()
    
   
    #Obtain ticker column names
    tickers = [col for col in price_weights.columns if '_adj_close_price_returns_weights' in col]
    tickers = [re.sub("_adj_close_price_returns_weights", "", x) for x in tickers]
          
    #Initializing empty columns
    ticker_val = [ticker + '_val' for ticker in tickers]     
    nan_df = pd.DataFrame(0 * np.random.randint(price_weights.shape[0] * len(ticker_val), 
                            size=(price_weights.shape[0], len(ticker_val)) ), 
          columns=ticker_val, index = price_weights.index)
    
    price_weights = pd.concat([price_weights, nan_df], axis = 1)
    price_weights['portfolio_value'] = np.nan; price_weights['portfolio_value'][0] = starting_amount
    price_weights[ticker_val] = price_weights[ticker_val].astype(float)
    
    #Compute returns for non-rebalancing date
    ticker_wts_names = [ticker + '_nested_weights' for ticker in tickers] 
    
    #Price day returns columns
    price_day_returns = [ticker + '_adj_close_price_day_returns' for ticker in tickers] 
   
    #Adjusted close returns
    adj_close_returns_cols = [ticker + '_adj_close_price_returns' for ticker in tickers]
   
    #Adjusted open price
    adj_open_price_cols = [ticker + '_adj_open_price' for ticker in tickers]
   
    #Adjusted close price
    adj_close_price_cols = [ticker + '_adj_close_price' for ticker in tickers]   
    
    #Store date back into index
    price_weights.index = price_weights['date']       

###########################################Non volatility targeting section################################################           
    
    ##################################################################################################
    #Obtain returns stream of non-volatility target portfolio to be fed into volatility target section
    ##################################################################################################
    for t in range(price_weights.shape[0]): 
       
        #Obtain today's return (Close - Open, not previous close)
        today_return = np.array(1 + price_weights.loc[price_weights.index[t], price_day_returns].astype(float)) 
       
        if t == 0: 

            #At init, assign ticker value
            price_weights.loc[price_weights.index[t], ticker_val] = \
            (price_weights.loc[price_weights.index[t], ticker_wts_names].astype(float).multiply(starting_amount)).tolist()           
           
            price_weights.loc[price_weights.index[t], ticker_val] = \
            np.multiply(np.array(price_weights.loc[price_weights.index[t], ticker_val].astype(float)), \
                    today_return
                    )            
            price_weights['portfolio_value'][t] = sum(price_weights.loc[price_weights.index[t], ticker_val])      
                      
        elif (t > 0):            
    
            #Obtain gap return at start of day. Compute portfolio value before rebalancing          
            open_price_array = np.array(price_weights.loc[price_weights.index[t], adj_open_price_cols].astype(float)) 
            close_price_lag_array = np.array(price_weights.loc[price_weights.index[t-1], adj_close_price_cols].astype(float))
            
            gap_return = 1 + (open_price_array - close_price_lag_array)/close_price_lag_array
            
            price_weights.loc[price_weights.index[t], ticker_val] = \
            np.multiply(np.array(price_weights.loc[price_weights.index[t-1], ticker_val].astype(float)), \
                   gap_return
                   )
    
            price_weights['portfolio_value'][t] = sum(price_weights.loc[price_weights.index[t], ticker_val])             
                 
            #Rebalance at start of the day after accounting for gap return
            #Use weights from previous day
            price_weights.loc[price_weights.index[t], ticker_val] = \
            (price_weights.loc[price_weights.index[t-1], ticker_wts_names].astype(float).multiply(price_weights['portfolio_value'][t])).tolist()            
    
            #Multiply by day's return
            price_weights.loc[price_weights.index[t], ticker_val] = \
            np.multiply(np.array(price_weights.loc[price_weights.index[t], ticker_val].astype(float)), \
                today_return
                )    
           
            #Find portfolio value today
            price_weights['portfolio_value'][t] = sum(price_weights.loc[price_weights.index[t], ticker_val])           
           
    #Compute returns for portfolio
    price_weights['portfolio_return'] = price_weights['portfolio_value'].pct_change(1) 
    
###########################################Volatility targeting section################################################        
    ########################################################################
    #Derive rolling vol target based on non volatility target returns stream
    ########################################################################
    price_weights['square_returns'] = price_weights['portfolio_return'] ** 2
    price_weights['ema_vol'] = price_weights['square_returns'].ewm(span=volatility_targeting_lookback, adjust=False).mean()    
    price_weights['ema_sd'] = (price_weights['ema_vol'] ** 0.5) * (252 ** 0.5) 
        
    #price_weights['ema_sd'] = uf.robust_vol_calc(price_weights['portfolio_return']) * (252 ** 0.5)     
    price_weights['max_allowable_leverage'] =  volatility_targeting/price_weights['ema_sd']
    price_weights['max_allowable_leverage'][0] = price_weights['max_allowable_leverage'][1]    
    price_weights['max_allowable_leverage_cap'] = np.where(price_weights['max_allowable_leverage'] > max_leverage, max_leverage, price_weights['max_allowable_leverage'])
        
    #Fill up first row
    price_weights['ema_sd'][0] = price_weights['ema_sd'][1]
    
    #########################################################
    #Initialize columns to store volatility target parameters
    #########################################################    
    ticker_vol_val = [ticker + '_vol_val' for ticker in tickers]     
    nan_df = pd.DataFrame(0 * np.random.randint(price_weights.shape[0] * len(ticker_vol_val), 
                            size=(price_weights.shape[0], len(ticker_vol_val)) ), 
          columns=ticker_vol_val, index = price_weights.index)
    
    price_weights = pd.concat([price_weights, nan_df], axis = 1)
    price_weights['portfolio_vol_value'] = np.nan; price_weights['portfolio_vol_value'][0] = starting_amount
    price_weights[ticker_vol_val] = price_weights[ticker_vol_val].astype(float)

    #Leveraged value & loan    
    price_weights['portfolio_lev_value'] = np.nan
    price_weights['loan'] = np.nan
    price_weights['actual_leverage'] = np.nan
    price_weights['rebal_vol_target'] = 0
    price_weights['financing_fees'] = 0
    
    total_comm_cost = 0
    
    ##############################################################################
    #Obtain returns streams based on volatility targeting and maximum leverage cap
    ##############################################################################    
    print('Run after volatility targeted portfolio')    
    
    for t in range(price_weights.shape[0]):         
        
        #Allowable leverage based on ytd volatility target
        if(t == 0):        
            lev_allowed = price_weights['max_allowable_leverage_cap'][t]
        elif(t > 0):
            lev_allowed = price_weights['max_allowable_leverage_cap'][t-1]        

        #Obtain today's return (Close - Open, not previous close)         
        today_return = np.array(1 + price_weights.loc[price_weights.index[t], price_day_returns].astype(float))      
        
        if(t == 0):
            
            price_weights['rebal_vol_target'][t] = 1
            
            #Multiply leverage with starting amount
            price_weights.loc[price_weights.index[t], ticker_vol_val] = \
            (price_weights.loc[price_weights.index[t], ticker_wts_names].astype(float).multiply(starting_amount * lev_allowed)).tolist()           
           
            #Multiply with today's return
            price_weights.loc[price_weights.index[t], ticker_vol_val] = \
            np.multiply(np.array(price_weights.loc[price_weights.index[t], ticker_vol_val].astype(float)), \
                    today_return
                    )            
            price_weights['portfolio_lev_value'][t] = sum(price_weights.loc[price_weights.index[t], ticker_vol_val])           
            price_weights['loan'][t] = starting_amount * (lev_allowed - 1)
            price_weights['portfolio_vol_value'][t] = price_weights['portfolio_lev_value'][t] - price_weights['loan'][t]
            price_weights['actual_leverage'][t] = price_weights['portfolio_lev_value'][t]/ price_weights['portfolio_vol_value'][t]
        
        elif (t > 0):  
            
            #Leveraged amount = NLV of previous day * Leverage
            #Does current leveraged amount differ with proposed leveraged amount. If more than x%, buy/sell
            #Obtain gap return at start of day. Compute portfolio value before rebalancing          
            open_price_array = np.array(price_weights.loc[price_weights.index[t], adj_open_price_cols].astype(float)) 
            close_price_lag_array = np.array(price_weights.loc[price_weights.index[t-1], adj_close_price_cols].astype(float))
            
            gap_return = 1 + (open_price_array - close_price_lag_array)/close_price_lag_array
                                                     
            #For each individual ticker asset value -->Asset value * gap return
            price_weights.loc[price_weights.index[t], ticker_vol_val] = \
            np.multiply(np.array(price_weights.loc[price_weights.index[t-1], ticker_vol_val].astype(float)), \
                  gap_return
                   )
    
            price_weights['portfolio_lev_value'][t] = sum(price_weights.loc[price_weights.index[t], ticker_vol_val]) 
            
            #Compute current volatility incorporating opening price/gap return and ensuing leverage
            #Using exponential MA formula
            portfolio_gap_return = (price_weights['portfolio_lev_value'][t] - price_weights['portfolio_lev_value'][t-1])/price_weights['portfolio_vol_value'][t-1]
            portfolio_gap_return_square = portfolio_gap_return ** 2                        
            sd_decay = 2/(volatility_targeting_lookback + 1)
            ema_var_t = (portfolio_gap_return_square * sd_decay) + price_weights['ema_vol'][t-1] * (1 - sd_decay)
            ema_sd_t = (ema_var_t ** 0.5) * (252 ** 0.5) 
            
            #Find maximum leverage allowed
            leverage_t =  volatility_targeting/ema_sd_t   
            max_allowable_leverage_cap = np.where(price_weights['max_allowable_leverage'] > max_leverage, max_leverage, price_weights['max_allowable_leverage'])            
            leverage_gap_cap_t = np.where(leverage_t > max_leverage, max_leverage, leverage_t) + 0
            #print(leverage_t)
            #print(leverage_gap_cap_t)
            
            #Store as variable for comparison later
            before_multiplying_day_ret = price_weights.loc[price_weights.index[t], ticker_vol_val]
                        
            #Check if leverage should be increased or reduced
            leverage_at_open = price_weights['portfolio_lev_value'][t]/ (price_weights['portfolio_lev_value'][t] - price_weights['loan'][t-1]) 
            diff = abs(lev_allowed - leverage_at_open)                                    
            lev_applied_today = leverage_gap_cap_t
            lev_multiplier = lev_applied_today/leverage_at_open

            #Rebalance at start of the day after accounting for gap return
            after_multiplying_day_ret = (price_weights.loc[price_weights.index[t-1], ticker_wts_names].astype(float).multiply(lev_multiplier * price_weights['portfolio_lev_value'][t])).tolist()
            
            #Should tickers rebalance today? Based on difference with optimal position value
            rebal_mask = (abs(after_multiplying_day_ret - before_multiplying_day_ret)/before_multiplying_day_ret) > perc_diff_before_rebalancing
            rebal_values = np.where(rebal_mask, after_multiplying_day_ret, before_multiplying_day_ret)
            
            comm_cost = sum(rebal_mask) * comm_fee
            total_comm_cost = float(total_comm_cost) + float(comm_cost)
            
            #is there rebalancing today?
            if(sum(rebal_mask) > 0):         
                price_weights['rebal_vol_target'][t] = sum(rebal_mask)
            else:
                price_weights['rebal_vol_target'][t] = 0
                                    
            price_weights.loc[price_weights.index[t], ticker_vol_val] = rebal_values  

            #Leverage multiplier adjusted
            lev_applied_adj_today = sum(rebal_values)/(price_weights['portfolio_lev_value'][t] - price_weights['loan'][t-1])
            lev_multiplier_adj = lev_applied_adj_today/leverage_at_open
            
            increase_loan = (lev_multiplier_adj - 1) * price_weights['portfolio_lev_value'][t] + comm_cost
            
            #Multiply by day's return (if difference is x%. Then purchase/ rebalance it)
            price_weights.loc[price_weights.index[t], ticker_vol_val] = \
            np.multiply(np.array(price_weights.loc[price_weights.index[t], ticker_vol_val].astype(float)), \
                today_return
                )    
           
            ###########################
            #Find portfolio value today
            ###########################    
            
            #Do I take full leverage or partial leverage? Take full to hedge against USD effect
            if leverage == 'partial':
                price_weights['financing_fees'][t] = price_weights['loan'][t-1] * financing_fee/365
            elif leverage == 'full':
                price_weights['financing_fees'][t] = price_weights['portfolio_vol_value'][t-1] * financing_fee/365                
            
            price_weights['portfolio_lev_value'][t] = sum(price_weights.loc[price_weights.index[t], ticker_vol_val])              
            price_weights['portfolio_vol_value'][t] = price_weights['portfolio_vol_value'][t-1] + \
                                                      price_weights['portfolio_lev_value'][t] - price_weights['portfolio_lev_value'][t-1] - increase_loan - price_weights['financing_fees'][t]                                                     
            price_weights['loan'][t] = price_weights['portfolio_lev_value'][t] - price_weights['portfolio_vol_value'][t]
            price_weights['actual_leverage'][t] = price_weights['portfolio_lev_value'][t]/ price_weights['portfolio_vol_value'][t]
          
    #Compute returns for portfolio
    price_weights['portfolio_vol_return'] = price_weights['portfolio_vol_value'].pct_change(1)          
    price_weights['square_vol_returns'] = price_weights['portfolio_vol_return'] ** 2
    price_weights['ema_after_vol'] = price_weights['square_vol_returns'].ewm(span=volatility_targeting_lookback, adjust=False).mean()    
    price_weights['ema_after_sd'] = (price_weights['ema_after_vol'] ** 0.5) * (252 ** 0.5)         
    #price_weights['ema_after_sd'] = uf.robust_vol_calc(price_weights['portfolio_vol_value']) * (252 ** 0.5)  
    
    #Print total comm cost
    print('Total comm:' +  str(total_comm_cost))
            
    #Return full info
    return price_weights

if __name__ == "__main__":
    
    #Test risk_parity_weights function
    vol = [0.05,0.05,0.07,0.1,0.15,0.15,0.15,0.18]
    cor = np.array([[100,  80,  60, -20, -10, -20, -20, -20],
                   [ 80, 100,  40, -20, -20, -10, -20, -20],
                   [ 60,  40, 100,  50,  30,  20,  20,  30],
                   [-20, -20,  50, 100,  60,  60,  50,  60],
                   [-10, -20,  30,  60, 100,  90,  70,  70],
                   [-20, -10,  20,  60,  90, 100,  60,  70],
                   [-20, -20,  20,  50,  70,  60, 100,  70],
                   [-20, -20,  30,  60,  70,  70,  70, 100]])/100
    cov = np.outer(vol,vol)*cor      
    rp_wts = risk_parity_weights(cov)

    #Test risk_parity_weights_single_chunk
    tickers = ['TLT', 'IEF', 'GLD', 'SPY']
    start_date = '2000-01-01'
    end_date = '2021-12-31' 
    api = 'yfinance'
    price_input = ut.get_adj_open_close(tickers = tickers, start_date = start_date, end_date = end_date, api = api)    
    adj_close_price_cols = [col for col in price_input.columns if 'adj_close_price' in col]
    price_input = price_input[adj_close_price_cols]
    price_w_input = price_input.resample('W').last().ffill()
    rp_wts_chunk = risk_parity_weights_single_chunk(price_w_input)
    
    #Test rolling_risk_parity_weights
    rp_wts_rolling = rolling_risk_parity_weights(price_w_input, window=52)    
    
    #Test rolling_risk_parity_returns
    price_w_input = rolling_risk_parity_returns (tickers = ['TLT', 'IEF', 'GLD', 'SPY'], 
                                                 start_date = '2015-01-01',
                                                 end_date = '2021-12-31', 
                                                 api = 'yfinance',
                                                 freq = 'W', window = 52, 
                                                 starting_amount = 1000000)
    
    #Test rolling_risk_parity_asset_class_returns
    asset_class_dict = {'asset_class1': ['GLD', 'IAU'],
                        'asset_class2': ['EEM','MCHI','VT','SPY', 'IXN'], 
                        'asset_class3': ['CBON','BNDX','WIP','TIP']}    
    start_date = '2015-01-01'
    end_date = '2021-12-31' 
    api = 'yfinance'
    freq = 'W'
    window = 252 
    window_week = 52
    volatility_targeting = 0.1
    volatility_targeting_lookback = 36
    max_leverage = 2.0         
    perc_diff_before_rebalancing = 0.15  
    comm_fee = 2
    financing_fee = 0.015
    leverage = 'partial'    #leverage: full or partial
    starting_amount = 1000000    
    
    full_df = rolling_risk_parity_asset_class_returns ( asset_class_dict, 
                                                        start_date,
                                                        end_date, 
                                                        api,
                                                        freq, 
                                                        window, 
                                                        window_week, #freq = 'W'
                                                        volatility_targeting,
                                                        volatility_targeting_lookback,
                                                        max_leverage,         
                                                        perc_diff_before_rebalancing,  
                                                        comm_fee,
                                                        financing_fee,
                                                        leverage,    #leverage: full or partial
                                                        starting_amount)    
    
    pass