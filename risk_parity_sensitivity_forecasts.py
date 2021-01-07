#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:36:38 2021

@author: jirong
"""

import util as ut
import risk_parity_class as risk		  	
import pandas as pd
import numpy as np
import pyfolio	 			     			  	   		   	  			  	
import seaborn as sns 	
import matplotlib.pyplot as plt	
import matplotlib
import re
import warnings
import yaml
	  		 			     			  	   		   	  			  	  	   		   	  			  	
class risk_parity_sensitivity_forecast(object): 		  		 			     			  	   		   	  			  	
 			  		 			     			  	   		   	  			  	
    # constructor 			  		 			     			  	   		   	  			  	
    def __init__(self, asset_class_dict, 
                 start_date, end_date,
                 api, freq, window, window_week,
                 volatility_targeting, volatility_targeting_lookback,
                 max_leverage, perc_diff_before_rebalancing,
                 comm_fee, financing_fee, leverage,
                 starting_amount,
                 volatility_targets, max_leverage_targets
                 ): 
        
        """
        Constructor for risk_parity_sensitivity_forecast (used to create forecasts for deployment and sensitivity analysis)
    
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
        :param volatility_targets: list of volatility targets
        :param max_leverage_targets: list of max leverage cap
        :return: returns risk_parity_sensitivity_forecast object
        """         
			  		 			     			  	   		   	  			  				  		 			     			  	   		   	  			  	
        self.asset_class_dict = asset_class_dict
        self.start_date = start_date
        self.end_date = end_date
        self.api = api
        self.freq = freq
        self.window = window
        self.window_week = window_week
        self.volatility_targeting = volatility_targeting
        self.volatility_targeting_lookback = volatility_targeting_lookback
        self.max_leverage = max_leverage         
        self.perc_diff_before_rebalancing = perc_diff_before_rebalancing
        self.comm_fee = comm_fee
        self.financing_fee = financing_fee
        self.leverage = leverage,    #leverage: full or partial
        self.starting_amount = starting_amount
        self.df = None
        self.vol_target_scenarios = volatility_targets
        self.leverage_scenarios = max_leverage_targets
        self.sharpe_grid = np.zeros((len(volatility_targets),len(max_leverage_targets)))
        self.sortino_grid = np.zeros((len(volatility_targets),len(max_leverage_targets)))
        self.max_drawdown_grid = np.zeros((len(volatility_targets),len(max_leverage_targets)))
        self.return_grid = np.zeros((len(volatility_targets),len(max_leverage_targets)))
        self.skewness_grid = np.zeros((len(volatility_targets),len(max_leverage_targets)))
        self.kurtosis_grid = np.zeros((len(volatility_targets),len(max_leverage_targets)))
    		  		 			     			  	   		   	  			  	
    # this method should use the existing policy and test it against new data 			  		 			     			  	   		   	  			  	
    def get_risk_parity_df(self, vol_target, leverage_num):
                   
        """
        Return rolling risk parity weights
    
        :param vol_target: Volatility target used in iteration
        :param leverage_num: Leverage ratio used in iteration
        :return: returns concatenated risk parity weights
        """                    
        comm_fee = self.comm_fee        
        
        self.df = risk.rolling_risk_parity_asset_class_returns (self.asset_class_dict, 
                                                                self.start_date,
                                                                self.end_date, 
                                                                self.api,
                                                                self.freq, 
                                                                self.window, 
                                                                self.window_week, #freq = 'W'
                                                                vol_target,  #ok
                                                                self.volatility_targeting_lookback,
                                                                leverage_num, #ok         
                                                                self.perc_diff_before_rebalancing,  
                                                                self.comm_fee,
                                                                self.financing_fee,
                                                                self.leverage,    #leverage: full or partial
                                                                self.starting_amount)           
                                        
        max_drawdown = ut.get_max_drawdown(self.df['portfolio_vol_return'].iloc[1:])
        sharpe = ut.get_sharpe(self.df['portfolio_vol_return'].iloc[1:])
        sortino = ut.get_sortino(self.df['portfolio_vol_return'].iloc[1:])
        annual_return = ut.get_compound_returns(self.df['portfolio_vol_return'].iloc[1:])
        skewness_char = ut.get_skewness(self.df['portfolio_vol_return'].iloc[1:])
        kurtosis_char = ut.get_kurtosis(self.df['portfolio_vol_return'].iloc[1:])
        
        print(max_drawdown, sharpe, sortino, annual_return, skewness_char, kurtosis_char)
        
        return max_drawdown, sharpe, sortino, annual_return, skewness_char, kurtosis_char
    
    def sensitivity_analysis(self):
        
        """
        Sensitivity analysis
        """           
            
        for i, vt in enumerate(self.vol_target_scenarios):
            for j, lev in enumerate(self.leverage_scenarios):
                                
                perf_stats = self.get_risk_parity_df(vt, lev)
                
                print('Volatility target: ' + str(vt) + ', Leverage: ' + str(lev))
                print(perf_stats)

                self.max_drawdown_grid[i][j] = perf_stats[0]                
                self.sharpe_grid[i][j] = perf_stats[1]
                self.sortino_grid[i][j] = perf_stats[2]                
                self.return_grid[i][j] = perf_stats[3]
                self.skewness_grid[i][j] = perf_stats[4]
                self.kurtosis_grid[i][j] = perf_stats[5]
        pass
    
    def plot_grid(self, perf_stat_name):
        
        """
        Plot sensitivity analysis grid
        
        :param perf_stat_name: Performance statistic name
        :return: returns concatenated risk parity weights        
        """           
        
        if(perf_stat_name == "sharpe"):
            grid = self.sharpe_grid
        if(perf_stat_name == "sortino"):
            grid = self.sortino_grid            
        elif(perf_stat_name == "drawdown"):
            grid = self.max_drawdown_grid
        elif(perf_stat_name == "return"):
            grid = self.return_grid
        elif(perf_stat_name == "skewness"):
            grid = self.skewness_grid
        elif(perf_stat_name == "kurtosis"):
            grid = self.kurtosis_grid            
        
        matplotlib.rc('figure', figsize=(10, 7))
        
        fig, ax = plt.subplots()
        sns.heatmap(grid, annot=True, cmap="RdYlGn")
        
        ax.set_xticks(np.arange(len(self.leverage_scenarios)))
        ax.set_yticks(np.arange(len(self.vol_target_scenarios)))
        
        # Label using the threshold ranges
        ax.set_xticklabels([round(x,4) for x in self.leverage_scenarios])
        ax.set_yticklabels([round(x,4) for x in self.vol_target_scenarios])
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
                    
        ax.set_title("Volatility target vs Max Leverage Grid (" + perf_stat_name + ")")
        fig.tight_layout()
        plt.show()        
        
        pass
    
    
    #Read from configuration file instead
    def rp_forecast(self, vol, lev):
        
        """
        Return risk parity forecast parameter used for deployment (not required in research jupyter notebook)
        
        :param vol: Volatility target
        :param lev: Max leverage        
        :return: returns full dataframe, weights to tickers and recommended leverage       
        """         
        
        warnings.filterwarnings("ignore")
        
        with open('config.yml') as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)            
        
        self.get_risk_parity_df(vol, lev) 
        forecasts = self.df
        
        #Get recommended leverage
        rec_lev = forecasts[configs['risk_parity']['max_leverage_column_name']].iloc[-1:,][0]
        column_names = ["leverage"]
        leverage = pd.DataFrame(columns = column_names)
        leverage = leverage.append({'leverage': rec_lev}, ignore_index=True)         

        #Get proportion of allocation (from yaml file)
        rec_weights_names = self.asset_class_dict['asset_class1'] + self.asset_class_dict['asset_class2'] + self.asset_class_dict['asset_class3']	             
        rec_weights_suffix = [ticker + configs['risk_parity']['weights_suffix'] for ticker in rec_weights_names] 

        #rec_weights = sum(forecasts.loc[forecasts.index[-1], rec_weights_suffix])
        rec_weights = forecasts.loc[forecasts.index[-2], rec_weights_suffix]
        rec_weights = rec_weights.to_frame()
        rec_weights['Ticker'] = rec_weights.index
        rec_weights['Ticker'] = [re.sub("_nested_weights", "", x) for x in rec_weights['Ticker']]
        rec_weights.index = rec_weights['Ticker']
        
        #Combine IAU and GLD. Delete IAU
        rec_weights.rename(columns={ rec_weights.columns[0]: "weights" }, inplace = True)                
        rec_weights.at['GLD', 'weights'] = rec_weights.at['GLD', 'weights'] + rec_weights.at['IAU', 'weights']
        rec_weights = rec_weights[rec_weights['Ticker'] != 'IAU']

        #Change CBON to CNYB
        rec_weights.at['CBON', 'Ticker'] = 'CNYB'   

        return {'forecasts': forecasts, 'weights': rec_weights, 'leverage': leverage}
    
if __name__=="__main__":

#########################Daily correlation matrix######################################           
    asset_class_dict =  {'asset_class1': ['TIP', 'WIP', 'CBON', 'BNDX'],
                         'asset_class2': ['EEM','MCHI','VT','SPY', 'IXN', 'KWEB'], 
                         'asset_class3': ['GLD','IAU']}       
        
    start_date = '2015-01-01'
    end_date = '2050-12-31'
    api = 'yfinance'
    freq = 'D'
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
    volatility_targets = np.arange(0.1,0.28,0.02)
    max_leverage_targets = np.arange(1, 2.1, 0.2) 
        
    strategy = risk_parity_sensitivity_forecast(asset_class_dict, 
                                                start_date, end_date,
                                                api, freq, window, window_week,
                                                volatility_targeting, volatility_targeting_lookback,
                                                max_leverage, perc_diff_before_rebalancing,
                                                comm_fee, financing_fee, leverage,
                                                starting_amount,
                                                volatility_targets, max_leverage_targets
                                                )	    

    strategy.get_risk_parity_df(0.1, 2.0) 
    results = strategy.df
        
    portfolio_return1 = pd.Series(results['portfolio_vol_return'], index = results.index)
    portfolio_return2 = pd.Series(results['portfolio_return'], index = results.index)
    benchmark_return = pd.Series(results['VT_adj_close_price_returns'], index = results.index)
    
    portfolio_return1.index = portfolio_return1.index.tz_localize('US/Eastern')
    portfolio_return2.index = portfolio_return2.index.tz_localize('US/Eastern')
    benchmark_return.index = benchmark_return.index.tz_localize('US/Eastern')
    
    pyfolio.create_full_tear_sheet(portfolio_return1, benchmark_rets=benchmark_return, live_start_date = '2020-08-21') 
    
    strategy.sensitivity_analysis()
    strategy.plot_grid('return')
    strategy.plot_grid('sharpe')
    strategy.plot_grid('sortino')    
    strategy.plot_grid('drawdown')
    strategy.plot_grid('skewness')
    strategy.plot_grid('kurtosis')    
    
#########################Weekly correlation matrix######################################    
    start_date = '2015-01-01'
    end_date = '2050-12-31'
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
    volatility_targets = np.arange(0.1,0.28,0.02)
    max_leverage_targets = np.arange(1, 2.1, 0.2) 
        
    strategy = risk_parity_sensitivity_forecast(asset_class_dict, 
                                                start_date, end_date,
                                                api, freq, window, window_week,
                                                volatility_targeting, volatility_targeting_lookback,
                                                max_leverage, perc_diff_before_rebalancing,
                                                comm_fee, financing_fee, leverage,
                                                starting_amount,
                                                volatility_targets, max_leverage_targets
                                                )	    

    strategy.get_risk_parity_df(0.1, 2.0) 
    results = strategy.df
        
    portfolio_return1 = pd.Series(results['portfolio_vol_return'], index = results.index)
    portfolio_return2 = pd.Series(results['portfolio_return'], index = results.index)
    benchmark_return = pd.Series(results['VT_adj_close_price_returns'], index = results.index)
    
    portfolio_return1.index = portfolio_return1.index.tz_localize('US/Eastern')
    portfolio_return2.index = portfolio_return2.index.tz_localize('US/Eastern')
    benchmark_return.index = benchmark_return.index.tz_localize('US/Eastern')
    
    pyfolio.create_full_tear_sheet(portfolio_return1, benchmark_rets=benchmark_return, live_start_date = '2020-08-21') 
    
    strategy.sensitivity_analysis()
    strategy.plot_grid('return')
    strategy.plot_grid('sharpe')
    strategy.plot_grid('sortino')    
    strategy.plot_grid('drawdown')
    strategy.plot_grid('skewness')
    strategy.plot_grid('kurtosis')  

#########################Weekly correlation matrix + alt groupings######################################  
    asset_class_dict =  {'asset_class1': ['CBON','BNDX','IGOV'],
                        'asset_class2': ['EEM','MCHI','VT','SPY', 'IXN', 'KWEB'], 
                        'asset_class3': ['GLD','IAU', 'TIP', 'WIP']}
    
    start_date = '2015-01-01'
    end_date = '2050-12-31'
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
    volatility_targets = np.arange(0.1,0.28,0.02)
    max_leverage_targets = np.arange(1, 2.1, 0.2) 
        
    strategy = risk_parity_sensitivity_forecast(asset_class_dict, 
                                                start_date, end_date,
                                                api, freq, window, window_week,
                                                volatility_targeting, volatility_targeting_lookback,
                                                max_leverage, perc_diff_before_rebalancing,
                                                comm_fee, financing_fee, leverage,
                                                starting_amount,
                                                volatility_targets, max_leverage_targets
                                                )	    

    strategy.get_risk_parity_df(0.1, 2.0) 
    results = strategy.df
    
    #Get forecasts
    forecasts = strategy.rp_forecast(0.1, 2.0)    
        
    portfolio_return1 = pd.Series(results['portfolio_vol_return'], index = results.index)
    portfolio_return2 = pd.Series(results['portfolio_return'], index = results.index)
    benchmark_return = pd.Series(results['VT_adj_close_price_returns'], index = results.index)
    
    portfolio_return1.index = portfolio_return1.index.tz_localize('US/Eastern')
    portfolio_return2.index = portfolio_return2.index.tz_localize('US/Eastern')
    benchmark_return.index = benchmark_return.index.tz_localize('US/Eastern')
    
    pyfolio.create_full_tear_sheet(portfolio_return1, benchmark_rets=benchmark_return, live_start_date = '2020-08-21') 
    
    strategy.sensitivity_analysis()
    strategy.plot_grid('return')
    strategy.plot_grid('sharpe')
    strategy.plot_grid('sortino')    
    strategy.plot_grid('drawdown')
    strategy.plot_grid('skewness')
    strategy.plot_grid('kurtosis') 
   