# Risk parity strategy

## Disclaimer

None of the contents constitute an offer to sell, a solicitation to buy, or a recommendation or endorsement for any security or strategy, nor does it constitute an offer to provide investment advisory services. Past performance is no indicator of future performance. Provided for informational purposes only. All investments involve risk, including loss of principal.

Note:

I have a vested interest in this strategy. In fact, this forms my core portfolio.
Variant of the strategy has been deployed since start of 8-Sep-2020 using my infrastructure here (https://medium.com/datadriveninvestor/designing-and-building-a-fully-automated-algorithmic-trading-portfolio-management-system-6945c6c87620)
Out-of-sample live results could be found in following link: https://jironghuang.github.io/portfolio/portfolio/

## Summary

In the 3 different configurations of risk parity portfolio that I explored (daily correlation matrix, weekly correlation matrix, weekly correlation matrix of alternate asset groupings),the performance characteristics are similar with sharpe between 1.9 to 2.0, annualized returns of 0.16-0.17, calmar ratio of 1.3 to 1.5, sortino of 2.5 to 2.7.

Note: Above corresponds to 10% volatility target and max leverage cap of 2.0. In my simulation, I find that this corresponds to the best sharpe, sortino, least negative skew and kurtosis.

I do not expect such outstanding performance to persist for decades. Based on the literature, sharpe is likely to be closer to 0.8 to 1.0.

## Motivation

This strategy is explored in a different way from other strategies I developed. Instead of carrying out robust statistical techniques such as bootstrapping, block-boostrapping, monte carlo and cross validation to gauge the viability of the strategy, the construction of portfolio is heavily influenced by the literature, white papers and books of hedge funds (e.g. AQR and Bridgewater) and reputable quants. In addition, the data history for my ETFs are pretty limited (dated till 2015); hence carrying out any form of robust statistical test is likened to data snooping.

In this study, I seek to create an all-weather low maintenance portfolio that could survive all economic regimes in the short and long term. Risky parity all weather portfolio has been studied extensively; and in numerous studies (see References), it was shown to perform better than a 60% equities - 40% bonds portfolio.

## Dataset

- Extracted from yfinance package

## How to use the repository

I developed 2 classes,

- risk_parity_class.py: Functions to generate risk parity weights on rolling basis. Also contained an event-driven backtesting function to contain all the constraints.
- risk_parity_sensitivity_forecasts.py: To carry out sensitivity analysis vis-a-vis max leverage cap and volatility target. 

Note: Second class is used to generate signals in my live deployment. But if you wish to use it, pls include further engineering safeguards which I set up in a broader framework.

