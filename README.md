# backtest
My take on backtesting - simple tool, which can be used to backtest investing in any asset, as long as monthly returns are defined.
Main features:
- Simple yet robust way to define investment instruments
- Simulation of weighted portfolios with rebalancing
- Data from YahooFinance or provided by csv
- Basic visualizations to compare different investing strategies

As an example of usage, take a look at the `main` function in `backtest.py`
```python
hfea = Portfolio(
    [
        (ProductFromCsv('TMFSIM'), 0.5),
        (ProductFromCsv('UPROSIM'), 0.5),
    ],
    rebalance_months=[1, 4, 7, 10],
    name='HFEA',
)
snp_500 = YahooProduct('^GSPC')

backtests = [
    Backtest(x, '1986-01-01', '2024-01-01', 10000) for x in [hfea, snp_500]
]
BacktestComparer(backtests).compare(show_total_invested=False, log_scale=True)
```
![image](https://github.com/mayerja1/backtest/assets/53050153/b64e7ac1-8d60-4080-8e3a-c4ee0cb4e09e)


![meme-stonks-162813497](https://github.com/mayerja1/backtest/assets/53050153/aac36a2b-19d3-40c3-a442-7fb1c8dbe445)
