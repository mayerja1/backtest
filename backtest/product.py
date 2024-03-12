import warnings
from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf


@dataclass()
class YahooProduct:
    """Gets price data from Yahoo finance."""

    symbol: str

    _yf_ticker: yf.Ticker = field(init=False)

    def __post_init__(self):
        self._yf_ticker = yf.Ticker(self.symbol)

    def get_price_history(self, start: str, end: str) -> pd.Series:
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            return self._yf_ticker.history(interval='1d', start=start, end=end)[
                'Close'
            ].rename(self.symbol)

    def get_historic_returns(self, start: str, end: str) -> pd.Series:

        price_history = self.get_price_history(start, end)

        return price_history.pct_change().fillna(0)

    def __str__(self) -> str:
        return self.symbol


@dataclass
class DailyLeveragedYahooProduct(YahooProduct):
    """Simulated daily reset leverage."""

    leverage: float

    def get_historic_returns(self, start: str, end: str) -> pd.Series:
        # very simplified computation - just multiply the daily returns
        return (super().get_historic_returns(start, end) * self.leverage).rename(
            str(self)
        )

    def __str__(self) -> str:
        return f'{self.leverage}x{self.symbol}'


@dataclass
class ProductFromCsv:
    """Gets daily returns from csv on disk in `resources` folder."""

    name: str

    date_format: str = '%m/%d/%y'

    def get_historic_returns(self, start: str, end: str) -> pd.Series:

        df = pd.read_csv(f'resources/{self.name}.csv').set_index('Date')
        [col] = df.columns

        returns = df[col].copy()
        returns.index = pd.to_datetime(returns.index, format=self.date_format)
        # remove % sign and convert to decimals
        returns = returns.str.rstrip('%').astype(float) / 100

        return returns.loc[start:end].copy()

    def __str__(self) -> str:
        return self.name
