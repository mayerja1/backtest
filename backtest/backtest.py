import pickle
from dataclasses import dataclass, field
from typing import Collection, Iterable, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numba import njit

from product import YahooProduct, ProductFromCsv
from typing_ import Investment

sns.set_theme()


@dataclass
class Portfolio:

    # collection of portfolio's investments with their allocations - weights
    investments: Collection[tuple[Investment, float]]

    # portfolio will be rebalanced first trading day of these months (1 = January)
    rebalance_months: list[int] = field(default_factory=list)

    # maximum allowed relative deviation from an investment's desired weight
    # for example for max_relative_imbalance = 0.1, and investment, whose desired
    # weight is 0.4, can take from 36-44% of the total portfolio value
    max_relative_imbalance: float = np.inf

    name: str = None

    def __post_init__(self):

        weights = [w for _, w in self.investments]
        assert np.isclose(sum(weights), 1) and all(w >= 0 for w in weights)

        self.name = self.name or '+'.join(f'{w}x{str(p)}' for p, w in self.investments)

    def get_historic_returns(self, start: str, end: str) -> pd.Series:

        returns = pd.concat(
            (p.get_historic_returns(start, end) for p, _ in self.investments), axis=1
        ).dropna()

        weights = np.array([w for _, w in self.investments])

        rebalance_mask = get_month_starts(
            returns.index, self.rebalance_months, as_mask=True
        )

        individual_positions = self._simulate_position(
            returns.to_numpy(), weights, rebalance_mask, self.max_relative_imbalance
        )
        portfolio_positions = np.sum(individual_positions, axis=1)

        return pd.Series(
            portfolio_positions,
            name=self.name,
            index=returns.index,
        ).pct_change()

    @staticmethod
    @njit
    def _simulate_position(
        investments_returns: np.ndarray,
        weights: np.ndarray,
        rebalance_mask: np.ndarray,
        max_imbalance_rel: float = np.inf,
    ) -> np.ndarray:
        """
        Simulates the portfolio's positions value. Rebalance is triggered either by
        rebalance mask, or by max relative imbalance.
        :param investments_returns: returns per investment in the portfolio
        :param weights: weights of the investments
        :param rebalance_mask: True value triggers portfolio rebalance
        :param max_imbalance_rel: max relative imbalance
        :return:
        """

        position = weights

        position_history = np.zeros((len(investments_returns), len(weights)))
        position_history[0, :] = weights

        for i in range(len(investments_returns)):

            position_history[i, :] = position

            new_position = position * (investments_returns[i, :] + 1)

            if (
                rebalance_mask[i]
                or np.max(np.abs((position / np.sum(position)) - weights) / weights)
                > max_imbalance_rel
            ):
                new_position = weights * np.sum(position)
                assert np.isclose(np.sum(new_position), np.sum(position))

            position = new_position

        return position_history

    def __str__(self) -> str:
        return self.name


class BacktestResult(NamedTuple):

    investment_value: pd.Series
    starting_sum_value: pd.Series
    total_invested: pd.Series

    backtest: 'Backtest'


@dataclass
class Backtest:
    """
    Simulates the bankroll evolution for the given investment.
    Supports lump sum and regular investing.
    """

    investment: Investment

    start: str
    end: str

    starting_sum: float

    # regular investing
    monthly_contribution: float = 0
    yearly_contribution_growth: float = 0
    contribution_growth_month: int = 1

    name: str = None

    def __post_init__(self):
        self.name = self.name or str(self.investment)

    def simulate(self) -> BacktestResult:

        r = self.investment.get_historic_returns(self.start, self.end)
        dt = r.index

        cum_ret = (r + 1).cumprod()
        starting_sum_value = cum_ret * self.starting_sum

        # compute gains from regular investing
        contribution_days = get_month_starts(dt, range(1, 13), as_mask=True)
        contribution_sums = np.where(contribution_days, self.monthly_contribution, 0)

        growth_days = get_month_starts(
            dt, [self.contribution_growth_month], as_mask=True
        )
        growth_coef = np.where(growth_days, 1 + self.yearly_contribution_growth, 1)

        contribution_sums = contribution_sums * np.cumprod(growth_coef)

        aux = contribution_sums / cum_ret
        contribution_value = cum_ret * np.cumsum(aux)

        total_value = starting_sum_value + contribution_value
        return BacktestResult(
            total_value,
            contribution_value,
            pd.Series(np.cumsum(contribution_sums) + self.starting_sum, index=dt),
            self,
        )


@dataclass
class BacktestComparer:

    backtests: list[Backtest]

    def __post_init__(self):

        assert len({b.start for b in self.backtests}) == 1
        assert len({b.end for b in self.backtests}) == 1

    def compare(self, log_scale: bool = False, show_total_invested: bool = True):

        results = [b.simulate() for b in self.backtests]

        fig, ax1 = plt.subplots()

        for result in results:
            ax1.plot(
                result.investment_value, label=f'{result.backtest.name} - total value'
            )
            if show_total_invested:
                ax1.plot(
                    result.total_invested,
                    label=f'{result.backtest.name} - total invested',
                )

        ax1.legend()

        if log_scale:
            ax1.set_yscale('log')

        plt.show()


def get_month_starts(
    dt: pd.DatetimeIndex, months: Iterable[int], as_mask: bool = False
) -> np.ndarray:

    month_mask = dt.month.isin(months)

    idxs = (
        pd.Series(np.arange(len(dt), dtype=int))[month_mask]
        .groupby([dt[month_mask].month, dt[month_mask].year])
        .first()
        .sort_values()
        .to_numpy()
    )

    if as_mask:
        mask = np.full(len(dt), False)
        mask[idxs] = True

        return mask

    return idxs


def compare_daily_returns(a: Investment, b: Investment, start: str, end: str):

    r1 = a.get_historic_returns(start, end)
    r2 = b.get_historic_returns(start, end)

    r1.index = r1.index.tz_localize(None)
    r2.index = r2.index.tz_localize(None)

    df = pd.concat([r1, r2], axis=1)

    sns.regplot(data=df, x=r1.name, y=r2.name)
    plt.show()


def main():
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


if __name__ == '__main__':
    main()
