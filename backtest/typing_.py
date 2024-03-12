from typing import runtime_checkable, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import Series


@runtime_checkable
class Investment(Protocol):

    def get_historic_returns(self, start: str, end: str) -> 'Series':
        pass
