from typing import List, Callable, Iterable, Optional
from dataclasses import dataclass, field
from queue import Queue

import tables as tb
import pandas as pd

from tqdm import tqdm


@dataclass
class LSAggregator:
    get_iterator: Callable[[int], Iterable]
    columns: List[str]
    lumisections_per_iteration: int = 1
    _next_row: int = field(init=False, default=0)
    _iterator: Iterable = field(init=False, default=None)

    def __post_init__(self):
        self._pbar = None
        self._iterator = iter(self.get_iterator(self._next_row))

    @property
    def pbar(self) -> Optional[tqdm]:
        return self._pbar

    @pbar.setter
    def pbar(self, pbar: Optional[tqdm]):
        self._pbar = pbar

    def __iter__(self):
        return self

    def __next__(self) -> pd.DataFrame:
        buffer = []
        current_ls_count = 0
        current_ls = None

        for row in self._iterator:
            lsnum = row["lsnum"]

            if current_ls is None:
                current_ls = lsnum

            if lsnum != current_ls:
                current_ls_count += 1
                current_ls = lsnum

                if current_ls_count >= self.lumisections_per_iteration:
                    break

            buffer.append(tuple(row.fetch_all_fields()))
            if self.pbar is not None:
                self.pbar.update()

        if not buffer:
            raise StopIteration

        self._next_row = row.nrow + 1

        return pd.DataFrame(buffer, columns=self.columns)

def async_aggregation(aggregator: LSAggregator, queue: Queue) -> None:
    for batch in aggregator:
        queue.put(batch)
    queue.put(None)