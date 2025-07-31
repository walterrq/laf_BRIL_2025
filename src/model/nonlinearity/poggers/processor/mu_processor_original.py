from typing import Any, List, Tuple, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Thread
from pathlib import Path
from queue import Queue

from tqdm import tqdm
from tables.exceptions import NoSuchNodeError

import pickle
import numpy as np
import pandas as pd
import tables as tb

from ..iterator import IterationContext
from .processor import HD5Processor
from ._aggregator import LSAggregator, async_aggregation


class MuProcessorExtension(ABC):
    @abstractmethod
    def process_batch(self, row, nbx: int, bxmask: np.ndarray) -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def build_dataframe(self, buffer: List[Tuple[Any, ...]], nbx: int) -> pd.DataFrame:
        pass

@dataclass
class MuProcessor(HD5Processor):
    extension: MuProcessorExtension
    node_path: str
    output_folder: Path
    beam_path: str = "/beam"
    

    def start(self):
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def process_iteration(self, ctx: IterationContext):
        output_file_path = self.output_folder / f"{ctx.fill}_{ctx.run}.pickle" #Only sets the name of the output.
        if output_file_path.exists():
            print(f"Fill: {ctx.fill} Run: {ctx.run} already processed. Skipping.")
            return

        try:
            c_handle: tb.Table = ctx.c_handle.get_node(self.node_path)
            b_handle: tb.Table = ctx.b_handle.get_node(self.beam_path)
        except NoSuchNodeError as e:
            print(f"HD5 ERROR: {e}.")
            return

        if c_handle.nrows == 0 or b_handle.nrows == 0:
            print(f"Fill: {ctx.fill} Run: {ctx.run} no data found. Skipping.")
            return

        aggregator = LSAggregator(lambda x: c_handle.iterrows(start=x), c_handle.colnames, 100)
        
        ls_query = self._get_ls_query(ctx.iov)
        nbx, bxmask = self._get_nbx_bxmask(b_handle, ls_query, ctx)
        
        buffer = []
        with tqdm(total=c_handle.nrows, desc=f"Processing rows", leave=False) as pbar:
            aggregator.pbar = pbar
            queue = Queue(maxsize=5)
            thread = Thread(target=async_aggregation, args=(aggregator, queue))
            thread.start()
            
            batch = queue.get()
            while batch is not None:
                result = self.extension.process_batch(batch, nbx, bxmask)
                buffer.append(result)
                batch = queue.get()
            thread.join()

        if buffer:
            result = (
                self.extension.build_dataframe(buffer, nbx),
                {
                   "nbx": nbx,
                   "bxmask": bxmask,
                   "node": self.node_path,
                   "tag": ctx.tag,
                   "ls_mask": ls_query
                }
            )

            with open(self.output_folder / f"{ctx.fill}_{ctx.run}.pickle", "wb") as fp:
                pickle.dump(result, fp)
        else:
            print(f"Fill: {ctx.fill} Run: {ctx.run} no data found for '{self.node_path}'.")

    def end(self):
        pass

    @staticmethod
    def _get_nbx_bxmask(beam: tb.Table, ls_query: str, ctx: IterationContext) -> Tuple[int, np.ndarray]:
        i = -1
        for i, row in enumerate(beam.where(ls_query)):
            if i == 20:
                return row["ncollidable"], row["collidable"]
        if i > -1:
            # we reach this if the table does not have at least 20 rows
            return row["ncollidable"], row["collidable"]
        else:
            # we reach this if the table does have a single row in the 'ls_query'
            # However it does have some rows of data since we reached this point
            row = next(beam.iterrows())
            print(f"WARNING: Fill: {ctx.fill} Run: {ctx.run} has few beam entries ({beam.nrows}). Bunch mask needs checking.")
            return row["ncollidable"], row["collidable"]

    @staticmethod
    def _get_ls_query(iov: List[Tuple[int, int]]) -> str:
        queries = []
        for start, end in iov:
            queries.append(f"((lsnum >= {start}) & (lsnum <= {end}))")
        return " | ".join(queries)
