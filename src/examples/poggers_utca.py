"""This module offers an implementation example of the poggers code in
order to perform the PLT zero counting and save it to the necessary pickle
file format.

An iterator is defined to go through all hd5 files in the 2023 data folder
and compute the PLT rates for fill 8880.

The PLTAggExtension specifies the per-batch processing steps required for applying
the zero counting algorithm.

Finally, the results will be saved under the specified output directory.

Run like: python examples/poggers_plt.py
"""

from pathlib import Path
from typing import *

import os
import sys
import argparse

from numba import njit, prange

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))
from poggers.iterator import CentralIterator
from poggers.processor.mu_processor import MuProcessorExtension, MuProcessor
from poggers.runner import runner

@njit(parallel=True)
def numba_process_batch(data: np.ndarray, nbx: int, bxmask: np.ndarray) -> np.ndarray:
    n_rows = data.shape[0]
    
    out = np.empty(n_rows)
    for i in prange(n_rows):
        idata = data[i]
        
        data_clipped = np.minimum(idata, 2**14)
        bxraw = -np.log(1 - data_clipped / (2**14))
        bxraw = np.where(np.isnan(bxraw) | np.isinf(bxraw), 0.0, bxraw)
        
        avgraw = np.sum(bxraw * bxmask)
        out[i] = avgraw

    return out

class UTCAAggExtension(MuProcessorExtension):
    def process_batch(self, batch: pd.DataFrame, nbx: int, bxmask: np.ndarray) -> pd.DataFrame:
        data = np.stack(batch.agghist)
        batch["avgraw"] = numba_process_batch(data, nbx, bxmask)
        return batch[["lsnum", "timestampsec", "channelid", "avgraw"]]

    def build_dataframe(self, buffer: List[pd.DataFrame], nbx: int) -> pd.DataFrame:
        df = pd.concat(buffer).rename(columns={
                "timestampsec": "time",
                "channelid": "chid",
        }).pivot_table(
            index=["lsnum", "time"], values=["avgraw"], columns=["chid"]
        )
        df.columns = df.columns.droplevel(0).rename(None)
        return df.reset_index()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--central", type=Path, help="Path to central folder.", default=Path("/brildata/23/"))
    parser.add_argument("--beam-central", type=Path, help="Path to beam central folder.", default=Path("/brildata/23/"))
    parser.add_argument("--fill", type=int, help="Fill to process.", default=8880)
    parser.add_argument("--output", type=Path, help="Path to output.", default=Path("poggers_utca_example"))
    args = parser.parse_args()


    iterator = CentralIterator(
        args.central,
        args.beam_central,
        fills = [args.fill]
    )
    processor = MuProcessor(
        UTCAAggExtension(),
        "/bcm1futca_agg_hist",
        output_folder=args.output
    )
    runner(iterator, processor)
