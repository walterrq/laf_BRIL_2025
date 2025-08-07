"""This module offers an implementation example of the poggers code in
order to get the DT data.

An iterator is defined to go through all hd5 files in the 2023 data folder
and compute the PLT rates for fill 8880.

The OrbitMuExtension specifies the per-batch processing steps required for
getting the DT data.

Finally, the results will be saved under the specified output directory.

Run like: python examples/poggers_dt.py
"""

from pathlib import Path
from typing import Any, List, Tuple

import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))
from poggers.iterator import CentralIterator
from poggers.processor.mu_processor import MuProcessorExtension, MuProcessor
from poggers.runner import runner


class DTMuExtension(MuProcessorExtension):
    def process_batch(self, batch: pd.DataFrame, nbx: int, bxmask: np.ndarray) -> pd.DataFrame:
        return batch[["lsnum", "timestampsec", "avgraw"]]

    def build_dataframe(self, buffer: List[pd.DataFrame], nbx: int) -> pd.DataFrame:
        return pd.concat(buffer).rename(
            columns={"avgraw": 0, "timestampsec": "time"}
        ).reset_index(drop=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--central", type=Path, help="Path to central folder.", default=Path("/brildata/23/"))
    parser.add_argument("--beam-central", type=Path, help="Path to beam central folder.", default=Path("/brildata/23/"))
    parser.add_argument("--fill", type=int, help="Fill to process.", default=8880)
    parser.add_argument("--output", type=Path, help="Path to output.", default=Path("poggers_dt_example"))
    args = parser.parse_args()

    iterator = CentralIterator(
        args.central,
        args.beam_central,
        fills = [args.fill]
    )
    processor = MuProcessor(
        DTMuExtension(),
        "/dtlumi",
        output_folder=args.output
    )
    runner(iterator, processor)
