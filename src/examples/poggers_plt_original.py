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
import json
import argparse

from numba import njit, prange

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))
from poggers.iterator import CentralIterator
from poggers.processor.mu_processor import MuProcessorExtension, MuProcessor
from poggers.models import sub_nl
from poggers.runner import runner

@njit(parallel=True)
def numba_process_batch(data: np.ndarray, nbx: int, bxmask: np.ndarray) -> np.ndarray:
    n_rows = data.shape[0]
    
    out = np.empty(n_rows)
    for i in prange(n_rows):
        idata = data[i]
        
        data_clipped = np.minimum(idata, 2**14)
        bxraw = -np.log(data_clipped / (2**14))
        bxraw = np.where(np.isnan(bxraw) | np.isinf(bxraw), 0.0, bxraw)
        
        avgraw = np.sum(bxraw * bxmask)
        out[i] = avgraw

    return out

class PLTAggExtension(MuProcessorExtension):
    def __init__(self, svs: Dict[int, float], effs: Dict[int, float], lins: Dict[int, float]):
        self.svs = svs
        self.effs = effs
        self.lins = lins

    def process_batch(self, batch: pd.DataFrame, nbx: int, bxmask: np.ndarray) -> pd.DataFrame:
        data = np.stack(batch.data)
        batch["avgraw"] = numba_process_batch(data, nbx, bxmask)
        return batch[["lsnum", "nbnum", "timestampsec", "channelid", "avgraw"]]

    def build_dataframe(self, buffer: List[pd.DataFrame], nbx: int) -> pd.DataFrame:
        df = pd.concat(buffer).rename(columns={
                "timestampsec": "time",
                "channelid": "chid",
        }).pivot_table(
            index=["lsnum", "nbnum"], values=["avgraw", "time"], columns=["chid"]
        )
        avg_time = df["time"].mean(axis=1).reset_index(drop=True)
        df = df.drop(columns=["time"], level=0)
        df.columns = df.columns.droplevel(0).rename(None)
        df = df.reset_index()
        df.insert(2, "time", avg_time)
        df = self._apply_lin(df, nbx)
        df = self._apply_svs(df)
        return df

    def _apply_lin(self, df: pd.DataFrame, nbx: int) -> pd.DataFrame:
        channels = list(self.lins.keys())
        lins = np.array(list(self.lins.values()))
        df[channels] = sub_nl(df[channels].T, lins[:, np.newaxis], nbx).T
        return df

    def _apply_svs(self, df: pd.DataFrame) -> pd.DataFrame:
        channels = list(self.svs.keys())
        svs = np.array(list(self.svs.values()))
        effs = np.array(list(self.effs.values()))
        df[channels] = df[channels] * 11245.5 / (svs * effs)
        return df


def load_corrections(fill: int, path: Path) -> Tuple[Dict[int, float], Dict[int, float]]:
    with open(path, "r") as f:
        corrections: Dict[str, Dict[str, float]] = json.load(f)[str(fill)]
    channels = np.array(list(map(int, corrections["eff"].keys())))
    effs = np.array([corrections["eff"][str(ch)] for ch in channels])
    lins = np.array([corrections["lin"][str(ch)] for ch in channels])
    lafs = np.array([corrections["laf"][str(ch)] for ch in channels])
    bad_channels_mask = np.isnan(effs) | np.isnan(lins) | (lafs == False)
    channels = channels[~bad_channels_mask]
    effs = effs[~bad_channels_mask]
    lins = lins[~bad_channels_mask]
    return dict(zip(channels.tolist(), effs)), dict(zip(channels.tolist(), lins))

def load_calibrations(path: Path) -> Dict[int, float]:
    with open(path, "r") as f:
        calibrations: Dict[str, float] = json.load(f)
    channels = np.array(list(map(int, calibrations.keys())))
    sigmas = np.array([calibrations[str(ch)] for ch in channels])
    bad_channels_mask = np.isnan(sigmas)
    channels = channels[~bad_channels_mask]
    sigmas = sigmas[~bad_channels_mask]
    return dict(zip(channels, sigmas))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fill", type=int, help="Fill to process.", default=8880)
    parser.add_argument("--central", type=Path, help="Path to central folder.", default=Path("/brildata/23/"))
    parser.add_argument("--beam-central", type=Path, help="Path to beam central folder.", default=Path("/brildata/23/"))
    parser.add_argument("--output", type=Path, help="Path to output.", default=Path("poggers_plt_example"))
    parser.add_argument("--corrections", type=Path, help="Path to per json channel alphas.", default=None)
    parser.add_argument("--calibrations", type=Path, help="Path to per json channel sigmas.", default=None)
    args = parser.parse_args()

    effs, lins = load_corrections(args.fill, args.corrections) if args.corrections else ({}, {})
    calibs = load_calibrations(args.calibrations) if args.calibrations else {}

    # Only keep the channels that are in both corrections and calibrations
    channels = set(effs.keys()) & set(calibs.keys())
    effs = {ch: effs[ch] for ch in channels}
    lins = {ch: lins[ch] for ch in channels}
    svs = {ch: calibs[ch] for ch in channels}

    all_results = []  # <- aquí se almacenarán todos los DataFrames

    
    iterator = CentralIterator(
        args.central,
        args.beam_central,
        fills = [args.fill]
    )
    processor = MuProcessor(
        PLTAggExtension(svs, effs, lins),
        "/pltaggzero",
        output_folder=args.output,
        all_results=all_results  # <- nuevo argumento
    )
    runner(iterator, processor)

    # Combinar todos los resultados en un único DataFrame
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        print(final_df)  # print df
        final_df.to_csv(args.output / "merged_results.csv", index=False)
        # final_df.to_csv("merged_results.csv", index=False)  # opcional
        