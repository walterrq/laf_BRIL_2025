from typing import Any, Tuple, Dict, List
from pathlib import Path

import pickle
import numpy as np
import pandas as pd

from ._utils import filter_scan_timestamps, get_scan_timestamps


def read_fill(
    path: Path,
    fill: int,
    name: str,
    agg_per_ls: bool = False,
    perform_ls_query: bool = False,
    remove_scans: bool = False,
    index_filter: Tuple[int, int] = (0, 1)
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    files = path.glob(f"{fill}*.pickle")

    dfs: List[pd.DataFrame] = []
    for file in files:
        run = int(file.stem.split("_")[1])
        with open(file, "rb") as fp:
            data: Tuple[pd.DataFrame, Dict[str, Any]] = pickle.load(fp)
            df, attrs = data

        if perform_ls_query:
            df = df.query(attrs["ls_mask"])

        df.insert(0, "run", run)
        dfs.append(df)

    if not dfs:
        raise Exception(f"No data found in {path} for fill {fill}.")

    result = pd.concat(dfs).sort_values(by=["run", "lsnum"]).reset_index(drop=True)
    attrs = {
        "fill": fill,
        "name": name,
        "nbx": attrs["nbx"],
        "bxmask": attrs["bxmask"],
    }
    
    if remove_scans:
        result = filter_scan_timestamps(result, get_scan_timestamps(fill))

    if index_filter is None:
        index_filter = (0, 1)
    start, stop = index_filter
    good_index = result.iloc[int(len(result)*start):int(len(result)*stop)].index
    filter_mask = result.index.isin(good_index)       
    result: pd.DataFrame = result.iloc[filter_mask]

    if agg_per_ls:
        result = result.groupby(["run", "lsnum"]).mean().reset_index()

    return attrs, result
    