from typing import Any, Dict, List, Tuple, Optional, Container
from pathlib import Path
from dataclasses import dataclass, field

import re
import json

import tables as tb
import pandas as pd


NormtagType = List[Tuple[str, Dict[str, List[Tuple[int, int]]]]]


@dataclass
class IterationContext:
    fill: int
    run: int
    tag: str
    iov: List[Tuple[int, int]]
    c_handle: tb.File = field(repr=False)
    b_handle: tb.File = field(repr=False)

@dataclass
class CentralIterator:
    central: Path
    beam_central: Path
    normtag_path: Optional[Path] = None
    fills: Optional[Container[int]] = None
    runs: Optional[Container[int]] = None

    def __post_init__(self):
        if not (
            self.normtag_path or
            self.fills or self.runs
        ):
            raise ValueError("A normtag, a range of fills or a range of runs must be specified.")

        self.current_path_index = -1
        self.c_handle: Optional[tb.File] = None
        self.b_handle: Optional[tb.File] = None
        self.paths = self._compute_in_range_paths()

    @property
    def total_iterations(self) -> int:
        return len(self.paths)

    @staticmethod
    def _parse_file_path(file_path: Path) -> Tuple[int, int]:
        mtch = re.match(r"(\d+)_(\d+)_.*", file_path.name)
        if mtch is None:
            raise ValueError(f"Could not extract run and fill from {file_path}")
        return int(mtch.group(1)), int(mtch.group(2))

    @staticmethod
    def _read_normtag(path: Path) -> pd.DataFrame:
        with open(path) as fp:
            content: NormtagType = json.load(fp)
        iov_dict: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}
        for tag, iov in content:
            for key, value_list in iov.items():
                run = int(key)
                if (tag, run) not in iov_dict:
                    iov_dict[(tag, run)] = []
                iov_dict[(tag, run)].extend(value_list)
        flattened_data = [
            {'tag': tag, 'run': run, 'iov': iov}
            for (tag, run), iov in iov_dict.items()
        ]
        normtag = pd.DataFrame(flattened_data)
        normtag["tag"] = normtag["tag"].astype("category")
        return normtag.set_index("run").sort_index()

    def _compute_in_range_paths(self) -> List[Dict[str, Any]]:
        normtag: Optional[pd.DataFrame] = self._read_normtag(self.normtag_path) if self.normtag_path else None

        def _in_requested_range(fill: int, run: int) -> bool:
            in_normtag = True if normtag is None else run in normtag.index
            in_runs = True if self.runs is None else fill in self.runs
            in_fills = True if self.fills is None else fill in self.fills
            return in_fills and in_runs and in_normtag

        in_range: List[Dict[str, Any]] = []
        for c_path in self.central.glob("**/*.hd5"):
            fill, run = self._parse_file_path(c_path)
            if _in_requested_range(fill, run):
                tag, iov = normtag.loc[run] if normtag else ("N/A", [[0, 9999]])
                b_path = self.beam_central / str(fill) / c_path.name
                in_range.append({
                    "fill": fill, "run": run,
                    "tag": tag, "iov": iov,
                    "c_path": c_path, "b_path": b_path
                })
        return in_range

    def __iter__(self):
        return self

    def __next__(self) -> IterationContext:
        if self.c_handle:
            self.c_handle.close()
            self.b_handle.close()

        while True:
            self.current_path_index += 1
            try:
                context = self.paths[self.current_path_index]
            except:
                if self.c_handle:
                    self.c_handle.close()
                    self.b_handle.close()
                raise StopIteration

            self.c_handle = tb.open_file(context.pop("c_path"), "r")
            self.b_handle = tb.open_file(context.pop("b_path"), "r")
            return IterationContext(
                **context,
                c_handle=self.c_handle,
                b_handle=self.b_handle,
            )
