"""
This script assumes you have already run the poggers code on fill 8880 for both PLT and DT.

Example: python dt_collinearity/dt_collinearity.py poggers_plt_example poggers_dt_example 8880 plt dt --output collinearity_8880
"""

from typing import * 
from pathlib import Path
from dataclasses import dataclass

import sys
import json
import argparse

from sklearn.linear_model import HuberRegressor
from scipy.stats import linregress

import numpy as np
import pandas as pd
import mplhep as hep
import statsmodels.api as sm
import matplotlib.pyplot as plt

sys.path.insert(0, Path(".").absolute().as_posix())
try:
    from poggers.io import read_fill
    from poggers.models import sub_nl
    from poggers.options import PoggerOptions
except ImportError as e:
    print(f"Error importing from poggers. Run as: 'python dt_collinearity/dt_collinearity.py --help'.")
    sys.exit(1)


def huber_regressor_with_uncertainties(huber, X, y):
    # Predict residuals and calculate weights
    residuals = y - huber.predict(X)
    delta = huber.epsilon * np.std(residuals)
    weights = np.where(np.abs(residuals) <= delta, 1, delta / np.abs(residuals))
    
    # Create the design matrix (add intercept term)
    X_design = sm.add_constant(X)
    
    # Perform weighted least squares regression
    wls_model = sm.WLS(y, X_design, weights=weights).fit()
    
    # Extract parameters and uncertainties
    intercept = wls_model.params[0]
    intercept_uncertainty = wls_model.bse[0]
    slope = wls_model.params[1]
    slope_uncertainty = wls_model.bse[1]

    red_chi2 = wls_model.ssr / (len(y) - 2)
    
    return red_chi2, intercept, intercept_uncertainty, slope, slope_uncertainty

def huber_regressor_with_uncertainties2(huber, X, y):
    res = linregress(X.reshape(-1), y)

    ypred = X * res.slope + res.intercept
    residuals = y - ypred
    red_chi2 = np.sum(residuals**2) / (len(y) - 2)

    return red_chi2, huber.intercept_, res.intercept_stderr, huber.coef_[0], res.stderr


def measure_collinearity(m: pd.DataFrame, channels: List[int], fill: int, ref_name: str, det_name: str, nbx: int) -> plt.Figure:
    m_ = m.filter(regex="\d+")
    print(len(channels))
    sqrt = np.sqrt(len(channels))
    ceil = np.ceil(np.sqrt(len(channels)))
    nrows = int(ceil)
    ncols = int(ceil)
    if ceil - sqrt >= 0.5:
        ncols -= 1
    elif np.allclose(ceil-sqrt, 0.0):
        pass
    else:
        ncols += 1

    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 18))
    fig.subplots_adjust(wspace=0.4, hspace=0.5)
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]

    results = {}
    for i, chid in enumerate(channels):
        ax: plt.Axes = axs[i]

        muA = (m_[chid] / nbx).values
        muB = (m_[f"{chid}{ref_name}"] / nbx).values

        # Remove NaN values in both detectors
        pre_ratio = muA / muB
        muA_no_nan = muA[np.isfinite(pre_ratio)]
        muB_no_nan = muB[np.isfinite(pre_ratio)]

        if len(muA_no_nan) == 0 or len(muB_no_nan) == 0:
            print(f"WARNING: Skipping channel {chid}. No valid data points.")
            ax.text(
                0.5, 0.5, "NO DATA", transform=ax.transAxes,
                fontsize=18, color="red", ha="center", va="center"
            )
            continue

        # Remove Any ratios outside 2 std
        ratio = pre_ratio[np.isfinite(pre_ratio)]
        mean, std = ratio.mean(), ratio.std()
        good_ratio_mask = ~((ratio > mean+std*2) | (ratio < mean-std*2))
        muA_filtered = muA_no_nan[good_ratio_mask]
        muB_filtered = muB_no_nan[good_ratio_mask]

        try:
            huber = HuberRegressor().fit(muB_filtered.reshape(-1, 1), muA_filtered/muB_filtered)
            (
                rchi2,
                i, istd,
                s, sstd
            ) = huber_regressor_with_uncertainties2(huber, muB_filtered.reshape(-1, 1), muA_filtered/muB_filtered)
            a = s / i**2
            astd = abs(s / i**2) * np.sqrt(
                (sstd / s)**2 + (2 * istd / i)**2
            )
        except ValueError as err:
            print(f"ERROR: Huber failed: {err}")
            ax.text(
                0.5, 0.5, "Huber Failed", transform=ax.transAxes,
                fontsize=18, color="red", ha="center", va="center"
            )
            continue
        except ZeroDivisionError as err:
            print(f"ERROR: Alpha Uncertainty Estimation Failed: {err}")
            ax.text(
                0.5, 0.5, "$\\alpha$ STD FAILED", transform=ax.transAxes,
                fontsize=16, color="red", ha="center", va="center"
            )
            continue
        
        results[chid] = {
            "slope": s,
            "rchi2": rchi2,
            "intercept": i,
            "alpha": a,
            "slope_uncertainty": sstd,
            "intercept_uncertainty": istd,
            "alpha_uncertainty": astd
        }
        muAc = sub_nl(muA_filtered, a, 1)

        hep.cms.label(
            exp="", ax=ax,
            llabel=f"$\\alpha = {a:.3f} \pm {astd/a*100:.3f} \\% (\\chi^2 = {rchi2:.3f})$",
            rlabel=f"ch{chid}"
        )
        ax.plot(muB_filtered, muA_filtered/muB_filtered, "o")
        ax.plot(muB_filtered, muAc/muB_filtered, "o")
        ax.plot(muB_filtered, muB_filtered*s+i, "--")
        ax.tick_params(axis='both', labelsize=8)
        ax.ticklabel_format(axis="both", style='plain', useOffset=False)

    fig.suptitle(f"DET: {det_name}, REF: {ref_name}, Fill {fill}", ha="center", y=0.99)
    return fig, results


@dataclass
class ParsedArgs:
    DET: Path
    REF: Path
    fill: int
    det_name: str
    ref_name: str
    index_filter: Tuple[float, float]
    vdm_central: Path
    burnoff_path: Path
    output: Path

    def __post_init__(self):
        self.index_filter = tuple(map(float, self.index_filter.split(",")))



def reference_is_ambiguous(data_columns: List[str]) -> bool:
    return len(data_columns) != 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="PLT vs DT Collinearity: Calculate collinearity between detector PLT and DT (assumed to be linear)",
    )
    
    parser.add_argument(
        "DET", type=Path, help="Path to the poggers data for detector."
    )
    parser.add_argument(
        "REF", type=Path, help="Path to the poggers data for the reference."
    )
    parser.add_argument(
        "fill", type=int, help="Fill to determine collinearity."
    )
    parser.add_argument(
        "det_name", type=str, help="Alias to use for DET."
    )
    parser.add_argument(
        "ref_name", type=str, help="Alias to use for DET."
    )
    parser.add_argument(
        "--index_filter", type=str, help="Ignore bottom and top percentile. Default '0.1,0.95'"
    )
    parser.add_argument(
        "--vdm_central", type=Path, help="Path to vdM central.", metavar=""
    )
    parser.add_argument(
        "--burnoff_path", type=Path, help="Path to burnoff data.", metavar=""
    )
    parser.add_argument(
        "--output", type=Path, default=None, metavar="",
        help="Path to save the collinearity plot and results."
        " If not specified the plot will be displayed and the results will be printed."
    )
    
    namespace = parser.parse_args()
    args = ParsedArgs(**vars(namespace))

    options = PoggerOptions()
    if args.vdm_central is not None:
        options.vdm_path = args.vdm_central
    if args.burnoff_path is not None:
        options.burnoff_path = args.burnoff_path

    attrsDET, dfDET = read_fill(
        args.DET, args.fill, args.det_name,
        remove_scans=True, index_filter=args.index_filter, agg_per_ls=True
    )
    attrsREF, dfREF = read_fill(
        args.REF, args.fill, args.ref_name,
        remove_scans=True, index_filter=args.index_filter, agg_per_ls=True
    )

    if reference_is_ambiguous(dfREF.filter(regex="\d+").columns):
        raise ValueError("Provided REF detector contains more than 1 data column making it ambigous.")

    channels = dfDET.filter(regex="\d+").columns
    dfREF = dfREF.rename(columns={0: args.ref_name})

    m = pd.merge(
        dfDET, dfREF, on=["run", "lsnum"],
        how="inner", suffixes=("", args.ref_name)
    ).drop(columns=[f"time{args.ref_name}"])
    
    # Normalizing detector B to detector A scale
    avg_ratios = m.filter(regex="\d+").div(m[args.ref_name], axis=0).mean(axis=0)
    m = m.assign(
        **{f"{chid}{args.ref_name}": m[args.ref_name] * avg_ratios[chid] for chid in avg_ratios.index}
    )

    fig, results = measure_collinearity(m, channels, args.fill, args.ref_name, args.det_name, attrsDET["nbx"])

    if args.output:
        args.output.mkdir(exist_ok=True, parents=True)
        fig.savefig(args.output / "collinearity.png")
        with open(args.output / "collinearity.json", "w") as fp:
            json.dump(results, fp)
    else:
        plt.show()
        print(results)
