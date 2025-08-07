"""
This example will run the fpm analysis on the the chosen hd5 scan file. It will run the framework with the QG fit and with no correction. It will run a total of 6 iterations and will not distinguish between lead and train bunches.

In brildev an example hd5 file path would be '/brildata/vdmdata23/8880/8880_230606034730_230606035606.hd5'.

Example: python fpm.py --framework_path <path-to-framework-repo> --hd5_path <path-to-one-of-the-scan-hd5> --calib_path calib.csv --output fpm_8880 --fit_function QG --correction noCorr --iterations 6 --use_cached --no_bcid_distinction --corr_flags ""
"""

import argparse
import logging
import os
import stat
import subprocess
from pathlib import Path
from typing import Tuple, Dict
import json
import pickle

import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit
from sklearn.linear_model import HuberRegressor
import matplotlib.pyplot as plt
import mplhep as hep

# Setup logging
logging.basicConfig(level=logging.WARN, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def read_output_csv(results_path: Path, fill: int, fit: str, det: str, corr: str) -> pd.DataFrame:
    det_path = results_path / det / "results" / corr
    if not det_path.exists():
        logger.error(f"{det_path.as_posix()} does not exist!")
        raise FileNotFoundError(f"{det_path.as_posix()} does not exist!")

    csv_files = [
        det_path / f"LumiCalibration_{det}_{fit}_{fill}.csv",
        det_path / f"FOM_LumiCalibration_{det}_{fit}_{fill}.csv"
    ]

    for csv_file in csv_files:
        if csv_file.exists():
            logger.debug(f"Reading CSV file: {csv_file}")
            return pd.read_csv(csv_file)

    logger.error(f"No results found for {fill}:{fit}:{det}:{corr}")
    raise FileNotFoundError(f"No results found for {fill}:{fit}:{det}:{corr}")

def read_filling_scheme_info(results_path: Path) -> np.lib.npyio.NpzFile:
    file_path = results_path / "cond" / "zero_indexed_filling_scheme_info.npz"
    if not file_path.exists():
        logger.error(f"No filling scheme found for {results_path.as_posix()}")
        raise FileNotFoundError(f"No filling scheme found for {results_path.as_posix()}")

    logger.debug(f"Loading filling scheme info from: {file_path}")
    return np.load(file_path)

def huber(x: pd.Series, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
    huber = HuberRegressor().fit(x.values.reshape(-1, 1), y)
    mask = huber.outliers_
    return x[~mask], y[~mask], huber.intercept_, huber.coef_[0]

def fit(x: pd.Series, y: pd.Series):
    return linregress(x, y)

def calculate_incomplete_alpha_no_bcid_distinction(
    ax: plt.Axes, data: pd.DataFrame, fill: int, calib: float, det: str, assumed_alpha: float, label: str
) -> Tuple[Dict, Dict]:
    hep.cms.label(
        exp="", llabel="", fontsize=14, ax=ax,
        rlabel=f"Assumed $\\alpha_{{{label}}}={{{assumed_alpha:.3f}}}$, {det}, Fill {fill} (2023, 13.6 TeV)"
    )

    x, y = data[label], data["xsec"] / calib
    ax.plot(x, y, "o")

    x, y, intercept, slope = huber(x, y)
    ax.plot(x, y, "o")

    alpha = slope / intercept**2

    ax.plot(
        x, x * slope + intercept, "-", color="black", lw=2,
        label=f"FOM = {slope:.3f}*{label} + {intercept:.3f} $\\rightarrow \\alpha_{{{label}}} = $ {alpha:.3f}/{label}"
    )

    ax.legend(fontsize=14, loc="upper left", bbox_to_anchor=(-0.05, -0.04))
    ax.set_ylabel("$\\sigma^{emit}_{vis} / \\sigma^{vdM}_{vis}$", fontsize=18)
    ax.set_xlabel(label, fontsize=18)
    ax.tick_params(axis="both", labelsize=12)

    res = {"slope": slope, "intercept": intercept}
    return res, res

def calculate_incomplete_alpha(
    ax: plt.Axes, data: pd.DataFrame, leading: np.ndarray, fill: int, calib: float, det: str,
    assumed_alpha_lead: float, assumed_alpha_train: float, label: str
) -> Tuple[Dict, Dict]:
    hep.cms.label(
        exp="", llabel="", fontsize=14, ax=ax,
        rlabel=f"Assumed $\\alpha^L_{{{label}}}={{{assumed_alpha_lead:.3f}}}$, $\\alpha^T_{{{label}}}={{{assumed_alpha_train:.3f}}}$, {det}, Fill {fill} (2023, 13.6 TeV)"
    )

    lead_mask = data["BCID"].isin(leading)

    ress = []
    for mask in [lead_mask, ~lead_mask]:
        subset = data[mask].copy()
        x, y = subset[label], subset["xsec"] / calib
        ax.plot(x, y, "o")

        x, y, intercept, slope = huber(x, y)
        ax.plot(x, y, "o")

        ress.append({"slope": slope, "intercept": intercept})

    for res, color, label in zip(ress, ["purple", "black"], ["Lead", "Train"]):
        alpha = res["slope"] / res["intercept"]**2
        ax.plot(
            x, x * res["slope"] + res["intercept"], "-", color=color, lw=2,
            label=f"{label} FOM = {res['slope']:.3f}*{label} + {res['intercept']:.3f} $\\rightarrow \\alpha = $ {alpha:.3f}/{label}"
        )

    ax.legend(fontsize=14, loc="upper left", bbox_to_anchor=(-0.05, -0.04))
    ax.set_ylabel("$\\sigma^{emit}_{vis} / \\sigma^{vdM}_{vis}$", fontsize=18)
    ax.set_xlabel(label, fontsize=18)
    ax.tick_params(axis="both", labelsize=12)

    return ress[0], ress[1]

def exp_decay(x, A, B):
    return A * np.exp(-B * x)

def make_exponential_fit(x_data, alphas):
    params, covariance = curve_fit(exp_decay, x_data, alphas)
    A_fit, B_fit = params
    return A_fit, B_fit, covariance

def combined_model(x, A, B, a, b, c):
    return A * np.exp(-B * x) + (a*x**2 + b*x + c)

def make_combined_fit(x_data, alphas):
    params, covariance = curve_fit(combined_model, x_data, alphas)
    A_fit, B_fit, a, b, c = params
    return A_fit, B_fit, a, b, c, covariance

def make_convoluted_fit(x_data, alphas):
    params, covariance = curve_fit(convoluted_model, x_data, alphas)
    A_fit, B_fit, a, b, c = params
    return A_fit, B_fit, a, b, c, covariance

def write_script(script_path: Path, fw_path: Path, hd5_path: Path, fit_function: str, json_config: dict, fom_path: Path, fw_folder: Path, corr_flags: str = ""):
    script_content = f"""#!/bin/bash
cd {fw_path}
cd src
source cmsenv.sh
python auto_analysis.py -f {hd5_path} {corr_flags} -fc {fit_function} -l {" ".join(json_config)} -fom {fom_path.absolute()} -o {fw_folder.absolute()}
"""
    script_path.write_text(script_content)
    st = os.stat(script_path)
    os.chmod(script_path, st.st_mode | stat.S_IEXEC)
    logger.debug(f"Execution script written at: {script_path}")

def run_analysis(script_path: Path):
    result = subprocess.run([script_path.as_posix()], shell=True)
    if result.returncode != 0:
        logger.error(f"Error in running the framework script {script_path}")
        raise RuntimeError(f"Script {script_path} failed with return code {result.returncode}")
    logger.debug("Framework script ran successfully")

def main(args):
    hep.style.use("CMS")

    output = args.output
    prefix = "no_bcid_distinction_" if args.no_bcid_distinction else ""
    output.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created at: {output}")

    # Setup basic configuration
    json_config, results_sbir, results_sbil = {}, {}, {}
    channels = [i for i in range(16) if i not in (6, 8, 9)]
    fill = int(args.hd5_path.stem.split("_")[0])
    calib_per_channel = pd.read_csv(args.calib_path).set_index("chid")

    for channel in channels:
        json_config[f"pltlumizero_{channel}"] = {str(fill): {"lead": 0.0, "train": 0.0}}
        results_sbir[f"pltlumizero_{channel}"] = {
            "assumed_lead": [],
            "assumed_train": [],
            "fit_results_lead": [],
            "fit_results_train": [],
        }
        results_sbil[f"pltlumizero_{channel}"] = {
            "assumed_lead": [],
            "assumed_train": [],
            "fit_results_lead": [],
            "fit_results_train": [],
        }

    for i in range(args.iterations):
        iteration_folder = output / f"iteration_{i:03d}"
        iteration_folder.mkdir(parents=True, exist_ok=True)
        fw_folder, plots_path = iteration_folder / f"{prefix}output", iteration_folder / f"{prefix}plots"
        plots_path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        fom_path = iteration_folder / f"{prefix}fom_config.json"
        fom_path.write_text(json.dumps(json_config, indent=4))

        # Write script
        script_path = iteration_folder / f"{prefix}script.sh"
        write_script(script_path, args.framework_path, args.hd5_path, args.fit_function, json_config, fom_path, fw_folder, args.corr_flags)

        # Run the analysis
        if not fw_folder.exists() or not args.use_cached:
            run_analysis(script_path)
        else:
            logger.warning("Using existing results; skipping framework execution.")

        results_path = next((fw_folder / "analysed_data").glob(f"{fill}*"))
        for results, label in [(results_sbir, "SBIR"), (results_sbil, "SBIL")]:
            for detname in results:
                ch = int(detname.split("_")[-1])
                det = f"PLT{ch}"
                calib = calib_per_channel.loc[ch]["calib"]

                data = read_output_csv(results_path, fill, args.fit_function, det, args.correction)
                leading = read_filling_scheme_info(results_path)["ileading"] + 1

                assumed_alpha_lead = json_config[detname][str(fill)]["lead"]
                assumed_alpha_train = json_config[detname][str(fill)]["train"]

                plt.figure(figsize=(8, 4))
                if args.no_bcid_distinction:
                    res_lead, res_train = calculate_incomplete_alpha_no_bcid_distinction(
                        plt.gca(), data, fill, calib,
                        det, assumed_alpha_lead, label
                    )
                else:
                    res_lead, res_train = calculate_incomplete_alpha(
                        plt.gca(), data, leading, fill, calib,
                        det, assumed_alpha_lead, assumed_alpha_train, label
                    )

                results[detname]["assumed_lead"].append(assumed_alpha_lead)
                results[detname]["assumed_train"].append(assumed_alpha_train)
                results[detname]["fit_results_lead"].append(res_lead)
                results[detname]["fit_results_train"].append(res_train)

                if label == "SBIR":
                    json_config[detname][str(fill)]["lead"] += res_lead["slope"] / res_lead["intercept"]**2
                    json_config[detname][str(fill)]["train"] += res_train["slope"] / res_train["intercept"]**2

                plt.savefig(plots_path / f"{label}_{det}.png", bbox_inches='tight')
                plt.close()

    for results, label in [(results_sbir, "SBIR"), (results_sbil, "SBIL")]:
        for detname, result in results.items():
            ch = int(detname.split("_")[-1])
            det = f"PLT{ch}"

            plt.figure(figsize=(10, 4))
            hep.cms.label(exp="", llabel="", rlabel=f"{det}, Fill {fill} (2023, 13.6 TeV)", fontsize=18)
            bx_labels = ["lead"] if args.no_bcid_distinction else ["lead", "train"]
            for i, bx_label in enumerate(bx_labels):
                alphas = [r["slope"] / r["intercept"]**2 for r in result[f"fit_results_{bx_label}"]]
                x_data = np.arange(len(alphas))

                if label == "SBIR":
                    A, B, covariance = make_exponential_fit(x_data, alphas)
                else:
                    A, B, a, b, c, covariance = make_combined_fit(x_data, alphas)

                infinite_sum = A / (1 - np.exp(-B))

                x_fit = np.linspace(0, args.iterations - 1, 100)
                
                if label == "SBIR":
                    y_fit = exp_decay(x_fit, A, B)
                else:
                    y_fit = combined_model(x_fit, A, B, a, b, c)
                
                if args.no_bcid_distinction:
                    plt.text(
                        0.2 + i * 0.4, -0.2, 
                        f"$\\sum \\alpha_i = \\frac{{{A:.4f}}}{{1 - e^{{-{B:.4f}}}}} = {infinite_sum:.3f}$",
                        fontsize=16, ha="center", transform=plt.gca().transAxes
                    )
                    plt.plot(x_data, alphas, 'o')
                    if label == "SBIR":
                        plt.plot(x_fit, y_fit, '-', label=f'Fitted Curve: $\\alpha^i_{{{label}}} = {A:.4f} e^{{-{B:.4f} i}}$')
                    else:
                        plt.plot(x_fit, y_fit, '-', label=f'Fitted Curve: $\\alpha^i_{{{label}}} = {A:.4f} e^{{-{B:.4f} i}} + 2ºPoly({a:.2e},{b:.2e},{c:.2e})$')
                else:
                    plt.text(
                        0.2 + i * 0.4, -0.2, 
                        f"$\\sum \\alpha{{{bx_label[0].upper()}}}_i = \\frac{{{A:.4f}}}{{1 - e^{{-{B:.4f}}}}} = {infinite_sum:.3f}$",
                        fontsize=16, ha="center", transform=plt.gca().transAxes
                    )
                    plt.plot(x_data, alphas, 'o', label=f'{bx_label.title()}')
                    if label == "SBIR":
                        plt.plot(x_fit, y_fit, '-', label=f'Fitted Curve: {bx_label[0].upper()} $\\alpha^i_{{{label}}} = {A:.4f} e^{{-{B:.4f} i}}$')
                    else:
                        plt.plot(x_fit, y_fit, '-', label=f'Fitted Curve: {bx_label[0].upper()} $\\alpha^i_{{{label}}} = {A:.4f} e^{{-{B:.4f} i}} + 2ºPoly({a:.2e},{b:.2e},{c:.2e})$')

                results[detname][f"A_{bx_label}"] = A
                results[detname][f"B_{bx_label}"] = B
                results[detname][f"alpha_{bx_label}"] = infinite_sum
                if args.no_bcid_distinction:
                    results[detname][f"A_train"] = A
                    results[detname][f"B_train"] = B
                    results[detname][f"alpha_train"] = infinite_sum

            plt.ylabel(f'Assumed $\\alpha_{{{label}}}$', fontsize=18)
            plt.xlabel('i', fontsize=18)
            ncol = 2 if args.no_bcid_distinction else 1
            plt.legend(fontsize=14, frameon=True, ncol=ncol)
            plt.tick_params(axis="both", labelsize=12)
            plt.savefig(output / f"{prefix}{label}_{det}_exp_fit.png", bbox_inches='tight')
            plt.close()
            logger.info(f"Exponential fit for {det} saved to {output / f'{det}_exp_fit.png'}")

        with open(output / f"{prefix}results_{label}.pickle", "wb") as fp:
            pickle.dump(results, fp)
        logger.info(f"{label} results saved to pickle")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analysis Script with Improvements")
    parser.add_argument("--framework_path", type=Path, required=True, help="Path to the VdM framework")
    parser.add_argument("--hd5_path", type=Path, required=True, help="Path to HD5 file")
    parser.add_argument("--calib_path", type=Path, required=True, help="Path to calibration CSV file")
    parser.add_argument("--output", type=Path, default=Path("fixed_point_results"), help="Output directory")
    parser.add_argument("--fit_function", type=str, default="QG", help="Fitting function to use")
    parser.add_argument("--correction", type=str, default="noCorr", help="Correction type")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations")
    parser.add_argument("--use_cached", action="store_true", help="Run the framework. If omitted, use existing results.")
    parser.add_argument("--no_bcid_distinction", action="store_true", help="Differentiate between lead and train bunches.")
    parser.add_argument("--corr_flags", type=str, help="Correction flags compatible with the vdM Framework.")
    args = parser.parse_args()

    main(args)
