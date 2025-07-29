"""
Run examples/poggers_plt.py and examples/poggers_dt.py.
Run dt_collinearity. Only then try the example
Example: python simultaneous_fitting/simultaneous_fitting.py 8880 poggers_plt_example/ 0 1 2 3 10 11 12 14 15 DT collinearity/collinearity.json
"""

import argparse
from pathlib import Path
from typing import *
import os
import sys
import json
import pickle

from tqdm.autonotebook import tqdm
from scipy.special import huber
from scipy.stats import linregress
from numpy.polynomial import Polynomial
from sklearn.linear_model import HuberRegressor

import optuna
import numpy as np
import pandas as pd
import mplhep as hep
import matplotlib.pyplot as plt

sys.path.insert(0, Path(".").absolute().as_posix())
try:
    from poggers.io import read_fill
    from poggers.models import sub_nl
except ImportError as e:
    print("Error importing from poggers. Run as: 'python simultaneous_fitting/simultaneous_fitting.py --help'.")
    sys.exit()

# Suppress Optuna INFO/DEBUG logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

class TQDMProgressBar:
    """Custom callback to integrate tqdm with Optuna optimization."""
    def __init__(self, total_trials: int):
        self.total_trials = total_trials
        self.pbar = tqdm(total=self.total_trials, desc="Optimization Progress", unit="trial")

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        self.pbar.update(1)

    def close(self):
        self.pbar.close()


def read_plt_fpm_alphas(path: Path, label: str="lead") -> Dict[int, float]:
    if label not in ("lead", "train"):
        raise ValueError(f"'label' parameter should only be 'lead' or 'train'. Got '{label}' instead.")
    results = pd.read_pickle(path)
    return {int(name.split("_")[-1]): res[f"alpha_{label}"] for name, res in results.items()}

def read_json_alphas(path: Path) -> Dict[int, float]:
    with open(path) as fp:
        results = json.load(fp)
    return {int(ch): res['alpha'] for ch, res in results.items()}

def get_channels(good_channels: Set[int], existing_channels: Set[int]) -> List[int]:
    existing_good_channels = existing_channels.intersection(good_channels)
    if existing_good_channels != good_channels:
        print(f"Fill data missing some channels. Missing channels are: {good_channels - existing_good_channels}")
    return existing_good_channels

SlopeInterceptFitter = Callable[[np.ndarray, np.ndarray], np.ndarray]
def default_fitter(x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
    deg = kwargs.pop("deg") if "deg" in kwargs else 1
    p: Polynomial = Polynomial.fit(x, y, deg, **kwargs)
    return p.convert().coef

def compute_intercepts_and_slopes(data: np.ndarray, fitter: SlopeInterceptFitter, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    N = data.shape[0]
    slopes = np.ones((N, N))
    intercepts = np.ones((N, N))

    I, J = np.triu_indices(N, k=1)
    for i, j in zip(I, J):
        ratio = data[i] / data[j]
        intercept, slope = fitter(data[j], ratio, **kwargs)
        intercepts[i, j] = intercept
        intercepts[j, i] = 1 / intercept
        slopes[i, j] = slope
        slopes[j, i] = -slope

    return intercepts, slopes

def ratio_model(x: np.ndarray, a1: float, a2: float, c: float) -> np.ndarray:
    x1, x2 = np.split(x, 2)
    N = 1 + np.sqrt(4 * a1 * x1 + 1)
    D = 1 + np.sqrt(4 * a2 * x2 + 1)
    return c * N / D

def log_ratio_model(x: np.ndarray, a1: float, a2: float, c: float) -> np.ndarray:
    x1, x2 = np.split(x, 2)
    N = 1 + np.sqrt(4 * a1 * x1 + 1)
    D = 1 + np.sqrt(4 * a2 * x2 + 1)
    return np.log(c) + np.log(N) - np.log(D)

def objective(data: np.ndarray, alphas: np.ndarray, intercepts: np.ndarray, use_log_form: bool=False, delta: float = 1.0):
    N = intercepts.shape[0]
    I, J = np.triu_indices(N, k=1)

    model = log_ratio_model if use_log_form else ratio_model
    if use_log_form:
        left_ratio_func = lambda rm1, rm2: np.log(rm1 / rm2)
    else:
        left_ratio_func = lambda rm1, rm2: rm1 / rm2

    total = 0
    for i, j in list(zip(I, J)):
        rm1, rm2 = data[i], data[j]
        intercept = intercepts[i, j]
        a1, a2 = alphas[i], alphas[j]

        y_fit = model(np.concatenate([rm1, rm2]), a1, a2, intercept)
        y = left_ratio_func(rm1, rm2)

        residual = y_fit - y
        total += np.sum(huber(delta, residual))

    return total

def optuna_objective_wrapper(trial: optuna.Trial, data: np.ndarray, intercepts: np.ndarray, alphas: np.ndarray, alpha_percentage: float, use_log_form: bool):
    lower_bounds = alphas * (1 - alpha_percentage)
    upper_bounds = alphas * (1 + alpha_percentage)

    test_alphas = np.array([
        trial.suggest_float(f"alpha_{i}", lower_bounds[i], upper_bounds[i])
        for i in range(len(alphas))
    ])
    return objective(data, test_alphas, intercepts, use_log_form=use_log_form)

def plot_corrected_ratios(channel_map: List[int], **datas: np.ndarray) -> plt.Figure:
    N = datas[next(iter(datas))].shape[0]
    fig, axes = plt.subplots(N - 1, N - 1, figsize=(25, 15))

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {label: color_cycle[i % len(color_cycle)] for i, label in enumerate(datas.keys())}

    for i in range(N - 1):
        for j in range(i + 1, N):
            for label, data in datas.items():
                original_ratio = data[i] / data[j]
                ax: plt.Axes = axes[i, j - 1]
                ax.scatter(data[j], original_ratio, s=10, color=color_map[label], label=label)

                chi, chj = channel_map[i], channel_map[j]
                ax.set_title(f'Ratio: Detector {chi}/{chj}', fontsize=10)
                ax.tick_params(axis="both", labelsize=8)

    for i in range(N - 1):
        for j in range(i):
            axes[i, j].axis("off")

    num_labels = len(datas)
    spacing = 1 / (num_labels + 1)
    for idx, (label, color) in enumerate(color_map.items()):
        x_position = spacing * (idx + 1)
        fig.text(x_position, 0.98, label, ha='center', va='top', color=color, fontsize=25, weight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def simfit(
    fill: int,
    PLT: Path,
    channels: List[int],
    alpha_source: str,
    alphas: Dict[int, float],
    use_log_form: bool=False,
    bound: float=0.2,
    output: Path=Path("simultaneous_fit_studies"),
    n_trials: int=100,
    seed: int=42,
    save_results: bool=True,
    add_pbar: bool=True
) -> Dict[int, float]:
    attrs, df = read_fill(PLT, fill, "det", remove_scans=True, index_filter=(0.2, 0.95))
    df = df.groupby(["run", "lsnum"]).mean().reset_index()
    channels = list(get_channels(set(channels), set(df.filter(regex="\d+").columns)))
    initial_alphas = np.array([alphas[ch] for ch in channels])

    normalized_rates = df[channels].values.T / attrs["nbx"]
    intercepts, _ = compute_intercepts_and_slopes(normalized_rates, default_fitter)

    study = optuna.create_study(
        study_name="NLStudy",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    
    callbacks = [] 
    if add_pbar:
        pbar = TQDMProgressBar(total_trials=n_trials)
        callbacks = [pbar] 
    try:
        study.optimize(
            lambda trial: optuna_objective_wrapper(trial, normalized_rates, intercepts, initial_alphas, bound, use_log_form),
            n_trials=n_trials, callbacks=callbacks
        )
    except:
        if add_pbar:
            pbar.close()

    best_params = dict(zip(channels, study.best_params.values()))

    if save_results:
        output.mkdir(exist_ok=True, parents=True)
        best_alphas = np.array(list(study.best_params.values()))
        source_rates = sub_nl(normalized_rates, initial_alphas[:, np.newaxis], 1)
        op_rates = sub_nl(normalized_rates, best_alphas[:, np.newaxis], 1)
        fig = plot_corrected_ratios(
            channels, 
            **{
                "original": normalized_rates,
                alpha_source: source_rates,
                "op"+use_log_form*"+log": op_rates
            }
        )
        fig.savefig(output/"collinearity.png", bbox_inches="tight")

        trials_df = study.trials_dataframe()
        trials_df.to_csv(output / "trials.csv", index=False)
        with open(output / "study.pickle", "wb") as fp:
            pickle.dump(study, fp)
        with open(output / "best_params.pickle", "wb") as fp:
            pickle.dump(best_params, fp)

    return best_params
 
def main():
    parser = argparse.ArgumentParser(description="Simultaneous Fit Studies Script")
    parser.add_argument("fill", type=int, help="Fill number")
    parser.add_argument("PLT", type=Path, help="Path to the poggers data for PLT.")
    parser.add_argument("channels", type=int, nargs="+", help="Channels to use in the fit.")
    parser.add_argument("alpha_source", type=str, choices=["CUSTOM", "DT", "FPM"], default="FPM", help="Source of alpha values")
    parser.add_argument("alpha_path", type=Path, help="Path to the either the DT or the FPM alphas.")
    parser.add_argument("--use_log_form", action="store_true", help="Use the log formulation of ratio model.")
    parser.add_argument("--bound", type=float, default=0.2, help="Alpha percentage bound")
    parser.add_argument("--output", type=Path, default="simultaneous_fit_studies", help="Output directory")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument("--seed", type=int, default=42, help="Seed given to optuna's sampler.")
    args = parser.parse_args()


    if args.alpha_source in ("DT", "CUSTOM"):
        alphas = read_json_alphas(args.alpha_path)
    else:
        alphas = read_plt_fpm_alphas(args.alpha_path)
    
    dict_args = vars(args)
    del dict_args["alpha_path"]
    dict_args["alphas"] = alphas
    simfit(**dict_args)

if __name__ == "__main__":
    main()
