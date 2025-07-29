from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import *

import sys
import pickle
import argparse
import functools

from scipy.stats import pearsonr
from tqdm.autonotebook import tqdm

import numpy as np
import mplhep as hep
import matplotlib.pyplot as plt

sys.path.insert(0, Path("..").absolute().as_posix())
from simultaneous_fitting import simfit


def generate_initial_alphas(initial: float, channels: List[int]) -> Dict[int, float]:
    return {ch: initial for ch in channels}

@dataclass
class ISimFitResult:
    # Meta
    fill: int
    iterations: int
    log_form: bool
    n_trials: int
    # Data
    bounds: np.ndarray = field(repr=False)
    alphas: np.ndarray = field(repr=False)
    channels: np.ndarray = field(repr=False)

def isimfit(iterations: int, alphas: Dict[int, float], *args, **kwargs) -> ISimFitResult:
    bound: float = kwargs.get("bound", 0.2)

    ibounds = np.empty(iterations, dtype=float)
    ialphas = np.empty((iterations, len(alphas)), dtype=float)
    for i in tqdm(range(iterations)):
        results = simfit(*args, alphas=alphas, **kwargs)
        old_alphas: np.ndarray = np.array(list(alphas.values()))
        new_alphas: np.ndarray = np.array(list(results.values()))

        ibounds[i] = bound
        ialphas[i] = old_alphas

        alphas = results
        if max((abs(new_alphas - old_alphas) / old_alphas)) < bound*0.8:
            bound = round(bound*0.8, 2)

    fill = args[0]
    n_trials: int = kwargs.get("n_trials", 100)
    use_log_form: bool = kwargs.get("use_log_form", False)
    return ISimFitResult(
        fill, iterations, use_log_form, n_trials,
        ibounds, ialphas, np.array(list(alphas.keys()))
    )

def plot_alpha_convergence(results: List[ISimFitResult]) -> plt.Figure:
    fig, axs = plt.subplots(3, 3, figsize=(20, 15), sharex=True)
    axs: List[plt.Axes] = axs.flatten()
    for ax in axs:
        ax.axis("off")

    hep.style.use("CMS")
    for i, ch in enumerate(results[0].channels):
        ax = axs[i]
        ax.axis("on")
        ch_alphas: np.ndarray = np.array([res.alphas[:, i] for res in results])

        avg = np.mean(ch_alphas[:, -1])
        std = np.std(ch_alphas[:, -1]) / avg * 100
        sem = std / np.sqrt(len(results))

        hep.cms.label(exp="", llabel="", rlabel=f"$\\alpha$ = {avg:.3f} rms: {std:.3f}%, sem: {sem:.3f}%, Ch{ch}", ax=ax, fontsize=16)
        ax.plot(ch_alphas.T, "--")
        ax.axhline(avg, 0, results[0].iterations, color="black", lw=3)
        xlim = ax.get_xlim()
        ax.fill_between(xlim, avg-std*avg/100, avg+std*avg/100, alpha=0.1, color="red")
        ax.set_xlim(xlim)
        ax.set_ylabel("$\\alpha_i$", fontsize=16)
        ax.set_xlabel("Iteration", fontsize=16)

    return fig

def plot_alpha_correlation(results: List[ISimFitResult]) -> plt.Figure:
    fig, axs = plt.subplots(3, 3, figsize=(20, 15), sharex=True)
    axs: List[plt.Axes] = axs.flatten()
    for ax in axs:
        ax.axis("off")

    initial_alphas = np.array([res.alphas[0,0] for res in results])

    hep.style.use("CMS")
    for i, ch in enumerate(results[0].channels):
        ax = axs[i]
        ax.axis("on")

        ch_alphas: np.ndarray = np.array([res.alphas[:, i] for res in results])
        end_alphas = ch_alphas[:, -1]

        corr, p_value = pearsonr(initial_alphas, end_alphas)

        hep.cms.label(exp="", llabel="", rlabel=f"Corr($\\alpha_i$, $\\alpha_f$) = {corr:.2f}, $R^2$ = {corr**2:.2f}, p = {p_value:.2e} Ch{ch}", ax=ax, fontsize=16)
        ax.plot(initial_alphas, end_alphas, "o")
        ax.set_ylabel("Final $\\alpha$ [$\mu^{-1}$]", fontsize=16)
        ax.set_xlabel("Initial $\\alpha$ [$\mu^{-1}$]", fontsize=16)

    return fig

def parse_parameter_space(range_str: str) -> Tuple[int, int, int]:
    """Parse a range string in the form 'start-end'."""
    try:
        start, end, gran = range_str.split(':')
        if start > end:
            raise ValueError("Start of the range cannot be greater than the end.")
        return np.linspace(float(start), float(end), int(gran))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid range format: '{range_str}'. Expected format 'start-end'.")

def run_isimfit(
    init: Union[float, np.ndarray],
    iterations: int, fill: int, plt_path: Path, channels: List[int],
    bound: float, n_trials: int, use_log_form: bool
):
    if isinstance(init, float):
        alphas = generate_initial_alphas(init, channels)
    elif isinstance(init, np.ndarray):
        alphas = dict(zip(channels, init))
    else:
        raise TypeError("init can only be float of np.ndarray.")

    return isimfit(
        iterations, alphas,
        fill, plt_path, channels, "CUSTOM",
        bound=bound, n_trials=n_trials, use_log_form=use_log_form,
        save_results=False, add_pbar=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Focused simultanous fitting.",
    )
    parser.add_argument("fill", type=int, help="Fill to analyse.")
    parser.add_argument("PLT", type=Path, help="Path to the poggers data for PLT.")
    parser.add_argument("channels", type=int, nargs="+", help="Channels to use in the fit.")
    parser.add_argument("--output", type=Path, default=Path("isimfit"), help="Output directory.")
    parser.add_argument("--iterations", type=int, default=15, help="Number of focussing iterations.")
    parser.add_argument("--starts", type=int, default=50, help="Number of starting positions.")
    parser.add_argument("--param-space", type=parse_parameter_space, default="0.1:0.9:50", help="Parameter space in the format start:end:granularity. To be used by numpy.linspace.")
    g_simfit = parser.add_argument_group("simfit", "simfit related parameters")
    g_simfit.add_argument("--use-log-form", action="store_true", help="Use the log formulation of ratio model.")
    g_simfit.add_argument("--bound", type=float, default=0.2, help="Alpha percentage bound")
    g_simfit.add_argument("--n-trials", type=int, default=100, help="Number of optimization trials")
    g_simfit.add_argument("--seed", type=int, default=42, help="Seed given to optuna's sampler.")
    args = parser.parse_args()
    
    alpha_choices = np.random.choice(args.param_space, (args.starts, len(args.channels)))

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_isimfit, alphas, args.iterations, args.fill, args.PLT, args.channels,
                args.bound, args.n_trials, args.use_log_form
            ) 
            for alphas in alpha_choices
        ]
        for future in futures:
            results.append(future.result())

    args.output.mkdir(exist_ok=True, parents=True)
    with open(args.output/f"{args.fill}_results.pkl", "wb") as fp:
        pickle.dump(results, fp)

    plot_alpha_convergence(results).savefig(args.output/f"{args.fill}_convergence.png", bbox_inches="tight")
    plot_alpha_correlation(results).savefig(args.output/f"{args.fill}_correlation.png", bbox_inches="tight")