from pathlib import Path

import sys
import argparse

import numpy as np
import pandas as pd
import mplhep as hep
import matplotlib.pyplot as plt

sys.path.insert(0, Path(".").absolute().as_posix())
try:
    from poggers.io import read_fill
    from poggers.models import sub_nl
except ImportError as e:
    print("Error importing from poggers. Run as: 'python plot.py --help'.")
    sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("PLT", type=Path, help="Path to poggers PLT folder.")
    parser.add_argument("fill", type=int, help="Fill to plot.")
    parser.add_argument("alphas", type=Path, help="Path to the 'best_params.pickle' file from performing the simultanwous fitting.")
    parser.add_argument("calib", type=Path, help="Path to PLT per channel calibrations.")
    parser.add_argument("fpm", type=Path, help="Path to fpm SBIR results for this fill.")
    parser.add_argument("--output", type=Path, help="Output folder. Displays plot if not provided", default=None)
    args = parser.parse_args()

    attrs, df = read_fill(args.PLT, args.fill, "plt", index_filter=(0.05, 0.98))
    alphas = pd.read_pickle(args.alphas)
    fpm_results = pd.read_pickle(args.fpm)

    channels = list(alphas.keys())
    alphas = np.array(list(alphas.values()))
    effs = np.array([fpm_results[f"pltlumizero_{ch}"]["fit_results_lead"][0]["intercept"] for ch in channels])
    calib = pd.read_csv(args.calib).set_index("chid").loc[channels]["calib"].values

    time = pd.to_datetime(df["time"], unit='s')
    df = df[channels]
    df_nl = sub_nl(df, alphas.T, attrs["nbx"])
    
    df *= 11245.5 / calib
    df_nl *= 11245.5 / calib
    df_nl_e = df_nl / effs

    hep.style.use("CMS")
    fig, axs = plt.subplots(2, 3, figsize=(20, 8), sharex=True)
    axs = axs.flatten()

    for i, (frame, label) in enumerate([(df, "Online"), (df_nl, "Nl corrected"), (df_nl_e, "NL+Eff corrected")]):
        top = axs[i]
        bot = axs[i+3]

        hep.cms.label(exp="", llabel="", rlabel=f"{label} Fill {args.fill} (Year, 13.6 TeV)", fontsize=16, ax=top)
        top.plot(time, frame.filter(regex="\d+"), "o", ms=2)
        if i == 0:
            top.set_ylabel("Inst. Lumi [Hz/$\mu$b]", fontsize=14)
        top.tick_params(axis="both", labelsize=12)

        ratios = frame.filter(regex="\d+").div(frame.mean(axis=1), axis=0)
        avg, std = ratios.sum(axis=0).describe().loc[["mean", "std"]]
        sem = std / np.sqrt(len(channels))
 
        hep.cms.label(exp="", llabel="", rlabel=f"avg: {avg:.3f} rms: {std/avg*100:.3f}% sem: {sem/avg*100:.3f}%", fontsize=16, ax=bot)
        bot.plot(time, ratios, "o", ms=2, label=ratios.columns if i == 1 else "")
        if i == 0:
            bot.set_ylabel("Ratio", fontsize=12)
        if i == 1:
            bot.legend(markerscale=2, loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=12, ncol=len(channels)//2)
        bot.set_xlabel("Time", fontsize=14)
        bot.tick_params(axis="both", labelsize=12)

    plt.tight_layout()
    if args.output:
        args.output.mkdir(exist_ok=True, parents=True)
        fig.savefig(args.output/"comparison.png", bbox_inches='tight')
    else:
        plt.show()
