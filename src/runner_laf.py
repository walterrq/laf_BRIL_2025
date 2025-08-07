
import numpy as np
from poggers.io import read_fill
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from model.preprocessor import DifferencePreprocessor
from model.figure_of_merit import Processor
import argparse

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(description="Run the Processor with given pickles_path and fill_number.")
parser.add_argument("--plt_path", 
                    type=str,
                    help="Path to the PLT pickles directory.")
parser.add_argument("--dt_path", 
                    type=str,
                    help="Path to the DT pickles directory.")
parser.add_argument("--fill", 
                    type=int, 
                    help="Fill number to process.")
parser.add_argument("--year",
                    type=int,
                    help="Year of the fill to be analyzed")
#parser.add_argument('--is_lumi',
#                   dest="is_lumi",
#                   type= str_to_bool,
#                   default= True,
#                   help = 'Lumi is being analyzed, or is it rates')
parser.add_argument("--corrs_path",
                    dest="corrs_path",
                    type=str,
                    default='/afs/cern.ch/user/f/fromeo/public/4Tomas/corrs_all.json',
                    help="Path to the corrections file")
#parser.add_argument("--study_corr",
#                    dest="study_corr",
#                    type=str_to_bool,
#                    default=True,
#                    help="Perform study with Isolation forest")
parser.add_argument("--out",
                    type=str,
                    default='.',
                    help="Output directory")

args = parser.parse_args()


pickles_path_plt = args.plt_path
pickles_path_dt = args.dt_path
fill_number = args.fill
#is_lumi = args.is_lumi
year = args.year
#study_corr = args.study_corr
corrs_path = args.corrs_path
output_path = args.out



searcher = Processor()
searcher(pickles_path_plt = pickles_path_plt,
         pickles_path_dt = pickles_path_dt,
         fill_number = fill_number, 
         year = year,
         corrs_path=corrs_path,
         store_path = output_path)
