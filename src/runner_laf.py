import argparse
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


parser = argparse.ArgumentParser(description="Run the Processor with given pickles_path and fill_number.")
parser.add_argument("--path", 
                    type=str,
                    help="Path to the pickles directory.")
parser.add_argument("--fill", 
                    type=int, 
                    help="Fill number to process.")
parser.add_argument("--year",
                    type=int,
                    help="Year of the fill to be analyzed")
parse.add_argument('--is_lumi',
                   dest="is_lumi",
                   type= bool,
                   default= False,
                   help = 'Lumi is being analyzed, or is it rates')
parser.add_argument("--out",
                    type=str,
                    default='.',
                    help="Output directory")

args = parser.parse_args()

pickles_path = args.path
fill_number = args.fill
is_lumi = args.is_lumi
year = args.year
output_path = args.out

searcher = Processor()
searcher(path = pickles_path, fill = fill_number, year = 2024, get_ratio = True, store_path = output_path)

