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

args = parser.parse_args()

pickles_path = args.path
fill_number = args.fill

searcher = Processor()
searcher(pickles_path, fill_number)

