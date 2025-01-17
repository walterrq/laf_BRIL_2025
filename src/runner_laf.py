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

def main():
    parser = argparse.ArgumentParser(description="Run the Processor with given pickles_path and fill_number.")
    parser.add_argument("pickles_path", 
                        type=str, 
                        default="/eos/user/t/tatehort/pylaf/laf/src/example_8880",
                        help="Path to the pickles directory.")
    parser.add_argument("fill_number", 
                        type=int, 
                        default=8880,
                        help="Fill number to process.")

    args = parser.parse_args()

    pickles_path = args.pickles_path
    fill_number = args.fill_number

    searcher = Processor()
    searcher(pickles_path, fill_number)

if __name__ == "__main__":
    main()