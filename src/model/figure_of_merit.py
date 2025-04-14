import json
import numpy as np
from poggers.io import read_fill
from poggers._utils import get_scan_timestamps
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import combinations
from pathlib import Path
from model.preprocessor import DifferencePreprocessor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from poggers.options import PoggerOptions
from adtk.detector import LevelShiftAD, PersistAD
from adtk.data import validate_series
import mplhep as hep
hep.style.use("CMS")


class Processor:
    def __init__(self):
        self.preprocessor = DifferencePreprocessor()

    def __call__(self, 
                 pickles_path: str,
                 fill_number: str,
                 year: int = 2024,
                 corrs_path: str = '/afs/cern.ch/user/f/fromeo/public/4Tomas/corrs_all.json',
                 store_path: str = '.') -> Any:

        self.year = year
        self.fill_number = fill_number
        save_path = f'{store_path}/results/{self.year}'
        if not os.path.exists(save_path) or not os.path.exists(f"{save_path}/plots"):
            os.makedirs(f"{save_path}/plots")
            os.makedirs(f"{save_path}/reports")
            
        self.save_path = save_path
        
        rates_df = self.read_pickles(pickles_path)
        #print(f"{rates_df.shape=} when prickles are read.")
        rates_df = self.filter_scanns(rates_df)
        #print(f"{rates_df.shape=} when scans are filtered.")
        useful_channels = self.read_useful_channels_corr(corrs_path)
        rates_df = rates_df.loc[:,useful_channels]
        rates_df_original = rates_df.copy()
        #print(f"{rates_df.shape} after filterin by corrected ones")
        ratio = self.get_cumulative_rates(rates_df)
        correlated_channels = self.filter_channels_by_ratio_correlation(ratio)
        #ratio = ratio.loc[:, correlated_channels]
        rates_df = rates_df.loc[:,correlated_channels]
        #print(f"{rates_df.shape} after filterin by correlation")
        true_ratio = self.get_cumulative_rates(rates_df)
        self.plot_results(rates_df_original, rates_df, true_ratio)
        self.save_results_JSON(correlated_channels)

    def save_results_JSON(self, correlated_channels):
        dictionary_channels = {}
        for channel in range(16):
            if channel in list(correlated_channels):
                dictionary_channels[channel] = True
            else:
                dictionary_channels[channel] = False
            path_file = f"{self.save_path}/reports/{self.fill_number}.json"
            with open(path_file, 'w') as json_file:
                json.dump(dictionary_channels, json_file, indent=4)
    
    def plot_results(self, 
                      rates_df_original,
                      rates_df,
                      true_ratio):
        fig, ax = plt.subplots(3, 1, figsize = (20, 16), sharex = True)

        hep.cms.label("Work in progress", rlabel = f'Fill {self.fill_number} ({self.year}, 13.6 TeV)', data = True, ax = ax[0])
        ax[0].plot(rates_df_original, ".",  label = [f"ch {ch}" for ch in rates_df_original.columns])
        ax[1].plot(rates_df, ".", label=[f"ch {ch}/ mean" for ch in rates_df.columns])
        ax[2].plot(true_ratio, ".", label=[f"ch {ch} / mean" for ch in true_ratio.columns])
        ax[0].set_ylabel(r'Lumi [$hz/\mu b$]')
        ax[1].set_ylabel(r'Lumi filtered[$hz/\mu b$]')
        ax[2].set_ylabel('Ratio [a.u.]')
        ax[2].set_xlabel('Time [$s$]')
        ax[1].set_xlabel(None)
        ax[0].legend(loc = 'upper right', bbox_to_anchor=(1.10, 1), fontsize = 16)
        ax[2].legend(loc = 'upper right', bbox_to_anchor=(1.15, 1), fontsize = 16)
        
        plt.savefig(f"{self.save_path}/plots/fill_{self.fill_number}.png")
        
    def read_pickles(self, 
                     pickles_path: str) -> pd.DataFrame:
        """
        Reads and processes pickle files containing data for a given year and fill number.
    
        Reads the fill data from the given pickle file path, removes unnecessary columns and 
        NaN values, converts the time column to a datetime format, and reindexes the DataFrame.
    
        Arguments:
        -----------
        year : int or str
            The year of the data to be processed.
        pickles_path : str or Path
            The path to the directory containing the pickle files.
        fill : int
            The fill number to be read and processed.
    
        Returns:
        --------
        pandas.DataFrame
            A cleaned and processed DataFrame indexed by time, with renamed columns from 0 to 15.
        """
        PoggerOptions().vdm_path = Path(f"/eos/cms/store/group/dpg_bril/comm_bril/{self.year}/vdm/")
        attrs, rates_df = read_fill(Path(pickles_path), 
                                    self.fill_number, 
                                    "plt",
                                    agg_per_ls=True,
                                    index_filter=(0.05,0.95))
        
        wanted_columns = ['time'] + [i for i in range(16)]
        rates_df = rates_df[[col for col in rates_df.columns if col in wanted_columns]]
        rates_df = rates_df.dropna()
        rates_df.time = pd.to_datetime(rates_df.time,
                                       unit = 's')
        rates_df.set_index('time',
                           inplace = True)
        rates_df.index.name = None
        #print(f"{rates_df.columns=}")
        rates_df.columns = [i for i in range(16)]
        return rates_df
    
    
    def filter_scanns(self, 
                      rates_df: pd.DataFrame)-> pd.DataFrame:
        """
        Filters out time intervals corresponding to scan timestamps within a time range slightly
        extended before and after each scan (by 3 minutes) from the given DataFrame.
    
        Parameters:
        -----------
        rates_df : pandas.DataFrame
            The DataFrame containing time-indexed rate data.
    
        Returns:
        --------
        pandas.DataFrame
            A DataFrame with scan-related time intervals removed.
        """
        stamps = get_scan_timestamps(self.fill_number)
        
        for stamp in stamps:
            start = pd.to_datetime(stamp[0], unit = 's')
            end = pd.to_datetime(stamp[1], unit = 's')
            
            start_minus = start - pd.DateOffset(minutes=3)
            end_plus = end + pd.DateOffset(minutes=3)
                
            rates_df = rates_df.drop(rates_df.loc[start_minus:end_plus].index)
    
        return rates_df
    
    def read_useful_channels_corr(self, corrections_path)-> list:
        """
        Loads the dictionary from the corrections JSON file. If the fill exists 
        in the JSON file, creates a dictionary where each channel is marked as 
        usable (True) if it's not NaN. If an error occurs (e.g., missing file or
        key), all 16 channels are assumed to be usable.
    
        Parameters:
        -----------
        corrs_path : str or Path
            The path to the JSON file containing correlation data.
    
        Returns:
        --------
        list
            A list of the usable channels for the corresponding fill.
        """
        try:
            with open(corrections_path, "r") as file:
                data = json.load(file)  # Convierte el JSON en un diccionario de Python
            
            dictio_channels = data[str(fill_number)]['eff']
            dictio_channels = {int(k): not np.isnan(v) for k, v in dictio_channels.items()}
        except Exception as e:
            dictio_channels = {int(k): True for k in range(16)}
    
        useful_channels = []
        for ch in dictio_channels.keys():
            if dictio_channels[ch] == True:
                useful_channels.append(ch)
        
        return useful_channels
    
    def get_cumulative_rates(self, rates)-> pd.DataFrame:
        """
        Computes the cumulative ratio of each channel with respect to its mean value.
        
        Parameters:
            rates (pd.DataFrame): DataFrame containing rate values for multiple channels.
        
        Returns:
            pd.DataFrame: DataFrame with the ratio of each value divided by the mean 
                          across rows.
        """
        chs = rates
        avg = chs.mean(axis=1)
        ratio = chs.div(chs.mean(axis=1), axis=0)
    
        return ratio
    
    def filter_channels_by_ratio_correlation(self, ratio: pd.DataFrame,
                                             threshold: float = 0.05) -> list:
        """
        Filters channels based on their correlation with others, selecting those with a median 
        correlation above a threshold.
        
        Parameters:
            ratio (pd.DataFrame): DataFrame containing ratio values for multiple channels.
            threshold (float, optional): Minimum median correlation value to consider a channel 
                                         usable. Default is 0.0.
        
        Returns:
            list: List of usable channel names that meet the correlation threshold.
        """
        corr = ratio.corr()
        fig, ax = plt.subplots(figsize = (18,15))
        hep.style.use('CMS')
        hep.cms.label("Work in progress", rlabel = f'Fill {self.fill_number} ({self.year}, 13.6 TeV)', data = True, ax = ax)
        sns.heatmap(corr, cmap = 'coolwarm', annot = True);
        plt.savefig(f"{self.save_path}/plots/{self.fill_number}_corr.png")
        corr['median'] = np.median(corr, axis = 1)
        usable_channels = list(corr[corr['median'] > threshold].index)
        return usable_channels
