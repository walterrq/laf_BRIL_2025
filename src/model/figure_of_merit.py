import json
import numpy as np
from poggers.io import read_fill
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from model.preprocessor import DifferencePreprocessor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from poggers.options import PoggerOptions


class Processor:
    def __init__(self):
        self.preprocessor = DifferencePreprocessor()
        #print(mount_path)

    def __call__(self, 
                 pickles_path: str,
                 fill_number: str,
                 year: int = 2023,
                 store_path: str = '.') -> Any:
        """
        Executes the pipeline with the specified parameters.
    
        Args:
            pickles_path (str): Path to the directory or file containing the pickles data.
            fill_number (str): Identifier for the specific fill number to process.
            year (int, optional): The year associated with the data. Defaults to 2023.
            store_path: Path to the directory where the outputs are going to be stored, if doesn't
                        exist, will be created
    
        Returns:
            Any: The output of the pipeline, depending on its implementation.
        """
        self.year = year
        PoggerOptions().vdm_path = Path(f"/eos/cms/store/group/dpg_bril/comm_bril/{self.year}/vdm/")
        self.fill_number = fill_number
        
        
        attrs, rates_df = read_fill(Path(pickles_path), 
                                    fill_number, 
                                    "plt", 
                                    remove_scans=True, 
                                    index_filter=(0.05,0.95))
        
        rates_df.drop(columns = ['run', 'lsnum'], 
                      inplace = True)
        
        rates_df = rates_df.dropna()
        rates_df.time = pd.to_datetime(rates_df.time, unit = 's')
        rates_df.set_index('time', inplace = True)
        rates_df.index.name = None
        rates_df.columns = [i for i in range(16)]
        
        if fill_number == 7921:
            rates_df = rates_df[np.sum(rates_df, axis = 1) < 11]
            
        preprocessed_df = self.preprocess_data(rates_df)
        scaler = StandardScaler()
        
        if year > 2022:
            preprocessed_df.drop(columns = [6,8,9, 13], inplace =True)
            rates_df.drop(columns = [6,8,9,13], inplace =True)

        save_path = f'{store_path}/results/{self.year}'
        if not os.path.exists(save_path):
            os.makedirs(f"{save_path}/plots")
            os.makedirs(f"{save_path}/reports")
            
        self.plot_rates_merit_fig(rates_df, preprocessed_df, f"{save_path}/plots")
        self.flag_channels_json(preprocessed_df, f"{save_path}")
        
    
    def flag_channels_json(self, 
                           preprocessed_df: pd.DataFrame, 
                           save_path: str):
        """
        Analyzes the correlation among channels in a preprocessed DataFrame and flags them as 
        active or inactive based on their isolation scores, saving the results as a JSON file.

        This method normalizes the input DataFrame, computes a correlation matrix, and determines
        a contamination level for an Isolation Forest model. The model is used to flag channels
        (columns of the DataFrame) as either active (`True`) or inactive (`False`). The results are
        saved to a JSON file in the specified directory.

        Args:
            preprocessed_df (pd.DataFrame): 
                The preprocessed DataFrame where each column represents a channel, 
                and rows contain data points for that channel.
            
            save_path (str): 
                The directory path where the JSON file containing the flagged channels
                will be saved.
        """
        
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(preprocessed_df)

        normed_df = pd.DataFrame(normalized_data, columns = preprocessed_df.columns)
        normed_df.index = preprocessed_df.index
        self.plot_correlation_matrix(normed_df, f"{save_path}/plots")
        correlation_matrix = np.corrcoef(normed_df.T)
        mean_corr, max_mean_corr = correlation_matrix.mean(axis = 1).mean(), correlation_matrix.mean(axis = 1).max()
        
        if (mean_corr < 0.2) and (max_mean_corr < 0.25):
            contamination = 0.01
        else:
            contamination = 0
            for ind in range(correlation_matrix.shape[0]):
                if correlation_matrix[ind].mean() < 0.35:
                    contamination += 1
            contamination = contamination / correlation_matrix.shape[0]
            
        
        model = IsolationForest(contamination=contamination, random_state=42)
        column_scores = model.fit_predict(normalized_data.T)
        
        l_scores = [True if i == 1 else False for i in column_scores]
        l_channels = preprocessed_df.columns.to_list()
        
        dictio_channels = {l_channels[i]:l_scores[i] for i in range(len(l_channels))}
        for element in range(16):
            if element not in dictio_channels.keys():
                dictio_channels[element] = False
        dictio_channels = {key: dictio_channels[key] for key in sorted(dictio_channels)}
        
        
        path_file = f"{save_path}/reports/{self.fill_number}.json"

        #print(dictio_channels)
        with open(path_file, 'w') as json_file:
            json.dump(dictio_channels, json_file, indent=4)
        
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Studies the fill in the dataframe

        Args:
            data (pd.DataFrame): Dataframe with the data
        """
        dfs = []
        for channel in data.columns:
            dfs.append(self.study_shannel(data, channel, name=channel))
        return pd.concat(dfs, axis=1)

    def study_shannel(self, 
                      data: pd.DataFrame, 
                      studied_channel: int, 
                      name="x") -> pd.DataFrame:
        """
        Studies the channel in the dataframe. It first add the column "m_agg" to 
        the dataframe, defined as the average of the channels that are not constant 
        (i.e. those channels which less than 90% consecutive equal values)

        Args:
            data (pd.DataFrame): Dataframe with the data
            studied_channel (int): Channel to be studied
            plot (bool, optional): If True, plots the data. Defaults to True.
        """
        df = data.copy()
        df["m_agg"] = df[
            (   #c is a given channel and the for loop returns the list of channels that are not constant. The mean is calculated only from the colums corresponding to these channels.
                c
                for c in self.list_nonconstant_channels(df, studied_channel)
                if c != studied_channel
            )
        ].mean(axis=1)
        X = self.preprocessor(df, ["m_agg", studied_channel]) # ["m_agg", studied_channel] is the list of columns to preprocess
        return self.preprocessor.build_dataframe(df, X, name=name)

    def list_nonconstant_channels(
        self, data: pd.DataFrame, exclude: int = None
    ) -> list:
        """
        Lists the non-constant channels in the dataframe

        Args:
            data (pd.DataFrame): Dataframe with the data

        Returns:
            list: List of the non-constant channels
        """
        are_constant = [
            self._is_constant(c, data)
            for c in data.columns if str(c).isnumeric()
        ]
        if exclude:
            return [
                ch
                for ch, is_constant in zip(data.columns, are_constant)
                if ch != exclude and not is_constant
            ]
        return [ch for ch, c in zip(data.columns, are_constant) if not c]

    def _is_constant(self, channel: int, data: pd.DataFrame) -> bool:
        """
        Checks if the channel is non-constant in the dataframe

        Args:
            channel (int): Channel to be checked
            data (pd.DataFrame): Dataframe with the data

        Returns:
            bool: True if the channel is non-constant, False otherwise
        """
        x = data[channel].values
        diffs = np.diff(x)
        if len(diffs[diffs == 0]) / len(x) > 0.9:
            return True
        return False

    def plot_correlation_matrix(self,
                                df: pd.DataFrame,
                                save_path: str):
        correlation_matrix = np.corrcoef(df.T)
        fig, ax = plt.subplots(1,1,figsize = (12, 9))
        sns.heatmap(correlation_matrix, cmap="coolwarm", xticklabels=df.columns, yticklabels=df.columns, annot=True, ax = ax)
        plt.savefig(f"{save_path}/{self.fill_number}_m.png");
        
    
    def plot_rates_merit_fig(self, 
                             rates_df: pd.DataFrame,
                             processed_diff: pd.DataFrame, 
                             save_path: str):
        fig, ax = plt.subplots(2,1, figsize = (18, 8), sharex = True)
        for ch in rates_df.columns:
            processed_diff[ch] = processed_diff[ch] / processed_diff[ch].mean()
            ax[0].plot(rates_df[ch].index, rates_df[ch].values, linewidth = 2, label=None)
            sns.lineplot(data=processed_diff[ch], label=ch, ax = ax[1])
        ax[0].set_ylabel('rates')
        ax[1].set_ylabel('Norm. Processed diff.')
        ax[1].set_xlabel(None)

        plt.savefig(f"{save_path}/{self.fill_number}_rf.png")
        #preprocessed.to_csv(f"{save_path}/{fill_number}_preprocessed_data.csv")