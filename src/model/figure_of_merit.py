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
from adtk.detector import LevelShiftAD


class Processor:
    def __init__(self):
        self.preprocessor = DifferencePreprocessor()
        
    def __call__(self, 
                 pickles_path: str,
                 fill_number: str,
                 study_corr: bool,
                 year: int = 2023,
                 get_ratio: bool = False,
                 corrs_path: str = '/eos/user/t/tatehort/nonlinearity/corrs_all.json',
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
        #print(f"{study_corr=}")
        #Set the year and the fill number as an attribute
        self.year = year
        PoggerOptions().vdm_path = Path(f"/eos/cms/store/group/dpg_bril/comm_bril/{self.year}/vdm/")
        self.fill_number = fill_number

        #Create the store path in case that it doesn't exist
        save_path = f'{store_path}/results/{self.year}'
        if not os.path.exists(save_path):
            os.makedirs(f"{save_path}/plots")
            os.makedirs(f"{save_path}/reports")
        
        #Read the fill from the pickle files
        attrs, rates_df = read_fill(Path(pickles_path), 
                                    fill_number, 
                                    "plt",
                                    agg_per_ls=True,
                                    remove_scans=True, 
                                    index_filter=(0.05,0.95))

        #Remove non-desired columns, drop na, and set the time as index
        rates_df.drop(columns = ['run', 'lsnum'], 
                      inplace = True)
        rates_df = rates_df.dropna()
        rates_df.time = pd.to_datetime(rates_df.time, unit = 's')
        rates_df.set_index('time', inplace = True)
        rates_df.index.name = None
        rates_df.columns = [i for i in range(16)]
        rates_df_original = rates_df.copy()
        
        #Create dictionary attribute with the information of anomalous channels 
        self.channels_dict = {i : True for i in range(16)} #True if the channel is non anomalous
        if self.year > 2022:
            self.channels_dict[6] = False
            self.channels_dict[8] = False
            self.channels_dict[9] = False
            self.channels_dict[13] = False
            #rates_df.drop(columns = [6, 8, 9, 13], inplace = True)


        channels_marked = self.read_non_usefull_channels_corr(corrs_path)
        self.channels_dict = {key: self.channels_dict[key] and channels_marked[key] for key in self.channels_dict}
        #look for not shifted channels in the luminosity
        
        try:
            channels = self.get_not_shifted_channels(rates_df)
        except:
            print(f"No enought instances to trust fill {self.fill_number}")
            dict_nans = {i : np.nan for i in range(16)}
            path_file_na = f"{save_path}/reports/{self.fill_number}_na.json"
            with open(path_file_na, 'w') as json_file:
                json.dump(dict_nans, json_file, indent=4)
            channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            
        for channel in range(16):
            if (self.channels_dict[channel] == False) and (channel in channels):
                channels.remove(channel)

        rates_df.drop(columns = [i for i in range(16) if i not in channels], 
                      inplace = True)

        #if rates_df.shape[0] < 200:
        #    
            
            
        
        #condition punctual, non-explained spikes in fill 7921
        if fill_number == 7921:
            rates_df = rates_df[np.sum(rates_df, axis = 1) < 11]

        
        #analyze ratios 
        if get_ratio:
            ratio, avg = self.get_cumulative_rates(rates_df, 
                                                   channels = channels)
            if type(ratio) != type(None):
                preprocessed_df = self.preprocess_data(ratio)
                self.plot_ratio_merit_fig(rates_df,#_original, 
                                          preprocessed_df, 
                                          ratio, 
                                          avg, 
                                          f"{save_path}/plots", 
                                          valid_channels = channels)
                if study_corr:
                    self.flag_channels_json(preprocessed_df, f"{save_path}")
                else:
                    #print("Isolation Forest not applied")
                    path_file = f"{save_path}/reports/{self.fill_number}.json"
                    with open(path_file, 'w') as json_file:
                        json.dump(self.channels_dict, json_file, indent=4)
                    
            else:
                if (rates_df.shape[0] == 0) or (rates_df.shape[1] < 2):
                    self.plot_nothing(save_path)
                    path_file = f"{save_path}/reports/{self.fill_number}.json"
                    with open(path_file, 'w') as json_file:
                        json.dump(self.channels_dict, json_file, indent=4)
                else:
                    preprocessed_df = self.preprocess_data(rates_df)#_original[[0,1,2,3,4,5,7,10,11,12,14,15]])
                    self.plot_rates_merit_fig(rates_df,#_original[[0,1,2,3,4,5,7,10,11,12,14,15]], 
                                              preprocessed_df, 
                                              f"{save_path}/plots")
                    
                    #print("Isolation Forest not applied, all channels anomalous")
                    path_file = f"{save_path}/reports/{self.fill_number}.json"
                    with open(path_file, 'w') as json_file:
                        json.dump(self.channels_dict, json_file, indent=4)
                

        #analize lumi
        else: #if not get ratio
            preprocessed_df = self.preprocess_data(rates_df)
                
            self.plot_rates_merit_fig(rates_df, 
                                      preprocessed_df, 
                                      f"{save_path}/plots")
            if study_corr:
                self.flag_channels_json(preprocessed_df, f"{save_path}")
            else:
                #print("Isolation Forest not applied")
                path_file = f"{save_path}/reports/{self.fill_number}.json"
                with open(path_file, 'w') as json_file:
                    json.dump(self.channels_dict, json_file, indent=4)

        print(self.channels_dict)

    
    def read_non_usefull_channels_corr(self, corrs_path):
        with open(corrs_path, "r") as file:
            datos = json.load(file)  # Convierte el JSON en un diccionario de Python
        
        channels = datos[str(self.fill_number)]['eff']
        channels = {int(k): not np.isnan(v) for k, v in channels.items()}
        return channels
    
    def get_not_shifted_channels(self, df):
        window = 40
        if df.shape[0] < 200:
            window = 10
        if self.year < 2023:
            channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        else:
            channels = [0, 1, 2, 3, 4, 5, 7, 10, 11, 12, 14, 15]
        ch_to_drop = []
        for channel in channels:
            detector = LevelShiftAD(c=3, side='negative', window=window)
            
            # Train the detector on the data
            detector.fit(df[channel])  # This is the training step
            anomalies = detector.detect(df[channel])
            anomalies.dropna(inplace=True)
            if any(anomalies):
                ch_to_drop.append(channel)
                self.channels_dict[channel] = False
        for channel in ch_to_drop:
            channels.remove(channel)
    
        return channels

    
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

        path_file = f"{save_path}/reports/{self.fill_number}.json"
        #scaler = StandardScaler()
        #normalized_data = scaler.fit_transform(preprocessed_df)

        normalized_data = preprocessed_df.values
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
            #if contamination == 0:
                #contamination = 0.001
        
        if contamination > 0.5:
            contamination = 0.4

        if contamination != 0:
            model = IsolationForest(contamination=contamination, random_state=42)
            column_scores = model.fit_predict(normalized_data.T)
            
            l_scores = [True if i == 1 else False for i in column_scores]
            l_channels = preprocessed_df.columns.to_list()
            
            dictio_channels = {l_channels[i]:l_scores[i] for i in range(len(l_channels))}
            for element in range(16):
                if element not in dictio_channels.keys():
                    dictio_channels[element] = False
            dictio_channels = {key: dictio_channels[key] for key in sorted(dictio_channels)}
            self.channels_dict = {key: self.channels_dict[key] and dictio_channels[key] for key in self.channels_dict}

        else:
            dictio_channels = self.channels_dict
        
        with open(path_file, 'w') as json_file:
            json.dump(dictio_channels, json_file, indent=4)


    
    def filter_channels(self, df: pd.DataFrame, channels: list[int]) -> list[int]:
        """Filters out channels where the average lumi is < 80% of the other channels' average."""
        chs = df[channels]
    
        # Compute mean lumi per channel over time
        avg_lumi_per_channel = chs.mean(axis=0)  # Average over time for each channel
    
        # Compute the average of the averages for all channels except each one in turn
        filtered_channels = []
        for ch in channels:
            remaining_channels = [c for c in channels if c != ch]
            if not remaining_channels:
                continue  # Avoid division by zero
    
            avg_other_channels = avg_lumi_per_channel[remaining_channels].mean()
    
            # Check if the channel is above 80% of the average of other channels
            if avg_lumi_per_channel[ch] >= 0.8 * avg_other_channels:
                filtered_channels.append(ch)
        #print(f"{filtered_channels=}")
        return filtered_channels


    def get_cumulative_rates(self, rates, channels: list[int] = [4, 11, 12]) -> plt.Figure:
        #hep.style.use("CMS")
        #fig, axs = plt.subplots(3, 1, figsize=(20, 6), sharex=True)
        #axs: List[plt.Axes] = axs.flatten()
    
        #attrs, df = read_fill(det, fill, "plt", agg_per_ls=True, index_filter=(0.05, 0.95))
        
        # Filter channels based on the 80% criterion
        valid_channels = self.filter_channels(rates, channels)
        if not valid_channels:
            print("No valid channels left after filtering!")
            channels_dict = {i : False for i in range(16)}
            return None, None
            #chs = rates[0, 1, 2, 3, 4, 5, 7, 10, 11, 12, 14, 15]
        #    return None  # Return empty figure
        else:
            chs = rates[valid_channels]
        
            #chs = rates[channels]
            avg = chs.mean(axis=1)
            #cumsum = avg.cumsum(skipna=True) * 23.31 / 1e9
            ratio = chs.div(chs.mean(axis=1), axis=0)
    
            return ratio, avg


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

    def study_shannel(
        self, data: pd.DataFrame, studied_channel: int, name="x"
    ) -> pd.DataFrame:
        """
        Studies the channel in the dataframe
        It first add the column "m_agg" to the dataframe, defined as the average of the channels that are not constant (i.e. those channels which less than 90% consecutive equal values)

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

    def plot_nothing(self, 
                     save_path):
        fig, ax = plt.subplots(3, 1, figsize = (18, 12), sharex = True)
        plt.savefig(f"{save_path}/plots/fill_{self.fill_number}.png")
        
    def plot_ratio_merit_fig(self, 
                             rates_df: pd.DataFrame,
                             processed_diff: pd.DataFrame,
                             ratio: pd.DataFrame,
                             avg: pd.Series,
                             save_path: str,
                             valid_channels = [1, 2, 3, 4, 5, 7, 10, 11, 12, 14, 15]):
        fig, ax = plt.subplots(3, 1, figsize = (18, 12), sharex = True)
        #for ch in processed_diff.columns:
            #processed_diff[ch] = processed_diff[ch] / processed_diff[ch].mean()

        avgs = ratio.mean(axis=0)
        stds = ratio.std(axis=0)
        rates_df = rates_df#[valid_channels]
        ax[0].plot(rates_df.index, rates_df, "o-", ms=2.5, label = [ch for ch in rates_df.columns])
        #ax[1].plot(ratio.index, ratio, "o", ms=2.5, label=[f"{ch}: {avgs[ch]:.3f} ({stds[ch]:.3f})" for ch in ratio.columns])
        ax[1].plot(ratio.index, ratio, "o-", ms=2.5, label=[ch for ch in ratio.columns])
        ax[2].plot(processed_diff.index, processed_diff, "o-", ms=2.5, label=[ch for ch in processed_diff.columns])
        ax[0].set_ylabel('rates')
        ax[1].set_ylabel('ratio')
        ax[2].set_ylabel('Norm. Processed diff.')
        ax[1].set_xlabel(None)
        ax[0].legend(loc ='best')
        ax[1].legend(loc ='best')
        ax[2].legend(loc ='best')

        #path_file = f"{save_path}/reports/{self.fill_number}.json"
        #save_path = f'results/{self.year}/{self.fill_number}'
        #save_path = 
        #if not os.path.exists(save_path):
        #    os.makedirs(save_path)
        plt.savefig(f"{save_path}/fill_{self.fill_number}.png")
        #preprocessed.to_csv(f"{save_path}/{fill_number}_preprocessed_data.csv")

    
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

        plt.savefig(f"{save_path}/fill_{self.fill_number}.png")
        #preprocessed.to_csv(f"{save_path}/{fill_number}_preprocessed_data.csv")