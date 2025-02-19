'''
    Script to investigate the full dataset and get some statistics.
    Mainly to get an idea of the data distribution and the range of values
    and how it changes when data is scaled.

'''

import os
import zarr
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from utils import *

def data_stats_from_args():
    '''
        Function to get arguments from the command line and run the data_stats function
    '''
    parser = argparse.ArgumentParser(description='Compute statistics of the data')
    parser.add_argument('--var', type=str, default='prcp', help='The variable to compute statistics for')
    parser.add_argument('--data_type', type=str, default='DANRA', help='The dataset to compute statistics for (DANRA or ERA5)')
    parser.add_argument('--split_type', type=str, default='test', help='The split type of the data (train, val, test)')
    parser.add_argument('--path_data', type=str, default='/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/Data/Data_DiffMod/', help='The path to the data')
    parser.add_argument('--create_figs', type=str2bool, default=True, help='Whether to create figures')
    parser.add_argument('--save_figs', type=str2bool, default=False, help='Whether to save the figures')
    parser.add_argument('--show_figs', type=str2bool, default=True, help='Whether to show the figures')
    parser.add_argument('--save_stats', type=str2bool, default=False, help='Whether to save the statistics')
    parser.add_argument('--fig_path', type=str, default='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_figures/', help='The path to save the figures')
    parser.add_argument('--stats_path', type=str, default='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_statistics', help='The path to save the statistics')
    parser.add_argument('--transformations', nargs='+', default=None, help='List of transformations to apply to the data', choices=['zscore', 'log', 'log01'])
    parser.add_argument('--show_only_transformed', type=str2bool, default=False, help='Whether to show only the transformed data')
    parser.add_argument('--time_agg', type=str, default='daily', choices=['daily', 'weekly', 'monthly'], help='Time aggregation for statistics (daily, weekly, monthly, yearly)')
    
    args = parser.parse_args()

    data_stats = DataStats(**vars(args))
    data_stats.run()



class DataStats:
    '''
        Class for investigating, visualizing and computing statistics of the data.
        Contains methods to load the data, compute statistics, apply transformations and visualize the data.
        The point is to be able to investigate the data and get an idea of the distribution of the data, as well
        as how it changes when transformed using various methods.

        CURRENTLY IMPLEMENTED:
            - compute_statistics: Method to compute statistics of the data
            - load_data: Method to load the data
            - apply_transformations: Method to apply transformations to the data
            - visualize_data: Method to visualize the data and the statistics

        DEVELOPMENT: 
            - More types of data (water vapor, CAPE, etc.)
            - Possibility for analyzing all data at once (not just one split)
            - Add possibility for monthly/weekkly stats instead of daily timeseries
    '''
    def __init__(self,
                var='prcp',
                data_type='DANRA',
                split_type='valid',
                path_data='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/',
                create_figs=True,
                save_figs=False,
                show_figs=True,
                save_stats=False,
                fig_path='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/',
                stats_path='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/',
                transformations=None,
                show_only_transformed=False,
                time_agg='daily'):

        self.var = var                  # The variable to compute statistics for (for now, prcp and temp)
        self.data_type = data_type      # The dataset to compute statistics for (DANRA or ERA5)
        self.split_type = split_type    # The split type of the data (train, val, test, all(DEVELOPMENT))
        self.path_data = path_data      # The path to the data
        self.create_figs = create_figs  # Whether to create figures
        self.save_figs = save_figs      # Whether to save the figures
        self.show_figs = show_figs      # Whether to show the figures
        self.save_stats = save_stats    # Whether to save the statistics
        self.fig_path = fig_path        # The path to save the figures
        self.stats_path = stats_path    # The path to save the statistics
        self.transformations = transformations if transformations else [] # List of transformations to apply to the data
        self.show_only_transformed = show_only_transformed # Whether to show only the transformed data
        self.time_agg = time_agg        # Time aggregation for statistics (daily, weekly, monthly, yearly)

        # Set some plot and variable specific parameters
        if self.var == 'temp':
            self.var_str = 't'
            self.cmap = 'plasma'
        elif self.var == 'prcp':
            self.var_str = 'tp'
            self.cmap = 'inferno'
            # IMPLEMENT MORE VARIABLES HERE

        # Set some naming parameters
        self.danra_size_str = '589x789'
        # Hard-coded cutout for now
        self.cutout = [170, 170+180, 340, 340+180]

    def compute_statistics(self, data):
        '''
            Basic descriptive stats on a NumPy array.
        '''
        
        mean = np.mean(data)
        median = np.median(data)
        std_dev = np.std(data)
        variance = np.var(data)
        min_temp = np.min(data)
        max_temp = np.max(data)
        percentiles = np.percentile(data, [25, 50, 75])
        return mean, median, std_dev, variance, min_temp, max_temp, percentiles

    def parse_file_date(self, filename):
        '''
            Attempt to parse filename into a datetime object.
            Example file name format: 'tp_tot_20030101' (xxxxxx_YYYYMMDD)
        '''
        if len(filename) < 8:
            return None
        date_str = filename[-8:]
        try:
            return datetime.datetime.strptime(date_str, '%Y%m%d')
        except ValueError:
            return None

    def load_data(self,
                    plot_cutout=True,
                    ):
        '''
            Loads data from zarr, computes daily statistics and optionally plots and returns them.
            Minimizes storing full data in memory by default.
        '''
        # Set the specific path to data and load the data from the zarr files
        PATH_DATA = self.path_data + 'data_' + self.data_type + '/size_' + self.danra_size_str + '/' + self.var + '_' + self.danra_size_str +  '/zarr_files/'
        data_dir_zarr = PATH_DATA + self.split_type + '.zarr'

        print(f'\nOpening zarr group: {data_dir_zarr}')
        zarr_group_img = zarr.open_group(data_dir_zarr, mode='r')
        files = list(zarr_group_img.keys())
        files.sort() # Ensure sorted by date
        
        # Prepare dict to store stats.
        stats_dict = {
            "date": [],
            "mean": [],
            "median": [],
            "std_dev": [],
            "variance": [],
            "min": [],
            "max": [],
        } 


        # Create a dir to store figures
        if self.save_figs and not os.path.exists(self.fig_path):
            os.makedirs(self.fig_path)

        # Make a list to store all the data and a df to store the statistics. Might become huge.
        all_data_list = []

        # Loop through all files and process the data
        for idx, file in enumerate(files):
            if idx % 10 == 0:
                print(f'\n\nProcessing File {idx+1}/{len(files)}: {file}')
            try:
                data = zarr_group_img[file][self.var_str][:].squeeze()
            except KeyError:
                for fallback_key in ['arr_0', 'data']:
                    if fallback_key in zarr_group_img[file]:
                        data = zarr_group_img[file][fallback_key][:].squeeze()
                        break
                    else:
                        print(f'Error reading data from {file}. Skipping.')
                        continue

            # Cutout specific region                        
            data = data[self.cutout[0]:self.cutout[1], self.cutout[2]:self.cutout[3]]
            
            # Convert to Celsius if temperature, set zeros to small value if precipitation
            if self.var == 'temp':
                data = data - 273.15
            if self.var == 'prcp':
                data[data <= 0] = 1e-4
                # IMPLEMENT MORE VARIABLES HERE
                
            all_data_list.append(data)
            # Compute statistics for single file data
            mean, median, std_dev, variance, min_temp, max_temp, percentiles = self.compute_statistics(data)

            # Parse date from last 8 characters of filename (if None, just storing filename)
            date_obj = self.parse_file_date(file)
            stats_dict["date"].append(date_obj if date_obj else file)
            stats_dict["mean"].append(mean)
            stats_dict["median"].append(median)
            stats_dict["std_dev"].append(std_dev)
            stats_dict["variance"].append(variance)
            stats_dict["min"].append(min_temp)
            stats_dict["max"].append(max_temp)

            # Save a figure of the first cutout, if specified
            if idx == 0 and plot_cutout:
                fig, ax = plt.subplots(1, 1, figsize=(5, 5), layout='tight')
                im = ax.imshow(data, cmap=self.cmap)
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"First cutout of {self.data_type} {self.var} data\nFile: {file}")
                ax.invert_yaxis()
                if self.save_figs:
                    fig.savefig(self.fig_path + f'{self.var}_{self.split_type}_{self.data_type}_cutout_example.png', dpi=300, bbox_inches='tight')
                if self.show_figs:
                    # Show for 5 seconds
                    plt.show(block=False)
                    plt.pause(5)
                    plt.close(fig)

                else:
                    plt.close(fig)


        # Convert lists to arrays for convenience
        for k in stats_dict:
            stats_dict[k] = np.array(stats_dict[k], dtype=object if k == "date" else float)

        # Optionally save the statistics to a csv file
        if self.save_stats:
            if not os.path.exists(self.stats_path):
                os.makedirs(self.stats_path)
            csv_path = os.path.join(self.stats_path, f'{self.var}_{self.split_type}_{self.data_type}_stats.csv')
            # Save the statistics to a csv file WITHOUT pandas
            with open(csv_path, 'w') as f:
                f.write('file,mean,median,std_dev,variance,min,max\n')
                for i in range(len(files)):
                    f.write(f'{stats_dict["date"][i]},{stats_dict["mean"][i]},{stats_dict["median"][i]},{stats_dict["std_dev"][i]},{stats_dict["variance"][i]},{stats_dict["min"][i]},{stats_dict["max"][i]}\n')


        return all_data_list, stats_dict

    def apply_transformations(self,
                                data_1d):
        '''
            Method to apply transformations to the data.
            Currently implemented transformations:
                - Z-Score Normalization
                - Log Transformation

            DEVELOPMENT:
                - Add raw -> log -> [0,1] transformation
        '''

        transformed_data = {}
        for transform in self.transformations:
            if transform == 'zscore':
                mu = np.mean(data_1d)
                sigma = np.std(data_1d) + 1e-8
                transformed_data['zscore'] = (data_1d - mu) / sigma
            elif transform == 'log':
                transformed_data['log'] = np.log(data_1d + 1e-8)
            elif transform == 'log01':
                # Log transformation to [0,1] range
                data_log = np.log(data_1d + 1e-8)
                transformed_data['log01'] = (data_log - np.min(data_log)) / (np.max(data_log) - np.min(data_log))
            # Add more transformations as needed

        return transformed_data

    def aggregate_stats(self, stats_dict):
        '''
            Aggregate daily stats into weekly or monthly bins as requested
            NOTE: Not the ACTUAL complete stats, just the mean of the daily stats.
        '''
        if self.time_agg == 'daily':
            return stats_dict

        # Group daily stats into weekly or monthly bins
        agg_map = {} # (year,weekOrMonth) -> list of indices
        date_list = stats_dict["date"]

        # Build groups of daily indices
        for i, d_obj in enumerate(date_list):
            # If not a datetime, skip or try to parse again
            if not isinstance(d_obj, datetime.datetime):
                continue
            if self.time_agg == 'weekly':
                # isocalendar give (year, week, weekday) tuple
                y, w, _ = d_obj.isocalendar()
                key = (y, w)
            else:
                key = (d_obj.year, d_obj.month)

            if key not in agg_map:
                agg_map[key] = []
            agg_map[key].append(i)

        # Prepare new agg. dict
        agg_stats_dict = {
            "date": [],
            "mean": [],
            "median": [],
            "std_dev": [],
            "variance": [],
            "min": [],
            "max": [],
        }

        # For labeling the new x-axis
        for key in sorted(agg_map.keys()):
            indices = agg_map[key]

            # Now just do average of daily stats in that bin
            mean_val = np.mean(stats_dict["mean"][indices])
            median_val = np.mean(stats_dict["median"][indices])
            std_val = np.mean(stats_dict["std_dev"][indices])
            var_val = np.mean(stats_dict["variance"][indices])
            min_val = np.mean(stats_dict["min"][indices])
            max_val = np.mean(stats_dict["max"][indices])

            # For new date array, just use the first date in the bin
            d_first = date_list[indices[0]]
            # If weekly, label with year-week, if monthly, label with year-month
            if self.time_agg == 'weekly':
                agg_label = f"{key[0]}-W{key[1]}"
            else:
                agg_label = f"{key[0]}-{key[1]:02d}"

            # Append to new stats dict
            agg_stats_dict["date"].append(agg_label)
            agg_stats_dict["mean"].append(mean_val)
            agg_stats_dict["median"].append(median_val)
            agg_stats_dict["std_dev"].append(std_val)
            agg_stats_dict["variance"].append(var_val)
            agg_stats_dict["min"].append(min_val)
            agg_stats_dict["max"].append(max_val)

        # Convert to np.array
        for k in ["mean", "median", "std_dev", "variance", "min", "max"]:
            agg_stats_dict[k] = np.array(agg_stats_dict[k], dtype=float)

        return agg_stats_dict

    def visualize_data(self,
                        all_data_list,
                        stats_dict,
                        timeseries_grouping='daily' # 'daily', 'weekly', 'monthly', 'yearly'
                        ):
        '''
            Visualize either time series stats or distribution histograms.
        '''

        # 1) Possibly aggregate stats
        agg_stats_dict = self.aggregate_stats(stats_dict) if self.time_agg != 'daily' else stats_dict
        
        # 2) Time-series stats 


        # stats_dict has arrays for each field
        if self.create_figs and len(agg_stats_dict["date"]) > 1:
            fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
            fig.suptitle(f'{self.data_type} {self.var.capitalize()} {self.split_type} Statistics', fontsize=14)

            # Integer x-axis
            x_vals = np.arange(len(agg_stats_dict["date"])) # or keep as strings on x-ticks

            # Plot mean +/- std_dev
            ax[0].errorbar(x_vals, agg_stats_dict["mean"],
                           yerr=agg_stats_dict["std_dev"],
                           label='Mean', marker='.', lw=0.5,
                           fmt='-.', ecolor='gray', capsize=2)
            ax[0].set_ylabel('Mean')
            ax[0].legend()

            # Plot min
            ax[1].plot(x_vals, agg_stats_dict["min"], label='Min', marker='.', lw=0.5)
            ax[1].set_ylabel('Min')
            ax[1].legend()

            # Plot max
            ax[2].plot(x_vals, agg_stats_dict["max"], label='Max', marker='.', lw=0.5)
            ax[2].set_ylabel('Max')
            ax[2].set_xlabel(f'Time ({self.time_agg.capitalize()})')
            ax[2].legend()

            # Use the date labels
            ax[2].set_xticks(x_vals)
            ax[2].set_xticklabels(agg_stats_dict["date"], rotation=45, ha='right')

            fig.tight_layout()

            if self.save_figs:
                out_path = os.path.join(self.fig_path, f'{self.var}_{self.split_type}_{self.data_type}_time_series_stats.png')
                fig.savefig(out_path, dpi=300, bbox_inches='tight')
            if self.show_figs:
                # Show for 5 seconds
                plt.show(block=False)
                plt.pause(5)
                plt.close(fig)
            else:
                plt.close(fig)


        # 3) Global Distribution histograms
        # For histograms of entire dataset, we need all in memory
        all_data_flat = np.concatenate(all_data_list, axis=0).flatten()
        if self.create_figs:
            # Original data histogram
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            if not self.show_only_transformed:
                mu = np.mean(all_data_flat)
                std = np.std(all_data_flat)
                label_orig = f'Original, mu={mu:.2f}, std={std:.2f}'
                ax.hist(all_data_flat, bins=100, alpha=0.7, label=label_orig)

            # Transformed data histograms
            transformed_data = self.apply_transformations(all_data_flat)
            for key, arr in transformed_data.items():
                mu_t = np.mean(arr)
                std_t = np.std(arr)
                label_t = f'{key.capitalize()} Transformed, mu={mu_t:.2f}, std={std_t:.2f}'
                ax.hist(arr, bins=100, alpha=0.7, label=label_t)

            ax.set_title(f"Global Distrbution - {self.data_type} {self.var.capitalize()}, {self.split_type}")
            if self.var == 'temp':
                ax.set_xlabel('Temperature [C]')
            elif self.var == 'prcp':
                ax.set_xlabel('Precipitation [mm]')
                # IMPLEMENT MORE VARIABLES HERE
            ax.set_ylabel('Frequency')
            ax.legend()
            # If transformation 'log', set y-scale to log
            if 'log' in self.transformations or 'log01' in self.transformations:
                ax.set_yscale('log')

            if self.save_figs:
                out_path = os.path.join(self.fig_path, f'{self.var}_{self.split_type}_{self.data_type}_all_data.png')
                fig.savefig(out_path, dpi=300, bbox_inches='tight')
            if self.show_figs:
                plt.show()
                # Show for 5 seconds
                # plt.show(block=False)
                # plt.pause(5)
                # plt.close(fig)
            else:
                plt.close(fig)

    def run(self):
        '''
            Method to run the DataStats class.
            Calls the load_data, apply_transformations and visualize_data methods.
        '''
        all_data_list, stats_dict = self.load_data(plot_cutout=True)
        self.visualize_data(all_data_list, stats_dict)

    




if __name__ == '__main__':
    data_stats_from_args()



























class DataComparison:
    '''
        Class for comparing two datasets with possibility for transformations.
        The idea is to be able to compare the two datasets and see how they differ.
        This can be useful for comparing the two datasets and see how they differ in terms of the distribution of the data.
    '''
    def __init__(self,
                var='prcp',
                data_type1='DANRA',
                data_type2='ERA5',
                split_type='valid',
                path_data='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/',
                create_figs=True,
                save_figs=False,
                show_figs=True,
                fig_path='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/',
                transformations=None):

        self.var = var                  # The variable to compute statistics for (for now, prcp and temp)
        self.data_type1 = data_type1    # The first dataset to compare
        self.data_type2 = data_type2    # The second dataset to compare
        self.split_type = split_type    # The split type of the data (train, val, test, all(DEVELOPMENT))
        self.path_data = path_data      # The path to the data
        self.create_figs = create_figs  # Whether to create figures
        self.save_figs = save_figs      # Whether to save the figures
        self.show_figs = show_figs      # Whether to show the figures
        self.fig_path = fig_path        # The path to save the figures
        self.transformations = transformations if transformations else [] # List of transformations to apply to the data

        if self.var == 'temp':
            self.var_str = 't'
            self.cmap = 'plasma'
        elif self.var == 'prcp':
            self.var_str = 'tp'
            self.cmap = 'inferno'
            # IMPLEMENT MORE VARIABLES HERE

    def load_data(self):
        '''
            To load the data from the two datasets
        '''