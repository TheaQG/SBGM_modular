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
import matplotlib
matplotlib.use('TkAgg') # For MacOS
# Set font params
plt.rcParams.update({'font.size': 11})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams.update({'axes.linewidth': 0.5})
plt.rcParams.update({'xtick.major.width': 0.5})
plt.rcParams.update({'ytick.major.width': 0.5})
plt.rcParams.update({'xtick.minor.width': 0.5})
plt.rcParams.update({'ytick.minor.width': 0.5})
plt.rcParams.update({'xtick.major.size': 2})
plt.rcParams.update({'ytick.major.size': 2})
plt.rcParams.update({'xtick.minor.size': 1})
plt.rcParams.update({'ytick.minor.size': 1})
plt.rcParams.update({'axes.labelsize': 11})
plt.rcParams.update({'legend.fontsize': 11})
plt.rcParams.update({'legend.frameon': False})
plt.rcParams.update({'legend.loc': 'upper right'})
plt.rcParams.update({'legend.handlelength': 1.5})
plt.rcParams.update({'legend.handletextpad': 1.0})
plt.rcParams.update({'legend.labelspacing': 0.4})
plt.rcParams.update({'legend.columnspacing': 1.0})
plt.rcParams.update({'lines.linewidth': 1.0})
plt.rcParams.update({'lines.markersize': 4})
plt.rcParams.update({'figure.dpi': 100})
plt.rcParams.update({'savefig.dpi': 300})
plt.rcParams.update({'savefig.bbox': 'tight'})

# matplotlib.use('Agg') # For Linux
from utils import str2bool, str2list_of_strings

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
    parser.add_argument('--save_figs', type=str2bool, default=True, help='Whether to save the figures')
    parser.add_argument('--show_figs', type=str2bool, default=True, help='Whether to show the figures')
    parser.add_argument('--save_stats', type=str2bool, default=False, help='Whether to save the statistics')
    parser.add_argument('--print_final_stats', type=str2bool, default=False, help='Whether to print the statistics')
    parser.add_argument('--fig_path', type=str, default='/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_figures/', help='The path to save the figures')
    parser.add_argument('--stats_path', type=str, default='/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_statistics', help='The path to save the statistics')
    parser.add_argument('--transformations', type=str2list_of_strings, default=None, help='List of transformations to apply to the data')#, choices=['zscore', 'log', 'log01', 'log_minus1_1', 'log_zscore'])
    parser.add_argument('--show_only_transformed', type=str2bool, default=False, help='Whether to show only the transformed data')
    parser.add_argument('--time_agg', type=str, default='daily', choices=['daily', 'weekly', 'monthly'], help='Time aggregation for statistics (daily, weekly, monthly, yearly)')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to use for CPU multiprocessing')
    
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
                print_final_stats=False,
                fig_path='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/',
                stats_path='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/',
                transformations=None,
                show_only_transformed=False,
                time_agg='daily',
                n_workers=1):

        self.var = var                  # The variable to compute statistics for (for now, prcp and temp)
        self.data_type = data_type      # The dataset to compute statistics for (DANRA or ERA5)
        self.split_type = split_type    # The split type of the data (train, val, test, all(DEVELOPMENT))
        self.path_data = path_data      # The path to the data
        self.create_figs = create_figs  # Whether to create figures
        self.show_figs = show_figs      # Whether to show the figures
        self.save_stats = save_stats    # Whether to save the statistics
        self.print_final_stats = print_final_stats  # Whether to print the statistics
        self.fig_path = fig_path        # The path to save the figures
        self.stats_path = stats_path    # The path to save the statistics
        self.transformations = transformations if transformations else [] # List of transformations to apply to the data
        self.show_only_transformed = show_only_transformed # Whether to show only the transformed data
        self.time_agg = time_agg        # Time aggregation for statistics (daily, weekly, monthly, yearly)
        self.n_workers = n_workers      # How many CPU processes to spawn for multiprocessing

        self.save_figs = save_figs      # Whether to save the figures
        if self.save_figs:
            self.transformation_str = '_'.join(self.transformations) if self.transformations else 'raw'


        # Set some plot and variable specific parameters
        if self.var == 'temp':
            self.var_str = 't'
            self.cmap = 'plasma'
            self.var_label = 'Temperature [C]'
        elif self.var == 'prcp':
            self.var_str = 'tp'
            self.cmap = 'inferno'
            self.var_label = 'Precipitation [mm]'
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

    def _process_single_file(self, zarr_group_img, file):
        '''
            Method called by each CPU worker to process a single file.
            1) Reads data from zarr
            2) Cuts out a specific region
            3) Basic variable correction
            4) Computes statistics
            returns: (stats_dict_for_this_file, data_array)
        '''
        # Print progress

        # Try read
        try:
            data = zarr_group_img[file][self.var_str][:].squeeze()
        except KeyError:
            data = None
            for fallback_key in ['arr_0', 'data']:
                if fallback_key in zarr_group_img[file]:
                    data = zarr_group_img[file][fallback_key][:].squeeze()
                    break
            if data is None:
                # Could not read data
                return None
        
        # Cutout
        data = data[self.cutout[0]:self.cutout[1], self.cutout[2]:self.cutout[3]]
        
        # Adjust for var
        if self.var == 'temp':
            data = data - 273.15
        elif self.var == 'prcp':
            data[data <= 0] = 1e-8
        
        # Compute log-min, log-max, log-mean, log-std for prcp
        if self.var == 'prcp':
            data_log = np.log(data)
            file_min_log = np.min(data_log)
            file_max_log = np.max(data_log)
            file_mean_log = np.mean(data_log)
            file_std_log = np.std(data_log)
        else:
            file_min_log = None
            file_max_log = None
            file_mean_log = None
            file_std_log = None

        # Compute stats
        mean, median, std_dev, variance, min_val, max_val, _ = self.compute_statistics(data)
        date_obj = self.parse_file_date(file)
        # Build a single-file stats dict
        single_stats = {
            "file": file,
            "date": date_obj if date_obj else file,
            "mean": mean,
            "median": median,
            "std_dev": std_dev,
            "variance": variance,
            "min": min_val,
            "max": max_val,
        }

        # also store min-log and max-log for prcp
        if self.var == 'prcp':
            single_stats["min_log"] = file_min_log
            single_stats["max_log"] = file_max_log
            single_stats["mean_log"] = file_mean_log
            single_stats["std_log"] = file_std_log

        return (single_stats, data)

    def load_data(self, plot_cutout=True):
        '''
            Method to load the data from the zarr files and compute statistics.
            Optionally plot the cutout for the first file.
            If n_workers > 1, parallelize the CPU processing of the files.
        '''
        path_data = os.path.join(
            self.path_data,
            f"data_{self.data_type}",
            f"size_{self.danra_size_str}",
            f"{self.var}_{self.danra_size_str}",
            "zarr_files"
        )
        data_dir_zarr = os.path.join(path_data, f"{self.split_type}.zarr")
        print(f"\nOpening zarr group: {data_dir_zarr}\n")
        zarr_group_img = zarr.open_group(data_dir_zarr, mode='r')
        files = list(zarr_group_img.keys())
        files.sort()

        # Prepare final stats_dict
        stats_dict = {
            "date": [],
            "mean": [],
            "median": [],
            "std_dev": [],
            "variance": [],
            "min": [],
            "max": [],
        }

        # If prcp, also add min-log, max-log
        if self.var == 'prcp':
            stats_dict["min_log"] = []
            stats_dict["max_log"] = []
            stats_dict["mean_log"] = []
            stats_dict["std_log"] = []
        all_data_list = []

        # Prepare figure and stats paths
        if not os.path.exists(self.fig_path):
            os.makedirs(self.fig_path, exist_ok=True)
        if not os.path.exists(self.stats_path):
            os.makedirs(self.stats_path, exist_ok=True)

        # (1) If user wants to parallelize, create a Pool, using imap to show progress
        if self.n_workers > 1:
            import multiprocessing
            from functools import partial

            worker_fn = partial(self._process_single_file, zarr_group_img)
            with multiprocessing.Pool(self.n_workers) as pool:
                # imap returns results in the order of fields, but yields them as they finish
                results_iter = pool.imap(worker_fn, files)

                # Loop over results
                for idx, out in enumerate(results_iter):
                    if out is None:
                        continue
                    single_stats, data = out

                    # Store daily stats
                    for k in ["date", "mean", "median", "std_dev", "variance", "min", "max"]:
                        stats_dict[k].append(single_stats[k])

                    # Also store min-log, max-log, mean-log, std-log for prcp
                    if self.var == 'prcp':
                        for k in ["min_log", "max_log", "mean_log", "std_log"]:
                            stats_dict[k].append(single_stats[k])

                    # Save data if we want global distribution
                    all_data_list.append(data)

                    # Print progress every 100 files
                    if idx % 100 == 0:
                        print(f"Processed {idx+1}/{len(files)} files...", flush=True)

                    # Optionally plot cutout for the first file
                    if idx == 0 and plot_cutout:
                        self._plot_cutout(data, single_stats["file"])
                        
        else:
            # Single process
            for idx, file in enumerate(files):
                out = self._process_single_file(zarr_group_img, file)
                if out is None:
                    continue
                single_stats, data = out

                # Store daily stats
                for k in ["date", "mean", "median", "std_dev", "variance", "min", "max"]:
                    stats_dict[k].append(single_stats[k])

                # Also store min-log, max-log, mean-log, std-log for prcp
                if self.var == 'prcp':
                    for k in ["min_log", "max_log", "mean_log", "std_log"]:
                        stats_dict[k].append(single_stats[k])

                # Save data if we want global distribution
                all_data_list.append(data)

                # Print progress every 100 files
                if idx % 100 == 0:
                    print(f"Processed {idx+1}/{len(files)} files...", flush=True)

                # Optionally plot cutout for the first file 
                if idx == 0 and plot_cutout:
                    self._plot_cutout(data, single_stats["file"])

        # Convert lists to arrays
        for k in ["mean","median","std_dev","variance","min","max"]:
            stats_dict[k] = np.array(stats_dict[k], dtype=float)

        # Also convert min-log, max-log, mean-log, std-log for prcp
        if self.var == 'prcp':
            for k in ["min_log", "max_log", "mean_log", "std_log"]:
                stats_dict[k] = np.array(stats_dict[k], dtype=float)

        # Print final stats
        if self.print_final_stats:
            print(f"\nFinal stats for {self.data_type} {self.var.capitalize()} {self.split_type}:")
            print(f"Mean: {np.mean(stats_dict['mean']):.2f}")
            print(f"Median: {np.mean(stats_dict['median']):.2f}")
            print(f"Std Dev: {np.mean(stats_dict['std_dev']):.2f}")
            print(f"Variance: {np.mean(stats_dict['variance']):.2f}")
            print(f"Min: {np.mean(stats_dict['min']):.2f}")
            print(f"Max: {np.mean(stats_dict['max']):.2f}")
            if self.var == 'prcp':
                print(f"Min Log: {np.mean(stats_dict['min_log']):.2f}")
                print(f"Max Log: {np.mean(stats_dict['max_log']):.2f}")
                print(f"Mean Log: {np.mean(stats_dict['mean_log']):.2f}")
                print(f"Std Log: {np.mean(stats_dict['std_log']):.2f}")

        # Optionally save stats (with date, mean, median, std_dev, variance, min, max - and min-log, max-log, mean-log, std-log for prcp)
        if self.save_stats:
            if not os.path.exists(self.stats_path):
                os.makedirs(self.stats_path)
            out_csv = os.path.join(self.stats_path, f"{self.var}_{self.split_type}_{self.data_type}_stats.csv")
            
            if self.var == 'prcp':
                np.savez(out_csv, date=stats_dict["date"], mean=stats_dict["mean"], median=stats_dict["median"],
                        std_dev=stats_dict["std_dev"], variance=stats_dict["variance"], min=stats_dict["min"],
                        max=stats_dict["max"], min_log=stats_dict["min_log"], max_log=stats_dict["max_log"],
                        mean_log=stats_dict["mean_log"], std_log=stats_dict["std_log"])
            else:
                np.savez(out_csv, date=stats_dict["date"], mean=stats_dict["mean"], median=stats_dict["median"],
                        std_dev=stats_dict["std_dev"], variance=stats_dict["variance"], min=stats_dict["min"],
                        max=stats_dict["max"])
                
            
        return all_data_list, stats_dict
    
    def _plot_cutout(self, data, file_label):
        """
            Helper to avoid duplicating the cutout plotting code.
        """
        fig, ax = plt.subplots(1,1, figsize=(5,5))
        im = ax.imshow(data, cmap=self.cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=self.var_label)
        ax.set_title(f"First cutout, {self.data_type} {self.var.capitalize()} {self.split_type}: {file_label}")
        ax.invert_yaxis()
        if self.save_figs:
            out_path = os.path.join(self.fig_path, f'{self.var}_{self.split_type}_{self.data_type}_{self.transformation_str}_cutout.png')
            fig.savefig(out_path, dpi=300, bbox_inches='tight')

        if self.show_figs:
            plt.show(block=False)
            plt.pause(2)
            plt.close(fig)
        else:
            plt.close(fig)

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
            elif transform == 'log_zscore':
                # Log transformation followed by z-score normalization
                data_log = np.log(data_1d + 1e-8)
                mu = np.mean(data_log)
                sigma = np.std(data_log) + 1e-8
                transformed_data['log_zscore'] = (data_log - mu) / sigma
            elif transform == 'log_minus1_1':
                # Log transformation to [-1,1] range
                data_log = np.log(data_1d + 1e-8)
                data_log_max = np.max(data_log)
                data_log_min = np.min(data_log)
                # Normalize to [-1,1]
                transformed_data['log_minus1_1'] = 2 * (data_log - data_log_min) / (data_log_max - data_log_min) - 1
            else:
                print(f"Transformation {transform} not implemented yet.")
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

        if self.var == 'prcp':
            agg_stats_dict["min_log"] = []
            agg_stats_dict["max_log"] = []
            agg_stats_dict["mean_log"] = []
            agg_stats_dict["std_log"] = []

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

            # Also for prcp, do average of min-log, max-log, mean-log, std-log
            if self.var == 'prcp':
                min_log_val = np.mean(stats_dict["min_log"][indices])
                max_log_val = np.mean(stats_dict["max_log"][indices])
                mean_log_val = np.mean(stats_dict["mean_log"][indices])
                std_log_val = np.mean(stats_dict["std_log"][indices])

            # For new date array, just use the first date in the bin
            d_first = date_list[indices[0]]
            # If weekly, label with year-week, if monthly, label with year-month, if yearly, just year
            if self.time_agg == 'weekly':
                agg_label = f"{key[0]}-W{key[1]}"
            elif self.time_agg == 'monthly':
                agg_label = f"{key[0]}-{key[1]:02d}"
            elif self.time_agg == 'yearly':
                agg_label = f"{key[0]}"

            # Append to new stats dict
            agg_stats_dict["date"].append(agg_label)
            agg_stats_dict["mean"].append(mean_val)
            agg_stats_dict["median"].append(median_val)
            agg_stats_dict["std_dev"].append(std_val)
            agg_stats_dict["variance"].append(var_val)
            agg_stats_dict["min"].append(min_val)
            agg_stats_dict["max"].append(max_val)

            # Also for prcp, append min-log, max-log, mean-log, std-log
            if self.var == 'prcp':
                agg_stats_dict["min_log"].append(min_log_val)
                agg_stats_dict["max_log"].append(max_log_val)
                agg_stats_dict["mean_log"].append(mean_log_val)
                agg_stats_dict["std_log"].append(std_log_val)

        # Convert to np.array
        for k in ["mean", "median", "std_dev", "variance", "min", "max"]:
            agg_stats_dict[k] = np.array(agg_stats_dict[k], dtype=float)

        # Also convert min-log, max-log, mean-log, std-log for prcp
        if self.var == 'prcp':
            for k in ["min_log", "max_log", "mean_log", "std_log"]:
                agg_stats_dict[k] = np.array(agg_stats_dict[k], dtype=float)

        return agg_stats_dict

    def visualize_data(self,
                        all_data_list,
                        stats_dict,
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

            # Use the date labels, but only show every 5th
            x_ticks = agg_stats_dict["date"]
            ax[2].set_xticks(x_vals[::5])
            ax[2].set_xticklabels(x_ticks[::5], rotation=45, ha='right')

            fig.tight_layout()

            if self.save_figs:
                out_path = os.path.join(self.fig_path, f'{self.var}_{self.split_type}_{self.data_type}_{self.time_agg}_{self.transformation_str}_timeseries.png')
                fig.savefig(out_path, dpi=300, bbox_inches='tight')
            if self.show_figs:
                # Show for 5 seconds
                plt.show(block=False)
                plt.pause(5)
                plt.close(fig)
            else:
                plt.close(fig)

        # 3) Global Distribution histograms (values)
        # For histograms of entire dataset, we need all in memory
        all_data_flat = np.concatenate(all_data_list, axis=0).flatten()
        
        if self.create_figs:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
            ax2 = ax1.twinx()  # second y-axis for transformed data

            max_freq_original = 0
            max_freq_transformed = 0
            color_orig = 'C0'
            color_trans = 'C1'
            # Plot original data on the left y-axis (ax1) with a unique color
            if not self.show_only_transformed:
                mu = np.mean(all_data_flat)
                std = np.std(all_data_flat)
                label_orig = f'Original, mu={mu:.2f}, std={std:.2f}'
                n1, bins1, patches1 = ax1.hist(
                    all_data_flat, bins=100, alpha=0.7, label=label_orig, color=color_orig
                )
                max_freq_original = n1.max()  # track max frequency for original distribution
                print(f"Max frequency for original: {max_freq_original}")
            min_global = np.min(all_data_flat)
            max_global = np.max(all_data_flat)

            min_x = min_global - 0.05 * np.abs(min_global)
            max_x = max_global + 0.05 * np.abs(max_global)
            # Plot transformed data on the right y-axis (ax2), track max frequencies
            transformed_data = self.apply_transformations(all_data_flat)
            for key, arr in transformed_data.items():
                mu_t = np.mean(arr)
                std_t = np.std(arr)
                max_t = np.max(arr)
                min_t = np.min(arr)
                label_t = f'{key.capitalize()} Transformed, mu={mu_t:.2f}, std={std_t:.2f}'
                n2, bins2, patches2 = ax2.hist(
                    arr, bins=100, alpha=0.7, label=label_t, color=color_trans
                )
                if n2.max() > max_freq_transformed:
                    max_freq_transformed = n2.max()
                    print(f"Max frequency for {key} transformed: {max_freq_transformed}")

                # If min/max is different, adjust x-limits
                if min_t < min_x:
                    min_x = min_t - 0.05 * np.abs(min_t)
                if max_t > max_x:
                    max_x = max_t + 0.05 * np.abs(max_t)

            # Apply y-limits based on max frequency for each distribution
            # if max_freq_original > 0:
            #     ax1.set_ylim([0, max_freq_original * 1.5])
            # if max_freq_transformed > 0:
            #     ax2.set_ylim([0, max_freq_transformed * 1.5])

            # Title and labels
            ax1.set_title(f"Global Distribution - {self.data_type} {self.var.capitalize()}, {self.split_type}")
            if self.var == 'temp':
                ax1.set_xlabel('Temperature [C]')
                ax1.set_xlim([min_x, max_x])
            elif self.var == 'prcp':
                ax1.set_xlabel('Precipitation [mm]')
                ax1.set_xlim([min_x, max_x])
                # IMPLEMENT MORE VARIABLES HERE
            
            
            # Y-labels and axis color to match histograms
            ax1.set_ylabel('Frequency (Original)', color=color_orig)
            ax1.tick_params(axis='y', labelcolor=color_orig)
            ax1.spines['left'].set_color(color_orig)

            ax2.set_ylabel('Frequency (Transformed)', color=color_trans)
            ax2.tick_params(axis='y', labelcolor=color_trans)
            ax2.spines['right'].set_color(color_trans)

            # Combine legends from both axes
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

            # If transformation any type of 'log', set right y-scale to log (search self.transformations strings for 'log')
            if any([t in self.transformations for t in ['log', 'log01', 'log_zscore', 'log_minus1_1']]):
                ax2.set_yscale('log')
                ax1.set_yscale('log')

            if self.save_figs:
                out_path = os.path.join(self.fig_path, f'{self.var}_{self.split_type}_{self.data_type}_{self.transformation_str}_all_data.png')
                fig.savefig(out_path, dpi=300, bbox_inches='tight')
            if self.show_figs:
                # Show for 5 seconds
                plt.show(block=False)
                plt.pause(5)
                plt.close(fig)
            else:
                plt.close(fig)
            plt.show()


        # 4) Global Distribution histograms (daily stats, mean, std_dev, etc.)
        # For histograms of entire dataset, we need all in memory
        if self.create_figs:
            n_plots = len(agg_stats_dict.keys()) - 1
            print(f"Plotting {n_plots} histograms for daily stats...")
            fig, ax = plt.subplots(2, n_plots//2, figsize=(12, 8))
            fig.suptitle(f'{self.data_type} {self.var.capitalize()} {self.split_type} {self.time_agg.capitalize()} Statistics', fontsize=14)
            axes = ax.flatten()
            for i, k in enumerate(agg_stats_dict.keys()):
                if k == 'date':
                    continue
                axes[i-1].hist(agg_stats_dict[k], bins=100, alpha=0.7)
                axes[i-1].set_title(f'{k.capitalize()} Distribution')
                axes[i-1].set_xlabel(k.capitalize())
                axes[i-1].set_ylabel('Frequency')

            fig.tight_layout()

            if self.save_figs:
                out_path = os.path.join(self.fig_path, f'{self.var}_{self.split_type}_{self.data_type}_{self.transformation_str}_{self.time_agg}_stats.png')
                fig.savefig(out_path, dpi=300, bbox_inches='tight')
            if self.show_figs:
                plt.show(block=False)
                plt.pause(5)
                plt.close(fig)
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