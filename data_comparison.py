import os
import matplotlib.pyplot as plt
import numpy as np
import datetime

# Reuse your existing DataStats class
from data_stats import DataStats




class DatasetComparison:
    """
    Class to compare multiple datasets (data_types) and/or splits side by side.

    Example usage:
        cmp = DatasetComparison(
            var='prcp',
            data_types=['DANRA','ERA5'],
            splits=['train','valid'],
            path_data='...',
            transformations=['zscore'],
            create_figs=True,
            save_figs=True,
            show_figs=False
        )
        cmp.run()
    """

    def __init__(self,
                 var='prcp',
                 data_types=('DANRA','ERA5'),
                 splits=('train','valid'),
                 path_data='.',
                 transformations=None,
                 time_agg='daily',
                 create_figs=True,
                 save_figs=False,
                 show_figs=True,
                 save_stats=False,
                 fig_path='./figs_comparison/',
                 stats_path='./stats_comparison/',
                 n_workers=1):
        
        # Core parameters
        self.var = var
        # One or more data types, e.g. ['DANRA','ERA5']
        self.data_types = data_types if isinstance(data_types, (list, tuple)) else [data_types]
        # One or more splits, e.g. ['train','valid','test']
        self.splits = splits if isinstance(splits, (list, tuple)) else [splits]
        
        self.path_data = path_data
        self.transformations = transformations if transformations else []
        self.time_agg = time_agg
        self.create_figs = create_figs
        self.save_figs = save_figs
        self.show_figs = show_figs
        self.save_stats = save_stats
        self.fig_path = fig_path
        self.stats_path = stats_path
        self.n_workers = n_workers

        # We'll store loaded data for each combination in a dict
        # Key: (data_type, split), Value: (all_data_list, stats_dict)
        self.data_map = {}

        # Make sure figure/stats directories exist
        if not os.path.exists(self.fig_path):
            os.makedirs(self.fig_path, exist_ok=True)
        if not os.path.exists(self.stats_path):
            os.makedirs(self.stats_path, exist_ok=True)

    def load_all_data(self):
        """
        Loads data for each (data_type, split) combination using DataStats.
        Stores results in self.data_map.
        """
        for dt in self.data_types:
            for sp in self.splits:
                print(f"\n=== Loading data for data_type={dt}, split={sp} ===\n")
                
                # Create a temporary DataStats instance for each combination
                ds = DataStats(
                    var=self.var,
                    data_type=dt,
                    split_type=sp,
                    path_data=self.path_data,
                    transformations=self.transformations,
                    time_agg=self.time_agg,
                    create_figs=False,      # We'll do our own comparison plots
                    save_figs=False,        # Turn off in the sub-run to avoid duplication
                    show_figs=False,
                    save_stats=self.save_stats,
                    fig_path=self.fig_path,
                    stats_path=self.stats_path,
                    n_workers=self.n_workers
                )

                # Load data (no final visualize_data call here)
                all_data_list, stats_dict = ds.load_data(plot_cutout=False)
                
                # Optionally, if you want to do transformations or time-aggregate now, you can:
                # stats_dict = ds.aggregate_stats(stats_dict)
                # But let's just store them raw
                self.data_map[(dt, sp)] = (all_data_list, stats_dict)

    def visualize_comparison(self):
        """
        Compare the loaded datasets. We'll do a simple side-by-side histogram
        for the raw data distribution or a boxplot comparing daily means.
        
        Extend or customize as needed.
        """
        if not self.create_figs or not self.data_map:
            print("No figures to create or no data loaded.")
            return
        
        # EXAMPLE 1: Compare global data distributions (flattened) side-by-side
        fig, ax = plt.subplots(figsize=(8,5))
        
        # We'll plot a histogram for each (data_type, split) in a loop
        for idx, key in enumerate(self.data_map.keys()):
            (all_data_list, stats_dict) = self.data_map[key]
            data_flat = np.concatenate(all_data_list, axis=0).flatten()

            label_ = f"{key[0]}_{key[1]}"  # e.g. DANRA_train
            mu_ = np.mean(data_flat)
            std_ = np.std(data_flat)
            ax.hist(
                data_flat, bins=100, alpha=0.3,
                label=f"{label_} (μ={mu_:.2f}, σ={std_:.2f})"
            )
        
        ax.set_title(f"Comparison of Global Distributions (var={self.var})")
        ax.set_xlabel(self.var)
        ax.set_ylabel("Frequency")
        ax.legend()
        
        if self.save_figs:
            out_path = os.path.join(self.fig_path, f"{self.var}_comparison_distribution.png")
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
        if self.show_figs:
            plt.show(block=False)
            plt.pause(3)
            plt.close(fig)
        else:
            plt.close(fig)

        # EXAMPLE 2: Compare daily means from each stats_dict with boxplots
        fig, ax = plt.subplots(figsize=(8,5))

        all_labels = []
        all_means_for_boxplot = []
        
        # Grab the daily means from stats_dict["mean"]
        for key in self.data_map.keys():
            (all_data_list, stats_dict) = self.data_map[key]
            # Only plotting the daily mean distribution
            daily_means = stats_dict["mean"]
            
            all_means_for_boxplot.append(daily_means)
            all_labels.append(f"{key[0]}_{key[1]}")
        
        # Create boxplots side by side
        ax.boxplot(all_means_for_boxplot, labels=all_labels, showfliers=False)
        ax.set_title(f"Boxplot of Daily Means (var={self.var})")
        ax.set_ylabel("Daily Mean")
        ax.tick_params(axis='x', rotation=45)

        if self.save_figs:
            out_path = os.path.join(self.fig_path, f"{self.var}_comparison_boxplot.png")
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
        if self.show_figs:
            plt.show(block=False)
            plt.pause(3)
            plt.close(fig)
        else:
            plt.close(fig)

    def run(self):
        """
        Main entry point: load data for all combinations, then visualize comparison.
        """
        self.load_all_data()
        self.visualize_comparison()


if __name__ == "__main__":
    # Example usage
    cmp = DatasetComparison(
        var='prcp',
        data_types=['DANRA','ERA5'],
        splits=['train','valid'],
        path_data='/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/Data/Data_DiffMod/',
        transformations=['zscore'],
        create_figs=True,
        save_figs=True,
        show_figs=False,
        save_stats=False,
        fig_path='/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_figures/figs_comparison/',
        stats_path='/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_figures/figs_comparison',
        n_workers=1
    )
    cmp.run()

    