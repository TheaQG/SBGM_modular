'''
    Script to investigate the full dataset and get some statistics.
    Mainly to get an idea of the data distribution and the range of values
    and how it changes when data is scaled.

'''

import os
import zarr
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_modules import DANRA_Dataset_cutouts_ERA5_Zarr
from scipy.stats import boxcox, yeojohnson
from scipy.optimize import minimize_scalar
from utils import *

def data_stats_from_args():

    parser = argparse.ArgumentParser(description='Compute statistics of the data')
    parser.add_argument('--var', type=str, default='prcp', help='The variable to compute statistics for')
    parser.add_argument('--data_type', type=str, default='ERA5', help='The dataset to compute statistics for (DANRA or ERA5)')
    parser.add_argument('--split_type', type=str, default='train', help='The split type of the data (train, val, test)')
    parser.add_argument('--path_data', type=str, default='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/', help='The path to the data')
    parser.add_argument('--create_figs', type=str2bool, default=True, help='Whether to create figures')
    parser.add_argument('--save_figs', type=str2bool, default=False, help='Whether to save the figures')
    parser.add_argument('--show_figs', type=str2bool, default=True, help='Whether to show the figures')
    parser.add_argument('--fig_path', type=str, default='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/', help='The path to save the figures')
    
    args = parser.parse_args()

    data_stats(args)


def compute_statistics(data):
    # Compute statistics of 2D data
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    variance = np.var(data)
    min_temp = np.min(data)
    max_temp = np.max(data)
    percentiles = np.percentile(data, [25, 50, 75])

    
    return mean, median, std_dev, variance, min_temp, max_temp, percentiles


def data_stats(args):
    '''
        Based on arguments check if data exists and in the right format
        If not, create right format
    '''
    var = args.var
    data_type = args.data_type
    split_type = args.split_type


    if var == 'temp':
        var_str = 't'
        cmap = 'plasma'
    elif var == 'prcp':
        var_str = 'tp'
        cmap = 'inferno'

    danra_size_str = '589x789'

    cutout = [170, 170+180, 340, 340+180]

    data_dir = args.path_data #'/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/'
    # To HR data: Path + '/data_DANRA/size_589x789/' + var + '_' + danra_size_str +  '/zarr_files/train.zarr'
    PATH_HR = data_dir + 'data_' + data_type + '/size_' + danra_size_str + '/' + var + '_' + danra_size_str +  '/zarr_files/'
    # Path to DANRA data (zarr files), full danra, to enable cutouts
    data_dir_zarr = PATH_HR + split_type + '.zarr'

    zarr_group_img = zarr.open_group(data_dir_zarr, mode='r')

    files = list(zarr_group_img.keys())

    all_data = []

    # Create a df to store daily stats of the data (mean, median, std_dev, variance, min, max, percentiles)
    df_stats = pd.DataFrame(columns=['mean', 'median', 'std_dev', 'variance', 'min', 'max', 'percentiles'], index=files)

    if args.save_figs:
        if not os.path.exists(args.fig_path):
            os.makedirs(args.fig_path)


    print(f'\n\nNumber of files: {len(files)}')
    # Get the data (all files)
    for idx, file in enumerate(files):
        if idx % 10 == 0:
            print(f'\n\nProcessing File {idx+1}/{len(files)}')

        # Get the data - try different keys
        try:
            data = zarr_group_img[file][var_str][:].squeeze()
        except:
            try:
                data = zarr_group_img[file]['arr_0'][:].squeeze()
            except:
                try:
                    data = zarr_group_img[file]['data'][:].squeeze()
                except ValueError:
                    print(f'Error with key in file.')
                    

        data = data[cutout[0]:cutout[1], cutout[2]:cutout[3]]
        # Convert to Celsius if temperature
        if var == 'temp':
            data = data - 273.15
        all_data.append(data)


        # Compute statistics
        mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(data)
        df_stats.loc[file] = [mean, median, std_dev, variance, min_temp, max_temp, percentiles]

        if idx == 0 and args.create_figs:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5), layout='tight')
            ax.imshow(data, cmap=cmap)
            # Add colorbar
            cbar = plt.colorbar(ax.imshow(data, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            # Flip the y-axis to match the image
            ax.invert_yaxis()

            if args.save_figs:
                fig.savefig(args.fig_path + f'{var}_{split_type}_cutout_example.png', dpi=600, bbox_inches='tight')

    # # Plot time series of the statistics
    # if args.create_figs:
    #     fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    #     for i in range(6):
    #         ax_i = ax.flatten()[i]
    #         ax_i.plot(df_stats.iloc[:, i])
    #         ax_i.set_title(df_stats.columns[i])
    #         ax_i.set_xlabel('Time')
    #         ax_i.set_ylabel(df_stats.columns[i])

    #     fig.tight_layout()
            




    # Concatenate the data
    all_data = np.concatenate(all_data, axis=0).flatten()
    print(f'Number of data points: {len(all_data.flatten())}')

    # If precipitation, transform the data
    if var == 'prcp':
        # Yeo-Johnson transform
        # yeojohnson_transformed_data, yeojohnson_lambda = yeojohnson(all_data)
        # Replace negative values with 0 for boxcox
        all_data_no_zeros = all_data.copy()
        all_data_no_zeros[all_data_no_zeros <= 0] = 1e-4
        # boxcox_transformed_data, boxcox_lambda = boxcox(all_data_no_zeros)
        # Square root transform
        #all_data_sqrt = np.sqrt(all_data_no_zeros)
        
        if args.create_figs:
            # Plot different transforms
            fig, ax = plt.subplots(1, 1, figsize=(8,5))
            ax.hist(all_data_no_zeros, bins=100, alpha=0.7, label=f'Original, mu={np.mean(all_data_no_zeros):.2f}, std={np.std(all_data_no_zeros):.2f}')
            #ax.hist(all_data_sqrt, bins=100, alpha=0.7, label='Square Root')
            # ax.hist(yeojohnson_transformed_data, bins=100, alpha=0.7, label=f'Yeo-Johnson, mu={np.mean(yeojohnson_transformed_data):.2f}, std={np.std(yeojohnson_transformed_data):.2f}')
            # ax.hist(boxcox_transformed_data, bins=100, alpha=0.7, label=f'Box-Cox, mu={np.mean(boxcox_transformed_data):.2f}, std={np.std(boxcox_transformed_data):.2f}')
            # Set log scale for y-axis
            ax.set_yscale('log')
            

            if var == 'temp':
                ax.set_xlabel('Temperature [C]')
            elif var == 'prcp':
                ax.set_xlabel('Precipitation [mm]')
            ax.set_ylabel('Frequency')
            ax.legend()

            if args.save_figs:
                fig.savefig(args.fig_path + f'{var}_{split_type}_transformed_data.png', dpi=600, bbox_inches='tight')


    print('\n\nGLOBAL STATISTICS BEFORE TRANSFORMATION\n\n')
    # Compute global statistics
    mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(all_data)
    print('Global statistics')
    print('Mean: ', mean)
    print('Median: ', median)
    print('Standard Deviation: ', std_dev)
    print('Variance: ', variance)
    print('Min: ', min_temp)
    print('Max: ', max_temp)
    print('Percentiles: ', percentiles)

    # Pool all pixels together and plot
    all_data = all_data.flatten()

    # Simplest transformation(z-score for temp and min-max for prcp)
    if var == 'temp':
        all_data_simpleTransform = (all_data - mean) / (std_dev + 1e-8)
    elif var == 'prcp':
        all_data_simpleTransform = (all_data - min_temp) / (max_temp - min_temp + 1e-8)


    if args.create_figs:
        fig, ax = plt.subplots(1, 1, figsize=(8,5))
        ax.hist(all_data, bins=100, alpha=0.7, label=f'Original, mu={mean:.2f}, std={std_dev:.2f}')
        if var == 'temp':
            ax.set_xlabel('Temperature [C]')
        elif var == 'prcp':
            ax.set_xlabel('Precipitation [mm]')
        ax.set_ylabel('Frequency')
        ax.hist(all_data_simpleTransform, bins=100, alpha=0.7, label=f'Z-Score Normalized, mu={np.mean(all_data_simpleTransform):.2f}, std={np.std(all_data_simpleTransform):.2f}')
        # if var == 'prcp':
        #     mean_bc, std_bc = boxcox_transformed_data.mean(), boxcox_transformed_data.std()
        #     bc_zScore = (boxcox_transformed_data - mean_bc) / (std_bc + 1e-8)

        #     ax.hist(boxcox_transformed_data, bins=100, alpha=0.7, label=f'Box-Cox Transformed, mu={mean_bc:.2f}, std={std_bc:.2f}')
        #     ax.hist(bc_zScore, bins=100, alpha=0.7, label=f'Box-Cox Z-Score Normalized, mu={np.mean(bc_zScore):.2f}, std={np.std(bc_zScore):.2f}')

        #     ax.set_yscale('log')
        # ax.legend()
        # fig.tight_layout()

        if args.save_figs:
            fig.savefig(args.fig_path + f'{var}_{split_type}_all_data.png', dpi=600, bbox_inches='tight')



        # Plot the distributions of individual means, medians, std_devs, variances, mins, maxs
        n_plots = 6
        fig, ax = plt.subplots(2, n_plots//2, figsize=(10, 7))
        
        for i in range(n_plots):
            ax_i = ax.flatten()[i]
            ax_i.hist(df_stats.iloc[:, i], bins=100, alpha=0.7)
            ax_i.set_title(df_stats.columns[i])
            ax_i.set_xlabel(df_stats.columns[i])
            ax_i.set_ylabel('Frequency')

        fig.tight_layout()

        if args.save_figs:
            fig.savefig(args.fig_path + f'{var}_{split_type}_stats_distributions.png', dpi=600, bbox_inches='tight')



    print('\n\nGLOBAL STATISTICS AFTER TRANSFORMATIONS\n\n')

    # Compute global statistics on the transformed data
    if var == 'temp':
        mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(all_data_simpleTransform)
        print('\nGlobal statistics')
        print('Mean: ', mean)
        print('Median: ', median)
        print('Standard Deviation: ', std_dev)
        print('Variance: ', variance)
        print('Min: ', min_temp)
        print('Max: ', max_temp)
        print('Percentiles: ', percentiles)

    elif var == 'prcp':
        mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(all_data_simpleTransform)
        print('\nGlobal statistics')
        print('Mean: ', mean)
        print('Median: ', median)
        print('Standard Deviation: ', std_dev)
        print('Variance: ', variance)
        print('Min: ', min_temp)
        print('Max: ', max_temp)
        print('Percentiles: ', percentiles)

        # mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(yeojohnson_transformed_data)
        # print('\nYeo Johnson Transformed')
        # print('Mean: ', mean)
        # print('Median: ', median)
        # print('Standard Deviation: ', std_dev)
        # print('Variance: ', variance)
        # print('Min: ', min_temp)
        # print('Max: ', max_temp)
        # print('Percentiles: ', percentiles)

        # mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(boxcox_transformed_data)
        # print('\nBox-Cox Transformed')
        # print('Mean: ', mean)
        # print('Median: ', median)
        # print('Standard Deviation: ', std_dev)
        # print('Variance: ', variance)
        # print('Min: ', min_temp)
        # print('Max: ', max_temp)
        # print('Percentiles: ', percentiles)

        # mean, median, std_dev, variance, min_temp, max_temp, percentiles = compute_statistics(bc_zScore)
        # print('\nBox-Cox Z-Score Normalized')
        # print('Mean: ', mean)
        # print('Median: ', median)
        # print('Standard Deviation: ', std_dev)
        # print('Variance: ', variance)
        # print('Min: ', min_temp)
        # print('Max: ', max_temp)
        # print('Percentiles: ', percentiles)





    if args.show_figs:
        plt.show()


    return 




if __name__ == '__main__':
    data_stats_from_args()