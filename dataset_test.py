'''
    Test the dataset class
    - Make a specific plotting function, using the layout developed in dataset_test.py, that I can employ across multiple files to have a consistent figure layout. Now, this is especially when I start generating results from my model (i.e. HR generated variable fields to compare with the HR truth). This means that the function must be able to either plot just the dataset (as for the previously written functionality), but also the dataset ALONG with the generated fields. 
- Employ the changes in the data_modules to the other scripts in my repo - especially in the main_sbgm.py, that I have written.

I want to start with developing the plotting function, and here is the codes that I need to align the plotting in:
dataset_test.py:

'''
import zarr 
import random
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import freeze_support

# Import DANRA dataset class from data_modules.py in src folder
from data_modules import DANRA_Dataset_cutouts_ERA5_Zarr_test
from special_transforms import *
from utils import *


# def launch_test_dataset_from_args():
#     '''
#         Launch the test dataset from the command line arguments
#     '''
#     parser = argparse.ArgumentParser(description='Test the dataset')
#     parser.add_argument('--var', type=str, default='prcp', help='The variable to use')
#     parser.add_argument('--img_dim', type=int, default=128, help='The image dimension')
#     parser.add_argument('--sample_w_lsm_topo', type=str2bool, default=True, help='Whether to sample with lsm and topo')
#     parser.add_argument('--sample_w_cutouts', type=str2bool, default=True, help='Whether to sample with cutouts')
#     parser.add_argument('--sample_w_cond_img', type=str2bool, default=True, help='Whether to sample with conditional images')
#     parser.add_argument('--sample_w_cond_season', type=str2bool, default=True, help='Whether to sample with conditional seasons')
#     parser.add_argument('--sample_w_sdf', type=str2bool, default=True, help='Whether to sample with sdf')
#     parser.add_argument('--scaling', type=str2bool, default=False, help='Whether to scale the data')
#     parser.add_argument('--scale_mean', type=float, default=8.69251, help='Mean of OG data distribution (Temperature [C])')
#     parser.add_argument('--scale_std', type=float, default=6.192434, help='STD of OG data distribution (Temperature [C])')
#     parser.add_argument('--scale_type_prcp', type=str, default='log_zscore', help='Type of scaling for precipitation', choices=['log_zscore', 'log_01', 'log', 'log_minus1_1', 'log_zscore'])
#     parser.add_argument('--scale_mean_log', type=float, default=-25.0, help='Mean of log-transformed data distribution (Precipitation [mm])')
#     parser.add_argument('--scale_std_log', type=float, default=10.0, help='STD of log-transformed data distribution (Precipitation [mm])')
#     parser.add_argument('--scale_min', type=float, default=0, help='Minimum of OG data distribution (Precipitation [mm])')
#     parser.add_argument('--scale_max', type=float, default=160, help='Maximum of OG data distribution (Precipitation [mm])')
#     parser.add_argument('--scale_min_log', type=float, default=-15, help='Minimum of log-transformed data distribution (Precipitation [mm])')
#     parser.add_argument('--scale_max_log', type=float, default=5, help='Maximum of log-transformed data distribution (Precipitation [mm])')
#     parser.add_argument('--buffer_frac', type=int, default=0.5, help='The percentage buffer size for precipition transformation')
#     parser.add_argument('--path_data', type=str, default='/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/Data/Data_DiffMod/', help='The path to the data')
#     parser.add_argument('--save_figs', type=str2bool, default=False, help='Whether to save the figures')
#     parser.add_argument('--show_figs', type=str2bool, default=True, help='Whether to show the figures')
#     parser.add_argument('--show_both_orig_scaled', type=str2bool, default=True, help='Whether to show both the original and scaled data in the same figure')
#     parser.add_argument('--show_ocean', type=str2bool, default=False, help='Whether to show the ocean')
#     parser.add_argument('--path_save', type=str, default='/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_figures/', help='The path to save the figures')
#     parser.add_argument('--cutout_domains', type=str2list, default=[170, 170+180, 340, 340+180], help='The cutout domains')
#     parser.add_argument('--topo_min', type=int, default=-12, help='The minimum value of the topological data')
#     parser.add_argument('--topo_max', type=int, default=330, help='The maximum value of the topological data')
#     parser.add_argument('--norm_min', type=int, default=0, help='The minimum value of the normalized topological data')
#     parser.add_argument('--norm_max', type=int, default=1, help='The maximum value of the normalized topological data')
#     parser.add_argument('--n_seasons', type=int, default=4, help='The number of seasons')
#     parser.add_argument('--n_gen_samples', type=int, default=3, help='The number of generated samples')
#     parser.add_argument('--num_workers', type=int, default=4, help='The number of workers')
    

#     args = parser.parse_args()


#     print(f'Scaling argument: {args.scaling}')
#     test_dataset(args)


def launch_test_dataset_from_args():
    '''
        Launch the test dataset from the command line arguments
    '''
    parser = argparse.ArgumentParser(description='Test the dataset')
    parser.add_argument('--hr_model', type=str, default='DANRA', help='The HR model to use')
    parser.add_argument('--hr_var', type=str, default='prcp', help='The HR variable to use')
    parser.add_argument('--hr_scaling_method', type=str, default='log_minus1_1', help='The scaling method for the HR variable')
    # Scaling params are provided as JSON-like strings
    parser.add_argument('--hr_scaling_params', type=str, default='{"glob_min": 0, "glob_max": 160, "glob_min_log": -20, "glob_max_log": 10, "glob_mean_log": -25.0, "glob_std_log": 10.0, "buffer_frac": 0.5}', #'{"glob_mean": 8.69251, "glob_std": 6.192434}', 
                        help='The scaling parameters for the HR variable, in JSON-like string format dict') #
    parser.add_argument('--lr_model', type=str, default='ERA5', help='The LR model to use')
    parser.add_argument('--lr_conditions', type=str2list, default=["prcp",
                                                                   "temp"],#,
                                                                #    "ewvf",#],
                                                                #    "nwvf"],
                        help='List of LR condition variables')
    parser.add_argument('--lr_scaling_methods', type=str2list, default=["log_minus1_1",#],
                                                                        "zscore"],
                                                                        # "zscore",#],
                                                                        # "zscore"],
                        help='List of scaling methods for LR conditions')
    # Scaling params are provided as JSON-like strings
    parser.add_argument('--lr_scaling_params', type=str2list, default=['{"glob_min": 0, "glob_max": 70, "glob_min_log": -10, "glob_max_log": 5, "glob_mean_log": -25.0, "glob_std_log": 10.0, "buffer_frac": 0.5}',
                                                                       '{"glob_mean": 8.69251, "glob_std": 6.192434}'],#,
                                                                    #    '{"glob_mean": 0.0, "glob_std": 500.0}',#],
                                                                    #    '{"glob_mean": 0.0, "glob_std": 500.0}'],
                        help='List of dicts of scaling parameters for LR conditions, in JSON-like string format dict')
    parser.add_argument('--lr_data_size', type=str2list, default=None, help='Target size for LR conditioning area as list, e.g. [589, 789]. If not provided and cutouts used, the whole LR image is used. If not provided and no cutouts used, the HR image size is used.')
    parser.add_argument('--lr_cutout_domains', type=str2list, default=None, help='Cutout domain for LR conditioning and geo variables as [x1, x2, y1, y2]. If not provided, HR cutout is used.')
    parser.add_argument('--force_matching_scale', type=str2bool, default=True, help='If True, force HR and LR images with the same variable to share the same color scale')
    parser.add_argument('--hr_dim', type=int, default=128, help='The image dimension')
    parser.add_argument('--sample_w_geo', type=str2bool, default=True, help='Whether to sample with lsm and topo')
    parser.add_argument('--sample_w_cutouts', type=str2bool, default=True, help='Whether to sample with cutouts')
    parser.add_argument('--sample_w_cond_season', type=str2bool, default=True, help='Whether to sample with conditional seasons')
    parser.add_argument('--sample_w_sdf', type=str2bool, default=True, help='Whether to sample with sdf')
    parser.add_argument('--scaling', type=str2bool, default=True, help='Whether to scale the data')
    parser.add_argument('--path_data', type=str, default='/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/Data/Data_DiffMod/', help='The path to the data')
    parser.add_argument('--save_figs', type=str2bool, default=False, help='Whether to save the figures')
    parser.add_argument('--specific_fig_name', type=str, default=None, help='If not None, saves figure with this name')
    parser.add_argument('--show_figs', type=str2bool, default=True, help='Whether to show the figures')
    parser.add_argument('--show_both_orig_scaled', type=str2bool, default=False, help='Whether to show both the original and scaled data in the same figure')
    parser.add_argument('--show_geo', type=str2bool, default=False, help='Whether to show the geo variables when plotting')
    parser.add_argument('--show_ocean', type=str2bool, default=False, help='Whether to show the ocean')
    parser.add_argument('--path_save', type=str, default='/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_figures/', help='The path to save the figures')
    parser.add_argument('--cutout_domains', type=str2list, default='170, 350, 340, 520', help='The cutout domains')
    parser.add_argument('--topo_min', type=int, default=-12, help='The minimum value of the topological data')
    parser.add_argument('--topo_max', type=int, default=330, help='The maximum value of the topological data')
    parser.add_argument('--norm_min', type=int, default=0, help='The minimum value of the normalized topological data')
    parser.add_argument('--norm_max', type=int, default=1, help='The maximum value of the normalized topological data')
    parser.add_argument('--n_seasons', type=int, default=4, help='The number of seasons')
    parser.add_argument('--n_gen_samples', type=int, default=3, help='The number of generated samples')
    parser.add_argument('--num_workers', type=int, default=4, help='The number of workers')
    

    args = parser.parse_args()
    test_dataset(args)


def test_dataset(args):
    # Use multiprocessing freeze_support() to avoid RuntimeError:
    freeze_support()

    # Print a large header for start of test
    print('\n\n')
    print('##################################################')
    print('#                                                #')
    print('#         Starting test of dataset class         #')
    print('#                                                #')
    print('##################################################')
    print('\n\n')

    # Set DANRA variable for use
    hr_var = args.hr_var
    if hr_var == 'temp':
        cmap_name = 'plasma'
        cmap_label = r'$^\circ$C'
    elif hr_var == 'prcp':
        cmap_name = 'inferno'
        cmap_label = 'mm'

    cmap_prcp = 'inferno'
    cmap_prcp_label = 'mm'
    cmap_temp = 'plasma'
    cmap_temp_label = r'$^\circ$C'
    cmap_nwvf = 'viridis'
    cmap_nwvf_label = 'm/s' # ?
    cmap_ewvf = 'viridis'
    cmap_ewvf_label = 'm/s' # ?
    cmap_topo = 'terrain'
    cmap_topo_label = ''
    cmap_sdf = 'coolwarm'
    cmap_sdf_label = ''
    cmap_lsm = 'binary'
    cmap_lsm_label = ''

    # Set units for hr and lr vars respectively
    prcp_units = 'mm'
    temp_units = r'$^\circ$C'
    ewvf_units = 'kg/m^2'
    nwvf_units = 'kg/m^2'

    if hr_var == 'prcp':
        hr_units = prcp_units
    elif hr_var == 'temp':
        hr_units = temp_units
    else:
        hr_units = 'Unknown'

    lr_units = []
    for cond in args.lr_conditions:
        if cond == 'prcp':
            lr_units.append(prcp_units)
        elif cond == 'temp':
            lr_units.append(temp_units)
        elif cond == 'nwvf':
            lr_units.append(nwvf_units)
        elif cond == 'ewvf':
            lr_units.append(ewvf_units)
        else:
            lr_units.append('Unknown')

    # Set size of DANRA images
    n_img_size = args.hr_dim
    image_size = (n_img_size, n_img_size)

    # LR data size and LR cutout domain. Convert lists (if provided) to tuples
    lr_data_size = tuple(args.lr_data_size) if args.lr_data_size is not None else None
    lr_cutout_domains = tuple(args.lr_cutout_domains) if args.lr_cutout_domains is not None else None


    # Set scaling to true or false
    scaling = args.scaling
    show_both_orig_scaled = args.show_both_orig_scaled
    force_matching_scale = args.force_matching_scale

    # Use hr_model and lr_model arguments to construct paths
    hr_model = args.hr_model # e.g. 'DANRA'
    lr_model = args.lr_model # e.g. 'ERA5'

    # Build HR zarr directory
    hr_zarr_dir = args.path_data + f'data_{hr_model}/size_589x789/{hr_var}_589x789/zarr_files/test.zarr'

    # For LR conditions, if more than one provided, build a dict mapping each LR condition to its zarr directory
    lr_conditions = args.lr_conditions
    lr_cond_dirs_zarr = {}
    for cond in lr_conditions:
        lr_path = args.path_data + f'data_{lr_model}/size_589x789/{cond}_589x789/zarr_files/test.zarr'
        print(f'Adding LR condition {cond} with path:\n\t{lr_path}')
        lr_cond_dirs_zarr[cond] = lr_path
    
    # Print what HR and LR data is being used
    print(f'\nHR data: {hr_model} {hr_var}, [{hr_units}]')# zarr directory: {hr_zarr_dir}')
    for cond, path in lr_cond_dirs_zarr.items():
        print(f'LR data: {lr_model} {cond}, [{lr_units[lr_conditions.index(cond)]}]')# zarr directory: {path}')
    print('\n')

    # Sampling options
    sample_w_geo = args.sample_w_geo
    sample_w_cutouts = args.sample_w_cutouts
    sample_w_cond_season = args.sample_w_cond_season
    sample_w_sdf = args.sample_w_sdf
    # Make sure that lsm is True if SDF is True
    if sample_w_sdf:
        print('\nSDF is True, setting lsm and topo to True\n')
        sample_w_geo = True
    # Set path to save figures
    PATH_SAVE = args.path_save

    # Determine number of samples from HR zarr group
    hr_zarr_group = zarr.open_group(hr_zarr_dir, mode='r')
    n_samples = len(list(hr_zarr_group.keys()))
    cache_size = n_samples
    
    if sample_w_cutouts:
        CUTOUT_DOMAINS = args.cutout_domains#[170, 170+180, 340, 340+180]
        # DOMAIN_2 = [80, 80+130, 200, 200+450]
    else:
        CUTOUT_DOMAINS = None
    
    # Load geo arrays if used
    if sample_w_geo:
        print(f'\nSampling with geo variables')
        geo_variables = ['lsm', 'topo']

        data_dir_lsm = args.path_data + 'data_lsm/truth_fullDomain/lsm_full.npz'
        data_dir_topo = args.path_data + 'data_topo/truth_fullDomain/topo_full.npz'

        data_lsm = np.flipud(np.load(data_dir_lsm)['data'])
        data_topo = np.flipud(np.load(data_dir_topo)['data'])


        if scaling:
            topo_min, topo_max = args.topo_min, args.topo_max
            norm_min, norm_max = args.norm_min, args.norm_max
            
            OldRange = (topo_max - topo_min)
            NewRange = (norm_max - norm_min)

            # Generating the new data based on the given intervals
            data_topo = (((data_topo - topo_min) * NewRange) / OldRange) + norm_min
    else:
        geo_variables = None
        data_lsm = None
        data_topo = None
    
    if scaling:
        print(f'\nScaling the data:')
        print(f'\tHR scaling method: {args.hr_scaling_method}')
        print(f'\tHR scaling params: \n\t\t{args.hr_scaling_params}\n')
        for i, cond in enumerate(lr_conditions):
            print(f'\tLR condition: {cond}')
            print(f'\tLR scaling method: {args.lr_scaling_methods[i]}')
            print(f'\tLR scaling params: \n\t\t{args.lr_scaling_params[i]}')
        print('\n')
    else:
        print(f'\nNot scaling the data - using raw values\n')

    # Parse the HR and LR scaling parameters from arguments
    hr_scaling_method = args.hr_scaling_method # Already string
    hr_scaling_params = ast.literal_eval(args.hr_scaling_params)
    lr_scaling_methods = args.lr_scaling_methods # Already list
    lr_scaling_params = [ast.literal_eval(param) for param in args.lr_scaling_params]

    # Initialize dataset with all options
    dataset = DANRA_Dataset_cutouts_ERA5_Zarr_test(hr_variable_dir_zarr = hr_zarr_dir,
                                                data_size = image_size,
                                                n_samples=n_samples,
                                                cache_size=cache_size,
                                                hr_variable=hr_var,
                                                hr_scaling_method=hr_scaling_method,
                                                hr_scaling_params=hr_scaling_params,
                                                lr_conditions=lr_conditions,
                                                lr_scaling_methods=lr_scaling_methods,
                                                lr_scaling_params=lr_scaling_params,
                                                lr_cond_dirs_zarr=lr_cond_dirs_zarr,
                                                geo_variables=geo_variables,
                                                lsm_full_domain=data_lsm,
                                                topo_full_domain=data_topo,
                                                shuffle=True,
                                                cutouts=sample_w_cutouts,
                                                cutout_domains=CUTOUT_DOMAINS if sample_w_cutouts else None,
                                                n_samples_w_cutouts=n_samples,
                                                sdf_weighted_loss=sample_w_sdf,
                                                scale=scaling,
                                                save_original=show_both_orig_scaled,
                                                conditional_seasons=sample_w_cond_season,
                                                n_classes=args.n_seasons,
                                                lr_data_size=lr_data_size,
                                                lr_cutout_domains=lr_cutout_domains,
                                                )

    # --- Build keys for plotting ---
    # HR key: append _hr suffix
    hr_key_plot = dataset.hr_variable + '_hr'
    # LR keys: for each LR condition, append _lr suffix
    lr_keys = [cond + '_lr' for cond in dataset.lr_conditions]

    # Create lists for scaled keys and (if requested) original keys
    scaled_keys = [hr_key_plot] + lr_keys
    original_keys = []
    if show_both_orig_scaled:
        original_keys = [hr_key_plot + '_original'] + [lr_key + '_original' for lr_key in lr_keys]
    
    # Final plotting order: scaled keys, then original keys (if any)
    if show_both_orig_scaled:
        subplot_keys = scaled_keys + original_keys
    else:
        subplot_keys = scaled_keys
    
    # Print keys
    print(f'\n\nKeys for plotting:')
    print(f'Subplot keys: {subplot_keys}\n')

    # Build subplot labels and colormap lists for the scaled keys
    scaled_labels = []
    scaled_cmaps = []

    for key in scaled_keys:
        if key.endswith('_hr'):
            scaled_labels.append(f'HR {hr_model} ({key})\nscaled')
            scaled_cmaps.append(cmap_name)
        else:
            # For LR, determine colormap based on the base condition
            base = key[:-3] # Remove '_lr' suffix
            if base == 'prcp':
                scaled_labels.append(f'LR {lr_model} ({key})\nscaled')
                scaled_cmaps.append(cmap_prcp)
            elif base == 'temp':
                scaled_labels.append(f'LR {lr_model} ({key})\nscaled')
                scaled_cmaps.append(cmap_temp)
            elif base == 'nwvf':
                scaled_labels.append(f'LR {lr_model} ({key})\nscaled')
                scaled_cmaps.append(cmap_nwvf)
            elif base == 'ewvf':
                scaled_labels.append(f'LR {lr_model} ({key})\nscaled')
                scaled_cmaps.append(cmap_ewvf)
            else:
                print(f'Warning: Unknown LR condition base: {base}')
                print(f'Using default colormap for key {key}')
                #   ADD MORE CONDITIONS HERE
                scaled_labels.append(f'LR {lr_model} ({key})\nscaled')
                scaled_cmaps.append(cmap_name)

    # Build labels and colormaps for original keys (if any)
    original_labels = []
    original_cmaps = []
    if show_both_orig_scaled:
        for key in original_keys:
            if key.endswith('_hr_original'):
                original_labels.append(f'HR {hr_model} ({key})\noriginal [{hr_units}]')
                original_cmaps.append(cmap_name)
            else:
                base = key[:-12] # Remove '_lr_original' suffix
                if base == 'prcp':
                    original_labels.append(f'LR {lr_model} ({key})\noriginal [{prcp_units}]')
                    original_cmaps.append(cmap_prcp)
                elif base == 'temp':
                    original_labels.append(f'LR {lr_model} ({key})\noriginal [{temp_units}]')
                    original_cmaps.append(cmap_temp)
                elif base == 'nwvf':
                    original_labels.append(f'LR {lr_model} ({key})\noriginal [{nwvf_units}]')
                    original_cmaps.append(cmap_nwvf)
                elif base == 'ewvf':
                    original_labels.append(f'LR {lr_model} ({key})\noriginal [{ewvf_units}]')
                    original_cmaps.append(cmap_ewvf)
                else:
                    print(f'Warning: Unknown LR condition base: {base}')
                    print(f'Using default colormap for key {key}')
                    #   ADD MORE CONDITIONS HERE
                    original_labels.append(f'LR {lr_model} ({key})\noriginal')
                    original_cmaps.append(cmap_name)                    

    # Final lists for scaled + original
    if show_both_orig_scaled:
        subplot_labels = scaled_labels + original_labels
        cmap_list = scaled_cmaps + original_cmaps
    else:
        subplot_labels = scaled_labels
        cmap_list = scaled_cmaps

    
    # Geo keys remain unchanged
    geo_keys = dataset.geo_variables
    # Append extra keys (geo and SDF)
    extra_keys = geo_keys.copy()
    extra_labels = []
    extra_cmaps = []
    for k in geo_keys:
        if k in geo_keys:
            if k == 'topo':
                extra_labels.append('Topography')
                extra_cmaps.append(cmap_topo)
            elif k == 'lsm':
                extra_labels.append('Land-sea mask')
                extra_cmaps.append(cmap_lsm)
    if sample_w_sdf:
        extra_keys.append('sdf')
        extra_labels.append('SDF')
        extra_cmaps.append(cmap_sdf)

    subplot_keys += extra_keys
    subplot_labels += extra_labels
    cmap_list += extra_cmaps

    n_subplots = len(subplot_keys)

    # --- Compute global min/max for HR/LR groups separately ---
    # Only use same colorbar if force_matching_scale is True
    sample_idxs = random.sample(range(len(dataset)), args.n_gen_samples)
    # For the scaled group
    if force_matching_scale:
        # Group scaled keys by base variable
        group_min = {}
        group_max = {}
        for key in scaled_keys:
            if key.endswith('_hr') or key.endswith('_lr'):
                base = key[:-3] # Remove '_hr' or '_lr' suffix
            else:
                base = key
            group_min.setdefault(base, []).append(float('inf'))
            group_max.setdefault(base, []).append(float('-inf'))
        # For each sample, update per-base min/max
        for idx in sample_idxs:
            sample = dataset[idx]
            for key in scaled_keys:
                if key in sample and sample[key] is not None:
                    base = key[:-3] if key.endswith('_hr') or key.endswith('_lr') else key
                    arr = sample[key].squeeze()
                    current_min = np.nanmin(arr)
                    current_max = np.nanmax(arr)
                    if base in group_min:
                        group_min[base].append(current_min)
                        group_max[base].append(current_max)
                    else:
                        print(f'Warning: Base {base} not found in group_min/max')
                        group_min[base] = [current_min]
                        group_max[base] = [current_max]
        # Final min/max per base
        final_scaled_min = {base: min(vals) for base, vals in group_min.items()}
        final_scaled_max = {base: max(vals) for base, vals in group_max.items()}
        scaled_global_min = {key: final_scaled_min[key[:-3]] if key.endswith('_hr') or key.endswith('_lr') else final_scaled_min[key] for key in scaled_keys}
        scaled_global_max = {key: final_scaled_max[key[:-3]] if key.endswith('_hr') or key.endswith('_lr') else final_scaled_max[key] for key in scaled_keys}
    else:
        # Compute global min/max for each scaled key individually
        scaled_global_min = {key: float('inf') for key in scaled_keys}
        scaled_global_max = {key: float('-inf') for key in scaled_keys}
        for idx in sample_idxs:
            sample = dataset[idx]
            for key in scaled_keys:
                if key in sample and sample[key] is not None:
                    arr = sample[key].squeeze()
                    scaled_global_min[key] = min(scaled_global_min[key], np.nanmin(arr))
                    scaled_global_max[key] = max(scaled_global_max[key], np.nanmax(arr))

    # For the original group
    if show_both_orig_scaled:
        if force_matching_scale:
            # Group original keys by base variable
            group_min_orig = {}
            group_max_orig = {}
            for key in original_keys:
                if key.endswith('_hr_original') or key.endswith('_lr_original'):
                    base = key[:-12] # Remove '_hr_original' or '_lr_original' suffix
                else:
                    base = key
                group_min.setdefault(base, []).append(float('inf'))
                group_max.setdefault(base, []).append(float('-inf'))
            for idx in sample_idxs:
                sample = dataset[idx]
                for key in original_keys:
                    if key in sample and sample[key] is not None:
                        base = key[:-12] if key.endswith('_hr_original') or key.endswith('_lr_original') else key
                        arr = sample[key].squeeze()
                        current_min = np.nanmin(arr)
                        current_max = np.nanmax(arr)
                        group_min_orig.setdefault(base, []).append(current_min)
                        group_max_orig.setdefault(base, []).append(current_max)
            # Final min/max per base
            final_orig_min = {base: min(vals) for base, vals in group_min_orig.items()}
            final_orig_max = {base: max(vals) for base, vals in group_max_orig.items()}
            orig_global_min = {key: final_orig_min[key[:-12]] if key.endswith('_hr_original') or key.endswith('_lr_original') else final_orig_min[key] for key in original_keys}
            orig_global_max = {key: final_orig_max[key[:-12]] if key.endswith('_hr_original') or key.endswith('_lr_original') else final_orig_max[key] for key in original_keys}
        else:
            # Compute global min/max for each original key individually
            orig_global_min = {key: float('inf') for key in original_keys}
            orig_global_max = {key: float('-inf') for key in original_keys}
            for idx in sample_idxs:
                sample = dataset[idx]
                for key in original_keys:
                    if key in sample and sample[key] is not None:
                        arr = sample[key].squeeze()
                        orig_global_min[key] = min(orig_global_min[key], np.nanmin(arr))
                        orig_global_max[key] = max(orig_global_max[key], np.nanmax(arr))





    # # Build subplot order - and add both scaled and original images if show_both_orig_scaled
    # subplot_keys = []
    # subplot_labels = []
    # cmap_list = []
    # cmap_label_list = []
    
    # # Add HR condition first (scaled with/without original, or unscaled)
    # if scaling:
    #     subplot_keys.append(hr_key_plot)
    #     subplot_labels.append(f'HR {hr_model} ({hr_key_plot})\nscaled')
    #     cmap_list.append(cmap_name)
    #     cmap_label_list.append(cmap_label)
    #     if show_both_orig_scaled:
    #         subplot_keys.append(hr_key_plot + '_original')
    #         subplot_labels.append(f'HR {hr_model} ({hr_key_plot})\noriginal')
    #         cmap_list.append(cmap_name)
    #         cmap_label_list.append(cmap_label)
    # else:
    #     subplot_keys.append(hr_key_plot)
    #     subplot_labels.append(f'HR {hr_model} ({hr_key_plot})\nunscaled')
    #     cmap_list.append(cmap_name)
    #     cmap_label_list.append(cmap_label)

    # for i, lr_key in enumerate(lr_keys):
    #     # Various colormaps for different LR conditions
    #     if lr_conditions[i] == 'prcp':
    #         cmap_use = cmap_prcp
    #         cmap_label_use = cmap_prcp_label
    #     elif lr_conditions[i] == 'temp':
    #         cmap_use = cmap_temp
    #         cmap_label_use = cmap_temp_label
    #     elif lr_conditions[i] == 'nwvf':
    #         cmap_use = cmap_nwvf
    #         cmap_label_use = cmap_nwvf_label
    #     elif lr_conditions[i] == 'ewvf':
    #         cmap_use = cmap_ewvf
    #         cmap_label_use = cmap_ewvf_label
    #     if scaling:
    #         subplot_keys.append(lr_key)
    #         subplot_labels.append(f'LR {lr_model} ({lr_key})\nscaled')
    #         cmap_list.append(cmap_use)
    #         cmap_label_list.append(cmap_label_use)
    #         if show_both_orig_scaled:
    #             subplot_keys.append(lr_key + '_original')
    #             subplot_labels.append(f'LR {lr_model} ({lr_key})\noriginal')
    #             cmap_list.append(cmap_use)
    #             cmap_label_list.append(cmap_label_use)
    #     else:
    #         subplot_keys.append(lr_key)
    #         subplot_labels.append(f'LR {lr_model} ({lr_key})\nunscaled')
    #         cmap_list.append(cmap_use)
    #         cmap_label_list.append(cmap_label_use)

    # # Then add geo variables
    # for k in geo_keys:
    #     if k == 'topo':
    #         subplot_keys.append(k)
    #         subplot_labels.append('Topography')
    #         cmap_list.append(cmap_topo)
    #         cmap_label_list.append(cmap_topo_label)
    #     elif k == 'lsm':
    #         subplot_keys.append(k)
    #         subplot_labels.append('Land-sea mask')
    #         cmap_list.append(cmap_lsm)
    #         cmap_label_list.append(cmap_lsm_label)

    # # If SDF is enabled, add SDF subplot
    # if sample_w_sdf:
    #     subplot_keys.append('sdf')
    #     subplot_labels.append('SDF')
    #     cmap_list.append(cmap_sdf)
    #     cmap_label_list.append(cmap_sdf_label)

    # n_subplots = len(subplot_keys)


    # # Compute global min/max for each subplot across all samples - if samples_on_same_scale is True
    # if samples_on_same_scale:
    #     global_min = {key: float('inf') for key in subplot_keys}
    #     global_max = {key: float('-inf') for key in subplot_keys}
    #     sample_idxs = random.sample(range(0, len(dataset)), args.n_gen_samples)

    #     for idx in sample_idxs:
    #         sample = dataset[idx]
    #         for key in subplot_keys:
    #             arr = sample[key].squeeze()
    #             global_min[key] = min(global_min[key], np.nanmin(arr))
    #             global_max[key] = max(global_max[key], np.nanmax(arr))
    # else:
    #     global_min = None
    #     global_max = None
    #     sample_idxs = random.sample(range(0, len(dataset)), args.n_gen_samples)

    if args.show_figs or args.save_figs:
        fig, axs = plt.subplots(args.n_gen_samples, n_subplots, figsize=(3*n_subplots, 2.5*args.n_gen_samples))
        # Set some properties for the boxplot
        flierprops = dict(marker='o', markerfacecolor='none', markersize=2,
                            linestyle='none', markeredgecolor='darkgreen', alpha=0.4)
        medianprops = dict(linestyle='-', linewidth=2, color='black')
        meanpointprops = dict(marker='x', markerfacecolor='firebrick',
                                markersize=5, markeredgecolor='firebrick')
        if scaling:
            hr_scaling_method = args.hr_scaling_method
            hr_scaling_params = ast.literal_eval(args.hr_scaling_params)
            fig.suptitle(f'Dataset with scaling, method: {hr_scaling_method}')#, params: {hr_scaling_params}')
        else:
            fig.suptitle(f'Dataset without scaling')

    print('\n\nDataset with options:\n')
    for i, idx in enumerate(sample_idxs):
        sample = dataset[idx]
        print(f"Sample '{idx}' keys: {list(sample.keys())}\n")

        if args.show_figs or args.save_figs:
            # go through all subplots
            for j, key in enumerate(subplot_keys):
                if key not in sample or sample[key] is None:
                    print(f"Warning: Key '{key}' not found in sample '{idx}'. Skipping...")
                    continue
                print(f"Plotting subplot '{key}'")
                img_data = sample[key].squeeze()

                # If show_ocean is False, only mask HR images using the HR mask ('lsm_hr')
                if not args.show_ocean:
                    if (key.endswith('_hr') or key.endswith('_hr_original')) and 'lsm_hr' in sample and sample['lsm_hr'] is not None:
                        mask = sample['lsm_hr'].squeeze()
                        img_data = np.where(mask < 1, np.nan, img_data)

                # Determine vmin and vmax
                if force_matching_scale:
                    if key in scaled_keys:
                        vmin = scaled_global_min.get(key, np.nanmin(img_data))
                        vmax = scaled_global_max.get(key, np.nanmax(img_data))
                    elif key in original_keys:
                        vmin = orig_global_min.get(key, np.nanmin(img_data))
                        vmax = orig_global_max.get(key, np.nanmax(img_data))
                    elif key in extra_keys:
                        vmin = np.nanmin(img_data)
                        vmax = np.nanmax(img_data)
                    else:
                        print(f'Warning: Key {key} not found in scaled or original keys')
                        vmin = np.nanmin(img_data)
                        vmax = np.nanmax(img_data)
                else:
                    vmin = np.nanmin(img_data)
                    vmax = np.nanmax(img_data)

                # If more than one sample, use axs[i, j], otherwise axs[j]
                ax = axs[i, j] if args.n_gen_samples > 1 else axs[j]
                im = ax.imshow(img_data, cmap=cmap_list[j], vmin=vmin, vmax=vmax, interpolation='nearest')
                ax.invert_yaxis()
                ax.set_xticks([])
                ax.set_yticks([])

                # Add colorbar and boxplot (for selected images)
                divider = make_axes_locatable(ax)

                # For HR/LR images (and their originals), add boxplot next to colorbar (if key has '_lr' or '_hr' suffix)
                if key.endswith('_lr') or key.endswith('_hr') or key.endswith('_lr_original') or key.endswith('_hr_original'):
                    print(f'Adding boxplot for key {key}')
                    bax = divider.append_axes("right", size="10%", pad=0.1)
                    cax = divider.append_axes("right", size="5%", pad=0.1)
                    
                    # To avoid warning of dtype mismatch
                    if isinstance(img_data, torch.Tensor):
                        mask = ~torch.isnan(img_data)
                        mask = mask.bool() # Ensure boolean dtype
                        img_data_bp = img_data[mask].flatten().cpu().numpy()
                    else:
                        img_data_bp = img_data[~np.isnan(img_data)].flatten()

                    bax.boxplot(img_data_bp,
                                vert=True,
                                widths=2,
                                patch_artist=True,
                                showmeans=True,
                                meanline=False,
                                flierprops=flierprops,
                                medianprops=medianprops,
                                meanprops=meanpointprops)
                    bax.set_xticks([])
                    bax.set_yticks([])
                    bax.set_frame_on(False)

                else:
                    # Only colorbar
                    cax = divider.append_axes("right", size="5%", pad=0.1) # Space for colorbar
                fig.colorbar(im, cax=cax)#, label=cmaps_label[j])

                if i == 0:
                    ax.set_title(subplot_labels[j], fontsize=10)
            fig.tight_layout()
    
    # Save the figure
    if args.save_figs:
        print(f'Saving figure to {PATH_SAVE}')
        if args.specific_fig_name is not None:
            fn = args.specific_fig_name
        else:
            if scaling:
                fn = f'Dataset_{hr_var}_{hr_scaling_method}'
            else:
                fn = f'Dataset_{hr_var}_unscaled'
        fig.savefig(PATH_SAVE + fn + '.png', dpi=300, bbox_inches='tight')
        print(f'with name {fn}')

    if args.show_figs:
        plt.show()

    print('\n\n')


# def test_dataset(args):
#     # Use multiprocessing freeze_support() to avoid RuntimeError:
#     freeze_support()


#     # Set DANRA variable for use
#     var = args.var
#     if var == 'temp':
#         cmap_name = 'plasma'
#         cmap_label = r'$^\circ$C'
#     elif var == 'prcp':
#         cmap_name = 'inferno'
#         cmap_label = 'mm'


#     # Set size of DANRA images
#     n_danra_size = args.img_dim
#     # Set DANRA size string for use in path
#     danra_size_str = str(n_danra_size) + 'x' + str(n_danra_size)
#     # Define different samplings
#     sample_w_lsm_topo = args.sample_w_lsm_topo
#     sample_w_cutouts = args.sample_w_cutouts
#     sample_w_cond_img = args.sample_w_cond_img
#     sample_w_cond_season = args.sample_w_cond_season
#     sample_w_sdf = args.sample_w_sdf

#     # Make sure that lsm is True if SDF is True
#     if sample_w_sdf:
#         print('\nSDF is True, setting lsm and topo to True\n')
#         sample_w_lsm_topo = True

#     # Set scaling to true or false
#     scaling = args.scaling
#     show_both_orig_scaled = args.show_both_orig_scaled
    
#     # Set paths to zarr data
#     data_dir_danra_zarr = args.path_data + 'data_DANRA/size_589x789/' + var + '_589x789/zarr_files/test.zarr'
#     data_dir_era5_zarr = args.path_data + 'data_ERA5/size_589x789/' + var + '_589x789/zarr_files/test.zarr'

#     # Set path to save figures
#     PATH_SAVE = args.path_save

#     # Set number of samples and cache size
#     danra_w_cutouts_zarr_group = zarr.open_group(data_dir_danra_zarr, mode='r')
#     n_samples = len(list(danra_w_cutouts_zarr_group.keys()))#365
#     cache_size = n_samples
#     # Set image size
#     image_size = (n_danra_size, n_danra_size)
    
#     CUTOUTS = args.sample_w_cutouts
#     if CUTOUTS:
#         CUTOUT_DOMAINS = args.cutout_domains#[170, 170+180, 340, 340+180]
#     # DOMAIN_2 = [80, 80+130, 200, 200+450]
#     # Set paths to lsm and topo if used
#     if sample_w_lsm_topo:
#         data_dir_lsm = args.path_data + 'data_lsm/truth_fullDomain/lsm_full.npz'
#         data_dir_topo = args.path_data + 'data_topo/truth_fullDomain/topo_full.npz'

#         data_lsm = np.flipud(np.load(data_dir_lsm)['data'])
#         data_topo = np.flipud(np.load(data_dir_topo)['data'])

#         if scaling:
#             topo_min, topo_max = args.topo_min, args.topo_max
#             norm_min, norm_max = args.norm_min, args.norm_max
            
#             OldRange = (topo_max - topo_min)
#             NewRange = (norm_max - norm_min)

#             # Generating the new data based on the given intervals
#             data_topo = (((data_topo - topo_min) * NewRange) / OldRange) + norm_min
#     else:
#         data_lsm = None
#         data_topo = None
    
#     if scaling:
#         print(f'\nScaling data')
#         if var == 'temp':
#             print(f'\tMean: {args.scale_mean}, std: {args.scale_std}')
#         elif var == 'prcp':
#             print(f'\tMin: {args.scale_min}, max: {args.scale_max}\n')


#     # Initialize dataset with all options
#     dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_danra_zarr, 
#                                     image_size, 
#                                     n_samples, 
#                                     cache_size, 
#                                     variable = var,
#                                     shuffle=True, 
#                                     cutouts = sample_w_cutouts, 
#                                     cutout_domains = CUTOUT_DOMAINS,
#                                     n_samples_w_cutouts = n_samples, 
#                                     lsm_full_domain = data_lsm,
#                                     topo_full_domain = data_topo,
#                                     sdf_weighted_loss = args.sample_w_sdf,
#                                     scale=scaling,
#                                     save_original=show_both_orig_scaled,
#                                     scale_mean=args.scale_mean,
#                                     scale_std=args.scale_std,
#                                     scale_type_prcp=args.scale_type_prcp,
#                                     scale_min=args.scale_min,
#                                     scale_max=args.scale_max,
#                                     scale_min_log=args.scale_min_log,
#                                     scale_max_log=args.scale_max_log,
#                                     scale_mean_log=args.scale_mean_log,
#                                     scale_std_log=args.scale_std_log,
#                                     buffer_frac=args.buffer_frac,
#                                     conditional_seasons=sample_w_cond_season,
#                                     conditional_images=sample_w_cond_img,
#                                     cond_dir_zarr = data_dir_era5_zarr,
#                                     n_classes=args.n_seasons
#                                     )

#     # Check if datasets work
#     # Get sample images


#     n_samples = args.n_gen_samples
#     idxs = random.sample(range(0, len(dataset)), n_samples)

#     # Build a fixed order of subplots:
#     # ['img', 'img_cond', 'img_original', 'img_cond_original'] if show_both_orig_scaled
#     # or ['img', 'img_cond'] if show_both_orig_scaled = False
#     # Then topo/lsm/sdf afterwards if applicable.

#     img_strs = []
#     cmaps = []
#     cmaps_label = []
#     img_labels = []
#     n_subplots = 0

#     # Always 'img'
#     n_subplots += 1
#     img_strs.append('img')
#     cmaps.append(cmap_name)
#     cmaps_label.append(cmap_label)
#     img_labels.append('HR DANRA, scaled' if scaling else 'HR DANRA, unscaled')

#     # 'img_cond' if sample_w_cond_img = True
#     if sample_w_cond_img:
#         n_subplots += 1
#         img_strs.append('img_cond')
#         cmaps.append(cmap_name)
#         cmaps_label.append(cmap_label)
#         img_labels.append('LR ERA5, scaled' if scaling else 'LR ERA5, unscaled')

#     # If show_both_orig_scaled, we then add 'img_original' 
#     # (and 'img_cond_original' if sample_w_cond_img)
#     if show_both_orig_scaled and scaling:
#         n_subplots += 1
#         img_strs.append('img_original')
#         cmaps.append(cmap_name)
#         cmaps_label.append(cmap_label)
#         img_labels.append('HR DANRA, original')

#         if sample_w_cond_img:
#             n_subplots += 1
#             img_strs.append('img_cond_original')
#             cmaps.append(cmap_name)
#             cmaps_label.append(cmap_label)
#             img_labels.append('LR ERA5, original')

#     # Finally, add topo/lsm/sdf if they exist:
#     if sample_w_lsm_topo:
#         n_subplots += 2
#         img_strs.append('topo')
#         cmaps.append('terrain')
#         cmaps_label.append('')
#         img_labels.append('Topography')

#         img_strs.append('lsm')
#         cmaps.append('binary')
#         cmaps_label.append('')
#         img_labels.append('Land-sea mask')

#     if sample_w_sdf:
#         n_subplots += 1
#         img_strs.append('sdf')
#         cmaps.append('coolwarm')
#         cmaps_label.append('')
#         img_labels.append('SDF')


#     if args.show_figs or args.save_figs:
#         fig, axs = plt.subplots(n_samples, n_subplots, figsize=(3*n_subplots, 2.5*n_samples))
#         if scaling:
#             if var == 'temp':
#                 fig.suptitle(f'Dataset with scaling, mean: {args.scale_mean}, std: {args.scale_std}')
#             elif var == 'prcp':
#                 fig.suptitle(f'Dataset with prcp scaling, type: {args.scale_type_prcp}')
                
#         else:
#             fig.suptitle(f'Dataset without scaling')


#     vmin_img = float('inf')
#     vmax_img = float('-inf')
#     vmin_img_original = float('inf')
#     vmax_img_original = float('-inf')

#     # Fixed min/max for topo and sdf (dependent on scaling)
#     if args.scaling:
#         vmin_topo = 0
#         vmax_topo = 1
#     else:
#         vmin_topo = 0#args.topo_min # wrong, makes DK completely flat
#         vmax_topo = args.topo_max 
#     vmin_sdf = 0
#     vmax_sdf = 1


    # print('\n\nDataset with options:\n')
    # for i, idx in enumerate(idxs):
    #     # Dataset with all options
    #     sample_full = dataset[idx]

    #     # Get min and max values for colorbar (between 'img' and 'img_cond')
    #     vmin_img = min(vmin_img, sample_full['img'].min(), sample_full['img_cond'].min())
    #     vmax_img = max(vmax_img, sample_full['img'].max(), sample_full['img_cond'].max())
    #     print('vmin_img: ', vmin_img)
    #     print('vmax_img: ', vmax_img)

    #     # If 'img_original' and 'img_cond_original' are in sample_full, get min and max values for colorbar
    #     if show_both_orig_scaled and scaling:
    #         vmin_img_original = min(vmin_img_original, sample_full['img_original'].min(), sample_full['img_cond_original'].min())
    #         vmax_img_original = max(vmax_img_original, sample_full['img_original'].max(), sample_full['img_cond_original'].max())
    #         print('vmin_img_original: ', vmin_img_original)
    #         print('vmax_img_original: ', vmax_img_original)

    #     print('Content of sample ', idx)
    #     print(sample_full.keys())
    #     print('\n\n')
        

    #     if args.show_figs or args.save_figs:
    #         # go through all subplots
    #         for j, img_str in enumerate(img_strs):
    #             img_data = sample_full[img_str].squeeze()

    #             # If show_ocean is False, set ocean to nan (i.e. land/sea mask )
    #             if not args.show_ocean and img_str not in ['lsm', 'sdf']:
    #                 lsm_from_sample = sample_full['lsm'].squeeze()
    #                 img_data = np.where(lsm_from_sample < 1, np.nan, img_data)
                
    #             # Plot, with different colorbars depending on the image (but same colorbar for img and img_cond, and img_original and img_cond_original, respectively)
    #             if img_str in ['img', 'img_cond']:
    #                 vmin, vmax = vmin_img, vmax_img
    #             elif img_str in ['img_original', 'img_cond_original']:
    #                 vmin, vmax = vmin_img_original, vmax_img_original
    #             elif img_str == 'topo':
    #                 vmin, vmax = vmin_topo, vmax_topo
    #             elif img_str == 'sdf':
    #                 vmin, vmax = vmin_sdf, vmax_sdf
    #             else:
    #                 vmin, vmax = None, None
                
    #             # Plot the image
    #             im = axs[i,j].imshow(img_data, cmap=cmaps[j], vmin=vmin, vmax=vmax, interpolation='nearest')
    #             axs[i,j].invert_yaxis()
    #             axs[i,j].set_xticks([])
    #             axs[i,j].set_yticks([])

    #             # Add colorbar and boxplot (for selected images)
    #             divider = make_axes_locatable(axs[i,j])

    #             if img_str in ['img', 'img_cond', 'img_original', 'img_cond_original']:
    #                 # Boxplot and colorbar 
    #                 bax = divider.append_axes("right", size="10%", pad=0.1) # Space for boxplot
    #                 cax = divider.append_axes("right", size="5%", pad=0.1) # Space for colorbar

    #                 # Set some boxplot properties
    #                 flierprops = dict(marker='o', markerfacecolor='none', markersize=2, linestyle='none', markeredgecolor='darkgreen', alpha=0.4)
    #                 medianprops = dict(linestyle='-', linewidth=2, color='black')
    #                 # meanlineprops = dict(linestyle=':', linewidth=2, color='firebrick')
    #                 meanpointprops = dict(marker='x', markerfacecolor='firebrick', markersize=5, markeredgecolor='firebrick')
                    
    #                 # Boxplot of pixel values (excluding NaNs, i.e. ocean)
    #                 img_data_bp = img_data[~np.isnan(img_data)].flatten()
    #                 bax.boxplot(img_data_bp,
    #                             vert=True,
    #                             widths=2,
    #                             patch_artist=True,
    #                             showmeans=True,
    #                             meanline=False,
    #                             meanprops=meanpointprops,
    #                             medianprops=medianprops,
    #                             flierprops=flierprops)
    #                 bax.set_xticks([])
    #                 bax.set_yticks([])
    #                 bax.set_frame_on(False) # Remove frame

    #             else:
    #                 # Only colorbar
    #                 cax = divider.append_axes("right", size="5%", pad=0.1) # Space for colorbar
    #             fig.colorbar(im, cax=cax)#, label=cmaps_label[j])


#                 if i == 0:
#                     axs[i,j].set_title(img_labels[j])
                
                    
#             fig.tight_layout()
    
#     # Save the figure
#     if args.save_figs:
#         print(f'Saving figure to {PATH_SAVE}')
#         if scaling:
#             if var == 'temp':
#                 fn = f'Dataset_{var}_ZScoreScaled'
#                 fig.savefig(PATH_SAVE + fn + '.png', dpi=300, bbox_inches='tight')
#             elif var == 'prcp':
#                 fn = f'Dataset_{var}_{args.scale_type_prcp}'
#                 fig.savefig(PATH_SAVE + fn + '.png', dpi=300, bbox_inches='tight')
#             print(f'with name {fn}')
#         else:
#             fn = f'Dataset_{var}_unscaled'
#             fig.savefig(PATH_SAVE + fn + '.png', dpi=300, bbox_inches='tight')
#             print(f'with name {fn}')

#     if args.show_figs:
#         plt.show()

#     print('\n\n')





# Test the dataset class 
if __name__ == '__main__':

    launch_test_dataset_from_args()
