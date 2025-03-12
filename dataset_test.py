'''
    Test the dataset class
'''
import zarr 
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import freeze_support

# Import DANRA dataset class from data_modules.py in src folder
from data_modules import DANRA_Dataset_cutouts_ERA5_Zarr
from special_transforms import *
from utils import *


def launch_test_dataset_from_args():
    '''
        Launch the test dataset from the command line arguments
    '''
    parser = argparse.ArgumentParser(description='Test the dataset')
    parser.add_argument('--var', type=str, default='prcp', help='The variable to use')
    parser.add_argument('--img_dim', type=int, default=128, help='The image dimension')
    parser.add_argument('--sample_w_lsm_topo', type=str2bool, default=True, help='Whether to sample with lsm and topo')
    parser.add_argument('--sample_w_cutouts', type=str2bool, default=True, help='Whether to sample with cutouts')
    parser.add_argument('--sample_w_cond_img', type=str2bool, default=True, help='Whether to sample with conditional images')
    parser.add_argument('--sample_w_cond_season', type=str2bool, default=True, help='Whether to sample with conditional seasons')
    parser.add_argument('--sample_w_sdf', type=str2bool, default=True, help='Whether to sample with sdf')
    parser.add_argument('--scaling', type=str2bool, default=False, help='Whether to scale the data')
    parser.add_argument('--scale_mean', type=float, default=8.69251, help='Mean of OG data distribution (Temperature [C])')
    parser.add_argument('--scale_std', type=float, default=6.192434, help='STD of OG data distribution (Temperature [C])')
    parser.add_argument('--scale_type_prcp', type=str, default='log_zscore', help='Type of scaling for precipitation', choices=['log_zscore', 'log_01', 'log', 'log_minus1_1', 'log_zscore'])
    parser.add_argument('--scale_mean_log', type=float, default=-25.0, help='Mean of log-transformed data distribution (Precipitation [mm])')
    parser.add_argument('--scale_std_log', type=float, default=10.0, help='STD of log-transformed data distribution (Precipitation [mm])')
    parser.add_argument('--scale_min', type=float, default=0, help='Minimum of OG data distribution (Precipitation [mm])')
    parser.add_argument('--scale_max', type=float, default=160, help='Maximum of OG data distribution (Precipitation [mm])')
    parser.add_argument('--scale_min_log', type=float, default=-15, help='Minimum of log-transformed data distribution (Precipitation [mm])')
    parser.add_argument('--scale_max_log', type=float, default=5, help='Maximum of log-transformed data distribution (Precipitation [mm])')
    parser.add_argument('--buffer_frac', type=int, default=0.5, help='The percentage buffer size for precipition transformation')
    parser.add_argument('--path_data', type=str, default='/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/Data/Data_DiffMod/', help='The path to the data')
    parser.add_argument('--save_figs', type=str2bool, default=False, help='Whether to save the figures')
    parser.add_argument('--show_figs', type=str2bool, default=True, help='Whether to show the figures')
    parser.add_argument('--show_both_orig_scaled', type=str2bool, default=True, help='Whether to show both the original and scaled data in the same figure')
    parser.add_argument('--show_ocean', type=str2bool, default=False, help='Whether to show the ocean')
    parser.add_argument('--path_save', type=str, default='/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_figures/', help='The path to save the figures')
    parser.add_argument('--cutout_domains', type=str2list, default=[170, 170+180, 340, 340+180], help='The cutout domains')
    parser.add_argument('--topo_min', type=int, default=-12, help='The minimum value of the topological data')
    parser.add_argument('--topo_max', type=int, default=330, help='The maximum value of the topological data')
    parser.add_argument('--norm_min', type=int, default=0, help='The minimum value of the normalized topological data')
    parser.add_argument('--norm_max', type=int, default=1, help='The maximum value of the normalized topological data')
    parser.add_argument('--n_seasons', type=int, default=4, help='The number of seasons')
    parser.add_argument('--n_gen_samples', type=int, default=3, help='The number of generated samples')
    parser.add_argument('--num_workers', type=int, default=4, help='The number of workers')
    

    args = parser.parse_args()


    print(f'Scaling argument: {args.scaling}')
    test_dataset(args)





def test_dataset(args):
    # Use multiprocessing freeze_support() to avoid RuntimeError:
    freeze_support()


    # Set DANRA variable for use
    var = args.var
    if var == 'temp':
        cmap_name = 'plasma'
        cmap_label = r'$^\circ$C'
    elif var == 'prcp':
        cmap_name = 'inferno'
        cmap_label = 'mm'


    # Set size of DANRA images
    n_danra_size = args.img_dim
    # Set DANRA size string for use in path
    danra_size_str = str(n_danra_size) + 'x' + str(n_danra_size)
    # Define different samplings
    sample_w_lsm_topo = args.sample_w_lsm_topo
    sample_w_cutouts = args.sample_w_cutouts
    sample_w_cond_img = args.sample_w_cond_img
    sample_w_cond_season = args.sample_w_cond_season
    sample_w_sdf = args.sample_w_sdf

    # Make sure that lsm is True if SDF is True
    if sample_w_sdf:
        print('\nSDF is True, setting lsm and topo to True\n')
        sample_w_lsm_topo = True

    # Set scaling to true or false
    scaling = args.scaling
    show_both_orig_scaled = args.show_both_orig_scaled
    
    # Set paths to zarr data
    data_dir_danra_zarr = args.path_data + 'data_DANRA/size_589x789/' + var + '_589x789/zarr_files/test.zarr'
    data_dir_era5_zarr = args.path_data + 'data_ERA5/size_589x789/' + var + '_589x789/zarr_files/test.zarr'

    # Set path to save figures
    PATH_SAVE = args.path_save

    # Set number of samples and cache size
    danra_w_cutouts_zarr_group = zarr.open_group(data_dir_danra_zarr, mode='r')
    n_samples = len(list(danra_w_cutouts_zarr_group.keys()))#365
    cache_size = n_samples
    # Set image size
    image_size = (n_danra_size, n_danra_size)
    
    CUTOUTS = args.sample_w_cutouts
    if CUTOUTS:
        CUTOUT_DOMAINS = args.cutout_domains#[170, 170+180, 340, 340+180]
    # DOMAIN_2 = [80, 80+130, 200, 200+450]
    # Set paths to lsm and topo if used
    if sample_w_lsm_topo:
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
        data_lsm = None
        data_topo = None
    
    if scaling:
        print(f'\nScaling data')
        if var == 'temp':
            print(f'\tMean: {args.scale_mean}, std: {args.scale_std}')
        elif var == 'prcp':
            print(f'\tMin: {args.scale_min}, max: {args.scale_max}\n')


    # Initialize dataset with all options
    dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_danra_zarr, 
                                    image_size, 
                                    n_samples, 
                                    cache_size, 
                                    variable = var,
                                    shuffle=True, 
                                    cutouts = sample_w_cutouts, 
                                    cutout_domains = CUTOUT_DOMAINS,
                                    n_samples_w_cutouts = n_samples, 
                                    lsm_full_domain = data_lsm,
                                    topo_full_domain = data_topo,
                                    sdf_weighted_loss = args.sample_w_sdf,
                                    scale=scaling,
                                    save_original=show_both_orig_scaled,
                                    scale_mean=args.scale_mean,
                                    scale_std=args.scale_std,
                                    scale_type_prcp=args.scale_type_prcp,
                                    scale_min=args.scale_min,
                                    scale_max=args.scale_max,
                                    scale_min_log=args.scale_min_log,
                                    scale_max_log=args.scale_max_log,
                                    scale_mean_log=args.scale_mean_log,
                                    scale_std_log=args.scale_std_log,
                                    buffer_frac=args.buffer_frac,
                                    conditional_seasons=sample_w_cond_season,
                                    conditional_images=sample_w_cond_img,
                                    cond_dir_zarr = data_dir_era5_zarr,
                                    n_classes=args.n_seasons
                                    )

    # Check if datasets work
    # Get sample images


    n_samples = args.n_gen_samples
    idxs = random.sample(range(0, len(dataset)), n_samples)

    # Build a fixed order of subplots:
    # ['img', 'img_cond', 'img_original', 'img_cond_original'] if show_both_orig_scaled
    # or ['img', 'img_cond'] if show_both_orig_scaled = False
    # Then topo/lsm/sdf afterwards if applicable.

    img_strs = []
    cmaps = []
    cmaps_label = []
    img_labels = []
    n_subplots = 0

    # Always 'img'
    n_subplots += 1
    img_strs.append('img')
    cmaps.append(cmap_name)
    cmaps_label.append(cmap_label)
    img_labels.append('HR DANRA, scaled' if scaling else 'HR DANRA, unscaled')

    # 'img_cond' if sample_w_cond_img = True
    if sample_w_cond_img:
        n_subplots += 1
        img_strs.append('img_cond')
        cmaps.append(cmap_name)
        cmaps_label.append(cmap_label)
        img_labels.append('LR ERA5, scaled' if scaling else 'LR ERA5, unscaled')

    # If show_both_orig_scaled, we then add 'img_original' 
    # (and 'img_cond_original' if sample_w_cond_img)
    if show_both_orig_scaled and scaling:
        n_subplots += 1
        img_strs.append('img_original')
        cmaps.append(cmap_name)
        cmaps_label.append(cmap_label)
        img_labels.append('HR DANRA, original')

        if sample_w_cond_img:
            n_subplots += 1
            img_strs.append('img_cond_original')
            cmaps.append(cmap_name)
            cmaps_label.append(cmap_label)
            img_labels.append('LR ERA5, original')

    # Finally, add topo/lsm/sdf if they exist:
    if sample_w_lsm_topo:
        n_subplots += 2
        img_strs.append('topo')
        cmaps.append('terrain')
        cmaps_label.append('')
        img_labels.append('Topography')

        img_strs.append('lsm')
        cmaps.append('binary')
        cmaps_label.append('')
        img_labels.append('Land-sea mask')

    if sample_w_sdf:
        n_subplots += 1
        img_strs.append('sdf')
        cmaps.append('coolwarm')
        cmaps_label.append('')
        img_labels.append('SDF')


    if args.show_figs or args.save_figs:
        fig, axs = plt.subplots(n_samples, n_subplots, figsize=(3*n_subplots, 2.5*n_samples))
        if scaling:
            if var == 'temp':
                fig.suptitle(f'Dataset with scaling, mean: {args.scale_mean}, std: {args.scale_std}')
            elif var == 'prcp':
                fig.suptitle(f'Dataset with prcp scaling, type: {args.scale_type_prcp}')
                
        else:
            fig.suptitle(f'Dataset without scaling')


    vmin_img = float('inf')
    vmax_img = float('-inf')
    vmin_img_original = float('inf')
    vmax_img_original = float('-inf')

    # Fixed min/max for topo and sdf (dependent on scaling)
    if args.scaling:
        vmin_topo = 0
        vmax_topo = 1
    else:
        vmin_topo = 0#args.topo_min # wrong, makes DK completely flat
        vmax_topo = args.topo_max 
    vmin_sdf = 0
    vmax_sdf = 1


    print('\n\nDataset with options:\n')
    for i, idx in enumerate(idxs):
        # Dataset with all options
        sample_full = dataset[idx]

        # Get min and max values for colorbar (between 'img' and 'img_cond')
        vmin_img = min(vmin_img, sample_full['img'].min(), sample_full['img_cond'].min())
        vmax_img = max(vmax_img, sample_full['img'].max(), sample_full['img_cond'].max())
        print('vmin_img: ', vmin_img)
        print('vmax_img: ', vmax_img)

        # If 'img_original' and 'img_cond_original' are in sample_full, get min and max values for colorbar
        if show_both_orig_scaled and scaling:
            vmin_img_original = min(vmin_img_original, sample_full['img_original'].min(), sample_full['img_cond_original'].min())
            vmax_img_original = max(vmax_img_original, sample_full['img_original'].max(), sample_full['img_cond_original'].max())
            print('vmin_img_original: ', vmin_img_original)
            print('vmax_img_original: ', vmax_img_original)

        print('Content of sample ', idx)
        print(sample_full.keys())
        print('\n\n')
        

        if args.show_figs or args.save_figs:
            # go through all subplots
            for j, img_str in enumerate(img_strs):
                img_data = sample_full[img_str].squeeze()

                # If show_ocean is False, set ocean to nan (i.e. land/sea mask )
                if not args.show_ocean and img_str not in ['lsm', 'sdf']:
                    lsm_from_sample = sample_full['lsm'].squeeze()
                    img_data = np.where(lsm_from_sample < 1, np.nan, img_data)
                
                # Plot, with different colorbars depending on the image (but same colorbar for img and img_cond, and img_original and img_cond_original, respectively)
                if img_str in ['img', 'img_cond']:
                    vmin, vmax = vmin_img, vmax_img
                elif img_str in ['img_original', 'img_cond_original']:
                    vmin, vmax = vmin_img_original, vmax_img_original
                elif img_str == 'topo':
                    vmin, vmax = vmin_topo, vmax_topo
                elif img_str == 'sdf':
                    vmin, vmax = vmin_sdf, vmax_sdf
                else:
                    vmin, vmax = None, None
                
                # Plot the image
                im = axs[i,j].imshow(img_data, cmap=cmaps[j], vmin=vmin, vmax=vmax, interpolation='nearest')
                axs[i,j].invert_yaxis()
                axs[i,j].set_xticks([])
                axs[i,j].set_yticks([])

                # Add colorbar and boxplot (for selected images)
                divider = make_axes_locatable(axs[i,j])

                if img_str in ['img', 'img_cond', 'img_original', 'img_cond_original']:
                    # Boxplot and colorbar 
                    bax = divider.append_axes("right", size="10%", pad=0.1) # Space for boxplot
                    cax = divider.append_axes("right", size="5%", pad=0.1) # Space for colorbar

                    # Set some boxplot properties
                    flierprops = dict(marker='o', markerfacecolor='none', markersize=2, linestyle='none', markeredgecolor='darkgreen', alpha=0.4)
                    medianprops = dict(linestyle='-', linewidth=2, color='black')
                    # meanlineprops = dict(linestyle=':', linewidth=2, color='firebrick')
                    meanpointprops = dict(marker='x', markerfacecolor='firebrick', markersize=5, markeredgecolor='firebrick')
                    
                    # Boxplot of pixel values (excluding NaNs, i.e. ocean)
                    img_data_bp = img_data[~np.isnan(img_data)].flatten()
                    bax.boxplot(img_data_bp,
                                vert=True,
                                widths=2,
                                patch_artist=True,
                                showmeans=True,
                                meanline=False,
                                meanprops=meanpointprops,
                                medianprops=medianprops,
                                flierprops=flierprops)
                    bax.set_xticks([])
                    bax.set_yticks([])
                    bax.set_frame_on(False) # Remove frame

                else:
                    # Only colorbar
                    cax = divider.append_axes("right", size="5%", pad=0.1) # Space for colorbar
                fig.colorbar(im, cax=cax)#, label=cmaps_label[j])


                if i == 0:
                    axs[i,j].set_title(img_labels[j])
                
                    
            fig.tight_layout()
    
    # Save the figure
    if args.save_figs:
        print(f'Saving figure to {PATH_SAVE}')
        if scaling:
            if var == 'temp':
                fn = f'Dataset_{var}_ZScoreScaled'
                fig.savefig(PATH_SAVE + fn + '.png', dpi=300, bbox_inches='tight')
            elif var == 'prcp':
                fn = f'Dataset_{var}_{args.scale_type_prcp}'
                fig.savefig(PATH_SAVE + fn + '.png', dpi=300, bbox_inches='tight')
            print(f'with name {fn}')
        else:
            fn = f'Dataset_{var}_unscaled'
            fig.savefig(PATH_SAVE + fn + '.png', dpi=300, bbox_inches='tight')
            print(f'with name {fn}')

    if args.show_figs:
        plt.show()

    print('\n\n')





# Test the dataset class 
if __name__ == '__main__':

    launch_test_dataset_from_args()
