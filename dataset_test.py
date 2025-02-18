'''
    Test the dataset class
'''
import zarr 
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

from multiprocessing import freeze_support

# Import DANRA dataset class from data_modules.py in src folder
from data_modules import DANRA_Dataset_cutouts_ERA5_Zarr
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
    parser.add_argument('--scale_min', type=float, default=0, help='Minimum of OG data distribution (Precipitation [mm])')
    parser.add_argument('--scale_max', type=float, default=160, help='Maximum of OG data distribution (Precipitation [mm])')
    parser.add_argument('--path_data', type=str, default='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/', help='The path to the data')
    parser.add_argument('--save_figs', type=str2bool, default=False, help='Whether to save the figures')
    parser.add_argument('--show_figs', type=str2bool, default=True, help='Whether to show the figures')
    parser.add_argument('--path_save', type=str, default='/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures/', help='The path to save the figures')
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
    
    # Set paths to zarr data
    data_dir_danra_zarr = args.path_data + 'data_DANRA/size_589x789/' + var + '_589x789/zarr_files/test.zarr'
    data_dir_era5_zarr = args.path_data + 'data_ERA5/size_589x789/' + var + '_589x789/zarr_files/test.zarr'

    # Set path to save figures
    SAVE_FIGS = args.save_figs
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
                                    scale_mean=args.scale_mean,
                                    scale_std=args.scale_std,
                                    scale_min=args.scale_min,
                                    scale_max=args.scale_max,
                                    conditional_seasons=sample_w_cond_season,
                                    conditional_images=sample_w_cond_img,
                                    cond_dir_zarr = data_dir_era5_zarr,
                                    n_classes=args.n_seasons
                                    )

    # Check if datasets work
    # Get sample images


    n_samples = args.n_gen_samples
    idxs = random.sample(range(0, len(dataset)), n_samples)

    # Figure out how many subplots are needed
    n_subplots = 1 # For HR image
    img_strs = ['img']
    cmaps = [cmap_name]
    cmaps_label = [cmap_label]

    if sample_w_cond_img:
        n_subplots += 1
        img_strs.append('img_cond')
        cmaps.append(cmap_name)
        cmaps_label.append(cmap_label)
    if sample_w_lsm_topo:
        n_subplots += 2
        img_strs.append('topo')
        cmaps.append('terrain')
        cmaps_label.append('')
        img_strs.append('lsm')
        cmaps.append('binary')
        cmaps_label.append('')
    if sample_w_sdf:
        n_subplots += 1
        img_strs.append('sdf')
        cmaps.append('coolwarm')
        cmaps_label.append('')

    if args.show_figs:
        fig, axs = plt.subplots(n_samples, n_subplots, figsize=(2*n_subplots, 2*n_samples))
        if scaling:
            if var == 'temp':
                fig.suptitle(f'Dataset with scaling, mean: {args.scale_mean}, std: {args.scale_std}')
            elif var == 'prcp':
                fig.suptitle(f'Dataset with scaling, min: {args.scale_min}, max: {args.scale_max}')
        else:
            fig.suptitle(f'Dataset without scaling')

    print('\n\nDataset with options:\n')
    for i, idx in enumerate(idxs):
        # Dataset with all options
        sample_full = dataset[idx]
        print('Content of sample ', idx)
        print(sample_full.keys())
        print('\n\n')
        

        if args.show_figs:
            # go through all subplots
            for j, img_str in enumerate(img_strs):
                

                im = axs[i,j].imshow(sample_full[img_str].squeeze(), cmap=cmaps[j])
                axs[i,j].invert_yaxis()
                axs[i,j].set_xticks([])
                axs[i,j].set_yticks([])
                # fig.colorbar(im, ax=axs[i,j], fraction=0.046, pad=0.04, label=cmaps_label[j])
                # Put truth and condition images in same colorbar
                

                if j == 0 or j == 1:
                    if scaling:
                        cb = fig.colorbar(im, ax=axs[i,j], fraction=0.046, pad=0.04, format=tkr.FormatStrFormatter('%.4f'))
                    else:
                        cb = fig.colorbar(im, ax=axs[i,j], fraction=0.046, pad=0.04, label=cmaps_label[j], format=tkr.FormatStrFormatter('%.2f'))

                else:
                    if scaling:
                        cb = fig.colorbar(im, ax=axs[i,j], fraction=0.046, pad=0.04, format=tkr.FormatStrFormatter('%.2f'))
                    else:
                        cb = fig.colorbar(im, ax=axs[i,j], fraction=0.046, pad=0.04, label=cmaps_label[j], format=tkr.FormatStrFormatter('%.2f'))

                if i == 0:
                    axs[i,j].set_title(img_str)
                    print(img_str)
    
    if args.show_figs:
        fig.tight_layout()
        plt.show()
    print('\n\n')





# Test the dataset class 
if __name__ == '__main__':

    launch_test_dataset_from_args()
