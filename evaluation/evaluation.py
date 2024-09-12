import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt


def evaluate_from_args():
    '''
        Launch the training from the command line arguments
    '''
    




    parser = argparse.ArgumentParser(description='Train a model for the downscaling of climate data')
    parser.add_argument('--HR_VAR', type=str, default='temp', help='The high resolution variable')
    parser.add_argument('--HR_SIZE', type=int, default=64, help='The shape of the high resolution data')
    parser.add_argument('--LR_VARS', type=list, default=['temp'], help='The low resolution variables')
    parser.add_argument('--LR_SIZE', type=int, default=32, help='The shape of the low resolution data')
    parser.add_argument('--path_data', type=str, default='/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/', help='The path to the data')
    parser.add_argument('--path_save', type=str, default='/Users/au728490/Documents/PhD_AU/Python_Scripts/DiffusionModels/SBGM_modular/', help='The path to save the results')
    parser.add_argument('--path_checkpoint', type=str, default='model_checkpoints/', help='The path to the checkpoints')
    parser.add_argument('--sample_w_cond_img', type=bool, default=True, help='Whether to sample with conditional images')
    parser.add_argument('--sample_w_cond_season', type=bool, default=True, help='Whether to sample with conditional seasons')
    parser.add_argument('--sample_w_cutouts', type=bool, default=True, help='Whether to sample with cutouts')
    parser.add_argument('--sample_w_lsm_topo', type=bool, default=True, help='Whether to sample with LSM and topography')
    parser.add_argument('--sample_w_sdf', type=bool, default=True, help='Whether to sample with SDF')
    parser.add_argument('--scaling', type=bool, default=True, help='Whether to scale the data')
    parser.add_argument('--calculate_stats', type=bool, default=True, help='Whether to calculate the statistics for scaling')
    parser.add_argument('--in_channels', type=int, default=1, help='The number of input channels')
    parser.add_argument('--out_channels', type=int, default=1, help='The number of output channels')
    parser.add_argument('--season_shape', type=list, default=(4,), help='The shape of the season data')
    parser.add_argument('--loss_type', type=str, default='sdfweighted', help='The type of loss function')
    parser.add_argument('--config_name', type=str, default='sbgm', help='The name of the configuration file')
    parser.add_argument('--create_figs', type=bool, default=True, help='Whether to create figures')
    parser.add_argument('--save_figs', type=bool, default=True, help='Whether to save the figures')
    parser.add_argument('--show_figs', type=bool, default=False, help='Whether to show the figures')
    parser.add_argument('--epochs', type=int, default=500, help='The number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='The batch size')
    parser.add_argument('--last_fmap_channels', type=int, default=512, help='The number of channels in the last feature map')
    parser.add_argument('--time_embedding_size', type=int, default=256, help='The size of the time embedding')
    parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='The minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='The weight decay')
    parser.add_argument('--n_timesteps', type=int, default=1000, help='The number of timesteps in diffusion process')
    parser.add_argument('--beta_min', type=float, default=1e-4, help='The minimum beta value')
    parser.add_argument('--beta_max', type=float, default=0.02, help='The maximum beta value')
    parser.add_argument('--beta_scheduler', type=str, default='cosine', help='The beta scheduler')
    parser.add_argument('--noise_variance', type=float, default=0.01, help='The noise variance')
    parser.add_argument('--CUTOUTS', type=bool, default=True, help='Whether to use cutouts')
    parser.add_argument('--CUTOUT_DOMAINS', type=list, default=[170, 350, 340, 520], help='The cutout domains')
    parser.add_argument('--num_workers', type=int, default=1, help='The number of workers')
    parser.add_argument('--topo_min', type=float, default=-12, help='The minimum topography value')
    parser.add_argument('--topo_max', type=float, default=330, help='The maximum topography value')
    parser.add_argument('--norm_min', type=float, default=0, help='The minimum normalized topography value')
    parser.add_argument('--norm_max', type=float, default=1, help='The maximum normalized topography value')
    parser.add_argument('--cache_size', type=int, default=0, help='The cache size')
    parser.add_argument('--data_split_type', type=str, default='random', help='The type of data split')
    parser.add_argument('--data_split_params', type=dict, default={'train_size': 0.8, 'val_size': 0.1, 'test_size': 0.1}, help='The data split parameters')
    parser.add_argument('--n_gen_samples', type=int, default=4, help='The number of generated samples')
    parser.add_argument('--num_heads', type=int, default=4, help='The number of heads')
    parser.add_argument('--optimizer', type=str, default='adam', help='The optimizer')
    parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau', help='The learning rate scheduler')
    parser.add_argument('--lr_scheduler_params', type=dict, default={'factor': 0.5, 'patience': 5, 'threshold': 0.01, 'min_lr': 1e-6}, help='The learning rate scheduler parameters')
    parser.add_argument('--early_stopping', type=bool, default=True, help='Whether to use early stopping')
    parser.add_argument('--early_stopping_params', type=dict, default={'patience': 50, 'min_delta': 0.0001}, help='The early stopping parameters')
    parser.add_argument('--device', type=str, default=None, help='The device')
    parser.add_argument('--test_diffusion', type=bool, default=True, help='Whether to test the diffusion process')
    parser.add_argument('--test_model', type=bool, default=True, help='Whether to test the model')
    parser.add_argument('--cfg_scale', type=float, default=0.0, help='The amount of added classifier free guidance training')
    #parser.add_argument('--init_weights', type=str, default='kaiming', help='The initialization weights')
    parser.add_argument('--plot_interval', type=int, default=5, help='Number of epochs between plots')
    parser.add_argument('--sampler', type=str, default='pc_sampler', help='The sampler to use for the langevin dynamics sampling')
    parser.add_argument('--with_ema', type=bool, default=True, help='Train the model with Exponential Moving Average of model weights')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='The decay rate of the EMA')
    parser.add_argument('--gen_years', type=list, default=[1991, 2020], help='The years to generate samples from')
    args = parser.parse_args()

    # Launch the training
    evaluate_ddpm(args)


class Evaluation:
    '''
        Class to evaluate generated samples (saved in npz files) from the SBGM model.

        Evaluates the generated samples using the following metrics:
        - All pixel values distribution (across space and time), generated and eval images
        - RMSE and MAE for all pixels in all samples
    '''


def evaluate_ddpm(args):
    print('\n\n')

    var = args.HR_VAR
    
    danra_size = args.HR_SIZE
    image_size = danra_size
    loss_type = args.loss_type
    n_seasons = args.season_shape[0]
    noise_variance = args.noise_variance

    config_name = args.config_name
    save_str = config_name + '__' + var + '__' + str(image_size) + 'x' + str(image_size) + '__' + loss_type + '__' + str(n_seasons) + '_seasons' + '__' + str(noise_variance) + '_noise' + '__' + str(args.num_heads) + '_heads' + '__' + str(args.n_timesteps) + '_timesteps'

    PATH_SAVE = args.path_save
    PATH_GENERATED_SAMPLES = PATH_SAVE + 'evaluation/generated_samples/' + save_str + '/'



    FIG_PATH = args.path_save + 'evaluation/plot_samples/' + save_str + '/' 
    print(FIG_PATH)
    if not os.path.exists(FIG_PATH):
        print(f'Creating directory {FIG_PATH}')
        os.makedirs(FIG_PATH)
    # Load generated images, truth evaluation images and lsm mask for each image
    gen_imgs = np.load(PATH_GENERATED_SAMPLES + 'gen_samples.npz')['arr_0']
    eval_imgs = np.load(PATH_GENERATED_SAMPLES + 'eval_samples.npz')['arr_0']
    lsm_imgs = np.load(PATH_GENERATED_SAMPLES + 'lsm_samples.npz')['arr_0']

    



    # Convert to torch tensors
    gen_imgs = torch.from_numpy(gen_imgs).squeeze()
    eval_imgs = torch.from_numpy(eval_imgs).squeeze()
    lsm_imgs = torch.from_numpy(lsm_imgs).squeeze()

    # Plot example of generated and eval images w/o masking
    plot_idx = np.random.randint(0, len(gen_imgs))

    fig, axs = plt.subplots(1, 2, figsize=(10,4))
    im1 = axs[0].imshow(eval_imgs[plot_idx])
    axs[0].set_ylim([0,eval_imgs[plot_idx].shape[0]])
    axs[0].set_title(f'Evaluation image', fontsize=14)
    # Remove ticks and labels
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    fig.colorbar(im1, ax=axs[0])

    im = axs[1].imshow(gen_imgs[plot_idx])
    axs[1].set_ylim([0,gen_imgs[plot_idx].shape[0]])
    axs[1].set_title(f'Generated image', fontsize=14)
    # Remove ticks and labels
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    fig.colorbar(im, ax=axs[1])
    fig.tight_layout()

#    fig.savefig(FIG_PATH + cond_str + '__example_eval_gen_images.png', dpi=600, bbox_inches='tight')


    # Mask out ocean pixels, set to nan
    for i in range(len(gen_imgs)):
        gen_imgs[i][lsm_imgs[i]==0] = np.nan
        eval_imgs[i][lsm_imgs[i]==0] = np.nan

    # Plot a sample of the generated and eval images
    fig, axs = plt.subplots(1, 2, figsize=(10,4))
    im1 = axs[0].imshow(eval_imgs[plot_idx])
    axs[0].set_ylim([0,eval_imgs[plot_idx].shape[0]])
    axs[0].set_title(f'Evaluation image', fontsize=14)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    fig.colorbar(im1, ax=axs[0])


    im = axs[1].imshow(gen_imgs[plot_idx])
    axs[1].set_ylim([0,gen_imgs[plot_idx].shape[0]])
    axs[1].set_title(f'Generated image', fontsize=14)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    fig.colorbar(im, ax=axs[1])
    fig.tight_layout()

    fig.savefig(FIG_PATH + save_str + '__example_eval_gen_images_masked.png', dpi=600, bbox_inches='tight')


    # Now evaluate the generated samples
    print("Evaluating samples...")

    # Calculate daily average MAE and RMSE for all samples (average over spatial dimensions) ignoring nans
    mae_daily = torch.abs(gen_imgs - eval_imgs).nanmean(dim=(1,2))
    rmse_daily = torch.sqrt(torch.square(gen_imgs - eval_imgs).nanmean(dim=(1,2)))


    # Calculate total single pixel-wise MAE and RMSE for all samples, no averaging
    # Flatten and concatenate the generated and eval images
    gen_imgs_flat = gen_imgs.flatten()
    eval_imgs_flat = eval_imgs.flatten()

    # Calculate MAE and RMSE for all samples ignoring nans
    mae_all = torch.abs(gen_imgs_flat - eval_imgs_flat)
    rmse_all = torch.sqrt(torch.square(gen_imgs_flat - eval_imgs_flat))

    # Make figure with four plots: MAE daily histogram, RMSE daily histogram, MAE pixel-wise histogram, RMSE pixel-wise histogram
    fig, axs = plt.subplots(2, 1, figsize=(12,6), sharex='col')
    # axs[0,0].hist(mae_daily, bins=50)
    # axs[0,0].set_title(f'MAE daily')
    # # axs[0,0].set_xlabel(f'MAE')
    # # axs[0,0].set_ylabel(f'Count')

    # axs[0].hist(mae_all, bins=70)
    # axs[0].set_title(f'MAE for all pixels')
    # axs[0].set_xlabel(f'MAE')
    # axs[0].set_ylabel(f'Count')

    axs[0].hist(rmse_daily, bins=150, alpha=0.7, label='RMSE daily', edgecolor='k')
    axs[0].set_title(f'RMSE daily', fontsize=16)
    axs[0].tick_params(axis='y', which='major', labelsize=14)
    #axs[0].set_xlabel(f'RMSE')
    axs[0].set_ylabel(f'Count', fontsize=16)

    axs[1].hist(rmse_all, bins=150, alpha=0.7, label='RMSE all pixels', edgecolor='k')
    axs[1].set_title(f'RMSE for all pixels', fontsize=16)
    axs[1].tick_params(axis='both', which='major', labelsize=14)
    axs[1].set_xlabel(f'RMSE', fontsize=16)
    axs[1].set_ylabel(f'Count', fontsize=16)
    axs[1].set_xlim([0, 25])

    fig.tight_layout()
    fig.savefig(FIG_PATH + save_str + '__RMSE_histograms.png', dpi=600, bbox_inches='tight')


    # Plot the pixel-wise distribution of the generated and eval images
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(gen_imgs.flatten(), bins=50, alpha=0.5, label='Generated')
    ax.hist(eval_imgs.flatten(), bins=50, alpha=0.5, color='r', label='Eval')
    ax.axvline(x=np.nanmean(eval_imgs.flatten()), color='r', alpha=0.5, linestyle='--', label=f'Eval mean, {np.nanmean(eval_imgs.flatten()):.2f}')
    ax.axvline(x=np.nanmean(gen_imgs.flatten()), color='b', alpha=0.5, linestyle='--', label=f'Generated mean, {np.nanmean(gen_imgs.flatten()):.2f}')
    ax.set_title(f'Distribution of generated and eval images, bias: {np.nanmean(gen_imgs.flatten())-np.nanmean(eval_imgs.flatten()):.2f}', fontsize=14)
    ax.set_xlabel(f'Pixel value', fontsize=14)
    ax.set_ylabel(f'Count', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    # Set the x-axis limits to 4 sigma around the mean of the eval images
    x_min = np.nanmin([np.nanmin(eval_imgs.flatten()), np.nanmin(gen_imgs.flatten())])
    x_max = np.nanmax([np.nanmax(eval_imgs.flatten()), np.nanmax(gen_imgs.flatten())])
    ax.set_xlim([x_min, x_max])
#    ax.set_xlim([np.nanmean(eval_imgs.flatten())-4*np.nanstd(eval_imgs.flatten()), np.nanmean(eval_imgs.flatten())+4*np.nanstd(eval_imgs.flatten())])
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_PATH + save_str + '__pixel_distribution.png', dpi=600, bbox_inches='tight')

    plt.show()



    # # Get the LSM mask for the area that the generated images are cropped from
    # CUTOUTS = True
    # CUTOUT_DOMAINS = [170, 170+180, 340, 340+180]

    # PATH_LSM_FULL = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_lsm/truth_fullDomain/lsm_full.npz'
    # PATH_TOPO_FULL = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_topo/truth_fullDomain/topo_full.npz'

    # data_lsm_full = np.flipud(np.load(PATH_LSM_FULL)['data'])[CUTOUT_DOMAINS[0]:CUTOUT_DOMAINS[1], CUTOUT_DOMAINS[2]:CUTOUT_DOMAINS[3]]
    # data_topo_full = np.flipud(np.load(PATH_TOPO_FULL)['data'])[CUTOUT_DOMAINS[0]:CUTOUT_DOMAINS[1], CUTOUT_DOMAINS[2]:CUTOUT_DOMAINS[3]]

    # # Load the points for the cutouts
    # points_imgs = np.load(SAVE_PATH + 'point_samples__' + SAVE_NAME)['arr_0']





    # NEED TO TAKE IMAGE SHIFT INTO ACCOUNT FOR THE PIXEL-WISE IMAGES 
    # # Calculate pixel-wise MAE and RMSE for all samples (average over temporal dimension)
    # mae_pixel = torch.abs(gen_imgs - eval_imgs).mean(dim=0)
    # rmse_pixel = torch.sqrt(torch.square(gen_imgs - eval_imgs).mean(dim=0))


    # # Plot image of MAE and RMSE for temporal average
    # fig, axs = plt.subplots(1, 2, figsize=(12,4))
    # im1 = axs[0].imshow(mae_pixel)
    # axs[0].set_ylim([0,mae_pixel.shape[0]])
    # axs[0].set_title(f'MAE pixel-wise')
    # fig.colorbar(im1, ax=axs[0])

    # im2 = axs[1].imshow(rmse_pixel)
    # axs[1].set_title(f'RMSE pixel-wise')
    # axs[1].set_ylim([0,rmse_pixel.shape[0]])
    # fig.colorbar(im2, ax=axs[1])



    # # Calculate Pearson correlation coefficient for all samples
    # for gen_im, ev_im in zip(gen_imgs, eval_imgs):
    #     corr = np.ma.corrcoef(np.ma.masked_invalid(gen_im), np.ma.masked_invalid(ev_im))
    #     print(corr)



    plt.show()



if __name__ == '__main__':
    evaluate_from_args()





# # FID score
# # Heidke/Pierce skill score (precipitation)
# # EV analysis (precipitation)
# # Moran's I (spatial autocorrelation)
# # Bias per pixel (spatial bias) 
# # Bias per image (temporal bias)
# # Bias per pixel per image (spatio-temporal bias)





