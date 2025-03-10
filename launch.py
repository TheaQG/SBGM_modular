import argparse

from main_sbgm import main_sbgm
from utils import *
#from ..src.main_ddpm import main_ddpm
#from ..src.utils import convert_npz_to_zarr

def data_checker(args):
    '''
        Based on arguments check if data exists and in the right format
        If not, create right format
    '''
    hr_var = args.HR_VAR
    lr_vars = args.LR_VARS

    data_path = args.path_data

    # Check if the data exists


# --path_data '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/'
# --path_save '/Users/au728490/Documents/PhD_AU/Python_Scripts/DiffusionModels/DDPM_clean/'

def launch_from_args():
    '''
        Launch the training from the command line arguments
    '''
    

    parser = argparse.ArgumentParser(description='Train a model for the downscaling of climate data')
    parser.add_argument('--HR_VAR', type=str, default='temp', help='The high resolution variable')
    parser.add_argument('--HR_SIZE', type=int, default=64, help='The shape of the high resolution data')
    parser.add_argument('--LR_VARS', type=str2list_of_strings, default=['temp'], help='The low resolution variables')
    parser.add_argument('--LR_SIZE', type=int, default=32, help='The shape of the low resolution data')
    parser.add_argument('--path_data', type=str, default='/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/Data/Data_DiffMod/', help='The path to the data')
    parser.add_argument('--path_save', type=str, default='/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/DiffusionModels/SBGM_modular/', help='The path to save the results')
    parser.add_argument('--path_checkpoint', type=str, default='model_checkpoints/', help='The path to the checkpoints')
    parser.add_argument('--sample_w_cond_img', type=str2bool, default=True, help='Whether to sample with conditional images')
    parser.add_argument('--sample_w_cond_season', type=str2bool, default=True, help='Whether to sample with conditional seasons')
    parser.add_argument('--sample_w_cutouts', type=str2bool, default=True, help='Whether to sample with cutouts')
    parser.add_argument('--sample_w_lsm_topo', type=str2bool, default=True, help='Whether to sample with LSM and topography')
    parser.add_argument('--sample_w_sdf', type=str2bool, default=True, help='Whether to sample with SDF')
    parser.add_argument('--scaling', type=str2bool, default=True, help='Whether to scale the data')
    parser.add_argument('--scale_mean', type=float, default=8.69251, help='Mean of OG data distribution (Temperature [C])')
    parser.add_argument('--scale_std', type=float, default=6.192434, help='STD of OG data distribution (Temperature [C])')
    parser.add_argument('--scale_min', type=float, default=0, help='Minimum of OG data distribution (Precipitation [mm])')
    parser.add_argument('--scale_max', type=float, default=160, help='Maximum of OG data distribution (Precipitation [mm])')
    parser.add_argument('--scale_min_log', type=float, default=-15, help='Minimum of log-transformed data distribution (Precipitation [mm])')
    parser.add_argument('--scale_max_log', type=float, default=5, help='Maximum of log-transformed data distribution (Precipitation [mm])')
    parser.add_argument('--buffer_frac', type=float, default=0.5, help='The percentage buffer size for precipition transformation')
    parser.add_argument('--calculate_stats', type=str2bool, default=True, help='Whether to calculate the statistics for scaling')
    parser.add_argument('--in_channels', type=int, default=1, help='The number of input channels')
    parser.add_argument('--out_channels', type=int, default=1, help='The number of output channels')
    parser.add_argument('--season_shape', type=str2list, default=(4,), help='The shape of the season data')
    parser.add_argument('--loss_type', type=str, default='sdfweighted', help='The type of loss function')
    parser.add_argument('--config_name', type=str, default='sbgm', help='The name of the configuration file')
    parser.add_argument('--create_figs', type=str2bool, default=True, help='Whether to create figures')
    parser.add_argument('--save_figs', type=str2bool, default=True, help='Whether to save the figures')
    parser.add_argument('--show_figs', type=str2bool, default=False, help='Whether to show the figures')
    parser.add_argument('--show_both_orig_scaled', type=str2bool, default=False, help='Whether to show both the original and scaled data')
    parser.add_argument('--transform_back_bf_plot', type=str2bool, default=True, help='Whether to transform back before plotting')
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
    parser.add_argument('--CUTOUTS', type=str2bool, default=True, help='Whether to use cutouts')
    parser.add_argument('--CUTOUT_DOMAINS', type=str2list, default=[170, 350, 340, 520], help='The cutout domains')
    parser.add_argument('--num_workers', type=int, default=1, help='The number of workers')
    parser.add_argument('--topo_min', type=float, default=-12, help='The minimum topography value')
    parser.add_argument('--topo_max', type=float, default=330, help='The maximum topography value')
    parser.add_argument('--norm_min', type=float, default=0, help='The minimum normalized topography value')
    parser.add_argument('--norm_max', type=float, default=1, help='The maximum normalized topography value')
    parser.add_argument('--cache_size', type=int, default=0, help='The cache size')
    parser.add_argument('--data_split_type', type=str, default='random', help='The type of data split')
    parser.add_argument('--data_split_params', type=str2dict, default={'train_size': 0.8, 'val_size': 0.1, 'test_size': 0.1}, help='The data split parameters. Format: {"train_size": 0.8, "val_size": 0.1, "test_size": 0.1}')
    parser.add_argument('--n_gen_samples', type=int, default=4, help='The number of generated samples')
    parser.add_argument('--num_heads', type=int, default=4, help='The number of heads')
    parser.add_argument('--optimizer', type=str, default='adam', help='The optimizer')
    parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau', help='The learning rate scheduler')
    parser.add_argument('--lr_scheduler_params', type=str2dict, default={'factor': 0.5, 'patience': 5, 'threshold': 0.01, 'min_lr': 1e-6}, help='The learning rate scheduler parameters. Format: {"factor": 0.5, "patience": 5, "threshold": 0.01, "min_lr": 1e-6}')
    parser.add_argument('--early_stopping', type=str2bool, default=True, help='Whether to use early stopping')
    parser.add_argument('--early_stopping_params', type=str2dict, default={'patience': 50, 'min_delta': 0.0001}, help='The early stopping parameters. Format: {"patience": 50, "min_delta": 0.0001}')
    parser.add_argument('--device', type=str, default=None, help='The device')
    parser.add_argument('--test_diffusion', type=str2bool, default=True, help='Whether to test the diffusion process')
    parser.add_argument('--test_model', type=str2bool, default=True, help='Whether to test the model')
    parser.add_argument('--cfg_scale', type=float, default=0.0, help='The amount of added classifier free guidance training')
    #parser.add_argument('--init_weights', type=str, default='kaiming', help='The initialization weights')
    parser.add_argument('--plot_interval', type=int, default=5, help='Number of epochs between plots')
    parser.add_argument('--sampler', type=str, default='pc_sampler', help='The sampler to use for the langevin dynamics sampling')
    parser.add_argument('--with_ema', type=str2bool, default=True, help='Train the model with Exponential Moving Average of model weights')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='The decay rate of the EMA')
    parser.add_argument('--back_transform_before_plot', type=str2bool, default=True, help='Whether to back-transform the data before plotting')

    args = parser.parse_args()

    # Launch the training
    main_sbgm(args)



if __name__ == '__main__':
    launch_from_args()