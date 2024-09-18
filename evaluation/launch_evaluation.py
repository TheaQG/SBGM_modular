
import argparse

import matplotlib.pyplot as plt

from evaluation import Evaluation

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
    evaluation_single = Evaluation(args, generated_sample_type='single')

    # fig, axs = evaluation_single.plot_example_images(masked=True, plot_with_cond=True, plot_with_lsm=True, show_figs=True)

    evaluation_repeated = Evaluation(args, generated_sample_type='repeated')

    # fig, axs = evaluation_repeated.plot_example_images(masked=True, plot_with_cond=False, plot_with_lsm=True, show_figs=True, n_samples=4)

    evaluation_multiple = Evaluation(args, generated_sample_type='multiple')

    evaluation_multiple.spatial_statistics(show_figs=True)
    # fig, axs = evaluation_multiple.plot_example_images(masked=False, plot_with_cond=False, plot_with_lsm=True, show_figs=True, n_samples=4, same_cbar=False)








if __name__ == '__main__':
    evaluate_from_args()