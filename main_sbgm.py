import os, torch
import pickle
import zarr

import numpy as np


from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

# Import objects from other files in this repository
from data_modules import DANRA_Dataset_cutouts_ERA5_Zarr
from special_transforms import *
from score_unet import *
from score_sampling import *
from training import *
from utils import *


def main_sbgm(args):
        
    print('\n\n')
    print('#'*50)
    print('Running ddpm')
    print('#'*50)
    print('\n\n')

    # Define different samplings
    sample_w_lsm_topo = args.sample_w_lsm_topo
    sample_w_cutouts = args.sample_w_cutouts
    sample_w_cond_img = args.sample_w_cond_img
    sample_w_cond_season = args.sample_w_cond_season
    sample_w_sdf = args.sample_w_sdf

    # General settings for use

    # Define DANRA data information 
    # Set variable for use
    var = args.HR_VAR 
    lr_vars = args.LR_VARS

    # If the length of lr_vars is 1, and var is not the same as lr_vars[0], raise warning and set lr_vars[0] as var
    if len(lr_vars) == 1 and var != lr_vars[0]:
        print(f'Warning: HR_VAR: {var} is not the same as LR_VARS[0]: {lr_vars[0]}. Setting LR_VARS[0] to HR_VAR')
        lr_vars[0] = var

    # Set scaling to true or false
    scaling = args.scaling
    noise_variance = args.noise_variance

    # Define wether to transform data back from normalization/log-space before plotting
    transform_back_bf_plot = args.transform_back_bf_plot

    # Set some color map settings
    if var == 'temp':
        cmap_name = 'plasma'
        cmap_label = 'Temperature [°C]'
    elif var == 'prcp':
        cmap_name = 'inferno'
        cmap_label = 'Precipitation [mm]'
    
    
    # Print what LR and HR variables are used
    print(f'High resolution variable: {var}')
    print(f'Low resolution variables: {lr_vars}')

    # Set DANRA size string for use in path
    danra_size_str = '589x789'

    PATH_SAVE = args.path_save
    
    # Path: .../Data_DiffMod
    # To HR data: Path + '/data_DANRA/size_589x789/' + var + '_' + danra_size_str +  '/zarr_files/train.zarr'
    PATH_HR = args.path_data + 'data_DANRA/size_' + danra_size_str + '/' + var + '_' + danra_size_str +  '/zarr_files/'
    # Path to DANRA data (zarr files), full danra, to enable cutouts
    data_dir_danra_train_w_cutouts_zarr = PATH_HR + 'train.zarr'
    data_dir_danra_valid_w_cutouts_zarr = PATH_HR + 'valid.zarr'
    data_dir_danra_test_w_cutouts_zarr = PATH_HR + 'test.zarr'

    # To LR data: Path + '/data_ERA5/size_589x789/' + var + '_' + danra_size_str +  '/'
    # Path to ERA5 data, 589x789 (same size as DANRA)
    if args.LR_VARS is not None:
        if len(lr_vars) == 1:
            lr_var = lr_vars[0]
            PATH_LR = args.path_data + 'data_ERA5/size_' + danra_size_str + '/' + lr_var + '_' + danra_size_str +  '/zarr_files/'
            data_dir_era5_train_zarr = PATH_LR + 'train.zarr'
            data_dir_era5_valid_zarr = PATH_LR + 'valid.zarr'
            data_dir_era5_test_zarr = PATH_LR + 'test.zarr'
        if len(lr_vars) > 1:
            # NEED TO IMPLEMENT CONCATENATION OF MULTIPLE VARIABLES (before training, and save to zarr file)
            KeyError('Multiple variables not yet implemented')
            # Check if .zarr files in 'concat_' str_lr_vars + '_' + danra_size_str + '/zarr_files/' exists
            str_lr_vars = '_'.join(lr_vars)
            PATH_LR = args.path_data + 'data_ERA5/size_' + danra_size_str + '/' + 'concat_' + str_lr_vars + '_' + danra_size_str + '/zarr_files/' 
            data_dir_era5_train_zarr = PATH_LR + 'train.zarr'
            # Check if .zarr file exists and otherwise raise error
            if not os.path.exists(data_dir_era5_train_zarr):
                raise FileNotFoundError(f'File not found: {data_dir_era5_train_zarr}. If multiple variables are used, the zarr file should be concatenated and saved in the directory: {PATH_LR}')
            
            data_dir_era5_valid_zarr = PATH_LR + 'valid.zarr'
            data_dir_era5_test_zarr = PATH_LR + 'test.zarr'
    else:
        data_dir_era5_train_zarr = None
        data_dir_era5_valid_zarr = None
        data_dir_era5_test_zarr = None
        
    # Make zarr groups
    data_danra_train_zarr = zarr.open_group(data_dir_danra_train_w_cutouts_zarr, mode='r')
    data_danra_valid_zarr = zarr.open_group(data_dir_danra_valid_w_cutouts_zarr, mode='r')
    data_danra_test_zarr = zarr.open_group(data_dir_danra_test_w_cutouts_zarr, mode='r')

    # /scratch/project_465000956/quistgaa/Data/Data_DiffMod/
    n_files_train = len(list(data_danra_train_zarr.keys()))
    n_files_valid = len(list(data_danra_valid_zarr.keys()))
    n_files_test = len(list(data_danra_test_zarr.keys()))

    CUTOUTS = sample_w_cutouts
    if CUTOUTS:
        CUTOUT_DOMAINS = args.CUTOUT_DOMAINS

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


    # Define data hyperparameters
    input_channels = args.in_channels
    output_channels = args.out_channels



    n_samples_train = n_files_train
    n_samples_valid = n_files_valid
    n_samples_test = n_files_test
    
    cache_size = args.cache_size
    if cache_size == 0:
        cache_size_train = n_samples_train//2
        cache_size_valid = n_samples_valid//2
        cache_size_test = n_samples_test//2
    else:
        cache_size_train = cache_size
        cache_size_valid = cache_size
        cache_size_test = cache_size
    print(f'\n\n\nNumber of training samples: {n_samples_train}')
    print(f'Number of validation samples: {n_samples_valid}')
    print(f'Number of test samples: {n_samples_test}\n')
    print(f'Total number of samples: {n_samples_train + n_samples_valid + n_samples_test}\n\n\n')

    print(f'\n\n\nCache size for training: {cache_size_train}')
    print(f'Cache size for validation: {cache_size_valid}')
    print(f'Cache size for test: {cache_size_test}\n')
    print(f'Total cache size: {cache_size_train + cache_size_valid + cache_size_test}\n\n\n')


    image_size = args.HR_SIZE
    image_dim = (image_size, image_size)
    
    n_seasons = args.season_shape[0]
    if n_seasons != 0:
        condition_on_seasons = True
    else:
        condition_on_seasons = False

    loss_type = args.loss_type
    if loss_type == 'sdfweighted':
        sdf_weighted_loss = True
    else:
        sdf_weighted_loss = False

    config_name = args.config_name
    save_str = config_name + '__' + var + '__' + str(image_size) + 'x' + str(image_size) + '__' + loss_type + '__' + str(n_seasons) + '_seasons' + '__' + str(noise_variance) + '_noise' + '__' + str(args.num_heads) + '_heads' + '__' + str(args.n_timesteps) + '_timesteps'
    
    # Set path to save figures
    PATH_SAMPLES = PATH_SAVE + 'samples' + f'/Samples' + '__' + config_name
    PATH_LOSSES = PATH_SAVE + '/losses'
    PATH_FIGURES = PATH_SAMPLES + '/Figures/'
    
    if not os.path.exists(PATH_SAMPLES):
        os.makedirs(PATH_SAMPLES)
    if not os.path.exists(PATH_LOSSES):
        os.makedirs(PATH_LOSSES)
    if not os.path.exists(PATH_FIGURES):
        os.makedirs(PATH_FIGURES)

    NAME_SAMPLES = 'Generated_samples' + '__' + save_str + '__' + 'epoch' + '_'
    NAME_FINAL_SAMPLES = f'Final_generated_sample' + '__' + save_str
    NAME_LOSSES = f'Training_losses' + '__' + save_str


    # Define the path to the pretrained model 
    PATH_CHECKPOINT = args.path_checkpoint
    try:
        os.makedirs(PATH_CHECKPOINT)
        print('\n\n\nCreating directory for saving checkpoints...')
        print(f'Directory created at {PATH_CHECKPOINT}')
    except FileExistsError:
        print('\n\n\nDirectory for saving checkpoints already exists...')
        print(f'Directory located at {PATH_CHECKPOINT}')


    NAME_CHECKPOINT = save_str + '.pth.tar'

    checkpoint_dir = PATH_CHECKPOINT
    checkpoint_name = NAME_CHECKPOINT
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    print(f'\nCheckpoint path: {checkpoint_path}')


    # Define model hyperparameters
    epochs = args.epochs
    batch_size = args.batch_size
    
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    last_fmap_channels = args.last_fmap_channels
    time_embedding = args.time_embedding_size
    
    learning_rate = args.lr
    min_lr = args.min_lr
    weight_decay = args.weight_decay

    # Define diffusion hyperparameters
    n_timesteps = args.n_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    beta_scheduler = args.beta_scheduler
    # noise_variance = args.noise_variance # Defined above

    # Define if samples should be moved around in the cutout domains
    CUTOUTS = args.CUTOUTS
    CUTOUT_DOMAINS = args.CUTOUT_DOMAINS

    # Define the loss function
    if loss_type == 'simple':
        lossfunc = SimpleLoss()
        use_sdf_weighted_loss = False
    elif loss_type == 'hybrid':
        lossfunc = HybridLoss(alpha=0.5, T=n_timesteps)#nn.MSELoss()#SimpleLoss()#
        use_sdf_weighted_loss = False
    elif loss_type == 'sdfweighted':
        lossfunc = SDFWeightedMSELoss(max_land_weight=1.0, min_sea_weight=0.0)
        use_sdf_weighted_loss = True
        # NEED TO ACCOUNT FOR POSSIBILITY OF MULTIPLE DOMAINS


    if args.LR_VARS is not None:
        condition_on_img = True
    else:
        condition_on_img = False

    
    # Define training dataset, with cutouts enabled and data from zarr files
    train_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_zarr=data_dir_danra_train_w_cutouts_zarr, 
                                            data_size=image_dim, 
                                            n_samples=n_samples_train, 
                                            cache_size=cache_size_train, 
                                            variable=var,
                                            shuffle=False,
                                            cutouts=CUTOUTS, 
                                            cutout_domains=CUTOUT_DOMAINS,
                                            n_samples_w_cutouts=n_samples_train,
                                            lsm_full_domain=data_lsm,
                                            topo_full_domain=data_topo,
                                            sdf_weighted_loss=use_sdf_weighted_loss,
                                            scale=scaling, 
                                            save_original=args.show_both_orig_scaled,
                                            scale_mean=args.scale_mean,
                                            scale_std=args.scale_std,
                                            scale_min=args.scale_min,
                                            scale_max=args.scale_max,
                                            scale_min_log=args.scale_min_log,
                                            scale_max_log=args.scale_max_log,
                                            buffer_frac=args.buffer_frac,
                                            conditional_seasons=condition_on_seasons,
                                            conditional_images=condition_on_img,
                                            cond_dir_zarr=data_dir_era5_train_zarr, 
                                            n_classes=n_seasons
                                            )
    # Define validation dataset, with cutouts enabled and data from zarr files
    valid_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_zarr=data_dir_danra_valid_w_cutouts_zarr, 
                                            data_size=image_dim, 
                                            n_samples=n_samples_valid, 
                                            cache_size=cache_size_valid, 
                                            variable=var,
                                            shuffle=False,
                                            cutouts=CUTOUTS, 
                                            cutout_domains=CUTOUT_DOMAINS,
                                            n_samples_w_cutouts=n_samples_valid,
                                            lsm_full_domain=data_lsm,
                                            topo_full_domain=data_topo,
                                            sdf_weighted_loss=use_sdf_weighted_loss,
                                            scale=scaling,
                                            save_original=args.show_both_orig_scaled,
                                            scale_mean=args.scale_mean,
                                            scale_std=args.scale_std,
                                            scale_min=args.scale_min,
                                            scale_max=args.scale_max,
                                            scale_min_log=args.scale_min_log,
                                            scale_max_log=args.scale_max_log,
                                            buffer_frac=args.buffer_frac,
                                            conditional_seasons=condition_on_seasons, 
                                            conditional_images=condition_on_img,
                                            cond_dir_zarr=data_dir_era5_valid_zarr,
                                            n_classes=n_seasons, 
                                            )
    # Define test dataset, with cutouts enabled and data from zarr files
    gen_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_zarr=data_dir_danra_test_w_cutouts_zarr,
                                            data_size = image_dim,
                                            n_samples = n_samples_test,
                                            cache_size = cache_size_test,
                                            variable=var,
                                            shuffle=True,
                                            cutouts=CUTOUTS,
                                            cutout_domains=CUTOUT_DOMAINS,
                                            n_samples_w_cutouts=n_samples_test,
                                            lsm_full_domain=data_lsm,
                                            topo_full_domain=data_topo,
                                            sdf_weighted_loss = use_sdf_weighted_loss,
                                            scale=scaling,
                                            save_original=args.show_both_orig_scaled,
                                            scale_mean=args.scale_mean,
                                            scale_std=args.scale_std,
                                            scale_min=args.scale_min,
                                            scale_max=args.scale_max,
                                            scale_min_log=args.scale_min_log,
                                            scale_max_log=args.scale_max_log,
                                            buffer_frac=args.buffer_frac,
                                            conditional_seasons=condition_on_seasons, 
                                            conditional_images=condition_on_img,    
                                            cond_dir_zarr=data_dir_era5_test_zarr,
                                            n_classes=n_seasons,
                                            )


    # Define the torch dataloaders for train and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#, num_workers=args.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)#, num_workers=args.num_workers)

    n_gen_samples = args.n_gen_samples
    gen_dataloader = DataLoader(gen_dataset, batch_size=n_gen_samples, shuffle=False)#, num_workers=args.num_workers)

    # Examine first batch from test dataloader

    # Define the seed for reproducibility, and set seed for torch, numpy and random
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Set torch to deterministic mode, meaning that the same input will always produce the same output
    torch.backends.cudnn.deterministic = False
    # Set torch to benchmark mode, meaning that the best algorithm will be chosen for the input
    torch.backends.cudnn.benchmark = True




    # Define the encoder and decoder from modules_DANRA_downscaling.py
    encoder = Encoder(input_channels, 
                        time_embedding,
                        cond_on_lsm=sample_w_lsm_topo,
                        cond_on_topo=sample_w_lsm_topo,
                        cond_on_img=sample_w_cond_img, 
                        cond_img_dim=(1, args.LR_SIZE, args.LR_SIZE), 
                        block_layers=[2, 2, 2, 2], 
                        num_classes=n_seasons,
                        n_heads=args.num_heads
                        )
    decoder = Decoder(last_fmap_channels, 
                        output_channels, 
                        time_embedding, 
                        n_heads=args.num_heads
                        )
    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn, encoder=encoder, decoder=decoder)
    score_model = score_model.to(device)


    # Define the optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(score_model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(score_model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(score_model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay)


    # Define the training pipeline
    pipeline = TrainingPipeline_general(score_model,
                                        loss_fn,
                                        marginal_prob_std_fn,
                                        optimizer,
                                        device,
                                        weight_init=True,
                                        sdf_weighted_loss=sdf_weighted_loss
                                        )
    
    # Define learning rate scheduler
    if args.lr_scheduler is not None:
        lr_scheduler_params = args.lr_scheduler_params
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(pipeline.optimizer,
                                                                  'min',
                                                                  factor = lr_scheduler_params['factor'],
                                                                  patience = lr_scheduler_params['patience'],
                                                                  threshold = lr_scheduler_params['threshold'],
                                                                  min_lr = min_lr,
                                                                  verbose = True
                                                                  )
        
    # Check if path to pretrained model exists
    if os.path.isfile(checkpoint_path):
        print('\n\nLoading pretrained weights from checkpoint:')
        print(checkpoint_path)

        checkpoint_state = torch.load(checkpoint_path, map_location=device)['network_params']
        pipeline.model.load_state_dict(checkpoint_state)

    # Check if device is cude, if so print information and empty cache
    if torch.cuda.is_available():
        print(f'\nModel is training on {torch.cuda.get_device_name()}')
        print(f'Model is using {torch.cuda.memory_allocated()} bytes of memory\n\n')
        torch.cuda.empty_cache()
        
    
    # Set empty lists for storing losses
    train_losses = []
    valid_losses = []

    # Set best loss to infinity
    best_loss = np.inf

    print('\n\n\nStarting training...\n\n\n')

    # Loop over epochs
    for epoch in range(epochs):

        print(f'\n\n\nEpoch {epoch+1} of {epochs}\n\n\n')
        if epoch == 0:
            PLOT_FIRST_IMG = True
        else:
            PLOT_FIRST_IMG = False

        # Calculate the training loss
        train_loss = pipeline.train(train_dataloader,
                                    verbose=False,
                                    PLOT_FIRST=PLOT_FIRST_IMG,
                                    SAVE_PATH = PATH_SAMPLES,
                                    SAVE_NAME = 'upsampled_images_example.png'
                                    )
        train_losses.append(train_loss)

        # Calculate the validation loss
        valid_loss = pipeline.validate(valid_dataloader,
                                        verbose=False,
                                        )
        valid_losses.append(valid_loss)

        # Print the training and validation losses
        print(f'\n\n\nTraining loss: {train_loss:.6f}')
        print(f'Validation loss: {valid_loss:.6f}\n\n\n')
        
        with open(PATH_LOSSES + '/' + NAME_LOSSES + '_train', 'wb') as fp:
            pickle.dump(train_losses, fp)
        with open(PATH_LOSSES + '/' + NAME_LOSSES + '_valid', 'wb') as fp:
            pickle.dump(valid_losses, fp)

        if args.create_figs:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.plot(train_losses, label='Train')
            ax.plot(valid_losses, label='Validation')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss')
            ax.legend(loc='upper right')
            fig.tight_layout()
            if args.show_figs:
                plt.show()
                
            if args.save_figs:
                fig.savefig(PATH_FIGURES + NAME_LOSSES + '.png', dpi=600, bbox_inches='tight', pad_inches=0.1)


        # If validation loss is lower than best loss, save the model. With possibility of early stopping
        if valid_loss < best_loss:
            best_loss = valid_loss
            pipeline.save_model(checkpoint_dir, checkpoint_name)
            print(f'Model saved at epoch {epoch+1} with validation loss: {valid_loss:.6f}')
            print(f'Saved to: {checkpoint_path} with name {checkpoint_name}\n\n')

            # If create figures is enabled, create figures
            if args.create_figs and n_gen_samples > 0 and epoch % args.plot_interval == 0:
                if epoch == 0:
                    print('First epoch, generating samples...')
                else:
                    print('Valid loss is better than best loss, and epoch is a multiple of plot interval, generating samples...')

                PATH_CHECKPOINT = args.path_save + args.path_checkpoint
                NAME_CHECKPOINT = save_str + '.pth.tar'

                checkpoint_path = os.path.join(PATH_CHECKPOINT, NAME_CHECKPOINT)

                # Load the model
                best_model_path = checkpoint_path
                best_model_state = torch.load(best_model_path)['network_params']

                # Load the model state
                pipeline.model.load_state_dict(best_model_state)
                # Set model to evaluation mode (remember to set back to training mode after generating samples)
                pipeline.model.eval()

                for idx, samples in tqdm.tqdm(enumerate(gen_dataloader), total=len(gen_dataloader)):
                    sample_batch_size = n_gen_samples
                    
                    # Define the sampler to use
                    sampler_name = args.sampler # ['pc_sampler', 'Euler_Maruyama_sampler', 'ode_sampler']

                    if sampler_name == 'pc_sampler':
                        sampler = pc_sampler
                    elif sampler_name == 'Euler_Maruyama_sampler':
                        sampler = Euler_Maruyama_sampler
                    elif sampler_name == 'ode_sampler':
                        sampler = ode_sampler
                    else:
                        raise ValueError('Sampler not recognized. Please choose between: pc_sampler, Euler_Maruyama_sampler, ode_sampler')

                    test_images, test_seasons, test_cond, test_lsm, test_sdf, test_topo, _ = extract_samples(samples, device=device)
                    data_plot = [test_images, test_cond, test_lsm, test_sdf, test_topo]
                    data_names = ['Truth', 'Condition', 'LSM', 'SDF', 'Topography']
                    # Filter out None samples
                    data_plot = [sample for sample in data_plot if sample is not None]
                    data_names = [name for name, sample in zip(data_names, data_plot) if sample is not None]

                    # Count length of data_plot
                    n_axs = len(data_plot)

                    generated_samples = sampler(score_model,
                                                marginal_prob_std_fn,
                                                diffusion_coeff_fn,
                                                sample_batch_size,
                                                device=device,
                                                img_size=image_size,
                                                y=test_seasons,
                                                cond_img=test_cond,
                                                lsm_cond=test_lsm,
                                                topo_cond=test_topo,
                                                ).squeeze()
                    generated_samples = generated_samples.detach().cpu()

                    # Add generated samples to data_plot and data_names
                    data_plot.append(generated_samples)
                    data_names.append('Generated')

                    # Plotting truth, condition, generated, lsm and topo for n different test images
                    fig, axs = plt.subplots(n_axs+1, n_gen_samples, figsize=(14,9)) 
                    
                    # Make the first row the generated images
                    for i in range(n_gen_samples):
                        img = data_plot[-1][i].squeeze()

                        # If transform back before plotting is enabled, transform back
                        if transform_back_bf_plot:
                            # If temperature, Z-score back transform
                            if var == 'temp':
                                img = ZScoreBackTransform(mean=args.scale_mean, std=args.scale_std)(img)
                            # If precipitation, log back transform
                            elif var == 'prcp':
                                img = PrcpLogBackTransform(min_log=args.scale_min_log, max_log=args.scale_max_log)(img)

                        image = axs[0, i].imshow(img, cmap=cmap_name)
                        axs[0, i].set_title(f'{data_names[-1]}')
                        axs[0, i].axis('off')
                        axs[0, i].set_ylim([0, img.shape[0]])
                        fig.colorbar(image, ax=axs[0, i], fraction=0.046, pad=0.04, orientation='vertical')


                    # Loop through the generated samples (and corresponding truth, condition, lsm and topo) and plot
                    for i in range(n_gen_samples):
                        for j in range(n_axs):
                            img = data_plot[j][i].squeeze()
                            if data_names[j] == 'Truth' or data_names[j] == 'Condition':
                                cmap_name_use = cmap_name
                                # If transform back before plotting is enabled, transform back (only for Truth and Condition)
                                if transform_back_bf_plot:
                                    # If temperature, Z-score back transform
                                    if var == 'temp':
                                        img = ZScoreBackTransform(mean=args.scale_mean, std=args.scale_std)(img)
                                    # If precipitation, log back transform
                                    elif var == 'prcp':
                                        img = PrcpLogBackTransform(min_log=args.scale_min_log, max_log=args.scale_max_log)(img)

                            else:
                                cmap_name_use = 'viridis'

                            
                            image = axs[j+1, i].imshow(img, cmap=cmap_name_use)
                            axs[j+1, i].set_title(f'{data_names[j]}')
                            axs[j+1, i].axis('off')
                            axs[j+1, i].set_ylim([0, img.shape[0]])
                            fig.colorbar(image, ax=axs[j+1, i], fraction=0.046, pad=0.04)

                    fig.tight_layout()
                    if args.show_figs:
                        plt.show()
                    
                    
                    # Save figure
                    if args.save_figs:
                        if epoch == (epochs - 1):
                            fig.savefig(PATH_FIGURES + NAME_FINAL_SAMPLES + '.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
                            print(f'Saving final generated sample in {PATH_SAMPLES} as {NAME_FINAL_SAMPLES}.png')
                        else:
                            fig.savefig(PATH_FIGURES + NAME_SAMPLES + str(epoch+1) + '.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
                            print(f'Saving generated samples in {PATH_FIGURES} as {NAME_SAMPLES}{epoch+1}.png')
                    
                    break
                
                # Set model back to train mode
                pipeline.model.train()
            


            # Early stopping
            PATIENCE = args.early_stopping_params['patience']
        else:
            PATIENCE -= 1
            if PATIENCE == 0:
                print(f'Early stopping at epoch {epoch+1}')
                break

        # Close any and all figures
        plt.close('all')
        
        # If learning rate scheduler is not None, step the scheduler
        if args.lr_scheduler is not None:
            lr_scheduler.step(valid_loss)                    

