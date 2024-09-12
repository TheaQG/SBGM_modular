'''
    Script to run evaluations of trained ddpm models on test DANRA dataset.
    Default evalutes on evaluation set of size equal to two years of data (730 samples), 2001-2002.
    The default size is 64x64.
    Script only creates samples, does not evaluate or plot.

    Need to develop evaluation methods:
    - Full test set generation + evaluation
    - Single sample generation + evaluation
    - Repeated single sample generation + evaluation

    Need to create scripts for:
    - Evaluation of generated samples
    - Plotting examples of generated samples
    


'''

import torch
import zarr
import os

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchsummary import summary

from data_modules import *
from score_unet import *
from score_sampling import *
from training import *
from utils import *



def generation_sbgm(args):
    from multiprocessing import freeze_support
    freeze_support()

    # Define different samplings
    sample_w_lsm_topo = args.sample_w_lsm_topo
    sample_w_cond_img = args.sample_w_cond_img

    # Set variable for use
    var = args.HR_VAR
    lr_vars = args.LR_VARS
    
    # If the length of lr_vars is 1, and var is not the same as lr_vars[0], raise warning and set lr_vars[0] as var
    if len(lr_vars) == 1 and var != lr_vars[0]:
        print(f'Warning: HR_VAR: {var} is not the same as LR_VARS[0]: {lr_vars[0]}. Setting LR_VARS[0] to HR_VAR')
        lr_vars[0] = var


    # Print what LR and HR variables are used
    print(f'\n\nHigh resolution variable: {var}')
    print(f'Low resolution variables: {lr_vars}')
    
    # Set scaling to true or false    
    scaling = args.scaling
    # Set the variance of the noise for the diffusion process
    noise_variance = args.noise_variance

    # Set DANRA size string for use in path
    danra_size_str = '589x789'



    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set torch to deterministic mode, meaning that the same input will always produce the same output
    torch.backends.cudnn.deterministic = False
    # Set torch to benchmark mode, meaning that the best algorithm will be chosen for the input
    torch.backends.cudnn.benchmark = True







    ####################################
    #                                  #
    # SELECTION OF DATA FOR EVALUATION #
    #                                  #
    ####################################


    # First choose the data to evaluate on (n random samples from the years 2001-2005)
    n_samples_gen = args.n_gen_samples
    year_start = args.gen_years[0]
    year_end = args.gen_years[1]

    # Select folder with .nc/.npz files for generation
    gen_dir_era5 = args.path_data + 'data_ERA5/size_' + danra_size_str + '/' + var + '_' + danra_size_str + '/test/' 
    gen_dir_danra = args.path_data + 'data_DANRA/size_' + danra_size_str + '/' + var + '_' + danra_size_str + '/test/'

    # List files in the directories in chronological order
    gen_files_era5 = os.listdir(gen_dir_era5)
    gen_files_danra = os.listdir(gen_dir_danra)


    # ERA5 startrs with 2000, DANRA with 2001
    # DANRA files are named as't2m_ave_YYYYMMDD.nc', ERA5 as 'temp_589x789_YYYYMMDD.npz'
    # Select only files from the years 2001-2005 and avoid .DS_Store file
    gen_files_era5 = [file for file in gen_files_era5 if (file != '.DS_Store') and (int(file[-12:-8]) >= year_start) and (int(file[-12:-8]) <= year_end)]
    gen_files_danra = [file for file in gen_files_danra if (file != '.DS_Store') and (int(file[-11:-7]) >= year_start) and (int(file[-11:-7]) <= year_end)]
    print(f'\n\nNumber of files in ERA5 generation dataset: {len(gen_files_era5)}')
    print(f'Number of files in DANRA generation dataset: {len(gen_files_danra)}')

    # Find the files that are not in both datasets (based on date)
    gen_files_era5_dates = [int(file[-12:-4]) for file in gen_files_era5]
    gen_files_danra_dates = [int(file[-11:-3]) for file in gen_files_danra]

    gen_files_era5_not_in_danra = [file for file in gen_files_era5_dates if file not in gen_files_danra_dates]
    gen_files_danra_not_in_era5 = [file for file in gen_files_danra_dates if file not in gen_files_era5_dates]

    if len(gen_files_era5_not_in_danra) > 0 or len(gen_files_danra_not_in_era5) > 0:
        print(f'Files in ERA5 not in DANRA: {gen_files_era5_not_in_danra}')
        print(f'Files in DANRA not in ERA5: {gen_files_danra_not_in_era5}')
        print('\n\nRemoving files not in both datasets from list (not dir)...')
    else:
        print('\n\nNo files in one dataset that are not in the other, continuing...')
    
    # Remove the files that are not in both datasets
    gen_files_era5 = sorted([file for file in gen_files_era5 if int(file[-12:-4]) not in gen_files_era5_not_in_danra])
    gen_files_danra = sorted([file for file in gen_files_danra if int(file[-11:-3]) not in gen_files_danra_not_in_era5])

    # Make lists with the updated dates
    gen_files_era5_dates = [int(file[-12:-4]) for file in gen_files_era5]
    gen_files_danra_dates = [int(file[-11:-3]) for file in gen_files_danra]

    # Sort the dates and check that the dates lists are equal
    gen_files_era5_dates.sort()
    gen_files_danra_dates.sort()
    assert gen_files_era5_dates == gen_files_danra_dates, "The dates in the two datasets are not equal!"

    if len(gen_files_era5_not_in_danra) > 0 or len(gen_files_danra_not_in_era5) > 0 and (gen_files_era5_dates == gen_files_danra_dates):
        print('\n\nThe dates in the two datasets are now the same!')
    
    print(f'\nNumber of files in ERA5 generation dataset: {len(gen_files_era5)}')
    print(f'Number of files in DANRA generation dataset: {len(gen_files_danra)}\n')


    # Select n random dates (the same for both datasets) to generate samples from
    print(f'\n\nSelecting {n_samples_gen} random dates to generate samples from...')
    
    # Check that there are enough files to generate n_samples_gen samples
    if n_samples_gen > len(gen_files_era5_dates):
        # If not, generate as many samples as there are files
        print(f'Not enough files to generate {n_samples_gen} samples, generating {len(gen_files_era5_dates)} samples instead!')
        n_samples_gen = len(gen_files_era5_dates)

    # Select n random dates from the dates in the two datasets
    gen_dates = np.random.choice(gen_files_era5_dates, size=n_samples_gen, replace=False)
    # Sort the dates
    gen_dates.sort()

    print('\n\nRandom dates selected:')
    print(gen_dates)
    # Find the indices of the selected dates in the two datasets
    gen_files_era5_idx = [gen_files_era5_dates.index(date) for date in gen_dates]
    gen_files_danra_idx = [gen_files_danra_dates.index(date) for date in gen_dates]
    # Check that the indices are the same
    print('\n\nIndices of selected dates:')
    print(gen_files_era5_idx)


    # Select the files corresponding to the selected dates
    gen_files_era5 = [gen_files_era5[idx] for idx in gen_files_era5_idx]
    gen_files_danra = [gen_files_danra[idx] for idx in gen_files_danra_idx]

    print(f'\n\nSelected files, n = {len(gen_files_era5)}:')
    # Print the selected files
    for i in range(len(gen_files_era5)):
        print(f'ERA5 file: {gen_files_era5[i]}')
        print(f'DANRA file: {gen_files_danra[i]}\n')

    # Check the seasonality of the selected dates 
    print('\n\nChecking seasonality of selected dates...')
    winter = ['12', '01', '02']
    spring = ['03', '04', '05']
    summer = ['06', '07', '08']
    autumn = ['09', '10', '11']

    # Sum the number of dates in each season
    winter_count = sum([1 for date in gen_dates if str(date)[-4:-2] in winter])
    spring_count = sum([1 for date in gen_dates if str(date)[-4:-2] in spring])
    summer_count = sum([1 for date in gen_dates if str(date)[-4:-2] in summer])
    autumn_count = sum([1 for date in gen_dates if str(date)[-4:-2] in autumn])

    print(f'\n\nWinter count: {winter_count}')
    print(f'Spring count: {spring_count}')
    print(f'Summer count: {summer_count}')
    print(f'Autumn count: {autumn_count}')



    # Now copy selected files to eval directories
    print('\n\nCopying selected files to eval directories...')

    eval_npz_dir_era5 = args.path_data + 'data_ERA5/size_589x789/' + var + '_589x789/eval'
    eval_nc_dir_danra = args.path_data + 'data_DANRA/size_589x789/' + var + '_589x789/eval'


    # Check if eval directories exist, if not create them. If not empty, empty them
    if not os.path.exists(eval_npz_dir_era5):
        print(f'\n\nLR Directory does not exist. Creating directory: {eval_npz_dir_era5}')
        os.mkdir(eval_npz_dir_era5)
    else:
        print(f'\n\nLR Directory exists. Emptying directory: {eval_npz_dir_era5}')
        for file in os.listdir(eval_npz_dir_era5):
            os.remove(os.path.join(eval_npz_dir_era5, file))

    if not os.path.exists(eval_nc_dir_danra):
        print(f'\n\nHR Directory does not exist. Creating directory: {eval_nc_dir_danra}')
        os.mkdir(eval_nc_dir_danra)

    else:
        print(f'\n\nHR Directory exists. Emptying directory: {eval_nc_dir_danra}')
        for file in os.listdir(eval_nc_dir_danra):
            os.remove(os.path.join(eval_nc_dir_danra, file))

    # Copy the selected files to the eval directories
    for i in range(len(gen_files_era5)):
        os.system(f'cp {os.path.join(gen_dir_era5, gen_files_era5[i])} {os.path.join(eval_npz_dir_era5, gen_files_era5[i])}')
        os.system(f'cp {os.path.join(gen_dir_danra, gen_files_danra[i])} {os.path.join(eval_nc_dir_danra, gen_files_danra[i])}')

    print(f'\n\nN files in ERA5 eval directory: {len(os.listdir(eval_npz_dir_era5))}')
    print(f'N files in DANRA eval directory: {len(os.listdir(eval_nc_dir_danra))}')

    # Create zarr files from the copied files
    print('\n\nCreating zarr files from copied files...')

    # Path tp zarr dirs
    data_dir_danra_eval_w_cutouts_zarr = args.path_data + 'data_DANRA/size_' + danra_size_str + '/' + var + '_' + danra_size_str + '/zarr_files/' + var + '_' + danra_size_str + '_eval.zarr'
    data_dir_era5_eval_zarr = args.path_data + 'data_ERA5/size_' + danra_size_str + '/' + var + '_' + danra_size_str + '/zarr_files/' + var + '_' + danra_size_str + '_eval.zarr'

    # Convert .nc files in DANRA eval dir to zarr files
    convert_nc_to_zarr(eval_nc_dir_danra, data_dir_danra_eval_w_cutouts_zarr)
    # Convert .npz files in ERA5 eval dir to zarr files
    convert_npz_to_zarr(eval_npz_dir_era5, data_dir_era5_eval_zarr)

    # Create zarr groups
    data_danra_eval_zarr = zarr.open_group(data_dir_danra_eval_w_cutouts_zarr, mode='r')
    data_era5_eval_zarr = zarr.open_group(data_dir_era5_eval_zarr, mode='r')






    ###################################
    #                                 #
    # SETTING UP MODEL FOR GENERATION #
    #                                 #
    ###################################

    # Define data hyperparameters
    input_channels = args.in_channels
    output_channels = args.out_channels

    n_samples = n_samples_gen
    cache_size = n_samples_gen
    
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
    
    PATH_SAVE = args.path_save
    PATH_GENERATED_SAMPLES = PATH_SAVE + 'evaluation/generated_samples/' + save_str + '/'

    # Check if the directory exists, if not create it
    if not os.path.exists(PATH_GENERATED_SAMPLES):
        os.makedirs(PATH_GENERATED_SAMPLES)
        print(f'\n\nCreated directory: {PATH_GENERATED_SAMPLES}')



    # Define model hyperparameters

    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    last_fmap_channels = args.last_fmap_channels
    time_embedding = args.time_embedding_size
    
    learning_rate = args.lr
    min_lr = args.min_lr
    weight_decay = args.weight_decay
    
    loss_type = args.loss_type
    if loss_type == 'sdfweighted':
        sdf_weighted_loss = True
    else:
        sdf_weighted_loss = False


    # Define diffusion hyperparameters
    n_timesteps = args.n_timesteps
    # noise_variance = args.noise_variance # Defined above

    CUTOUTS = args.CUTOUTS
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


    # Create evaluation dataset
    eval_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_zarr=data_dir_danra_eval_w_cutouts_zarr, 
                                            data_size = image_dim, 
                                            n_samples = n_samples_gen, 
                                            cache_size = cache_size, 
                                            variable=var,
                                            scale=scaling,
                                            scale_mean=8.69251,
                                            scale_std=6.192434,
                                            scale_max=160.0,
                                            scale_min=0.0,
                                            shuffle=True,
                                            conditional_seasons=condition_on_seasons,
                                            conditional_images=condition_on_img,
                                            cond_dir_zarr=data_dir_era5_eval_zarr,
                                            n_classes=n_seasons, 
                                            cutouts=CUTOUTS, 
                                            cutout_domains=CUTOUT_DOMAINS,
                                            lsm_full_domain=data_lsm,
                                            topo_full_domain=data_topo,
                                            sdf_weighted_loss = use_sdf_weighted_loss
                                            )
    
    # Make a dataloader with batch size equal to n
    eval_dataloader = DataLoader(eval_dataset, batch_size=n_samples_gen, shuffle=False, num_workers=args.num_workers)


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

    # Define training pipeline
    pipeline = TrainingPipeline_general(score_model,
                                        loss_fn,
                                        marginal_prob_std_fn,
                                        optimizer,
                                        device,
                                        weight_init=True,
                                        sdf_weighted_loss=sdf_weighted_loss
                                        )

    # Define the learning rate scheduler
    if args.lr_scheduler is not None:
        lr_scheduler_params = args.lr_scheduler_params
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(pipeline.optimizer,
                                                                'min',
                                                                factor = lr_scheduler_params['factor'],
                                                                patience = lr_scheduler_params['patience'],
                                                                threshold = lr_scheduler_params['threshold'],
                                                                min_lr = min_lr
                                                                )
    
    # Define path to checkpoint
    checkpoint_dir = args.path_checkpoint

    name_checkpoint = save_str + '.pth.tar'

    checkpoint_path = os.path.join(checkpoint_dir, name_checkpoint)

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f'\n\nCheckpoint {os.path.join(checkpoint_dir, name_checkpoint)} does not exist, exiting...')
        exit(1)

    # Load model checkpoint
    map_location=torch.device('cpu')
    best_model_state = torch.load(checkpoint_path, map_location=map_location)['network_params']

    # Load best state model into model
    print('\nLoading best model state from checkpoint: ')
    print(checkpoint_path)
    print('\n\n')

    # Load the model state and set the model to evaluation mode
    pipeline.model.load_state_dict(best_model_state)
    pipeline.model.eval()


    # Print the summary of the model
    # print('\n\nModel summary:')
    #x, t, y, cond_img, lsm_cond, topo_cond
    # input_size_summary = (input_channels, image_size[0], image_size[1]), (1,), (1,), (1, image_size[0], image_size[1]), (1, image_size[0], image_size[1]), (1, image_size[0], image_size[1])
    # summary(model, input_size=[input_size_summary], batch_size=batch_size, device=device)










    ###############################
    #                             #
    # GENERATE SAMPLES FROM MODEL #
    #                             #
    ###############################
    
    print("Generating samples...")

    n_gen_samples = n_samples_gen

    for idx, samples in tqdm.tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), position=0):
        sample_batch_size = n_gen_samples

        # Define sampler to use
        sampler_name = args.sampler # ['pc_sampler', 'Euler_Maruyama', 'ode_sampler']

        if sampler_name == 'pc_sampler':
            sampler = pc_sampler
        elif sampler_name == 'Euler_Maruyama':
            sampler = Euler_Maruyama_sampler
        elif sampler_name == 'ode_sampler':
            sampler = ode_sampler
        else:
            raise ValueError(f'Invalid sampler name: {sampler_name}. Please choose from: ["pc_sampler", "Euler_Maruyama", "ode_sampler"]')
        
        # Extract the samples
        eval_img, eval_season, eval_cond, eval_lsm, eval_topo, eval_sdf, eval_point = extract_samples(samples, device)
        
        # Generate images from model
        generated_samples = sampler(score_model,
                                    marginal_prob_std_fn,
                                    diffusion_coeff_fn,
                                    sample_batch_size,
                                    device=device,
                                    img_size=image_size,
                                    y=eval_season,
                                    cond_img=eval_cond,
                                    lsm_cond=eval_lsm,
                                    topo_cond=eval_topo,
                                    ).squeeze()
        generated_samples = generated_samples.detach().cpu()

        # Stop after first iteration, all samples are generated
        break


    # Save the generated and corresponding eval images
    print(f'\n\nSaving generated images to {PATH_GENERATED_SAMPLES}...')
    np.savez_compressed(PATH_GENERATED_SAMPLES + 'gen_samples', generated_samples)
    np.savez_compressed(PATH_GENERATED_SAMPLES + 'eval_samples', eval_img)
    np.savez_compressed(PATH_GENERATED_SAMPLES + 'lsm_samples', eval_lsm)
    np.savez_compressed(PATH_GENERATED_SAMPLES + 'cond_samples', eval_cond)
    np.savez_compressed(PATH_GENERATED_SAMPLES + 'season_samples', eval_season)
    np.savez_compressed(PATH_GENERATED_SAMPLES + 'point_samples', eval_point)





class GenerationSBGM():
    '''
        Class to generate samples from trained SBGM model.
        Can generate samples as:
        - Single samples
        - Multiple samples
        - Repeated samples of single sample

    '''
    def __init__(self, args):
        self.args = args

    def setup_model():
        '''
            Method to setup the SBGM model for generation.
        '''
        pass

    def setup_data_folder():
        '''
            Method to setup the data for genera∑tion.
        '''
        pass

    def setup_data_loader():
        '''
            Method to setup the data loader for generation.
        '''
        pass

    def load_checkpoint():
        '''
            Method to load the checkpoint of the trained SBGM model.
        '''
        pass

    def generate_multiple_samples():
        '''
            Method to generate multiple samples from the trained SBGM model.
        '''
        pass

    def generate_single_sample():
        '''
            Method to generate a single sample from the trained SBGM model.
        '''
        pass

    def generate_repeated_single_sample():
        '''
            Method to generate repeated samples of a single sample from the trained SBGM model.
        '''
        pass