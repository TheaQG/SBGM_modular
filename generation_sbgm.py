'''
    Script to run evaluations of trained ddpm models on test DANRA dataset.
    Default evalutes on evaluation set of size equal to two years of data (730 samples), 2001-2002.
    The default size is 64x64.
    Script only creates samples, does not evaluate or plot.

    !!! MISSING: No shift in image, steady over specific area of DK
    


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




class GenerationSBGM():
    '''
        Class to generate samples from trained SBGM model.
        Can generate samples as:
        - Single samples
        - Multiple samples
        - Repeated samples of single sample

    '''
    def __init__(self, args):
        '''
            Constructor for the GenerationSBGM class.
            Args:
                args: Namespace, arguments from the command line.
        '''
        
        
        
        
        ##################################
        #                                #
        # SETUP WITH NECESSARY ARGUMENTS #
        #                                #
        ##################################

        self.args = args
        
        # Define different samplings
        self.sample_w_lsm_topo = args.sample_w_lsm_topo
        self.sample_w_cond_img = args.sample_w_cond_img

        # Set variable for use
        self.var = args.HR_VAR
        self.lr_vars = args.LR_VARS
        
        # Set paths for data and save
        self.path_data = args.path_data
        self.path_save = args.path_save

        # If the length of lr_vars is 1, and var is not the same as lr_vars[0], raise warning and set lr_vars[0] as var
        if len(self.lr_vars) == 1 and self.var != self.lr_vars[0]:
            print(f'Warning: HR_VAR: {self.var} is not the same as LR_VARS[0]: {self.lr_vars[0]}. Setting LR_VARS[0] to HR_VAR')
            self.lr_vars[0] = self.var


        # Print what LR and HR variables are used
        print(f'\n\nHigh resolution variable: {self.var}')
        print(f'Low resolution variables: {self.lr_vars}')
        
        # Set scaling to true or false
        self.scaling = args.scaling
        # Set the variance of the noise for the diffusion process
        self.noise_variance = args.noise_variance

        # Set DANRA size string for use in path
        self.danra_size_str = '589x789'



        # Set seed for reproducibility
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Set torch to deterministic mode, meaning that the same input will always produce the same output
        torch.backends.cudnn.deterministic = False
        # Set torch to benchmark mode, meaning that the best algorithm will be chosen for the input
        torch.backends.cudnn.benchmark = True




        ##############################
        # SETUP DATA HYPERPARAMETERS #
        ##############################

        self.input_channels = args.in_channels
        self.output_channels = args.out_channels

        self.num_workers = args.num_workers

        self.image_size = self.args.HR_SIZE
        self.image_dim = (self.image_size, self.image_size)

        self.LR_SIZE = args.LR_SIZE

        self.n_seasons = args.season_shape[0]
        if self.n_seasons != 0:
            self.condition_on_seasons = True
        else:
            self.condition_on_seasons = False

        self.loss_type = args.loss_type
        if self.loss_type == 'sdfweighted':
            self.sdf_weighted_loss = True
        else:
            self.sdf_weighted_loss = False

        self.optimizer = args.optimizer

        config_name = args.config_name
        self.save_str = config_name + '__' + self.var + '__' + str(self.image_size) + 'x' + str(self.image_size) + '__' + self.loss_type + '__' + str(self.n_seasons) + '_seasons' + '__' + str(args.noise_variance) + '_noise' + '__' + str(args.num_heads) + '_heads' + '__' + str(args.n_timesteps) + '_timesteps'
        
        self.PATH_SAVE = self.path_save
        self.PATH_GENERATED_SAMPLES = self.PATH_SAVE + 'evaluation/generated_samples/' + self.save_str + '/'

        # Check if the directory exists, if not create it
        if not os.path.exists(self.PATH_GENERATED_SAMPLES):
            os.makedirs(self.PATH_GENERATED_SAMPLES)
            print(f'\n\nCreated directory: {self.PATH_GENERATED_SAMPLES}')


        # Set the year range for generation
        self.gen_years = args.gen_years
        self.year_start = self.gen_years[0]
        self.year_end = self.gen_years[1]



        
        ###############################
        # SETUP MODEL HYPERPARAMETERS #
        ###############################

        if args.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = args.device
        
        self.last_fmap_channels = args.last_fmap_channels
        self.time_embedding = args.time_embedding_size
        self.num_heads = args.num_heads
        
        self.learning_rate = args.lr
        self.min_lr = args.min_lr
        self.weight_decay = args.weight_decay
        self.lr_scheduler = args.lr_scheduler
        self.lr_scheduler_params = args.lr_scheduler_params
        self.min_lr = args.min_lr
        
        self.loss_type = args.loss_type
        if self.loss_type == 'sdfweighted':
            self.sdf_weighted_loss = True
        else:
            self.sdf_weighted_loss = False


        # Define diffusion hyperparameters
        self.n_timesteps = args.n_timesteps
        # noise_variance = args.noise_variance # Defined above

        self.CUTOUTS = args.CUTOUTS
        if self.CUTOUTS:
            self.CUTOUT_DOMAINS = args.CUTOUT_DOMAINS

        if self.sample_w_lsm_topo:
            self.data_dir_lsm = self.path_data + 'data_lsm/truth_fullDomain/lsm_full.npz'
            self.data_dir_topo = self.path_data + 'data_topo/truth_fullDomain/topo_full.npz'

            self.data_lsm = np.flipud(np.load(self.data_dir_lsm)['data'])
            self.data_topo = np.flipud(np.load(self.data_dir_topo)['data'])

            if self.scaling:
                topo_min, topo_max = args.topo_min, args.topo_max
                norm_min, norm_max = args.norm_min, args.norm_max
                
                OldRange = (topo_max - topo_min)
                NewRange = (norm_max - norm_min)

                # Generating the new data based on the given intervals
                self.data_topo = (((self.data_topo - topo_min) * NewRange) / OldRange) + norm_min
        else:
            self.data_lsm = None
            self.data_topo = None

        # Define the loss function
        if self.loss_type == 'simple':
            self.lossfunc = SimpleLoss()
            self.use_sdf_weighted_loss = False
        elif self.loss_type == 'hybrid':
            self.lossfunc = HybridLoss(alpha=0.5, T=self.n_timesteps)#nn.MSELoss()#SimpleLoss()#
            self.use_sdf_weighted_loss = False
        elif self.loss_type == 'sdfweighted':
            self.lossfunc = SDFWeightedMSELoss(max_land_weight=1.0, min_sea_weight=0.0)
            self.use_sdf_weighted_loss = True
            # NEED TO ACCOUNT FOR POSSIBILITY OF MULTIPLE DOMAINS

        if self.lr_vars is not None:
            self.condition_on_img = True
        else:
            self.condition_on_img = False

        # Set the path to the model checkpoint 
        self.path_checkpoint = args.path_checkpoint





    def setup_data_folder(self, n_gen_samples):
        '''
            Method to setup the data for genera∑tion.
        '''
        

        ####################################
        #                                  #
        # SELECTION OF DATA FOR EVALUATION #
        #                                  #
        ####################################


        # First choose the data to evaluate on (n random samples from the years 2001-2005)
        self.n_samples_gen = n_gen_samples
        self.cache_size = self.n_samples_gen
        self.year_start = self.gen_years[0]
        self.year_end = self.gen_years[1]

        # Select folder with .nc/.npz files for generation
        gen_dir_era5 = self.path_data + 'data_ERA5/size_' + self.danra_size_str + '/' + self.var + '_' + self.danra_size_str + '/test/' 
        gen_dir_danra = self.path_data + 'data_DANRA/size_' + self.danra_size_str + '/' + self.var + '_' + self.danra_size_str + '/test/'

        # List files in the directories in chronological order
        gen_files_era5 = os.listdir(gen_dir_era5)
        gen_files_danra = os.listdir(gen_dir_danra)


        # ERA5 startrs with 2000, DANRA with 2001
        # DANRA files are named as't2m_ave_YYYYMMDD.nc', ERA5 as 'temp_589x789_YYYYMMDD.npz'
        # Select only files from the years 2001-2005 and avoid .DS_Store file
        gen_files_era5 = [file for file in gen_files_era5 if (file != '.DS_Store') and (int(file[-12:-8]) >= self.year_start) and (int(file[-12:-8]) <= self.year_end)]
        gen_files_danra = [file for file in gen_files_danra if (file != '.DS_Store') and (int(file[-11:-7]) >= self.year_start) and (int(file[-11:-7]) <= self.year_end)]
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
        print(f'\n\nSelecting {self.n_samples_gen} random dates to generate samples from...')
        
        # Check that there are enough files to generate n_samples_gen samples
        if self.n_samples_gen > len(gen_files_era5_dates):
            # If not, generate as many samples as there are files
            print(f'Not enough files to generate {self.n_samples_gen} samples, generating {len(gen_files_era5_dates)} samples instead!')
            self.n_samples_gen = len(gen_files_era5_dates)

        # Select n random dates from the dates in the two datasets
        gen_dates = np.random.choice(gen_files_era5_dates, size=self.n_samples_gen, replace=False)
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

        eval_npz_dir_era5 = self.path_data + 'data_ERA5/size_' + self.danra_size_str + '/' + self.var + '_' + self.danra_size_str + '/eval'
        eval_nc_dir_danra = self.path_data + 'data_DANRA/size_' + self.danra_size_str + '/' + self.var + '_' + self.danra_size_str + '/eval'


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
        self.data_dir_danra_eval_w_cutouts_zarr = self.path_data + 'data_DANRA/size_' + self.danra_size_str + '/' + self.var + '_' + self.danra_size_str + '/zarr_files/' + self.var + '_' + self.danra_size_str + '_eval.zarr'
        self.data_dir_era5_eval_zarr = self.path_data + 'data_ERA5/size_' + self.danra_size_str + '/' + self.var + '_' + self.danra_size_str + '/zarr_files/' + self.var + '_' + self.danra_size_str + '_eval.zarr'

        # Convert .nc files in DANRA eval dir to zarr files
        convert_nc_to_zarr(eval_nc_dir_danra, self.data_dir_danra_eval_w_cutouts_zarr)
        # Convert .npz files in ERA5 eval dir to zarr files
        convert_npz_to_zarr(eval_npz_dir_era5, self.data_dir_era5_eval_zarr)

        print('\n\nData setup complete!')



    def setup_data_loader(self, n_gen_samples):
        '''
            Method to setup the data loader for generation.
        '''
        # Call the setup_data_folder method to setup the data
        self.setup_data_folder(n_gen_samples)

        # Create evaluation dataset
        eval_dataset = DANRA_Dataset_cutouts_ERA5_Zarr(data_dir_zarr=self.data_dir_danra_eval_w_cutouts_zarr, 
                                                data_size = self.image_dim, 
                                                n_samples = self.n_samples_gen, 
                                                cache_size = self.cache_size, 
                                                variable=self.var,
                                                scale=self.scaling,
                                                scale_mean=8.69251,
                                                scale_std=6.192434,
                                                scale_max=160.0,
                                                scale_min=0.0,
                                                shuffle=True,
                                                conditional_seasons=self.condition_on_seasons,
                                                conditional_images=self.condition_on_img,
                                                cond_dir_zarr=self.data_dir_era5_eval_zarr,
                                                n_classes=self.n_seasons, 
                                                cutouts=self.CUTOUTS, 
                                                cutout_domains=self.CUTOUT_DOMAINS,
                                                lsm_full_domain=self.data_lsm,
                                                topo_full_domain=self.data_topo,
                                                sdf_weighted_loss = self.use_sdf_weighted_loss
                                                )
        

        # Make a dataloader with batch size equal to n
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=n_gen_samples, shuffle=False, num_workers=self.num_workers)



    def setup_model(self):
        '''
            Method to setup the SBGM model for generation.
        '''
        ###################################
        #                                 #
        # SETTING UP MODEL FOR GENERATION #
        #                                 #   
        ###################################        


        # Define the encoder and decoder from modules_DANRA_downscaling.py
        encoder = Encoder(self.input_channels, 
                            self.time_embedding,
                            cond_on_lsm=self.sample_w_lsm_topo,
                            cond_on_topo=self.sample_w_lsm_topo,
                            cond_on_img=self.sample_w_cond_img, 
                            cond_img_dim=(1, self.LR_SIZE, self.LR_SIZE), 
                            block_layers=[2, 2, 2, 2], 
                            num_classes=self.n_seasons,
                            n_heads=self.num_heads
                            )
        decoder = Decoder(self.last_fmap_channels, 
                            self.output_channels, 
                            self.time_embedding, 
                            n_heads=self.num_heads
                            )
        self.score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn, encoder=encoder, decoder=decoder)
        self.score_model = self.score_model.to(self.device)


        # Define the optimizer
        if self.optimizer == 'adam':
            optimizer = torch.optim.AdamW(self.score_model.parameters(),
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_decay)
        elif self.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.score_model.parameters(),
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_decay)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.score_model.parameters(),
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_decay)

        # Define training pipeline
        self.pipeline = TrainingPipeline_general(self.score_model,
                                            loss_fn,
                                            marginal_prob_std_fn,
                                            optimizer,
                                            self.device,
                                            weight_init=True,
                                            sdf_weighted_loss=self.sdf_weighted_loss
                                            )

        # Define the learning rate schedulerß
        if self.lr_scheduler is not None:
            lr_scheduler_params = self.lr_scheduler_params
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.pipeline.optimizer,
                                                                    'min',
                                                                    factor = lr_scheduler_params['factor'],
                                                                    patience = lr_scheduler_params['patience'],
                                                                    threshold = lr_scheduler_params['threshold'],
                                                                    min_lr = self.min_lr
                                                                    )
        
    def load_checkpoint(self):
        '''
            Method to load the checkpoint of the trained SBGM model.
        '''

        # Call the setup_model method to setup the model
        self.setup_model()

        # Define path to checkpoint
        checkpoint_dir = self.path_checkpoint

        name_checkpoint = self.save_str + '.pth.tar'

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
        self.pipeline.model.load_state_dict(best_model_state)
        self.pipeline.model.eval()





    def generate_multiple_samples(self, n_gen_samples, sampler='pc_sampler'):
        '''
            Method to generate multiple samples from the trained SBGM model.
        '''
        
        # Call the setup_data_loader method to setup the data loader
        self.setup_data_loader(n_gen_samples)

        # Call the load_checkpoint method to load the model checkpoint
        self.load_checkpoint()

        
        print(f"Generating {n_gen_samples} samples from the trained SBGM model...")


        for idx, samples in tqdm.tqdm(enumerate(self.eval_dataloader), total=len(self.eval_dataloader), position=0):
            sample_batch_size = n_gen_samples

            # Define sampler to use
            sampler_name = sampler # ['pc_sampler', 'Euler_Maruyama', 'ode_sampler']

            if sampler_name == 'pc_sampler':
                sampler = pc_sampler
            elif sampler_name == 'Euler_Maruyama':
                sampler = Euler_Maruyama_sampler
            elif sampler_name == 'ode_sampler':
                sampler = ode_sampler
            else:
                raise ValueError(f'Invalid sampler name: {sampler_name}. Please choose from: ["pc_sampler", "Euler_Maruyama", "ode_sampler"]')
            
            # Extract the samples
            eval_img, eval_season, eval_cond, eval_lsm, eval_topo, eval_sdf, eval_point = extract_samples(samples, self.device)
            
            # Generate images from model
            generated_samples = sampler(self.score_model,
                                        marginal_prob_std_fn,
                                        diffusion_coeff_fn,
                                        sample_batch_size,
                                        device=self.device,
                                        img_size=self.image_size,
                                        y=eval_season,
                                        cond_img=eval_cond,
                                        lsm_cond=eval_lsm,
                                        topo_cond=eval_topo,
                                        ).squeeze()
            generated_samples = generated_samples.detach().cpu()

            # Stop after first iteration, all samples are generated
            break


        # Save the generated and corresponding eval images
        print(f'\n\nSaving generated images to {self.PATH_GENERATED_SAMPLES}...')
        np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'gen_samples_' + str(n_gen_samples) + '_distinctSamples', generated_samples)
        np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'eval_samples_' + str(n_gen_samples) + '_distinctSamples', eval_img)
        np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'lsm_samples_' + str(n_gen_samples) + '_distinctSamples', eval_lsm)
        np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'cond_samples_' + str(n_gen_samples) + '_distinctSamples', eval_cond)
        np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'season_samples_' + str(n_gen_samples) + '_distinctSamples', eval_season)
        np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'point_samples_' + str(n_gen_samples) + '_distinctSamples', eval_point)

        print(f'\n\n{n_gen_samples} generated samples saved to {self.PATH_GENERATED_SAMPLES}!')


    def generate_single_sample(self, sampler='pc_sampler', idx=0, date=None):
        '''
            Method to generate a single sample from the trained SBGM model.
            So far, only generates a single sample from the first batch of the data loader.
            DEVELOPMENT NEEDED: Generate a single sample from a specific date/index in the gen_years list.
                Not straightforward: setup_data_folder needs to be modified to be able to select specific dates.
        '''

        # Call the setup_data_loader method to setup the data loader
        self.setup_data_loader(1)

        # Call the load_checkpoint method to load the model checkpoint
        self.load_checkpoint()

        print(f"Generating single (random) sample from the trained SBGM model...")

        for idx, samples in tqdm.tqdm(enumerate(self.eval_dataloader), total=len(self.eval_dataloader), position=0):
            sample_batch_size = 1

            # Define sampler to use
            sampler_name = sampler

            if sampler_name == 'pc_sampler':
                sampler = pc_sampler
            elif sampler_name == 'Euler_Maruyama':
                sampler = Euler_Maruyama_sampler
            elif sampler_name == 'ode_sampler':
                sampler = ode_sampler
            else:
                raise ValueError(f'Invalid sampler name: {sampler_name}. Please choose from: ["pc_sampler", "Euler_Maruyama", "ode_sampler"]')
            
            # Extract the sample
            eval_img, eval_season, eval_cond, eval_lsm, eval_topo, eval_sdf, eval_point = extract_samples(samples, self.device)
            
            # Generate images from model
            generated_samples = sampler(self.score_model,
                                        marginal_prob_std_fn,
                                        diffusion_coeff_fn,
                                        sample_batch_size,
                                        device=self.device,
                                        img_size=self.image_size,
                                        y=eval_season,
                                        cond_img=eval_cond,
                                        lsm_cond=eval_lsm,
                                        topo_cond=eval_topo,
                                        ).squeeze()
            generated_samples = generated_samples.detach().cpu()

            # Stop after first iteration, all samples are generated
            break


        # Save the generated and corresponding eval images
        print(f'\n\nSaving generated images to {self.PATH_GENERATED_SAMPLES}...')
        np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'gen_singleSample', generated_samples)
        np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'eval_singleSample', eval_img)
        np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'lsm_singleSample', eval_lsm)
        np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'cond_singleSample', eval_cond)
        np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'season_singleSample', eval_season)
        np.savez_compressed(self.PATH_GENERATED_SAMPLES + 'point_singleSample', eval_point)

        print(f'\n\nSingle generated sample saved to {self.PATH_GENERATED_SAMPLES}!')



    

    def generate_repeated_single_sample(self, n_repeats, sampler='pc_sampler', idx=0, date=None):
        '''
            Method to generate multiple samples from the trained SBGM model based on the same conditionals.
            Generates n_repeats samples from the first batch of the data loader, or from a specific date/index if implemented.
        '''

        # Call the setup_data_loader method to setup the data loader
        self.setup_data_loader(1)

        # Call the load_checkpoint method to load the model checkpoint
        self.load_checkpoint()

        print(f"Generating {n_repeats} samples from the trained SBGM model with the same conditionals...")

        for idx, samples in tqdm.tqdm(enumerate(self.eval_dataloader), total=len(self.eval_dataloader), position=0):
            sample_batch_size = 1

            # Define sampler to use
            sampler_name = sampler

            if sampler_name == 'pc_sampler':
                sampler = pc_sampler
            elif sampler_name == 'Euler_Maruyama':
                sampler = Euler_Maruyama_sampler
            elif sampler_name == 'ode_sampler':
                sampler = ode_sampler
            else:
                raise ValueError(f'Invalid sampler name: {sampler_name}. Please choose from: ["pc_sampler", "Euler_Maruyama", "ode_sampler"]')

            # Extract the sample
            eval_img, eval_season, eval_cond, eval_lsm, eval_topo, eval_sdf, eval_point = extract_samples(samples, self.device)

            # Initialize a list to store generated samples
            generated_samples_list = []

            # Generate multiple samples
            for _ in range(n_repeats):
                # Generate images from model
                generated_sample = sampler(self.score_model,
                                        marginal_prob_std_fn,
                                        diffusion_coeff_fn,
                                        sample_batch_size,
                                        device=self.device,
                                        img_size=self.image_size,
                                        y=eval_season,
                                        cond_img=eval_cond,
                                        lsm_cond=eval_lsm,
                                        topo_cond=eval_topo,
                                        ).squeeze()
                generated_sample = generated_sample.detach().cpu()
                generated_samples_list.append(generated_sample)

            # Convert list of generated samples to a tensor
            generated_samples = torch.stack(generated_samples_list)

            # Stop after first iteration, all samples are generated
            break

        # Save the generated and corresponding eval images
        print(f'\n\nSaving {n_repeats} generated images to {self.PATH_GENERATED_SAMPLES}...')
        np.savez_compressed(self.PATH_GENERATED_SAMPLES + f'gen_repeatedSamples_{n_repeats}', generated_samples)
        np.savez_compressed(self.PATH_GENERATED_SAMPLES + f'eval_repeatedSamples_{n_repeats}', eval_img)
        np.savez_compressed(self.PATH_GENERATED_SAMPLES + f'lsm_repeatedSamples_{n_repeats}', eval_lsm)
        np.savez_compressed(self.PATH_GENERATED_SAMPLES + f'cond_repeatedSamples_{n_repeats}', eval_cond)
        np.savez_compressed(self.PATH_GENERATED_SAMPLES + f'season_repeatedSamples_{n_repeats}', eval_season)
        np.savez_compressed(self.PATH_GENERATED_SAMPLES + f'point_repeatedSamples_{n_repeats}', eval_point)

        print(f'\n\n{n_repeats} generated samples saved to {self.PATH_GENERATED_SAMPLES}!')
