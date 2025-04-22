import os, tqdm, torch
import torch.nn as nn
import copy

from torch.cuda.amp import autocast, GradScaler

from utils import *
from data_modules import *
from score_unet import *
from score_sampling import *

class TrainingPipeline_general_new:
    '''
        Class for building a training pipeline for the SBGM.
        To run through the training batches in one epoch.
    '''

    def __init__(self,
                 model,
                 loss_fn,
                 marginal_prob_std_fn,
                 optimizer,
                 device,
                 weight_init=None,
                 custom_weight_initializer=None,
                 sdf_weighted_loss=False,
                 with_ema=False
                 ):
        '''
            Initialize the training pipeline.
            Args:
                model: PyTorch model to be trained.
                loss_fn: Loss function for the model.
                optimizer: Optimizer for the model.
                device: Device to run the model on.
                weight_init: Weight initialization method.
                custom_weight_initializer: Custom weight initialization method.
                sdf_weighted_loss: Boolean to use SDF weighted loss.
        '''

        # Set class variables
        self.model = model
        self.loss_fn = loss_fn
        self.marginal_prob_std_fn = marginal_prob_std_fn
        self.optimizer = optimizer
        self.weight_init = weight_init
        self.custom_weight_initializer = custom_weight_initializer
        self.sdf_weighted_loss = sdf_weighted_loss
        self.with_ema = with_ema

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Initialize weights if needed
        if self.weight_init:
            if self.custom_weight_initializer:
                self.model.apply(self.custom_weight_initializer)
            else:
                self.model.apply(self.xavier_init_weights)

        # Set Exponential Moving Average (EMA) if needed
        if self.with_ema:
            # Create a copy of the model for EMA
            self.ema_model = copy.deepcopy(self.model)
            # Detach the EMA model parameters to not update them
            for param in self.ema_model.parameters():
                param.detach_()

    def xavier_init_weights(self, m):
        '''
            Xavier weight initialization.
            Args:
                m: Model to initialize weights for.
        '''

        # Check if the layer is a linear or convolutional layer
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            # Initialize weights with Xavier uniform
            nn.init.xavier_uniform_(m.weight)
            # If model has bias, initialize with 0.01 constant
            if torch.is_tensor(m.bias):
                m.bias.data.fill_(0.01)

    def save_model(self,
                   dirname='./model_params',
                   filename='SBGM.pth'
                   ):
        '''
            Save the model parameters.
            Args:
                dirname: Directory to save the model parameters.
                filename: Filename to save the model parameters.
        '''
        # Create directory if it does not exist
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # Set state dictionary to save
        state_dicts = {
            'network_params': self.model.state_dict(),
            'optimizer_params': self.optimizer.state_dict()
        }

        return torch.save(state_dicts, os.path.join(dirname, filename))
    
    def train(self,
              dataloader,
              verbose=True,
              PLOT_FIRST=False,
              SAVE_PATH='./',
              SAVE_NAME='upsampled_image.png',
              use_mixed_precision=False
              ):
        '''
            Method to run through the training batches in one epoch.
            Args:
                dataloader: Dataloader to run through.
                verbose: Boolean to print progress.
                PLOT_FIRST: Boolean to plot the first image.
                SAVE_PATH: Path to save the image.
                SAVE_NAME: Name of the image to save.
                use_mixed_precision: Boolean to use mixed precision training.
        '''

        # Set model to training mode
        self.model.train()
        # Set initial loss to 0
        loss = 0.0

        # Check if cuda is available and set scaler for mixed precision training if needed
        if torch.cuda.is_available() and use_mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Get information on variable for plotting
        var = dataloader.dataset.hr_variable

        # Set colormaps depending on variable
        if var == 'temp':
            cmap = 'plasma'
            cmap_label = 'Temperature [°C]'
        elif var == 'prcp':
            cmap = 'inferno'
            cmap_label = 'Precipitation [mm/day]'

        # Iterate through batches in dataloader (tuple of images and seasons)
        for idx, samples in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            # Samples is a dict with following available keys: 'img', 'classifier', 'img_cond', 'lsm', 'sdf', 'topo', 'points'
            # Extract samples
            x, seasons, cond_images, lsm, sdf, topo, hr_points, lr_points = extract_samples(samples, self.device)

            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Use mixed precision training if needed
            if self.scaler:
                with autocast():
                    # Pass the score model and samples+conditions to the loss_fn
                    batch_loss = loss_fn(self.model,
                                x,
                                self.marginal_prob_std_fn,
                                y = seasons,
                                cond_img = cond_images,
                                lsm_cond = lsm,
                                topo_cond = topo,
                                sdf_cond = sdf)
                # Mixed precision: scale loss and update weights
                self.scaler.scale(batch_loss).backward()
                # Update weights
                self.scaler.step(self.optimizer)
                # Update scaler
                self.scaler.update()
            else:
                batch_loss = loss_fn(self.model,
                            x,
                            self.marginal_prob_std_fn,
                            y = seasons,
                            cond_img = cond_images,
                            lsm_cond = lsm,
                            topo_cond = topo,
                            sdf_cond = sdf)
                # Backward pass
                batch_loss.backward()
                # Update weights
                self.optimizer.step()

            # Add batch loss to total loss
            loss += batch_loss.item()

        # Calculate average loss
        avg_loss = loss / len(dataloader)

        # Print average loss if verbose
        if verbose:
            print(f'Training Loss: {avg_loss}')

        return avg_loss
    
    def validate(self,
                 dataloader,
                 verbose=True
                 ):
        '''
            Method to run through the validation batches in one epoch.
            Args:
                dataloader: Dataloader to run through.
                verbose: Boolean to print progress.
        '''

        # Set model to evaluation mode
        self.model.eval()
        # Set initial loss to 0
        loss = 0.0

        # Iterate through batches in dataloader (tuple of images and seasons)
        for idx, samples in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            # Samples is a dict with following available keys: 'img', 'classifier', 'img_cond', 'lsm', 'sdf', 'topo', 'points'
            # Extract samples
            x, seasons, cond_images, lsm, sdf, topo, _ = extract_samples(samples, self.device)

            # Pass the score model and samples+conditions to the loss_fn
            batch_loss = loss_fn(self.model,
                           x,
                           self.marginal_prob_std_fn,
                           y = seasons,
                           cond_img = cond_images,
                           lsm_cond = lsm,
                           topo_cond = topo,
                           sdf_cond = sdf)

            # Add batch loss to total loss
            loss += batch_loss.item()

        # Calculate average loss
        avg_loss = loss / (idx + 1)

        # Print average loss if verbose
        if verbose:
            print(f'Validation Loss: {avg_loss}')

        return avg_loss




class TrainingPipeline_general:
    '''
        Class for building a training pipeline for the SBGM.
        To run through the training batches in one epoch.
    '''

    def __init__(self,
                 model,
                 loss_fn,
                 marginal_prob_std_fn,
                 optimizer,
                 device,
                 weight_init=None,
                 custom_weight_initializer=None,
                 sdf_weighted_loss=False,
                 with_ema=False
                 ):
        '''
            Initialize the training pipeline.
            Args:
                model: PyTorch model to be trained.
                loss_fn: Loss function for the model.
                optimizer: Optimizer for the model.
                device: Device to run the model on.
                weight_init: Weight initialization method.
                custom_weight_initializer: Custom weight initialization method.
                sdf_weighted_loss: Boolean to use SDF weighted loss.
        '''

        # Set class variables
        self.model = model
        self.loss_fn = loss_fn
        self.marginal_prob_std_fn = marginal_prob_std_fn
        self.optimizer = optimizer
        self.weight_init = weight_init
        self.custom_weight_initializer = custom_weight_initializer
        self.sdf_weighted_loss = sdf_weighted_loss
        self.with_ema = with_ema

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Initialize weights if needed
        if self.weight_init:
            if self.custom_weight_initializer:
                self.model.apply(self.custom_weight_initializer)
            else:
                self.model.apply(self.xavier_init_weights)

        # Set Exponential Moving Average (EMA) if needed
        if self.with_ema:
            # Create a copy of the model for EMA
            self.ema_model = copy.deepcopy(self.model)
            # Detach the EMA model parameters to not update them
            for param in self.ema_model.parameters():
                param.detach_()

    def xavier_init_weights(self, m):
        '''
            Xavier weight initialization.
            Args:
                m: Model to initialize weights for.
        '''

        # Check if the layer is a linear or convolutional layer
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            # Initialize weights with Xavier uniform
            nn.init.xavier_uniform_(m.weight)
            # If model has bias, initialize with 0.01 constant
            if torch.is_tensor(m.bias):
                m.bias.data.fill_(0.01)

    def save_model(self,
                   dirname='./model_params',
                   filename='SBGM.pth'
                   ):
        '''
            Save the model parameters.
            Args:
                dirname: Directory to save the model parameters.
                filename: Filename to save the model parameters.
        '''
        # Create directory if it does not exist
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # Set state dictionary to save
        state_dicts = {
            'network_params': self.model.state_dict(),
            'optimizer_params': self.optimizer.state_dict()
        }

        return torch.save(state_dicts, os.path.join(dirname, filename))
    
    def train(self,
              dataloader,
              verbose=True,
              PLOT_FIRST=False,
              SAVE_PATH='./',
              SAVE_NAME='upsampled_image.png',
              use_mixed_precision=False
              ):
        '''
            Method to run through the training batches in one epoch.
            Args:
                dataloader: Dataloader to run through.
                verbose: Boolean to print progress.
                PLOT_FIRST: Boolean to plot the first image.
                SAVE_PATH: Path to save the image.
                SAVE_NAME: Name of the image to save.
                use_mixed_precision: Boolean to use mixed precision training.
        '''

        # Set model to training mode
        self.model.train()
        # Set initial loss to 0
        loss = 0.0

        # Check if cuda is available and set scaler for mixed precision training if needed
        if torch.cuda.is_available() and use_mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Get information on variable for plotting
        var = dataloader.dataset.variable

        # Set colormaps depending on variable
        if var == 'temp':
            cmap = 'plasma'
            cmap_label = 'Temperature [°C]'
        elif var == 'prcp':
            cmap = 'inferno'
            cmap_label = 'Precipitation [mm/day]'

        # Iterate through batches in dataloader (tuple of images and seasons)
        for idx, samples in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            # Samples is a dict with following available keys: 'img', 'classifier', 'img_cond', 'lsm', 'sdf', 'topo', 'points'
            # Extract samples
            x, seasons, cond_images, lsm, sdf, topo, _ = extract_samples(samples, self.device)

            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Use mixed precision training if needed
            if self.scaler:
                with autocast():
                    # Pass the score model and samples+conditions to the loss_fn
                    batch_loss = loss_fn(self.model,
                                x,
                                self.marginal_prob_std_fn,
                                y = seasons,
                                cond_img = cond_images,
                                lsm_cond = lsm,
                                topo_cond = topo,
                                sdf_cond = sdf)
                # Mixed precision: scale loss and update weights
                self.scaler.scale(batch_loss).backward()
                # Update weights
                self.scaler.step(self.optimizer)
                # Update scaler
                self.scaler.update()
            else:
                batch_loss = loss_fn(self.model,
                            x,
                            self.marginal_prob_std_fn,
                            y = seasons,
                            cond_img = cond_images,
                            lsm_cond = lsm,
                            topo_cond = topo,
                            sdf_cond = sdf)
                # Backward pass
                batch_loss.backward()
                # Update weights
                self.optimizer.step()

            # Add batch loss to total loss
            loss += batch_loss.item()

        # Calculate average loss
        avg_loss = loss / len(dataloader)

        # Print average loss if verbose
        if verbose:
            print(f'Training Loss: {avg_loss}')

        return avg_loss
    
    def validate(self,
                 dataloader,
                 verbose=True
                 ):
        '''
            Method to run through the validation batches in one epoch.
            Args:
                dataloader: Dataloader to run through.
                verbose: Boolean to print progress.
        '''

        # Set model to evaluation mode
        self.model.eval()
        # Set initial loss to 0
        loss = 0.0

        # Iterate through batches in dataloader (tuple of images and seasons)
        for idx, samples in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            # Samples is a dict with following available keys: 'img', 'classifier', 'img_cond', 'lsm', 'sdf', 'topo', 'points'
            # Extract samples
            x, seasons, cond_images, lsm, sdf, topo, _ = extract_samples(samples, self.device)

            # Pass the score model and samples+conditions to the loss_fn
            batch_loss = loss_fn(self.model,
                           x,
                           self.marginal_prob_std_fn,
                           y = seasons,
                           cond_img = cond_images,
                           lsm_cond = lsm,
                           topo_cond = topo,
                           sdf_cond = sdf)

            # Add batch loss to total loss
            loss += batch_loss.item()

        # Calculate average loss
        avg_loss = loss / (idx + 1)

        # Print average loss if verbose
        if verbose:
            print(f'Validation Loss: {avg_loss}')

        return avg_loss


