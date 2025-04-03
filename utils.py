'''
    Utility functions for the project.

    Functions:
    ----------
    - model_summary: Print the model summary
    - SimpleLoss: Simple loss function
    - HybridLoss: Hybrid loss function
    - SDFWeightedMSELoss: Custom loss function for SDFs
    - convert_npz_to_zarr: Convert DANRA .npz files to zarr files
    - create_concatenated_data_files: Create concatenated data files
    - convert_npz_to_zarr_based_on_time_split: Convert DANRA .npz files to zarr files based on a time split
    - convert_npz_to_zarr_based_on_percent_split: Convert DANRA .npz files to zarr files based on a percentage split
    - convert_nc_to_zarr: Convert ERA5 .nc files to zarr files
    - extract_samples: Extract samples from dictionary
    
    For argparse:
    - str2bool: Convert string to boolean
    - str2list: Convert string to list
    - str2list_of_strings: Convert string to list of strings
    - str2dict: Convert string to dictionary

'''

import argparse
import torch 
import zarr
import os
import json

import netCDF4 as nc
import torch.nn as nn
import numpy as np


def model_summary(model):
    '''
        Simple function to print the model summary
    '''

    print("model_summary")
    print()
    print("Layer_name" + "\t"*7 + "Number of Parameters")
    print("="*100)
    
    model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
    layer_name = [child for child in model.children()]
    j = 0
    total_params = 0
    print("\t"*10)
    for i in layer_name:
        print()
        param = 0
        try:
            bias = (i.bias is not None)
        except:
            bias = False  
        if not bias:
            param =model_parameters[j].numel()+model_parameters[j+1].numel()
            j = j+2
        else:
            param =model_parameters[j].numel()
            j = j+1
        print(str(i) + "\t"*3 + str(param))
        total_params+=param
    print("="*100)
    print(f"Total Params:{total_params}")     


class SimpleLoss(nn.Module):
    def __init__(self):
        super(SimpleLoss, self).__init__()
        self.mse = nn.MSELoss()#nn.L1Loss()#$

    def forward(self, predicted, target):
        return self.mse(predicted, target)

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, T=10):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.T = T
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        loss = self.mse(predictions[-1], targets[0])
        
        for t in range(1, self.T):
            loss += self.alpha * self.mse(predictions[t-1], targets[t])
        
        return loss

class SDFWeightedMSELoss(nn.Module):
    '''
        Custom loss function for SDFs.

    '''
    def __init__(self, max_land_weight=1.0, min_sea_weight=0.5, device=None):
        super().__init__()
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.to(self.device)

        self.max_land_weight = max_land_weight
        self.min_sea_weight = min_sea_weight
#        self.mse = nn.MSELoss(reduction='none')

    def forward(self, input, target, sdf):
        # Convert SDF to weights, using a sigmoid function (or similar)
        # Scaling can be adjusted to control sharpness of transition
        weights = torch.sigmoid(sdf) * (self.max_land_weight - self.min_sea_weight) + self.min_sea_weight

        # Calculate the squared error
        input = input.to(self.device)
        target = target.to(self.device)
        squared_error = (input - target)**2

        squared_error = squared_error.to(self.device)
        weights = weights.to(self.device)

        # Apply the weights
        weighted_squared_error = weights * squared_error

        # Return mean of weighted squared error
        return weighted_squared_error.mean()
    

def convert_npz_to_zarr(npz_directory, zarr_file, VERBOSE=False):
    '''
        Function to convert DANRA .npz files to zarr files
        
        Parameters:
        -----------
        npz_directory: str
            Directory containing .npz files
        zarr_file: str
            Name of zarr file to be created
    '''
    print(f'\n\nConverting {len(os.listdir(npz_directory))} .npz files to zarr file...')

    # Create zarr group (equivalent to a directory) 
    zarr_group = zarr.open_group(zarr_file, mode='w')
    
    # Make iterator to keep track of progress
    i = 0

    # Loop through all .npz files in the .npz directory
    for npz_file in os.listdir(npz_directory):
        
        # Check if the file is a .npz file (not dir or .DS_Store)
        if npz_file.endswith('.npz'):
            if VERBOSE:
                print(os.path.join(npz_directory, npz_file))
            # Load the .npz file
            npz_data = np.load(os.path.join(npz_directory, npz_file))            

            # Loop through all keys in the .npz file
            for key in npz_data:

                if i == 0:
                    print(f'Key: {key}')
                # Save the data as a zarr array
                zarr_group.array(npz_file.replace('.npz', '') + '/' + key, npz_data[key], chunks=True, dtype=np.float32)
            
            # Print progress if iterator is a multiple of 100
            if (i+1) % 100 == 0:
                print(f'Converted {i+1} files...')
            i += 1


def create_concatenated_data_files(data_dir_all:list, data_dir_concatenated:str, variables:list, n_images:int=4):
    '''
        Function to create concatenated data files.
        The function will concatenate the data from the data_dir_all directories
        and save the concatenated data to the data_dir_concatenated directory.

        Parameters:
        -----------
        data_dir_all: list
            List of directories containing the data
        data_dir_concatenated: str
            Directory to save the concatenated data
        variables: list
            List of variables to concatenate
        n_images: int
            Number of images to concatenate
    '''
    print(f'\n\nCreating concatenated data files from {len(data_dir_all)} directories...')

    # Create zarr group (equivalent to a directory)
    zarr_group = zarr.open_group(data_dir_concatenated, mode='w')

    # Loop through all directories in the data_dir_all list
    for data_dir in data_dir_all:
        # Loop through all files in the directory
        for data_file in os.listdir(data_dir):
            # Check if the file is a .npz file (not dir or .DS_Store)
            if data_file.endswith('.npz'):
                # Load the .npz file
                npz_data = np.load(os.path.join(data_dir, data_file))
                # Loop through all variables in the .npz file
                for var in variables:
                    # Select the data from the variable
                    data = npz_data[var][:n_images]
                    # Save the data as a zarr array
                    zarr_group.array(data_file.replace('.npz', '') + '/' + var, data, chunks=True, dtype=np.float32)
    
    print(f'Concatenated data saved to {data_dir_concatenated}...')




class data_preperation():
    '''
        Class to handle data preperation for the project.
        All data is located in an /all/ directory.
        This class can handle:
            - Check if data already exists
            - Creating train, val and test splits (based on years or percentages) - and saving to zarr
            - Clean up after training (remove zarr files)

    '''
    def __init__(self, args):
        self.args = args
        self.data_dir_all = args.data_dir_all

    def create_concatenated_data_files(self, data_dir_all, data_dir_concatenated, n_images=4):
        '''
            Function to create concatenated data files.
            The function will concatenate the data from the data_dir_all directories
            and save the concatenated data to the data_dir_concatenated directory.

            Parameters:
            -----------
            data_dir_all: list
                List of directories containing the data
            data_dir_concatenated: str
                Directory to save the concatenated data
            n_images: int
                Number of images to concatenate
        '''
        print(f'\n\nCreating concatenated data files from {len(data_dir_all)} directories...')

        # Create zarr group (equivalent to a directory)
        zarr_group = zarr.open_group(data_dir_concatenated, mode='w')

        # Loop through all directories in the data_dir_all list
        for data_dir in data_dir_all:
            # Loop through all files in the directory
            for data_file in os.listdir(data_dir):
                # Check if the file is a .npz file (not dir or .DS_Store)
                if data_file.endswith('.npz'):
                    # Load the .npz file
                    npz_data = np.load(os.path.join(data_dir, data_file))
                    # Loop through all variables in the .npz file
                    for var in npz_data:
                        # Select the data from the variable
                        data = npz_data[var][:n_images]
                        # Save the data as a zarr array
                        zarr_group.array(data_file.replace('.npz', '') + '/' + var, data, chunks=True, dtype=np.float32)
        
        print(f'Concatenated data saved to {data_dir_concatenated}...')


def convert_npz_to_zarr_based_on_time_split(npz_directory:str, year_splits:list):
    '''
        Function to convert DANRA .npz files to zarr files based on a time split.
        The function will select the files based on the year_splits list.
        Will create train, val and test splits based on the years specified in the list.

        Parameters:
        -----------
        npz_directory: str
            Directory containing all .npz data files
        year_splits: list
            List of 3 lists containing the years for train, val and test splits
    '''
    # Print the years for each split
    # Print number of files in the split years
    print(f'\nTrain years: {year_splits[0]}')
    print(f'Number of files in train years: {len([f for f in os.listdir(npz_directory) if any(str(year) in f for year in year_splits[0])])}')

    print(f'\nVal years: {year_splits[1]}')
    print(f'Number of files in val years: {len([f for f in os.listdir(npz_directory) if any(str(year) in f for year in year_splits[1])])}')

    print(f'\nTest years: {year_splits[2]}')
    print(f'Number of files in test years: {len([f for f in os.listdir(npz_directory) if any(str(year) in f for year in year_splits[2])])}')

    # 



def convert_npz_to_zarr_based_on_percent_split(npz_directory:str, percent_splits:list, random_selection:bool=False):
    '''
        Function to convert DANRA .npz files to zarr files based on a percentage split.
        The function will randomly select the percentage of files specified in the percent_splits list
        or select the first files in the directory.
        Will create train, val and test splits based on the percentages specified in the list.
        
        Parameters:
        -----------
        npz_directory: str
            Directory containing .npz files
        percent_splits: list
            List of floats representing the percentage splits
    
    '''
    print(f'\n\nConverting {len(os.listdir(npz_directory))} .npz files to zarr file...')



def convert_nc_to_zarr(nc_directory, zarr_file, VERBOSE=False):
    '''
        Function to convert ERA5 .nc files to zarr files
        
        Parameters:
        -----------
        nc_directory: str
            Directory containing .nc files
        zarr_file: str
            Name of zarr file to be created
    '''
    print(f'Converting {len(os.listdir(nc_directory))} .nc files to zarr file...')
    # Create zarr group (equivalent to a directory)
    zarr_group = zarr.open_group(zarr_file, mode='w')
    
    # Loop through all .nc files in the .nc directory 
    for nc_file in os.listdir(nc_directory):
        # Check if the file is a .nc file (not dir or .DS_Store)
        if nc_file.endswith('.nc'):
            if VERBOSE:
                print(os.path.join(nc_directory, nc_file))
            # Load the .nc file
            nc_data = nc.Dataset(os.path.join(nc_directory, nc_file))
            # Loop through all variables in the .nc file
            for var in nc_data.variables:
                # Select the data from the variable
                data = nc_data[var][:]
                # Save the data as a zarr array
                zarr_group.array(nc_file.replace('.nc', '') + '/' + var, data, chunks=True, dtype=np.float32)


def extract_samples(samples, device=None):
    '''
        Function to extract samples from dictionary.
        Returns the samples as variables.
        If not in dictionary, returns None.
        Also sends the samples to the device and converts to float.
    '''
    images = None
    seasons = None
    cond_images = None
    lsm = None
    sdf = None
    topo = None

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

    if 'img' in samples.keys() and samples['img'] is not None:
        images = samples['img'].to(device)
        images = samples['img'].to(torch.float)
    else:
        # Stop if no images (images are required)
        raise ValueError('No images in samples dictionary.')
    
    if 'classifier' in samples.keys() and samples['classifier'] is not None:
        seasons = samples['classifier'].to(device)

    if 'img_cond' in samples.keys() and samples['img_cond'] is not None:
        cond_images = samples['img_cond'].to(device)
        cond_images = samples['img_cond'].to(torch.float)

    if 'lsm' in samples.keys() and samples['lsm'] is not None:
        lsm = samples['lsm'].to(device)
        lsm = samples['lsm'].to(torch.float)
    
    if 'sdf' in samples.keys() and samples['sdf'] is not None:
        sdf = samples['sdf'].to(device)
        sdf = samples['sdf'].to(torch.float)
    
    if 'topo' in samples.keys() and samples['topo'] is not None:
        topo = samples['topo'].to(device)
        topo = samples['topo'].to(torch.float)
    
    if 'point' in samples.keys() and samples['point'] is not None:
        point = samples['point'].to(device)
        point = samples['point'].to(torch.float)
        # Print type of topo
        #print(f'Topo is of type: {topo.dtype}')
    else:
        point = None

    return images, seasons, cond_images, lsm, sdf, topo, point



def str2bool(v):
    '''
        Function to convert string to boolean.
        Used for argparse.
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def str2list(v):
    """
    Convert a string to a list.
    If the string is in JSON list format (begins with '[' and ends with ']'),
    this function uses json.loads to parse it. Otherwise, it splits the string
    on commas and tries to convert each element to an integer, then float,
    and if neither conversion applies, leaves it as a string.
    
    Examples:
      - '[1,2,3]' --> [1, 2, 3]
      - '1,2,3'   --> [1, 2, 3]
      - '["a", "b", "c"]' --> ["a", "b", "c"]
      - '1.1,2.2,3.3' --> [1.1, 2.2, 3.3]
    """
    v = v.strip()
    if v.startswith('[') and v.endswith(']'):
        try:
            return json.loads(v)
        except Exception:
            # If JSON loading fails, fallback to splitting
            pass
    # Fallback: split on commas and try to convert each element.
    items = [x.strip() for x in v.split(',')]
    result = []
    for x in items:
        try:
            result.append(int(x))
        except ValueError:
            try:
                result.append(float(x))
            except ValueError:
                result.append(x)
    return result    

# def str2list(v):
#     '''
#         Function to convert string to list.
#         Used for argparse.
#     '''
#     try:
#         # Try to split commas and convert each part to an integer
#         return [int(i) for i in v.split(',')]
#     except:
#         # If there is a ValueError, it means the list contains floats or strings
#         return [float(x) if '.' in x else x for x in v.split(',')]
    

def str2list_of_strings(v):
    # Split the input string by commas, strip any extra whitespace around the strings
    return [s.strip() for s in v.split(',')]

import ast
def str2dict(v):
    try:
        # Try to safely evaluate the string as a Python literal
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Invalid dictionary format.")