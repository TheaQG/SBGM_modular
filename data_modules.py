"""
    Script for generating a pytorch dataset for the DANRA data.
    The dataset is loaded as a single-channel image - either prcp or temp.
    Different transforms are be applied to the dataset.
    A custom transform is used to scale the data to a new interval.
"""

# Import libraries and modules 
from special_transforms import *
import zarr
import random, torch
import numpy as np
import multiprocessing
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import distance_transform_edt as distance
from special_transforms import PrcpLogTransform, ZScoreTransform

def preprocess_lsm_topography(lsm_path, topo_path, target_size, scale=False, flip=False):
    '''
        Preprocess the lsm and topography data.
        Function loads the data, converts it to tensors, normalizes the topography data to [0, 1] interval,
        and upscales the data to match the target size.

        Input:
            - lsm_path: path to lsm data
            - topo_path: path to topography data
            - target_size: tuple containing the target size of the data
    '''
    # 1. Load the Data and flip upside down if flip=True
    if flip:
        lsm_data = np.flipud(np.load(lsm_path)['data']).copy() # Copy to avoid negative strides
        topo_data = np.flipud(np.load(topo_path)['data']).copy() # Copy to avoid negative strides
        
    else:
        lsm_data = np.load(lsm_path)['data']
        topo_data = np.load(topo_path)['data']

    # 2. Convert to Tensors
    lsm_tensor = torch.tensor(lsm_data).float().unsqueeze(0)  # Add channel dimension
    topo_tensor = torch.tensor(topo_data).float().unsqueeze(0)
    
    if scale: # SHOULD THIS ALSO BE A Z-SCALE TRANSFORM?
        # 3. Normalize Topography to [0, 1] interval
        topo_tensor = (topo_tensor - topo_tensor.min()) / (topo_tensor.max() - topo_tensor.min())
    
    # 4. Upscale the Fields to match target size
    resize_transform = transforms.Resize(target_size, antialias=True)
    lsm_tensor = resize_transform(lsm_tensor)
    topo_tensor = resize_transform(topo_tensor)
    
    return lsm_tensor, topo_tensor

def preprocess_lsm_topography_from_data(lsm_data, topo_data, target_size, scale=True):
    '''
        Preprocess the lsm and topography data.
        Function loads the data, converts it to tensors, normalizes the topography data to[0, 1] interval (if scale=True)),
        and upscales the data to match the target size.

        Input:
            - lsm_data: lsm data
            - topo_data: topography data
            - target_size: tuple containing the target size of the data
            - scale: whether to scale the topography data to [0, 1] interval
    '''    
    # 1. Convert to Tensors
    lsm_tensor = torch.tensor(lsm_data.copy()).float().unsqueeze(0)  # Add channel dimension
    topo_tensor = torch.tensor(topo_data.copy()).float().unsqueeze(0)
    
    if scale:
        # 2. Normalize Topography to [0, 1] interval
        topo_tensor = (topo_tensor - topo_tensor.min()) / (topo_tensor.max() - topo_tensor.min())
    
    # 3. Upscale the Fields to match target size
    resize_transform = transforms.Resize(target_size, antialias=True)
    lsm_tensor = resize_transform(lsm_tensor)
    topo_tensor = resize_transform(topo_tensor)
    
    return lsm_tensor, topo_tensor

def generate_sdf(mask):
    # Ensure mask is boolean
    binary_mask = mask > 0 

    # Distance transform for sea
    dist_transform_sea = distance(~binary_mask)

    # Set land to 1 and subtract sea distances
    sdf = 10*binary_mask.astype(np.float32) - dist_transform_sea

    return sdf

def normalize_sdf(sdf):
    # Find min and max in the SDF
    min_val = np.min(sdf)
    max_val = np.max(sdf)

    # Normalize the SDF
    sdf_normalized = (sdf - min_val) / (max_val - min_val)

    return sdf_normalized

class DateFromFile:
    '''
    General class for extracting date from filename.
    Can take .npz, .nc and .zarr files.
    Not dependent on the file extension.
    '''
    def __init__(self, filename):
        # Remove file extension
        self.filename = filename.split('.')[0]
        # Get the year, month and day from filename ending (YYYYMMDD)
        self.year = int(self.filename[-8:-4])
        self.month = int(self.filename[-4:-2])
        self.day = int(self.filename[-2:])

    def determine_season(self):
        # Determine season based on month
        if self.month in [3, 4, 5]:
            return 0
        elif self.month in [6, 7, 8]:
            return 1
        elif self.month in [9, 10, 11]:
            return 2
        else:
            return 3

    def determine_month(self):
        # Returns the month as an integer in the interval [0, 11]
        return self.month - 1

    @staticmethod
    def is_leap_year(year):
        """Check if a year is a leap year"""
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return True
        return False

    def determine_day(self):
        # Days in month for common years and leap years
        days_in_month_common = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        days_in_month_leap = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # Determine if the year is a leap year
        if self.is_leap_year(self.year):
            days_in_month = days_in_month_leap
        else:
            days_in_month = days_in_month_common

        # Compute the day of the year
        day_of_year = sum(days_in_month[:self.month]) + self.day - 1  # "-1" because if it's January 1st, it's the 0th day of the year
        return day_of_year
    


def find_rand_points(rect, crop_dim):
    '''
    Function to find random quadrants in a given rectangle
    '''
    x1 = rect[0]
    x2 = rect[1]
    y1 = rect[2]
    y2 = rect[3]

    d = crop_dim 

    l_x = x2 - x1
    l_y = y2 - y1

    a_x = l_x - d
    a_y = l_y - d

    x_rand = random.randint(0, a_x)
    y_rand = random.randint(0,a_y)

    x1_new = x1 + x_rand
    x2_new = x1_new + d
    y1_new = y1 + y_rand 
    y2_new = y1_new + d 

    point = [x1_new, x2_new, y1_new, y2_new]
    return point



class DANRA_Dataset_cutouts_ERA5_Zarr(Dataset):
    '''
        Class for setting the DANRA dataset with option for random cutouts from specified domains.
        Along with DANRA data, the land-sea mask and topography data is also loaded at same cutout.
        Possibility to sample more than n_samples if cutouts are used.
        Option to shuffle data or load sequentially.
        Option to scale data to new interval.
        Option to use conditional (classifier) sampling (season, month or day).
    '''
    def __init__(self, 
                data_dir_zarr:str,                  # Path to data
                data_size:tuple,                    # Size of data (2D image, tuple)
                n_samples:int = 365,                # Number of samples to load
                cache_size:int = 365,               # Number of samples to cache
                variable:str = 'temp',              # Variable to load (temp or prcp)
                shuffle:bool = False,               # Whether to shuffle data (or load sequentially)
                cutouts:bool = False,               # Whether to use cutouts 
                cutout_domains:list = None,         # Domains to use for cutouts
                n_samples_w_cutouts:int = None,     # Number of samples to load with cutouts (can be greater than n_samples)
                lsm_full_domain = None,             # Land-sea mask of full domain
                topo_full_domain = None,            # Topography of full domain
                sdf_weighted_loss:bool = False,     # Whether to use weighted loss for SDF
                scale:bool = True,                  # Whether to scale data to new interval
                save_original:bool = False,         # Whether to save original data
                scale_mean:float = 8.69251,         # Global mean of data for scaling
                scale_std:float = 6.192434,         # Global standard deviation of data for scaling
                scale_type_prcp:str = 'log_zscore', # Type of scaling for precipitation data
                scale_min:float = 0,                # Global minimum value of data for scaling (in mm) - in precipitation data
                scale_max:float = 160,              # Global maximum value of data for scaling (in mm) - in precipitation data
                scale_min_log:float = 1e-10,        # Global minimum value for log transform
                scale_max_log:float = 1,            # Maximum value for log transform
                scale_mean_log:float = 0,           # Global mean value for log transform
                scale_std_log:float = 1,            # Global standard deviation for log transform
                buffer_frac:float = 0.5,            # Percentage buffer_frac for scaling
                conditional_seasons:bool = False,   # Whether to use seasonal conditional sampling
                conditional_images:bool = False,    # Whether to use image conditional sampling
                cond_dir_zarr:str = None,           # Path to directory containing conditional data
                n_classes:int = None                # Number of classes for conditional sampling
                ):                       
        '''n_samples_w_cutouts

        '''
        self.data_dir_zarr = data_dir_zarr
        self.n_samples = n_samples
        self.data_size = data_size
        self.cache_size = cache_size
        self.variable = variable
        
        self.scale = scale
        self.save_original = save_original
        self.scale_mean = scale_mean
        self.scale_std = scale_std

        self.scale_type_prcp = scale_type_prcp
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_min_log = scale_min_log
        self.scale_max_log = scale_max_log
        self.scale_mean_log = scale_mean_log
        self.scale_std_log = scale_std_log
        self.buffer_frac = buffer_frac
        
        self.shuffle = shuffle
        self.cutouts = cutouts
        self.cutout_domains = cutout_domains
        if n_samples_w_cutouts is None:
            self.n_samples_w_cutouts = self.n_samples
        else:
            self.n_samples_w_cutouts = n_samples_w_cutouts
        
        self.lsm_full_domain = lsm_full_domain
        self.topo_full_domain = topo_full_domain
        self.sdf_weighted_loss = sdf_weighted_loss
        
        self.conditional_seasons = conditional_seasons
        self.conditional_images = conditional_images
        self.cond_dir_zarr = cond_dir_zarr
        #self.zarr_group_cond = zarr_group_cond
        self.n_classes = n_classes

        if self.variable == 'prcp':
            print('DANRA data in [mm], ERA5 data in [m]')
            print('Converting ERA5 data to [mm] by multiplying by 1000\n')
            if self.scale:
                print('Using PrcpLogTransform to scale data to new interval')
                print(f'Original interval in [mm]: [{self.scale_min}, {self.scale_max}]')
                print(f'Original interval in [log(mm)]: [{self.scale_min_log}, {self.scale_max_log}]')
                if self.scale_type_prcp == 'log_zscore':
                    print(f'Using Z-score transform to scale data to new distribution')
                    print(f'Old distribution: mean={self.scale_mean_log}, std={self.scale_std_log}')
                    print(f'New distribution: mean=0, std=1\n\n')
                elif self.scale_type_prcp == 'log_01':
                    log_range = self.scale_max_log - self.scale_min_log
                    scale_buffer_min = self.scale_min_log - self.buffer_frac*log_range
                    scale_buffer_max = self.scale_max_log + self.buffer_frac*log_range
                    print(f'Using log_01 transform to scale data to new interval, with buffer fraction {buffer_frac}')
                    print(f'Old interval: [{self.scale_min_log}, {self.scale_max_log}], ([{scale_buffer_min}, {scale_buffer_max}])')
                    print(f'New interval: [0, 1]\n\n')
                elif self.scale_type_prcp == 'log_minus1_1':
                    log_range = self.scale_max_log - self.scale_min_log
                    scale_buffer_min = self.scale_min_log - self.buffer_frac*log_range
                    scale_buffer_max = self.scale_max_log + self.buffer_frac*log_range
                    print(f'Using log_minus1_1 transform to scale data to new interval with buffer fraction {buffer_frac}')
                    print(f'Old interval: [{self.scale_min_log}, {self.scale_max_log}] ([{scale_buffer_min}, {scale_buffer_max}])')
                    print(f'New interval: [-1, 1]\n\n')
                
        elif self.variable == 'temp':
            print('DANRA and  ERA5 data in [Kelvin]')
            print('Converting data to [Celsius] by subtracting 273.15\n')
            if self.scale:
                print('Using Z-score transform to scale data to new distribution')
                print(f'Old distribution: mean={self.scale_mean}, std={self.scale_std}')
                print(f'New distribution: mean=0, std=1\n\n')

        # Make zarr groups of data
        self.zarr_group_img = zarr.open_group(data_dir_zarr, mode='r')

        # Load files from directory (both data and conditional data)
        self.files = list(self.zarr_group_img.keys())
        
        if self.conditional_images:
            # If no conditional images are used, use mean of samples as conditional image
            if self.cond_dir_zarr is None:
                self.files_cond = self.files
            # If using individual samples as conditional images
            else:
                self.zarr_group_cond = zarr.open_group(cond_dir_zarr)
                self.files_cond = list(self.zarr_group_cond.keys())

        # If not using cutouts, no possibility to sample more than n_samples
        if self.cutouts == False:
            # If shuffle is True, sample n_samples randomly
            if self.shuffle:
                n_samples = min(len(self.files), len(self.files_cond))
                random_idx = random.sample(range(n_samples), n_samples)
                self.files = [self.files[i] for i in random_idx]
                # If using conditional images, also sample conditional images randomly
                if self.conditional_images:
                    # If no conditional images are used, use mean of samples as conditional image
                    if self.cond_dir_zarr is None:
                        self.files_cond = [self.files[i] for i in random_idx]
                    # If using individual samples as conditional images
                    else:
                        self.files_cond = [self.files_cond[i] for i in random_idx]
            # If shuffle is False, sample n_samples sequentially
            else:
                self.files = self.files[0:n_samples]
                # If using conditional images, also sample conditional images sequentially
                if self.conditional_images:
                    # If no conditional images are used, use mean of samples as conditional image
                    if self.cond_dir_zarr is None:
                        self.files_cond = self.files[0:n_samples]
                    # If using individual samples as conditional images
                    else:
                        self.files_cond = self.files_cond[0:n_samples]
        # If using cutouts, possibility to sample more than n_samples
        else:
            # If shuffle is True, sample n_samples randomly
            if self.shuffle:
                # If no conditional samples are given, n_samples equal to length of files
                if self.cond_dir_zarr is None:
                    n_samples = len(self.files)
                # Else n_samples equal to min val of length of files and length of conditional files
                else:
                    n_samples = min(len(self.files), len(self.files_cond))
                # Sample n_samples randomly
                print(f'\nNumber of samples in zarr dir: {len(self.files)}')
                print(f'Number of samples to generate: {self.n_samples_w_cutouts}')
                print(f'Number of samples to load: {n_samples}')
                random_idx = random.sample(range(n_samples), self.n_samples_w_cutouts)
                self.files = [self.files[i] for i in random_idx]
                if self.conditional_images:
                    # If no conditional images are used, use mean of samples as conditional image
                    if self.cond_dir_zarr is None:
                        self.files_cond = [self.files[i] for i in random_idx]
                    # If using individual samples as conditional images
                    else:
                        self.files_cond = [self.files_cond[i] for i in random_idx]
            # If shuffle is False, sample n_samples sequentially
            else:
                n_individual_samples = len(self.files)
                factor = int(np.ceil(self.n_samples_w_cutouts/n_individual_samples))
                self.files = self.files*factor
                self.files = self.files[0:self.n_samples_w_cutouts]
                if self.conditional_images:
                    # If no conditional images are used, use mean of samples as conditional image
                    if self.cond_dir_zarr is None:
                        self.files_cond = self.files*factor
                        self.files_cond = self.files_cond[0:self.n_samples_w_cutouts]
                    # If using individual samples as conditional images
                    else:
                        self.files_cond = self.files_cond*factor
                        self.files_cond = self.files_cond[0:self.n_samples_w_cutouts]
        
        # Set cache for data loading - if cache_size is 0, no caching is used
        self.cache = multiprocessing.Manager().dict()
        # #self.cache = SharedMemoryManager().dict()
        
        # Set transforms
        if self.scale:
            self.transforms_topo = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.data_size, antialias=True)
                ])
            if self.variable == 'temp':
                self.transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(self.data_size, antialias=True),
                    # Use if z-score transform (transformed to 10 year ERA5 (mean=8.714, std=6.010) training data):
                    ZScoreTransform(self.scale_mean, self.scale_std)
                    # Use if scaling in interval (not z-score transform):
                    ])
            elif self.variable == 'prcp':
                self.transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(self.data_size, antialias=True),
                    # Use log transform for precipitation data:
                    PrcpLogTransform(eps=1e-10,
                                     scale_type=self.scale_type_prcp,
                                     glob_mean_log=self.scale_mean,
                                     glob_std_log=self.scale,
                                     glob_min_log=self.scale_min_log,
                                     glob_max_log=self.scale_max_log,
                                     buffer_frac=self.buffer_frac)
                    # # Use Scale transform to scale data to new interval (based on ERA5 training data, (min = 0, max = 0.160m)):
                    # Scale(0, 0.160, 0, 1)
                    ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.data_size, antialias=True)
                ])
            self.transforms_topo = self.transforms
        
    def __len__(self):
        '''
            Return the length of the dataset.
        '''
        return len(self.files)

    def _addToCache(self, idx:int, data:torch.Tensor):
        '''
            Add item to cache. 
            If cache is full, remove random item from cache.
            Input:
                - idx: index of item to add to cache
                - data: data to add to cache
        '''
        # If cache_size is 0, no caching is used
        if self.cache_size > 0:
            # If cache is full, remove random item from cache
            if len(self.cache) >= self.cache_size:
                # Get keys from cache
                keys = list(self.cache.keys())
                # Select random key to remove
                key_to_remove = random.choice(keys)
                # Remove key from cache
                self.cache.pop(key_to_remove)
            # Add data to cache
            self.cache[idx] = data
    
    def __getitem__(self, idx:int):
        '''
            Get item from dataset based on index.
            Modified to load data from zarr files.
            Input:
                - idx: index of item to get
        '''

        # Get file name
        file_name = self.files[idx]

        # If conditional directory exists (i.e. LR conditions are used) get file name from conditional directory
        if self.cond_dir_zarr is None:
            # Set file names to file names of truth data
            file_name_cond = self.files[idx]
        else:
            file_name_cond = self.files_cond[idx]


        # Check if conditional sampling on season is used
        if self.conditional_seasons:

            # Determine class from filename
            if self.n_classes is not None:
                # Seasonal condittion
                if self.n_classes == 4:
                    dateObj = DateFromFile(file_name)
                    classifier = dateObj.determine_season()
                    
                # Monthly condition
                elif self.n_classes == 12:
                    dateObj = DateFromFile(file_name)
                    classifier = dateObj.determine_month()
                
                # Daily condition
                elif self.n_classes == 366:
                    dateObj = DateFromFile(file_name)
                    classifier = dateObj.determine_day()

                else:
                    raise ValueError('n_classes must be 4, 12 or 365')
            
            # Convert classifier to tensor
            classifier = torch.tensor(classifier)
        

        elif not self.conditional_seasons:
            # Set classifier to None
            classifier = None
        else:
            raise ValueError('conditional_seasons must be True or False')

        # Load data from zarr files, either temp or prcp
        if self.variable == 'temp':
            try:
                img = self.zarr_group_img[file_name]['t'][()][0,0,:,:] - 273.15
            except:
                img = self.zarr_group_img[file_name]['data'][()][:,:] - 273.15

            if self.conditional_images:
                if self.cond_dir_zarr is None:
                    # Compute the mean of sample 
                    mu = np.mean(img)
                    # Create conditional image with mean value
                    img_cond = np.ones(img.shape)*mu
                else:
                    img_cond = self.zarr_group_cond[file_name_cond]['arr_0'][()][:,:] - 273.15

        elif self.variable == 'prcp':
            try:
                # DANRA in [mm]
                img = self.zarr_group_img[file_name]['tp'][()][0,0,:,:] 
            except:
                # DANRA in [mm]
                img = self.zarr_group_img[file_name]['data'][()][:,:]
            # If sample has negative values, set them to eps=1e-8
            # Print negative values found
            if (img < 0).any():
                # print(f'Negative values found in sample {file_name}')
                # print(f'Number of negative values: {np.sum(img < 0)}')
                # print(f'Number of values in sample: {img.size}')
                # print(f'Minimum value in sample: {np.min(img)}')
                # print(f'Setting negative values to eps=1e-10')
                # Set negative values to eps=1e-8
                img[img <= 0] = 1e-10

            if self.conditional_images:
                if self.cond_dir_zarr is None:
                    # Compute the mean of sample
                    mu = np.mean(img)
                    # Create conditional image with mean value
                    img_cond = np.ones(img.shape)*mu
                else:
                    # ERA5 in [m] - convert to [mm] by multiplying by 1000
                    img_cond = self.zarr_group_cond[file_name_cond]['arr_0'][()][:,:] * 1000
                img_cond[img_cond <= 0] == 1e-10
            


        if self.cutouts:
            # Get random point
            point = find_rand_points(self.cutout_domains, 128)
            # Crop image, lsm and topo
            img = img[point[0]:point[1], point[2]:point[3]]

            if self.lsm_full_domain is not None:
                lsm_use = self.lsm_full_domain[point[0]:point[1], point[2]:point[3]]
            if self.topo_full_domain is not None:
                topo_use = self.topo_full_domain[point[0]:point[1], point[2]:point[3]]
    
            if self.sdf_weighted_loss:
                sdf_use = generate_sdf(lsm_use)
                sdf_use = normalize_sdf(sdf_use)

            if self.conditional_images:
                img_cond = img_cond[point[0]:point[1], point[2]:point[3]]
        else:
            point = None

        # If save_original is True, save original image and conditional image
        if self.save_original:
            img_original = img.copy()
            if self.conditional_images:
                img_cond_original = img_cond.copy()

        # Apply transforms if any
        if self.transforms:
            img = self.transforms(img)

            if self.cutouts:
                if self.lsm_full_domain is not None:
                    lsm_use = self.transforms_topo(lsm_use.copy())
                if self.topo_full_domain is not None:
                    topo_use = self.transforms_topo(topo_use.copy())
                if self.sdf_weighted_loss:
                    sdf_use = self.transforms_topo(sdf_use.copy())

            if self.conditional_images:
                img_cond = self.transforms(img_cond)

        if self.conditional_images:
            # Return sample image and classifier
            if self.conditional_seasons:
                # Make a dict with image and conditions (and originals, if save_original is True)
                if self.save_original:
                    sample_dict = {'img':img, 'img_original':img_original, 'classifier':classifier, 'img_cond':img_cond, 'img_cond_original':img_cond_original}
                else:
                    sample_dict = {'img':img, 'classifier':classifier, 'img_cond':img_cond}
                #sample = (img, classifier, img_cond)
            else:
                # Make a dict with image and conditions
                if self.save_original:
                    sample_dict = {'img':img, 'img_original':img_original, 'img_cond':img_cond, 'img_cond_original':img_cond_original}
                else:
                    sample_dict = {'img':img, 'img_cond':img_cond}
                #sample = (img, img_cond)
        else:
            # Return sample image as dict (with original image if save_original is True)
            if self.save_original:
                sample_dict = {'img':img, 'img_original':img_original}
            else:
                sample_dict = {'img':img}
            sample = (img)
        
        # Add item to cache
        self._addToCache(idx, sample_dict)

        
        # Return data based on whether cutouts are used or not
        if self.cutouts:
            # If sdf weighted loss is used, add sdf to return
            if self.sdf_weighted_loss:
                # Make sure lsm and topo also provided, otherwise raise error
                if self.lsm_full_domain is not None and self.topo_full_domain is not None:
                    # Add lsm, sdf, topo and point to dict
                    sample_dict['lsm'] = lsm_use
                    sample_dict['sdf'] = sdf_use
                    sample_dict['topo'] = topo_use
                    sample_dict['points'] = point
                    # Return sample image and classifier and random point for cropping (lsm and topo)
                    return sample_dict #sample, lsm_use, topo_use, sdf_use, point
                else:
                    raise ValueError('lsm_full_domain and topo_full_domain must be provided if sdf_weighted_loss is True')
            # If sdf weighted loss is not used, only return lsm and topo if they are provided
            else:
                # Return lsm and topo if provided
                if self.lsm_full_domain is not None and self.topo_full_domain is not None:
                    # Add lsm, topo and point to dict
                    sample_dict['lsm'] = lsm_use
                    sample_dict['topo'] = topo_use
                    sample_dict['points'] = point
                    # Return sample image and classifier and random point for cropping (lsm and topo)
                    return sample_dict #sample, lsm_use, topo_use, point
                # If lsm and topo not provided, only return sample and point
                else:
                    # Add point to dict
                    sample_dict['points'] = point
                    return sample_dict #sample, point
        else:
            # Return sample image and classifier only
            return sample_dict #sample
        
    
    def __name__(self, idx:int):
        '''
            Return the name of the file based on index.
            Input:
                - idx: index of item to get
        '''
        return self.files[idx]



    