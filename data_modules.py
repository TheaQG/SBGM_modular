"""
    Script for generating a pytorch dataset for the DANRA data.
    The dataset is loaded as a single-channel image - either prcp or temp.
    Different transforms are be applied to the dataset.
    A custom transform is used to scale the data to a new interval.
"""

# Import libraries and modules 
from special_transforms import *
import zarr
import re
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
    sdf = 10*binary_mask.float() - dist_transform_sea

    return sdf

def normalize_sdf(sdf):
    # Find min and max in the SDF
    if isinstance(sdf, torch.Tensor):
        min_val = torch.min(sdf)
        max_val = torch.max(sdf)
    elif isinstance(sdf, np.ndarray):
        min_val = np.min(sdf)
        max_val = np.max(sdf)
    else:
        raise ValueError('SDF must be either torch.Tensor or np.ndarray')

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
    
def FileDate(filename):
    """
    Extract the last 8 digits from the filename as the date string.
    E.g. for 't2m_ave_19910122' or 'temp_589x789_19910122', returns '19910122'
    """

    m = re.search(r'(\d{8})$', filename)
    if m:
        return m.group(1)
    else:
        raise ValueError(f"Could not extract date from filename: {filename}")


def find_rand_points(rect, crop_size):
    '''
    Randomly selects a crop region within a given rectangle
    Input:
        - rect (list or tuple): [x1, x2, y1, y2] rectangle to crop from
        - crop_size (tuple): (crop_width, crop_height) size of the desired crop
    Output:
        - point (list): [x1_new, x2_new, y1_new, y2_new] random crop region

    Raises: 
        - ValueError if crop_size is larger than the rectangle
    '''
    x1 = rect[0]
    x2 = rect[1]
    y1 = rect[2]
    y2 = rect[3]

    crop_width = crop_size[0]
    crop_height = crop_size[1]

    full_width = x2 - x1
    full_height = y2 - y1

    if crop_width > full_width or crop_height > full_height:
        raise ValueError('Crop size is larger than the rectangle dimensions.')

    # Calculate available offsets
    max_x_offset = full_width - crop_width
    max_y_offset = full_height - crop_height

    offset_x = random.randint(0, max_x_offset)
    offset_y = random.randint(0, max_y_offset)

    x1_new = x1 + offset_x
    x2_new = x1_new + crop_width
    y1_new = y1 + offset_y
    y2_new = y1_new + crop_height

    point = [x1_new, x2_new, y1_new, y2_new]
    return point


def random_crop(data, target_size):
    """
        Randomly crops a 2D 'data' to shape (target_size[0], target_size[1]).
        Assumes data is a 2D numpy array
        Input:
            - data: 2D numpy array
            - target_size: tuple containing the target size of the data
        Output:
            - data: 2D numpy array with shape (target_size[0], target_size[1])
        Raises:
            - ValueError if target size is larger than the data dimensions    
    """
    H, W = data.shape

    if target_size[0] > H or target_size[1] > W:
        raise ValueError('Target size is larger than the data dimensions.')
    
    if H == target_size[0] and W == target_size[1]:
        return data

    y = random.randint(0, H - target_size[0])
    x = random.randint(0, W - target_size[1])
    return data[y:y + target_size[0], x:x + target_size[1]]

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



def list_all_keys(zgroup):
    all_keys = []
    for key in zgroup.keys():
        all_keys.append(key)
        member = zgroup[key]
        if isinstance(member, zarr.hierarchy.Group):
            sub_keys = list_all_keys(member)
            all_keys.extend([f"{key}/{sub}" for sub in sub_keys])
    return all_keys

# all_keys = list_all_keys(self.lr_cond_zarr_dict[cond])
# print(all_keys)


class DANRA_Dataset_cutouts_ERA5_Zarr_test(Dataset):
    '''
        Class for setting the DANRA dataset with option for random cutouts from specified domains.
        Along with DANRA data, the land-sea mask and topography data is also loaded at same cutout.
        Possibility to sample more than n_samples if cutouts are used.
        Option to shuffle data or load sequentially.
        Option to scale data to new interval.
        Option to use conditional (classifier) sampling (season, month or day).
    '''
    def __init__(self, 
                # Must-have parameters
                hr_variable_dir_zarr:str,           # Path to high resolution data
                data_size:tuple,                    # Size of data (2D image, tuple)
                # HR target variable and its scaling parameters
                hr_variable:str = 'temp',           # Variable to load (temp or prcp)
                hr_scaling_method:str = 'zscore',   # Scaling method for high resolution data
                hr_scaling_params:dict = {'glob_mean':8.69251, 'glob_std':6.192434}, # Scaling parameters for high resolution data (if prcp, 'log_minus1_1' or 'log_01' include 'glob_min_log' and 'glob_max_log' and optional buffer_frac)
                # LR conditions and their scaling parameters (not including geo variables. they are handled separately)
                lr_conditions:list = ['temp'], # Variables to load as low resolution conditions
                lr_scaling_methods:list = ['zscore'], # Scaling methods for low resolution conditions
                lr_scaling_params:list = [{'glob_mean':8.69251, 'glob_std':6.192434}], # Scaling parameters for low resolution conditions
                lr_cond_dirs_zarr:dict = None,           # Path to directories containing conditional data (in format dict({'condition1':dir1, 'condition2':dir2}))
                # Geo variables (stationary) and their full domain arrays
                geo_variables:list = ['lsm', 'topo'], # Geo variables to load
                lsm_full_domain = None,             # Land-sea mask of full domain
                topo_full_domain = None,            # Topography of full domain
                # Other dataset parameters
                n_samples:int = 365,                # Number of samples to load
                cache_size:int = 365,               # Number of samples to cache
                shuffle:bool = False,               # Whether to shuffle data (or load sequentially)
                cutouts:bool = False,               # Whether to use cutouts 
                cutout_domains:list = None,         # Domains to use for cutouts
                n_samples_w_cutouts:int = None,     # Number of samples to load with cutouts (can be greater than n_samples)
                sdf_weighted_loss:bool = False,     # Whether to use weighted loss for SDF
                scale:bool = True,                  # Whether to scale data to new interval
                save_original:bool = False,         # Whether to save original data
                conditional_seasons:bool = False,   # Whether to use seasonal conditional sampling
                n_classes:int = None,               # Number of classes for conditional sampling
                # NEW: LR conditioning area size (if cropping is desired)
                lr_data_size: tuple = None,         # Size of low resolution data (2D image, tuple), e.g. (589,789) for full LR domain
                # Optionally a separate cutout domain for LR conditions
                lr_cutout_domains: list = None    # Domains to use for cutouts for LR conditions
                ):                       
        '''
        Initializes the dataset.
        '''
        
        # Basic dataset parameters
        self.hr_variable_dir_zarr = hr_variable_dir_zarr
        self.n_samples = n_samples
        self.data_size = data_size
        self.cache_size = cache_size

        # LR conditions and scaling parameters
        # (Remove any geo variable from conditions list, if accidentally included)
        self.geo_variables = geo_variables
        # Check that there are the same number of scaling methods and parameters as conditions
        if len(lr_conditions) != len(lr_scaling_methods) or len(lr_conditions) != len(lr_scaling_params):
            raise ValueError('Number of conditions, scaling methods and scaling parameters must be the same')

        # Go through the conditions, and if condition is in geo_variables, remoce from list, and remove scaling methods and params associated with it
        for geo_var in self.geo_variables:
            if geo_var in lr_conditions:
                idx = lr_conditions.index(geo_var)
                lr_conditions.pop(idx)
                lr_scaling_methods.pop(idx)
                lr_scaling_params.pop(idx)
        self.lr_conditions = lr_conditions
        self.lr_scaling_methods = lr_scaling_methods
        self.lr_scaling_params = lr_scaling_params
        # If any conditions exist, set with_conditions to True
        self.with_conditions = len(self.lr_conditions) > 0

        # Save new LR parameters
        self.lr_data_size = lr_data_size
        self.lr_cutout_domains = lr_cutout_domains
        # Specify target LR size (if different from HR size)
        self.target_lr_size = self.lr_data_size if self.lr_data_size is not None else self.data_size


        # Save LR condition directories
        # lr_cond_dirs_zarr is a dict mapping each condition to its own zarr directory path
        self.lr_cond_dirs_zarr = lr_cond_dirs_zarr
        # Open each LR condition's zarr group and list its files
        self.lr_cond_zarr_dict = {} 
        self.lr_cond_files_dict = {}
        if self.lr_cond_dirs_zarr is not None:
            for cond in self.lr_cond_dirs_zarr:
                print(f'Loading zarr group for condition {cond}')
                # print(f'Path to zarr group: {self.lr_cond_dirs_zarr[cond]}')
                group = zarr.open_group(self.lr_cond_dirs_zarr[cond], mode='r')
                self.lr_cond_zarr_dict[cond] = group
                self.lr_cond_files_dict[cond] = list(group.keys())
            print('\n\n')
        else:
            raise ValueError(f'LR condition {cond} not found in dict')

        # HR target variable parameters
        self.hr_variable = hr_variable
        self.hr_scaling_method = hr_scaling_method
        self.hr_scaling_params = hr_scaling_params
        
        # Save geo variables full-domain arrays
        self.lsm_full_domain = lsm_full_domain
        self.topo_full_domain = topo_full_domain

        # Save other parameters
        self.shuffle = shuffle
        self.cutouts = cutouts
        self.cutout_domains = cutout_domains
        self.sdf_weighted_loss = sdf_weighted_loss
        self.scale = scale
        self.save_original = save_original
        self.conditional_seasons = conditional_seasons
        self.n_classes = n_classes
        self.n_samples_w_cutouts = self.n_samples if n_samples_w_cutouts is None else n_samples_w_cutouts
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
        #                               #
        # PRINT INFORMATION ABOUT SCALING
        #                               #
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! #


        # Build file maps based on the date in the file name      
        # Open main (HR) zarr group, and get HR file keys (pure filenames)
        self.zarr_group_img = zarr.open_group(hr_variable_dir_zarr, mode='r')
        hr_files_all = list(self.zarr_group_img.keys())
        self.hr_file_map = {}
        for file in hr_files_all:
            try:
                date = FileDate(file)
                self.hr_file_map[date] = file
            except Exception as e:
                print(f"Warning: Could not extract date from file {file}. Skipping file. Error: {e}")

        # For each LR condition, build a file map: date -> file key
        self.lr_file_map = {}
        for cond in self.lr_conditions:
            self.lr_file_map[cond] = {}
            for file in self.lr_cond_files_dict[cond]:
                try:
                    date = FileDate(file)
                    self.lr_file_map[cond][date] = file
                except Exception as e:
                    print(f"Warning: Could not extract date from file {file} for condition {cond}. Skipping file. Error: {e}")

        # Compute common dates across HR and all LR conditions
        common_dates = set(self.hr_file_map.keys())
        for cond in self.lr_conditions:
            common_dates = common_dates.intersection(set(self.lr_file_map[cond].keys()))
        self.common_dates = sorted(list(common_dates))
        if len(self.common_dates) < self.n_samples:
            self.n_samples = len(self.common_dates)
            print(f"Warning: Number of common dates is less than n_samples. Setting n_samples to {self.n_samples}")
        if self.shuffle:
            self.common_dates = random.sample(self.common_dates, self.n_samples)

        # Set cache for data loading - if cache_size is 0, no caching is used
        self.cache = multiprocessing.Manager().dict()

        # Set transforms for conditions, and use specified scaling methods and parameters
        if self.scale:
            self.transforms_dict = {}
            for cond, method, params in zip(self.lr_conditions, self.lr_scaling_methods, self.lr_scaling_params):
                # Base transform: to tensor and resize
                transform_list = [
                    transforms.ToTensor(),
                    transforms.Resize(self.target_lr_size, antialias=True)
                ]
                # Use per-variable buffer_frac
                buff = params.get('buffer_frac', 0.5)
                if method == 'zscore':
                    # ADD BUFFER FRACTION TO ZSCORE TRANSFORM
                    transform_list.append(ZScoreTransform(params['glob_mean'], params['glob_std']))
                elif method in ['log', 'log_01', 'log_minus1_1', 'log_zscore']:
                    transform_list.append(PrcpLogTransform(eps=1e-10,
                                                           scale_type=method,
                                                           glob_mean_log=params['glob_mean_log'],
                                                           glob_std_log=params['glob_std_log'],
                                                           glob_min_log=params['glob_min_log'],
                                                           glob_max_log=params['glob_max_log'],
                                                           buffer_frac=buff))
                elif method == '01':
                    transform_list.append(Scale(0, 1, params['glob_min'], params['glob_max']))
                self.transforms_dict[cond] = transforms.Compose(transform_list) 
        else:
            self.transforms_dict = {cond: transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.target_lr_size, antialias=True)
            ]) for cond in self.lr_conditions}

        # Build transform for the HR target variable similarly
        if self.scale:
            hr_transform_list = [
                transforms.ToTensor(),
                transforms.Resize(self.data_size, antialias=True)
            ]
            hr_buff = self.hr_scaling_params.get('buffer_frac', 0.5)
            if self.hr_scaling_method == 'zscore':
                hr_transform_list.append(ZScoreTransform(self.hr_scaling_params['glob_mean'], self.hr_scaling_params['glob_std']))
            elif self.hr_scaling_method in ['log', 'log_01', 'log_minus1_1', 'log_zscore']:
                hr_transform_list.append(PrcpLogTransform(eps=1e-10,
                                                          scale_type=self.hr_scaling_method,
                                                          glob_mean_log=self.hr_scaling_params['glob_mean_log'],
                                                          glob_std_log=self.hr_scaling_params['glob_std_log'],
                                                          glob_min_log=self.hr_scaling_params['glob_min_log'],
                                                          glob_max_log=self.hr_scaling_params['glob_max_log'],
                                                          buffer_frac=hr_buff))
            elif self.hr_scaling_method == '01':
                hr_transform_list.append(Scale(0, 1, self.hr_scaling_params['glob_min'], self.hr_scaling_params['glob_max']))
            self.hr_transform = transforms.Compose(hr_transform_list)
        else:
            self.hr_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.data_size, antialias=True)
            ])

        # Build a transform for the geo variables, with possible scaling  to [0,1]
        if self.scale:
            self.geo_transform_topo = transforms.Compose([
                transforms.Lambda(lambda x: np.ascontiguousarray(x)), # To make sure np.flipud is not messing up the tensor
                transforms.ToTensor(),
                transforms.Resize(self.target_lr_size, antialias=True),
                Scale(0, 1, self.topo_full_domain.min(), self.topo_full_domain.max())
            ])
            self.geo_transform_lsm = transforms.Compose([
                transforms.Lambda(lambda x: np.ascontiguousarray(x)), # To make sure np.flipud is not messing up the tensor
                transforms.ToTensor(),
                transforms.Resize(self.target_lr_size, antialias=True),
            ])
        else:
            self.geo_transform_topo = transforms.Compose([
                transforms.Lambda(lambda x: np.ascontiguousarray(x)), # To make sure np.flipud is not messing up the tensor
                transforms.ToTensor(),
                transforms.Resize(self.target_lr_size, antialias=True)
            ])
            self.geo_transform_lsm = self.geo_transform_topo
    
    def print_transformation_info(self):
        print("=== Data Transformation Summary ===")
        print("Low-Resolution (LR) Conditions:")
        for cond, method, params in zip(self.conditions, self.scaling_methods, self.scaling_params):
            print(f"\nCondition: '{cond}'")
            print(f"  Scaling Method: {method}")
            if cond.lower() == 'temp':
                print("  Note: Data is in Kelvin; conversion to Celsius is applied (subtract 273.15).")
            if method == 'zscore':
                print(f"  Using ZScore Transform:")
                print(f"    Original distribution: mean = {params['glob_mean']}, std = {params['glob_std']}")
                print("    Transformed distribution: mean = 0, std = 1")
            elif method in ['log_01', 'log_minus1_1']:
                if 'glob_log_min' in params and 'glob_log_max' in params:
                    log_range = params['glob_log_max'] - params['glob_log_min']
                    buff = params.get('buffer_frac', 0.5)
                    scale_buffer_min = params['glob_log_min'] - buff * log_range
                    scale_buffer_max = params['glob_log_max'] + buff * log_range
                    print(f"  Using {method} Transform:")
                    print(f"    Original log interval: [{params['glob_log_min']}, {params['glob_log_max']}]")
                    print(f"    Buffer fraction: {buff}")
                    print(f"    Buffered interval: [{scale_buffer_min}, {scale_buffer_max}]")
            elif method == '01':
                print(f"  Using ScaleTransform (Min-Max):")
                print(f"    Original interval: [{params['glob_min']}, {params['glob_max']}]")
                print("    Transformed interval: [0, 1]")
            else:
                print("  No specific transform information available.")
        
        print("\nHigh-Resolution (HR) Target Variable:")
        print(f"  Variable: '{self.hr_variable}'")
        print(f"  Scaling Method: {self.hr_scaling_method}")
        if self.hr_variable.lower() == 'temp':
            print("  Note: Data is in Kelvin; conversion to Celsius is applied (subtract 273.15).")
        hr_params = self.hr_scaling_params
        if self.hr_scaling_method == 'zscore':
            print("  Using ZScore Transform:")
            print(f"    Original distribution: mean = {hr_params['glob_mean']}, std = {hr_params['glob_std']}")
            print("    Transformed distribution: mean = 0, std = 1")
        elif self.hr_scaling_method in ['log_01', 'log_minus1_1']:
            if 'glob_log_min' in hr_params and 'glob_log_max' in hr_params:
                log_range = hr_params['glob_log_max'] - hr_params['glob_log_min']
                buff = hr_params.get('buffer_frac', 0.5)
                scale_buffer_min = hr_params['glob_log_min'] - buff * log_range
                scale_buffer_max = hr_params['glob_log_max'] + buff * log_range
                print(f"  Using {self.hr_scaling_method} Transform:")
                print(f"    Original log interval: [{hr_params['glob_log_min']}, {hr_params['glob_log_max']}]")
                print(f"    Buffer fraction: {buff}")
                print(f"    Buffered interval: [{scale_buffer_min}, {scale_buffer_max}]")
        elif self.hr_scaling_method == '01':
            print("  Using ScaleTransform (Min-Max):")
            print(f"    Original interval: [{hr_params['glob_min']}, {hr_params['glob_max']}]")
            print("    Transformed interval: [0, 1]")
        else:
            print("  No specific transform information available.")

    def __len__(self):
        '''
            Return the length of the dataset.
        '''
        return len(self.common_dates)

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
            For each sample:
            - Loads LR conditions from the main zarr group (and, if applicable, from additional condition directories)
            - Loads HR target variable from the main zarr group
            - Loads the stationary geo variables (lsm and topo) from provided full-domain arrays
            - Applies cutouts and the appropriate transforms
        '''

        # Get the common date corresponding to the index
        date = self.common_dates[idx]
        sample_dict = {}

        # Determine crop region, if cutouts are used
        if self.cutouts:
            # hr_point is computed using HR cutout domain and HR data size
            hr_point = find_rand_points(self.cutout_domains, self.data_size)
            # For LR conditions, if lr_data_size and separate LR cutout domains are provided, use them
            if self.lr_data_size is not None and self.lr_cutout_domains is not None:
                lr_point = find_rand_points(self.lr_cutout_domains, self.lr_data_size)
            else:
                lr_point = hr_point # We will use random crop later
        else:
            hr_point = None
            lr_point = None

        # Look up HR file using the common date
        hr_file_name = self.hr_file_map[date]

        # Look up LR files for each condition using the common date
        for cond in self.lr_conditions:
            lr_file_name = self.lr_file_map[cond][date]
            # Load LR condition data from its own zarr group
            try:
                # print(f'Loading LR {cond} data for {lr_file_name}')
                # print(self.lr_cond_zarr_dict[cond].tree())
                if cond == "temp":
                    try:
                        data = self.lr_cond_zarr_dict[cond][lr_file_name]['t']
                        data = data[()][0,0,:,:] - 273.15
                        # print("Key 't' found")
                    except:
                        data = self.lr_cond_zarr_dict[cond][lr_file_name]['arr_0']
                        data = data[()][:,:] - 273.15
                        # print("Key 'data' found")
                elif cond == "prcp":
                    try:
                        data = self.lr_cond_zarr_dict[cond][lr_file_name]['tp']
                        data = data[()][0,0,:,:] * 1000
                        data[data <= 0] = 1e-10
                        # print("Key 'tp' found")
                    except:
                        data = self.lr_cond_zarr_dict[cond][lr_file_name]['arr_0']
                        data = data[()][:,:] * 1000
                        data[data <= 0] = 1e-10
                        # print("Key 'arr_0' found")
                else:
                    # Add custom logic for other LR conditions when needed
                    data = self.lr_cond_zarr_dict[cond][lr_file_name]['data'][()]
            except Exception as e:
                print(f'Error loading {cond} data for {lr_file_name}')
                print(e)
                data = None
            
            # Crop LR data using lr_point if cutouts are enabled
            if self.cutouts and data is not None:
                # lr_point is in format [x1, x2, y1, y2] - note: for slicing, use [y1:y2, x1:x2]
                data = data[lr_point[0]:lr_point[1], lr_point[2]:lr_point[3]]
            print(f"Data shape for {cond}: {data.shape if data is not None else None}")
                
            # If save_original is True, save original conditional data
            if self.save_original:
                sample_dict[f"{cond}_lr_original"] = data.copy() if data is not None else None

            # Apply specified transform (specific to various conditions)
            if data is not None and self.transforms_dict.get(cond, None) is not None:
                data = self.transforms_dict[cond](data)
            sample_dict[cond + "_lr"] = data
        print('\n')

        # Load HR target variable data
        try:
            # print(f'Loading HR {self.hr_variable} data for {hr_file_name}')
            # print(self.zarr_group_img[hr_file_name].tree())
            if self.hr_variable == 'temp':
                try:
                    hr = self.zarr_group_img[hr_file_name]['t'][()][0,0,:,:] - 273.15
                except:
                    hr = self.zarr_group_img[hr_file_name]['data'][()][:,:] - 273.15
            elif self.hr_variable == 'prcp':
                try:
                    hr = self.zarr_group_img[hr_file_name]['tp'][()][0,0,:,:]
                except:
                    hr = self.zarr_group_img[hr_file_name]['data'][()][:,:]
                hr[hr <= 0] = 1e-10
            else:
                # Add custom logic for other HR variables when needed
                hr = self.zarr_group_img[hr_file_name]['data'][()]
        except Exception as e:
            print(f'Error loading {self.hr_variable} data for {hr_file_name}')
            print(e)
            hr = None

        if self.cutouts and (hr is not None):
            hr = hr[hr_point[0]:hr_point[1], hr_point[2]:hr_point[3]]
        if self.save_original and (hr is not None):
            sample_dict[f"{self.hr_variable}_hr_original"] = hr.copy()
        if hr is not None:
            hr = self.hr_transform(hr)
        sample_dict[self.hr_variable + "_hr"] = hr

        # Process a separate HR mask for geo variables (if 'lsm' is needed for HR SDF and masking HR images)
        if 'lsm' in self.geo_variables and self.lsm_full_domain is not None:
            lsm_hr = self.lsm_full_domain
            if self.cutouts and lsm_hr is not None:
                lsm_hr = lsm_hr[hr_point[0]:hr_point[1], hr_point[2]:hr_point[3]]
            # Ensure the mask is contiguous and transform
            lsm_hr = np.ascontiguousarray(lsm_hr)
            # Separate geo transform, with resize to HR size
            geo_transform_lsm_hr = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.data_size, antialias=True)
            ])
            lsm_hr = geo_transform_lsm_hr(lsm_hr)
            sample_dict['lsm_hr'] = lsm_hr


        # Load geo variables (stationary) from full-domain arrays (may be cropped using lr_data_size and lr_cutout_domains)
        if self.geo_variables is not None:
            for geo in self.geo_variables:
                if geo == 'lsm':
                    if self.lsm_full_domain is None:
                        raise ValueError("lsm_full_domain must be provided if 'lsm' is in geo_variables")
                    geo_data = self.lsm_full_domain
                    print('lsm_full_domain shape:', geo_data.shape)
                    geo_transform = self.geo_transform_lsm
                elif geo == 'topo':
                    if self.topo_full_domain is None:
                        raise ValueError("topo_full_domain must be provided if 'topo' is in geo_variables")
                    geo_data = self.topo_full_domain
                    print('topo_full_domain shape:', geo_data.shape)
                    geo_transform = self.geo_transform_topo
                else:
                    # Add custom logic for other geo variables when needed
                    geo_data = None
                    geo_transform = None
                if geo_data is not None and self.cutouts:
                    # For geo data, if an LR-specific size and domain are provided, use lr_point
                    if self.lr_data_size is not None and self.lr_cutout_domains is not None:
                        geo_data = geo_data[lr_point[0]:lr_point[1], lr_point[2]:lr_point[3]]
                    else:
                        geo_data = geo_data[hr_point[0]:hr_point[1], hr_point[2]:hr_point[3]]
                if geo_data is not None:
                    geo_data = geo_transform(geo_data)
                sample_dict[geo] = geo_data

        # Check if conditional sampling on season (or monthly/daily) is used
        if self.conditional_seasons:
            # Determine class from filename
            if self.n_classes is not None:
                # Seasonal condittion
                if self.n_classes == 4:
                    dateObj = DateFromFile(hr_file_name)
                    classifier = dateObj.determine_season()
                # Monthly condition
                elif self.n_classes == 12:
                    dateObj = DateFromFile(hr_file_name)
                    classifier = dateObj.determine_month()
                # Daily condition
                elif self.n_classes == 366:
                    dateObj = DateFromFile(hr_file_name)
                    classifier = dateObj.determine_day()
                else:
                    raise ValueError('n_classes must be 4, 12 or 365')
            # Convert classifier to tensor
            classifier = torch.tensor(classifier)
            sample_dict['classifier'] = classifier
        else:
            sample_dict['classifier'] = None

        # For SDF, ensure that it is computed for the HR mask (lsm_hr) to get it in same shape as HR
        if self.sdf_weighted_loss:
            if 'lsm_hr' in sample_dict and sample_dict['lsm_hr'] is not None:
                sdf = generate_sdf(sample_dict['lsm_hr'])
                sdf = normalize_sdf(sdf)
                sample_dict['sdf'] = sdf
            else:
                raise ValueError("lsm_hr must be provided for SDF computation if sdf_weighted_loss is True")
            
        # Attach cutout points for reference
        if self.cutouts:
            sample_dict['hr_points'] = hr_point
            sample_dict['lr_points'] = lr_point

        # Add item to cache
        self._addToCache(idx, sample_dict)

        return sample_dict #sample

    def __name__(self, idx:int):
        '''
            Return the name of the file based on index.
            Input:
                - idx: index of item to get
        '''
        date = self.common_dates[idx]
        return date #self.hr_file_map[date]
