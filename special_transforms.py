'''
    This file contains custom transforms for data preprocessing.
'''

# Import libraries and modules 
import torch

# Define custom transforms
class Scale(object):
    '''
        Class for scaling the data to a new interval. 
        The data is scaled to the interval [in_low, in_high].
        The data is assumed to be in the interval [data_min_in, data_max_in].
    '''
    def __init__(self,
                 in_low,
                 in_high,
                 data_min_in = 0,
                 data_max_in = 1
                 ):
        '''
            Initialize the class.
            Input:
                - in_low: lower bound of new interval
                - in_high: upper bound of new interval
                - data_min_in: lower bound of data interval
                - data_max_in: upper bound of data interval
        '''
        self.in_low = in_low
        self.in_high = in_high
        self.data_min_in = data_min_in 
        self.data_max_in = data_max_in

    def __call__(self, sample):
        '''
            Call function for the class - scales the data to the new interval.
            Input:
                - sample: datasample to scale to new interval
        '''
        data = sample
        OldRange = (self.data_max_in - self.data_min_in)
        NewRange = (self.in_high - self.in_low)

        # Generating the new data based on the given intervals
        DataNew = (((data - self.data_min_in) * NewRange) / OldRange) + self.in_low

        return DataNew

# Back transform the scaled data
class ScaleBackTransform(object):
    '''
    Class for back-transforming the scaled data.
    The data is back-transformed to the original interval.
    '''
    def __init__(self,
                 data_min_in = 0,
                 data_max_in = 1
                 ):
        '''
        Initialize the class.
        Input:
            - data_min_in: lower bound of data interval
            - data_max_in: upper bound of data interval
        '''
        self.data_min_in = data_min_in
        self.data_max_in = data_max_in

    def __call__(self, sample):
        '''
        Call function for the class - back-transforms the scaled data.
        Input:
            - sample: data sample to be back-transformed
        '''
        data = sample
        OldRange = (1 - 0)
        NewRange = (self.data_max_in - self.data_min_in)

        # Back-transforming the data
        DataNew = (((data - 0) * NewRange) / OldRange) + self.data_min_in

        return DataNew


import torch

class ZScoreTransform(object):
    '''
    Class for Z-score standardizing the data. 
    The data is standardized to have a mean of 0 and a standard deviation of 1.
    The mean and standard deviation of the training data should be provided.
    '''
    def __init__(self, mean, std):
        '''
        Initialize the class.
        Input:
            - mean: the mean of the global training data
            - std: the standard deviation of the global training data
        '''
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        '''
        Call function for the class - standardizes the data.
        Input:
            - sample: data sample to be standardized
        '''
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)  # Ensure the input is a Tensor
        
        # Ensure mean and std are tensors for broadcasting, preserve their shapes if they are not scalars.
        if not isinstance(self.mean, torch.Tensor):
            self.mean = torch.tensor(self.mean, dtype=torch.float32)
        if not isinstance(self.std, torch.Tensor):
            self.std = torch.tensor(self.std, dtype=torch.float32)

        # Expand as necessary to match the sample dimensions
        if len(sample.shape) > len(self.mean.shape):
            shape_diff = len(sample.shape) - len(self.mean.shape)
            for _ in range(shape_diff):
                self.mean = self.mean.unsqueeze(0)
                self.std = self.std.unsqueeze(0)

        # Standardizing the sample
        standardized_sample = (sample - self.mean) / (self.std + 1e-8)  # Add a small epsilon to avoid division by zero

        return standardized_sample
    
# Back transform the standardized data
class ZScoreBackTransform(object):
    '''
    Class for back-transforming the Z-score standardized data.
    The data is back-transformed to the original distribution with mean and standard deviation.
    '''
    def __init__(self, mean, std):
        '''
        Initialize the class.
        Input:
            - mean: the mean of the training data
            - std: the standard deviation of the training data
        '''
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        '''
        Call function for the class - back-transforms the standardized data.
        Input:
            - sample: data sample to be back-transformed
        '''
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)  # Ensure the input is a Tensor

        # Ensure mean and std are tensors for broadcasting, preserve their shapes if they are not scalars.
        if not isinstance(self.mean, torch.Tensor):
            self.mean = torch.tensor(self.mean, dtype=torch.float32)
        if not isinstance(self.std, torch.Tensor):
            self.std = torch.tensor(self.std, dtype=torch.float32)

        # Expand as necessary to match the sample dimensions
        if len(sample.shape) > len(self.mean.shape):
            shape_diff = len(sample.shape) - len(self.mean.shape)
            for _ in range(shape_diff):
                self.mean = self.mean.unsqueeze(0)
                self.std = self.std.unsqueeze(0)

        # Back-transforming the sample
        back_transformed_sample = (sample * (self.std + 1e-8)) + self.mean  # Add a small epsilon to avoid division by zero

        return back_transformed_sample
    




class PrcpLogTransform(object):
    '''
    Class for log-transforming the precipitation data.
    The data is transformed by taking the logarithm of the data.
    '''
    def __init__(self, eps=1e-8, scale01 = True,
                 glob_max=None, buffer=0.5): 
        '''
        Initialize the class.
        '''
        self.eps = eps
        self.scale01 = scale01
        self.glob_max = glob_max
        self.buffer = buffer

        pass

    def __call__(self, sample):
        '''
        Call function for the class - log-transforms the data.
        Input:
            - sample: data sample to be log-transformed
            - eps: small epsilon to avoid log(0)
        '''
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)  # Ensure the input is a Tensor

        # Ensure the sample is non-negative
        if (sample < 0).any():
            raise ValueError('The sample contains negative values and cannot be log-transformed.')
        

        # Log-transform the sample
        log_sample = torch.log(sample + self.eps) # Add a small epsilon to avoid log(0)

        if self.scale01:
            # Find the global maximum value in the training data or use global statistics max
            if glob_max is None:
                glob_max = torch.max(log_sample)
            # Scale the log-transformed data to [0, 1], with a % buffer
            log_sample = log_sample / (glob_max + self.buffer * glob_max)

        return log_sample
    
# Back transform the log-transformed data
class PrcpLogBackTransform(object):
    '''
    Class for back-transforming the log-transformed precipitation data.
    The data is back-transformed by taking the exponential of the data.
    '''
    def __init__(self, scale01 = True,
                 glob_max=None, buffer=0.5): 
        '''
        Initialize the class.
        '''
        self.scale01 = scale01
        self.glob_max = glob_max
        self.buffer = buffer

        pass

    def __call__(self, sample):
        '''
        Call function for the class - back-transforms the log-transformed data.
        Input:
            - sample: data sample to be back-transformed
        '''
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)  # Ensure the input is a Tensor

        if self.scale01:
            # Find the global maximum value in the training data or use global statistics max
            if glob_max is None:
                glob_max = torch.max(sample)
            # Scale the log-transformed data to [0, 1], with a % buffer
            sample = sample * (glob_max + self.buffer * glob_max)

        # Back-transform the log-transformed sample
        back_transformed_sample = torch.exp(sample)

        return back_transformed_sample



