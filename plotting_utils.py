'''
    This file contains utility functions for plotting.

    Functions:
        - plot_dataset: Plots the images in the dataset. Can plot LSM, Topography, SDF, truth, condition and prediction.
        - plot_psds: Plots the power spectral densities of the given images. Can compare truth, condition and prediction.
        - plot_psd: Plots the power spectral density of a single image.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_dataset(data_dict, num_images=5, figsize=(15, 15), cmap='viridis', title=None, save_path=None):
    '''
    Plots the images in the dataset. Can plot LSM, Topography, SDF, truth, condition and prediction.

    Args:
        data_dict (dict): Dictionary containing the dataset.
        num_images (int): Number of images to plot.
        figsize (tuple): Size of the figure.
        cmap (str): Colormap to use.
        title (str): Title of the plot.
        save_path (str): Path to save the plot.
    '''
    fig, axs = plt.subplots(3, num_images, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    for i in range(num_images):
        axs[0, i].imshow(data_dict['lsm'][i], cmap=cmap)
        axs[0, i].set_title('LSM')
        axs[0, i].axis('off')
        axs[1, i].imshow(data_dict['topo'][i], cmap=cmap)
        axs[1, i].set_title('Topography')
        axs[1, i].axis('off')
        axs[2, i].imshow(data_dict['sdf'][i], cmap=cmap)
        axs[2, i].set_title('SDF')
        axs[2, i].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()