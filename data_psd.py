import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, fftshift
from torch.utils.data import DataLoader



class DataPSD:
    def __init__(self, truth_file,
                 generated_file,
                 conditional_file=None,
                 keys=['arr_0', 'arr_0', 'arr_0'],
                 save_figs=True,
                 save_path='.',
                 show_figs=True
                 ):
        self.save_figs = save_figs
        self.save_path = save_path
        self.show_figs = show_figs

        self.truth_data = np.load(truth_file)[keys[0]]
        self.generated_data = np.load(generated_file)[keys[1]]
        print('truth_data', self.truth_data.shape)
        print('generated_data', self.generated_data.shape)
        
        # Unsqueeze the data if it is 2D
        if len(self.generated_data.shape) == 2:
            # Unsqueeze once to add the channel dimension   
            self.generated_data = np.expand_dims(self.generated_data, axis=0)
            # Unsqueeze again to add the batch dimension
            self.generated_data = np.expand_dims(self.generated_data, axis=0)





        if conditional_file is not None:
            self.conditional_data = np.load(conditional_file)[keys[2]]
            print('conditional_data', self.conditional_data.shape)
        else:
            self.conditional_data = None
        
        self.truth_ps_list = []
        self.generated_ps_list = []
        self.conditional_ps_list = []

    def compute_power_spectrum(self, image):
        # Squeeze to make sure the image is 2D
        image = np.squeeze(image)

        # Perform the Fourier transform
        ft = fftn(image)
        
        # Shift the zero frequency component to the center
        ft_shifted = fftshift(ft)
        
        # Compute the power spectrum (magnitude squared of the Fourier transform)
        power_spectrum = np.abs(ft_shifted) ** 2
        
        # Compute the radial average of the power spectrum
        ps_radial = self.radial_average(power_spectrum)
        
        return ps_radial

    def radial_average(self,
                       data
                       ):
        if data.ndim > 2:
            data = data.mean(axis=tuple(range(data.ndim - 2)))

        y, x = np.indices(data.shape)
        center = np.array([(x.max() - x.min()) / 2, (y.max() - y.min()) / 2])
        r = np.hypot(x - center[0], y - center[1])

        # Bin the radii 
        rbin = (np.rint(r)).astype(int)
        radial_mean = np.bincount(rbin.ravel(), data.ravel()) / np.bincount(rbin.ravel())
        
        return radial_mean

    def process_data(self,
                     psd_data_mode='pooled' # 'mean' or 'pooled'
                     ):
        # Process truth and generated datasets
        for truth_image, generated_image in zip(self.truth_data, self.generated_data):
            truth_ps = self.compute_power_spectrum(truth_image)
            generated_ps = self.compute_power_spectrum(generated_image)
            
            self.truth_ps_list.append(truth_ps)
            self.generated_ps_list.append(generated_ps)

        if self.conditional_data is not None:
            # Process conditional dataset if available
            for conditional_image in self.conditional_data:
                conditional_ps = self.compute_power_spectrum(conditional_image)
                self.conditional_ps_list.append(conditional_ps)
        
        if psd_data_mode == 'mean':
            # Average the power spectra over all images
            self.truth_ps = np.mean(self.truth_ps_list, axis=0)
            self.generated_ps = np.mean(self.generated_ps_list, axis=0)
            if self.conditional_data is not None:
                self.conditional_ps = np.mean(self.conditional_ps_list, axis=0)
        
        elif psd_data_mode == 'pooled':
            # Pool the power spectra over all images
            self.truth_ps = np.array(self.truth_ps_list)
            self.generated_ps = np.array(self.generated_ps_list)
            if self.conditional_data is not None:
                self.conditional_ps = np.array(self.conditional_ps_list)
        print('truth_ps', self.truth_ps)


    def plot_psd(self):
        # Plot and compare the average power spectra
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Loop through the number of images in the dataset and plot the power spectra
        
        for i, (truth_ps, generated_ps) in enumerate(zip(self.truth_ps_list, self.generated_ps_list)):
            color = plt.cm.viridis(i / len(self.truth_ps_list))

            ax.plot(truth_ps, color=color)
            ax.plot(generated_ps, linestyle='--', color=color)

            # If last image, add legend
            if i == len(self.truth_ps_list) - 1:
                ax.plot(truth_ps, label=f'Truth Data {i}', color=color)
                ax.plot(generated_ps, label=f'Generated Data {i}', linestyle='--', color=color)
                

        if self.conditional_data is not None:
            for i, conditional_ps in enumerate(self.conditional_ps_list):
                color = plt.cm.viridis(i / len(self.conditional_ps_list))

                ax.plot(conditional_ps, linestyle='-.', color=color)

                if i == len(self.conditional_ps_list) - 1:
                    ax.plot(conditional_ps, label=f'Conditional Data {i}', linestyle='-.', color=color)

        # Set y-axis to log scale
        ax.set_yscale('log')
        
        ax.set_xlabel('Wavenumber')
        ax.set_ylabel('Power Spectrum')
        ax.legend()
        ax.set_title('Power Spectrum Comparison')

        if self.save_figs:
            fig.savefig(self.save_path + 'psd_comparison.png')

        if self.show_figs:
            plt.show()

# Example usage:
# Assuming you have saved .npz files containing your datasets

# Initialize the DataPSD class
# psd_comparer = DataPSD('truth_data.npz', 'generated_data.npz', conditional_file='conditional_data.npz')

# Process the data to compute the power spectra
# psd_comparer.process_data()

# Plot the power spectrum comparison
# psd_comparer.plot_psd()


if __name__ == '__main__':
    # Try with generated samples from evaluation/generated_samples

    # Condition samples called 'cond_samples_8_distinctSamples.npz
    # Generated samples called 'gen_samples_8_distinctSamples.npz
    # Truth samples called 'eval_samples_8_distinctSamples.npz

    PATH_SAMPLES = 'evaluation/generated_samples/sbgm__temp__64x64__sdfweighted__4_seasons__0.01_noise__4_heads__1000_timesteps/'

    im_type_str = 'samples_8_distinctSamples' #'samples_8_distinctSamples' # 'singleSample', 'repeatedSamples', 'samples_8_distinctSamples'

    # Load the samples
    path_cond_samples = os.path.join(PATH_SAMPLES, 'cond_' + im_type_str + '.npz')
    path_gen_samples = os.path.join(PATH_SAMPLES, 'gen_' + im_type_str + '.npz')
    path_truth_samples = os.path.join(PATH_SAMPLES, 'eval_' + im_type_str + '.npz')

    psd_comparer = DataPSD(path_truth_samples, path_gen_samples, conditional_file=path_cond_samples)

    # Process the data to compute the power spectra
    psd_comparer.process_data()

    # Plot the power spectrum comparison
    psd_comparer.plot_psd()

    




    