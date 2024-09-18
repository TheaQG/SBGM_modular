import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt




class Evaluation:
    '''
        Class to evaluate generated samples (saved in npz files) from the SBGM model.

        Evaluates the generated samples using the following metrics:
        - All pixel values distribution (across space and time), generated and eval images
        - RMSE and MAE for all pixels in all samples
    '''
    def __init__(self, args, generated_sample_type='repeated', n_samples=8):
        '''
            Setup the evaluation class.
            Get configuration parameters from args.
            Load samples from config path.
            
            - sample_type: What type of generated samples to evaluate. ['repeated', 'single', 'multiple'] 
        '''


        self.var = args.HR_VAR
    
        self.danra_size = args.HR_SIZE
        self.image_size = self.danra_size
        self.loss_type = args.loss_type
        self.n_seasons = args.season_shape[0]
        self.noise_variance = args.noise_variance

        self.config_name = args.config_name
        self.save_str = self.config_name + '__' + self.var + '__' + str(self.image_size) + 'x' + str(self.image_size) + '__' + self.loss_type + '__' + str(self.n_seasons) + '_seasons' + '__' + str(self.noise_variance) + '_noise' + '__' + str(args.num_heads) + '_heads' + '__' + str(args.n_timesteps) + '_timesteps'

        self.PATH_SAVE = args.path_save
        self.PATH_GENERATED_SAMPLES = self.PATH_SAVE + 'evaluation/generated_samples/' + self.save_str + '/'

        print(f'\nLoading generated samples from {self.PATH_GENERATED_SAMPLES}')

        self.generated_sample_type = generated_sample_type
        print(f'Type of generated samples: {self.generated_sample_type}\n')


        self.FIG_PATH = args.path_save + 'evaluation/plot_samples/' + self.save_str + '/' 
        

        if not os.path.exists(self.FIG_PATH):
            print(f'Creating directory {self.FIG_PATH}')
            os.makedirs(self.FIG_PATH)

        if self.generated_sample_type == 'repeated':
            load_str = '_repeatedSamples_' + str(n_samples) + '.npz'
        elif self.generated_sample_type == 'single':
            load_str = '_singleSample.npz'
        elif self.generated_sample_type == 'multiple':
            load_str ='_samples_' + str(n_samples) + '_distinctSamples.npz'

        # Load generated images, truth evaluation images and lsm mask for each image
        self.gen_imgs = np.load(self.PATH_GENERATED_SAMPLES + 'gen' + load_str)['arr_0']
        self.eval_imgs = np.load(self.PATH_GENERATED_SAMPLES + 'eval' + load_str)['arr_0']
        self.lsm_imgs = np.load(self.PATH_GENERATED_SAMPLES + 'lsm' + load_str)['arr_0']
        self.cond_imgs = np.load(self.PATH_GENERATED_SAMPLES + 'cond' + load_str)['arr_0']

        # Convert to torch tensors
        self.gen_imgs = torch.from_numpy(self.gen_imgs).squeeze()
        self.eval_imgs = torch.from_numpy(self.eval_imgs).squeeze()
        self.lsm_imgs = torch.from_numpy(self.lsm_imgs).squeeze()
        self.cond_imgs = torch.from_numpy(self.cond_imgs).squeeze()


        print(self.gen_imgs.shape)
        print(self.eval_imgs.shape)
        print(self.lsm_imgs.shape)
        print(self.cond_imgs.shape)


    def plot_example_images(self,
                            masked=False,
                            plot_with_cond=False,
                            plot_with_lsm=False,
                            n_samples=0,
                            same_cbar=True,
                            show_figs=False,
                            save_figs=False
                            ):
        '''
            Plot example of generated and eval images w or w/o masking
        '''


        # Set number of samples to plot
        if self.generated_sample_type == 'single':
            n_samples = 1
        else:
            if n_samples == 0 and self.gen_imgs.shape[0] < 8:
                n_samples = self.gen_imgs.shape[0]
            elif n_samples == 0 and self.gen_imgs.shape[0] >= 8:
                n_samples = 8

        print(f'Plotting {n_samples} samples')
        # If only one sample is plotted, unsqueeze to add batch dimension
        if n_samples == 1:
            gen_imgs = self.gen_imgs.unsqueeze(0)
            eval_imgs = self.eval_imgs.unsqueeze(0)
        else:
            gen_imgs = self.gen_imgs[:n_samples]
            if self.generated_sample_type == 'repeated':
                # Repeate the eval image n_samples times
                eval_imgs = self.eval_imgs.repeat(n_samples, 1, 1)
            else:
                eval_imgs = self.eval_imgs[:n_samples]
            

        # Define lists of images to plot
        plot_list = [gen_imgs, eval_imgs]
        plot_titles = ['Generated image', 'Evaluation image']


        # Add conditional images and LSM mask to plot list if specified
        if plot_with_cond:
            if self.generated_sample_type == 'repeated':
                cond_imgs = self.cond_imgs.repeat(n_samples, 1, 1)
            else:
                cond_imgs = self.cond_imgs

            if n_samples == 1:
                cond_imgs = self.cond_imgs.unsqueeze(0)
            plot_list.append(cond_imgs)
            plot_titles.append('Conditional image')


        if self.generated_sample_type == 'repeated':
            lsm_imgs = self.lsm_imgs.repeat(n_samples, 1, 1)
        else:
            lsm_imgs = self.lsm_imgs
        if n_samples == 1:
            lsm_imgs = self.lsm_imgs.unsqueeze(0)
        
        if plot_with_lsm:
            plot_list.append(lsm_imgs)
            plot_titles.append('LSM mask')


        # Set number of axes and figure/axis object
        n_axs = len(plot_list)
        fig, axs = plt.subplots(n_axs, n_samples, figsize=(n_samples*2, n_axs*2))

        # If only one sample is plotted, unsqueeze axis to be able to iterate over it
        if n_samples == 1:
            axs = np.expand_dims(axs, axis=0)


        # Plot on same colorbar, if same_cbar is True
        if same_cbar:
            vmin = np.nanmin([plot_list[i].min() for i in range(n_axs)])
            vmax = np.nanmax([plot_list[i].max() for i in range(n_axs)])
        else:
            vmin = None
            vmax = None


        # Loop over samples and plot (n_axs x n_samples) images
        for i in range(n_samples):
            for j in range(n_axs):
                # If masked, set ocean pixels to nan
                if masked and plot_titles[j] != 'LSM mask':
                    plot_list[j][i][lsm_imgs[i]==0] = np.nan

                # Add plot_title to first image in each row (as y-label)
                if i == 0:
                    axs[j, i].set_ylabel(plot_titles[j], fontsize=14)
                # If lsm, set vmin and vmax to 0 and 0.1
                if plot_titles[j] == 'LSM mask':
                    im = axs[j, i].imshow(plot_list[j][i], vmin=0, vmax=0.1)
                else:
                    im = axs[j, i].imshow(plot_list[j][i], vmin=vmin, vmax=vmax)
                axs[j, i].set_ylim([0,plot_list[j][i].shape[0]])
                axs[j, i].set_xticks([])
                axs[j, i].set_yticks([])
                
                # If colorbar is the same for all images, only add it to the last image in each row
                if same_cbar and i == n_samples-1:
                    fig.colorbar(im, ax=axs[j, i], fraction=0.046, pad=0.04)
                elif not same_cbar:
                    fig.colorbar(im, ax=axs[j, i], fraction=0.046, pad=0.04)

        fig.tight_layout()

        # Save figure if specified
        if save_figs:
            if masked:
                fig.savefig(self.FIG_PATH + self.save_str + '__example_eval_gen_images_masked.png', dpi=600, bbox_inches='tight')
            else:
                fig.savefig(self.FIG_PATH + self.save_str + '__example_eval_gen_images.png', dpi=600, bbox_inches='tight')

        # Show figure if specified
        if show_figs:
            plt.show()
        else:
            plt.close(fig)

        return fig, axs



    def full_pixel_statistics(self,
                              show_figs=False,
                              save_figs=False,
                              save_stats=False,
                              save_path=None
                              ):
        '''
            Calculate pixel-wise statistics for the generated samples.
            - RMSE and MAE for all pixels in all samples
            - Pixel value distribution for all samples
        '''
        
        
        #########################
        #                       #
        # FULL PIXEL STATISTICS #
        #                       #
        #########################
        
        # Calculate total single pixel-wise MAE and RMSE for all samples, no averaging
        # Flatten and concatenate the generated and eval images
        gen_imgs_flat = self.gen_imgs.flatten()
        eval_imgs_flat = self.eval_imgs.flatten()

        # Plot the pixel-wise distribution of the generated and eval images
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(gen_imgs_flat, bins=50, alpha=0.5, label='Generated')
        ax.hist(eval_imgs_flat, bins=50, alpha=0.5, color='r', label='Eval')
        ax.axvline(x=np.nanmean(eval_imgs_flat), color='r', alpha=0.5, linestyle='--', label=f'Eval mean, {np.nanmean(eval_imgs_flat):.2f}')
        ax.axvline(x=np.nanmean(gen_imgs_flat), color='b', alpha=0.5, linestyle='--', label=f'Generated mean, {np.nanmean(gen_imgs_flat):.2f}')
        ax.set_title(f'Distribution of generated and eval images, bias: {np.nanmean(gen_imgs_flat)-np.nanmean(eval_imgs_flat):.2f}', fontsize=14)
        ax.set_xlabel(f'Pixel value', fontsize=14)
        ax.set_ylabel(f'Count', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        # Set the x-axis limits to 4 sigma around the mean of the eval images
        x_min = np.nanmin([np.nanmin(eval_imgs_flat), np.nanmin(gen_imgs_flat)])
        x_max = np.nanmax([np.nanmax(eval_imgs_flat), np.nanmax(gen_imgs_flat)])
        ax.set_xlim([x_min, x_max])
        ax.legend()

        fig.tight_layout()

        # Save figure if specified
        if save_figs:
            fig.savefig(self.FIG_PATH + self.save_str + '__pixel_distribution.png', dpi=600, bbox_inches='tight')

        # Show figure if specified
        if show_figs:
            plt.show()
        else:
            plt.close(fig)

        # Save statistics if specified
        if save_stats:
            np.savez(save_path + self.save_str + '__pixel_distribution.npz', gen_imgs_flat, eval_imgs_flat)
        


        ###############################
        #                             #
        # RMSE AND MAE FOR ALL PIXELS #
        #                             # 
        ###############################

        # Calculate MAE and RMSE for all samples ignoring nans
        mae_all = torch.abs(gen_imgs_flat - eval_imgs_flat)
        rmse_all = torch.sqrt(torch.square(gen_imgs_flat - eval_imgs_flat))

        # Make figure with two plots: RMSE and MAE for all pixels
        fig, axs = plt.subplots(2, 1, figsize=(12,6))#, sharex='col')

        axs[0].hist(rmse_all, bins=150, alpha=0.7, label='RMSE all pixels', edgecolor='k')
        axs[0].set_title(f'RMSE for all pixels', fontsize=16)
        axs[0].tick_params(axis='y', which='major', labelsize=14)
        axs[0].set_ylabel(f'Count', fontsize=16)

        axs[1].hist(mae_all, bins=70, alpha=0.7, label='MAE all pixels', edgecolor='k')
        axs[1].set_title(f'MAE for all pixels', fontsize=16)
        axs[1].tick_params(axis='both', which='major', labelsize=14)
        axs[1].set_xlabel(f'RMSE', fontsize=16)
        axs[1].set_ylabel(f'Count', fontsize=16)

        fig.tight_layout()

        # Save figure if specified
        if save_figs:
            fig.savefig(self.FIG_PATH + self.save_str + '__RMSE_MAE_histograms.png', dpi=600, bbox_inches='tight')
        
        # Show figure if specified
        if show_figs:
            plt.show()
        else:
            plt.close(fig)

        # Save statistics if specified
        if save_stats:
            np.savez(save_path + self.save_str + '__pixel_statistics.npz', mae_all, rmse_all)





    def daily_statistics(self,
                        plot_stats=False,
                        save_plots=False,
                        save_plot_path=None,
                        save_stats=False,
                        save_path=None
                        ):
        '''
            Calculate daily average MAE and RMSE for all samples (average over spatial dimensions) ignoring nans
        '''

        # Calculate daily average MAE and RMSE for all samples (average over spatial dimensions) ignoring nans
        mae_daily = torch.abs(self.gen_imgs - self.eval_imgs).nanmean(dim=(1,2))
        rmse_daily = torch.sqrt(torch.square(self.gen_imgs - self.eval_imgs).nanmean(dim=(1,2)))


                                

    def spatial_statistics(self,
                           show_figs=False,
                           save_figs=False,
                           save_plot_path=None,
                           save_stats=False,
                           save_path=None
                           ):
        '''
            Calculate spatial statistics for the generated samples.
            - Moran's I
            - Bias per pixel
            - Bias per image
            - Bias per pixel per image
        '''
        
        # Calculate rmse per pixel
        rmse_per_pixel = torch.sqrt(torch.square(self.gen_imgs - self.eval_imgs).nanmean(dim=0))

        # Calculate mae per pixel
        mae_per_pixel = torch.abs(self.gen_imgs - self.eval_imgs).nanmean(dim=0)

        # Calculate bias per pixel
        bias_per_pixel = self.gen_imgs.nanmean(dim=0) - self.eval_imgs.nanmean(dim=0)

        # Plot the spatial statistics
        fig, axs = plt.subplots(2, 2, figsize=(12,12))

        im = axs[0,0].imshow(rmse_per_pixel)
        axs[0,0].set_title(f'RMSE per pixel', fontsize=16)
        fig.colorbar(im, ax=axs[0,0])

        im = axs[0,1].imshow(mae_per_pixel)
        axs[0,1].set_title(f'MAE per pixel', fontsize=16)
        fig.colorbar(im, ax=axs[0,1])

        im = axs[1,0].imshow(bias_per_pixel)
        axs[1,0].set_title(f'Bias per pixel', fontsize=16)
        fig.colorbar(im, ax=axs[1,0])

        fig.tight_layout()

        # Save figure if specified
        if save_figs:
            fig.savefig(self.FIG_PATH + self.save_str + '__spatial_statistics.png', dpi=600, bbox_inches='tight')

        # Show figure if specified
        if show_figs:
            plt.show()
        else:
            plt.close(fig)






# def evaluate_sbgm(args):
#     print('\n\n')

#     var = args.HR_VAR
    
#     danra_size = args.HR_SIZE
#     image_size = danra_size
#     loss_type = args.loss_type
#     n_seasons = args.season_shape[0]
#     noise_variance = args.noise_variance

#     config_name = args.config_name
#     save_str = config_name + '__' + var + '__' + str(image_size) + 'x' + str(image_size) + '__' + loss_type + '__' + str(n_seasons) + '_seasons' + '__' + str(noise_variance) + '_noise' + '__' + str(args.num_heads) + '_heads' + '__' + str(args.n_timesteps) + '_timesteps'

#     PATH_SAVE = args.path_save
#     PATH_GENERATED_SAMPLES = PATH_SAVE + 'evaluation/generated_samples/' + save_str + '/'



#     FIG_PATH = args.path_save + 'evaluation/plot_samples/' + save_str + '/' 
#     print(FIG_PATH)
#     if not os.path.exists(FIG_PATH):
#         print(f'Creating directory {FIG_PATH}')
#         os.makedirs(FIG_PATH)
#     # Load generated images, truth evaluation images and lsm mask for each image
#     gen_imgs = np.load(PATH_GENERATED_SAMPLES + 'gen_samples.npz')['arr_0']
#     eval_imgs = np.load(PATH_GENERATED_SAMPLES + 'eval_samples.npz')['arr_0']
#     lsm_imgs = np.load(PATH_GENERATED_SAMPLES + 'lsm_samples.npz')['arr_0']

    



#     # Convert to torch tensors
#     gen_imgs = torch.from_numpy(gen_imgs).squeeze()
#     eval_imgs = torch.from_numpy(eval_imgs).squeeze()
#     lsm_imgs = torch.from_numpy(lsm_imgs).squeeze()

#     # Plot example of generated and eval images w/o masking
#     plot_idx = np.random.randint(0, len(gen_imgs))

#     fig, axs = plt.subplots(1, 2, figsize=(10,4))
#     im1 = axs[0].imshow(eval_imgs[plot_idx])
#     axs[0].set_ylim([0,eval_imgs[plot_idx].shape[0]])
#     axs[0].set_title(f'Evaluation image', fontsize=14)
#     # Remove ticks and labels
#     axs[0].set_xticks([])
#     axs[0].set_yticks([])
#     fig.colorbar(im1, ax=axs[0])

#     im = axs[1].imshow(gen_imgs[plot_idx])
#     axs[1].set_ylim([0,gen_imgs[plot_idx].shape[0]])
#     axs[1].set_title(f'Generated image', fontsize=14)
#     # Remove ticks and labels
#     axs[1].set_xticks([])
#     axs[1].set_yticks([])
#     fig.colorbar(im, ax=axs[1])
#     fig.tight_layout()

# #    fig.savefig(FIG_PATH + cond_str + '__example_eval_gen_images.png', dpi=600, bbox_inches='tight')


#     # Mask out ocean pixels, set to nan
#     for i in range(len(gen_imgs)):
#         gen_imgs[i][lsm_imgs[i]==0] = np.nan
#         eval_imgs[i][lsm_imgs[i]==0] = np.nan

#     # Plot a sample of the generated and eval images
#     fig, axs = plt.subplots(1, 2, figsize=(10,4))
#     im1 = axs[0].imshow(eval_imgs[plot_idx])
#     axs[0].set_ylim([0,eval_imgs[plot_idx].shape[0]])
#     axs[0].set_title(f'Evaluation image', fontsize=14)
#     axs[0].set_xticks([])
#     axs[0].set_yticks([])
#     fig.colorbar(im1, ax=axs[0])


#     im = axs[1].imshow(gen_imgs[plot_idx])
#     axs[1].set_ylim([0,gen_imgs[plot_idx].shape[0]])
#     axs[1].set_title(f'Generated image', fontsize=14)
#     axs[1].set_xticks([])
#     axs[1].set_yticks([])
#     fig.colorbar(im, ax=axs[1])
#     fig.tight_layout()

#     fig.savefig(FIG_PATH + save_str + '__example_eval_gen_images_masked.png', dpi=600, bbox_inches='tight')


#     # Now evaluate the generated samples
#     print("Evaluating samples...")

#     # Calculate daily average MAE and RMSE for all samples (average over spatial dimensions) ignoring nans
#     mae_daily = torch.abs(gen_imgs - eval_imgs).nanmean(dim=(1,2))
#     rmse_daily = torch.sqrt(torch.square(gen_imgs - eval_imgs).nanmean(dim=(1,2)))


#     # Calculate total single pixel-wise MAE and RMSE for all samples, no averaging
#     # Flatten and concatenate the generated and eval images
#     gen_imgs_flat = gen_imgs.flatten()
#     eval_imgs_flat = eval_imgs.flatten()

#     # Calculate MAE and RMSE for all samples ignoring nans
#     mae_all = torch.abs(gen_imgs_flat - eval_imgs_flat)
#     rmse_all = torch.sqrt(torch.square(gen_imgs_flat - eval_imgs_flat))

#     # Make figure with four plots: MAE daily histogram, RMSE daily histogram, MAE pixel-wise histogram, RMSE pixel-wise histogram
#     fig, axs = plt.subplots(2, 1, figsize=(12,6), sharex='col')
#     # axs[0,0].hist(mae_daily, bins=50)
#     # axs[0,0].set_title(f'MAE daily')
#     # # axs[0,0].set_xlabel(f'MAE')
#     # # axs[0,0].set_ylabel(f'Count')

#     # axs[0].hist(mae_all, bins=70)
#     # axs[0].set_title(f'MAE for all pixels')
#     # axs[0].set_xlabel(f'MAE')
#     # axs[0].set_ylabel(f'Count')

#     axs[0].hist(rmse_daily, bins=150, alpha=0.7, label='RMSE daily', edgecolor='k')
#     axs[0].set_title(f'RMSE daily', fontsize=16)
#     axs[0].tick_params(axis='y', which='major', labelsize=14)
#     #axs[0].set_xlabel(f'RMSE')
#     axs[0].set_ylabel(f'Count', fontsize=16)

#     axs[1].hist(rmse_all, bins=150, alpha=0.7, label='RMSE all pixels', edgecolor='k')
#     axs[1].set_title(f'RMSE for all pixels', fontsize=16)
#     axs[1].tick_params(axis='both', which='major', labelsize=14)
#     axs[1].set_xlabel(f'RMSE', fontsize=16)
#     axs[1].set_ylabel(f'Count', fontsize=16)
#     axs[1].set_xlim([0, 25])

#     fig.tight_layout()
#     fig.savefig(FIG_PATH + save_str + '__RMSE_histograms.png', dpi=600, bbox_inches='tight')


#     # Plot the pixel-wise distribution of the generated and eval images
#     fig, ax = plt.subplots(figsize=(8,4))
#     ax.hist(gen_imgs.flatten(), bins=50, alpha=0.5, label='Generated')
#     ax.hist(eval_imgs.flatten(), bins=50, alpha=0.5, color='r', label='Eval')
#     ax.axvline(x=np.nanmean(eval_imgs.flatten()), color='r', alpha=0.5, linestyle='--', label=f'Eval mean, {np.nanmean(eval_imgs.flatten()):.2f}')
#     ax.axvline(x=np.nanmean(gen_imgs.flatten()), color='b', alpha=0.5, linestyle='--', label=f'Generated mean, {np.nanmean(gen_imgs.flatten()):.2f}')
#     ax.set_title(f'Distribution of generated and eval images, bias: {np.nanmean(gen_imgs.flatten())-np.nanmean(eval_imgs.flatten()):.2f}', fontsize=14)
#     ax.set_xlabel(f'Pixel value', fontsize=14)
#     ax.set_ylabel(f'Count', fontsize=14)
#     ax.tick_params(axis='both', which='major', labelsize=12)
#     # Set the x-axis limits to 4 sigma around the mean of the eval images
#     x_min = np.nanmin([np.nanmin(eval_imgs.flatten()), np.nanmin(gen_imgs.flatten())])
#     x_max = np.nanmax([np.nanmax(eval_imgs.flatten()), np.nanmax(gen_imgs.flatten())])
#     ax.set_xlim([x_min, x_max])
# #    ax.set_xlim([np.nanmean(eval_imgs.flatten())-4*np.nanstd(eval_imgs.flatten()), np.nanmean(eval_imgs.flatten())+4*np.nanstd(eval_imgs.flatten())])
#     ax.legend()

#     fig.tight_layout()
#     fig.savefig(FIG_PATH + save_str + '__pixel_distribution.png', dpi=600, bbox_inches='tight')

#     plt.show()



#     # # Get the LSM mask for the area that the generated images are cropped from
#     # CUTOUTS = True
#     # CUTOUT_DOMAINS = [170, 170+180, 340, 340+180]

#     # PATH_LSM_FULL = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_lsm/truth_fullDomain/lsm_full.npz'
#     # PATH_TOPO_FULL = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_topo/truth_fullDomain/topo_full.npz'

#     # data_lsm_full = np.flipud(np.load(PATH_LSM_FULL)['data'])[CUTOUT_DOMAINS[0]:CUTOUT_DOMAINS[1], CUTOUT_DOMAINS[2]:CUTOUT_DOMAINS[3]]
#     # data_topo_full = np.flipud(np.load(PATH_TOPO_FULL)['data'])[CUTOUT_DOMAINS[0]:CUTOUT_DOMAINS[1], CUTOUT_DOMAINS[2]:CUTOUT_DOMAINS[3]]

#     # # Load the points for the cutouts
#     # points_imgs = np.load(SAVE_PATH + 'point_samples__' + SAVE_NAME)['arr_0']





#     # NEED TO TAKE IMAGE SHIFT INTO ACCOUNT FOR THE PIXEL-WISE IMAGES 
#     # # Calculate pixel-wise MAE and RMSE for all samples (average over temporal dimension)
#     # mae_pixel = torch.abs(gen_imgs - eval_imgs).mean(dim=0)
#     # rmse_pixel = torch.sqrt(torch.square(gen_imgs - eval_imgs).mean(dim=0))


#     # # Plot image of MAE and RMSE for temporal average
#     # fig, axs = plt.subplots(1, 2, figsize=(12,4))
#     # im1 = axs[0].imshow(mae_pixel)
#     # axs[0].set_ylim([0,mae_pixel.shape[0]])
#     # axs[0].set_title(f'MAE pixel-wise')
#     # fig.colorbar(im1, ax=axs[0])

#     # im2 = axs[1].imshow(rmse_pixel)
#     # axs[1].set_title(f'RMSE pixel-wise')
#     # axs[1].set_ylim([0,rmse_pixel.shape[0]])
#     # fig.colorbar(im2, ax=axs[1])



#     # # Calculate Pearson correlation coefficient for all samples
#     # for gen_im, ev_im in zip(gen_imgs, eval_imgs):
#     #     corr = np.ma.corrcoef(np.ma.masked_invalid(gen_im), np.ma.masked_invalid(ev_im))
#     #     print(corr)



#     plt.show()








# # FID score
# # Heidke/Pierce skill score (precipitation)
# # EV analysis (precipitation)
# # Moran's I (spatial autocorrelation)
# # Bias per pixel (spatial bias) 
# # Bias per image (temporal bias)
# # Bias per pixel per image (spatio-temporal bias)





