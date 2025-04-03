#!/bin/bash
# run_dataset_test.sh
# This script runs the dataset_test.py with configurable parameters.
# Edit variables below to change configuration.
#   To run: 
#       - Make executable: chmod +x run_dataset_test.sh
#       - Run: ./run_dataset_test.sh

# Since scripts are one folder up, we need to set the PYTHONPATH to the parent directory
export PYTHONPATH=$(dirname "$0")/..


# HR parameters
HR_MODEL="DANRA"
HR_VAR="prcp"
HR_DIM="128"
HR_SCALING_METHOD="log_minus1_1"
HR_SCALING_PARAMS='{"glob_min": 0, "glob_max": 160, "glob_min_log": -20, "glob_max_log": 10, "glob_mean_log": -25.0, "glob_std_log": 10.0, "buffer_frac": 0.5}'
# HR cutout domain (e.g. for HR images over Denmark)
CUTOUT_DOMAINS="[170,350,340,520]" #"[195,323,365,493]" #"[170,350,340,520]" # # 170, 170+180, 340, 340+180

# LR parameters
LR_MODEL="ERA5"
# List LR conditions as a JSON like string (assuming use of str2list helper function)
LR_CONDITIONS='["prcp", "temp"]'
LR_SCALING_METHODS='["log_minus1_1", "zscore"]'
# LR_SCALING_PARAMS='["{"glob_min": 0, "glob_max": 70, "glob_min_log": -10, "glob_max_log": 5, "glob_mean_log": -25.0, "glob_std_log": 10.0, "buffer_frac": 0.5}", "{"glob_mean": 8.69251, "glob_std": 6.192434}"]'
LR_SCALING_PARAMS='["{\"glob_min\": 0, \"glob_max\": 70, \"glob_min_log\": -10, \"glob_max_log\": 5, \"glob_mean_log\": -25.0, \"glob_std_log\": 10.0, \"buffer_frac\": 0.5}", "{\"glob_mean\": 8.69251, \"glob_std\": 6.192434}"]'

# New parameters for LR cropping
# To use full LR size (589x789), set:
#   LR_DATA_SIZE="[589, 789]"
# To use cropped LR size, e.g. 128x128, set:
#   LR_DATA_SIZE="[128, 128]"
LR_DATA_SIZE="[589, 789]" #"[128, 128]"  #
# Optionally, if you want a seperate cutout region for LR conditions and geo variables:
# For full LR, you might as well set this to the full extent:
LR_CUTOUT_DOMAINS="[0,589,0,789]"
# If you want to use the same as HR cutout, set to None
# LR_CUTOUT_DOMAINS="None"

# Other parameters 
FORCE_MATCHING_SCALE="false"                            # if true, HR and LR variables of same kind forced on matching colorbar scale
SAMPLE_W_GEO="true"                                     # if true, sample with geo variables
SAMPLE_W_CUTOUTS="true"                                 # if true, sample with cutouts 
SAMPLE_W_COND_SEASON="true"                             # if true, sample with seasonal conditions
SAMPLE_W_SDF="true"                                     # if true, sample with SDF (for loss weighting)
SCALING="true"                                          # if true, scale the data         
PATH_DATA="/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/Data/Data_DiffMod/" # Path to the data
SAVE_FIGS="true"                                        # if true, save the figures
SPECIFIC_FIG_NAME="test_with_prcp_temp_full_LR_domain'" # Name of the specific figure to save
SHOW_FIGS="true"                                        # if true, show the figures                
SHOW_BOTH_ORIG_SCALED="false"                           # if true, show both original and scaled data in figures
SHOW_GEO="true"                                         # if true, show geo variables in figures
SHOW_OCEAN="false"                                      # if true, show ocean variables in figures                
PATH_SAVE="/Users/au728490/Library/CloudStorage/OneDrive-Aarhusuniversitet/PhD_AU/Python_Scripts/DiffusionModels/SBGM_modular/test_bash/figures_test/" # Path to save the figures
TOPO_MIN="-12"                                          # minimum value for topography (in LR domain)         
TOPO_MAX="12"                                           # maximum value for topography (in LR domain)
NORM_MIN="0"                                            # minimum value for normalization (in LR domain)
NORM_MAX="1"                                            # maximum value for normalization (in LR domain)
N_SEASONS="4"                                           # number of seasons
N_GEN_SAMPLES="3"                                       # number of generated samples
NUM_WORKERS="4"                                         # number of workers for data loading



# Run the dataset_test.py script with the specified parameters
python ${PYTHONPATH}/dataset_test.py \
    --hr_model "$HR_MODEL" \
    --hr_var "$HR_VAR" \
    --hr_dim "$HR_DIM" \
    --hr_scaling_method "$HR_SCALING_METHOD" \
    --hr_scaling_params "$HR_SCALING_PARAMS" \
    --cutout_domains "$CUTOUT_DOMAINS" \
    --lr_model "$LR_MODEL" \
    --lr_conditions "$LR_CONDITIONS" \
    --lr_scaling_methods "$LR_SCALING_METHODS" \
    --lr_scaling_params "$LR_SCALING_PARAMS" \
    --lr_data_size "$LR_DATA_SIZE" \
    --lr_cutout_domains "$LR_CUTOUT_DOMAINS" \
    --force_matching_scale "$FORCE_MATCHING_SCALE" \
    --sample_w_geo "$SAMPLE_W_GEO" \
    --sample_w_cutouts "$SAMPLE_W_CUTOUTS" \
    --sample_w_cond_season "$SAMPLE_W_COND_SEASON" \
    --sample_w_sdf "$SAMPLE_W_SDF" \
    --scaling "$SCALING" \
    --path_data "$PATH_DATA" \
    --save_figs "$SAVE_FIGS" \
    --specific_fig_name "$SPECIFIC_FIG_NAME" \
    --show_figs "$SHOW_FIGS" \
    --show_both_orig_scaled "$SHOW_BOTH_ORIG_SCALED" \
    --show_geo "$SHOW_GEO" \
    --show_ocean "$SHOW_OCEAN" \
    --path_save "$PATH_SAVE" \
    --topo_min "$TOPO_MIN" \
    --topo_max "$TOPO_MAX" \
    --norm_min "$NORM_MIN" \
    --norm_max "$NORM_MAX" \
    --n_seasons "$N_SEASONS" \
    --n_gen_samples "$N_GEN_SAMPLES" \
    --num_workers "$NUM_WORKERS"