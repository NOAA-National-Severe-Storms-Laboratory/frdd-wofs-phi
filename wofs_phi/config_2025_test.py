#==================================
# This is the configuration file
# for wofs_phi.py. It will hold 
# all of the constants necessary to
# run wofs_phi.py 
#==================================

import os
import pathlib

#TODO: It's possible the below won't be needed--Not sure yet. 
#Options are: 
#"forecast" for forecast mode in wofs viewer, 
#"warning" for warning mode in wofs viewer, and

#NOTE: mode will now be a parameter passed in to MLGenerator
#mode = "warning" 
#mode = "forecast"

#True if used for training; False if used for prediction/real-time
#NOTE: Not used, except in Ryan's code (with TORP stuff) 
is_train_mode = True

base_path = pathlib.Path(__file__).parent.resolve()

train_mode = 'train'
train_type = 'obs_and_warnings' #NOTE: Probably no longer needed, or change to train_types
train_radii = ['7.5', '15', '30', '39']
train_hazards = ['hail', 'wind', 'tornado']
train_lead_times = [90, 120]
forecast_length = 60
num_folds = 5
wofs_spinup_time = 25

ps_version = 2 #i.e., which probSevere version is being used 2 for v2, 3 for v3
include_torp_in_predictors = False

if (ps_version == 2) and (not include_torp_in_predictors):
    model_type = 'wofs_psv2_no_torp'
elif (ps_version == 2) and (include_torp_in_predictors):
    model_type = 'wofs_psv2_with_torp'
elif (ps_version == 3) and (not include_torp_in_predictors):
    model_type = 'wofs_psv3_no_torp'
elif (ps_version == 3) and (include_torp_in_predictors):
    model_type = 'wofs_psv3_with_torp'

plot_in_training = True

num_training_vars = 269
num_torp_vars = 90
if include_torp_in_predictors:
    num_training_vars += num_torp_vars

model_save_dir = '/work/ryan.martz/wofs_phi_data/%s_train/models/%s' %(train_type, model_type)
validation_dir = '/work/ryan.martz/wofs_phi_data/%s_train/validation_fcsts/%s' %(train_type, model_type)
test_dir = '/work/ryan.martz/wofs_phi_data/%s_train/test_fcsts/%s' %(train_type, model_type)
warning_dir = '' #will fill this in when we get around to re making new warnings

torp_vars_filename = '/work/ryan.martz/wofs_phi_data/training_data/predictors/torp_predictors.txt'

#Path to the trained rfs 
#rf_dir = "/work/ryan.martz/wofs_phi_data/models/wofs_psv2_no_torp/hail/wofslag_25/length_60"
rf_dir = "/work/eric.loken/wofs/2024_update/SFE2024/rf_models/latest"
sr_dir = ""

#Path where to save the ncdf files 
#ncdf_save_dir = "/work/eric.loken/wofs/2024_update/SFE2024/ncdf_files"
#ncdf_save_dir = "/home/eric.loken/python_packages/frdd-wofs-phi/wofs_phi/ncdf"

ncdf_save_dir = "/home/eric.loken/ncdf_test"

#TODO: Path indicating where to save the png files to
#May implement this later; for now, let's just save to the same location as the 
#netcdf files 
png_outdir = ""

######## Main things to change on cloud#######

#May or may not eventually use these
generate_forecasts = True #Generates the predictors array if True
generate_reports = False #Generates the reports file if True 
save_npy = False #Tells whether or not to save the npy predictor files 
save_ncdf = True #Tells whether or not to create/save the ncdf (realtime) files
plot_forecasts = True #Tells whether or not to create the .png files for wofs viewer

nc_outdir = "." #Where to place the final netcdf files #Needed for real time
#nc_outdir = "/home/eric.loken/python_packages/frdd-wofs-phi/wofs_phi/ncdf"
nc_outdir = "/home/eric.loken/ncdf_test"

#If True, use the ALL naming convention (will be true on cloud) 
#If False, use the legacy naming convention (e.g., ENS, ENV, SVR, etc.) 
use_ALL_files = True
#use_ALL_files = False

###############################################

#Buffer time for a report in minutes: 
#i.e., consider reports within this many minutes of the valid period
#as "yes" observations as well. 
#e.g., 10 means if our original valid period is from 0000-0100Z, we would
#consider yes reports to be from 2350-0110Z. 
report_time_buffer = 10 

#"lsrs" for training on local storm reports 
#"warnings" for training on warnings 
report_target = "lsrs"

#Fraction of data to randomly sample for training
sample_rate = 0.1 

#Path to full_npy directory for training
train_fcst_full_npy_dir = "/work/eric.loken/wofs/2024_update/SFE2024/fcst/full_npy"

train_obs_full_npy_dir = "/work/eric.loken/wofs/2024_update/SFE2024/obs/full_npy"

#Path to dat directory for training
train_fcst_dat_dir = "/work/eric.loken/wofs/2024_update/SFE2024/fcst/dat"
train_obs_dat_dir = "/work/eric.loken/wofs/2024_update/SFE2024/obs/dat"

train_obs_and_warnings_full_2d_npy_dir = '/work/ryan.martz/wofs_phi_data/training_data/obs_and_warnings/full_2d_obs_and_warnings'
train_obs_and_warnings_full_1d_npy_dir = '/work/ryan.martz/wofs_phi_data/training_data/obs_and_warnings/full_1d_obs_and_warnings'
train_obs_and_warnings_sampled_dat_dir = '/work/ryan.martz/wofs_phi_data/training_data/obs_and_warnings/sampled_1d_obs_and_warnings'

train_warnings_csv_dir = '/work/ryan.martz/wofs_phi_data/training_data/warnings/warning_csvs'
train_warnings_full_2d_npy_dir = '/work/ryan.martz/wofs_phi_data/training_data/warnings/full_2d_warnings'
train_warnings_full_1d_npy_dir = '/work/ryan.martz/wofs_phi_data/training_data/warnings/full_1d_warnings'
train_warnings_sampled_1d_dat_dir = '/work/ryan.martz/wofs_phi_data/training_data/warnings/sampled_1d_warnings'

raw_torp_training_path = "/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp"

#need to change this to reflect the real time directory!
#real_time_sr_map_dir = '/work/eric.loken/wofs/2024_update/SFE2024/sr_csv/latest'
real_time_sr_map_dir = '/work/eric.loken/wofs/2024_update/SFE2024/sr_csv/sfe2025_test'

#Path to the reports coordinates directory -- i.e., where are the coords.txt 
#files that need to be read in during training? 
reps_coords_dir = "/work/eric.loken/wofs/new_torn/storm_events_reports/fromThea"



#If True, use the ALL naming convention (will be true on cloud) 
#If False, use the legacy naming convention (e.g., ENS, ENV, SVR, etc.) 
#use_ALL_files = False

#wofs_base_path = "/work/mflora/SummaryFiles" #Obviously, will need to change on cloud

wofs_base_path = "/work2/wof/SummaryFiles"
max_cores = 30 #max number of cores to use for parallelization

dx_km = 3.0 #horizontal grid spacing of wofs in km 
ps_thresh = 0.01 #ps objects must have probs greater than or equal to this amount to be considered

#max_ps_extrap_time = 181.0 #Maximum amount of PS extrapolation time (used for setting min and max radius) 
max_ps_extrap_time = 241.0 #Maximum amount of PS extrapolation time (used for setting min and max radius) 

#Amount of time (in minutes) to go back (relative to first PS file) 
ps_time_to_go_back = 180.0
ps_wofs_buffer = 0

nan_replace_value = 0.0 #Replace nans in wofs files with this value 
#NOTE: We were getting nans primarily in mid-level lapse rate, 0-6km shear components, 
#STP, SCP, and freezing level

#radius (in km) for probSevere objects at time 0
min_radius = 1.5 #in km (for probSevere objects)

#radius (in km) for probSevere objects at the maximum extrapolation time 
#generally taken to be 181 minutes. 
#max_radius = 20.0 #in km (for probSevere objects) #Used to be 30.0, but that was much too big
max_radius = 1.5 

#conv_type = "square" #"square" or "circle" -- tells how to do the convolutions 
conv_type = "circle"

predictor_radii_km = [0.0, 15.0, 30.0, 45.0, 60.0] #how far to "look" spatially in km for predictors


#Fields not part of the standard wofs or ProbSevere fields that we'd like to include as extras 
extra_predictor_names = ["lat", "lon", "wofs_x", "wofs_y", "wofs_init_time"]


#obs_radii = ["39", "30", "15", "7.5"]
#obs_radii_str = ["39.0", "30.0", "15.0", "7.5"] 
#obs_radii_float = [39, 30, 15, 7.5]
#final_str_obs_radii = ["39", "30","15", "7_5"] #form to use for final ncdf files

obs_radii = ["15", "39"]
obs_radii_str = ["15.0", "39.0"]
obs_radii_float = [15, 39]
final_str_obs_radii = ["15", "39"] #form to use for final ncdf files 

final_hazards = ["hail", "wind", "tornado"] #for naming in final ncdf file 

wofs_fields_file = (base_path / "standard_wofs_variables.txt")
wofs_methods_file = (base_path / "standard_wofs_methods.txt")

all_fields_file = (base_path / "all_fields.txt") #Holds all the predictor fields
all_methods_file = (base_path / "all_methods.txt") #Holds all the preprocessing methods

#wofs_dir = "/work/mflora/SummaryFiles/"
wofs_dir = "/work2/wof/SummaryFiles/"
#if ps_version == 2:
#    ps_dir = "/work/eric.loken/wofs/probSevere/"
#else:
#    ps_dir = "/work/eric.loken/wofs/probSevere3_data/new_cintineo/PS_json/"

ps_dir = "/work/eric.loken/wofs/probSevere3_data/new_cintineo/PS_json/"

ps_search_minutes = 180 #how long before start time do we need to search for ps files to generate predictors
ps_recent_file_threshold = 10 #need a file in last __ minutes to do training/real time running

#Holds all the variables that just get taken from the point of prediction
single_pt_file = (base_path / "single_point_fields.txt")


bottom_hour_inits = ["1730", "1830", "1930", "2030", "2130", "2230", "2330", "0030", "0130",\
                     "0230", "0330", "0430"]
top_hour_inits = ["1700", "1800", "1900", "2000", "2100", "2200", "2300", "0000", "0100",\
                     "0200", "0300", "0400", "0500"]

all_wofs_init_times = ["1700", "1730", "1800", "1830", "1900", "1930", "2000", "2030", "2100",\
                        "2130", "2200", "2230", "2300", "2330", "0000", "0030", "0100", "0130",\
                        "0200", "0230", "0300", "0330", "0400", "0430", "0500", "0530", "0600",\
                        "0630", "0700", "0730", "0800", "0830", "0900", "0930", "1000", "1030",\
                        "1100", "1130"]

next_day_inits = ["0000", "0030", "0100", "0130", "0200", "0230", "0300", "0330", "0400", "0430", "0500",\
                    "0530", "0600", "0630", "0700", "0730", "0800", "0830", "0900", "0930", "1000", \
                    "1030", "1100", "1130"]

next_day_times = ["0000", "0005", "0010", "0015", "0020", "0025", "0030",\
                    "0030", "0035", "0040","0045", "0050", "0055",\
                    "0100", "0105", "0110", "0115", "0120", "0125", "0130",\
                    "0130", "0135", "0140","0145", "0150", "0155",\
                    "0200", "0205", "0210", "0215", "0220", "0225", "0230",\
                    "0230", "0235", "0240","0245", "0250", "0255",\
                    "0300", "0305", "0310", "0315", "0320", "0325", "0330",\
                    "0330", "0335", "0340","0345", "0350", "0355",\
                    "0400", "0405", "0410", "0415", "0420", "0425", "0430",\
                    "0430", "0435", "0440","0445", "0450", "0455",\
                    "0500", "0505", "0510", "0515", "0520", "0525", "0530",\
                    "0530", "0535", "0540","0545", "0550", "0555",\
                    "0600", "0605", "0610", "0615", "0620", "0625", "0630",\
                    "0630", "0635", "0640","0645", "0650", "0655",\
                    "0700", "0705", "0710", "0715", "0720", "0725", "0730",\
                    "0730", "0735", "0740","0745", "0750", "0755",\
                    "0800", "0805", "0810", "0815", "0820", "0825", "0830",\
                    "0830", "0835", "0840","0845", "0850", "0855",\
                    "0900", "0905", "0910", "0915", "0920", "0925", "0930",\
                    "0930", "0935", "0940","0945", "0950", "0955",\
                    "1000", "1005", "1010", "1015", "1020", "1025", "1030",\
                    "1030", "1035", "1040","1045", "1050", "1055"]            
    



wofs_reset_hour = 12
wofs_update_rate = 5 #wofs updates every __ minutes, currently 5, don't think that will change, but here just in case
ps_update_rate = 2 #ProbSevere updates every 2 minutes currently
pkl_dir = "." #Will probably need to update later

wofs_spinup_time = 25 #in minutes 
ps_spinup_time = 5 #in minutes; mostly used for warning mode 
wofs_bottom_init_min = 30

wofs_time_between_runs = 30 #in minutes: time between new wofs initializations 

torp_point_buffer = 7.5
torp_prob_change_1 = 5
torp_prob_change_2 = 10
torp_prob_change_1_str = 'torp_p_change_' + str(torp_prob_change_1) + '_min'
torp_prob_change_2_str = 'torp_p_change_' + str(torp_prob_change_2) + '_min'

torp_search_minutes = max(torp_prob_change_1, torp_prob_change_2)

torp_max_time_skip = 10

torp_conv_dists = [15, 30, 45, 60]

### If this environment variable is set, then it likely running in the cloud
if 'WOFS_ML_PATH' in os.environ:
    real_time_sr_map_dir = os.environ['WOFS_ML_PATH']
    rf_dir = os.environ['WOFS_ML_PATH']

torp_max_convs = ["torp_prob","torp_age",torp_prob_change_1_str,torp_prob_change_2_str,'azshear_max', 'azshear_max_trend', 'divshear_min',
       'divshear_min_trend', 'reflectivity_max', 'spectrumwidth_max',
       'spectrumwidth_max_trend', 'absolute_velocity_max',
       'absolute_velocity_max_trend', 'vrot']

torp_min_convs = ['cc_min', 'phidp_min', 'reflectivity_min', 'vrot_distance']

torp_predictors = ['torp_prob', 'azshear_max', 'azshear_max_trend', 'divshear_min',
       'divshear_min_trend', 'reflectivity_max', 'spectrumwidth_max',
       'spectrumwidth_max_trend', 'absolute_velocity_max',
       'absolute_velocity_max_trend', 'vrot', 'vrot_distance', 'cc_min',
       'cc_max', 'phidp_min', 'phidp_max', 'reflectivity_min']

torp_all_predictors = ['torp_prob','torp_age',torp_prob_change_1_str,torp_prob_change_2_str,
                       'azshear_max', 'azshear_max_trend', 'divshear_min',
                       'divshear_min_trend', 'reflectivity_max', 'spectrumwidth_max',
                       'spectrumwidth_max_trend', 'absolute_velocity_max',
                       'absolute_velocity_max_trend', 'vrot','cc_min', 'phidp_min',
                       'reflectivity_min', 'vrot_distance']
