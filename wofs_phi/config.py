#==================================
# This is the configuration file
# for wofs_phi.py. It will hold 
# all of the constants necessary to
# run wofs_phi.py 
#==================================


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

train_mode = 'validate'
train_type = 'obs'
train_radii = ['7.5', '15', '30', '39']
train_hazards = ['tornado']
train_lead_times = [30]
forecast_length = 60
num_folds = 5
wofs_spinup_time = 25

plot_in_training = True

num_training_vars = 269

model_save_dir = '/work/ryan.martz/wofs_phi_data/%s_train/models/wofs_psv2_no_torp' %(train_type)
validation_dir = '/work/ryan.martz/wofs_phi_data/%s_train/validation_fcsts/wofs_psv2_no_torp' %(train_type)
test_dir = '/work/ryan.martz/wofs_phi_data/%s_train/test_fcsts/wofs_psv2_no_torp' %(train_type)
warning_dir = '' #will fill this in when we get around to re making new warnings
torp_vars_filename = '/work/ryan.martz/wofs_phi_data/%s_train/training_data/predictors/torp_predictors.txt' %(train_type)


#Path to the trained rfs 
#rf_dir = "/work/ryan.martz/wofs_phi_data/models/wofs_psv2_no_torp/hail/wofslag_25/length_60"
rf_dir = "/work/eric.loken/wofs/2024_update/SFE2024/rf_models"
sr_dir = ""

#Path where to save the ncdf files 
#ncdf_save_dir = "/work/eric.loken/wofs/2024_update/SFE2024/ncdf_files"
ncdf_save_dir = "/home/eric.loken/python_packages/frdd-wofs-phi/wofs_phi/ncdf"

#TODO: Path indicating where to save the png files to
#May implement this later; for now, let's just save to the same location as the 
#netcdf files 
png_outdir = ""

#May or may not eventually use these
#generate_forecasts = True #Generates the predictors array if True
generate_forecasts = True
generate_reports = False #Generates the reports file if True 
save_npy = False #Tells whether or not to save the npy predictor files 
save_ncdf = True #Tells whether or not to create/save the ncdf (realtime) files
plot_forecasts = True #Tells whether or not to create the .png files for wofs viewer

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

#Path to the reports coordinates directory -- i.e., where are the coords.txt 
#files that need to be read in during training? 
reps_coords_dir = "/work/eric.loken/wofs/new_torn/storm_events_reports/fromThea"


#If True, use the ALL naming convention (will be true on cloud) 
#If False, use the legacy naming convention (e.g., ENS, ENV, SVR, etc.) 
use_ALL_files = False

wofs_base_path = "/work/mflora/SummaryFiles" #Obviously, will need to change on cloud 

nc_outdir = "." #Where to place the final netcdf files 

max_cores = 30 #max number of cores to use for parallelization

ps_version = 2 #i.e., which probSevere version is being used 2 for v2, 3 for v3

dx_km = 3.0 #horizontal grid spacing of wofs in km 
ps_thresh = 0.01 #ps objects must have probs greater than or equal to this amount to be considered

max_ps_extrap_time = 181.0 #Maximum amount of PS extrapolation time (used for setting min and max radius) 

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


obs_radii = ["39", "30", "15", "7.5"]

#obs_radii_str = ["30.0", "15.0", "7.5", "39.0"]
obs_radii_str = ["39.0", "30.0", "15.0", "7.5"] 
obs_radii_float = [39, 30, 15, 7.5]
#final_str_obs_radii = ["30", "15", "7_5", "39"] #form to use for final ncdf files
final_str_obs_radii = ["39", "30","15", "7_5"] #form to use for final ncdf files

final_hazards = ["hail", "wind", "tornado"] #for naming in final ncdf file 

wofs_fields_file = "standard_wofs_variables.txt"
wofs_methods_file = "standard_wofs_methods.txt"

all_fields_file = "all_fields.txt" #Holds all the predictor fields
all_methods_file = "all_methods.txt" #Holds all the preprocessing methods

wofs_dir = "/work/mflora/SummaryFiles/"
ps_dir = "/work/eric.loken/wofs/probSevere/"

ps_search_minutes = 180 #how long before start time do we need to search for ps files to generate predictors
ps_recent_file_threshold = 10 #need a file in last __ minutes to do training/real time running

#Holds all the variables that just get taken from the point of prediction
single_pt_file = "single_point_fields.txt" 


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
torp_prob_change_1_str = 'p_change_' + str(torp_prob_change_1) + '_min'
torp_prob_change_2_str = 'p_change_' + str(torp_prob_change_2) + '_min'

torp_search_minutes = max(torp_prob_change_1, torp_prob_change_2)

torp_max_time_skip = 10

torp_conv_dists = [15, 30, 45, 60]

torp_max_convs = ["prob","age",torp_prob_change_1_str,torp_prob_change_2_str,"AzShear_max","AzShear_min","AzShear_25th_percentile","AzShear_median","DivShear_max","DivShear_min","DivShear_median","DivShear_75th_percentile","PhiDP_AzGradient_median","PhiDP_DivGradient_min","PhiDP_DivGradient_25th_percentile","PhiDP_DivGradient_median","PhiDP_DivGradient_75th_percentile","PhiDP_Gradient_max","PhiDP_Gradient_min","PhiDP_Gradient_median","PhiDP_MedianFiltered_max","PhiDP_MedianFiltered_min","Reflectivity_MedianFiltered_max","Reflectivity_MedianFiltered_min","SpectrumWidth_MedianFiltered_max","SpectrumWidth_MedianFiltered_min","Zdr_MedianFiltered_max","Zdr_MedianFiltered_min","Zdr_MedianFiltered_median"]

torp_abs_convs = ["Reflectivity_AzGradient_max","Reflectivity_AzGradient_min","Reflectivity_AzGradient_median","Reflectivity_DivGradient_min","Reflectivity_DivGradient_median","Reflectivity_Gradient_max","Reflectivity_Gradient_min","RhoHV_AzGradient_25th_percentile","RhoHV_AzGradient_median","RhoHV_AzGradient_75th_percentile","RhoHV_DivGradient_median","RhoHV_Gradient_max","RhoHV_Gradient_min","SpectrumWidth_AzGradient_min","SpectrumWidth_AzGradient_25th_percentile","SpectrumWidth_AzGradient_median","SpectrumWidth_AzGradient_75th_percentile","SpectrumWidth_DivGradient_min","SpectrumWidth_DivGradient_25th_percentile","SpectrumWidth_DivGradient_median","SpectrumWidth_DivGradient_75th_percentile","SpectrumWidth_Gradient_min","Velocity_Gradient_min","Velocity_Gradient_25th_percentile","Velocity_MedianFiltered_absmax","Velocity_MedianFiltered_absmin","Velocity_MedianFiltered_median","Zdr_AzGradient_median","Zdr_DivGradient_min","Zdr_DivGradient_25th_percentile","Zdr_DivGradient_median","Zdr_DivGradient_75th_percentile","Zdr_Gradient_min"]

torp_min_convs = ["RhoHV_MedianFiltered_max","RhoHV_MedianFiltered_min","RhoHV_MedianFiltered_median"]

torp_no_conv = ["RangeInterval"]

torp_predictors = ["RangeInterval","AzShear_max","AzShear_min","AzShear_25th_percentile","AzShear_median","DivShear_max","DivShear_min","DivShear_median","DivShear_75th_percentile","PhiDP_AzGradient_median","PhiDP_DivGradient_min","PhiDP_DivGradient_25th_percentile","PhiDP_DivGradient_median","PhiDP_DivGradient_75th_percentile","PhiDP_Gradient_max","PhiDP_Gradient_min","PhiDP_Gradient_median","PhiDP_MedianFiltered_max","PhiDP_MedianFiltered_min","Reflectivity_AzGradient_max","Reflectivity_AzGradient_min","Reflectivity_AzGradient_median","Reflectivity_DivGradient_min","Reflectivity_DivGradient_median","Reflectivity_Gradient_max","Reflectivity_Gradient_min","Reflectivity_MedianFiltered_max","Reflectivity_MedianFiltered_min","RhoHV_AzGradient_25th_percentile","RhoHV_AzGradient_median","RhoHV_AzGradient_75th_percentile","RhoHV_DivGradient_median","RhoHV_Gradient_max","RhoHV_Gradient_min","RhoHV_MedianFiltered_max","RhoHV_MedianFiltered_min","RhoHV_MedianFiltered_median","SpectrumWidth_AzGradient_min","SpectrumWidth_AzGradient_25th_percentile","SpectrumWidth_AzGradient_median","SpectrumWidth_AzGradient_75th_percentile","SpectrumWidth_DivGradient_min","SpectrumWidth_DivGradient_25th_percentile","SpectrumWidth_DivGradient_median","SpectrumWidth_DivGradient_75th_percentile","SpectrumWidth_Gradient_min","SpectrumWidth_MedianFiltered_max","SpectrumWidth_MedianFiltered_min","Velocity_Gradient_min","Velocity_Gradient_25th_percentile","Velocity_MedianFiltered_absmax","Velocity_MedianFiltered_absmin","Velocity_MedianFiltered_median","Zdr_AzGradient_median","Zdr_DivGradient_min","Zdr_DivGradient_25th_percentile","Zdr_DivGradient_median","Zdr_DivGradient_75th_percentile","Zdr_Gradient_min","Zdr_MedianFiltered_max","Zdr_MedianFiltered_min","Zdr_MedianFiltered_median"]

torp_all_predictors = ["age","prob",torp_prob_change_1_str,torp_prob_change_2_str,"RangeInterval","AzShear_max","AzShear_min","AzShear_25th_percentile","AzShear_median","DivShear_max","DivShear_min","DivShear_median","DivShear_75th_percentile","PhiDP_AzGradient_median","PhiDP_DivGradient_min","PhiDP_DivGradient_25th_percentile","PhiDP_DivGradient_median","PhiDP_DivGradient_75th_percentile","PhiDP_Gradient_max","PhiDP_Gradient_min","PhiDP_Gradient_median","PhiDP_MedianFiltered_max","PhiDP_MedianFiltered_min","Reflectivity_AzGradient_max","Reflectivity_AzGradient_min","Reflectivity_AzGradient_median","Reflectivity_DivGradient_min","Reflectivity_DivGradient_median","Reflectivity_Gradient_max","Reflectivity_Gradient_min","Reflectivity_MedianFiltered_max","Reflectivity_MedianFiltered_min","RhoHV_AzGradient_25th_percentile","RhoHV_AzGradient_median","RhoHV_AzGradient_75th_percentile","RhoHV_DivGradient_median","RhoHV_Gradient_max","RhoHV_Gradient_min","RhoHV_MedianFiltered_max","RhoHV_MedianFiltered_min","RhoHV_MedianFiltered_median","SpectrumWidth_AzGradient_min","SpectrumWidth_AzGradient_25th_percentile","SpectrumWidth_AzGradient_median","SpectrumWidth_AzGradient_75th_percentile","SpectrumWidth_DivGradient_min","SpectrumWidth_DivGradient_25th_percentile","SpectrumWidth_DivGradient_median","SpectrumWidth_DivGradient_75th_percentile","SpectrumWidth_Gradient_min","SpectrumWidth_MedianFiltered_max","SpectrumWidth_MedianFiltered_min","Velocity_Gradient_min","Velocity_Gradient_25th_percentile","Velocity_MedianFiltered_absmax","Velocity_MedianFiltered_absmin","Velocity_MedianFiltered_median","Zdr_AzGradient_median","Zdr_DivGradient_min","Zdr_DivGradient_25th_percentile","Zdr_DivGradient_median","Zdr_DivGradient_75th_percentile","Zdr_Gradient_min","Zdr_MedianFiltered_max","Zdr_MedianFiltered_min","Zdr_MedianFiltered_median"]
