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
#"phi_tool" for forecast mode in PHI Tool. 
mode = "forecast" 

is_train_mode = False
is_on_cloud = False 
max_cores = 30 #max number of cores to use for parallelization

ps_version = 2 #i.e., which probSevere version is being used 2 for v2, 3 for v3

dx_km = 3.0 #horizontal grid spacing of wofs in km 
ps_thresh = 0.01 #ps objects must have probs greater than or equal to this amount to be considered

min_radius = 1.5 #in km (for probSevere objects) 
max_radius = 1.5 #in km (for probSevere objects) #Used to be 30.0, but that was much too big
conv_type = "square" #"square" or "circle" -- tells how to do the convolutions 
predictor_radii_km = [0.0, 15.0, 30.0, 45.0, 60.0] #how far to "look" spatially in km for predictors
obs_radii = ["30.0", "15.0", "7.5", "39.0"]
final_str_obs_radii = ["30", "15", "7_5", "39"] #form to use for final ncdf files
final_hazards = ["hail", "wind", "tornado"] #for naming in final ncdf file 
bottom_hour_inits = ["1730", "1830", "1930", "2030", "2130", "2230", "2330", "0030", "0130",\
                     "0230", "0330", "0430", "0530", "0630", "0730", "0830", "0930", "1030",\
                     "1130", "1230", "1330", "1430", "1530", "1630"]

next_day_inits = ["0000", "0030", "0100", "0130", "0200", "0230", "0300", "0330", "0400", "0430", "0500"]
pkl_dir = "." #Will probably need to update later

torp_point_buffer = 7.5
torp_prob_change_1 = 5
torp_prob_change_2 = 10

torp_predictors = ["RangeInterval","AzShear_max","AzShear_min","AzShear_25th_percentile","AzShear_median","DivShear_max","DivShear_min","DivShear_median","DivShear_75th_percentile","PhiDP_AzGradient_median","PhiDP_DivGradient_min","PhiDP_DivGradient_25th_percentile","PhiDP_DivGradient_median","PhiDP_DivGradient_75th_percentile","PhiDP_Gradient_max","PhiDP_Gradient_min","PhiDP_Gradient_median","PhiDP_MedianFiltered_max","PhiDP_MedianFiltered_min","Reflectivity_AzGradient_max","Reflectivity_AzGradient_min","Reflectivity_AzGradient_median","Reflectivity_DivGradient_min","Reflectivity_DivGradient_median","Reflectivity_Gradient_max","Reflectivity_Gradient_min","Reflectivity_MedianFiltered_max","Reflectivity_MedianFiltered_min","RhoHV_AzGradient_25th_percentile","RhoHV_AzGradient_median","RhoHV_AzGradient_75th_percentile","RhoHV_DivGradient_median","RhoHV_Gradient_max","RhoHV_Gradient_min","RhoHV_MedianFiltered_max","RhoHV_MedianFiltered_min","RhoHV_MedianFiltered_median","SpectrumWidth_AzGradient_min","SpectrumWidth_AzGradient_25th_percentile","SpectrumWidth_AzGradient_median","SpectrumWidth_AzGradient_75th_percentile","SpectrumWidth_DivGradient_min","SpectrumWidth_DivGradient_25th_percentile","SpectrumWidth_DivGradient_median","SpectrumWidth_DivGradient_75th_percentile","SpectrumWidth_Gradient_min","SpectrumWidth_MedianFiltered_max","SpectrumWidth_MedianFiltered_min","Velocity_Gradient_min","Velocity_Gradient_25th_percentile","Velocity_MedianFiltered_absmax","Velocity_MedianFiltered_absmin","Velocity_MedianFiltered_median","Zdr_AzGradient_median","Zdr_DivGradient_min","Zdr_DivGradient_25th_percentile","Zdr_DivGradient_median","Zdr_DivGradient_75th_percentile","Zdr_Gradient_min","Zdr_MedianFiltered_max","Zdr_MedianFiltered_min","Zdr_MedianFiltered_median"]
