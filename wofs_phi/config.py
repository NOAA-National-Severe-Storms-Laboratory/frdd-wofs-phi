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


max_ps_extrap_time = 181.0 #Maximum amount of PS extrapolation time (used for setting min and max radius) 

#radius (in km) for probSevere objects at time 0
min_radius = 1.5 #in km (for probSevere objects)

#radius (in km) for probSevere objects at the maximum extrapolation time 
#generally taken to be 181 minutes. 
#max_radius = 20.0 #in km (for probSevere objects) #Used to be 30.0, but that was much too big
max_radius = 1.5 

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


 
