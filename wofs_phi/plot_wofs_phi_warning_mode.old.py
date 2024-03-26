#=================================================
# Plots the WoFS PHI ML products  
# Author: Montgomery Flora 
# Email : monte.flora@noaa.gov
#=================================================

# Internal modules

import sys, os
import numpy as np
sys.path.insert(0, '/home/monte.flora/python_packages/WoF_post')
#sys.path.insert(0, '/work/eric.loken/wofs/realtime_like/test_realtime/phi_tool')

#from wofs.plotting.wofs_plotter import WoFSPlotter
from wofs_plotter_copy import WoFSPlotter
from wofs.plotting.data_preprocessing import DataPreProcessor
from wofs.plotting.util import check_file_type
#import wofs_phi_cmaps
from datetime import datetime, timedelta

##
#from ..wofs_plotter import WoFSPlotter
#from ..data_preprocessing import DataPreProcessor
#from ..util import check_file_type

#plot_wofs_phi(wofs_phi_file, final_png_dir, new_init_dt, new_valid_dt, phi_timestep, \
#                            time_window, timestep_index=ts, dt=DT)
#def plot_wofs_phi(file, outdir, init_dt, valid_dt, ps_init, wofs_init, timeWindow, timestep_index=None, dt=None,): 

#plot_wofs_phi(wofs_phi_file, final_png_dir, new_init_dt, new_valid_dt, \
#                                time_window, timestep_index=ts, dt=DT)

def plot_wofs_phi(file, outdir, init_dt, valid_dt, timeWindow, timestep_index=None, dt=None,): 
    """
    Plots the ensemble mean satellite fields. 

    Parameters
    -------------
    file : path-like string
        Path to the PHI file. 

    outdir   : path-like string
        Path to where the PNG file be will be saved. 
        
    init_dt : datetime object
        The initial date time of the 30min, 60min, or 180min 
        forecast period
    
    valid_dt : datetime object
        The end date time of the 30min, 60min, or 180 min 
        forecast period
    
    ps_init : str
        The PHI initialization time 

    wofs_init : str
        The WoFS initialization time 

    timeWindow : int 
        The length of the forecast valid period, in minutes
        
        
    dt : integer (default=None)
        Time between forecast time steps (in minutes).
        Used for naming the resulting .png file.
    
    timestep_index : int (default=None)
        The forecast time step index. 
        Used for naming the resulting .png file.
        If None, then WoFSPlotter attempts to 
        determine the index from the summary file name.
        
    """
    #vars_to_plot=['wofsphi__hail__30km__%smin' %timeWindow, 
    #              'wofsphi__wind__30km__%smin' %timeWindow,
    #              'wofsphi__tornado__30km__%smin' %timeWindow,
    #              'wofsphi__hail__39km__%smin' %timeWindow,
    #              'wofsphi__wind__39km__%smin' %timeWindow,
    #              'wofsphi__tornado__39km__%smin' %timeWindow,
    #              'wofsphi__hail__15km__%smin' %timeWindow,
    #              'wofsphi__wind__15km__%smin' %timeWindow,
    #              'wofsphi__tornado__15km__%smin' %timeWindow,
    #              'wofsphi__hail__7_5km__%smin' %timeWindow,
    #              'wofsphi__wind__7_5km__%smin' %timeWindow,
    #              'wofsphi__tornado__7_5km__%smin' %timeWindow,
    #             ]
   
    vars_to_plot=['wofsphi__hail__30km__%smin' %timeWindow, 
                  'wofsphi__wind__30km__%smin' %timeWindow,
                  'wofsphi__tornado__30km__%smin' %timeWindow,
                  'wofsphi__hail__15km__%smin' %timeWindow,
                  'wofsphi__wind__15km__%smin' %timeWindow,
                  'wofsphi__tornado__15km__%smin' %timeWindow,
                 ]

    #Plotting kwargs 
    #levels = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,\
    #          0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50] 
    #levels = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, \
    #            0.40, 0.45, 0.50] 

    #levels = np.linspace(0,0.50,11) 
    #prob_thresh = 0.50 #Probability at which to transition to other cmap
    orig_levels = np.linspace(0,1,21) 
    orig_levels[0] = 0.01 #Start contouring at 1% instead of 0%

    extend_var = "max"  
    #percent_trunc = 0.4 #percent of original colorbar to truncate. 

    #OLD: 
    #hail_cmap = wofs_phi_cmaps.darker_non_inverted('Greens', percent_trunc, levels) 
    #wind_cmap = wofs_phi_cmaps.darker_non_inverted('Blues', percent_trunc, levels) 
    #torn_cmap = wofs_phi_cmaps.darker_non_inverted('Reds', percent_trunc, levels) 

 
    #cmaps = ['Greens_r', 'Blues_r', 'Reds_r', 'Greens_r', 'Blues_r', 'Reds_r',\
    #         'Greens_r', 'Blues_r', 'Reds_r', 'Greens_r', 'Blues_r', 'Reds_r']
    #cmaps = [hail_cmap, wind_cmap, torn_cmap, hail_cmap, wind_cmap, torn_cmap, \
    #        hail_cmap, wind_cmap, torn_cmap, hail_cmap, wind_cmap, torn_cmap] 
    #alpha = 0.3

    #Make it so alpha and linewidth values increase with increasing probabilities
    #linewidths = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,\
    #                  2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    #linewidths = np.array([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0,\
    #                        2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) 

    orig_linewidths = np.linspace(0.1, 2.0, 21) 

    #linewidths = np.array([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, \
    #                        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]) 



    #alphas = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,\
    #                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    alpha = 1.0 


    # Load the data and close the netCDF file.
    processor = DataPreProcessor(file) 
    data = processor.load_data(vars_to_plot)

    

    #valid_time_string = datetime.strptime(valid_dt, "%Y%m%d%H%M")
    #valid_time_string = datetime.strftime(valid_dt, "%Y%m%d%H%M") 
    init_time_string = datetime.strftime(init_dt, "%Y%m%d%H%M")
    
    #outtime = f"f{dt*timestep_index:03d}"
    #outtime = f"f{dt*timestep_index:03d}"
    #outtime = f{"valid_time_string"}
    #pstime = f"{phi_init}"
   
    var_index = 0  
    for v in data.keys(): 
        print (v) 
        #initialize the plotter
        plotter = WoFSPlotter(file_path=file,
                          #gis_kwargs = {'add_states' : True},
                          outdir=outdir,
                          dt=dt,
                          timestep_index=timestep_index,
                          init_time = init_dt,
                          valid_time = valid_dt,
                         )

        #cmap = cmaps[var_index] 
    
        # Mask out low or zero probabilities to highlight the tracks.
        color_cont_data = data[v]
        color_cont_data[color_cont_data<0.01] = -1.0 

        #Maybe the levels should vary based on the max contour? 

        max_data = max(color_cont_data.flatten())

        #if (max_data > 0.40):
        if (max_data > 0.10): 
            levels = np.array([0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]) 
            linewidths = orig_linewidths[0::2]
        else: 
            levels = orig_levels
            linewidths = orig_linewidths

        ##Update the levels 
        #levels[0] = 0.01

        #Make it so alpha and linewidth values increase with increasing probabilities
        #alphas = np.linspace(0.3, 1.0, len(levels))
        #linewidths = 
        
        #cont_kwargs_dict = {'levels': levels, 'extend': extend_var, 'linewidths': 2, 'linestyles': 'solid', 'colors': 'black',\
        #                    'add_labels': True}
        
        cont_kwargs_dict = {'levels': levels, 'extend': extend_var, 'linewidths': linewidths, 'linestyles': 'solid', 'colors': 'black',\
                            'alpha': alpha, 'add_labels': True}
        
        
        #TODO: Label the contours. 
        #clabels = ["0%","5%", "10%", "15%", "20%", "25%", "30%", "35%", "40%", "45%", "50%", "55%", "60%","65%",\
        #            "70%", "75%", "80%", "85%", "90%", "95%", "100%"] 

        plotter.plot(var_name=v, 
                    line_cont_data=color_cont_data, 
                    add_sharpness=False,
                    #color_contf_kws=conf_kwargs_dict,
                    line_cont_kws=cont_kwargs_dict,
                    save_name = f'{v}_{init_time_string}.png'
                    )

        #TODO: We'd like to have the contours labeled. Not sure if this will work. 
        



        #Orig: 
        #plotter.plot(var_name=v, 
        #            color_cont_data=color_cont_data, 
        #            add_sharpness=False,
        #            color_contf_kws=conf_kwargs_dict,
        #            save_name = f'{v}_{init_time_string}.png'
        #            )

        var_index += 1
        
    # Close the figure.
    plotter.close_figure()
