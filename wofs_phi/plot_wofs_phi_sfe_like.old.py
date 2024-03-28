#=================================================
# Plots the WoFS PHI ML products  
# Author: Montgomery Flora 
# Email : monte.flora@noaa.gov
#=================================================

# Internal modules

import sys, os
import numpy as np
sys.path.insert(0, '/home/monte.flora/python_packages/WoF_post')
sys.path.insert(0, '/work/eric.loken/wofs/realtime_like/test_realtime/phi_tool') 
from wofs.plotting.wofs_plotter import WoFSPlotter
from wofs.plotting.data_preprocessing import DataPreProcessor
from wofs.plotting.util import check_file_type
import wofs_phi_cmaps
#from wofs_phi_cmaps import double_cmap
from datetime import datetime, timedelta


##
#from ..wofs_plotter import WoFSPlotter
#from ..data_preprocessing import DataPreProcessor
#from ..util import check_file_type

#plot_wofs_phi(wofs_phi_file, final_png_dir, new_init_dt, new_valid_dt, phi_timestep, \
#                            time_window, timestep_index=ts, dt=DT)
#def plot_wofs_phi(file, outdir, init_dt, valid_dt, ps_init, wofs_init, timeWindow, timestep_index=None, dt=None,): 

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
    vars_to_plot=['wofsphi__hail__30km__%smin' %timeWindow, 
                  'wofsphi__wind__30km__%smin' %timeWindow,
                  'wofsphi__tornado__30km__%smin' %timeWindow,
                  'wofsphi__hail__39km__%smin' %timeWindow,
                  'wofsphi__wind__39km__%smin' %timeWindow,
                  'wofsphi__tornado__39km__%smin' %timeWindow,
                  'wofsphi__hail__15km__%smin' %timeWindow,
                  'wofsphi__wind__15km__%smin' %timeWindow,
                  'wofsphi__tornado__15km__%smin' %timeWindow,
                  'wofsphi__hail__7_5km__%smin' %timeWindow,
                  'wofsphi__wind__7_5km__%smin' %timeWindow,
                  'wofsphi__tornado__7_5km__%smin' %timeWindow,
                 ]
   
    #vars_to_plot=['wofsphi__hail__30km__%smin' %timeWindow, 
    #              'wofsphi__wind__30km__%smin' %timeWindow,
    #              'wofsphi__tornado__30km__%smin' %timeWindow,
    #              'wofsphi__hail__15km__%smin' %timeWindow,
    #              'wofsphi__wind__15km__%smin' %timeWindow,
    #              'wofsphi__tornado__15km__%smin' %timeWindow,
    #             ]

    #Plotting kwargs 
    #levels = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,\
    #          0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50] 
    #levels = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, \
    #            0.40, 0.45, 0.50] 

    #levels = np.linspace(0,0.50,11) 
    #prob_thresh = 0.50 #Probability at which to transition to other cmap
    #levels = np.linspace(0,1,21) 
    #levels = np.linspace(0,0.6, 13) 
    levels = np.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    extend_var = "max"  
    cmap_levels1 = np.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    cmap_levels2 = np.array([ 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])

    #n_cbar_segments = 1001 #initial resolution of cbar (for making new cbars) 
    #percent_trunc = 0.4 #percent of original colorbar to truncate. 

    #hail_cmap = wofs_phi_cmaps.inverted_severe_cmap_w_grays('Greens_r', 'Greys',\
    #                prob_thresh, levels) 
    #wind_cmap = wofs_phi_cmaps.inverted_severe_cmap_w_grays('Blues_r', 'Greys',\
    #                prob_thresh, levels) 
    #torn_cmap = wofs_phi_cmaps.inverted_severe_cmap_w_grays('Reds_r', 'Greys',\
    #                prob_thresh, levels) 

    #hail_cmap = wofs_phi_cmaps.darker_non_inverted('Greens', percent_trunc, levels) 
    #wind_cmap = wofs_phi_cmaps.darker_non_inverted('Blues', percent_trunc, levels) 
    #torn_cmap = wofs_phi_cmaps.darker_non_inverted('Reds', percent_trunc, levels) 

    hail_cmap = wofs_phi_cmaps.doubleCmap('Greens', 'Greys_r', 0.25, 0.1, 0.25, cmap_levels1, cmap_levels2) 
    wind_cmap = wofs_phi_cmaps.doubleCmap('Blues', 'Purples_r', 0.25, 0.0, 0.40, cmap_levels1, cmap_levels2) 
    torn_cmap = wofs_phi_cmaps.doubleCmap('Reds', 'Oranges_r', 0.18, 0.1, 0.18, cmap_levels1, cmap_levels2) 


 
    #cmaps = ['Greens_r', 'Blues_r', 'Reds_r', 'Greens_r', 'Blues_r', 'Reds_r',\
    #         'Greens_r', 'Blues_r', 'Reds_r', 'Greens_r', 'Blues_r', 'Reds_r']
    cmaps = [hail_cmap, wind_cmap, torn_cmap, hail_cmap, wind_cmap, torn_cmap, \
            hail_cmap, wind_cmap, torn_cmap, hail_cmap, wind_cmap, torn_cmap] 
    alpha = 0.3

    # Load the data and close the netCDF file.
    processor = DataPreProcessor(file) 
    data = processor.load_data(vars_to_plot)

    #valid_time_string = datetime.strptime(valid_dt, "%Y%m%d%H%M")
    valid_time_string = datetime.strftime(valid_dt, "%Y%m%d%H%M") 
    
    #outtime = f"f{dt*timestep_index:03d}"
        #outtime = f"f{dt*timestep_index:03d}"
    #outtime = f{"valid_time_string"}
    #pstime = f"{phi_init}"
   
    var_index = 0  
    for v in data.keys(): 
        #initialize the plotter
        plotter = WoFSPlotter(file_path=file,
                          #gis_kwargs = {'add_states' : True},
                          outdir=outdir,
                          dt=dt,
                          timestep_index=timestep_index,
                          init_time = init_dt,
                          valid_time = valid_dt,
                         )

        cmap = cmaps[var_index] 
    
        # Mask out low or zero probabilities to highlight the tracks.
        color_cont_data = data[v]
        color_cont_data[color_cont_data<0.01] = -1.0 
        
        conf_kwargs_dict = {'levels': levels, 'extend': extend_var, 'cmap': cmap, 'alpha': alpha} 

        plotter.plot(var_name=v, 
                    color_cont_data=color_cont_data, 
                    add_sharpness=True,
                    color_contf_kws=conf_kwargs_dict,
                    save_name = f'{v}_{valid_time_string}.png'
                    )

        var_index += 1
        
    # Close the figure.
    plotter.close_figure()
