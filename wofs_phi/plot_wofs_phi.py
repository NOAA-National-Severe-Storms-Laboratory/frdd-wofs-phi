#======================================
# Will handle the plotting of wofs-phi
#======================================

#_wofs = '/home/monte.flora/python_packages/frdd-wofs-post'
_wofs = '/home/eric.loken/python_packages/frdd-wofs-post'

import sys, os
import numpy as np

#Add path to Monte's package to path 
sys.path.insert(0, _wofs)
from wofs.plotting.wofs_plotter import WoFSPlotter
from wofs.plotting.data_preprocessing import DataPreProcessor
from wofs.plotting.util import check_file_type
from wofs.plotting.wofs_colors import WoFSColors as wc
from datetime import datetime, timedelta
#from . import config as c 
#import config as c 
import config_2025_test as c
import matplotlib
import copy 



def plot_wofs_phi_forecast_mode(nc_fname, png_outdir, wofs_init_dt, \
                ps_init_dt, start_valid_dt, end_valid_dt, time_window,\
                training_types):
    '''Plots the wofs-phi severe hazard probabilities in forecast mode
        @nc_fname: path-like string
            Path to the wofs-phi ncdf file 
        @png_outdir: Path-like string
            Path to where the png file will be saved 
        @wofs_init_dt: datetime object
            Corresponding to the wofs intialization time
        @ps_init_dt: datetime object
            Corresponding to the initialization time of the ProbSevere Object
        @start_valid_dt: datetime object
            Corresponding to the start of the forecast valid period 
        @end_valid_dt: datetime object
            Corresponding to the end o the forecast valid period 
        @time_window : integer 
            Forecast valid period time in minutes 
        @training_types : List of strings
            corresponding to how the training was done: 
            "obs" for only LSRs,
            "warnings" for only warnings
            "obs_and_warnings" for LSRs + warnings
    '''


    #TODO: Might eventually make this a passed in parameter. 
    vars_to_plot = []
    for t in range(len(training_types)):
        training_type = training_types[t] 
        for h in c.final_hazards: 
            for radius in c.final_str_obs_radii:
                vars_to_plot.append('wofsphi__%s__%skm__%smin__%s' \
                        %(h, radius, time_window, training_type))


    #NOTE/TODO: Might need to make this different for tornadoes 
    levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) 
    levels_tornado = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,\
                                0.4, 0.45]) 
    
    #levels_tornado = np.array([0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,\
    #                            0.45, 0.50]) 

    extend_var = "max" 


    # Load the data and close the netCDF file.
    processor = DataPreProcessor(nc_fname)
    data = processor.load_data(vars_to_plot)


    #Define my colormap 
    #My custom cmap 
    #my_cmap = matplotlib.colors.ListedColormap([wc.blue3, wc.blue4, wc.orange3, wc.orange4,\
    #            wc.red3, wc.red4, wc.red5, wc.red6, wc.red7]) 

    #Standard cmap 
    my_cmap = wc.wz_cmap

    alpha = 0.55

    #conf_kwargs_dict = {'levels': levels, 'extend': extend_var, 'cmap': my_cmap, 'alpha': alpha}

    #png_save_name = "wofsphi__hail__30km__60min_{wofs_init_date}_{wofs_init_time}_{ps_init_time}_\
    #                        f{lead_time}.png"


    wofs_init_date_str, wofs_init_time_str = dattime_to_str(wofs_init_dt)
    
    __, ps_init_time_str = dattime_to_str(ps_init_dt) 


    ps_init_time_date_str = ps_init_dt.strftime("%Y-%m-%d")

    #Compute lead time in minutes 
    lead_time_minutes = subtract_dt(end_valid_dt, wofs_init_dt, True )

    for v in data.keys():
        print (v) 

        #initialize the plotter 
        #png_save_name = "%s_%s_%s_%s_f%s.png" %(v, wofs_init_date_str, wofs_init_time_str,\
        #                    ps_init_time_str, str(lead_time_minutes).zfill(3))

        png_save_name = "%s_f%s.png" %(v, str(lead_time_minutes).zfill(3))

        color_cont_data = data[v]

        if ("tornado" in v):
            conf_kwargs_dict = {'levels': levels_tornado, 'extend': extend_var, 'cmap': my_cmap, 'alpha': alpha}
            
            #Mask out low or zero probabilities
            color_cont_data[color_cont_data<levels_tornado[0]] = -1.0


        else: 
            conf_kwargs_dict = {'levels': levels, 'extend': extend_var, 'cmap': my_cmap, 'alpha': alpha}

            #Mask out low or zero probabilities
            color_cont_data[color_cont_data<levels[0]] = -1.0

        plotter = WoFSPlotter(file_path=nc_fname, \
                                outdir = png_outdir,\
                                dt=c.wofs_update_rate,\
                                timestep_index=None,\
                                init_time = start_valid_dt,\
                                valid_time = end_valid_dt)

       
        #Mask out low or zero probabilities
        #color_cont_data = data[v] 
        #color_cont_data[color_cont_data<0.01] = -1.0 
       
     
        
        #if ("tornado" in v):
        #    color_cont_data[color_cont_data < levels_tornado[0]] = -1.0
        #else: 
        #    color_cont_data[color_cont_data < levels[0]] = -1.0
        
        ps_init_time_full_string = "ProbSevere Init: %s, %s UTC" %(ps_init_time_date_str, ps_init_time_str)

        plotter.plot(var_name=v, \
                        color_cont_data=color_cont_data,\
                        add_sharpness=True,\
                        color_contf_kws=conf_kwargs_dict,\
                        save_name=png_save_name, \
                        second_title=ps_init_time_full_string)


    #Close the plotter
    plotter.close_figure() 



def plot_wofs_phi_warning_mode(nc_fname, png_outdir, wofs_init_dt, \
                ps_init_dt, start_valid_dt, end_valid_dt, time_window,\
                training_types):
    '''Plots the wofs-phi severe hazard probabilities in forecast mode
        @nc_fname: path-like string
            Path to the wofs-phi ncdf file 
        @png_outdir: Path-like string
            Path to where the png file will be saved 
        @wofs_init_dt: datetime object
            Corresponding to the wofs intialization time
        @ps_init_dt: datetime object
            Corresponding to the initialization time of the ProbSevere Object
        @start_valid_dt: datetime object
            Corresponding to the start of the forecast valid period 
        @end_valid_dt: datetime object
            Corresponding to the end o the forecast valid period 
        @time_window : integer 
            Forecast valid period time in minutes 
        @training_types : List of strings
            corresponding to how the training was done: 
            "obs" for only LSRs,
            "warnings" for only warnings
            "obs_and_warnings" for LSRs + warnings
    '''

    #TODO: Maybe eventually pass this in ?? 

    #vars_to_plot = ['wofsphi__hail__39km__%smin' %time_window,
    #              'wofsphi__wind__39km__%smin' %time_window,
    #              'wofsphi__tornado__39km__%smin' %time_window]

    vars_to_plot = []
    for t in range(len(training_types)):
        training_type = training_types[t]
        for h in c.final_hazards:
            vars_to_plot.append('wofsphi__%s__39km__%smin__%s' \
                    %(h, time_window, training_type))


    #levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    orig_levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9]) 

    #TODO: Might have to make different contours for tornadoes
    #orig_levels_tornado = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,\
    #                            0.4, 0.45, 0.5]) 

    #orig_levels_tornado = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5]) 
    orig_levels_tornado = np.array([0.1, 0.3, 0.5, 0.7, 0.9]) 

    #levels_tornado = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35,\
    #                    0.40, 0.45, 0.5]) 

    orig_linewidths = np.linspace(0.1, 2.0, len(orig_levels))
    #orig_linewidths_tornado = np.linspace(0.1, 2.0, len(orig_levels_tornado))
    orig_linewidths_tornado = np.linspace(0.5, 2.5, len(orig_levels_tornado))

    colors = ["black" for l in orig_levels]

    extend_var = "max"


    # Load the data and close the netCDF file.
    processor = DataPreProcessor(nc_fname)
    data = processor.load_data(vars_to_plot)


    #Define my colormap 
    #my_cmap = matplotlib.colors.ListedColormap([wc.blue3, wc.blue4, wc.orange3, wc.orange4,\
    #            wc.red3, wc.red4, wc.red5, wc.red6, wc.red7])

    alpha = 0.9

    #cont_kwargs_dict = {'levels': orig_levels, 'extend': extend_var, 'linewidths': orig_linewidths, \
    #                    'linestyles': 'solid', 'colors': colors,\
    #                    'alpha': alpha, 'add_labels': True}

    #png_save_name = "wofsphi__hail__30km__60min_{wofs_init_date}_{wofs_init_time}_{ps_init_time}_\
    #                        f{lead_time}.png"


    #wofs_init_date_str, wofs_init_time_str = dattime_to_str(wofs_init_dt)

    __, ps_init_time_str = dattime_to_str(ps_init_dt)

    start_valid_date_str, start_valid_time_str = dattime_to_str(start_valid_dt) 
    end_valid_date_str, end_valid_time_str = dattime_to_str(end_valid_dt) 

    ps_init_time_date_str = ps_init_dt.strftime("%Y-%m-%d")

    #Compute lead time in minutes 
    lead_time_minutes = subtract_dt(end_valid_dt, wofs_init_dt, True )

    for v in data.keys():
        #initialize the plotter 
        #png_save_name = "%s_%s_%s_%s_f%s.png" %(v, wofs_init_date_str, wofs_init_time_str,\
        #                    ps_init_time_str, str(lead_time_minutes).zfill(3))

        png_save_name = "%s_%s%s.png" %(v, start_valid_date_str, start_valid_time_str)

        color_cont_data = data[v]

        if ("tornado" in v):

            cont_kwargs_dict = {'levels': orig_levels_tornado, 'extend': extend_var, 'linewidths': orig_linewidths_tornado, \
                        'linestyles': 'solid', 'colors': colors,\
                        'alpha': alpha, 'add_labels': True}

            #Mask out low or zero probabilities
            color_cont_data[color_cont_data<orig_levels_tornado[0]] = -1.0 


        else: 
            #Start contouring at 30% instead of 10% if we're trained on LSRs + warnings
            
            #if ("obs_and_warnings" in v):
            levels = orig_levels[1:]
            linewidths = orig_linewidths[1:]
            #else:
                #levels = copy.deepcopy(orig_levels)
                #linewidths = copy.deepcopy(orig_linewidths) 

            cont_kwargs_dict = {'levels': levels, 'extend': extend_var, 'linewidths': linewidths, \
                        'linestyles': 'solid', 'colors': colors,\
                        'alpha': alpha, 'add_labels': True}

            #Mask out low probabilities 
            color_cont_data[color_cont_data < levels[0]] = -1.0

        ps_init_time_full_string = "ProbSevere Init: %s, %s UTC, Valid: %s - %s UTC" \
                %(ps_init_time_date_str, ps_init_time_str, start_valid_time_str, \
                    end_valid_time_str)

        plotter = WoFSPlotter(file_path=nc_fname, \
                                outdir = png_outdir,\
                                dt=c.wofs_update_rate,\
                                timestep_index=None,\
                                init_time = start_valid_dt,\
                                valid_time = end_valid_dt)

        #Mask out low or zero probabilities
        #color_cont_data[color_cont_data<0.01] = -1.0 
        #color_cont_data[color_cont_data < levels[0]] = -1.0

        plotter.plot(var_name = v,\
                        line_cont_data=color_cont_data,\
                        add_sharpness=False,\
                        line_cont_kws=cont_kwargs_dict,\
                        color_cont_data=None,\
                        add_text=False,\
                        save_name=png_save_name,
                        second_title=ps_init_time_full_string)



    #Close the plotter
    plotter.close_figure()


def dattime_to_str(in_dt):
        '''Converts incoming datetime object (@in_dt) to 8 character date string
            and 4-character time string
            @Returns 8-character date string (YYYYMMDD) and 4-character time 
                string (HHMM) 
        '''

        new_date_string = in_dt.strftime("%Y%m%d")
        new_time_string = in_dt.strftime("%H%M")


        return new_date_string, new_time_string



def timedelta_to_min(in_dt):
    '''Converts the incoming timedelta object (@in_dt) to minutes.'''


    minutes = int(in_dt.total_seconds()/60)

    return minutes


def subtract_dt(dt1, dt2, inMinutes):
    ''' Takes dt1 - dt2 and returns the difference (in datetime format)
            @dt1 and @dt2 are both datetime objects 
            @inMinutes is boolean. If True, returns the subtraction in minutes, 
                if false, returns a timedelta object 
    '''

    difference = dt1 - dt2

    if (inMinutes == True):
        difference = timedelta_to_min(difference)

    return difference
