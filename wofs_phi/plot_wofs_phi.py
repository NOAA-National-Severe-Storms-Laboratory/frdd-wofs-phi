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
import config as c 
import matplotlib



def plot_wofs_phi_forecast_mode(nc_fname, png_outdir, wofs_init_dt, \
                ps_init_dt, start_valid_dt, end_valid_dt, time_window):
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
    '''




    #TODO: Maybe eventually pass this in ?? 

    vars_to_plot = ['wofsphi__hail__39km__%smin' %time_window,
                  'wofsphi__wind__39km__%smin' %time_window,
                  'wofsphi__tornado__39km__%smin' %time_window]


    levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) 

    extend_var = "max" 


    # Load the data and close the netCDF file.
    processor = DataPreProcessor(nc_fname)
    data = processor.load_data(vars_to_plot)


    #Define my colormap 
    #My custom cmap 
    my_cmap = matplotlib.colors.ListedColormap([wc.blue3, wc.blue4, wc.orange3, wc.orange4,\
                wc.red3, wc.red4, wc.red5, wc.red6, wc.red7]) 

    #Standard cmap 
    my_cmap = wc.wz_cmap

    alpha = 0.55

    conf_kwargs_dict = {'levels': levels, 'extend': extend_var, 'cmap': my_cmap, 'alpha': alpha}

    #png_save_name = "wofsphi__hail__30km__60min_{wofs_init_date}_{wofs_init_time}_{ps_init_time}_\
    #                        f{lead_time}.png"


    wofs_init_date_str, wofs_init_time_str = dattime_to_str(wofs_init_dt)
    
    __, ps_init_time_str = dattime_to_str(ps_init_dt) 

    #Compute lead time in minutes 
    lead_time_minutes = subtract_dt(end_valid_dt, wofs_init_dt, True )

    for v in data.keys():
        print (v) 

        #initialize the plotter 
        png_save_name = "%s_%s_%s_%s_f%s.png" %(v, wofs_init_date_str, wofs_init_time_str,\
                            ps_init_time_str, str(lead_time_minutes).zfill(3))

        plotter = WoFSPlotter(file_path=nc_fname, \
                                outdir = png_outdir,\
                                dt=c.wofs_update_rate,\
                                timestep_index=None,\
                                init_time = start_valid_dt,\
                                valid_time = end_valid_dt)


        #Mask out low or zero probabilities
        color_cont_data = data[v] 
        #color_cont_data[color_cont_data<0.01] = -1.0 
        color_cont_data[color_cont_data < levels[0]] = -1.0
        
        plotter.plot(var_name=v, \
                        color_cont_data=color_cont_data,\
                        add_sharpness=True,\
                        color_contf_kws=conf_kwargs_dict,\
                        save_name=png_save_name)


    #Close the plotter
    plotter.close_figure() 



def plot_wofs_phi_warning_mode(nc_fname, png_outdir, wofs_init_dt, \
                ps_init_dt, start_valid_dt, end_valid_dt, time_window):
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
    '''

    #TODO: Maybe eventually pass this in ?? 

    vars_to_plot = ['wofsphi__hail__39km__%smin' %time_window,
                  'wofsphi__wind__39km__%smin' %time_window,
                  'wofsphi__tornado__39km__%smin' %time_window]


    #levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9]) 

    linewidths = np.linspace(0.1, 2.0, 9)

    colors = ["black" for l in levels]

    extend_var = "max"


    # Load the data and close the netCDF file.
    processor = DataPreProcessor(nc_fname)
    data = processor.load_data(vars_to_plot)


    #Define my colormap 
    #my_cmap = matplotlib.colors.ListedColormap([wc.blue3, wc.blue4, wc.orange3, wc.orange4,\
    #            wc.red3, wc.red4, wc.red5, wc.red6, wc.red7])

    alpha = 0.9

    cont_kwargs_dict = {'levels': levels, 'extend': extend_var, 'linewidths': linewidths, \
                        'linestyles': 'solid', 'colors': colors,\
                        'alpha': alpha, 'add_labels': True}

    #png_save_name = "wofsphi__hail__30km__60min_{wofs_init_date}_{wofs_init_time}_{ps_init_time}_\
    #                        f{lead_time}.png"


    #wofs_init_date_str, wofs_init_time_str = dattime_to_str(wofs_init_dt)

    #__, ps_init_time_str = dattime_to_str(ps_init_dt)

    start_valid_date_str, start_valid_time_str = dattime_to_str(start_valid_dt) 

    #Compute lead time in minutes 
    lead_time_minutes = subtract_dt(end_valid_dt, wofs_init_dt, True )

    for v in data.keys():
        #initialize the plotter 
        #png_save_name = "%s_%s_%s_%s_f%s.png" %(v, wofs_init_date_str, wofs_init_time_str,\
        #                    ps_init_time_str, str(lead_time_minutes).zfill(3))

        png_save_name = "%s_%s%s.png" %(v, start_valid_date_str, start_valid_time_str)

        plotter = WoFSPlotter(file_path=nc_fname, \
                                outdir = png_outdir,\
                                dt=c.wofs_update_rate,\
                                timestep_index=None,\
                                init_time = start_valid_dt,\
                                valid_time = end_valid_dt)

        #Mask out low or zero probabilities
        color_cont_data = data[v]
        #color_cont_data[color_cont_data<0.01] = -1.0 
        color_cont_data[color_cont_data < levels[0]] = -1.0

        plotter.plot(var_name = v,\
                        line_cont_data=color_cont_data,\
                        add_sharpness=False,\
                        line_cont_kws=cont_kwargs_dict,\
                        color_cont_data=None,\
                        add_text=True,\
                        save_name=png_save_name)




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
