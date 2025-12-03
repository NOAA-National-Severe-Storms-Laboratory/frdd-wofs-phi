#=====================================================
# Module that will handle the forecast specifications
# (e.g., forecast start and end valid times, 
# time windows, etc. 
#====================================================

import numpy as np 
import utilities as utils 
import json 


class ForecastSpecs: 

    """Handles/stores the forecast specifications."""


    def __init__(self, start_valid, end_valid, start_valid_dt, end_valid_dt, \
                    wofs_init_time, wofs_init_time_dt, forecast_window, ps_init_time,\
                    ps_lead_time_start, ps_lead_time_end, ps_init_time_dt, ps_ages,\
                    adjustable_radii_gridpoint, fieldsMethodsDict, singlePtFields,\
                    before_00z_date):


        """ Handles the key forecast specifications, especially related to date/time 
            aspects. 
            @start_valid : 4-char string : Start of the forecast valid period (e.g., "0225")
            @end_valid : 4-char string : End of the forecast valid period (e.g., "0255") 
            @start_valid_dt : datetime obj : Corresponding to the start of the forecast valid
                period
            @end_valid_dt : datetime obj : Corresponding to the end of the forecast valid period 
            @wofs_init_time : 4-char string : Corresponding to the wofs initialization time 
                (e.g., "0200") 
            @wofs_init_time_dt : datetime obj : WoFS initialization time 
            @forecast_window : int : Length of the forecast valid period (in minutes) 
            @ps_init_time : 4-char string : PS initialization time (e.g., "0224") 
            @ps_lead_time_start : int/floating pt : Time (in minutes) between the PS initialization
                time and the start of the forecast valid period 
            @ps_lead_time_end : int/floating pt : Time (in minutes) between the PS initialization
                time and the end of the forecast valid period
            @ps_init_time_dt : datetime object : ProbSevere initialization time
            @ps_ages : list : List of (potential) probSevere ages (in minutes; relative to the
                most recent ProbSevere file) based on the ProbSevere input files 
            @adjustable_radii_gridpoint : array/arr-like : Array of radii (in grid points)
                showing how much extrapolation should be done at each extrapolation time 
                from PS file initiation time to the maximum extrapolation time (which 
                is set in the json config file) 
            @fieldsMethodsDict : Dict : Keys are the predictor fields (ml name notation); 
                Values are the convolution methods (e.g., "max", "min", "abs", "minbut"). 
            @singlePtFields : List : List of single point fields (ml name notation) 
            @before_00z_date : 8-char string ("YYYYMMDD") : Corresponding to the date of the 
                forecast before 00z. 
            
        """    

        self.start_valid = start_valid
        self.end_valid = end_valid

        self.start_valid_dt = start_valid_dt
        self.end_valid_dt = end_valid_dt

        self.wofs_init_time = wofs_init_time
        self.wofs_init_time_dt = wofs_init_time_dt

        self.forecast_window = forecast_window

        self.ps_init_time = ps_init_time
        self.ps_lead_time_start = ps_lead_time_start
        self.ps_lead_time_end = ps_lead_time_end

        self.ps_init_time_dt = ps_init_time_dt

        self.ps_ages = ps_ages

        self.adjustable_radii_gridpoint = adjustable_radii_gridpoint

        self.fieldsMethodsDict = fieldsMethodsDict

        self.singlePtFields = singlePtFields

        self.before_00z_date = before_00z_date
    
        return  


    @classmethod
    def create_forecast_specs(cls, ps_files, wofs_files, json_config_filename): 
        """Factory method for creating a ForecastSpecs object. Creates and @Returns a
            ForecastSpecs object based on:  
            @ps_files : List : List of PS files
            @wofs_files : List : List of WoFS files 
            @json_config_filename : str : full path to json config file
        """

        #Find start/end valid and wofs initialization time/datetime objects from 
        #the wofs summary file names. 
        start_valid, start_valid_date, start_valid_dt = utils.find_date_time_from_wofs(\
            wofs_files[0], "valid")

        end_valid, end_valid_date, end_valid_dt = utils.find_date_time_from_wofs(\
            wofs_files[-1], "valid") 

        wofs_init_time, wofs_init_date, wofs_init_time_dt = utils.find_date_time_from_wofs(\
            wofs_files[0], "init") 

        #Get the date before 00z; Will be most relevant for training, I believe. 
        date_before_00z = utils.get_date_before_00z(wofs_init_time_dt, json_config_filename) 


        #Compute valid window based on the wofs files provided 
        valid_window = utils.subtract_dt(end_valid_dt, start_valid_dt, True)

        #Find PS init time from the first (most recent) PS file 
        ps_init_time, ps_init_date = utils.find_ps_date_time(ps_files[0], json_config_filename) 

        #Get ps datetime objects 
        ps_init_time_dt = utils.str_to_dattime(ps_init_time, ps_init_date) 
        
        #COME BACK HERE




        return 

