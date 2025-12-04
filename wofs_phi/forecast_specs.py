#=====================================================
# Module that will handle the forecast specifications
# (e.g., forecast start and end valid times, 
# time windows, etc. 
#====================================================

import numpy as np 
import utilities as utils 
import json 
import math 


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



    #NOTE: TODO: Make instance classes. 
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

        #Find PS init time from the first (most recent) PS file 
        ps_init_time, ps_init_date = utils.find_ps_date_time(ps_files[0], json_config_filename) 

        #Get ps datetime objects 
        ps_init_time_dt = utils.str_to_dattime(ps_init_time, ps_init_date) 
        
        #Find PS lead time for end of valid period (in minutes) 
        #based on PS initialization time and end of the valid period


        #Create new ForecastSpecs object; 
        #Initialize with None Values for variables that will be set with instance methods
        new_fspecs = ForecastSpecs(start_valid, end_valid, start_valid_dt, end_valid_dt,\
                        wofs_init_time, wofs_init_time_dt, None, ps_init_time, \
                        None, None, ps_init_time_dt, \
                        None, None, None, None, None)


        #Populate the None instance variables with instance methods 
        new_fspecs.set_before_00z_date(json_config_filename) 

        new_fspecs.set_ps_lead_time_start()

        new_fspecs.set_ps_lead_time_end() 

        new_fspecs.set_forecast_window()

        new_fspecs.set_ps_ages(ps_files, json_config_filename)

        new_fspecs.set_adjustable_radii_gridpoint(json_config_filename)

        new_fspecs.set_fieldsMethodsDict(json_config_filename)

        new_fspecs.set_singlePtFields(json_config_filename)


        return new_fspecs


    def set_before_00z_date(self, jsonConfigFile):
        """ Sets the date before 00z. Will be most useful for training, I believe.
            @jsonConfigFile : str : full path to .json config file
        """
    
        date_before_00z = utils.get_date_before_00z(self.wofs_init_time_dt,\
                            jsonConfigFile)

        self.before_00z_date = date_before_00z
        
        return 


    def set_ps_lead_time_start(self):
        """ Sets the ProbSevere lead time to the start of the valid period (i.e.,
            time between PS initialization and start of the forecast valid period)
        """

        ps_start_lead_time = utils.subtract_dt(self.start_valid_dt, self.ps_init_time_dt, True)

        self.ps_lead_time_start = ps_start_lead_time


        return 


    def set_ps_lead_time_end(self): 
        """ Sets the ProbSevere lead time to the end of the forecast valid period 
            (i.e., time between PS initialization and the end of the forecast
            valid period) 
        """ 

        ps_end_lead_time = utils.subtract_dt(self.end_valid_dt, self.ps_init_time_dt, True)
        
        self.ps_lead_time_end = ps_end_lead_time

        return 


    def set_forecast_window(self): 
        """ Sets the forecast_window attribute (i.e., length of the forecast valid period 
            in minutes) from end_valid_dt and start_valid_dt attributes
        """

        valid_window = utils.subtract_dt(self.end_valid_dt, self.start_valid_dt, True)

        self.forecast_window = valid_window

        return 


    def set_ps_ages(self, psFiles, jsonConfigFile): 
        """Sets the ps_ages instance variable: Will be a list of ages (in minutes) of
            the various PS files 
            @psFiles : List : List of ProbSevere files, with most recent (newest) first
            @jsonConfigFile : str : Full path to .json config file
        """
        
        ages = [] 

        first_ps_file = psFiles[0]
        first_ps_dt = utils.datetime_from_ps(first_ps_file, jsonConfigFile)

        for p in range(len(psFiles)): 
            curr_ps_file = psFiles[p] 
            curr_dt = utils.datetime_from_ps(curr_ps_file, jsonConfigFile) 

            #Find the difference between the current dt and the first ps_dt in minutes
            diff = utils.subtract_dt(first_ps_dt, curr_dt, True) 

            #Append to ages list
            ages.append(diff) 

        #Update the instance variable 
        self.ps_ages = ages

        return 


    def set_adjustable_radii_gridpoint(self, jsonConfigFile): 
        """Sets the adjustable radii at each extrapolation time based on attributes
            from the json config file. 
            The adjustable_radii_gridpoint variable is an array of radii of
            influence (in gridpoint coordinates) at each 1 minute of extrapolation 
            time. NOTE: In the current config file, the min and max radii are specified
            to be small, making this array all zeros for current applications.
            This signifies no additional spatial "buffering" of the extrapolated 
            ProbSevere "tracks" from ProbSevere objects. This doesn't have to be the case
            and can be changed in the .json config file (e.g., by specifying a greater
            max_radius) 
            @jsonConfigFile : str : Full path to .json config file
        """

        #Extract relevant items from config file 
        config_data = utils.read_json(jsonConfigFile)
        min_radius = config_data['min_radius'] 
        max_radius = config_data['max_radius'] 
        km_grid_spacing = config_data['dx_km'] 
        max_extrap_time = config_data['max_ps_extrap_time']

        adjustable_radii_km = np.linspace(min_radius, max_radius, int(max_extrap_time)+1)
        adjustable_radii = [math.ceil((r - 1.5)/km_grid_spacing) for r in adjustable_radii_km]

        #Only extrapolate until the end of the ps lead time; any further extrapolation is 
        #unnecessary. 

        adjustable_radii = adjustable_radii[0:int(self.ps_lead_time_end)+1]

        self.adjustable_radii_gridpoint = adjustable_radii

        return 


    def set_fieldsMethodsDict(self, jsonConfigFile):

        #Set the fields-methods dictionary from the .json config file 
        config_data = utils.read_json(jsonConfigFile)
        
        use_dict = config_data['fields_methods_dict']

        self.fieldsMethodsDict = use_dict


        return 


    def set_singlePtFields(self, jsonConfigFile): 
        
        config_data = utils.read_json(jsonConfigFile)

        singlePtFields = config_data['single_point_fields'] 

        self.singlePtFields = singlePtFields
    
        return 



