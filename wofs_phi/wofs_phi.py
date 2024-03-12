#===========================================
# This file runs the wofs-phi probabilities
#
# It needs an input date and time (which will be used 
# to set WoFS and PS initializations) and 
#
#
# Created by Eric Loken, 2/6/2024
#
#
#===========================================


#======================
# Imports
#======================
from shapely.geometry import Point, MultiPolygon, Polygon, LineString
from shapely.prepared import prep
from shapely import geometry
from pyproj import Transformer
from shapely.ops import transform
import numpy as np
#from mpl_toolkits.basemap import Basemap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import netCDF4 as nc
import pandas as pd
import json
from matplotlib.patches import Polygon as PolygonMPL
import math
#from scipy.ndimage.filters import maximum_filter, minimum_filter
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
import sys
import geopandas as gpd
import multiprocessing as mp
import itertools
from multiprocessing.pool import Pool
from datetime import datetime, timedelta
import datetime as dt
#from skexplain.common.multiprocessing_utils import run_parallel, to_iterator
import netCDF4 as nc
import os
import copy
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xarray as xr
import config as c
import utilities
import predictor_extractor as pex

class MLGenerator: 
    ''' This class will handle the ML generator functions. '''


    def __init__(self, wofs_files, ps_files, ps_path, wofs_path, torp_files, nc_outdir): 

        ''' @wofs_files is the list of wofs_files to use for the prediction (in chronological order,
            beginning with the start of the prediction window/valid period and ending with the end of
            the prediction window/valid period. 

            @ps_files are the list of probSevere files needed in reverse chronological order (i.e.,
            starting with the most recent file/time step.). Typically, we go back to 180 minutes ago.

            @ps_path is the string path to the probSevere files 

            @wofs_path is the string path to the wofs files   

            @nc_outdir is the directory to save the final .ncdf files to 
            
        '''

        self.wofs_files = wofs_files
        self.ps_files = ps_files
        self.ps_path = ps_path
        self.wofs_path = wofs_path
        self.nc_outdir = nc_outdir
        self.torp_files = torp_files


    def generate(self):
        '''Instance method to generate the predictions given an MLGenerator object.
            Kind of like the "main method" for generating the predictions. ''' 

        # Get grid stats from first wofs file 
        fcst_grid = Grid.create_wofs_grid(self.wofs_path, self.wofs_files[0])

        #Get the forecast specifications (e.g., start valid, end_valid, ps_lead time, wofs_lead_time, etc.) 
        #These will be determined principally by the wofs files we're dealing with
        forecast_specs = ForecastSpecs.create_forecast_specs(self.ps_files, self.wofs_files, c.all_fields_file, c.all_methods_file, c.single_pt_file)


        if (c.generate_forecasts == True):
            #Do PS preprocessing -- parallel track 1 -- Returns a ps object that holds an xarray and 
            #extrapolated geodataframe (xarray is what we likely most care about)

            #Skip for now for debugging the Wofs.preprocess_wofs method 
            ps = PS.preprocess_ps(fcst_grid, forecast_specs, self.ps_path, self.ps_files) 

            #Do WoFS preprocessing -- parallel track 2 
            wofs = Wofs.preprocess_wofs(forecast_specs, fcst_grid, self.wofs_path, self.wofs_files)

            #Concatenate parallel tracks 
            combined_xr = pex.xr_from_ps_and_wofs(ps, wofs) 

            #Add predictors -- wofs lat/lon, wofs point, wofs initialization time
       
            #Add gridded fields
 
            #Add latitude points to xarray 
            combined_xr = pex.add_gridded_field(combined_xr, fcst_grid.lats, "lat")

            #Add longitude points to xarray 
            combined_xr = pex.add_gridded_field(combined_xr, fcst_grid.lons, "lon") 
        
            #Add wofs y points 
            combined_xr = pex.add_gridded_field(combined_xr, fcst_grid.ypts, "wofs_y") 

            #Add wofs x points
            combined_xr = pex.add_gridded_field(combined_xr, fcst_grid.xpts, "wofs_x") 

            #Add wofs initialization time 
            combined_xr = pex.add_single_field(combined_xr, Wofs.INIT_TIME_DICT[forecast_specs.wofs_init_time]\
                                , "wofs_init_time", fcst_grid.ny, fcst_grid.nx) 

            #Add convolutions
            #What is needed? combined_xr, footprint_type, all_var_names, all_var_methods
            #Probably can compute stuff using the predictor_radii_km in config file 
            #rf_sizes, grid spacing of wofs
            conv_predictors_ds = pex.add_convolutions(combined_xr, c.conv_type, forecast_specs.allFields, \
                                forecast_specs.allMethods, forecast_specs.singlePtFields, \
                                c.predictor_radii_km, c.dx_km)

            #Get 1d predictor names 
            predictor_list = pex.get_predictor_list(forecast_specs.allFields, \
                                forecast_specs.singlePtFields, c.predictor_radii_km, \
                                c.extra_predictor_names)

       
            #Save predictor names to file -- if desired 
            #MLGenerator.save_predictor_names(predictor_list)

            #Extract 1d predictors  
            one_d_pred_array = pex.extract_1d(conv_predictors_ds, predictor_list, \
                            forecast_specs, fcst_grid)

            #Save predictors to file (if we're training)
            if (c.is_train_mode == True):
        
                #torp_predictors = TORP_List.gen_torp_npy(self.torp_files, fcst_grid, forecast_specs)

                #Get the filenames used for saving 
                full_npy_fname = MLGenerator.get_full_npy_filename(forecast_specs)
                dat_fname = MLGenerator.get_dat_filename(forecast_specs)
                rand_inds_fname = MLGenerator.get_rand_inds_filename(forecast_specs)

                #Save predictors to appropriate files
                pex.save_predictors(one_d_pred_array, c.sample_rate, fcst_grid, \
                        c.train_fcst_full_npy_dir, full_npy_fname, c.train_fcst_dat_dir,\
                        dat_fname, rand_inds_fname)

        
            else: 

                #NOTE: All radii and hazards will go into the same ncdf file 

                MLPrediction.create_rf_predictions(forecast_specs, fcst_grid, one_d_pred_array) 

                #Get RF filenames
                #rf_filenames = MLGenerator.get_rf_filenames(forecast_specs.forecast_window)
                #print (rf_filenames) 

                #Load RFs


                #Get predictions (for each hazard) 

                

                #Save predictions to ncdf 

                pass


        #Get the reports if we're supposed to 
        if (c.generate_reports == True):

            #TODO: input the one_d_pred_array
            rep = ReportGenerator.generate(forecast_specs, fcst_grid)

        




        return


   
    @staticmethod
    def save_predictor_names(pred_list):
        '''Saves predictor names to text file
            @pred_list is list of predictor names.'''

        with open("rf_variables.txt", 'w') as f:
            for name in pred_list: 

                f.write("%s\n" %name)

        f.close() 


        return 

    @staticmethod 
    def get_full_npy_filename(fSpecs):
        '''Sets and returns the filename for the full_npy file (holding the full array of predictors)
            given a ForecastSpecs object (@fSpecs) 
        '''

        use_fname = "wofs1d_%s_%s_%s_v%s-%s.npy" %(fSpecs.before_00z_date, fSpecs.wofs_init_time,\
                        fSpecs.ps_init_time, fSpecs.start_valid, fSpecs.end_valid) 

        return use_fname


    @staticmethod
    def get_dat_filename(fSpecs):
        '''Sets and returns the filename for the dat file (holding the randomly-sampled 
            array of predictors) given a ForecastSpecs object (@fSpecs) 
        '''
       
        use_fname = "wofs1d_%s_%s_%s_v%s-%s.dat" %(fSpecs.before_00z_date, fSpecs.wofs_init_time,\
                        fSpecs.ps_init_time, fSpecs.start_valid, fSpecs.end_valid)

 
        return use_fname


    @staticmethod 
    def get_rand_inds_filename(fSpecs):
        
        use_fname = "rand_inds_%s_%s_%s_v%s-%s.npy" %(fSpecs.before_00z_date, fSpecs.wofs_init_time,\
                        fSpecs.ps_init_time, fSpecs.start_valid, fSpecs.end_valid)

        return use_fname

    @staticmethod
    def get_obs_full_npy_filenames(fSpecs, haz_name, str_radius):
        '''Returns the filenames for the 1d and 2d full_npy files (for obs)
            @fSpecs is a ForecastSpecs object
            @haz_name is the hazard name
            @str_radius is the radius of the obs in string format
        '''
        
        use_fname_1d = "%s_reps1d_%s_v%s-%s_r%skm.npy" %(haz_name, fSpecs.before_00z_date,\
                        fSpecs.start_valid, fSpecs.end_valid, str_radius)

        use_fname_2d = "%s_reps2d_%s_v%s-%s_r%skm.npy" %(haz_name, fSpecs.before_00z_date,\
                        fSpecs.start_valid, fSpecs.end_valid, str_radius)


        return use_fname_1d, use_fname_2d  

    @staticmethod
    def get_obs_dat_filename(fSpecs, haz_name, str_radius):
        ''' Returns the filename for the dat file (for obs)
            @fSpecs is a ForecastSpecs object
            @haz_name is the hazard name
            @str_radius is the radius of the obs in string format
        '''
       
        use_fname = "%s_reps1d_%s_%s_%s_v%s-%s_r%skm.npy" %(haz_name, fSpecs.before_00z_date,\
                        fSpecs.wofs_init_time,fSpecs.ps_init_time, fSpecs.start_valid,\
                        fSpecs.end_valid, str_radius)
 

        return use_fname


class MLPrediction:
    ''' This class handles the ML prediction'''


    def __init__(self, rf_filename, hazard, radius_str, radius_float, probs,\
                    rf_model, predictor_arr):
        '''@rf_filename is the full name of the pickled model (including the path)
            @hazard is the string hazard name (e.g., "wind", "hail", or "tornado") 
            @radius_str is the radius in string form (e.g., "7.5", "15", "30", "39", 
                or "warning" if trained on warnings)
            @radius_float is the radius in float form (e.g., 7.5, 15.0, 30.0, etc.; 
                or 0.0 if trained on warnings)
            @probs is the 2-d np array of probabilities 
            @rf_model is the unpickled RF model used to make predictions 
            @predictor_arr is the array of predictors (in format 
                (rows: examples, columns: predictors))
        '''


        self.rf_filename = rf_filename
        self.hazard = hazard
        self.time_window = time_window 
        self.radius_str = radius_str
        self.radius_float = radius_float
        self.probs = probs 
        self.rf_model = rf_model 
        self.predictor_arr = predictor_arr
    

        return 


    @classmethod
    def create_rf_predictions(cls, specs, grid_obj, predictor_array):
        '''Factory method for creating RF predictions.
            @specs is a ForecastSpecs object
            @grid_obj is a Grid object
            @predictor_array is the array of predictors to run through the RF 
                to make predictions. Format: (rows: examples, columns: predictors)
        '''

        #TODO: Might need to add 

        zero_probs_2d = np.zeros((grid_obj.ny, grid_obj.nx))
        
        #Will hold the set of mlp objects over the different hazards and radii 
        mlp_list = [] 

        for hazard in c.final_hazards:

            for r in range(len(c.obs_radii_str)):
                radius_str = c.obs_radii_str[r]
                radius_float = c.obs_radii_float[r] 

                #Get filename 
                rf_filename = MLPrediction.get_rf_filename(specs.forecast_window,\
                                hazard, radius_str) 


                rfModel = MLPrediction.load_rf_model(rf_filename) 
                #Create new MLPrediction object 
                mlp = MLPrediction(rf_filename, hazard, radius_str, radius_float, \
                            zero_probs_2d, rfModel, predictor_array)

                #Now, implement instance methods -- TODO 
                mlp.set_probs(grid_obj)

                #Append to list 
                mlp_list.append(mlp) 


        #Once we have a list of mlp objects, save to netcdf file 
                


        return 


    def set_probs(self, curr_grid):
        '''Sets the final output probabilities based on the instance's attributes
            @curr_grid is the current Grid object for this situation'''

        one_d_probs = self.rf_model.predict_proba(self.predictor_arr)[:,1] 

        two_d_probs = MLPrediction(one_d_probs, curr_grid.ny, curr_grid.nx) 

        #Set the probs attribute 
        self.probs = two_d_probs 

        return 

   

    @staticmethod 
    def save_to_ncdf(list_of_mlps):
        '''Obtains/saves ncdf files based on a list of MLPrediction objects.
            @list_of_mlps is the list of MLPrediction objects.
        '''
        

        return 
 
    @staticmethod
    def convert_1d_field_to_2d(one_d_field, ny, nx):
        '''Maps a 1d field to 2d -- for a given day/forecast
            @one_d_field is incoming 1-d field (e.g., of probabilities)
            @ny is the number of y grid points
            @nx is the number of x grid points
        '''

        two_d_field = one_d_field.reshape(nY, nX)


        return two_d_field
   
    @staticmethod 
    def load_rf_model(filename):
        '''Loads given pkl file (@filename) into an array'''

        f = open(Filename, 'rb')
        new = pickle.load(f)
        f.close()

    return new



    #TODO: Will have to fully implmement this later -- and probably make this an instance method
    @staticmethod
    def get_rf_filename(window_minutes, haz_name, radius):
        '''Gets the full filenames to the rf file. Returns the pkl files 
            in the hazard order as given in the config file
            @window_minutes is the time window of the valid period in minutes

        '''

        pkl_filename = "%s/wofsphi_%s_%smin_%s_mode_r%skm.pkl" \
                        %(c.rf_dir, haz_name, window_minutes, c.mode, radius)


        return pkl_filename


class FinalNCFile:
    '''This class handles obtaining the final ncdf file based on a list of MLPrediction objects'''



    def __init__(self, list_of_mlps, xr ):
        '''@list_of_mlps is a list of MLPrediction objects needed for the current final netcdf file
            @xr is the xarray dataset that will be used to construct the final netcdf file'''

        self.list_of_mlps = list_of_mlps
        self.xr = xr 

        return 


    #TODO
    @classmethod
    def create_FNCF_from_mlps(cls, mlps_list):
        ''' Factory method for creating FinalNCFile object from list of MLPrediction objects.
            @mlps_list is a list of MLPrediction objects.'''

        return 


    #TODO:
    def set_xr(self):

        return 



class ReportGenerator:
    '''This class handles generating the report grid'''

    #Variable names corresponding with how the coords.txt files are set up. 
    COORDS_FILE_COLUMNS = ['time', 'lon', 'lat'] 


    def __init__(self, date_before_00z, rep_start_dt, rep_end_dt,\
                    buffer_minutes, start_valid_dt, end_valid_dt, start_valid, end_valid,\
                    report_coords_df, report_gdf, report_grid_2d, report_grid_1d,\
                    rand_inds, sampled_grid_1d, hazard, radius, target):
        '''@date_before_00z is the 8-character string (YYYYMMDD) before 00z
            @rep_start_dt is a datetime object corresponding to the start of 
                the period we care about for reports i.e., 
                start of the valid period - buffer time 
            @rep_end_dt is a datetime object corresponding to the end of the 
                period we care about for reports. i.e., 
                end of the valid period + buffer time 
            @buffer_minutes is the period of time to add on either side of the valid
                period in minutes (e.g., 10 means add +/- 10 minutes to the valid period)
            @start_valid_dt is the datetime object corresponding to the start
                of the forecast valid period (no buffer)
            @end_valid_dt is the datetime object corresponding to the end of the 
                forecast valid period (no buffer) 
            @start_valid is the 4-character string (HHMM) corresponding to the 
                start of the valid period (no buffer)
            @end_valid is the 4-character string (HHMM) corresponding to the end
                of the valid period (no buffer) 
            @report_coords_df is a dataframe of reports with columns ['time', 'lon', 'lat']
            @report_gdf is a geodataframe of report examples (locations are Points (or 
                Polygons for warnings)) 
            @report_grid_2d is a 2-d grid of binary reports 
            @report_grid_1d is a 1-d "grid" of binary reports (correpsonding to above) 
            @rand_inds is a list of random indices used to sample the 1-d grid of binary
                reports (for training) 
            @sampled_grid_1d is the list of randomly-sampled 1-d points 
                (to use during training) 
            @hazard is a string corresponding to the hazard investigated (e.g., 
                "hail", "wind", or "tornado") 
            @radius is the spatial radius to apply (e.g., 7.5, 15, 30, 39km) 
            @target is a string telling what we're using as our target:
                "lsrs" for local storm reports or "warnings" for warnings
        '''

        self.date_before_00z = date_before_00z
        self.rep_start_dt = rep_start_dt
        self.rep_end_dt = rep_end_dt
        self.buffer_minutes = buffer_minutes
        self.start_valid_dt = start_valid_dt 
        self.end_valid_dt = end_valid_dt 
        self.start_valid = start_valid
        self.end_valid = end_valid
        self.report_coords_df = report_coords_df
        self.report_gdf = report_gdf
        self.report_grid_2d= report_grid_2d
        self.report_grid_1d= report_grid_1d
        self.rand_inds = rand_inds
        self.sampled_grid_1d = sampled_grid_1d
        self.hazard = hazard 
        self.radius = radius
        self.target = target

        return 

    #TODO: Come back here
    @classmethod
    def generate(cls, fcst_specs, fcst_grid):
        '''Factory method for creating a report object
            @fcst_specs is a ForecastSpecs object for the current 
                situation. 
            @grid is a 
        '''

        #We'll want to generate ReportGenerator objects for each hazard 
        
        #Set the date and pre-00z date (probably from fcst_specs, etc.)
        dateBefore00z = fcst_specs.before_00z_date

        startValidDT = fcst_specs.start_valid_dt 
        endValidDT = fcst_specs.end_valid_dt 

        startValid_str = fcst_specs.start_valid
        endValid_str = fcst_specs.end_valid

        #Obtain the reps_start_dt and reps_end_dt objects by adding in the 
        #buffer time 
        repsStartDT = startValidDT - timedelta(minutes=c.report_time_buffer)
        
        repsEndDT = endValidDT + timedelta(minutes=c.report_time_buffer) 


        zero_2d_grid = np.zeros((fcst_grid.ny, fcst_grid.nx)) #Used to initialize the binary report grid 
        zero_1d_grid = np.zeros(fcst_grid.ny*fcst_grid.nx) 
        zero_rand_inds_arr = np.zeros(int(fcst_grid.ny*fcst_grid.nx*c.sample_rate))

        #Need to obtain ReportsGenerator object for each hazard
        for hazard in c.final_hazards:
            for radius in c.obs_radii_float:
                reps_file = "%s/%s_coords_%s.txt" %(c.reps_coords_dir, hazard, dateBefore00z)
           
                #Obtain the coordinates in a pandas dataframe  -- NOTE: We'll add a datetime column 
                #as well 
                coords_df = ReportGenerator.get_daily_reps_df(reps_file, cls.COORDS_FILE_COLUMNS) 

                #Add a datetime column to the coords_df 
                coords_df = ReportGenerator.add_datetime_to_df(coords_df, dateBefore00z) 

                #Get the subset of reports we're interested in based on the repsStartDT and repsEndDT

                subset_df = ReportGenerator.get_subset_reps_df(coords_df, repsStartDT, repsEndDT)

                subset_df = ReportGenerator.add_latlon_to_df(coords_df) 


                #Initialize a ReportGenerator object -- with gdf initialized as None. Will be updated later. 

                rep = ReportGenerator(dateBefore00z, repsStartDT, repsEndDT, c.report_time_buffer, startValidDT, endValidDT, \
                        startValid_str, endValid_str, subset_df, None, zero_2d_grid, zero_1d_grid, \
                        zero_rand_inds_arr, zero_rand_inds_arr, hazard, radius, c.report_target)


                #Set the coordinates in a pandas geodataframe 
                rep.set_gdf()

                #Now we want to get the binary report grid  
                rep.set_report_grid_2d(fcst_grid)

                #Set the 1d binary report grid 
                rep.set_report_grid_1d(fcst_grid) 

                #Set the random indices list
                rep.set_rand_inds(fcst_specs)

                #Set the sampled 1d report grid 
                rep.set_sampled_grid_1d()

                #Now we just need to save to file -- with the random sampling as well 
    

                rep.save_to_file(fcst_specs)

        return rep 


    def save_to_file(self, specs):
        '''Saves the binary 2-d array to file
            @specs is a ForecastSpecs object
        '''

        #Get filenames 
        full_npy_one_d_name, full_npy_two_d_name = MLGenerator.get_obs_full_npy_filenames(\
                specs, self.hazard, str(self.radius))

        dat_name = MLGenerator.get_obs_dat_filename(specs, self.hazard, str(self.radius))
        

        #Save to appropriate files 
        np.save("%s/%s" %(c.train_obs_full_npy_dir, full_npy_two_d_name), self.report_grid_2d)
        np.save("%s/%s" %(c.train_obs_full_npy_dir, full_npy_one_d_name), self.report_grid_1d)
        self.sampled_grid_1d.tofile("%s/%s" %(c.train_obs_dat_dir, dat_name))
        
        return 
    
    def set_rand_inds(self, specs):
        ''' Obtains the list of random indices to use.
            @specs is a ForecastSpecs object.
         '''
        #Get rand_ind filename 
        rand_ind_name = "%s/%s" %(c.train_fcst_dat_dir, \
                            MLGenerator.get_rand_inds_filename(specs))

        #Read in rand_inds
        rand_inds = np.load(rand_ind_name) 
    
        #Set random indices
        self.rand_inds = rand_inds


        return 

    def set_sampled_grid_1d(self):
        ''' Sets the sampled_grid_1d using the random indices and report_grid_1d
            attributes.
        '''
        sampled_grid = self.report_grid_1d[self.rand_inds]
        self.sampled_grid_1d = np.float32(sampled_grid)

        return 

    def set_report_grid_1d(self, curr_grid):
        '''Sets the 1d reports attribute from the report_grid_2d attribute 
        '''

        #Only need to update this if we have reports 
        if (len(self.report_coords_df) > 0):
            n_points = curr_grid.ny*curr_grid.nx #Total number of grid points 
        
            one_d_grid = self.report_grid_2d.reshape(n_points, -1) 

            self.report_grid_1d = np.float32(one_d_grid)

        return 


    def set_report_grid_2d(self, curr_grid):

        '''Sets the binary report grid from other instance variables/attributes.
            @curr_grid is the current Grid object (wofs forecast grid).
        '''
        #Only need to do this if we have reports 
        if (len(self.report_coords_df) > 0):
           

            #NOTE: We also need a wofs geodataframe 
            #get wofs dataframe from grid object
            wofs_gdf = PS.get_wofs_gdf(curr_grid)

            #Merge the two geodataframes 
            new_gdf = ReportGenerator.merge_wofs_and_reps(wofs_gdf, self.report_gdf, c.dx_km)


            #Get the 2d footprints 

            footprints = pex.get_footprints([self.radius], c.dx_km)
            
            #There should only be one element here--the first
            footprints = footprints[0]

            #Next, binarize the wofs grid 

            binary_grid = ReportGenerator.binarize_wofs(new_gdf, curr_grid.ny, curr_grid.nx, footprints)

            #Make float32
            binary_grid = np.float32(binary_grid) 

            #Update the instance
            self.report_grid_2d = binary_grid 

        return 


    def set_gdf(self):
        '''sets the report gdf from other instance variables/attibutes'''

        #If there are no elements to set, then the report_gdf attribute will be
        #left as None
        if (len(self.report_coords_df) == 0):
            self.report_gdf = None

        else: 
            out_gdf = gpd.GeoDataFrame(data=self.report_coords_df, geometry=self.report_coords_df['latlon'], crs="EPSG:4326")
            #Set the instance variable
            self.report_gdf = out_gdf

        return 


    @staticmethod
    def binarize_wofs(in_gdf, nY, nX, footprint_arr):
        '''Binarizes the wofs grid (1 with observed LSR, 0 otherwise) and applies the spatial 
            radius 
            @Returns a 2-d array corresponding to the convolved binarized field (ny,nx). 
            @in_gdf is the merged wofs-reports geodataframe
            @nY is the number of y points
            @nX is the number of x points
            @footprint_arr is the 2-d footprints array
        '''

        #First, create an initial wofs grid and binarize
        wofs_binary = np.zeros((nY, nX))

        num_reps = len(in_gdf)
        jpts = in_gdf['wofs_j'].values
        ipts = in_gdf['wofs_i'].values

        for n in range(num_reps):
            #Get the wofs point
            jpt = jpts[n]
            ipt = ipts[n]
            wofs_binary[jpt,ipt] = 1.0

        #Now, we can apply the convolution -- and save to 2d array

        conv_wofs_binary = maximum_filter(wofs_binary, footprint=footprint_arr)

        return conv_wofs_binary


    @staticmethod 
    def merge_wofs_and_reps(wofsGDF, repsGDF, max_distance_km):

        '''Returns a new geodataframe that essentially maps the reports geodataframe
            to the wofs grid.
            @wofsGDF is the wofs geodataframe
            @repsGDF is the reports geodataframe 
            @max_distance is the maximum distance in km 
            @Returns merged wofs and reps geodataframe 
                (merged based on nearest point within @max_distance_km
            NOTE (from docs): Results will include multiple output records for a single input record where 
                there are multiple equidistant nearest or intersected neighbors (probably fine because
                we apply a buffer anyway) 
        '''

        #Convert maxDistance to meters
        search_radius = max_distance_km*1000.0

        #Convert both geodataframes to meters geometry
        wofsGDF.to_crs("EPSG:32634", inplace=True)
        repsGDF.to_crs("EPSG:32634", inplace=True)

        #Apply the merger 
        out_gdf = wofsGDF.sjoin_nearest(repsGDF, how='inner', max_distance=search_radius)

        #Convert back to latlon coords
        out_gdf.to_crs("EPSG:4326", inplace=True)
        wofsGDF.to_crs("EPSG:4326", inplace=True)
        repsGDF.to_crs("EPSG:4326", inplace=True)

        return out_gdf


    @staticmethod 
    def add_latlon_to_df(in_df):
        ''' Adds a "latlon" column to the incoming dataframe  [form: Point(lon, lat)]
            @in_df is the incoming dataframe (columns "time", "lon", "lat", "datetime") 
        '''
        if (len(in_df) > 0):
            #Add latlon column 

            in_df['latlon'] = [Point(in_df.iloc[a,1], in_df.iloc[a,2]) for a in range(len(in_df))]

        return in_df

    @staticmethod
    def get_subset_reps_df(reps_df, start_time_dt, end_time_dt):
        '''Returns the subset of the dataframe that we're interested in. 
            @reps_df is the incoming dataframe of reports (columns of "time", "lon", "lat", "datetime")
            @start_time_dt is the datetime object corresponding to the reports start time 
                with buffer applied
            @end_time_dt is the datetime object corresponding to the reports end time 
                with buffer applied
        '''

        if (len(reps_df) > 0):

            reps_df = reps_df.loc[(reps_df['datetime'] >= start_time_dt) & (reps_df['datetime'] <= end_time_dt)]


        return reps_df


    @staticmethod
    def add_datetime_to_df(reps_df, datebefore00z):
        '''
            Adds a column of datetimes to the reports dataframe if the reps_df is not empty
            @Returns this new dataframe 
            @reps_df is the incoming dataframe of reports (columns of "time", "lon", "lat") 
            @datebefore00z is the string date (YYYYMMDD) before 00z 
        '''

        #Need to Add column of datetimes 
        if (len(reps_df) > 0):

            #Get a list of time strings from the df 
            times = [str(int(t)).zfill(4) for t in reps_df['time']]

            #12z datetime object -- will be used to know whether or not to increment the datetime objects after 00z
            dattime_12z = ForecastSpecs.str_to_dattime("1200", datebefore00z)

            #Get corresponding datetime objects 
            datetimes = [] 
            for r in range(len(times)):
                time = times[r] 
                dt_obj = ForecastSpecs.str_to_dattime(time, datebefore00z)

                #If this datetime object is earlier than the 12z datetime object given above, we need to increment
                #by 1 day 
                if (dt_obj < dattime_12z):
                    dt_obj += timedelta(days=1) 

                #Append to list 
                datetimes.append(dt_obj) 

            #Add column to dataframe 
            reps_df['datetime'] = datetimes
        else: 
            all_columns = ReportGenerator.COORDS_FILE_COLUMNS
            all_columns.append('datetime') 
            reps_df = pd.DataFrame(columns=all_columns) 

        return reps_df

    @staticmethod
    def get_daily_reps_df(coordsFile, header_names):
        '''Reads in the file containing the reports for the given day.
            @Returns a pandas dataframe with the given @header names as columns
            @coordsFile is the filename containing the reports for the given
                day. 
            @header_names is the list of column names corresponding to the structure of 
                the @coordsFile  

        '''

        reps_df = pd.read_csv(coordsFile, names=header_names, dtype='float32', header=None)

        return reps_df 



class MLTrainer:
    '''This class handles the training of the model'''
    
    def __init__(self, dates, forecast_length, num_fcsts, num_folds, haz, n_jobs = 36, n_trees = 200, criterion = 'entropy', max_depth = 15, min_samples_leaf = 20,
                max_features = "sqrt"):
        self.dates = dates #this needs to be in a list of date strings (YYYYMMDD)
        self.forecast_length = forecast_length
        self.num_fcsts = num_fcsts
        self.num_folds = num_folds
        self.hazard = haz
        
        #RF variables/hyperparameters as optional inputs:
        self.n_jobs = n_jobs
        self.n_trees = n_trees
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
    
    def do_training_observations(self):
        '''Will do the training, run on validation and map probs to success ratios, then run on testing data with the mapped SRs'''
        
        inits = c.top_hour_inits.extend(c.bottom_hour_inits)
        self.set_chunk_splits()
        #TODO: get txt files of all variable names
        #self.wofs_vars = 
        #self.ps_vars = 
        #self.torp_vars = 
        #self.n_vars = len(self.wofs_vars) + len(self.ps_vars)
        train_start_ind = 0
        train_end_ind = self.num_folds - 3
        val_ind = self.num_folds - 2
        test_ind = self.num_folds - 1
        for r in c.obs_radii:
            for k in range(num_folds):
                self.save_model(k, r)
                #self.run_on_validation()
                #self.save_test_predictions()
    
    def save_model(self, fold, r):
        '''Trains and saves a single model (one hazard, forecast window, fold, radius)'''
        
        fold_train_dates = self.date_train_folds[fold]
        for i in range(self.num_fcsts):
            for date in fold_train_dates:
                for init in inits:
                    init_dt = utilities.make_dt_from_str(date, init)
                    start_dt = init_dt + datetime.timedelta(seconds = ((c.wofs_spinup_time*60) + (self.forecast_length*i*60)))
                    end_dt = start_dt + datetime.timedelta(seconds = forecast_length*60)
                    predictor_fname, fcst_specs = self.get_predictors_fname(init_dt, start_dt, end_dt)
                    if predictor_fname == '':
                        continue
                    os.system('cat %s >> ./training_fdata_%s_%smin_w%s_r%skm.dat' %(predictor_fname, self.hazard, self.forecast_length, str(i+1), str(r)))
                    obs_fname = self.get_events_fname(start_dt, end_dt)
                    os.system('cat %s >> ./training_odata_%s_%smin_w%s_r%skm.dat' %(obs_fname, self.hazard, self.forecast_length, str(i+1), str(r)))
            
            forecast_file = './training_fdata_%s_%smin_w%s_r%skm.dat' %(self.hazard, self.forecast_length, str(i+1), str(r))
            obs_file = './training_odata_%s_%smin_w%s_r%skm.dat' %(self.hazard, self.forecast_length, str(i+1), str(r))

            training_x = self.read_binary(forecast_file)
            training_y = self.read_obs_binary(obs_file)

            clf = RandomForestClassifier(n_jobs = self.n_jobs, n_estimators = self.n_trees, criterion = self.criterion, max_depth = self.max_depth,
                                         min_samples_leaf = self.min_samples_leaf, max_features = self.max_features)
            clf.fit(training_x, training_y)

            clf_filename = '%s/%s/wofsphi_%s_%smin_w%s_r%skm_fold%s.pkl' %(c.model_save_dir, self.hazard, self.hazard, self.forecast_length, str(i+1), str(r), fold)
                    
            clf_pkl = open(clf_filename, 'wb')
            pickle.dump(clf, clf_pkl)
            clf_pkl.close()
            
            os.system('rm %s' %(forecast_file))
            os.system('rm %s' %(obs_file))
    
    def get_predictors_fname(self, init_dt, start_dt, end_dt):
        '''gets the full filename (including directory) of where the predictors *should* be so that we can see if they still need to be generated'''
        
        wofs_files, wofs_dir = self.find_wofs_ALL_files(init_dt, start_dt, end_dt)
        ps_files, is_recent_ps_file = self.find_ps_files(start_dt)
        num_wofs_fcsts = (self.forecast_length/c.wofs_update_rate) + 1
        if (not (len(wofs_files) == num_wofs_fcsts) and is_recent_ps_file):
            #we either don't have all the necessary wofs file or don't have a recent ps file to train on --> don't generate/load predictors
            return ''
        fcst_specs = ForecastSpecs.create_forecast_specs(ps_files, wofs_files, c.all_fields_file, c.all_methods_file, c.single_pt_file)
        pred_filename = c.train_fcst_dat_dir + '/' + MLGenerator.get_dat_filename(fcst_specs)
        if not os.path.isfile(pred_filename):
            #predictor files don't exist, we need to generate them
            torp_files = [] #needed for generator, but not training on this... yet
            nc_outdir = '.' #doesn't matter since we're just using the generator to make/save predictor files
            generator = MLGenerator(wofs_files, ps_files, c.ps_dir, wofs_dir, torp_files, nc_outdir)
            generator.generate()
        return pred_filename, fcst_specs
    
    def get_events_fname(self, init_dt, start_dt, end_dt, r, fcst_specs):
        '''gets the full obs files then randomly samples them and saves them to a dat file, returns the dat filename'''
        
        if init_dt.hour < 12:
            init_dt = init_dt - datetime.timedelta(days = 1)
        if start_dt.hour < 12:
            start_dt = start_dt - datetime.timedelta(days = 1)
        if end_dt.hour < 12:
            end_dt = end_dt - datetime.timedelta(days = 1)
        day_str = init_dt.strftime('%Y%m%d-%H%M').split('-')[0]
        init_str = init_dt.strftime('%Y%m%d-%H%M').split('-')[1]
        start_str = start_dt.strftime('%Y%m%d-%H%M').split('-')[1]
        end_str = end_dt.strftime('%Y%m%d-%H%M').split('-')[1]
        
        full_npy_fname = '%s/%s_reps1d_%s_v%s-%s_r%skm.npy' %(c.train_obs_full_npy_dir, self.hazard, day_str, start_str, end_str, str(r))
        obs_full_npy = np.load(full_npy_fname)
        sample_indices_fname = MLGenerator.get_rand_inds_filename(fcst_specs)
        sample_indices_full_fname = '%s/%s' %(c.train_fcst_dat_dir, sample_indices_fname)
        sample_indices = np.load(sample_indices_full_fname)
        sampled_obs = obs_full_npy[sample_indices]
        
        sampled_obs_dat_fname = '%s/%s_reps1d_%s_%s_v%s-%s_r%skm.dat' %(c.train_obs_dat_dir, self.hazard, day_str, init_str, start_str, end_str, str(r))
        sampled_obs.tofile(sampled_obs_dat_fname)
        
        return sampled_obs_dat_fname
        
    
    def set_chunk_splits(self):
        '''Does the k fold logic. The class stores a list of dates to train on, and this samples it into training, testing, validation for each fold.
        This allows us to just use the fold number as an index to each of the lists to get the list of dates for the current fold for testing, validation,
        and training.'''
        
        num_dates = len(self.dates)
        indices = np.arange(num_dates)
        splits = np.array_split(indices, self.num_folds)
        chunk_splits = []
        for i in range(len(splits)):
            split = splits[i]
            chunk_splits.append(split[0])
        chunk_splits.append(num_dates)
        self.date_test_folds = []
        self.date_val_folds = []
        self.date_train_folds = []
        for i in range(self.num_folds):
            if i == 0:
                self.date_test_folds.append(self.dates[chunk_splits[i]:chunk_splits[i+1]])
                self.date_val_folds.append(self.dates[chunk_splits[-2]:chunk_splits[-1]])
                self.date_train_folds.append(self.dates[chunk_splits[i+1]:chunk_splits[i+1+(self.num_folds-2)]])
            elif i == self.num_folds - 1:
                self.date_test_folds.append(self.dates[chunk_splits[i]:chunk_splits[-1]])
                self.date_val_folds.append(self.dates[chunk_splits[i-1]:chunk_splits[i]])
                self.date_train_folds.append(self.dates[chunk_splits[0]:chunk_splits[self.num_folds-2]])
            elif (i+1+(self.num_folds-2)) > (self.num_folds):
                self.date_test_folds.append(self.dates[chunk_splits[i]:chunk_splits[i+1]])
                self.date_val_folds.append(self.dates[chunk_splits[i-1]:chunk_splits[i]])
                mid_ind = ((i+1+(self.num_folds-2)) - (self.num_folds - 1))
                train_folds = self.dates[chunk_splits[i+1]:chunk_splits[-1]]
                train_folds.extend(self.dates[chunk_splits[0]:chunk_splits[mid_ind]])
                self.date_train_folds.append(train_folds)
            else:
                self.date_test_folds.append(self.dates[chunk_splits[i]:chunk_splits[i+1]])
                self.date_val_folds.append(self.dates[chunk_splits[i-1]:chunk_splits[i]])
                self.date_train_folds.append(self.dates[chunk_splits[i+1]:chunk_splits[i+1+(self.num_folds-2)]])
    
    def find_wofs_files(self, init_dt, start_dt, end_dt):
        '''Returns the wofs files (and wofs directory for the date/initialization) for a given init, start, and end.
        Part of class because it uses the forecast length'''
        
        if init_dt.hour < c.wofs_reset_hour:
            init_dt = init_dt - datetime.timedelta(days = 1)
        init_date_str = init_dt.strftime('%Y%m%d-%H%M').split('-')[0]
        init_time_str = init_dt.strftime('%Y%m%d-%H%M').split('-')[1]
        wofs_date_dir = c.wofs_dir + init_date_str + '/' + init_time_str + '/'
        wofs_files = []
        if c.use_ALL_files:
            for i in range((self.forecast_length/c.wofs_update_rate) + 1):
                valid_dt = init_dt + datetime.timedelta(seconds = ((i*c.wofs_update_rate)*60))
                valid_time_str = valid_dt.strftime('%Y%m%d-%H%M').split('-')[1]
                if i < 10:
                    wofs_file = 'wofs_ALL_0%s_%s_%s_%s.nc' %(str(i), init_date_str, init_time_str, valid_time_str)
                else:
                    wofs_file = 'wofs_ALL_%s_%s_%s_%s.nc' %(str(i), init_date_str, init_time_str, valid_time_str)
                wofs_files.append(wofs_file)
        else: #use ENV files instead
            for i in range((self.forecast_length/c.wofs_update_rate) + 1):
                valid_dt = init_dt + datetime.timedelta(seconds = ((i*c.wofs_update_rate)*60))
                valid_time_str = valid_dt.strftime('%Y%m%d-%H%M').split('-')[1]
                if i < 10:
                    wofs_file = 'wofs_ENV_0%s_%s_%s_%s.nc' %(str(i), init_date_str, init_time_str, valid_time_str)
                else:
                    wofs_file = 'wofs_ENV_%s_%s_%s_%s.nc' %(str(i), init_date_str, init_time_str, valid_time_str)
                if os.path.isfile(wofs_file) or (os.path.isfile(wofs_file.replace('ALL', 'ENV')) and os.path.isfile(wofs_file.replace('ALL', 'ENS')) and os.path.isfile(wofs_file.replace('ALL', 'SVR'))):
                    wofs_files.append(wofs_file)
        
        return wofs_files, wofs_date_dir
    
    def find_ps_files(start_dt):
        '''Returns the ps files associated with a given start time. Part of class to associated with wofs file finder
        (and torp file finder when it gets created). Also returns true/false to determine if there is a ps_file
        within 10 minutes of the inputted start time'''
        
        ps_files = []
        recent_files = False
        earliest_time = start_dt - start_dt.timedelta(hours = 3)
        curr_time = earliest_time
        for i in range(c.ps_search_minutes):
            curr_date_str = curr_time.strftime('%Y%m%d-%H%M').split('-')[0]
            curr_time_str = curr_time.strftime('%Y%m%d-%H%M').split('-')[1] + '00'
            ps_file = 'MRMS_EXP_PROBSEVERE_%s.%s.json' %(curr_date_str, curr_time_str)
            if os.path.isfile(ps_file):
                ps_files.append(ps_file)
                if i >= (c.ps_search_minutes - c.ps_recent_file_threshold):
                    recent_files = True
            curr_time = curr_time + datetime.timedelta(seconds = 60)
        
        
        return ps_files, recent_files
    
    def read_binary(self, infile, header = False):
        '''read in the .dat forecast files'''
        
        #@infile is the filename of unformatted binary file
        #Returns a numpy array of data
        #nvars is the number of RF variables 
        #@header is a binary variable -- True if contains a 1-elmnt header/footer at 
        #beginning and end of file; else False


        f = open ( infile , 'rb' )
        arr = np.fromfile ( f , dtype = np.float32 , count = -1 )
        f.close()

        if (header == True):
            arr = arr[1:-1]


        #print arr.shape
        #Rebroadcast the data into 2-d array with proper format
        arr.shape = (-1, nvars)
        return arr
    
    def read_obs_binary(self, infile):
        '''read in the .dat obs files'''
        
        #@infile is the filename of unformatted binary file
        #Returns a numpy array of data

        f = open ( infile , 'rb' )
        arr = np.fromfile ( f , dtype = np.float32 , count = -1 )
        f.close()

        return arr
    
class Wofs:
    '''Handles the wofs forecasting/processing'''
    
    #Number of WoFS members 
    N_MEMBERS = 18

    #Will compute number of members exceeding this threshold 
    DBZ_THRESH = 40 

    #Dictionary converting wofs initialization time strings to integers in order 
    #(to be used for converting wofs initialization times to 1-d predictors
    INIT_TIME_DICT = {'1700':1, '1730':2, '1800':3, '1830':4, '1900':5, '1930':6,\
                        '2000':7, '2030':8, '2100':9, '2130':10, '2200':11, '2230':12,\
                        '2300':13, '2330':14, '0000':15, '0030':16, '0100':17, '0130':18,\
                        '0200':19, '0230':20, '0300':21, '0330':22, '0400':23, '0430':24,\
                        '0500':25, '0530':26, '0600':27} 

    #Legacy file naming conventions 
    ENS_VARS = ["ws_80", "dbz_1km", "wz_0to2_instant", "uh_0to2_instant",  "uh_2to5", "w_up",\
                     "w_1km", "w_down", "buoyancy", "div_10m", "10-500m_bulkshear", "ctt", "fed",\
                     "rh_avg", "okubo_weiss", "hail", "hailcast", "freezing_level", "comp_dz"]

    ENV_VARS = ["mslp", "u_10", "v_10", "td_2", "t_2", "qv_2", "theta_e", "omega", "psfc", \
                            "pbl_mfc", "mid_level_lapse_rate", "low_level_lapse_rate" ]
    
    SVR_VARS = ["shear_u_0to1", "shear_v_0to1", "shear_u_0to3", "shear_v_0to3", "shear_u_0to6", "shear_v_0to6",\
                      "srh_0to500", "srh_0to1", "srh_0to3", "cape_sfc", "cin_sfc", "lcl_sfc", "lfc_sfc",\
                       "stp", "scp", "stp_srh0to500"]
    

    def __init__(self, xarr):
        ''' Wofs object will contain a geodataframe of attributes and an xarray of predictors.
            @gdf is the geodataframe -- exclude for now because we might not need
            @xarr is the xarray

        '''


        #self.gdf = gdf
        self.xarr = xarr

        return 


    @classmethod
    def preprocess_wofs(cls, specs, grid, wofs_path, wofs_files):
        ''' Handles the WoFS preprocessing--like the factory/blueprint method for the WoFS side of things.
            @specs is the current ForecastSpecs object
            @grid is the current Grid object (should be current WoFS grid
            @wofs_path is the path to the wofs files 
            @wofs_files is a list of wofs files that are considered
        '''

        #Get the wofs fields and methods from text files (set in the config.py file)
        wofs_fields = np.genfromtxt(c.wofs_fields_file, dtype='str') 
        wofs_methods = np.genfromtxt(c.wofs_methods_file, dtype='str') 
         
        
        #First, obtain the list of WoFS_Agg objects for all/all standard variables 
        
        temporal_agg_list = WoFS_Agg.create_wofs_agg_list(wofs_fields, wofs_methods, specs, grid,\
                                wofs_path, wofs_files)

        #print (temporal_agg_list) 
        wofs_xr = Wofs.list_to_xr(temporal_agg_list) 

        #Create new wofs object that holds the xarray 
        wofs_obj = Wofs(wofs_xr) 

        return wofs_obj


    @staticmethod 
    def list_to_xr(obj_list):
        '''Converts list of WoFS_Agg objects to an xarray with dimensions
            (nY, nX)
            @obj_list is a list of WoFS_Agg objects that will be used to 
                create the xarray
        '''

        nY = obj_list[0].ny #number of y points
        nX = obj_list[0].nx #number of x points 

        #Create new x array dataset with dimensions (nY, nX)
        new_xr = xr.Dataset(data_vars=None, coords={"y": (range(nY)), "x": (range(nX))})

        for obj in obj_list: 
            mlName = obj.ml_var_name
            new_xr[mlName] = (["y", "x"], obj.agg_grid)
             

        return new_xr


class WoFS_Agg: 
    '''
        WoFS_Agg handles the temporal aggregation of wofs files 

    '''

    def __init__(self, wofs_var_name, ml_var_name, mem_index, \
                    filepath, filenames, method, nx, ny, grid_time_list, agg_grid,
                    threshold, legacy_filenames):
        '''
            @wofs_var_name is the name of the field as represented in wofs file
            @ml_var_name is the name of the field as represented in the ML 
            @mem_index is an integer corresponding to what member we care about
                #0 is first member, 1 is second, 2 is third, etc. 
                #-1 means we care about the ensemble mean 
                #-2 means we care about the number of members exceeding a threshold
                #(@threshold) 
            @filepath is the path to the wofs files 
            @filenames is the list of wofs filenames (no path) 
            @method is the temporal aggregation method (e.g., "min", "max")
            @nx is the number of x wofs points
            @ny is the number of y wofs points 
            @grid_time_list is the list of initial wofs grids at the relevant time
                [list of np array (ny,nx) at each time over the period]
            @agg_grid is the time-aggregated grid for the given variable  
            @threshold is the threshold used to compute threshold exceedance 
                (if applicable; for most variables, probably won't be applicable) 
            @legacy_filenames is the list of filenames with the "ALL" replaced with, 
                e.g., ENV, ENS, SWT, etc., as appropriate, based on the legacy
                file naming convention. 
        '''

        self.wofs_var_name = wofs_var_name
        self.ml_var_name = ml_var_name 
        self.mem_index = mem_index
        self.filepath = filepath 
        self.filenames = filenames
        self.method = method 
        self.nx = nx 
        self.ny = ny 
        self.grid_time_list = grid_time_list
        self.agg_grid = agg_grid  
        self.threshold = threshold
        self.legacy_filenames = legacy_filenames

        return 


    @classmethod
    def create_wofs_agg_list(cls, wofsFields, wofsMethods, specsObj, gridObj, \
                                wofsPath, wofsFilenames):
        ''' Creates/returns a list of (complete) WoFS_Agg objects (for each variable) 
            @wofsFields is a list of wofs fields (from text file) 
            @wofsMethods is a list of computation methods to apply to the wofs fields
            @specsObj is the ForecastSpecs object for the current situation
            @gridObj is the Grid object for the current situation. 
            @wofsPath is the path to the wofs files 
            @wofsFilenames is the list of wofs filenames (without path) 
        '''

        agg_files = [] #Will hold the list of time-aggregated WoFS_Agg objects

        #Used to initialize WoFS_Agg object
        initial_grid = np.zeros((gridObj.ny, gridObj.nx))

        initial_grid_list = [initial_grid for g in wofsFilenames] 
            

        #Create an object for each variable 
        for v in range(len(wofsFields)):
        
            #set ml variable name
            ml_variable = wofsFields[v] 

            #Set wofs variable name and member index from ml variable name 
            wofs_variable, member_index, threshold_value = WoFS_Agg.find_var_attributes(ml_variable) 

            #Set computational method to be used for aggregation
            wofs_method = wofsMethods[v] 

            #Get list of legacy filenames -- e.g., replace ALL with ENS, ENV, SVR, etc. 
            #as was done in the old naming convention. 
            legacyFilenames = WoFS_Agg.get_legacy_filenames(wofs_variable, wofsFilenames) 
    
            #Create an initial WoFS_Agg object
            wofs_agg_obj = WoFS_Agg(wofs_variable, ml_variable, member_index, wofsPath, wofsFilenames,\
                                wofs_method, gridObj.nx, gridObj.ny, initial_grid_list, initial_grid, \
                                threshold_value, legacyFilenames)


            #Set the object's grid time list 
            wofs_agg_obj.set_grid_time_list() 

            #Set the object's temporal aggregation
            wofs_agg_obj.set_temporal_aggregation()

            #Add wofs_agg_obj to list 
            agg_files.append(wofs_agg_obj) 


        return agg_files

    def set_temporal_aggregation(self):
        ''' Sets an instance's agg_grid attribute based on the other attributes of 
                the instance.'''


        #Convert list to np array 
        #Dimensions will be (number of times, number of members, y points, x points)
        time_array = WoFS_Agg.time_list_to_array(self.grid_time_list) 

        #How to proceed will depend on the instance's member index and method (i.e.,
        #agg strategy. 

      
        #If mem_index is -1, we will take a time aggregation over the individual 
        #members, followed by an ensemble mean at each grid point 
        #(This will be for the majority of variables) 
        if (self.mem_index == -1):
            
            #Take time aggregation depending on the instance's method 
            if (self.method == "max"):
                time_agg = np.amax(time_array, axis=0)
            elif (self.method == "min"):
                time_agg = np.amin(time_array, axis=0) 


            #Take ensemble mean 
            time_agg = np.mean(time_agg, axis=0) 

            #Get rid of the masking element of the array    
            time_agg = np.ma.getdata(time_agg) 


        #In this case, we will take the aggregation of the individual member 
        #indicated by the mem_index 
        elif (self.mem_index >= 0):
            
            #First, extract the relevant member from time_array
            member_array = time_array[:,self.mem_index,:,:]

            #Take the time aggregation (depending on the instance's method) 
            #of the single member 
            if (self.method == "max"):
                time_agg = np.amax(member_array, axis=0) 
            elif (self.method == "min"):
                time_agg = np.amin(member_array, axis=0) 

        #In this case, we'd want to first apply the threshold to get a 
        #"probability" at each point and time and take the max/min of 
        #this probability over time.             
        elif (self.mem_index == -2):
    

            #Apply the threshold to each member
            probability_array = np.where(time_array >= self.threshold, 1/Wofs.N_MEMBERS, 0)

            #Sum over the ensemble members
            probability_array = np.sum(probability_array, axis=1) 

            #Take the max/min over time
            if (self.method == "max"):
                time_agg = np.amax(probability_array, axis=0)
            elif (self.method == "min"):
                time_agg = np.amin(probability_array, axis=0)


        #Get rid of nans and inf values -- NOTE: Moved this to where we 
        #first read in the WoFS data. 
        #np.nan_to_num(time_agg, copy=False, nan=c.nan_replace_value,\
        #        posinf=c.nan_replace_value, neginf=c.nan_replace_value)

        #Convert to float32 
        time_agg = np.float32(time_agg)

        #Update the instance's attribute 
        self.agg_grid = time_agg


        return 


    def set_grid_time_list(self):
        #Reads in the list of gridded data  over time from the relevant files

        data_list = self.populate_data_list()

        #Update the instance
        self.grid_time_list = data_list


        return 


    def populate_data_list(self):
        #Read in the data from the file at each time step 

        var_data_list = [] 
        for n in range(len(self.filenames)):

            full_filename = "%s/%s" %(self.filepath, self.filenames[n])
            full_legacy_filename = "%s/%s" %(self.filepath, self.legacy_filenames[n])

            if (c.use_ALL_files == False):
                try: 
                    data = nc.Dataset(full_legacy_filename) 
                except FileNotFoundError:
                    try: 
                        #Try to load the ALL file if we can if the
                        #legacy file isn't there 
                        data = nc.Dataset(full_filename) 
                    except FileNotFoundError:
                        print ("Neither %s nor %s found. Moving on." \
                                %(full_legacy_filename, full_filename))
                        continue 


            else: #if we are using the ALL files 
                try: 
                    data = nc.Dataset(full_filename)
                except FileNotFoundError:
                    print ("%s not found. Moving on." %full_filename) 
                    continue 


            #Extract relevant variable 
            var_data = data[self.wofs_var_name][:]

            #Get rid of nans and inf values
            np.nan_to_num(var_data, copy=False, nan=c.nan_replace_value,\
                posinf=c.nan_replace_value, neginf=c.nan_replace_value)
            
            #Append to list 
            var_data_list.append(var_data) 


        return var_data_list 


    @staticmethod
    def time_list_to_array(time_list):
        '''Converts the list of wofs data (nm, ny, nx) to an array 
            (nt, nm, ny, nx), where nt is the number of times
            @time_list is the list of wofs data at successive lead times 

        '''

        nt = len(time_list)
        shape_of_list_elements = np.shape(time_list[0]) 
        nnm = shape_of_list_elements[0]
        nny = shape_of_list_elements[1]
        nnx = shape_of_list_elements[2] 

        out_array = np.zeros((nt, nnm, nny, nnx))

        for t in range(nt):
            out_array[t, :, :,:] = time_list[t] 
            


        return out_array

    @staticmethod
    def get_legacy_filenames(wofs_var_name, wofs_files_list):
        ''' Returns a list of wofs filenames in legacy format. 
            e.g., replacing the "ALL" with "ENS" or "ENV", etc. 
            @wofs_var_name is the string wofs variable name for 
                the current variable
            @wofs_files_list is the list of wofs filenames
                (with the ALL convention) 
        '''
    
        #Default
        new_names = copy.deepcopy(wofs_files_list) 

        if (wofs_var_name in Wofs.ENS_VARS):
            new_names = [s.replace("ALL", "ENS") for s in wofs_files_list] 

        elif (wofs_var_name in Wofs.ENV_VARS):
            new_names = [s.replace("ALL", "ENV") for s in wofs_files_list]

        elif (wofs_var_name in Wofs.SVR_VARS):
            new_names = [s.replace("ALL", "SVR") for s in wofs_files_list] 
            

        return new_names


    @staticmethod
    def find_var_attributes(ml_var):
        #Returns the appropriate wofs variable name, member index, and
        #threshold value given the incoming ml variable name (@ml_var) 

        #Default threshold
        #threshold_val is the threshold we use for finding the 
        #number of wofs members meeting or exceeding this value; 
        #only used for a small number of variables, so most of
        #the time, will be set to -999.0
        threshold_val = -999.0
        

        if (ml_var == "m1_uh_2to5"):
            wofs_var = "uh_2to5"
            mem_index = 0
        elif (ml_var == "m2_uh_2to5"):
            wofs_var = "uh_2to5"
            mem_index = 1
        elif (ml_var == "m3_uh_2to5"):
            wofs_var = "uh_2to5"
            mem_index = 2
        elif (ml_var == "m4_uh_2to5"):
            wofs_var = "uh_2to5"
            mem_index = 3
        elif (ml_var == "m5_uh_2to5"):
            wofs_var = "uh_2to5"
            mem_index = 4
        elif (ml_var == "m6_uh_2to5"):
            wofs_var = "uh_2to5"
            mem_index = 5
        elif (ml_var == "m7_uh_2to5"):
            wofs_var = "uh_2to5"
            mem_index = 6
        elif (ml_var == "m8_uh_2to5"):
            wofs_var = "uh_2to5"
            mem_index = 7
        elif (ml_var == "m9_uh_2to5"):
            wofs_var = "uh_2to5"
            mem_index = 8
        elif (ml_var == "m10_uh_2to5"):
            wofs_var = "uh_2to5"
            mem_index = 9
        elif (ml_var == "m11_uh_2to5"):
            wofs_var = "uh_2to5"
            mem_index = 10
        elif (ml_var == "m12_uh_2to5"):
            wofs_var = "uh_2to5"
            mem_index = 11
        elif (ml_var == "m13_uh_2to5"):
            wofs_var = "uh_2to5"
            mem_index = 12
        elif (ml_var == "m14_uh_2to5"):
            wofs_var = "uh_2to5"
            mem_index = 13
        elif (ml_var == "m15_uh_2to5"):
            wofs_var = "uh_2to5"
            mem_index = 14
        elif (ml_var == "m16_uh_2to5"):
            wofs_var = "uh_2to5"
            mem_index = 15
        elif (ml_var == "m17_uh_2to5"):
            wofs_var = "uh_2to5"
            mem_index = 16
        elif (ml_var == "m18_uh_2to5"):
            wofs_var = "uh_2to5"
            mem_index = 17
        elif (ml_var == "prob40dbz"):
            wofs_var = "comp_dz"
            mem_index = -2
            threshold_val = Wofs.DBZ_THRESH

        #Default
        else: 
            wofs_var = copy.deepcopy(ml_var)     
            mem_index = -1       

        return wofs_var, mem_index, threshold_val



class Grid: 
    '''Handles the (wofs) grid attributes.'''


    def __init__(self, ny, nx, lats, lons, tlat1, tlat2, stlon, sw_lat, ne_lat, sw_lon, ne_lon, ypts, xpts): 
        ''' @ny is number of y grid points,
            @nx is number of x grid points,
            @lats is list of latitudes 
            @lons is list of longitudes
            @tlat1 is true latitude 1
            @tlat2 is true latitude 2
            @stlon is standard longitude
            @sw_lat is the southwest corner latitude
            @ne_lat is the northeast corner latitude
            @sw_lon is the southwest corner longitude
            @ne_lon is the northeast corner longitude 
            @ypts is a 2-d array of y values (ny,nx) 
            @xpts is a 2-d array of x values (ny,nx) 
        '''

        self.ny = ny
        self.nx = nx
        self.lats = lats
        self.lons = lons
        self.tlat1 = tlat1
        self.tlat2 = tlat2
        self.stlon = stlon
        self.sw_lat = sw_lat
        self.ne_lat = ne_lat
        self.sw_lon = sw_lon
        self.ne_lon = ne_lon
        self.ypts = ypts
        self.xpts = xpts 


    @classmethod
    def create_wofs_grid(cls, wofs_path, wofs_file): 
        '''Creates a Grid object from a wofs path and wofs file.'''

        full_wofs_file = "%s/%s" %(wofs_path, wofs_file)

        #Get legacy file
        legacy_fnames = WoFS_Agg.get_legacy_filenames("mslp", [wofs_file])
        legacy_fname = legacy_fnames[0]

        full_legacy_wofs_file = "%s/%s" %(wofs_path, legacy_fname)

        #Add capabilities to account for ALL or legacy file names
        if (c.use_ALL_files == True):
            try: 
                ds = nc.Dataset(full_wofs_file)
            except FileNotFoundError:
                try: 
                    ds = nc.Dataset(full_legacy_wofs_file)
                except:
                    print ("Neither %s nor %s found" %(full_wofs_file, full_legacy_wofs_file))
                    quit()


        else: 

            try: 
                ds = nc.Dataset(full_legacy_wofs_file)
            except FileNotFoundError:
                try: 
                    ds = nc.Dataset(full_wofs_file) 
                except:
                    print ("Neither %s nor %s found" \
                            %(full_legacy_wofs_file, full_wofs_file))
                    quit() 


        ny = int(ds.__dict__['ny'])
        nx = int(ds.__dict__['nx'])


        wofsLats = ds['xlat'][:]
        wofsLons = ds['xlon'][:]

        Tlat1 = ds.__dict__['TRUELAT1']
        Tlat2 = ds.__dict__['TRUELAT2']
        Stlon = ds.__dict__['STAND_LON']

        SW_lat = wofsLats[0,0]
        NE_lat = wofsLats[-1,-1]
        SW_lon = wofsLons[0,0]
        NE_lon = wofsLons[-1,-1]

        #Find arrays of x and y points 
        xArr, yArr = Grid.get_xy_points(ny, nx) 

        #Create new wofs Grid object 
        wofs_grid = Grid(ny, nx, wofsLats, wofsLons, Tlat1, Tlat2, Stlon, SW_lat, NE_lat, SW_lon, NE_lon, yArr, xArr)  

        return wofs_grid

    @staticmethod 
    def get_xy_points(num_y, num_x):
        '''Gets a grid of x and y points given the 
            number of points in the y direction (@num_y)
            and number of points in the x direction (@num_x) 
        '''

        x_arr = np.zeros((num_y, num_x)) 
        y_arr = np.zeros((num_y, num_x)) 
    
        for x in range(num_x):
            for y in range(num_y): 
                x_arr[y,x] = x
                y_arr[y,x] = y 
                


        return x_arr, y_arr
        


class PS_WoFS:
    ''' Handles the remapping of PS objects onto WoFS grid'''

    
    #Dataframe columns relevant to each hazard 
    HAIL_COLS = ["wofs_j", "wofs_i", "t", "hail_prob", "age", "fourteen_change_hail", "thirty_change_hail" ]
    TORN_COLS = ["wofs_j", "wofs_i", "t", "torn_prob", "age", "fourteen_change_torn", "thirty_change_torn"]
    WIND_COLS = ["wofs_j", "wofs_i", "t", "wind_prob", "age", "fourteen_change_wind", "thirty_change_wind"] 
   

    #Dictionaries that handle the hazard-specific changing of variable keywords 
    #to geodataframes (for remapping to wofs grid) 


    HAIL_RENAME_DICT = {'hail_prob':'prob', 'fourteen_change_hail':'fourteen_change',\
                                     'thirty_change_hail':'thirty_change'}

    WIND_RENAME_DICT = {'wind_prob':'prob', 'fourteen_change_wind':'fourteen_change',\
                                     'thirty_change_wind':'thirty_change'}

    TORN_RENAME_DICT = {'torn_prob':'prob', 'fourteen_change_torn':'fourteen_change',\
                                     'thirty_change_torn':'thirty_change'}



    def __init__(self, nx, ny, hazard, probs, smoothed_probs, ages, lead_times, \
                    fourteen_change, thirty_change):

        #NOTE: Might potentially add storm motion east/south to this as well 
                #(so it could be used with TORP)

        self.nx = nx 
        self.ny = ny
        self.hazard = hazard
        self.probs = probs
        self.smoothed_probs = smoothed_probs
        self.ages = ages
        self.lead_times = lead_times
        self.fourteen_change = fourteen_change
        self.thirty_change = thirty_change 

        return 


    @classmethod
    def new_PS_WoFS_from_Grid(cls, hazard_name, gridObj):
        '''Creates new PS_WoFS object from Grid object
            and hazard name. 
            @hazard_name is the string hazard name
            @gridObj is the Grid object (e.g., corresponding to WoFS grid) 
            
        '''

        use_nx = gridObj.nx
        use_ny = gridObj.ny
        
        #Initialize these fields to 0s 
        initial_probs = np.zeros((use_ny, use_nx))
        initial_smoothed_probs = np.zeros((use_ny, use_nx))
        initial_fourteens = np.zeros((use_ny, use_nx))
        initial_thirtys = np.zeros((use_ny, use_nx))
        
        #Initialize these fields to -1s
        initial_ages = np.ones((use_ny, use_nx))*-1
        initial_leads = np.ones((use_ny, use_nx))*-1
        
        #Create/return new PS_WoFS object 
        new_object = PS_WoFS(use_nx, use_ny, hazard_name, initial_probs, \
                    initial_smoothed_probs, initial_ages, initial_leads,\
                    initial_fourteens, initial_thirtys)

        return new_object

    @classmethod
    def filter_and_rename_gdf(cls, haz_name, gdf_to_change):
        ''' 
            Returns a geodataframe that is only relevant for the given hazard. 
            i.e., removes irrelevant columns and renames the relevant ones to 
            standardized names 
            @haz_name is the string hazard name. 
            @gdf_to_change is the geodataframe that needs to be changed
        '''

        if (haz_name == "hail"):
            haz_subset = gdf_to_change[cls.HAIL_COLS]
        #Rename the columns 
            haz_subset.rename(columns = cls.HAIL_RENAME_DICT, inplace=True)

        elif (haz_name == "wind"):
            haz_subset = gdf_to_change[cls.WIND_COLS]
            haz_subset.rename(columns = cls.WIND_RENAME_DICT, inplace=True) 
    
        elif (haz_name == "tornado"):
            haz_subset = gdf_to_change[cls.TORN_COLS]
            haz_subset.rename(columns = cls.TORN_RENAME_DICT, inplace=True) 


        return haz_subset 


    def update(self, points_to_change, geodf):
        '''
            Updates the instance based on a set of points to change (@points_to_change)
            and an incoming geodataframe of PS examples

            How do we want to assign a value? Want all values coming from the same storm
            1. Highest Prob
            2. Greatest 14-min (positive) change
            3. Greatest 30-min (positive) change
            4. Oldest storm
            5. Smallest extrapolation 

        '''
        #First need to filter/rename the columns on geodataframe 
        hazard_gdf = PS_WoFS.filter_and_rename_gdf(self.hazard, geodf) 

        #Apply the probability threshold 
        hazard_gdf = PS_WoFS.threshold_probability(hazard_gdf, c.ps_thresh)
    
        #Do the assignments/updates -- do point by point
        for l in range(len(points_to_change)): 
            y = points_to_change['wofs_j'].iloc[l]
            x = points_to_change['wofs_i'].iloc[l]

            df_subset = hazard_gdf.loc[(hazard_gdf['wofs_j'] == y) & (hazard_gdf['wofs_i'] == x)]
            df_subset_sorted = df_subset.sort_values(['prob', 'fourteen_change', 'thirty_change', 'age', 't'], \
                                        ascending=[False, False, False, False, True])

            if (len(df_subset_sorted) > 0):
                maxValue = df_subset_sorted.iloc[0,:]
            
                #Update the object. 
                #How do we want to assign a value? Want all values coming from the same storm
                #1. Highest Prob
                #2. Greatest 14-min (positive) change
                #3. Greatest 30-min (positive) change
                #4. Oldest storm
                #5. Smallest extrapolation 
                self.probs[y,x] = maxValue['prob']
                self.ages[y,x] = maxValue['age'] 
                self.lead_times[y,x] = maxValue['t']
                self.fourteen_change[y,x] = maxValue['fourteen_change'] 
                self.thirty_change[y,x] = maxValue['thirty_change'] 

        #Next, at the end we need to compute the smoothed prob field 
        self.update_smoothed_probs()


        return 

    def update_smoothed_probs(self):
        ''' Updates the instance's smoothed_probs attribute based on the probs attribute.
            Applies 2d Gaussian kernel density function.
        '''

        #NOTE: Here sigma=3 is hardcoded. i.e., spatial smoothing parameter is 9km. (3x3km grid spacing) 
        smoothed_probs = gaussian_filter(self.probs, sigma=3, order=0, mode='constant', truncate=3.5) 

        self.smoothed_probs = smoothed_probs

        return 
        


    @staticmethod
    def threshold_probability(incoming_gdf, probThresh):
        '''
        Applies a probability threshold such that only rows in the @incoming_gdf
        with 'prob' greater than or equal to @probThresh are retained. 
        '''
        return incoming_gdf.loc[incoming_gdf['prob'] >= probThresh]



class PS:
    '''Handles the ProbSevere forecasts/processing'''


    #Class constants 

    #Variables we will take from older probSevere files that will help us
    #construct current predictors 
    HISTORICAL_VARIABLES = ["id", "hail_prob", "torn_prob", "wind_prob", "age"]

    #List of variables to extract from current probsevere files 
    CURR_PS_VARIABLES = ["id", "hail_prob", "torn_prob", "wind_prob", "east_motion", "south_motion", "points"]

    #Buffer to add around wofs points in m. 2.15km guarantees that we cover the full grid cell
    WOFS_BUFFER = 2.15*10**3 

    #Final PS variable order 
    FINAL_PS_VAR_ORDER = ['raw_probs', 'smooth_probs', 'leads', 'ages', 'changes14', 'changes30']
    
    #WoFS_PS keys corresponding to Final PS variable order 
    FINAL_ORDER_WOFS_PS_KEYS = ['probs', 'smoothed_probs', 'lead_times', 'ages', 'fourteen_change', 'thirty_change']

    def __init__(self, gdf, xarr):
        ''' @gdf is a geodataframe containing all the relevant predictors
            @xarr is an xarray of all the relevant predictors
        '''

        #TODO: Could add a list of past ProbSevereObject objects, and a list of 
        #current ProbSevereObject objects. 


        self.gdf = gdf
        self.xarr = xarr
        

        return 

    @classmethod
    def preprocess_ps(cls, grid, specs, ps_path, ps_files):
        ''' Like the "main method"/blueprint method for doing the probSevere preprocessing.
            @grid is the forecast Grid object for the current case
            @specs is the ForecastSpecs object for the current case. 
            @ps_path is the string path to the ProbSevere files
            @ps_files is the list of ProbSevere files, beginning with the most recent and 
                working backward in time. 
            Ultimately creates a PS object with a gdf and xarrray of the relevant predictors 
        '''
        
        #Get PS geodataframe (i.e., for current PS object) 
        ps_gdf = PS.get_ps_gdf(ps_path, ps_files[0])

        #Get a list of unique ids from gdf 
        curr_ids = PS.get_ids_from_gdf(ps_gdf) 

        #Get a dataframe of all past objects (including their IDs, hazard probabilities, and ages) 
        past_ps_df = PS.get_past_ps_df(specs, ps_path, ps_files, curr_ids)

        #Get Wofs geodataframe
        wofs_gdf = PS.get_wofs_gdf(grid) 

        #Add buffer to wofs geodataframe (since wofs grid points are larger than a literal "point") 
        buffered_wofs_gdf = PS.add_gpd_buffer(wofs_gdf, cls.WOFS_BUFFER)

        #Merge the wofs and probSevere geodataframes -- this is how we will eventually "map" 
        #PS points to wofs grid
        merged_gdf = buffered_wofs_gdf.sjoin(ps_gdf, how='inner', predicate='intersects')

        #Do the extrapolation 
        extrapolated_gdf = PS.do_extrapolation(past_ps_df, merged_gdf, specs)

        #Restrict lead times to relevant (e.g., 30-min) period 
        extrapolated_gdf = PS.filter_lead_time(extrapolated_gdf, specs) 

        #Map to wofs grid - obtain a list of PS_WoFS objects, one for each hazard. 
        list_of_ps_wofs = PS.gdf_to_wofs(extrapolated_gdf, grid) 

        #Convert to xarray 
        ps_xr = PS.ps_wofs_list_to_xr(list_of_ps_wofs, grid)

        #Create new PS object -- will hold geodataframe of predictors and xarray 
        ps_object = PS(extrapolated_gdf, ps_xr) 

        return ps_object


    @classmethod
    def ps_wofs_list_to_xr(cls, ps_wofs_list, wofs_grid):
        '''Converts a list of PS_WoFS objects to a single xarray of 
            ProbSevere predictors.
            @ps_wofs_list is a list of PS_WoFS objects, 1 PS_WoFS object per hazard, in order
                of c.final_hazards
            @wofs_grid is a Grid object with the wofs stats 
        '''
        
        nY = wofs_grid.ny 
        nX = wofs_grid.nx 
        

        #Want to create a giant array and then create an xarray from that 
        #FINAL_PS_VAR_ORDER = ['raw_probs', 'smooth_probs', 'leads', 'ages', 'changes14', 'changes30']

        varnames = (["%s_%s" %(h,v) for h in c.final_hazards for v in cls.FINAL_PS_VAR_ORDER])

        nH = len(ps_wofs_list) #should be number of hazards
        nV = len(cls.FINAL_PS_VAR_ORDER) 

        #Create a final array to hold all data, and then create an xarray from that.
        #Want shape: (nY, nX, nV, nH) 

        new_arr = np.zeros((nY, nX, nV, nH))

        for v in range(nV):
            for h in range(nH):
                new_arr[:,:,v,h] = getattr(ps_wofs_list[h], cls.FINAL_ORDER_WOFS_PS_KEYS[v])


    
        #Now create x array -- Maybe make into a general method in future if needed. 
        new_xr = xr.Dataset(data_vars=None, coords={"y": (range(nY)), "x": (range(nX))})
        count = 0
        for h in range(nH):
            for v in range(nV):
                varname = varnames[count]
                new_xr[varname] = (["y", "x"], new_arr[:,:,v,h])
                count += 1

        return new_xr
    
    @staticmethod
    def get_ids_from_gdf(in_gdf):
        '''Returns a list of unique id numbers from the current
            ProbSevere file given the current ProbSevere
            geodataframe (@in_gdf)
        '''

             

        return in_gdf['id'].unique()


    @staticmethod
    def gdf_to_wofs(in_gdf, fcst_grid):
        '''Converts a geodataframe of example probSevere objects/extrapolation points to 
            a list of PS_WoFS objects; these contain the set of gridded PS predictors. 
            @Returns a list of PS_WoFS objects, with one list element for each hazard, in
                order of c.final_hazards (i.e., the order set in the config.py file) 
            @in_gdf is the incoming geodataframe where each row is an example/point --
                it has been filtered to exclude the irrelevant lead times 
            @fcst_grid is the current Grid object (i.e., the WoFS grid in this case) 
        '''
       
        #Obtain list of wofs_points that need to be updated for this case
        wofs_change_points = PS.get_wofs_change_points(in_gdf, fcst_grid) 

 
        #We will create 3 PS_WoFS objects -> 1 for each hazard, which will hold the wofs grids 
        #haz_names = c.final_hazards
        
        #We will create 3 PS_WoFS objects: 1 for each hazard       
        ps_wofs_objects = [] 
 
        for haz in c.final_hazards: 

            #Initialize PS_WoFS object
            ps_wofs = PS_WoFS.new_PS_WoFS_from_Grid(haz, fcst_grid) 
            
            #Update fields with list of points 
            ps_wofs.update(wofs_change_points, in_gdf) 
            
            #append to list 
            ps_wofs_objects.append(ps_wofs) 

        #Return list of PS_WoFS objcts (should be one for each hazard, in order of c.final_hazards) 

        return ps_wofs_objects


    @staticmethod 
    def get_wofs_change_points(ps_geodataframe, wofs_grid_obj):
        '''Obtains the wofs points that need to be changed from the given ps_geodataframe
            @ps_geodataframe is the incoming geodataframe 
            @wofs_grid_obj is the incoming Grid object corresponding to wofs grid '''

        #Number of y, x grid points (for convenience) 
        Ny = wofs_grid_obj.ny
        Nx = wofs_grid_obj.nx 
        

        unique_points = ps_geodataframe.drop_duplicates(subset=['wofs_j', 'wofs_i'], inplace=False, ignore_index=True)

        #Also, we need to filter out the points that are outside of the wofs grid. Only save points between 0 and 299 
        unique_points = unique_points.loc[(unique_points['wofs_j'] >= 0) & (unique_points['wofs_j'] < Ny) & \
                                      (unique_points['wofs_i'] >= 0) & (unique_points['wofs_i'] < Nx)]


        return unique_points  



    @staticmethod
    def filter_lead_time(in_gdf, fcst_specs):
        '''
            Only keeps geodataframe examples that are within the relevant lead 
            times for this 30-min period. 
            @in_gdf is the incoming geodataframe where each row is an example
            @fcst_specs is a ForecastSpecs object for the current case
        '''
   
        #self.ps_lead_time_start = ps_lead_time_start
        #self.ps_lead_time_end
        subset_gdf = in_gdf.loc[(in_gdf['t'] >= fcst_specs.ps_lead_time_start) & \
                        (in_gdf['t'] <= fcst_specs.ps_lead_time_end)]

        #Drop duplicates
        subset_gdf.drop_duplicates(keep='first', inplace=True, ignore_index=True) 

 

        return subset_gdf 


    @staticmethod
    def do_extrapolation(prev_ps_df, curr_gdf, fcst_specs):
        ''' This method creates/returns a "final" geopandas dataframe with all relevant PS 
            attriutes/predictors over the relevant time period for all PS objects in wofs
            domain
            @prev_ps_df is the dataframe of past PS objects
            @curr_gdf is the merged "WoFS/PS geopandas dataframe. 
            @fcst_specs is a ForecastSpecs object for the current situation.
        '''

        #If there are no objects, then there's nothing to apply the extrapolation to
        if (len(curr_gdf) == 0):
            output_gdf = curr_gdf.copy(deep=True)


        else: 

            #Get unique object IDs from current PS file/gdf
            obj_ids = curr_gdf['id'].unique() 
       
            subsets = [] #Will hold the gdfs from individual objects  

            #Loop over unique object IDs -- parallelize eventually?? 
            for obj_id in obj_ids: 

                #Find the gdf entries corresponding to the relevant object
                obj_gdf = curr_gdf.loc[curr_gdf['id']==obj_id]

                past_obj_df = prev_ps_df.loc[prev_ps_df['id']==obj_id]

                #Get object attributes -- hazard probs and storm motion components

                obj_hail_prob = PS.get_object_attribute(obj_gdf, "hail_prob")
                obj_torn_prob = PS.get_object_attribute(obj_gdf, "torn_prob")
                obj_wind_prob = PS.get_object_attribute(obj_gdf, "wind_prob")
                obj_motion_east = PS.get_object_attribute(obj_gdf, "east_motion") 
                obj_motion_south = PS.get_object_attribute(obj_gdf, "south_motion") 

                #Find Storm Age
                obj_age = PS.find_obj_age(obj_id, prev_ps_df) 
                
                #Find 14- and 30-minute changes 
                fourteen_min_change_hail, fourteen_min_change_torn, fourteen_min_change_wind = \
                        PS.find_prob_change(obj_id, prev_ps_df, 14)

                thirty_min_change_hail, thirty_min_change_torn, thirty_min_change_wind = \
                        PS.find_prob_change(obj_id, prev_ps_df, 30) 

                #Get extrapolation points (no adjustable radius) 

                #NOTE: fcst_specs.ps_lead_time_end is essentially the time to which we need to extrapolate. 
                orig_extrap_points = PS.get_extrapolation_points(fcst_specs.ps_lead_time_end,\
                    obj_motion_east, obj_motion_south, c.dx_km, \
                    fcst_specs.adjustable_radii_gridpoint)


                #Now, add the adjustable radii to the set of extrapolation points 
                extrap_points = PS.add_adjustable_radii(orig_extrap_points, fcst_specs, c.dx_km)

                subset_gdf = PS.apply_extrapolation(obj_gdf, extrap_points)

                #Add the probabilities, age, 14-, and 30-minute prob changes  
                attribute_names = ["hail_prob", "torn_prob", "wind_prob", "age",\
                                    "fourteen_change_hail", "fourteen_change_torn", \
                                    "fourteen_change_wind", "thirty_change_hail", \
                                    "thirty_change_torn", "thirty_change_wind"]

                dataToAdd = [obj_hail_prob, obj_torn_prob, obj_wind_prob, obj_age, \
                            fourteen_min_change_hail, fourteen_min_change_torn, \
                            fourteen_min_change_wind, thirty_min_change_hail, \
                            thirty_min_change_torn, thirty_min_change_wind] 

                subset_gdf = PS.add_ps_attributes(subset_gdf, attribute_names, dataToAdd) 

                #Add to subsets list 
                subsets.append(subset_gdf) 

            #Concatenate all subsets
            output_gdf = pd.concat(subsets, axis=0, ignore_index=True) 
    
        return output_gdf 

        
    @staticmethod 
    def add_ps_attributes(in_gdf, var_names, data_to_add):
        ''' Adds new columns of data to an existing geopandas dataframe.
            @in_gdf the existing geopandas dataframe
            @var_names is a list of (string) names of the new variables 
            @data_to_add is a list of the corresponding data to be added. 
        '''

        for d in range(len(var_names)):
            var_name = var_names[d] 
            curr_data = data_to_add[d] 
            
            in_gdf[var_name] = curr_data

        return in_gdf


    @staticmethod
    def apply_extrapolation(in_gdf, in_extrap_points):
        '''Applies the (spatially expanded) extrapolated points to the original geodataframe.
            @in_gdf is an incoming geopandas dataframe of PS/wofs points (here, will generally
            be of a single object)
            @in_extrap_points is a pandas dataframe of relative extrapolation points to be 
            added/applied to the current set of points 
        '''

        parts = [] #Will hold dataframes to concat

        for a in range(len(in_gdf)):
            #print (in_gdf)
            part = in_extrap_points.copy()
            part['wofs_j'] = in_extrap_points['y'] + in_gdf['wofs_j'].iloc[a]
            part['wofs_i'] = in_extrap_points['x'] + in_gdf['wofs_i'].iloc[a]
            part['t'] = in_extrap_points['t']

            parts.append(part)


        #Now concatenate
        out_df = pd.concat(parts, axis=0, ignore_index=True)

        #Return only the columns that matter: 't', 'wofs_j', 'wofs_i'

        return out_df[['wofs_j', 'wofs_i', 't']]


    @staticmethod 
    def add_adjustable_radii(all_pts_df, curr_specs, km_spacing):
        ''' Adds the adjustable radius to the dataframe. i.e., 
            Add in points within a radius -- for each element in all_pts_df 
            Returns an updated version of all_pts_df with these points added.
            @all_pts_df is the dataframe of extrapolation points.
            @curr_specs is the current ForecastSpecs object
        '''


        #Only proceed if there are adjustable radii to add, otherwise skip this
        #and just return what was passed in 
        if (max(curr_specs.adjustable_radii_gridpoint) > 0):

            all_column_names = ['y','x', 't', 'max_xy', 'radius_km']

            patch_coords = []

            for a in range(len(all_pts_df)):
                y = all_pts_df['y'][a]
                x = all_pts_df['x'][a]
                ymax = all_pts_df['max_xy'][a]
                xmax = ymax
                time = all_pts_df['t'][a]
                rradius = all_pts_df['radius_km'][a]


                #Need to add all relative points within circle 

                #Obtain square patch 
                real_ys = np.arange(y-ymax, y+ymax+1)
                real_xs = np.arange(x-xmax, x+xmax+1)

                #Check if points are within radius. 
                patch_xs = []
                patch_ys = []
                patch_inds = []
                for xx in real_xs:
                    for yy in real_ys:
                        x_rad = abs(x-xx)
                        y_rad = abs(y-yy)
                        if ( (math.sqrt(x_rad**2 + y_rad**2)*km_spacing <= rradius) and ((yy,xx) not in patch_coords)):
                            patch_coords.append((yy,xx))
                            #We need y,x,t,max_xy,radius_km
                            patch_inds.append((yy,xx, time, ymax, rradius))
                            new_point = pd.DataFrame(patch_inds, columns=all_column_names)
                            #Append new point to original dataframe 
                            all_pts_df = pd.concat([all_pts_df, new_point], \
                                            names=all_column_names, ignore_index=True, copy=False) 
        
        return all_pts_df 


    @staticmethod 
    def get_extrapolation_points(time, e_motion, s_motion, km_spacing, max_xy):
        '''
        Finds the (relative) wofs points that would be hit by extrapolating a storm object
        over time according to the supplied storm motion vectors. 
    
        Returns a set of (unique) relative coordinates that are "hit" by the extrapolation. 
            E.g., (0,0) is the initial point

        @time is the time in minutes over which to apply the extrapolation
        @e_motion is the eastward storm motion in km/min
        @s_motion is the southward storm motion in km/min
        @km_spacing is the grid spacing of the output (e.g., wofs) grid in km
        @max_xy is a list of max_x/max_y points associated with each time. --i.e., 
            the adjustable radii points 

        '''

        minutes = np.arange(time+1) 

        new_xs = [round((e_motion*m)/(km_spacing)) for m in minutes]
        new_ys = [round((-s_motion*m)/(km_spacing)) for m in minutes]


        #Put these together and add time dimension
        three_d_extrap_pts = [(new_ys[p], new_xs[p], p, max_xy[p]) for p in range(len(new_xs))]

        #Let's make this a pandas dataframe and then extract the unique points from that
        all_pts_df = pd.DataFrame(three_d_extrap_pts, columns=['y','x', 't', 'max_xy'])

        #NOTE: Don't drop the duplicates here. 
        all_pts_df['radius_km'] = all_pts_df['max_xy']*km_spacing
    
        return all_pts_df


    @staticmethod 
    def find_prob_change(object_id, past_ps_df, time):
        '''Finds/returns the change in probabilities over a given time range.
            @object_id is the object id number, 
            @past_ps_df is the dataframe of previous PS objects,
            @time is the time in minutes over which to compute the change  '''


        final = past_ps_df.loc[(past_ps_df['id'] == object_id) & (past_ps_df['age'] == 0)]
        initial = past_ps_df.loc[(past_ps_df['id'] == object_id) & (past_ps_df['age'] == time)]

        if (len(final) > 0):
            final_hail_prob = final['hail_prob'].iloc[0]
            final_torn_prob = final['torn_prob'].iloc[0]
            final_wind_prob = final['wind_prob'].iloc[0]

            if (len(initial) > 0):
                initial_hail_prob = initial['hail_prob'].iloc[0]
                initial_torn_prob = initial['torn_prob'].iloc[0]
                initial_wind_prob = initial['wind_prob'].iloc[0]

                change_hail = final_hail_prob - initial_hail_prob
                change_torn = final_torn_prob - initial_torn_prob
                change_wind = final_wind_prob - initial_wind_prob
            else: #if initial (i.e., older age file) doesn't exist, just set to current probability
                change_hail = final_hail_prob
                change_torn = final_torn_prob
                change_wind = final_wind_prob 

        else: #if final (i.e., current object) doesn't exist, set all probs to 0 ; 
                #shouldn't be the case very often
            change_hail = 0.0 
            change_torn = 0.0
            change_wind = 0.0 


        return change_hail, change_torn, change_wind


    @staticmethod
    def find_obj_age(object_id, previous_ps_df):
        ''' Finds/returns the age of a given object--based on how long the object
            ID number has previously existed.
        '''

        #Need to take the maximum of age columns -- look for the maximum time this
        #storm has been around. 
        possible_ages = previous_ps_df.loc[previous_ps_df['id'] == object_id]
        final_age = max(possible_ages['age'])

        return final_age


    @staticmethod
    def get_object_attribute(object_gdf, var_name):
        ''' Extracts/returns a given piece of information (@var_name) from a 
            geodataframe corresponding to one PS object (@object_gdf) 
        '''
        
        output = object_gdf[var_name].iloc[0]


        return output


    @staticmethod
    def add_gpd_buffer(in_gdf, buffer_dist):
        '''
        Adds a buffer to the points in @in_gdf in meters. 
        Returns new, buffered, geopandas dataframe. 
        @in_gdf: Incoming geopandas dataframe with list of lat/lon points
        @buffer_dist: Buffer distance to be applied, in meters

        '''
       
        #Make a copy of the incoming gdf
        copy_gdf = in_gdf.copy(deep=True)

        #Convert to meters 
        copy_gdf.to_crs("EPSG:32634", inplace=True)

        #Apply buffer 
        copy_gdf.geometry = copy_gdf.geometry.buffer(buffer_dist)

        #Convert back to lat/lon coords
        copy_gdf.to_crs("EPSG:4326", inplace=True) 



        return copy_gdf


    @staticmethod
    def get_wofs_gdf(wofs_grid):
        ''' Obtains/Returns a geodataframe of the wofs grid based 
            on a wofs Grid object (@wofs_grid)'''
        
        points = []
        wofs_i = []
        wofs_j = []
        for j in range(wofs_grid.ny):
            for i in range(wofs_grid.nx):
                pt = Point((wofs_grid.lons[j,i], wofs_grid.lats[j,i]))
                points.append(pt)
                wofs_j.append(j)
                wofs_i.append(i)


        wofs_j = np.array(wofs_j)
        wofs_i = np.array(wofs_i)
        wofs_df_dict = {"wofs_j": wofs_j, "wofs_i": wofs_i}

        gdf_wofs = gpd.GeoDataFrame(data=wofs_df_dict, geometry=points, crs="EPSG:4326") 

        return gdf_wofs


    @staticmethod
    def get_ps_gdf(ps_path, ps_file):
        ''' Obtains probSevere geoDataFrame from current probsevere file (@ps_file)'''

        #Read in the data 
        ps_data = PS.get_ps_data(ps_path, ps_file) 

        #Extract the relevant information from the ps data. 
        #NOTE: 0 for age of 0 since the file is current (although won't matter much for this) 
        ps_df = PS.extract_ps_info(ps_data, 0, c.ps_version, True)
        
        #Find polygons from the set of points from each PS object
        polygons = PS.get_polygons_from_points(ps_df['points']) 

        #Create the gdf
        df_dict = {"hail_prob": ps_df['hail_prob'], "torn_prob": ps_df['torn_prob'], "wind_prob": ps_df['wind_prob'],\
                     "east_motion": ps_df['east_motion'], "south_motion": ps_df['south_motion'], "id": ps_df['id']}

        gdf = gpd.GeoDataFrame(data=df_dict, geometry=polygons, crs="EPSG:4326")
            

        return gdf

    @staticmethod
    def get_polygons_from_points(ps_points):
        ''' Returns a list of polygons from the sets of points associated with each object.
            @ps_points is a pandas dataframe of a series of points 
        '''

        polygons = [] 

        for obj in ps_points: 
            pgon = Polygon(obj[0])
            polygons.append(pgon)

    
        return polygons


    @classmethod
    def get_past_ps_df(cls, specs, ps_path, ps_files, current_ids):
        ''' Returns a dataframe with statistics from past PS files (that will  be relevant
            for our predictors 
            @specs is the ForecastSpecs object for our situation
            @ps_path is the path to the probSevere files
            @ps_files is the list of probSevere files (ordered most recent to oldest)
            @current_ids is a numpy array of current id numbers from the storms in the 
                most recent probSevere file. 

        '''

        #Create new dataframe 
        prev_df = pd.DataFrame(columns = cls.HISTORICAL_VARIABLES)

        for p in range(len(ps_files)):

            ps_file = ps_files[p]
            age = specs.ps_ages[p]

            #Extract the information 
            ps_data = PS.get_ps_data(ps_path, ps_file) 

            if (ps_data != ""):  #i.e., if there's at least one storm object
                #extract historical info
                #False at the end because this is for past/historical data 
                curr_df = PS.extract_ps_info(ps_data, age, c.ps_version, False)
           
                #Maybe filter out the extraneous data here. 
                mask = curr_df['id'].isin(current_ids)
                curr_df = curr_df[mask]

                #Merge dataframe
                if (len(curr_df) > 0):
                    prev_df = pd.concat([prev_df, curr_df], axis=0, ignore_index=True, copy=False)
        


        return prev_df 

    @classmethod
    def extract_ps_info(cls, ps_data, age, ps_version, isCurrent): 
        ''' Extracts information from given set of ps_data (from one ps_file) 
            and stores this information in a pandas dataframe. Ultimately, returns the
            dataframe. 
            @ps_data is an array of probSevere data, 
            @age is the age corresponding to the given probSevere file, 
            @ps_version is the probSevere version (e.g., 2 or 3)
            @isCurrent is boolean: True if we're extracting information relevant to 
                the current ProbSevere file (e.g., storm motion, points, hazard probs, etc.), 
                False if we're extracting information relevant for past/historical probSevere
                (e.g., hazard probs, ids, and ages only)  
        '''

        hail_probs = [] 
        torn_probs = [] 
        wind_probs = [] 
        ids = [] 
        ages = [] 
        points = [] 
        east_motion = [] 
        south_motion = []         

        if (ps_version == 2): 

            if (len(ps_data['features']) > 0):
                for i in ps_data['features']:
                    hail_probs.append(float(i['models']['probhail']['PROB'])/100.)
                    torn_probs.append(float(i['models']['probtor']['PROB'])/100.)
                    wind_probs.append(float(i['models']['probwind']['PROB'])/100.)

                    east_motion.append(float(i['properties']['MOTION_EAST'])*0.06) #multiply by 0.06 to convert to km/min
                    south_motion.append(float(i['properties']['MOTION_SOUTH'])*0.06) #multiply by 0.06 to convert to km/min

                    points.append(i['geometry']['coordinates'])

                    ids.append(i['properties']['ID'])
                    ages.append(age) 

        #TODO: Implement ps version 3 code here
        elif (ps_version == 3):
            pass 

        #if we're dealing with past PS files: 
        if (isCurrent == False): 
            df = pd.DataFrame(list(zip(ids, hail_probs, torn_probs, wind_probs, ages)), columns=cls.HISTORICAL_VARIABLES)

        #if we're dealing with current PS files
        else: 
            df = pd.DataFrame(list(zip(ids, hail_probs, torn_probs, wind_probs, east_motion, south_motion, points)),\
                                columns=cls.CURR_PS_VARIABLES)

        return df


    @staticmethod
    def get_ps_data(ps_path, ps_file):
        ''' Opens ps file given a path and filename.
            Returns the ps data (if file is there) or a blank string
            (i.e., "", if the data is not found. 
        '''

        try: 
            full_fname = "%s/%s" %(ps_path, ps_file) 
            f = open (full_fname) 
            data = json.load(f)   
        
        except FileNotFoundError:
            print ("%s not found. Adding as if it had no information" %full_fname)
            data = ""

        except json.decoder.JSONDecodeError:
            print ("%s Extra data in file. Proceeding as if it had no information." %full_fname)
            data = ""

        return data

class ForecastSpecs: 

    '''Class to handle/store the forecast specifications.'''

    def __init__(self, start_valid, end_valid, start_valid_dt, end_valid_dt, \
                    wofs_init_time, wofs_init_time_dt, forecast_window, ps_init_time,\
                    ps_lead_time_start, ps_lead_time_end, ps_init_time_dt, ps_ages,\

                    adjustable_radii_gridpoint, allFields, allMethods, singlePtFields,\
                    before_00z_date):

        ''' @start valid is the start of the forecast valid period (4-character string)
            @end_valid is the end of the forecast valid period (4-character string) 
            @start_valid_dt is the start of the forecast valid period in datetime form
                (i.e., datetime object) 
            @end_valid_dt is the end of the forecast valid period in datetime form 
                (i.e., datetime object) 
            @wofs_init_time is the 4-character string of the wofs initialization time
            @wofs_init_time_dt is the wofs initialization time in datetime form
                (i.e., datetime object) 
            @forecast_window is an integer corresponding to the size of the forecast
                valid period (in minutes) 
            @ps_init_time is the 4-character string of the PS initialization time 
            @ps_lead_time_start is an integer/floating point showing the time (in minutes)
                between the PS initialization time and the start of the forecast
                valid period. 
            @ps_lead_time_end is an integer/floating point showing the time (in minutes) 
                between the PS initialization time and the end of the forecast valid
                period. 
            @ps_init_time_dt is the probSevere initialization time in datetime form
                (i.e., datetime object) 

            @ps_ages is a list of (potential) probSevere ages (in minutes; relative to
                the most recent ProbSevere file) based on the probSevere input files 

            @adjustable_radii_gridpoint is an array of radii (in grid points) showing 
                how much extrapolation should be done at each extrapolation time from
                PS file initiation time to the maximum extrapolation time (set in config
                file)

            @allFields is a list of all predictor fields (ml name notation)
            @allMethods is a list of preprocessing methods corresponding to
                allFields (e.g., max, min, abs, minbut) 
            @singlePtFields is a list of the single point fields (ml name notation) 
            @before_00z_date is the 8-character date string (YYYMMDD) corresponding
                to the date of the forecast before 00Z

        '''

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
        
        self.allFields = allFields
        self.allMethods = allMethods
        self.singlePtFields = singlePtFields

        self.before_00z_date = before_00z_date

        return 


    @classmethod
    
    def create_forecast_specs(cls, ps_files, wofs_files, allFieldsFile, \
                allMethodsFile, singlePtFile):
        '''Blueprint method for creating a ForecastSpecs object based on the list of
            PS files (@ps_files) and the list of wofs files (@wofs_files)  
            @allFieldsFile is the name of the file containing the list of 
                all predictor fields (ml name format) 
            @allMethodsFile is the name of the file containing the list of 
                preprocessing methods 
            @singlePtFile is the name of the file containing the list of predictors
                that will only be taken at a single point (the point of prediction);
                i.e., predictors for which no convolutions will be done. 
        '''

        #Find start/end valid and wofs initialization time from wofs files 
        start_valid, start_valid_date = ForecastSpecs.find_date_time_from_wofs(wofs_files[0], "forecast")
        end_valid, end_valid_date = ForecastSpecs.find_date_time_from_wofs(wofs_files[-1], "forecast") 
        wofs_init_time, wofs_init_date = ForecastSpecs.find_date_time_from_wofs(wofs_files[0], "init") 

        #Obtain datetime versions of the above (i.e., start_valid, end_valid, wofs_init_time, etc.) 
        start_valid_dt = ForecastSpecs.str_to_dattime(start_valid, start_valid_date) 
        end_valid_dt = ForecastSpecs.str_to_dattime(end_valid, end_valid_date) 

        wofs_init_time_dt = ForecastSpecs.str_to_dattime(wofs_init_time, wofs_init_date) 

        



        #TODO: Yes, I think there's a bug in this now. 
        #Get the date before 00z -- Like our UseDate in previous script iterations
        date_before_00z = ForecastSpecs.get_date_before_00z(wofs_init_time_dt, c.next_day_inits)

        #Find the length of the forecast time window based on the start_valid_dt and end_valid_dt
        #datetime objects 

        valid_window = ForecastSpecs.subtract_dt(end_valid_dt, start_valid_dt, True) 

        #Find PS init time from the first (most recent) PS file 
        ps_init_time, ps_init_date = ForecastSpecs.find_ps_date_time(ps_files[0], c.ps_version)

        #Obtain datetime objects
        ps_init_time_dt = ForecastSpecs.str_to_dattime(ps_init_time, ps_init_date) 

        #Find PS lead time for start of valid period (in minutes) 
        #based on PS initialization time and start of the valid period 
        ps_start_lead_time = ForecastSpecs.subtract_dt(start_valid_dt, ps_init_time_dt, True) 
        
        #Find PS lead time for end of valid period (in minutes) 
        #based on PS initailization time and end of the valid period
        ps_end_lead_time = ForecastSpecs.subtract_dt(end_valid_dt, ps_init_time_dt, True) 

        #Find the ages associated with the different ps files 
        ps_ages = ForecastSpecs.find_ps_ages(ps_files)

        #Find array of adjustable radii (in grid points) at each extrapolation time/i.e., 
        #how much the radius should be at each extrapolation time. 
        #NOTE: We limit the size of "adjustable_radii_gridpoint" to the size of the ps_end_lead_time--
        #there's no point in taking the extrapolation out farther than that. 
        #We still need c.max_ps_extrap_time though so that the adjustable radii are applied consistently; 
        #i.e., c.max_radius corresponds to the same (end) time for all forecasts. 
        adjustable_radii_gridpoint = ForecastSpecs.find_adjustable_radii(c.min_radius, c.max_radius,\
                                        c.dx_km, ps_end_lead_time, c.max_ps_extrap_time)


        #Read in the all fields, all methods, and single point files
        #allFieldsFile, #allMethodsFile, singlePtFile
        all_fields = np.genfromtxt(allFieldsFile, dtype='str')
        all_methods = np.genfromtxt(allMethodsFile, dtype='str') 
        single_points = np.genfromtxt(singlePtFile, dtype='str') 
    

        #Create ForecastSpecs object  

        new_specs = ForecastSpecs(start_valid, end_valid, start_valid_dt, end_valid_dt, wofs_init_time, \
                            wofs_init_time_dt, valid_window, ps_init_time, ps_start_lead_time, ps_end_lead_time,\
                            ps_init_time_dt, ps_ages, adjustable_radii_gridpoint,\
                            all_fields, all_methods, single_points, date_before_00z) 

        return new_specs


    @staticmethod 
    def get_date_before_00z(wofsInitTimeDT, nextDayWofsInits):
        '''Returns the 8 character string (YYYYMMDD) corresponding to the forecast
            date before 00z
            @wofsInitTimeDT is the datetime object corresponding to the wofs 
                intitialization time 
            @nextDayWofsInits is the list of next-day wofs initialization times 
                (i.e., wofs initialization times at and after 00z). These will
                be used to determine if the forecast period is after 00z. 
        '''


        #Get the 8-character string from date time object 
        date_string, time_string = ForecastSpecs.dattime_to_str(wofsInitTimeDT)

        #If this string is in the nextDayWoFSinits, then we need to find the
        #string corresponding to the day before. 

        if (time_string in nextDayWofsInits):
                
            #Get datetime object corresponding to one day before
            day_before_dt = wofsInitTimeDT - timedelta(days=1)
            
            #Get updated date_string
            date_string, time_string = ForecastSpecs.dattime_to_str(day_before_dt) 


        return date_string


    @staticmethod
    def find_adjustable_radii(radius_min, radius_max, km_grid_spacing, ps_end_lead, max_extrap_time):
        '''Finds the adjustable radii at each extrapolation time. Returns an array of radii 
            at each 1-min of extrapolation time. 
            @radius_min is the minimum radius (at time 0; generally set in the config file)
            @radius_max is the maximum radius (at max_extrap_time; generally set in the config file) 
            @km_grid_spacing is the grid spacing in km
            @max_extrap_time is the maximum amount of time to do the extrapolation 
        '''

        adjustable_radii_km = np.linspace(radius_min, radius_max, int(max_extrap_time)+1)
        adjustable_radii = [math.ceil((r - 1.5)/km_grid_spacing) for r in adjustable_radii_km]

        #Filter the adjustable radii by time -- only extrapolate until the end of the ps lead time;
        #Any further extrapolation is unnecessary. 
        adjustable_radii = adjustable_radii[0:int(ps_end_lead)+1]

        return adjustable_radii


    @staticmethod
    def datetime_from_ps(ps_file):
        ''' Creates/returns a datetime object from probsevere file'''

        time, date = ForecastSpecs.find_ps_date_time(ps_file, c.ps_version)
        dt_obj = ForecastSpecs.str_to_dattime(time, date) 

        return dt_obj


    @staticmethod
    def find_ps_ages(ps_files): 
        ''' Finds/returns an array of ages (in minutes) of the various PS files'''

        ages = [] 

        first_ps_file = ps_files[0]
        first_ps_dt = ForecastSpecs.datetime_from_ps(first_ps_file) 

        for p in range(len(ps_files)):
            curr_ps_file = ps_files[p]
            
            #Get datetime object 
            curr_dt = ForecastSpecs.datetime_from_ps(curr_ps_file)

            #Find the difference between the current dt and the first_ps_dt in minutes 
            diff = ForecastSpecs.subtract_dt(first_ps_dt, curr_dt, True) 

            #append to ages array 

            ages.append(diff) 


        return ages

    @staticmethod 
    def timedelta_to_min(in_dt):
        '''Converts the incoming timedelta object (@in_dt) to minutes.'''


        minutes = int(in_dt.total_seconds()/60)

        return minutes


    @staticmethod 
    def subtract_dt(dt1, dt2, inMinutes):
        ''' Takes dt1 - dt2 and returns the difference (in datetime format)
            @dt1 and @dt2 are both datetime objects 
            @inMinutes is boolean. If True, returns the subtraction in minutes, 
                if false, returns a timedelta object 
        '''
    
        difference = dt1 - dt2

        if (inMinutes == True):
            difference = ForecastSpecs.timedelta_to_min(difference)

        return difference

    @staticmethod 
    def dattime_to_str(in_dt):
        '''Converts incoming datetime object (@in_dt) to 8 character date string
            and 4-character time string
            @Returns 8-character date string (YYYYMMDD) and 4-character time 
                string (HHMM) 
        '''

        new_date_string = in_dt.strftime("%Y%m%d") 
        new_time_string = in_dt.strftime("%H%M")
    

        return new_date_string, new_time_string


    @staticmethod 
    def str_to_dattime(string_time, string_date):
        ''' Converts a string time and string date to a datetime object. 
            Returns the datetime object. 
            @string_time is the 4-character string time (e.g., "0025") 
            @string_date is the 8-character string date (e.g., "20190504") 
        '''

        #Combine the string date and time 
        full_string = "%s%s" %(string_date, string_time) 

        dt_obj = datetime.strptime(full_string, "%Y%m%d%H%M") 

        return dt_obj

    @staticmethod
    def find_date_time_from_wofs(wofs_file, fcst_type):
        '''
        Finds/returns the (string) time and date (e.g., start or end of the forecast valid window) associated with the given WoFS file. 
        @wofs_file is the string of the wofs file 
        @fcst_type is a string telling what type of forecast this is (i.e., which time
            to return): "forecast" means it will return the valid time; "init" means 
            return the initialization time 

        NOTE: The date returned will be the date associated with the valid time, 
        NOT the initialization time. As a result, the date string returned might 
        differ from the date in the wofs file name. 
        '''

        #Need to check if the time is in next_day_times 


        #Split the string based on underscores 
        split_str = wofs_file.split("_") 
        
        #Find initialization time
        initTime = split_str[4] 

        validTime = split_str[5] #In this case, it'll be the 5th element of the wofs file string

        #Have to remove the .nc
        validTime = validTime.split(".")[0]


        date = split_str[3] #date listed in wofs file 

        #Increment the date listed if the init time is in the previous day but the
        #valid time is in the next day -- but only if we're concerned with returning
        #the forecast date/time and not the initialization date/time 
        if ((fcst_type=="forecast") and (validTime in c.next_day_times)\
                 and (initTime not in c.next_day_inits)):
            dt = ForecastSpecs.str_to_dattime(validTime, date)
            dt += timedelta(days=1) 
            date, __ = ForecastSpecs.dattime_to_str(dt) 

        if (fcst_type == "forecast"):
            time = validTime
        elif (fcst_type == "init"):
            time = initTime
        


        return time, date 


    @staticmethod 
    def find_ps_date_time(ps_file, ps_version):
        '''
        Finds/returns the (string) probSevere initialization time/date from the ProbSevere file. 

        '''

        if (ps_version == 2):
            split_str = ps_file.split(".") 
            time = split_str[1]

            #now get the date
            date_split = ps_file.split("_")[3]

            #get *only* the date (first 8 characters)
            date = date_split[0:8]

        #NOTE: Haven't debugged this yet 
        elif (ps_version == 3):

            split_str = ps_file.split("_") 
            time = split_str[4] 
            time = time.split(".")[0]

            #Now get the date 
            date = split_str[3]
       
        #Now, remove the seconds 
        time = time[0:4] 
    

        return time, date

class TORP:
    
    def __init__(self, ID, prob, lat, lon, last_update_str, torp_df_row, radar, fcst_specs):
        self.fcst_duration = fcst_specs.forecast_window
        delta_t_min = (fcst_specs.start_valid_dt - fcst_specs.wofs_init_time_dt).seconds/60
        lead_index = int(((delta_t_min - c.wofs_spinup_time)/fcst_specs.forecast_window) + 1)
        self.num_fcsts = lead_index #if we are on the 3rd forecast period for example, we only care about making out to 3 forecasts
        self.predictors = {'prob': prob}
        self.lats = [lat]
        self.lons = [lon]
        self.Points = [Point(self.lons[0], self.lats[0])]
        self.long_id = ID
        self.ID = int(self.long_id.split('_')[0])
        self.detection_time = utilities.parse_date(self.long_id.split('_')[1])
        self.last_update = utilities.parse_date(last_update_str)
        self.set_start_valid()
        self.set_time_to_start_valid()
        self.fill_predictors(torp_df_row)
        self.set_storm_motion()
        self.set_future_lats_lons()
        self.update_buffers()
        self.radar = torp_df_row['Radar']
    
    def __gt__(self, other):
        '''This method overloads the greater than comparison for TORP_List sorting.
        The TORP_List will store by date with newest at index 0. Thus, "greater than"
        will be determined by which object's detection time is later. Tie is broken
        by ID value.'''
        
        if self.detection_time > other.detection_time:
            return True
        elif (self.detection_time == other.detection_time) and (self.ID < other.ID):
            return True
        elif (self.detection_time == other.detection_time) and (self.ID == other.ID) and (self.last_update > other.last_update):
            return True
        else:
            return False
    
    def __lt__(self, other):
        '''This method overloads the less than comparison for TORP_List sorting.
        The TORP_List will store by date with newest at index 0. Thus, "less than"
        will be determined by which object's detection time is earlier. Tie is broken
        by ID value.'''
        
        if self.detection_time < other.detection_time:
            return True
        elif (self.detection_time == other.detection_time) and (self.ID > other.ID):
            return True
        elif (self.detection_time == other.detection_time) and (self.ID == other.ID) and (self.last_update < other.last_update):
            return True
        else:
            return False
    
    def __eq__(self, other):
        '''This method will overload the equals to operator. If two TORP objects have
        the same ID and detection time, then they can be used to track storm motion
        and probability changes through time.'''
        
        if (self.detection_time == other.detection_time) and (self.ID == other.ID):
            return True
        else:
            return False
    
    def show(self):
        '''This is more of a helpful tool for the creation process, can probably
        be deleted once the product is created'''
        id_str = 'ID: ' + str(self.ID)
        coord_str = 'Location: (' + str(self.lats[0]) + ', ' + str(self.lons[0]) + ')'
        prob_str = 'Prob Tor: ' + str(self.predictors['prob']*100) + '%'
        time_str = 'Detected: ' + str(self.detection_time) + ',\nLast Updated: ' + str(self.last_update)
        age_str = 'Age: ' + str(self.predictors['age']) + ' minutes'
        prob_change_1_str = str(c.torp_prob_change_1) + '-Minute Probability Change: ' + str(self.predictors['p_change_' + str(c.torp_prob_change_1) + '_min']*100) + '%'
        prob_change_2_str = str(c.torp_prob_change_2) + '-Minute Probability Change: ' + str(self.predictors['p_change_' + str(c.torp_prob_change_2) + '_min']*100) + '%'
        
        print(id_str)
        print(coord_str)
        print(prob_str)
        print(time_str)
        print(age_str)
        print(prob_change_1_str)
        print(prob_change_2_str)
        print()
    
    def set_storm_motion(self):
        '''return storm motion in m/s'''
        self.storm_motion_north = 20
        self.storm_motion_east = 10
    
    def set_future_lats_lons(self):
        minutes_to_0 = self.time_to_start_valid
        x_dist = self.storm_motion_east * (minutes_to_0 * 60)/1000
        y_dist = self.storm_motion_north * (minutes_to_0 * 60)/1000
        
        for i in range(1, self.num_fcsts+2):
            self.lons.append(utilities.haversine_get_lon(self.lats[i-1], self.lons[i-1], x_dist))
            self.lats.append(utilities.haversine_get_lat(self.lats[i-1], self.lons[i-1], self.lons[i], y_dist))
            self.Points.append(Point(self.lons[i], self.lats[i]))
            
            x_dist = self.storm_motion_east * (self.fcst_duration * 60)/1000
            y_dist = self.storm_motion_north * (self.fcst_duration * 60)/1000
    
    def fill_predictors(self, row):
        for predictor in c.torp_predictors:
            try:
                self.predictors[predictor] = row[predictor]
            except:
                if predictor == 'RangeInterval':
                    self.predictors[predictor] = row['rng_int']
    
    def set_start_valid(self):
        curr_hour = self.last_update.hour
        curr_min = self.last_update.minute
        curr_sec = self.last_update.second
        if ((curr_min < (c.wofs_bottom_init_min + c.wofs_spinup_time)) or (curr_min == (c.wofs_bottom_init_min + c.wofs_spinup_time) and curr_sec == 0)) and (curr_min >= c.wofs_spinup_time):
            self.start_valid = self.last_update.replace(minute = c.wofs_bottom_init_min + c.wofs_spinup_time, second = 0)
        else:
            if curr_min >= (c.wofs_bottom_init_min + c.wofs_spinup_time):
                self.start_valid = (self.last_update + dt.timedelta(hours = 1))
                self.start_valid = self.last_update.replace(minute = c.wofs_spinup_time, second = 0)
            else:
                self.start_valid = self.last_update.replace(minute = c.wofs_spinup_time, second = 0)
    
    def set_time_to_start_valid(self):
        self.time_to_start_valid = ((self.start_valid - self.last_update).seconds)/60
    
    def update_buffers(self):
        '''Applies a geodesic point buffer to get a polygon (with many points to approximate
        a circle) centered around the lat/lon coords of the point using a geodesic buffer.
        The buffer represents the buffer in km (for instance, to get a 15km buffer, enter
        15 for buffer, not 15000.'''
        
        self.geometrys = [utilities.geodesic_point_buffer(self.lons[0], self.lats[0], c.torp_point_buffer)]
        
        for i in range(2, self.num_fcsts+2):
            line = LineString([self.Points[i-1], self.Points[i]])
            
            local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m +lat_0={} +lon_0={}".format((self.lats[i-1] + self.lats[i]) / 2, (self.lons[i-1] + self.lons[i]) / 2)
            
            wgs84_to_aeqd = Transformer.from_proj('+proj=longlat +datum=WGS84 +no_defs',local_azimuthal_projection)
            aeqd_to_wgs84 = Transformer.from_proj(local_azimuthal_projection,'+proj=longlat +datum=WGS84 +no_defs')

            line_transformed = transform(wgs84_to_aeqd.transform, line)

            buffer = line_transformed.buffer(c.torp_point_buffer * 1000)
            line_wgs84 = transform(aeqd_to_wgs84.transform, buffer)
            
            self.geometrys.append(line_wgs84)
    
    def check_bounds(self, grid):
        if (self.lats[0] > grid.ne_lat) or (self.lons[0] > grid.ne_lon) or (self.lats[0] < grid.sw_lat) or (self.lons[0] < grid.sw_lon):
            return False
        else:
            return True
    
    def get_wofs_overlap_points(self, wofs_gdf, torp_dict, *args):
        '''return the i, j components of a wofs grid that this torp object overlaps.
        Can change this in the future to deal with the different valid time torp swaths'''
        
        torp_gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[self.geometrys[self.num_fcsts]])
        overlap_gdf = gpd.overlay(wofs_gdf, torp_gdf, how='intersection')
        
        self.overlap_i = overlap_gdf.wofs_i.values
        self.overlap_j = overlap_gdf.wofs_j.values
        
        if len(args) == 0:
            prev_overlap_gdf = overlap_gdf
            prev_overlap_gdf['torp_id'] = self.long_id
        else:
            prev_overlap_gdf = args[0]
            self.check_overlap(prev_overlap_gdf, torp_dict)
            for i in range(len(self.overlap_i)):
                try:
                    row = np.where((wofs_gdf.wofs_i.values == self.overlap_i[i]) & (wofs_gdf.wofs_j.values == self.overlap_j[i]))[0][0]
                    point = wofs_gdf.geometry.values[row]
                    new_overlap_row = [{'wofs_i': self.overlap_i[i], 'wofs_j': self.overlap_j[i], 'geometry': point, 'torp_id': self.long_id}]
                    new_overlap_row_gdf = gpd.GeoDataFrame(new_overlap_row)
                    prev_overlap_gdf = pd.concat((prev_overlap_gdf,new_overlap_row_gdf.set_crs('epsg:4326')), axis=0)
                except:
                    raise Exception("Failed on coords: wofs_i: " + str(self.overlap_i[i]) + ", wofs_j: " + str(self.overlap_j[i]))
        
        return prev_overlap_gdf
    
    def check_overlap(self, overlap_gdf, torp_dict):
        '''Find areas of overlap between TORP objects on the wofs grid'''
        
        existing_overlap_i = overlap_gdf.wofs_i.values
        existing_overlap_j = overlap_gdf.wofs_j.values
        
        del_indices = []
        for i in range(len(self.overlap_i)):
            try:
                row = np.where((existing_overlap_i == self.overlap_i[i]) & (existing_overlap_j == self.overlap_j[i]))[0][0]
            except:
                #no overlap at this point
                continue
            
            overlapped_torp_id = overlap_gdf.torp_id.values[row]
            overlapped_torp_list = torp_dict[overlapped_torp_id]
            overlapped_torp = overlapped_torp_list.array[0] #get the most recent torp from this list
            if self.assert_dominance(overlapped_torp):
                #change the gdf
                overlap_gdf.torp_id.values[row] = self.long_id
            else:
                #mark this point for deletion from the overlap points for this TORP object as another TORP will be represented
                del_indices.append(i)
        
        self.overlap_i = np.delete(self.overlap_i, del_indices)
        self.overlap_j = np.delete(self.overlap_j, del_indices)
    
    def assert_dominance(self, battle_torp):
        '''Determine which TORP object will be represented at a grid point,
        can add more levels of determination here as the TORP object is flushed out,
        but for now, just higher probability works'''
        
        if self.predictors['prob'] > battle_torp.predictors['prob']:
            return True
        else:
            return False
    
    def find_link(self, torp_dict, cutoff):
        '''Finds the link (if one exists) for a torp object across 00z'''
        
        link_torp = None
        
        eligible_tls = []
        eligible_tl_dists = []
        
        for t_id in torp_dict:
            if t_id == self.long_id:
                continue
            tl = torp_dict[t_id]
            back = tl.array[-1]
            
            #does the linkable torp start within 10 minutes of 00z?
            if (back.last_update - cutoff).seconds/60 > 10:
                continue
            
            #is there a small enough time delta to consider linking the objects?
            time_delta = back.detection_time - self.last_update
            if (time_delta.seconds/60 > c.torp_max_time_skip) or (time_delta.seconds == 0):
                continue
            
            #are they from the same radar?
            if not (back.radar == self.radar):
                continue
            
            td_sec = time_delta.seconds
            north_motion = (self.storm_motion_north * td_sec)/1000
            east_motion = (self.storm_motion_east * td_sec)/1000
            
            extrap_lon = utilities.haversine_get_lon(self.lats[0], self.lons[0], east_motion)
            extrap_lat = utilities.haversine_get_lat(self.lats[0], self.lons[0], extrap_lon, north_motion)
            
            dist = utilities.haversine(extrap_lat, extrap_lon, back.lats[0], back.lons[0])
            
            #if within 4.5km --> they are linked since it is not closer to any other linkable torp object
            #if in 4.5-9 range --> they are linkable, but we need to check to make sure that no other
            #linkable torp object is closer
            #if 9+km away, then its feasible that they are separate torp objects, cannot link
            if dist < 4.5:
                return tl
            elif (dist >= 4.5) and (dist < 9):
                eligible_tls.append(tl)
                eligible_tl_dists.append(dist)
            else:
                continue
            
        if len(eligible_tls) == 0:
            return None
        
        eligible_tls = np.array(eligible_tls)
        eligible_tl_dists = np.array(eligible_tl_dists)
        
        min_dist_index = np.where(eligible_tl_dists == np.min(eligible_tl_dists))[0][0]
        
        return eligible_tls[min_dist_index]

class TORP_List:
    
    def __init__(self, torp_list = None):
        if torp_list == None:
            self.array = np.array([])
        else:
            self.array = torp_list
    
    def insert(self, torp):
        '''Insert a torp object into the list while maintaining newest to
        oldest order. This will allow for easy calculations about storms
        changing since all TORP objects in a list refer to the same storm
        at different times.'''
        
        if len(self.array) == 0:
            self.array = np.array([torp])
        else:
            for i in range(len(self.array)):
                if torp > self.array[i]:
                    new_array = list(self.array[0:i])
                    new_array.append(torp)
                    new_array.extend(list(self.array[i:]))
                    self.array = np.array(new_array)
                    del new_array
                    self.check_for_old_objects
                    self.update_front()
                    self.update_i(i)
                    return
            #append to the end of the array since it would have returned
            #out of the function if it was to be inserted in the middle
            new_array = list(self.array)
            new_array.append(torp)
            self.array = np.array(new_array)
            del new_array
            self.check_for_old_objects()
            self.update_i(i+1)
        self.update_front()
            
    def update_front(self):
        '''update age and prob changes over time'''
        
        front = self.array[0]
        back = self.array[-1]
        front.predictors['age'] = round(((front.last_update - front.detection_time).seconds)/60, 2)
        
        if (front.predictors['age'] < c.torp_prob_change_2) or (len(self.array) == 1):
            front.predictors['p_change_' + str(c.torp_prob_change_2) + '_min'] = front.predictors['prob']
        else:
            closeTorp = self.find_temporal_closest(c.torp_prob_change_2)
            front.predictors['p_change_' + str(c.torp_prob_change_2) + '_min'] = round(front.predictors['prob'] - closeTorp.predictors['prob'], 6)
        
        if front.predictors['age'] < c.torp_prob_change_1 or (len(self.array) == 1):
            front.predictors['p_change_' + str(c.torp_prob_change_1) + '_min'] = front.predictors['prob']
        else:
            closeTorp = self.find_temporal_closest(c.torp_prob_change_1)
            front.predictors['p_change_' + str(c.torp_prob_change_1) + '_min'] = round(front.predictors['prob'] - closeTorp.predictors['prob'], 6)
        
        front.update_buffers()
        
        self.front_time = front.last_update
        self.front_lat = front.lats[0]
        self.front_lon = front.lons[0]
        
        self.back_time = back.last_update
        self.back_lat = back.lats[0]
        self.back_lon = back.lons[0]
    
    def update_i(self, i):
        '''update a specific index of the array'''
        torp = self.array[i]
        torp.predictors['age'] = round(((torp.last_update - torp.detection_time).seconds)/60, 2)
        
        if (torp.predictors['age'] < c.torp_prob_change_2) or (len(self.array) == 1):
            torp.predictors['p_change_' + str(c.torp_prob_change_2) + '_min'] = torp.predictors['prob']
        else:
            closeTorp = self.find_temporal_closest(c.torp_prob_change_2, i)
            torp.predictors['p_change_' + str(c.torp_prob_change_2) + '_min'] = round(torp.predictors['prob'] - closeTorp.predictors['prob'], 6)
        
        if torp.predictors['age'] < c.torp_prob_change_1 or (len(self.array) == 1):
            torp.predictors['p_change_' + str(c.torp_prob_change_1) + '_min'] = torp.predictors['prob']
        else:
            closeTorp = self.find_temporal_closest(c.torp_prob_change_1, i)
            torp.predictors['p_change_' + str(c.torp_prob_change_1) + '_min'] = round(torp.predictors['prob'] - closeTorp.predictors['prob'], 6)
        
        torp.update_buffers()
            
    def find_temporal_closest(self, time, i = 0):
        '''Find torp in list closest to 'time' minutes ago'''
        
        minTime = 100000
        returnTorp = None
        for torp in self.array:
            if abs(((((self.array[i].last_update - torp.last_update).seconds)/60) - time)) < abs(minTime):
                minTime = abs(((((self.array[i].last_update - torp.last_update).seconds)/60) - time))
                returnTorp = torp
        
        return returnTorp
        
    #add functionality to delete itself from dictionary if all objects are 3+ hours old
    def check_for_old_objects(self):
        '''Get rid of objects from 3+ hours ago unless they are ongoing'''
        
        last_update = self.array[0].last_update
        cutoff_time = last_update - dt.timedelta(hours = 3)
        
        del_indices = []
        for i in range(len(self.array)):
            t = self.array[i]
            if cutoff_time > t.last_update:
                del_indices.append(i)
            
        self.array = np.delete(self.array, del_indices)
    
    def link_torp(self, tl):
        '''used for appending torp lists across 00z'''
        for torp in tl.array:
            torp.long_id = self.array[0].long_id
            torp.detection_time = self.array[0].detection_time
            self.insert(torp)
    
    @staticmethod
    def link_torps(torp_dict, cutoff):
        '''If training, we need to link torp objects to have the same id across 00z'''
        
        keys = list(torp_dict.keys())
        for t_id in keys:
            try:
                tl = torp_dict[t_id]
            except:
                #torp has already been linked
                continue
            front_t = tl.array[0]
            back_t = tl.array[-1]
            
            if (cutoff - front_t.last_update).seconds/60 < 10:
                #find torp_list to link to, if exists
                to_link = front_t.find_link(torp_dict, cutoff)
                if not (to_link == None):
                    del_id = to_link.array[0].long_id
                    tl.link_torp(to_link)
                    del torp_dict[del_id]
        
        return torp_dict
    
    @staticmethod
    def gen_torp_dict_from_file(path, grid, fcst_specs, td = None, cutoff = None):
        '''Given a torp csv file from a radar, this function will create
        TORP objects and add them to a dictionary of TORP lists. Each list
        will be full of TORP objects with the same long id but different
        'last updated' times. This will allow for all TORP objects of the
        same storm to be grouped together, but all storms separated. Also,
        TORP objects/lists are not added to the dictionary if they are not
        on the given wofs grid'''
        
        #if no torp list was passed, create one to add new torp objects to,
        #otherwise, use passed torp list
        if td == None:
            torp_dict = {}
        else:
            torp_dict = copy.deepcopy(td)
        
        torp_df = pd.read_csv(path)
        IDs = torp_df.ID_date
        probs = torp_df.Probability
        lats = torp_df.Lat
        lons = torp_df.Lon
        
        file = path.split('/')[-1]
        last_update = file.split('_')[0]
        for i in range(len(IDs)):
            #if its not a string, its a nan object and there are no TORPs in this file
            if not isinstance(IDs[0], str):
                continue
            #create the TORP object
            
            torp = TORP(IDs[i], probs[i], lats[i], lons[i], last_update, torp_df.iloc[i], torp_df, fcst_specs)
            #if it's out of bounds for the wofs grid of the day, then ignore it
            if not torp.check_bounds(grid):
                continue
            #if this torp object is already in the dictionary, add it to its existing torp list
            if torp.long_id in torp_dict:
                torp_list = torp_dict[torp.long_id]
                torp_list.insert(torp)
                torp_dict[torp.long_id] = torp_list
            #otherwise, create a new torp list and add to dictionary
            else:
                torp_list = TORP_List()
                torp_list.insert(torp)
                torp_dict[torp.long_id] = torp_list
        
        if c.is_train_mode:
            torp_dict = TORP_List.link_torps(torp_dict, cutoff)
            
        return torp_dict
    
    #change to generating a dictionary
    @staticmethod
    def gen_full_dict_from_file_list(paths, grid, fcst_specs):
        '''Get the torp dictionary full of torp lists representing each object through time.
        Only keep torps that are on the wofs grid. Also, this finds the "true" init time
        after the dictionary is created. This is used in subsequent functions when mapping
        to the wofs grid.'''
        
        if c.is_train_mode:
            #set the cutoff to the last file on the date, sometimes a bit after 00z
            #ok to use my file paths here since it's training and i know the
            #training torp directory
            cutoff_candidates = []
            for path in paths:
                #if trained with file paths in a different spot, may need to change the [7]
                dir_date = path.split('/')[7] + '-000000'
                date_time = utilities.parse_date(path.split('/')[-1].split('_')[0])
                date = utilities.parse_date(dir_date)
                if not (date.day == date_time.day):
                    cutoff_candidates.append(date_time)
            if len(cutoff_candidates) == 0:
                date = utilities.parse_date(paths[0].split('/')[-1].split('_')[0])
                if date.hour > 0:
                    date.replace(hour=0, minute=0, second=0)
                else:
                    date.replace(day=date.day+1, hour=0, minute=0, second=0)
                cutoff = date
            else:
                cutoff = np.max(cutoff_candidates)
            
            for i, path in enumerate(paths):
                if i == 0:
                    torp_dict = TORP_List.gen_torp_dict_from_file(path, grid, fcst_specs, cutoff=cutoff)
                else:
                    torp_dict = TORP_List.gen_torp_dict_from_file(path, grid, fcst_specs, cutoff=cutoff, torp_dict=torp_dict)
            
            return torp_dict
        
        torp_dict = {}
        
        #if not training mode, then don't need to deal with calculating cutoff point
        for i, path in enumerate(paths):
            if i == 0:
                torp_dict = TORP_List.gen_torp_dict_from_file(path, grid, fcst_specs)
            else:
                torp_dict = TORP_List.gen_torp_dict_from_file(path, grid, fcst_specs, torp_dict)
        
        return torp_dict
    
    @staticmethod
    def gen_wofs_points_gdf(torp_dict, fcst_specs, wofs_gdf):
        '''This method will return a gdf with wofs_i and wofs_j values along with
        the associated torp_id. This will allow for easily applying TORP object
        predictors to each point on the wofs map.
        
        After extrapolation is implemented, this will need to be updated to specify
        the lead time at which the resulting wofs points and torp_ids are valid.'''
        
        delta_t_min = (fcst_specs.start_valid_dt - fcst_specs.wofs_init_time_dt).seconds/60
        lead_index = ((delta_t_min - c.wofs_spinup_time)/fcst_specs.forecast_window) + 1
        
        i = 0
        gdf = None
        for long_id in torp_dict:
            l = torp_dict[long_id]
            t = l.array[0]
            if not (t.start_valid == fcst_specs.wofs_init_time_dt + dt.timedelta(seconds = c.wofs_spinup_time*60)):
                continue
            if i == 0:
                gdf = t.get_wofs_overlap_points(wofs_gdf, torp_dict)
                i += 1
            else:
                gdf = t.get_wofs_overlap_points(wofs_gdf, torp_dict, gdf)
        
        return gdf
    
    @staticmethod
    def overlap_gdf_to_npy(gdf, torp_dict, txt_file):
        '''This function takes the gdf of overlap points and corresponding torp objects and makes the final npy
        of torp predictors. It also applies the convolutions to get the max values on the grid within x km.
        These radii to search in the convolutions as well as the method of convolution (max, min, absolute value)
        can be set in the config file.'''
        
        km_spacing = c.dx_km
        radii_km = c.torp_conv_dists
        n_sizes = []
        for i in range(len(radii_km)):
            r = radii_km[i]
            n_sizes.append(int(((r/km_spacing)*2)+3))

        conv_footprints = utilities.get_footprints(n_sizes, radii_km, km_spacing)
        
        predictors = c.torp_all_predictors
        
        npy_predictors_dict = {}
        for predictor in predictors:
            npy_predictors_dict[predictor] = np.zeros((300, 300))
            if predictor in ['age', 'RangeInterval']:
                npy_predictors_dict[predictor] -= 1
        
        if  isinstance(gdf, pd.DataFrame):
            wofs_i = gdf.wofs_i.values
            wofs_j = gdf.wofs_j.values
        else:
            wofs_i = []
            wofs_j = []
        
        for m in range(len(wofs_i)):
            i = wofs_i[m]
            j = wofs_j[m]
            torp_predictors = torp_dict[gdf.torp_id.values[m]].array[0].predictors
            for predictor in npy_predictors_dict:
                array = npy_predictors_dict[predictor]
                array[i, j] = torp_predictors[predictor]
                npy_predictors_dict[predictor] = array
        
        if not os.path.isfile(txt_file):  
            f = open(txt_file, "w")
        
        for i, predictor in enumerate(predictors):
            array_2d = npy_predictors_dict[predictor]
            if predictor in c.torp_max_convs:
                var_method = "max"
            elif predictor in c.torp_min_convs:
                var_method = "min"
            elif predictor in c.torp_abs_convs:
                var_method = "abs"
            else:
                var_method = "none"
            
            if not (var_method == "none"):
                array_2d_15km = utilities.add_convolutions(var_method, array_2d, conv_footprints[0])
                array_1d_15km = array_2d_15km.reshape((90000,1))
                array_2d_30km = utilities.add_convolutions(var_method, array_2d, conv_footprints[1])
                array_1d_30km = array_2d_30km.reshape((90000,1))
                array_2d_45km = utilities.add_convolutions(var_method, array_2d, conv_footprints[2])
                array_1d_45km = array_2d_45km.reshape((90000,1))
                array_2d_60km = utilities.add_convolutions(var_method, array_2d, conv_footprints[3])
                array_1d_60km = array_2d_60km.reshape((90000,1))
            
            array_1d = array_2d.reshape((90000,1))
            if i == 0:
                full_npy = array_1d
                if not (var_method == "none"):
                    full_npy = np.append(full_npy, array_1d_15km, axis = 1)
                    full_npy = np.append(full_npy, array_1d_30km, axis = 1)
                    full_npy = np.append(full_npy, array_1d_45km, axis = 1)
                    full_npy = np.append(full_npy, array_1d_60km, axis = 1)
                i += 1
            else:
                full_npy = np.append(full_npy, array_1d, axis = 1)
                if not (var_method == "none"):
                    full_npy = np.append(full_npy, array_1d_15km, axis = 1)
                    full_npy = np.append(full_npy, array_1d_30km, axis = 1)
                    full_npy = np.append(full_npy, array_1d_45km, axis = 1)
                    full_npy = np.append(full_npy, array_1d_60km, axis = 1)
            
            if not os.path.isfile(txt_file):
                f.write(predictor + '\n')
                f.write(predictor + '_' + var_method + '_15km' + '\n')
                f.write(predictor + '_' + var_method + '_30km' + '\n')
                f.write(predictor + '_' + var_method + '_45km' + '\n')
                f.write(predictor + '_' + var_method + '_60km' + '\n')
        if not os.path.isfile(txt_file):    
            f.close()
        return full_npy
    
    @staticmethod
    def gen_torp_npy(torp_files, wofs_grid, fcst_specs):
        
        wofs_gdf = PS.get_wofs_gdf(wofs_grid)
        td = TORP_List.gen_full_dict_from_file_list(torp_files, wofs_grid, fcst_specs)
        gdf = TORP_List.gen_wofs_points_gdf(td, fcst_specs, wofs_gdf)
        npy = TORP_List.overlap_gdf_to_npy(gdf, td, c.torp_vars_filename)
        
        return npy

def main():
    '''Main Method'''

    #For debugging 
    #TODO: Put these in and start seeing if I can get something reasonable. 
    wofs_direc = "/work/mflora/SummaryFiles/20210604/0200"
    ps_direc = "/work/eric.loken/wofs/probSevere"


    #TODO: We'd need to develop some code (maybe in an outside script, etc. to determine these files/filenames
    wofs_files = ["wofs_ALL_05_20210605_0200_0225.nc", "wofs_ALL_06_20210605_0200_0230.nc", \
                    "wofs_ALL_07_20210605_0200_0235.nc", "wofs_ALL_08_20210605_0200_0240.nc",\
                    "wofs_ALL_09_20210605_0200_0245.nc", "wofs_ALL_10_20210605_0200_0250.nc",\
                    "wofs_ALL_11_20210605_0200_0255.nc"] 
    
    wofs_files2 = ["wofs_ALL_11_20210605_0200_0255.nc", "wofs_ALL_12_20210605_0200_0300.nc",\
                    "wofs_ALL_13_20210605_0200_0305.nc", "wofs_ALL_14_20210605_0200_0310.nc",\
                    "wofs_ALL_15_20210605_0200_0315.nc", "wofs_ALL_16_20210605_0200_0320.nc",\
                    "wofs_ALL_17_20210605_0200_0325.nc"]

    #ps_files = ["MRMS_EXP_PROBSEVERE_20210605.022400.json", "MRMS_EXP_PROBSEVERE_20210605.022200.json",\
    #            "MRMS_EXP_PROBSEVERE_20210605.021400.json", "MRMS_EXP_PROBSEVERE_20210605.021000.json",\
    #            "MRMS_EXP_PROBSEVERE_20210605.015400.json", "MRMS_EXP_PROBSEVERE_20210605.014000.json",\
    #            "MRMS_EXP_PROBSEVERE_20210605.012400.json", "MRMS_EXP_PROBSEVERE_20210605.011000.json",\
    #            "MRMS_EXP_PROBSEVERE_20210605.005400.json", "MRMS_EXP_PROBSEVERE_20210605.004000.json",\
    #            "MRMS_EXP_PROBSEVERE_20210605.002400.json", "MRMS_EXP_PROBSEVERE_20210605.001000.json",\
    #            "MRMS_EXP_PROBSEVERE_20210604.235400.json", "MRMS_EXP_PROBSEVERE_20210604.234000.json",\
    #            "MRMS_EXP_PROBSEVERE_20210604.232400.json"]

    ps_times = ["0224", "0222", "0220", "0218", "0216", "0214", "0212", "0210", "0208", "0206",\
                "0204", "0202", "0200", "0158", "0156", "0154", "0152", "0150", "0148", "0146",\
                "0144", "0142", "0140", "0138", "0136", "0134", "0132", "0130", "0128", "0126",\
                "0124", "0122", "0120", "0118", "0116", "0114", "0112", "0110", "0108", "0106",\
                "0104", "0102", "0100", "0058", "0056", "0054", "0052", "0050", "0048", "0046",\
                "0044", "0042", "0040", "0038", "0036", "0034", "0032", "0030", "0028", "0026",\
                "0024", "0022", "0020", "0018", "0016", "0014", "0012", "0010", "0008", "0006",\
                "0004", "0002", "0000", "2358", "2356", "2354", "2352", "2350", "2348", "2346",\
                "2344", "2342", "2340", "2338", "2336", "2334", "2332", "2330", "2328", "2326",\
                "2324"] 

    ps_times = ["%s00" %p for p in ps_times] 

    ps_dates = ["20210605" for p in ps_times] 
    for a in range(73, len(ps_times)):
        ps_dates[a] = "20210604"


    ps_files = ["MRMS_EXP_PROBSEVERE_%s.%s.json" %(ps_dates[p], ps_times[p]) for p in range(len(ps_times))]
   
 
    torp_files = ['/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-023037_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-023337_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-023628_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-023928_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-024219_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-024519_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-024808_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-025102_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-025350_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-025644_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-025931_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-030225_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-030512_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-030812_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-031103_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-031356_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-031643_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-031937_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-032238_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-032532_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-032804_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-033013_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-033234_KUDX_tordetections.csv',
                  '/work/ryan.martz/wofs_phi_data/training_data/predictors/raw_torp/20210605/20210605-033456_KUDX_tordetections.csv']

    ml_obj = MLGenerator(wofs_files2, ps_files, ps_direc, wofs_direc, torp_files, c.nc_outdir)

    #Do the generation 
    ml_obj.generate() 



if (__name__ == '__main__'):

    main()


