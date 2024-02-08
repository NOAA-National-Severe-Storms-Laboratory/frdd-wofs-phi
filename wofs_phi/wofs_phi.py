#===========================================
# This file runs the wofs-phi probabilities
#
# It needs an input date and time (which will be used 
# to set WoFS and PS initializations) and 


# Created by Eric Loken, 2/6/2024
#
#
#===========================================


#======================
# Imports
#======================
from shapely.geometry import Point, MultiPolygon, Polygon
from shapely.prepared import prep
from shapely import geometry
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
from datetime import datetime
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




class MLGenerator: 
    ''' This class will handle the ML generator functions. '''


    def __init__(self, wofs_files, ps_files, ps_path, wofs_path, nc_outdir): 

        ''' @wofs_files is the list of wofs_files to use for the prediction (in chronological order,
            beginning with the start of the prediction window/valid period and ending with the end of
            the prediction window/valid period. 

            @ps_files are the list of probSevere files needed in reverse chronological order (i.e.,
            starting with the most recent file/time step. So we'd have t = 0, -2, -10, -14, -30, -44,
            -60, -74, -90, -104, -120, -134, -150, -164, -180 minutes

            @ps_path is the string path to the probSevere files 

            @wofs_path is the string path to the wofs files   

            @nc_outdir is the directory to save the final .ncdf files to 
            
        '''

        self.wofs_files = wofs_files
        self.ps_files = ps_files
        self.ps_path = ps_path
        self.wofs_path = wofs_path
        self.nc_outdir = nc_outdir


    def generate(self):
        '''Instance method to generate the predictions given an MLGenerator object.
            Kind of like the "main method" for generating the predictions. ''' 

        #TODO: Should build roadmap 
       
        # Get grid stats from first wofs file 
        grid = Grid.create_wofs_grid(self.wofs_path, self.wofs_files[0])

        #Get the forecast specifications (e.g., start valid, end_valid, ps_lead time, wofs_lead_time, etc.) 
        #These will be determined principally by the wofs files we're dealing with
        forecast_specs = ForecastSpecs.create_forecast_specs(self.ps_files[0], self.wofs_files)


        #Do PS preprocessing -- parallel track 1 -- should return a PS xarray 
        

        
        #Do WoFS preprocessing -- parallel track 2 


        #Concatenate parallel tracks 


        #Add convolutions 


        #Convert to 1d predictor list 



        #Save predictors to file (if we're training) 



        #Load RF, run the predictors through RF 


        #Save predictions to ncdf 


        pass


    @staticmethod
    def get_full_path(pathToFiles, filenames):
        '''Returns a list of the full path to a list of files given a path and a list of filenames.'''
        full_list = ["%s/%s" %(pathToFiles, f) for f in filenames]
    
        return full_list  


class Wofs:
    '''Handles the wofs forecasting/processing'''


    def __init__(self):
        #self.ny = ny
        #self.nx = nx

        pass


class Grid: 
    '''Handles the (wofs) grid attributes.'''


    def __init__(self, ny, nx, lats, lons, tlat1, tlat2, stlon, sw_lat, ne_lat, sw_lon, ne_lon): 
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


    @classmethod
    def create_wofs_grid(cls, wofs_path, wofs_file): 
        '''Creates a Grid object from a wofs path and wofs file.'''

        full_wofs_file = "%s/%s" %(wofs_path, wofs_file) 

        ds = nc.Dataset(full_wofs_file)
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

        #Create new wofs Grid object 
        wofs_grid = Grid(ny, nx, wofsLats, wofsLons, Tlat1, Tlat2, Stlon, SW_lat, NE_lat, SW_lon, NE_lon)  

        return wofs_grid
        


class PS:
    '''Handles the ProbSevere forecasts/processing'''

    def __init__(self):
        pass


    @classmethod
    def preprocess_ps(cls):
        ''' Like the "main method"/blueprint method for doing the probSevere preprocessing.'''


        #Current procedure: #TODO
        #Get a dataframe of all past objects (including their IDs, hazard probabilities, and ages) 

        #Get PS geodataframe 

        #Get Wofs geodataframe -- and, ultimately, buffered WoFS geodataframe --> for merging purposes

        #Get merged PS/WoFS geodataframe, which is how we "map" PS points to wofs grid 

        #Do the extrapolation 

        #Put the key predictor fields in geodataframe 

        #Convert to xarray 

        #Create new PS object -- will hold 

        pass


class ForecastSpecs: 

    '''Class to handle/store the forecast specifications.'''

    def __init__(self, start_valid, end_valid, start_valid_dt, end_valid_dt, \
                    wofs_init_time, wofs_init_time_dt, forecast_window, ps_init_time,\
                    ps_lead_time_start, ps_lead_time_end, ps_init_time_dt):

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

        pass

    @classmethod
    def create_forecast_specs(cls, first_ps_file, wofs_files):
        '''Blueprint method for creating a ForecastSpecs object based on the first 
            PS file and the list of wofs files. 
        '''


        #TODO: Might have to do this with datetime objects 
        #Find start/end valid and wofs initialization time from wofs files 
        start_valid, start_valid_date = ForecastSpecs.find_date_time_from_wofs(wofs_files[0], "forecast")
        end_valid, end_valid_date = ForecastSpecs.find_date_time_from_wofs(wofs_files[-1], "forecast") 
        wofs_init_time, wofs_init_date = ForecastSpecs.find_date_time_from_wofs(wofs_files[0], "init") 

        #Obtain datetime versions of the above (i.e., start_valid, end_valid, wofs_init_time, etc.) 
        start_valid_dt = ForecastSpecs.str_to_dattime(start_valid, start_valid_date) 
        end_valid_dt = ForecastSpecs.str_to_dattime(end_valid, end_valid_date) 
        wofs_init_time_dt = ForecastSpecs.str_to_dattime(wofs_init_time, wofs_init_date) 

        #Find the length of the forecast time window based on the start_valid_dt and end_valid_dt
        #datetime objects 

        valid_window = ForecastSpecs.subtract_dt(end_valid_dt, start_valid_dt, True) 

        #Find PS init time from PS file 
        ps_init_time, ps_init_date = ForecastSpecs.find_ps_date_time(first_ps_file, c.ps_version)

        #Obtain datetime objects
        ps_init_time_dt = ForecastSpecs.str_to_dattime(ps_init_time, ps_init_date) 

        #Find PS lead time for start of valid period (in minutes) 
        #based on PS initialization time and start of the valid period 
        ps_start_lead_time = ForecastSpecs.subtract_dt(start_valid_dt, ps_init_time_dt, True) 
        
        #Find PS lead time for end of valid period (in minutes) 
        #based on PS initailization time and end of the valid period
        ps_end_lead_time = ForecastSpecs.subtract_dt(end_valid_dt, ps_init_time_dt, True) 

        #Create ForecastSpecs object  

        new_specs = ForecastSpecs(start_valid, end_valid, start_valid_dt, end_valid_dt, wofs_init_time, \
                            wofs_init_time_dt, valid_window, ps_init_time, ps_start_lead_time, ps_end_lead_time,\
                            ps_init_time_dt) 

        return new_specs


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
    def find_date_time_from_wofs(wofs_file, time_type):
        '''
        Finds/returns the (string) time and date (e.g., start or end of the forecast valid window) associated with the given WoFS file. 
        # @wofs_file is the string of the wofs file 
        # @time_type is a string: "forecast" is a forecast time period; "init" is initialization time 
        '''

        #Split the string based on underscores 
        split_str = wofs_file.split("_") 
        if (time_type == "forecast"):
            time = split_str[5] #In this case, it'll be the 5th element of the wofs file string

            #Have to remove the .nc
            time = time.split(".")[0]

        elif (time_type == "init"):
            time = split_str[4]

        date = split_str[3] 


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

def main():
    '''Main Method'''

    #For debugging 
    #TODO: Put these in and start seeing if I can get something reasonable. 
    wofs_direc = "/work/mflora/SummaryFiles/20210604/0200"
    ps_direc = "/work/eric.loken/wofs/probSevere"
    nc_outdir = "."


    #TODO: We'd need to develop some code (maybe in an outside script, etc. to determine these files/filenames
    wofs_files = ["wofs_ALL_05_20210605_0200_0225.nc", "wofs_ALL_06_20210605_0200_0230.nc", \
                    "wofs_ALL_07_20210605_0200_0235.nc", "wofs_ALL_08_20210605_0200_0240.nc",\
                    "wofs_ALL_09_20210605_0200_0245.nc", "wofs_ALL_10_20210605_0200_0250.nc",\
                    "wofs_ALL_11_20210605_0200_0255.nc"] 
    ps_files = ["MRMS_EXP_PROBSEVERE_20210605.022400.json", "MRMS_EXP_PROBSEVERE_20210605.022200.json",\
                "MRMS_EXP_PROBSEVERE_20210605.021400.json", "MRMS_EXP_PROBSEVERE_20210605.021000.json",\
                "MRMS_EXP_PROBSEVERE_20210605.015400.json", "MRMS_EXP_PROBSEVERE_20210605.014000.json",\
                "MRMS_EXP_PROBSEVERE_20210605.012400.json", "MRMS_EXP_PROBSEVERE_20210605.011000.json",\
                "MRMS_EXP_PROBSEVERE_20210605.005400.json", "MRMS_EXP_PROBSEVERE_20210605.004000.json",\
                "MRMS_EXP_PROBSEVERE_20210605.002400.json", "MRMS_EXP_PROBSEVERE_20210605.001000.json",\
                "MRMS_EXP_PROBSEVERE_20210604.235400.json", "MRMS_EXP_PROBSEVERE_20210604.234000.json",\
                "MRMS_EXP_PROBSEVERE_20210604.232400.json"] 

    ml_obj = MLGenerator(wofs_files, ps_files, ps_direc, wofs_direc, nc_outdir)

    #Do the generation 
    ml_obj.generate() 



if (__name__ == '__main__'):

    main()


