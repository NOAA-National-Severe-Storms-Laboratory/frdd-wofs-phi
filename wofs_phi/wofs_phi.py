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
import datetime
import copy
import utilities




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
        fcst_grid = Grid.create_wofs_grid(self.wofs_path, self.wofs_files[0])

        #Get the forecast specifications (e.g., start valid, end_valid, ps_lead time, wofs_lead_time, etc.) 
        #These will be determined principally by the wofs files we're dealing with
        forecast_specs = ForecastSpecs.create_forecast_specs(self.ps_files[0], self.wofs_files)

        #Do PS preprocessing -- parallel track 1 -- should return a PS xarray 
        ps = PS.preprocess_ps(fcst_grid, forecast_specs) 

        
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

    def __init__(self, gdf, xarr):
        ''' @gdf is a geodataframe containing all the relevant predictors
            @xarr is an xarray of all the relevant predictors
        '''

        self.gdf = gdf
        self.xarr = xarr
        

        return 

    @classmethod
    def preprocess_ps(cls, grid, specs):
        ''' Like the "main method"/blueprint method for doing the probSevere preprocessing.
            @grid is the forecast Grid object for the current case
            @specs is the ForecastSpecs object for the current case. 
            Ultimately creates a PS object with a gdf and xarrray of the relevant predictors 
        '''

        #Current procedure: #TODO
        #Get a dataframe of all past objects (including their IDs, hazard probabilities, and ages) 

        #Get PS geodataframe 

        #Get Wofs geodataframe -- and, ultimately, buffered WoFS geodataframe --> for merging purposes

        #Get merged PS/WoFS geodataframe, which is how we "map" PS points to wofs grid 

        #Do the extrapolation 

        #Put the key predictor fields in geodataframe 

        #Convert to xarray 

        #Create new PS object -- will hold geodataframe of predictors and xarray 

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

class TORP:
    
    def __init__(self, ID, prob, lat, lon, last_update_str, torp_df_row):
        self.predictors = {'prob': prob}
        self.lats = [lat]
        self.lons = [lon]
        self.Points = [Point(self.lons[0], self.lats[0])]
        self.long_id = ID
        self.ID = int(self.long_id.split('_')[0])
        self.detection_time = utilities.parse_date(self.long_id.split('_')[1])
        self.last_update = utilities.parse_date(last_update_str)
        self.set_init_start()
        self.set_time_to_init_start()
        self.fill_predictors(torp_df_row)
        self.set_storm_motion()
        self.set_future_lats_lons()
        self.update_buffers()
    
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
        minutes_to_0 = self.time_to_init_start + 25
        x_dist = self.storm_motion_east * (minutes_to_0 * 60)/1000
        y_dist = self.storm_motion_north * (minutes_to_0 * 60)/1000
        
        for i in range(1, 8):
            self.lons.append(utilities.haversine_get_lon(self.lats[i-1], self.lons[i-1], x_dist))
            self.lats.append(utilities.haversine_get_lat(self.lats[i-1], self.lons[i-1], self.lons[i], y_dist))
            self.Points.append(Point(self.lons[i], self.lats[i]))
            
            x_dist = self.storm_motion_east * (30 * 60)/1000
            y_dist = self.storm_motion_north * (30 * 60)/1000
    
    def fill_predictors(self, row):
        for predictor in c.torp_predictors:
            try:
                self.predictors[predictor] = row[predictor]
            except:
                if predictor == 'RangeInterval':
                    self.predictors[predictor] = row['rng_int']
    
    def set_init_start(self):
        curr_hour = self.last_update.hour
        curr_min = self.last_update.minute
        curr_sec = self.last_update.second
        if (curr_min < 30) or (curr_min == 30 and curr_sec == 0):
            self.init_start = self.last_update.replace(minute = 30, second = 0)
        else:
            if curr_hour < 23:
                self.init_start = self.last_update.replace(hour = curr_hour + 1, minute = 0, second = 0)
            else:
                self.init_start = self.last_update.replace(hour = 0, minute = 0, second = 0)
    
    def set_time_to_init_start(self):
        self.time_to_init_start = ((self.init_start - self.last_update).seconds)/60
    
    def update_buffers(self):
        '''Applies a geodesic point buffer to get a polygon (with many points to approximate
        a circle) centered around the lat/lon coords of the point using a geodesic buffer.
        The buffer represents the buffer in km (for instance, to get a 15km buffer, enter
        15 for buffer, not 15000.'''
        
        self.geometrys = [utilities.geodesic_point_buffer(self.lons[0], self.lats[0], c.torp_point_buffer)]
        
        for i in range(2, 8):
            line = LineString([self.Points[i-1], self.Points[i]])
            
            local_azimuthal_projection = "+proj=aeqd +R=6371000 +units=m +lat_0={} +lon_0={}".format((self.lats[i-1] + self.lats[i]) / 2, (self.lons[i-1] + self.lons[i]) / 2)
            
            wgs84_to_aeqd = Transformer.from_proj('+proj=longlat +datum=WGS84 +no_defs',local_azimuthal_projection)
            aeqd_to_wgs84 = Transformer.from_proj(local_azimuthal_projection,'+proj=longlat +datum=WGS84 +no_defs')

            line_transformed = transform(wgs84_to_aeqd.transform, line)

            buffer = line_transformed.buffer(c.torp_point_buffer * 1000)
            line_wgs84 = transform(aeqd_to_wgs84.transform, buffer)
            
            self.geometrys.append(line_wgs84)
        #self.geometrys.append(0-30)
        #self.geometrys.append(30-60)
        #self.geometrys.append(60-90)
        #self.geometrys.append(90-120)
        #self.geometrys.append(120-150)
        #self.geometrys.append(150-180)
    
    def check_bounds(self, grid):
        if (self.lats[0] > grid.ne_lat) or (self.lons[0] > grid.ne_lon) or (self.lats[0] < grid.sw_lat) or (self.lons[0] < grid.sw_lon):
            return False
        else:
            return True
    
    def get_wofs_overlap_points(self, wofs_gdf, torp_dict, lead_time_int, *args):
        '''return the i, j components of a wofs grid that this torp object overlaps.
        Can change this in the future to deal with the different valid time torp swaths'''
        
        torp_gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[self.geometrys[lead_time_int]])
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
        
        return prev_overlap_gdf

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
                    return
            #append to the end of the array since it would have returned
            #out of the function if it was to be inserted in the middle
            new_array = list(self.array)
            new_array.append(torp)
            self.array = np.array(new_array)
            del new_array
            self.check_for_old_objects()
        self.update_front()
            
    def update_front(self):
        '''update age and prob changes over time'''
        
        front = self.array[0]
        back = self.array[-1]
        front.predictors['age'] = round(((front.last_update - back.last_update).seconds)/60, 2)
        
        if front.predictors['age'] < c.torp_prob_change_2:
            front.predictors['p_change_' + str(c.torp_prob_change_2) + '_min'] = front.predictors['prob']
        else:
            closeTorp = self.find_temporal_closest(c.torp_prob_change_2)
            front.predictors['p_change_' + str(c.torp_prob_change_2) + '_min'] = round(front.predictors['prob'] - closeTorp.predictors['prob'], 6)
        
        if front.predictors['age'] < c.torp_prob_change_1:
            front.predictors['p_change_' + str(c.torp_prob_change_1) + '_min'] = front.predictors['prob']
        else:
            closeTorp = self.find_temporal_closest(c.torp_prob_change_1)
            front.predictors['p_change_' + str(c.torp_prob_change_1) + '_min'] = round(front.predictors['prob'] - closeTorp.predictors['prob'], 6)
        
        front.update_buffers()
            
    def find_temporal_closest(self, time):
        '''Find torp in list closest to 'time' minutes ago'''
        
        minTime = 100000
        returnTorp = None
        for torp in self.array:
            if abs(((((self.array[0].last_update - torp.last_update).seconds)/60) - time)) < abs(minTime):
                minTime = abs(((((self.array[0].last_update - torp.last_update).seconds)/60) - time))
                returnTorp = torp
        
        return returnTorp
        
    #add functionality to delete itself from dictionary if all objects are 3+ hours old
    def check_for_old_objects(self):
        '''Get rid of objects from 3+ hours ago unless they are ongoing'''
        
        last_update = self.array[0].last_update
        cutoff_time = last_update - datetime.timedelta(hours = 3)
        
        del_indices = []
        for i in range(len(self.array)):
            t = self.array[i]
            if cutoff_time > t.last_update:
                del_indices.append(i)
            
        self.array = np.delete(self.array, del_indices)
                
    @staticmethod
    def gen_torp_dict_from_file(path, grid, td = None):
        '''Given a torp csv file from a radar, this function will create
        TORP objects and add them to a dictionary of TORP lists. Each list
        will be full of TORP objects with the same long id but different
        'last updated' times. This will allow for all TORP objects of the
        same storm to be grouped together, but all storms separated'''
        
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
            torp = TORP(IDs[i], probs[i], lats[i], lons[i], last_update, torp_df.iloc[i])
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
            
        return torp_dict
    
    #change to generating a dictionary
    @staticmethod
    def gen_full_dict_from_file_list(paths, grid):
        true_init = datetime.datetime(1970, 1, 1, 0, 0, 0)
        for i, path in enumerate(paths):
            if i == 0:
                torp_dict = TORP_List.gen_torp_dict_from_file(path, grid)
            else:
                torp_dict = TORP_List.gen_torp_dict_from_file(path, grid, torp_dict)
            
            file = path.split('/')[-1]
            last_update = file.split('_')[0]
            init = utilities.get_init_time(last_update)
            
            if init > true_init:
                true_init = init
        
        return torp_dict, true_init
    
    @staticmethod
    def gen_wofs_points_gdf(torp_dict, true_init, lead_time_int):
        '''This method will return a gdf with wofs_i and wofs_j values along with
        the associated torp_id. This will allow for easily applying TORP object
        predictors to each point on the wofs map.
        
        After extrapolation is implemented, this will need to be updated to specify
        the lead time at which the resulting wofs points and torp_ids are valid.'''
        
        i = 0
        for long_id in torp_dict:
            l = td[long_id]
            t = l.array[0]
            if not (t.init_start == true_init):
                continue
            if i == 0:
                gdf = t.get_wofs_overlap_points(wofs_gdf, torp_dict, lead_time_int)
                i += 1
            else:
                gdf = t.get_wofs_overlap_points(wofs_gdf, torp_dict, lead_time_int, gdf)
        
        return gdf
    
    @staticmethod
    def overlap_gdf_to_npy(gdf, torp_dict):
        wofs_i = gdf.wofs_i.values
        wofs_j = gdf.wofs_j.values
        
        for t_id in torp_dict:
            t = torp_dict[t_id]
            predictors = t.array[0].predictors
            break
        
        npy_predictors_dict = {}
        for predictor in predictors:
            npy_predictors_dict[predictor] = np.zeros((300, 300))
        
        for m in range(len(wofs_i)):
            i = wofs_i[m]
            j = wofs_j[m]
            torp_predictors = torp_dict[gdf.torp_id.values[m]].array[0].predictors
            for predictor in npy_predictors_dict:
                array = npy_predictors_dict[predictor]
                array[i, j] = torp_predictors[predictor]
                npy_predictors_dict[predictor] = array
        
        i = 0
        for predictor in npy_predictors_dict:
            array_2d = npy_predictors_dict[predictor]
            array_1d = array_2d.reshape((90000,1))
            if i == 0:
                full_npy = array_1d
                i += 1
            else:
                full_npy = np.append(full_npy, array_1d, axis = 1)
        
        return full_npy

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


