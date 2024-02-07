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
from mpl_toolkits.basemap import Basemap
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
from skexplain.common.multiprocessing_utils import run_parallel, to_iterator
import netCDF4 as nc
import os
import copy
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xarray as xr


class Setup: 


    ''' Handles the setup/initialization aspects of wofs-phi, including 
        parameters passed in as well as constants set by the system'''

    #======================================================
    #Class "constants" that can be adjusted in-code here. 
    #======================================================

    is_train_mode = False #True if we're using this script for training; False if used for testing
    is_on_cloud = False # True if we're using this script on the cloud; False if using locally
    forecast_mode_time_window = 30 #time window of forecast mode in minutes (e.g., 30 means 30-min forecasts)
    warning_mode_time_window = 90 #in minutes; e.g., 90 means predict 0-90 minutes
    max_cores = 30 #maximum number of computing cores to use for parallelization

    wofs_dir = "" #Directory to wofs files 
    ps_dir = "" #Directory to PS files 
    nc_outdir = "./" #Directory to save the .ncdf files to 
    pkl_dir = "" #Directory pointing to the trained pkl files 

    dx_km = 3.0 #horizontal grid spacing in km
    ps_thresh = 0.01 #ps objects must have probs greater than or equal to this amount to be considered
    dm = 18 #number of wofs members 
    min_radius = 1.5 #in km (for probSevere objects) 
    max_radius = 1.5 #in km (for probSevere objects) #Used to be 30.0, but that was much too big
    conv_type = "square" #"square" or "circle" -- tells how to do the convolutions 
    predictor_radii_km = [0.0, 15.0, 30.0, 45.0, 60.0] #how far to "look" spatially in km for predictors
    obs_radii = ["30.0", "15.0", "7.5", "39.0"]
    final_str_obs_radii = ["30", "15", "7_5", "39"] #form to use for final ncdf files
    final_hazards = ["hail", "wind", "tornado"] #for naming in final ncdf file 
    bottom_hour_inits = ["1730", "1830", "1930", "2030", "2130", "2230", "2330", "0030", "0130",\
                     "0230", "0330", "0430", "0530", "0630", "0730", "0830", "0930", "1030",\
                     "1130", "1230", "1330", "1430", "1530", "1630"]

    next_day_inits = ["0000", "0030", "0100", "0130", "0200", "0230", "0300", "0330", "0400", "0430", "0500"]


    #=========================================================

        
    def __init__(self, list_of_wofs_files, list_of_ps_files, lead_time_window): 
        '''
            @list_of_wofs_files is a list of wofs summary files
            @list_of_ps_files is a list of probSevere summary files
            @lead_time_window is an integer indicating what lead time window
            this prediction is for (e.g., for 30-min windows, 1 would be 0-30, 
            2 would be 30-60, 3 would be 60-90, etc.)

        '''
        self.list_of_wofs_files = list_of_wofs_files
        self.list_of_ps_files = list_of_ps_files
        self.lead_time_window = lead_time_window


        pass


    @classmethod
    def create_setup(cls):
        ''' This is a blueprint method designed to initialize a Setup object based on user-specified inputs'''

        pass






class Wofs:
    '''Handles the wofs forecasting/processing'''

    def __init__(self):
        self.ny = ny
        self.nx = nx

        pass

class PS:
    '''Handles the ProbSevere forecasts/processing'''

    def __init__(self):
        pass


class DataGenerator: 
    '''Handles the merging of wofs and ps data (through inheriting Wofs and PS objects)'''

    def __init__(self, Wofs, PS):

        pass 

def main():
    '''Main Method'''

    #Get the key setup details 
    #Create a Setup object with the key setup/logistic details 
    #setup = Setup(currDate, currTime) 

    #For debugging 

    print ("Hello, World") 

    #TODO: Put these in and start seeing if I can get something reasonable. 
    wofs_files = [] 
    ps_files = [] 
    prediction_window = 1 #first 30 minutes 

    setup = Setup(wofs_files, ps_files, prediction_window)


    #Roadmap: #TODO 
    #Get Setup object
    

    #Do ProbSevere preprocessing 



    #Do WoFS Preprocessing 



    #
    


if (__name__ == '__main__'):

    main()


