#=====================================================
# This script will function as the main method for 
# running WoFS-PHI (v2; after refactoring). 
# i.e., Will handle functions related to actually
# running the (various pieces of) code. 
#
# Created by: Eric Loken 11/18/2025
# Previous development work on WoFS-PHI (v1) done by:
# Eric Loken, Ryan Martz, Joshua Martin. 
#=====================================================


#===============
# Imports
#===============

import sys
_wofs = '/home/eric.loken/python_packages/frdd-wofs-post'
_wofsphi = '/home/eric.loken/python_packages/frdd-wofs-phi'
_monte_python = '/home/eric.loken/python_packages/monte-python'
sys.path.insert(0, _wofs)
sys.path.insert(0, _wofsphi)
sys.path.insert(0, _monte_python) 
#sys.path.insert(0, '/home/monte.flora/python_packages/MontePython')


from shapely.geometry import Point, MultiPolygon, Polygon, LineString
from shapely.prepared import prep
from shapely import geometry
from pyproj import Transformer
from shapely.ops import transform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import netCDF4 as nc
import pandas as pd
import json
from matplotlib.patches import Polygon as PolygonMPL
import math
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
#import sys
import geopandas as gpd
import multiprocessing as mp
import itertools
from multiprocessing.pool import Pool
from datetime import datetime, timedelta
import datetime as dt
import netCDF4 as nc
import os
import copy
import re
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xarray as xr

import warnings
import cartopy.feature as cfeature
import cartopy.crs as ccrs

#Other imports here (eventually): 
import grid 
import utilities
import wofs 
import forecast_specs

#Will eventually have to take this form: 
#from . import grid 
#from . import utilities 

#from . import config as c
#from . import utilities
#from . import predictor_extractor as pex
#from wofs.common.zarr import open_dataset
#from wofs.common import remove_reserved_keys
#from wofs.post.utils import save_dataset

#from .plot_wofs_phi import plot_wofs_phi_forecast_mode, plot_wofs_phi_warning_mode


#=================================================================


class MLGenerator: 

    """This class handles the overall setup/running of the WoFS-PHI code, given
        relevant inputs."""

    def __init__(self, wofs_files, ps_files, ps_path, wofs_path, nc_outdir, mode, \
                    json_config_file, torp_files=[]): 

        """
            @wofs_files : list of wofs files (excluding path) to use for the prediction
                in chronological order, beginning with the start of the prediction window/
                valid period and ending with the end of the prediction window/valid period.
            @ps_files : list of probSevere files (excluding path) needed in reverse 
                chronological order (i.e., starting with the most recent file/time step.) 
                Typically, we go back to 180 minutes ago. 
            @ps_path : str path to the ProbSevere files 
            @wofs_path : str path to the WoFS files. 
            @nc_outdir : str path showing the directory to save the final .ncdf files to
            @mode : str either "forecast" for forecast mode or "warning" for warning mode
                predictions. 
            @json_config_file : str : Full file path to json file, which shows 
                how various user-defined variables are set for wofs-phi
            @torp_files : list of Torp files to use. Defaults as an empty list, which 
                indicates TORP will not be used. 
        """
    

        self.wofs_files = wofs_files 
        self.ps_files = ps_files
        self.ps_path = ps_path 
        self.wofs_path = wofs_path
        self.nc_outdir = nc_outdir
        self.mode = mode
        self.json_config_file = json_config_file
        self.torp_files = torp_files


    def generate(self): 
        """Generates the wofs-phi forecasts based on the instance variables provided. 
            Kind of like the de facto "main method" for generating the predictions.
        """

        #full_wofs_file = f"{self.wofs_path}/{self.wofs_files[0]}"

        #Get the forecast Grid object from the first wofs file 
        #TODO 
        fcst_grid = grid.Grid.create_wofs_grid(self.wofs_path, self.wofs_files[0])

        #Need to create forecast specifications object -- Will pass in json config
        #file
        f_specs = forecast_specs.ForecastSpecs.create_forecast_specs(self.ps_files,\
                    self.wofs_files, self.json_config_file)
        

        

        return 


#Might be worth adding methods for training and real time usage


def get_ps_files_from_times_and_dates(list_of_times, list_of_dates): 
    """ Convenience function for use in main. Converts a list of times 
            and dates into a list of Probsevere filenames. 
        @Returns : List of probSevere files in format 
            "MRMS_EXP_PROBSEVERE_{date}.{time}.json
        @list_of_times : List of 4-character strings (e.g., ["0224", "0222", ...]) 
        @list_of_dates : List of 8-character string dates of the same length
            as list_of_times. e.g., ["20210605", "20210605", ...]
    """ 

    ps_times = [f"{p}00" for p in list_of_times]

    ps_files = [f"MRMS_EXP_PROBSEVERE_{list_of_dates[p]}.{ps_times[p]}.json" \
                     for p in range(len(ps_times))]


    return ps_files 


def main(): 

    """ NOTE: You can use this main method to 'drive' the code, especially if you
        want to run locally, but this is not required. You can simply
        create an MLGenerator object and call the .generate() function.
    """


    #=========================
    # User defined parameters
    #=========================

    #NOTE: Eventually might not need these if we have the proper variables
    #set in the json file. 
    on_cloud = False #True if on cloud, False otherwise 
    is_training = True # True if using for model training; False if using for realtime 

    do_warning_mode = False #True if we want to generate warning mode predictions
    do_forecast_mode = True #True if we want to generate forecast mode predictions

    #Full path to the json file/dictionary that will set some key user-defined 
    #wofs-phi parameters. 
    json_config_file = "/home/eric.loken/python_packages/frdd-wofs-phi/wofs_phi/config_json_files/config_training.json"


    #NOTE: Will have to change these to reflect updated paths for 
    #wofs summary files and (good) probSevere data on local machines.  
    #wofs_direc = "/work/mflora/SummaryFiles/20210604/0200"
    #wofs_direc = "/work2/wof/SummaryFiles" 
    wofs_direc = "/work2/wof/SummaryFiles/20210604/0200"
    ps_direc = "/work/eric.loken/wofs/probSevere"


    #Will need to set these

    wofsFiles = ["wofs_ALL_05_20210605_0200_0225.nc", "wofs_ALL_06_20210605_0200_0230.nc", \
                    "wofs_ALL_07_20210605_0200_0235.nc", "wofs_ALL_08_20210605_0200_0240.nc",\
                    "wofs_ALL_09_20210605_0200_0245.nc", "wofs_ALL_10_20210605_0200_0250.nc",\
                    "wofs_ALL_11_20210605_0200_0255.nc"]

    #For testing second forecast period 
    wofsFiles2 = ["wofs_ALL_11_20210605_0200_0255.nc", "wofs_ALL_12_20210605_0200_0300.nc",\
                    "wofs_ALL_13_20210605_0200_0305.nc", "wofs_ALL_14_20210605_0200_0310.nc",\
                    "wofs_ALL_15_20210605_0200_0315.nc", "wofs_ALL_16_20210605_0200_0320.nc",\
                    "wofs_ALL_17_20210605_0200_0325.nc"]

    #wofsFiles = [] 
    psTimes = ["0224", "0222", "0220", "0218", "0216", "0214", "0212", "0210", "0208", "0206",\
                "0204", "0202", "0200", "0158", "0156", "0154", "0152", "0150", "0148", "0146",\
                "0144", "0142", "0140", "0138", "0136", "0134", "0132", "0130", "0128", "0126",\
                "0124", "0122", "0120", "0118", "0116", "0114", "0112", "0110", "0108", "0106",\
                "0104", "0102", "0100", "0058", "0056", "0054", "0052", "0050", "0048", "0046",\
                "0044", "0042", "0040", "0038", "0036", "0034", "0032", "0030", "0028", "0026",\
                "0024", "0022", "0020", "0018", "0016", "0014", "0012", "0010", "0008", "0006",\
                "0004", "0002", "0000", "2358", "2356", "2354", "2352", "2350", "2348", "2346",\
                "2344", "2342", "2340", "2338", "2336", "2334", "2332", "2330", "2328", "2326",\
                "2324"]

    psDates = ["20210605" for p in psTimes]
    for a in range(73, len(psTimes)):
        psDates[a] = "20210604"

    psFiles = get_ps_files_from_times_and_dates(psTimes, psDates) 

    nc_output_dir = "." 


    #=========================


    if (do_forecast_mode == True): 
    
        ml_generator = MLGenerator(wofsFiles, psFiles, ps_direc, wofs_direc,\
                            nc_output_dir, "forecast", json_config_file)

        ml_generator.generate() 

    print ("Done with forecast mode generation (if applicable)") 

    if (do_warning_mode == True): 

        ml_generator = MLGenerator(wofsFiles, psFiles, ps_direc, wofs_direc,\
                            nc_output_dir, "warning", json_config_file)

        ml_generator.generate()     

    
    print ("Done with warning mode generation (if applicable)") 

    return 




if (__name__ == '__main__'): 

    main() 


