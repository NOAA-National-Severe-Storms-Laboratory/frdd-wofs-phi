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
sys.path.insert(0, _wofs)
sys.path.insert(0, _wofsphi)

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
        self.torp_files = torp_files


    def generate(self): 
        """Generates the wofs-phi forecasts based on the instance variables provided. 
            Kind of like the de facto "main method" for generating the predictions.
        """


        return 


#Might be worth adding methods for training and real time usage


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
    wofs_direc = "/work/mflora/SummaryFiles/20210604/0200"
    ps_direc = "/work/eric.loken/wofs/probSevere"


    #Will need to set these
    wofsFiles = [] 
    psFiles = [] 

    nc_output_dir = "." 


    #=========================


    if (do_forecast_mode == True): 
    
        ml_generator = MLGenerator(wofsFiles, psFiles, ps_direc, wofs_direc,\
                            nc_output_dir, "forecast", json_config_file)

        ml_generator.generate() 


    if (do_warning_mode == True): 

        ml_generator = MLGenerator(wofsFiles, psFiles, ps_direc, wofs_direc,\
                            nc_output_dir, "warning", json_config_file)

        ml_generator.generate()     

    


    return 




if (__name__ == '__main__'): 

    main() 


