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



#Might be worth adding methods for training and real time usage


def main(): 

    #=========================
    # User defined parameters
    #=========================

    #Q: Is it worth having user-defined parameters here or in a separate
    #   file?     

    #=========================


    return 




if (__name__ == '__main__'): 

    main() 


