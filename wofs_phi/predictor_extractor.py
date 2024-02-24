#=======================================================
# This module provides methods to extract RF predictors
# from PS and Wofs objects 
#=======================================================



#=====================
# Imports 
#=====================

import numpy as np
import pickle
import netCDF4 as nc
import pandas as pd
import math
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
import sys
import geopandas as gpd
import os
import copy
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xarray as xr
import config as c


#========================

def xr_from_ps_and_wofs(psObj, wofsObj):

        '''Creates a combined xarray from a PS object (@psObj) and
            Wofs object (@wofsObj)
        '''

        #merge probSevere and WoFS xarrays 
        merged_xr = xr.merge([psObj.xarr, wofsObj.xarr])

        return merged_xr



#predictors_ds['lat'] = (["y", "x"], wofs_lats)
#predictors_ds['lon'] = (["y", "x"], wofs_lons)
def add_gridded_field(in_xr, gridded_field, name):
    '''' Adds gridded field (e.g., latitudes) to xarray of predictors
        @in_xr is the xarray to add the predictors to
        @gridded_field (ny, nx) is the 2-d field to add to the xarray
        @name is the name that this new field will have in the xarray
    '''

    in_xr[name] = (["y", "x"], gridded_field)
    
    return in_xr


