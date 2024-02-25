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



def add_gridded_field(in_xr, gridded_field, name):
    '''' Adds gridded field (e.g., latitudes) to xarray of predictors
        @in_xr is the xarray to add the predictors to
        @gridded_field (ny, nx) is the 2-d field to add to the xarray
        @name is the name that this new field will have in the xarray
    '''

    in_xr[name] = (["y", "x"], gridded_field)
    
    return in_xr



def add_convolutions(in_xr, footprint_type, allFields, allMethods, \
                singlePtFields, pred_radii_km, grid_spacing_km):

    ''' Adds the convolutions within various radii to the predictor
        xarray. 
        @in_xr is the xarray of predictors
        @footprint_type is "square" or "circle" depending on how 
            convolutions should be done.
        @allFields is a list of all the predictor fields (in ml name
            convention)
        @allMethods is a corresponding list of convolution methods/
            strategies (i.e., "max", "min", "abs", "minbut") 
        @singlePtFields is a list of the points where we will NOT
            take convolutions; i.e., we will only use the value at
            a single point. 
        @pred_radii_km is a list of radii (in km) over which to 
            take the convolutions
        @grid_spacing_km is the grid spacing of wofs in km
    '''


    return 


