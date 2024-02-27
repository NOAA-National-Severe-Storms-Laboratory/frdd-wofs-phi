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


def get_footprints(predRadiiKm, gridSpacingKm):

    '''Gets/returns list of binary grid of "footprints" defining the circular 
        kernel for each element in predRadiiKm
        @predRadiiKm is the array containing the radii of the neighborhoods
            (km)
        @gridSpacingKm is the grid spacing of WoFS in km
    '''
    n_neighborhoods = len(predRadiiKm) 

    #First, need to compute the n_sizes -- i.e., size (one of the dimensions)
    #of the footprint grid 
    n_sizes = compute_radii_sizes(predRadiiKm, gridSpacingKm)


    grids = [] #Will hold a list of grids corresponding to each neighborhood
    for n in range(n_neighborhoods):
        n_size = n_sizes[n] 
        spatial_radius = predRadiiKm[n]
        grid = np.zeros((n_size, n_size)) 
        center = n_size//2    

        #Now, traverse the grid and test if each point falls within the cicrcular radius 
        for i in range(n_size):
            for j in range(n_size):
                #Compute distance between (i,j) and center point 
                dist = (((i-center)**2 + (j-center)**2)**0.5)*gridSpacingKm #want distance in km
                if (dist <= spatial_radius):
                    grid[i,j] = 1.0

        grids.append(grid) 
   

    return grids


def compute_radii_sizes(radii_km_list, horiz_spacing_km):
    '''Computes/returns the list of square "diameter" of wofs grid points 
        to consider (@nSizes) 
        given an array of radii in im (@radii_km_list)
        and the horizontal grid spacing in km (@horiz_spacing_km) 
    '''

    nSizes = [(round(p/horiz_spacing_km)*2 + 1) for p in radii_km_list] 


    return nSizes


def radii_list_to_points_list(radii_km_list, horiz_spacing_km):
    '''Converts a list of radii in km (@radii_km_list)
         to a list of radii in grid points given the horizontal 
         grid spacing of the grid in km (@horiz_spacing_km) 
         @Returns the list of radii in grid points (@radii_pts_list)'''

    radii_pts_list = [math.ceil((p-horiz_spacing_km/2)/horiz_spacing_km) \
                        for p in radii_km_list]

    return radii_pts_list


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

    #Obtain circular footprint if necessary 
    if (footprint_type == "circle"):
        circular_footprints = get_footprints(pred_radii_km, grid_spacing_km)


    #Compute the predictor radii in grid point coordinates 
    pred_radii_pts = radii_list_to_points_list(pred_radii_km, grid_spacing_km) 


    #We will add the convolutions as a separate variable. 
    #Each convolution will be named according to the km_radius 

    #Span the different radii/convolutions
    #Starting at 1 because the first radius is always 0; i.e., no convolution, 
    #so we can skip. 
    for r in range(1,len(pred_radii_km)):
        km_radius = pred_radii_km[r]
        pt_radius = pred_radii_pts[r]
        if (footprint_type == "circle"):
            circular_footprint = circular_footprints[r] 

        #Span the variables -- ml naming convention -- and apply the convolution
        #if necessary 
        for v in range(len(allFields)):
            var_name = allFields[v]
            var_method = allMethods[v] 
            
            #Only add the convolution if the variable is not in singlePtFields
            if (var_name not in singlePtFields): 

                #Add the convolution 
                orig_field = in_xr[var_name].values

                #Do the convolution based on the convolution method/strategy
                if (var_method == "max"):
                    if (footprint_type == "circle"):
                        conv = maximum_filter(orig_field, footprint = circular_footprint)
                    elif (footprint_type == "square"):
                        #NOTE: Have to give tuple (0,radius, radius) to prevent the smoothing from happening in time 
                        conv = maximum_filter(orig_field, (pt_radius,pt_radius))
                elif (var_method == "min"):
                    if (footprint_type == "circle"):
                        conv = minimum_filter(orig_field, footprint=circular_footprint)
                    elif (footprint_type == "square"):
                        conv = minimum_filter(orig_field, (pt_radius,pt_radius))
                elif (var_method == "abs"):
                    if (footprint_type == "circle"):
                        conv_hi = maximum_filter(orig_field, footprint=circular_footprint)
                        conv_low = minimum_filter(orig_field, footprint=circular_footprint)
                    elif (footprint_type == "square"):
                        conv_hi = maximum_filter(orig_field, (pt_radius,pt_radius))
                        conv_low = minimum_filter(orig_field, (pt_radius,pt_radius))

                    #now, take the max absolute value between the two
                    conv = np.where(abs(conv_low) > abs(conv_hi), conv_low, conv_hi)

                elif (var_method == "minbut"): #Handle the case where we want to take min, but keep -1 if there is no meaningful min
                    #Can start by renaming the -1--assigning this a high value 
                    conv_hi = np.where(orig_field == -1, 999999.9, orig_field)

                    if (footprint_type == "circle"):
                        conv_low = minimum_filter(conv_hi, footprint=circular_footprint)
                    elif (footprint_type == "square"):
                        conv_low = minimum_filter(conv_hi, (pt_radius,pt_radius))
     
                    #Now reassign 999999.9 to -1
                    conv = np.where(conv_low==999999.9, -1, conv_low)

                #Now, add the convolved field as a new variable 
                new_name = "%s_r%s" %(var_name, km_radius)

                in_xr[new_name] = (["y", "x"], conv)


    return in_xr


