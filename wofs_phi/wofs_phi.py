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


    def __init__(self, wofs_files, ps_files, ps_direc, wofs_direc, nc_outdir): 

        ''' @wofs_files is the list of wofs_files to use for the prediction (in chronological order,
            beginning with the start of the prediction window/valid period and ending with the end of
            the prediction window/valid period. 

            @ps_files are the list of probSevere files needed in reverse chronological order (i.e.,
            starting with the most recent file/time step. So we'd have t = 0, -2, -10, -14, -30, -44,
            -60, -74, -90, -104, -120, -134, -150, -164, -180 minutes

            @ps_direc is the string path to the probSevere files 

            @wofs_direc is the string path to the wofs files   

            @nc_outdir is the directory to save the final .ncdf files to 
            
        '''

        self.wofs_files = wofs_files
        self.ps_files = ps_files
        self.ps_direc = ps_direc
        self.wofs_direc = wofs_direc
        self.nc_outdir = nc_outdir
        self.pkl_dir = pkl_dir 


    def generate(self):
        '''Instance method to generate the predictions given an MLGenerator object.
            Kind of like the "main method" for generating the predictions. ''' 

        #TODO: Should build roadmap 
       
        # Get grid stats from first wofs file 
        

        #Do PS preprocessing -- parallel track 1

        
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

class PS:
    '''Handles the ProbSevere forecasts/processing'''

    def __init__(self):
        pass


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


