#####################################################
# Script to map the reps to the wofs grid for a given
# time window and spatial scale
##################################################### 

#==============
# Imports 
#==============
import warnings 
#Ignore user warnings
warnings.filterwarnings("ignore", category=UserWarning)

from shapely.geometry import Point, MultiPolygon, Polygon
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import netCDF4 as nc
import pandas as pd
import math
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
import sys
import geopandas as gpd
import multiprocessing as mp
import itertools
from multiprocessing.pool import Pool
from datetime import datetime
from skexplain.common.multiprocessing_utils import run_parallel, to_iterator
import os
import copy



#==================
# Functions
#==================


#Gets the wofs geodataframe 
#WOFS_lats is the wofs latitudes
#WOFS_lons is the wofs longitudes
#nnny is number of y-grid points
#nnnx is number of x-grid points
def get_wofs_gdf(WOFS_lats, WOFS_lons, nnny, nnnx):

    points = []
    wofs_i = []
    wofs_j = []
    for j in range(nnny):
        for i in range(nnnx):
            pt = Point((WOFS_lons[j,i], WOFS_lats[j,i]))
            points.append(pt)
            wofs_j.append(j)
            wofs_i.append(i)


    wofs_j = np.array(wofs_j)
    wofs_i = np.array(wofs_i)
    wofs_df_dict = {"wofs_j": wofs_j, "wofs_i": wofs_i}

    gdf_wofs = gpd.GeoDataFrame(data=wofs_df_dict, geometry=points, crs="EPSG:4326")

    return gdf_wofs

#Finds the grid statistics based on a wofs netCDF file (@templateFile) 
def find_grid_stats(templateFile):

    ds = nc.Dataset(templateFile)
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

    return ny, nx, wofsLats, wofsLons, Tlat1, Tlat2, Stlon, SW_lat, NE_lat, SW_lon, NE_lon

#decomposes a given time string (e.g., "0025") into hours and minutes 
#e.g., "0025" returns 00, 25
def get_hhmm(in_str_time):
    hh = str(in_str_time)[0:2]
    mm = str(in_str_time)[2:4] 
    return hh, mm 

#Takes a time in hours and minutes and converts to minutes 
def to_minutes(hours_in, minutes_in):

    return hours_in*60 + minutes_in

#Converts total minutes into hhmm format 
def minutes_to_hh_mm(in_minutes):

    total_time = in_minutes/60.0
    total_hours = math.floor(total_time)
    total_minutes = round((total_time - total_hours)*60.0)

    if (total_hours >= 24):
        total_hours = total_hours - 24

    hour_str = str(total_hours).zfill(2)
    min_str = str(int(total_minutes)).zfill(2)

    return "%s%s" %(hour_str, min_str)

#Computes the end time (string) given the start time and length of the time window 
#@StartTime is a string indicating start time 
#@TimeWindow is an integer indicating the length of the forecast period in minutes 
def findEndTime(StartTime, TimeWindow):
    
    #First, decompose StartTime to hours and minutes 
    hhi, mmi = get_hhmm(StartTime) 
    
    #Next, convert to sum minutes -- essentially, minutes since 00z (the day before if after 00z) 
    sum_minutes = to_minutes(int(hhi), int(mmi))
   
    #Add the time window 
    sum_minutes += TimeWindow

    #Now convert back to hhmm format 
    final_time = minutes_to_hh_mm(sum_minutes) 

    return final_time

#Returns GeoDataFrame of reports given the filename of the reps coordinates 
#(which contain time, latitude, and longitude of each report) between certain
#start and end times
#@repsFile: The name of the .txt file with the time, lon, and lat data for each report
#@StartTime: The string start time
#@EndTime: The string end time 
#@reportBuffer: The report buffer time in minutes (integer or float). i.e., count reps before and after
#the time period by reportBuffer minutes (or less) as belonging to the given time period 
def get_reps_gdf(repsFile, StartTime, EndTime, reportBuffer):

    reps_df = pd.read_csv(repsFile, names=['time', 'lon', 'lat'], dtype='float32', header=None)

    #Add in the sum time (i.e., add 2400 for stuff after 00z (or before 12z the next day)) 
    reps_df['sum_time'] = reps_df['time']
    reps_df['sum_time'].where(reps_df['sum_time']>=1200, other=reps_df['sum_time']+2400, inplace=True)
    reps_df['latlon'] = [Point(reps_df.iloc[a,1], reps_df.iloc[a,2]) for a in range(len(reps_df))]

    #Obtain the subset of reports within the given start and end times 
    #low_time = float(StartTime)
    #high_time = float(EndTime)

    buffer_low_time = float(int(findEndTime(StartTime, -reportBuffer)))
    buffer_high_time = float(int(findEndTime(EndTime, reportBuffer)))
    
    if (buffer_low_time < 1200):
        buffer_low_time = buffer_low_time + 2400
    
    if (buffer_high_time < 1200): 
        buffer_high_time = buffer_high_time + 2400 


    #if (float(StartTime) < 1200):
    #    low_time = low_time + 2400 
    #
    #if (float(EndTime) < 1200):
    #    high_time = high_time + 2400
   

    #NOTE: This is a technical bug. Need to convert things to make sure it's not. Can't just linearly add reportBuffer
    #to high time. Need to convert. 

    #buffer_low_time = findEndTime(low_time
    #buffer_high_time = 

    #Original 
    #subset_reps_df = reps_df.loc[(reps_df['sum_time'] >= low_time - reportBuffer) & (reps_df['sum_time'] < high_time + \
    #                    reportBuffer)] 

    subset_reps_df = reps_df.loc[(reps_df['sum_time'] >= buffer_low_time) & (reps_df['sum_time'] <= buffer_high_time)]

    #print (StartTime, EndTime) 
    #print ("reps_df")
    #print (subset_reps_df) 

    #Convert to geodataframe 
    repsGDF = gpd.GeoDataFrame(data=subset_reps_df, geometry=subset_reps_df['latlon'], crs="EPSG:4326")
    
    #print ("repsGDF:") 
    #print (repsGDF) 
    #print ("****") 
    #quit() 

    return repsGDF

#Returns merged wofs and reps geodataframe (merged based on nearest point within 
#maxDistance
#NOTE (from docs): Results will include multiple output records for a single input record where 
#there are multiple equidistant nearest or intersected neighbors (probably fine because
#we apply a buffer anyway) 
#@wofsGDF is incoming wofs geodataframe
#@repsGDF is incoming reports geodataframe 
#@maxDistance is float of maximum distance to search within, in km
def merge_wofs_and_reps(wofsGDF, repsGDF, maxDistance):

    #Convert maxDistance to meters
    search_radius = maxDistance*1000.0 
    
    #Convert both geodataframes to meters geometry
    wofsGDF.to_crs("EPSG:32634", inplace=True) 
    repsGDF.to_crs("EPSG:32634", inplace=True) 
        
    #Apply the merger 
    
    out_gdf = wofsGDF.sjoin_nearest(repsGDF, how='inner', max_distance=search_radius)

    #Convert back to latlon coords
    out_gdf.to_crs("EPSG:4326", inplace=True)
    wofsGDF.to_crs("EPSG:4326", inplace=True) 
    repsGDF.to_crs("EPSG:4326", inplace=True) 


    return out_gdf

#Returns binary grid of "footprints" defining the circular kernel 
#given the sizes of the square neighborhood (n_sizes), the radii of the neighborhoods (km), 
#and the grid spacing in km 
def get_footprints_3d(n_sizes, radii_km, km_spacing):

    grids = [] #Will hold a list of grids 
    for n in range(len(n_sizes)):
        n_size = n_sizes[n]
        spatial_radius = radii_km[n]
        grid = np.zeros((1, n_size, n_size)) #Need to make 3D b/c we're working with (nT, nY, nX) 
        center = n_size//2



        #Now, traverse the grid and test if each point falls within circular radius
        for i in range(n_size):
            for j in range(n_size):
                #Compute distance between (i,j) and center point
                dist = (((i-center)**2 + (j-center)**2)**0.5)*km_spacing #want distance in km
                if (dist <= spatial_radius):
                    grid[0,i,j] = 1.0

        grids.append(grid)

    return grids


#Returns binary grid of "footprints" defining the circular kernel 
#given the sizes of the square neighborhood (n_sizes), the radii of the neighborhoods (km), 
#and the grid spacing in km
def get_footprints_2d(n_sizes, radii_km, km_spacing):

    grids = [] #Will hold a list of grids 
    for n in range(len(n_sizes)):
        n_size = n_sizes[n]
        spatial_radius = radii_km[n]
        grid = np.zeros((n_size, n_size)) #Need to make 3D b/c we're working with (nT, nY, nX) 
        center = n_size//2

        #Now, traverse the grid and test if each point falls within circular radius
        for i in range(n_size):
            for j in range(n_size):
                #Compute distance between (i,j) and center point
                dist = (((i-center)**2 + (j-center)**2)**0.5)*km_spacing #want distance in km
                if (dist <= spatial_radius):
                    grid[i,j] = 1.0

        grids.append(grid)

    return grids


#Binarizes the wofs grid (1 with observed LSR, 0 otherwise) and applies the spatial 
#radius 
#Returns a 3-d array of convolved binarized fields. (ny,nx,nn), where nn is the number of 
#spatial neighborhoods
#@in_gdf is the geodataframe containing the wofs points associated with LSRs (within the proper
#time window, including buffer) 
#@Ny is the number of y points on the wofs grid
#@Nx is the number of x points on the wofs grid 
#@ObsRadii_km is the list of observation radii in km 
#@conv_footprint_list is a list of the 2-d convolution footprints. Each list element corresponds
#to one element in ObsRadii_km
def binarize_wofs(in_gdf, Ny, Nx, ObsRadii_km, conv_footprint_list):

    n_neighborhoods = len(ObsRadii_km) 
    
    #First, create an initial wofs grid and binarize
    wofs_binary = np.zeros((Ny, Nx))

    num_reps = len(in_gdf) 
    jpts = in_gdf['wofs_j'].values
    ipts = in_gdf['wofs_i'].values

    for n in range(num_reps):
        #Get the wofs point
        jpt = jpts[n]
        ipt = ipts[n]
        wofs_binary[jpt,ipt] = 1.0 

    #Now, we can apply the convolution -- and save to 3d array
    conv_wofs_binary = np.zeros((Ny, Nx, n_neighborhoods))

    for n in range(n_neighborhoods):
        footprint = conv_footprint_list[n]
        conv_field = maximum_filter(wofs_binary, footprint=footprint)
        conv_wofs_binary[:,:,n] = conv_field


    return conv_wofs_binary

#Saves the (convolved) lsr grids (sampled and full) to specified files 
#@lsr_grids is a 3-d array of lsr grids (ny,nx,nr), where nr is number of spatial radii
#@ObsRadii_km is a list of observation radii in km (integer)
#@rand_inds is a list of (1D) random indices that has been used for subsampling the wofs files
#@Outdir_full is the outdirectory for the full npy files 
#@Outdir_dat is the outdirectory for the sampled dat files 
#@haz is the string hazard name (e.g., "hail", "wind", "torn") 
#@start_valid is the start valid time period
#@end_valid is the end valid time period 
#@wofs_i_time is the wofs initialization time (4-character string) 
#@fdate is the forecast date (does not update after 00z ) 
#@Ny is number of wofs y points
#Nx is number of wofs x points
def save_files(lsr_grids, ObsRadii_km, rand_inds, Outdir_full, Outdir_dat, haz, start_valid, end_valid, \
                wofs_i_time, fdate, Ny, Nx):

    #Total number of points
    n_points = int(Ny*Nx) 

    n_neighborhoods = len(ObsRadii_km)

    for n in range(n_neighborhoods):
        outRadius = float(ObsRadii_km[n])
        lsr_grid = lsr_grids[:,:,n] 

        full_npy_two_d_name = "%s/%s_reps2d_%s_v%s-%s_r%skm.npy" %(Outdir_full, haz, fdate, start_valid, end_valid, \
                            str(outRadius))
        full_npy_name = "%s/%s_reps1d_%s_v%s-%s_r%skm.npy" %(Outdir_full, haz, fdate, start_valid, end_valid, \
                            str(outRadius))

        dat_name = "%s/%s_reps1d_%s_%s_v%s-%s_r%skm.dat" %(Outdir_dat, haz, fdate, wofs_i_time, start_valid, end_valid, \
                            str(outRadius))
        

        #Convert lsr grid to 1d 
        lsr_grid_1d = lsr_grid.reshape(n_points, -1) 

        #Save full npy -- both 2-d and 1-d

        lsr_grid = np.float32(lsr_grid) 
        lsr_grid_1d = np.float32(lsr_grid_1d)

        #lsr_grid.tofile(full_npy_two_d_name)
        #lsr_grid_1d.tofile(full_npy_name) 
        np.save(full_npy_two_d_name, lsr_grid)
        np.save(full_npy_name, lsr_grid_1d)

        #Apply the random samples
        sampled_grid = lsr_grid_1d[rand_inds] 

        #Save random samples dat 
        sampled_grid = np.float32(sampled_grid) 
        sampled_grid.tofile(dat_name) 



    return 



#Gets the gridded reports for the given start and end time
#Updated for parallelization
#@hazard_ind is the integer index corresponding to unique hazard
#@hazard: string showing which hazard this is. E.g., "wind", "hail", "torn"
#@long_hazards: string showing which hazard, but completely spelled out (e.g., "wind", "hail", "tornado") 
#@date: Current date (the before-00z date)
#@timeBuffer: Amount of time buffer to add (in minutes) 
#@startTime: (string) Start time of period in question (excluding buffer)
#@timeWindow: (integer) Length of time window/forecast window in minutes 
#@wofs_template_dir: directory for template wofs file
#@wofs_template_iTime: initialization time of template wofs file  
#@repsDir is the reports directory 
#NOTE: We will ultimately parallelize this method 
#@maxDist: float or integer of the maximum distance (in km) needed to associate a report to a wofs point
#(i.e., when to stop searching for nearest wofs point). 
#@obsRadii_km is a list of (floats) observation radii in km
#@nSizes_obs is a list of neighborhood sizes (in wofs grid points) corresponding to obsRadii_km
#@wofs_spacing is the wofs horizontal grid spacing (in km; float) 
#@samplingDir is the main/top directory that hold the random indices .npy files (showing which points were sampled
#for wofs training) 
#@init_time is wofs initialization time (string; e.g., "2200" or "0130")
#@outdirFull is the directory where the full npy files should be saved
#@outdirDat is the directory where the sampled dat files should be saved 

def get_gridded_reps(hazard_ind, hazard_list, long_hazard_list, date, timeBuffer, startTime, \
                        init_time, timeWindow, wofs_template_iTime, repsDir, maxDist, obsRadii_km, \
                        nSizes_obs, wofs_spacing, samplingDir, outdirFull, outdirDat):

    hazard = hazard_list[hazard_ind]
    long_hazard = long_hazard_list[hazard_ind]
    #startTime = startTimes[startTimeInd]
    #init_time = init_times[startTimeInd]

    #Get wofs stats 
    wofs_template_dir = "/work/mflora/SummaryFiles/%s/%s" %(date, wofs_template_iTime)
    wofs_template_file = "%s/wofs_ENV_00_%s_%s_%s.nc" %(wofs_template_dir, date, wofs_template_iTime, wofs_template_iTime)
    try: 
        ny, nx, wofs_lats, wofs_lons, tlat1, tlat2, stlon, sw_lat, ne_lat, sw_lon, ne_lon = find_grid_stats(wofs_template_file)
    except FileNotFoundError:
        #print ("%s file not found. Skipping and moving on to the next iteration." %wofs_template_file) 
        return 


    #Define endTime of period based on startTime and timeWindow
    endTime = findEndTime(startTime, timeWindow) 

    #Sampling file 
    sample_inds_file = "%s/%s/dat/rand_inds_%s_%s_v%s-%s.npy" %(samplingDir, str(timeWindow), date, init_time, startTime,\
                            endTime)

    #Try to load the sample inds file. If it doesn't exist, then we can skip this and go to the next iteration
    try:
        random_inds = np.load(sample_inds_file)
    except FileNotFoundError: 
        #print ("%s file not found. Skipping and moving on to the next iteration." %sample_inds_file) 
        return 

    #Get reps file 
    reps_file = "%s/%s_coords_%s.txt" %(repsDir, long_hazard, date)
    #print (reps_file) 
    #quit() 

    #Now, load the data --TODO: Come back here. 
    #reps_gpd = gpd.read_csv(reps_file, names=['time', 'lon', 'lat',)

    #Probably make below a function 
    try: 
        reps_df = pd.read_csv(reps_file, names=['time', 'lon', 'lat'], dtype='float32', header=None) 
    except FileNotFoundError:
        #print ("%s file not found. Skipping and moving on to the next iteration." %reps_file)
        return 

    reps_gdf = get_reps_gdf(reps_file, startTime, endTime, timeBuffer) 

    #print (startTime, endTime) 
    #print (reps_gdf) 
    #print ("*****") 
    
    #print ("reps_gdf") 
    #print (reps_gdf) 
    #Now get the wofs geodataframe 

    wofs_gdf = get_wofs_gdf(wofs_lats, wofs_lons, ny, nx) 

    #merge wofs and reps geodataframes
    #Wait...Might not be what we want... 
    #new_gdf = wofs_gdf.sjoin_nearest(reps_gdf, how='inner', predicate='intersects') 

    #Merge wofs and reps geodataframes -- find closest wofs point, within a certain
    #max distance
    new_gdf = merge_wofs_and_reps(wofs_gdf, reps_gdf, maxDist) 

    #print ("new_gdf") 
    #print (new_gdf) 
    #print ("done") 
    #stahp
    #quit() 

    #print (new_gdf)
    #print (len(new_gdf))
    #quit() 

    #Get the circular footprints 
    footprints = get_footprints_2d(nSizes_obs, obsRadii_km, wofs_spacing)

    #TODO Come back here
    #Next, binarize wofs grid. Assign all points in new_gdf a value of 1, all others 0.
    #And do for the different neighborhoods 
    wofs_grids = binarize_wofs(new_gdf, ny, nx, obsRadii_km, footprints)

    #print ("done" )  
    #Save the file for each neighborhood
    save_files(wofs_grids, obsRadii_km, random_inds, outdirFull, outdirDat,\
                hazard, startTime, endTime, init_time, date, ny, nx) 

    #Apply the random indices 

    #Save the sampled file 

    return 



#=========================
# User defined parameters
#=========================

#Ignore user warnings
#warnings.filterwarnings("ignore", category=UserWarning)

dx_km = 3.0 #horizontal grid spacing in km

dx_r_km = dx_km/2.0

n_threads = 30 #number of cores for multiprocessing

sampling_top_dir = "/work/eric.loken/wofs/parallelized_v10/fcst" 

outdir_full_npy = "/work/eric.loken/wofs/parallelized_v10/obs/full_npy"
outdir_dat = "/work/eric.loken/wofs/parallelized_v10/obs/dat"

#time_windows = [30, 60, 90, 180] #in minutes
time_windows = [30, 60, 90, 180] 
#time_windows = [30] 

#obs_radii = ["7.5", "15.0", "30.0", "39.0"] #in km 
obs_radii_km = [7.5, 15.0, 30.0, 39.0] 
#obs_radii_pts = [math.ceil((o-dx_km/2)/dx_km) for o in obs_radii_km]
n_sizes_obs = [(round(o/dx_km)*2 + 1) for o in obs_radii_km]

time_buffer = 10 #Time buffer to apply to the reports, in minutes

#Wait, what if I apply the obs_radii at this stage? Do I then not have to smooth? 
#maximum_wofs_distance = 2.13 #in km ; Stop searching for nearest wofs point if distance exceeds this

hazards = ["hail", "wind", "torn"]
long_hazards = ["hail", "wind", "tornado"] 


#Stop searching for nearest wofs point if distance (in km) exceeds this. This is the maximum farthest
#distance (in km) that the point could probably be. 
#maximum_wofs_distance = dx_r_km*math.sqrt(2) + 0.0001
#maximum_wofs_distance = 2.5 #Before, with the above code, not all reports were getting plotted
maximum_wofs_distance = dx_km
#(likely due to rounding error).

#print (maximum_wofs_distance)
#quit() 

date_file = "probSevere_dates.txt" 
dates = np.genfromtxt(date_file, dtype='str') 
#dates = ["20190430"] 

main_init_time = "2200" #For wofs template file 

reps_dir = "/work/eric.loken/wofs/new_torn/storm_events_reports/fromThea"

#template_dir = "/work/mflora/SummaryFiles/%s/%s" %(date, main_init_time)
#template_file = "%s/wofs_ENV_00_%s_%s_%s.nc" %(template_dir, Use_Date, init_time, init_time)
#========================
    
#Test the script 
#current_date = "20190506" 
#hazard = "hail" 
#long_hazard = "hail" 
#wofs_init_time = "0000" 
##start_time = "2225" 
#start_time = "0025" 
#time_window = 180

#dates = ["20190506"] 


#NOTE: Will probably eventually need files every 5 minutes. For now this is fine though. 
start_times = ["1925", "1955", "2025", "2055", "2125", "2155", "2225", "2255", "2325", "2355",\
                "0025", "0055", "0125", "0155", "0225", "0255", "0325", "0355", "0425", "0455"] 
wofs_init_times = ["1900", "1930", "2000", "2030", "2100", "2130", "2200", "2230", "2300", "2330", \
                "0000", "0030", "0100", "0130", "0200", "0230"] 


#start_times = ["1955"]
#wofs_init_times = ["1930"] 

#dates, time_windows, hazards, start_times 

#OK--works! Now just need to parallelize! 

#itr = to_iterator(hazard_inds, [hazards], [long_hazards], [dates], [time_buffer], start_time)

hazard_inds = np.arange(len(hazards))
start_time_inds = np.arange(len(start_times))

args_itr = to_iterator(hazard_inds, [hazards], [long_hazards], dates, [time_buffer], start_times, \
                    wofs_init_times, time_windows, [main_init_time], [reps_dir], [maximum_wofs_distance], \
                    [obs_radii_km], [n_sizes_obs], [dx_km], [sampling_top_dir], [outdir_full_npy], [outdir_dat])


run_parallel(get_gridded_reps, args_itr, n_jobs = n_threads) 

#original: 
#get_gridded_reps(hazard, long_hazard, current_date, time_buffer, start_time, time_window, main_init_time, reps_dir,\
#                    maximum_wofs_distance, obs_radii_km, n_sizes_obs, dx_km, sampling_top_dir, wofs_init_time,\
#                    outdir_full_npy, outdir_dat) 




