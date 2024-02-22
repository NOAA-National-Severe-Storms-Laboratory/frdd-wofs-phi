#########################################
# Master script designed to 
# handle everything related to
# WoFS-ProbSevere severe weather
# forecasts.
#########################################

############
#Imports 
############

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
from skexplain.common.multiprocessing_utils import run_parallel, to_iterator
import netCDF4 as nc
import os
import copy 
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xarray as xr 


#################################################

############
# Functions 
############

#Returns a list of start valids and end valids for a given forecast period 
#based on the start valid time (@startValid), the time window in minutes (@timeWindow),
#and the number of time windows (@nWindows) 
def find_start_end_leads(startValid, timeWindow, nWindows):

    #If the start valid time is odd, then the extrapolation lead time
    #should start at 1 (b/c probSevere files are only produced at even times)
    start_min = 1.0 
    #Even start valid times will start the extrapolation at 0
    if (int(startValid) %2 == 0):
        start_min = 0


    start_valids = [start_min + n*timeWindow for n in range(nWindows)]
    end_valids = [start_min + n*timeWindow for n in range(1, nWindows+1)] 

    return start_valids, end_valids


#Returns list of appropriately-formatted valid times (based on lead time). 
#i.e., We will take the temporal max/mean over these times. 
#start_v is starting valid time
#num_wofs is the number of wofs files (determined based on lead time)
#@increment is the time increment in minutes 
def get_curr_valids(start_v, num_wofs, increment):
    good_valids = np.arange(start_v, start_v+increment*num_wofs, increment)
    new_valids = []

    bad_mins = np.arange(60,240)
    
    count_flag = 0
    for c in range(len(good_valids)):
        #Only need to get the starting valid
        if (c == 0):
            valid = str(good_valids[c]).zfill(6)
            valid = valid[0:4]
            hh = valid[0:2]
            mm = valid[2:4]

        #if (mm == "60"):
        if (int(mm) in bad_mins):
            mm = str(int(mm) - 60 )
            #mm = "00"
            mm = str(mm).zfill(2) 
            hh = str(int(hh)+1)
            if (int(hh) >= 24):
                hh = str(int(hh) - 24)
            if (int(hh) < 10):
                hh = hh.zfill(2)

            valid = "%s%s" %(hh,mm)

        #Append valids and increment
        new_valids.append(valid)

        mm = str(int(mm)+increment).zfill(2)

        if (int(mm) in bad_mins):
            mm = str(int(mm) - 60 )
            #mm = "00"
            mm = str(mm).zfill(2)
            hh = str(int(hh)+1)
            if (int(hh) >= 24):
                hh = str(int(hh) - 24)
            if (int(hh) < 10):
                hh = hh.zfill(2)

        valid = "%s%s" %(hh,mm)

    good_valids = np.array(new_valids, dtype='str')

    return good_valids


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


#Gets the list of past PS times (for assessing object ID) 
#@minute_increment is the time in minutes for each increment (so 15 looks back at 15 min intervals) 
def get_past_ps_list(start_v, lead_time, curr_date, past_date, minute_increment):
    #only actually used as sort of a placeholder; actual heavy lifting done below. 
    #NOTE: Includes the start time itself 
    good_valids = np.arange(start_v, start_v-lead_time-minute_increment, -minute_increment)
    #print (good_valids) 
    new_valids = []

    good_dates = [] # Will store the relevant dates for files 
    #binary variable; if true, use the current date; if false, use the previous date
    use_curr = True

    count_flag = 0
    for c in range(len(good_valids)):
        #Only need to get the starting valid
        if (c == 0):
            valid = str(good_valids[c]).zfill(4)
            valid = valid[0:4]
            hh = valid[0:2]
            mm = valid[2:4]

        if (int(mm) < 0 ):
            mm = str(int(mm)+60)
            hh = str(int(hh)-1)
            if (int(hh) < 0):
                hh = str(int(hh) + 24)
                hh = hh.zfill(2)
                #use_curr = False
            if (int(hh) < 10):
                hh = hh.zfill(2)
            if (int(hh) >= 12):
                use_curr = False
            else:
                use_curr = True

            valid = "%s%s" %(hh,mm)

        #Append valids and increment
        new_valids.append(valid)
        if ( use_curr == True):
            good_dates.append(curr_date)
        else:
            good_dates.append(past_date)

        #print (mm) 
        #print (hh) 
        mm = str(int(mm)-minute_increment).zfill(2)
        valid = "%s%s" %(hh,mm)

    #good_valids = np.array(new_valids, dtype='str')
    #print (good_valids) 

    #Now modify: Since PS is only every 2 minutes: Get rid of the odd values. Round up
    #Round up to nearest even integer. 
    keep = [math.ceil((int(g)/2))*2 for g in new_valids]
    keep = [str(k).zfill(4) for k in keep]

    #keep = np.array(keep, dtype='str') 

    #return keep, good_dates

    return keep, good_dates


#Returns a dataframe with object ids, hazard probabilities, and lead times 
#aage is (approx) object age in minutes (integer) 
def get_ids_probs_ages(json_file, aage, in_df, names):

    try:
        f = open(json_file)
        ps_data = json.load(f)

    except FileNotFoundError:
        print ("%s not found. Adding as if it had no information" %json_file)
        return in_df

    except json.decoder.JSONDecodeError:
        print ("%s Extra data in file. Proceeding as if it had no information." %json_file)
        return in_df

    hail_probs = []
    wind_probs = []
    torn_probs = []
    ids = []

    if (len(ps_data['features']) == 0):
        return in_df

    for i in ps_data['features']:
        ids.append(i['properties']['ID'])

        #NOTE: Let's get all hazards 

        hail_probs.append(float(i['models']['probhail']['PROB'])/100.)
        torn_probs.append(float(i['models']['probtor']['PROB'])/100.)
        wind_probs.append(float(i['models']['probwind']['PROB'])/100.)

    aages = np.ones(len(ids))*aage

    df = pd.DataFrame(list(zip(ids, hail_probs, torn_probs, wind_probs, aages)), columns=names)

    out_df = pd.concat([in_df, df], axis=0, ignore_index=True, copy=False)

    return out_df


def get_prev_ps_df(columnNames, psPreviousList, psDateList, ps_direc, age_list):
    #Get past ps files 

    past_ps_files = []
    for p in range(len(psPreviousList)):
        ps_previous = psPreviousList[p]
        ps_date = psDateList[p]
        past_ps_file = "%s/MRMS_EXP_PROBSEVERE_%s.%s00.json" %(ps_direc, ps_date, ps_previous)
        age = age_list[p] 
        
        if (p == 0): #start an empty dataframe 
            prev_df = pd.DataFrame(columns = columnNames)

        #Get the relevant information from the previous files 
        prev_df = get_ids_probs_ages(past_ps_file, age, prev_df, columnNames)


    return prev_df 



#Reads ProbSevere json file--
#Returns a list of each objects' lat points, lon points, and tornado probabilities. 
#Also want to get object storm motion 
#Returns geopandas array with these attributes. 
def read_probSevere(json_file):
    f = open(json_file)
    ps_data = json.load(f)
    points = []
    hail_probs = []
    torn_probs = []
    wind_probs = []
    east_motion = []
    south_motion = []
    ids = []
    for i in ps_data['features']:
        east_motion.append(float(i['properties']['MOTION_EAST'])*0.06) #multiply by 0.06 to convert to km/min
        south_motion.append(float(i['properties']['MOTION_SOUTH'])*0.06) #multiply by 0.06 to convert to km/min

        points.append(i['geometry']['coordinates'])
        ids.append(i['properties']['ID'])
        hail_probs.append(float(i['models']['probhail']['PROB'])/100.)
        torn_probs.append(float(i['models']['probtor']['PROB'])/100.)
        wind_probs.append(float(i['models']['probwind']['PROB'])/100.)

    #NOTE: Might also be worth extracting other variables -- e.g., modeled vs. simulated MLCAPE, etc. 

    polygons = []

    #Can I start a geopandas array with the polygons as geometry? 

    #Span the different objects
    for objj in points:
        #print (objj[0]) 
        #print ("***") 
        pgon = Polygon(objj[0])
        polygons.append(pgon)

    df_dict = {"hail_prob": hail_probs, "torn_prob": torn_probs, "wind_prob": wind_probs, "east_motion": east_motion,\
                 "south_motion": south_motion, "id": ids}

    gdf = gpd.GeoDataFrame(data=df_dict, geometry=polygons, crs="EPSG:4326")

    return gdf

#obtains geodataframe of probSevere objects for a specific hazard
#@psFile is the name of the current probSevere file
#@backupPsFile is the name of the backup probSevere file (2-minutes previous)
#@orig_time_extrap is the extrapolation time in minutes (e.g., 181) 
def get_ps_geoDataFrame(psFile, backupPsFile, orig_time_extrap, radius_min, radius_max,\
                            origStartLeads, origEndLeads):

    #Make a copy of origStartLeads and origEndLeads
    startLeads = origStartLeads.copy() 
    endLeads = origEndLeads.copy() 
    time_extrap = orig_time_extrap

    #Start off by trying current PS file. Only use backup if that file is not available
    use_backup = False    
    
    try: 
        ps_gdf = read_probSevere(psFile)
    

    #if main PS file is missing, try the previous time 
    except OSError:
        use_backup = True
        print ("PS file not found: %s. Trying %s." %(ps_file, backup_ps_file) )
        pass

    if (use_backup == True):
        #Now try the backup file 
        try:
            #lats, lons, probs, east, south, iids = read_probSevere(backup_ps_file, hazard)
            ps_gdf = read_probSevere(backup_ps_file)

            #extrap_time = orig_extrap_time + 2.0
            time_extrap = time_extrap + 2.0 
            startLeads = startLeads + 2.0 
            endLeads = endLeads + 2.0 

            #TODO: Have to handle start_leads and end_leads 
            #start_leads = np.array(orig_start_leads) + 2.0
            #end_leads = np.array(orig_end_leads) + 2.0

        #If backup file is not found, then quit
        except OSError:
            print ("Backup PS file not found: %s. Ending for this initialization time." %backup_ps_file)
            quit() 
            #continue


    #NOTE: We need min_radius, max_radius, and ps_thresh 
    #Find the radius at each time (based on the extrapolation time) 
    adjustable_radii = np.linspace(radius_min, radius_max, int(time_extrap)+1)
    adjustable_max_x = [math.ceil((r - 1.5)/dx_km) for r in adjustable_radii]
    #adjustable_max_y = [math.ceil((r - 1.5)/dx_km) for r in adjustable_radii]

    #NOTE: I will probably now need to do this later. 
    #apply the probability threhsold; e.g., ignore objects with 0% prob.
    #ps_gdf = ps_gdf.loc[ps_gdf['prob']>=thresh_ps]

    return ps_gdf, time_extrap, startLeads, endLeads, adjustable_max_x 


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


    #points = np.array(points) 
    wofs_j = np.array(wofs_j)
    wofs_i = np.array(wofs_i)
    wofs_df_dict = {"wofs_j": wofs_j, "wofs_i": wofs_i}

    gdf_wofs = gpd.GeoDataFrame(data=wofs_df_dict, geometry=points, crs="EPSG:4326")

    #print (gdf_wofs) 

    #Now make wofs dataframe             


    return gdf_wofs



def find_age(obj_id, in_df):

    possible_leads = in_df.loc[in_df['id'] == obj_id] #need to take max of lead time columns 
    final_lead_time = max(possible_leads['age'])

    return final_lead_time

#age_change is integer over which to calculate change
#(e.g., 2 or 10 for 2- or 10-minute change) 
def find_change(obj_id, in_df, age_change):

    final = in_df.loc[(in_df['id'] == obj_id) & (in_df['age'] == 0)]
    initial = in_df.loc[(in_df['id'] == obj_id) & (in_df['age'] == age_change)]

    if (len(final) > 0 and len(initial) > 0 ):

        final_hail_prob = final['hail_prob'].iloc[0]
        initial_hail_prob = initial['hail_prob'].iloc[0]

        final_torn_prob = final['torn_prob'].iloc[0]
        initial_torn_prob = initial['torn_prob'].iloc[0]

        final_wind_prob = final['wind_prob'].iloc[0]
        initial_wind_prob = initial['wind_prob'].iloc[0]

        change_hail = final_hail_prob - initial_hail_prob
        change_torn = final_torn_prob - initial_torn_prob
        change_wind = final_wind_prob - initial_wind_prob

    else:
        change_hail = 0.0
        change_torn = 0.0 
        change_wind = 0.0

    return change_hail, change_torn, change_wind

#finds the (relative) wofs points that would be hit by extrapolating a storm object
#over time according to the supplied storm motion vectors. 
#time is the time in minutes over which to apply the extrapolation
#e_motion is the eastward storm motion in km/min
#s_motion is the southward storm motion in km/min
#km_spacing is the grid spacing of the output (e.g., wofs) grid in km
#returns a set of (unique) relative coordinates that are "hit" by the extrapolation
#e.g., (0,0) is initial point. 
#max_xy is a list of max_x/max_y points associated with each time 
def find_extrap_points(time, e_motion, s_motion, km_spacing, max_xy):

    minutes = np.arange(time+1)
    #minutes = np.arange(0, time+0.5, 0.5) 
    new_xs = [round((e_motion*m)/(km_spacing)) for m in minutes]
    new_ys = [round((-s_motion*m)/(km_spacing)) for m in minutes]


    #TODO: Add time dimension

    three_d_extrap_pts = [(new_ys[p], new_xs[p], p, max_xy[p]) for p in range(len(new_xs))]

    #Let's make this a pandas dataframe and then extract the unique points from that
    all_pts_df = pd.DataFrame(three_d_extrap_pts, columns=['y','x', 't', 'max_xy'])

    #Don't drop the duplicates here 
    #all_pts_df.drop_duplicates(subset=['y','x'],keep='last', inplace=True, ignore_index=True)
    all_pts_df['radius_km'] = all_pts_df['max_xy']*dx_km

    return all_pts_df

#Adds the adjustable radius to the dataframe 
def add_adjustable_radius(all_pts_df):

    #Applies the adjustable radius -- should eliminate the need to use the apply_radius
    #function later in the code 
    #Add in points within a radius -- for each element in all_pts_df 
    #Let's move this to the other section of code -- so we only grab the points we care about 

    all_column_names = ['y','x', 't', 'max_xy', 'radius_km']

    patch_coords = []
    for c in range(len(all_pts_df)):
        y = all_pts_df['y'][c]
        x = all_pts_df['x'][c]
        ymax = all_pts_df['max_xy'][c]
        xmax = ymax
        time = all_pts_df['t'][c]
        rradius = all_pts_df['radius_km'][c]

        #print (x, y, ymax, time, rradius)

        #Need to add all relative points within circle 

        #Obtain square patch 
        real_ys = np.arange(y-ymax, y+ymax+1)
        real_xs = np.arange(x-xmax, x+xmax+1)

        #print (real_xs) 
        #print (real_ys)
        #Get rid of indices that are less than zero or greater than nx (or ny)
        #real_ys = ys[(ys>=0) & (ys < ny)]
        #real_xs = xs[(xs>=0) & (xs < nx)]

        #TODO: Check if points are within radius. 
        patch_xs = []
        patch_ys = []
        patch_inds = []
        for xx in real_xs:
            for yy in real_ys:
                x_rad = abs(x-xx)
                y_rad = abs(y-yy)
                if ( (math.sqrt(x_rad**2 + y_rad**2)*dx_km <= rradius) and ((yy,xx) not in patch_coords)):
                #if ( (math.sqrt(x_rad**2 + y_rad**2)*dx_km <= rradius)):
                    #print ("added:" , xx,yy) 
                    patch_coords.append((yy,xx))
                    #We need y,x,t,max_xy,radius_km
                    patch_inds.append((yy,xx, time, ymax, rradius))
                    new_point = pd.DataFrame(patch_inds, columns=all_column_names)
                    #Append new point to original dataframe 
                    all_pts_df = pd.concat([all_pts_df, new_point], names=all_column_names, ignore_index=True, copy=False)

    #all_pts_df.sort_values(by='t', ignore_index=True, inplace=True)
    #all_pts_df.drop_duplicates(subset=['y','x'],keep='first', inplace=True, ignore_index=True)

    return all_pts_df


#Applies the (spatially expanded) extrapolated points (@in_extrap_points) to the
#original subset geodataframe (@in_gdf) 
def add_radius_points(in_gdf, in_extrap_points):


    #print (in_extrap_points)
    #print (in_gdf.loc[in_gdf['wofs_j'] == 3]) 
    #print (in_extrap_points['y']) 
    #quit() 

    parts = [] #Will hold dataframes to concat
    for a in range(len(in_gdf)):
        #print (in_gdf)
        part = in_extrap_points.copy()
        part['wofs_j'] = in_extrap_points['y'] + in_gdf['wofs_j'].iloc[a]
        part['wofs_i'] = in_extrap_points['x'] + in_gdf['wofs_i'].iloc[a]
        part['t'] = in_extrap_points['t']

        parts.append(part)

        #orig_ys = in_gdf['wofs_j']
        #orig_xs = in_gdf['wofs_i']
        #add_ys = in_extrap_points['x']

    #Now concatenate
    out_df = pd.concat(parts, axis=0, ignore_index=True)
    #Sort by time and drop duplicates 
    #out_df.sort_values(by='t', ignore_index=True, inplace=True) 
    #out_df.drop_duplicates(subset=['wofs_j', 'wofs_i'], keep='first', inplace=True, ignore_index=True)

    #Return only the columns that matter: 't', 'wofs_j', 'wofs_i'
    return out_df[['wofs_j', 'wofs_i', 't']]

#Gets the extrapolation (and other info from a given probSevere object 
#@obj_id is the object id number of given PS object 
#@in_geo_df: geopandas dataframe containing the intersection of the wofs and ps geodataframes. 
#So contains object attributes (e.g., storm motion components, PS probabilities, etc.), lat/lon points corresponding to probSevere
#objects, as well as corresponding wofs points for each object. 
#@df_prev is a dataframe containing attributes of the past PS objects
#@time_extrap is the extrapolation time (float) 
#@adjustable_radii_pts is the array showing how much extrapolation (in grid points) to apply
# at each extrapolation time (corresponds to adjustable_max_x in main code) 
#@Returns an updated geodataframe with ALL information (e.g., adds the probability changes) as well as the 
#extrapolation points for each object 
def get_obj_info(obj_id, in_geo_df, Df_Prev, Time_Extrap, adjustable_radii_pts):

    obj_subset_gdf = in_geo_df.loc[in_geo_df['id']==obj_id]
    #print (obj_subset_gdf)
    subset_hail_prob = obj_subset_gdf['hail_prob'].iloc[0]
    subset_torn_prob = obj_subset_gdf['torn_prob'].iloc[0]
    subset_wind_prob = obj_subset_gdf['wind_prob'].iloc[0]
    subset_motion_east = obj_subset_gdf['east_motion'].iloc[0]
    subset_motion_south = obj_subset_gdf['south_motion'].iloc[0]

    #Step 1: Compute storm age and probability changes 
    ps_age = find_age(obj_id, Df_Prev)

    #Step 2: Add 14- and 30-minute changes 
    if (ps_age >= 14):
        fourteen_min_change_hail, fourteen_min_change_torn, fourteen_min_change_wind = find_change(obj_id, Df_Prev, 14)
    else: #Assign current probability as 14-min change
        fourteen_min_change_hail = subset_hail_prob
        fourteen_min_change_torn = subset_torn_prob
        fourteen_min_change_wind = subset_wind_prob

    if (ps_age >= 30):
        thirty_min_change_hail, thirty_min_change_torn, thirty_min_change_wind = find_change(obj_id, Df_Prev, 30)
    else: #Assign current probability as 30-min change
        thirty_min_change_hail = subset_hail_prob
        thirty_min_change_torn = subset_torn_prob
        thirty_min_change_wind = subset_wind_prob

    #Step 3a: Get extrapolation points

    #If no extrapolation, just use raw objects 
    if (Time_Extrap == 0.0):
        zero_data = np.zeros((1,5))
        zero_data[0,3] = adjustable_radii_pts[0]
        zero_data[0,4] = adjustable_radii_pts[0]*dx_km
        orig_extrap_points = pd.DataFrame(zero_data, columns=['y','x','t','max_xy', 'radius_km'])
    else: #Do the extrapolation 
        orig_extrap_points = find_extrap_points(Time_Extrap, subset_motion_east, subset_motion_south, dx_km, adjustable_radii_pts)

    #print (orig_extrap_points) 
    #Step 3b: Get extrapolation points with adjustable radius applied 

    #Now, add adjustable radii 
    extrap_points = add_adjustable_radius(orig_extrap_points)


    updated_subset_df = add_radius_points(obj_subset_gdf, extrap_points)


    #Step 4: Add the prob, age, 14-change, and 30-change
    updated_subset_df['hail_prob'] = subset_hail_prob
    updated_subset_df['torn_prob'] = subset_torn_prob
    updated_subset_df['wind_prob'] = subset_wind_prob
    updated_subset_df['age'] = ps_age
    updated_subset_df['fourteen_change_hail'] = fourteen_min_change_hail
    updated_subset_df['fourteen_change_torn'] = fourteen_min_change_torn
    updated_subset_df['fourteen_change_wind'] = fourteen_min_change_wind
    updated_subset_df['thirty_change_hail'] = thirty_min_change_hail
    updated_subset_df['thirty_change_torn'] = thirty_min_change_torn
    updated_subset_df['thirty_change_wind'] = thirty_min_change_wind

    #subsets.append(updated_subset_df) 

    return updated_subset_df


#Gets the points, variables, and extrapolation points for each object in the @in_geo_df
#@in_geo_df: geopandas dataframe containing the intersection of the wofs and ps geodataframes. 
#So contains object attributes (e.g., storm motion components, PS probabilities, etc.), lat/lon points corresponding to probSevere
#objects, as well as corresponding wofs points for each object. 
#@df_prev is a dataframe containing attributes of the past PS objects
#@timeExtrap is the extrapolation time (float) 
#@adjustable_radii_pts is the array showing how much extrapolation (in grid points) to apply
#@MaxCores is maximum number of cores for multiprocessing 
# at each extrapolation time (corresponds to adjustable_max_x in main code) 
#@Returns an updated geodataframe with ALL information (e.g., adds the probability changes) as well as the 
#extrapolation points for each object 
def get_extrap_df(in_geo_df, df_prev, timeExtrap, adjustable_radii_pts, MaxCores):

    #Now, we'll have to loop over the unique objects that fall in the wofs domain 
    obj_ids = in_geo_df['id'].unique()

    n_objects = len(obj_ids)

    #Do multiprocessing 
    args_itr = to_iterator(obj_ids, [in_geo_df], [df_prev], [timeExtrap], [adjustable_radii_pts])

    if (n_objects <= MaxCores):
        n_threads = n_objects
    else:
        n_threads = MaxCores

    #Feed these args into the function 
    subsets = run_parallel(get_obj_info, args_itr, n_jobs=n_threads)

    #Concatenate all subsets 
    final_df = pd.concat(subsets, axis=0, ignore_index=True)

    return final_df


#Returns 2-d fields (on the wofs grid) of prob, lead time, age, 14-, and 30-minute changes in prob
#@in_df is a dataframe containing wofs j pts, ipts, lead times, ages, probabilities, and 14- and 30-minute changes
#@Ny is number of wofs y points
#@Nx is number of wofs x points
#@sstart_leads is the list of start lead times (extrapolation times in min.) 
#@eend_leads is the list of end lead times (extrapolation times in min.) 
def get_2d_wofs_fields(in_df, Ny, Nx):

    #nnt is the number of lead times 
    #nnt = len(sstart_leads) 

    out_probs = np.zeros((Ny,Nx))
    out_ages = np.ones((Ny, Nx))*-1
    out_leads = np.ones((Ny, Nx))*-1
    out_fourteens = np.zeros((Ny, Nx)) #14-min changes
    out_thirtys = np.zeros((Ny, Nx)) #30-min changes 


    #How do we want to assign a value? Want all values coming from the same storm
    #1. Highest Prob
    #2. Greatest 14-min (positive) change
    #3. Greatest 30-min (positive) change
    #4. Oldest storm
    #5. Smallest extrapolation  


    #Get list of x/y points to traverse
    #extrap_points.drop_duplicates(subset=['y','x'],keep='first', inplace=True, ignore_index=True) 
    unique_points = in_df.drop_duplicates(subset=['wofs_j', 'wofs_i'], inplace=False, ignore_index=True)
    #print (len(unique_points))

    #Also, we need to filter out the points that are outside of the wofs grid. Only save points between 0 and 299 
    unique_points = unique_points.loc[(unique_points['wofs_j'] >= 0) & (unique_points['wofs_j'] < Ny) & \
                                      (unique_points['wofs_i'] >= 0) & (unique_points['wofs_i'] < Nx)]


    #print (unique_points.loc[unique_points['wofs_j']==3])
    #print (unique_points.loc[unique_points['wofs_j']==101])
    #quit() 

    #print (unique_points) 
    for l in range(len(unique_points)):
        y = unique_points['wofs_j'].iloc[l]
        x = unique_points['wofs_i'].iloc[l]
        #print (y,x) 

        #print ("%s/%s" %(str(l), str(len(unique_points))))
        df_subset = in_df.loc[(in_df['wofs_j'] == y) & (in_df['wofs_i'] == x)]

        #Sort dataframe by greatest prob, greatest 14-min change, greatest 30-min change, 
        #oldest storm, and smallest extrapolation. 
        df_subset_sorted = df_subset.sort_values(['prob', 'fourteen_change', 'thirty_change', 'age', 't'], \
                                        ascending=[False, False, False, False, True])

        #print (df_subset_sorted)
        maxValue = df_subset_sorted.iloc[0,:]
        
        #print (maxValue['prob']) 
        #print ("done" ) 
        #quit() 
        #print (maxValue)
        #print (maxValue['prob'], maxValue['age'], maxValue['t'], maxValue['fourteen_change'], maxValue['thirty_change']) 
        #print ("****") 

        #Now assign the appropriate values to the grid 
        out_probs[y,x] = maxValue['prob']
        out_ages[y,x] = maxValue['age']
        out_leads[y,x] = maxValue['t']
        out_fourteens[y,x] = maxValue['fourteen_change']
        out_thirtys[y,x] = maxValue['thirty_change']


    #print (out_probs[3,101]) 
    #quit() 

    #np.set_printoptions(threshold=sys.maxsize)
    #print (out_probs)
    #quit() 

    #print (out_probs.reshape(int(Ny*Nx), -1)[1001]) 
    #quit() 

    #print (out_probs)
    #print (out_probs.shape) 
    #quit() 

    return out_probs, out_leads, out_ages, out_fourteens, out_thirtys


#Parses the object extrapolations by lead time and saves them to file 
#lead_time_index, [start_leads], [end_leads], [start_valids_multi[dd]], \
#                            [end_valids_multi[dd]], [final_df], [ny], [nx], [outdir], [hazard], [date], \
#                            [init_time], [start_t], [radius]

#@ind is the index corresponding to the lead time (e.g., 0=0-30, 1=30-60, 2=60-90, etc. 
#@start_leads_arr is the array of start lead times 
#@end_leads_arr is the array of end lead times 
#@start_valids_arr is the array of start valid times 
#@end_valids_arr is the array of end valid times 
#@in_df is the incoming dataframe with all of the grid points and attributes (e.g., lead times, probabilities, prob changes, etc.) 
#@nny is the number of y grid points in the wofs domain
#@nnx is the number of x grid points in the wofs domain
#@out_dir is the directory in which to save the ouput files 
#@haz is the hazard (string; e.g., "hail") 
#@date is the current date (string) 
#@time_init is the initialization time 
#@rradius is the maximum radius (will be included in output file name) 
#@thresh_ps (float) is the probability threshold to apply. i.e., only consider objects at or above this threshold. 

def save_lead_times(ind, haz, start_leads_arr, end_leads_arr, in_df, nnny, nnnx, thresh_ps):

    #Dataframe columns relevant to each hazard 
    hail_cols = ["wofs_j", "wofs_i", "t", "hail_prob", "age", "fourteen_change_hail", "thirty_change_hail" ]
    torn_cols = ["wofs_j", "wofs_i", "t", "torn_prob", "age", "fourteen_change_torn", "thirty_change_torn"]
    wind_cols = ["wofs_j", "wofs_i", "t", "wind_prob", "age", "fourteen_change_wind", "thirty_change_wind"]


    if (haz == "hail"):
        haz_subset = in_df[hail_cols]
        #Rename the columns 
        haz_subset.rename(columns = {'hail_prob':'prob', 'fourteen_change_hail':'fourteen_change',\
                                     'thirty_change_hail':'thirty_change'}, inplace=True)

    elif (haz == "torn"):
        haz_subset = in_df[torn_cols] 
        #Rename the columns 
        haz_subset.rename(columns = {'torn_prob':'prob', 'fourteen_change_torn':'fourteen_change',\
                                     'thirty_change_torn':'thirty_change'}, inplace=True)
        
    elif (haz == "wind"):
        haz_subset = in_df[wind_cols]
        #Rename the columns 
        haz_subset.rename(columns = {'wind_prob':'prob', 'fourteen_change_wind':'fourteen_change',\
                                     'thirty_change_wind':'thirty_change'}, inplace=True)

    #Now apply the probability threshold
    haz_subset = haz_subset.loc[haz_subset['prob']>=thresh_ps]

    #If there are no objects, simply return empty fields 
    if (len(haz_subset) == 0):
        #6 hard-coded here because we have 6 different fields 
        #Oh, but wait. How to handle the ages? 
        #Order: raw_probs, smoothed_probs, leads, ages, changes14, changes30
        #Ages and leads should be -1 if no object 
        neg_ones = -1*np.ones((nnny, nnnx))
        ps_outarr = np.zeros((nnny, nnnx, 6))
        ps_outarr[:,:,2] = neg_ones
        ps_outarr[:,:,3] = neg_ones 
        return ps_outarr
        

    start_lead = start_leads_arr[ind]
    end_lead = end_leads_arr[ind]

    time_subset = haz_subset.loc[(haz_subset['t'] >= start_lead) &\
                                    (haz_subset['t'] <= end_lead)]

    #Now, here is where we'd want to drop the duplicates -- but only drop the duplicates where everything is the same
    time_subset.drop_duplicates(keep='first', inplace=True, ignore_index=True)

    #Get the time subset 

    #Now, we'll give this to a function to return 2-d wofs grids of what we're interested in.
    raw_probs, leads, ages_2d, changes14, changes30 = get_2d_wofs_fields(time_subset, nnny, nnnx)

    #Get spatially smoothed probs 
    smoothed_probs = gaussian_filter(raw_probs, sigma=3, order=0, mode='constant', truncate=3.5)

    #Convert to float32 for saving 
    raw_probs = np.float32(raw_probs)
    smoothed_probs = np.float32(smoothed_probs)
    leads = np.float32(leads)
    ages_2d = np.float32(ages_2d)
    changes14 = np.float32(changes14)
    changes30 = np.float32(changes30)

    #print (raw_probs.reshape(int(nnny*nnnx), -1)[1001])
    #quit() 

    #smoothed_probs.tofile("%s/smoothed_expanded_probsevere_%s_%s_%s_v%s-%s_r%skm.dat" %(out_dir, haz, date, time_init, start_t, end_t, str(rradius)))
    #raw_probs.tofile("%s/raw_expanded_probsevere_%s_%s_%s_v%s-%s_r%skm.dat" %(out_dir, haz, date, time_init, start_t, end_t, str(rradius)))
    #ages.tofile("%s/raw_ps_age_%s_%s_%s_v%s-%s_r%skm.dat" %(out_dir, haz, date, time_init, start_t, end_t, str(rradius)))
    #leads.tofile("%s/raw_ps_leads_%s_%s_%s_v%s-%s_r%skm.dat" %(out_dir, haz, date, time_init, start_t, end_t, str(rradius)))
    #changes14.tofile("%s/raw_ps_14min_change_%s_%s_%s_v%s-%s_r%skm.dat" %(out_dir, haz, date, time_init, start_t, end_t, str(rradius)))
    #changes30.tofile("%s/raw_ps_30min_change_%s_%s_%s_v%s-%s_r%skm.dat" %(out_dir, haz, date, time_init, start_t, end_t, str(rradius)))

    #Let's actually package this information in a single 3-d array
    #6 hard-coded here because we have 6 different fields 
    ps_outarr = np.zeros((nnny, nnnx, 6))
    #Order: raw_probs, smoothed_probs, leads, ages, changes14, changes30
    ps_outarr[:,:,0] = raw_probs
    ps_outarr[:,:,1] = smoothed_probs
    ps_outarr[:,:,2] = leads
    ps_outarr[:,:,3] = ages_2d
    ps_outarr[:,:,4] = changes14
    ps_outarr[:,:,5] = changes30


    #return raw_probs, smoothed_probs, ages, leads, changes14, changes30
    return ps_outarr

#Adds a buffer to the points in @in_gdf in meters. 
#@in_gdf: Incoming geopandas dataframe with list of lat/lon points
#@buffer_dist: Buffer distance to be applied, in meters
def add_gpd_buffer(in_gdf, buffer_dist):

    #Make a copy of the incoming gdf
    copy_gdf = in_gdf.copy(deep=True)

    #Convert to meters 
    copy_gdf.to_crs("EPSG:32634", inplace=True) 

    #Apply buffer 
    copy_gdf.geometry = copy_gdf.geometry.buffer(buffer_dist)
 
    #Convert back to lat/lon coords
    copy_gdf.to_crs("EPSG:4326", inplace=True) 

    return copy_gdf


#Returns the list of 3-d grids from probSevere for multiple hazards and lead times.  
def do_probSevere_preprocessing(initTime, startValids, endValids, psStart, currDate, nextDate, \
        nextDayInits, psDir, hazs, extrapTime, psThresh, minRadius, maxRadius, start_leads, end_leads,\
        wofsLats, wofsLons, nny, nnx, maxCores, nWindows):

    #Hard coded variables/method constants (for now):
    time_to_go_back = 180 #Previous ProbSevere times/files to look back over (in minutes) 
    back_time_increments = 15 #Look back in 15 min. increments
    column_names = ['id', 'hail_prob', 'torn_prob', 'wind_prob', 'age'] 
    ps_variable_order = ['raw_probs', 'smooth_probs', 'leads', 'ages', 'changes14', 'changes30']
    wofs_buffer = 2.15*10**3 #Buffer to add around wofs points in m. 2.15km guarantees that we cover the full grid cell 

    #First, get use date 
    if (initTime in nextDayInits):
        use_date = nextDate
    else: 
        use_date = currDate 


    #Get a list of relevant PS files 
    ps_previous_list, ps_past_date_list = get_past_ps_list(int(psStart), time_to_go_back, \
            use_date, currDate, back_time_increments)

    #Also add 2- and 10-minute past ps files 
    two_min_list, two_min_date_list = get_past_ps_list(int(psStart), 2, use_date, currDate, 2)
    ten_min_list, ten_min_date_list = get_past_ps_list(int(psStart), 10, use_date, currDate, 10)
    ps_previous_list.append(two_min_list[1])
    ps_previous_list.append(ten_min_list[1])
    ps_past_date_list.append(two_min_date_list[1])
    ps_past_date_list.append(ten_min_date_list[1])

    #Set the current and backup PS files to use 
    ps_file = "%s/MRMS_EXP_PROBSEVERE_%s.%s00.json" %(psDir, use_date, psStart)
    
    #If ps_file is not found, let's see if we can use the file from 2 minutes before 
    backup_ps_file = "%s/MRMS_EXP_PROBSEVERE_%s.%s00.json" %(psDir, use_date, str(two_min_list[1]))

    ages = [0, 14, 30, 44, 60, 74, 90, 104, 120, 134, 150, 164, 180, 2, 10]    

    #Get dataframe of past PS objects -- to get age and prob changes with time. 
    #NOTE: prev_df now contains probabilities for each hazard 
    prev_df = get_prev_ps_df(column_names, ps_previous_list, ps_past_date_list, psDir, ages)

    #Now, read in the probSevere json files. Use backup file if the current file is not available 
   
    #Get the ProbSevere geodataframe  
    ps_gdf, extrapTime, start_leads, end_leads, adjustable_maximum_x = get_ps_geoDataFrame(ps_file, backup_ps_file, \
            extrapTime, minRadius, maxRadius, start_leads, end_leads)

    #Get wofs geodataframe 
    wofs_gdf = get_wofs_gdf(wofsLats, wofsLons, nny, nnx)     

    #TODO: We need to get a buffer of wofs points. I think that's why we're excluding some points now. 
    buffered_wofs_gdf = add_gpd_buffer(wofs_gdf, wofs_buffer)

    #print (buffered_wofs_gdf) 
    #quit() 

    #Merge the wofs and probSevere geodataframes 
    new_gdf = buffered_wofs_gdf.sjoin(ps_gdf, how='inner', predicate='intersects')

    #Implement function to get the points, variables, and extrapolation points for each object in
    #the given geodataframe (i.e., new_gdf) 
    if (len(new_gdf) > 0):
        final_df = get_extrap_df(new_gdf, prev_df, extrapTime, adjustable_maximum_x, maxCores)
    else: 
        final_df = new_gdf

    #print (final_df.iloc[0:20,:])
    #print (list(final_df.columns))
    #quit() 

    #Now make the forecasts for the different lead times. -- And different hazards 
    lead_time_index = np.arange(len(start_leads))

    lead_times_itr = to_iterator(lead_time_index, hazs, [start_leads], [end_leads], [final_df], [ny], [nx], [psThresh])

    #Order: Time Window 1: Hail, Wind, Torn ; Time Window2: Hail, Wind, Torn, etc. 


    if (maxCores <= (len(start_leads)*len(hazs))):
        njobs = maxCores
    else:
        njobs = int(len(start_leads)*len(hazs))

    #TODO: Not sure how to handle this. 
    #results should have 6 3-d arrays (1 for each lead time)
    #In each array, shape should be (ny, nx, nv), where here nv is 6 for 6 fields 
    #Order of the 6 fields is: raw probs, smoothed probs, leads, ages, 14-min change, 30-min change 
    #NOTE: I wonder if we shouldn't try to increase the number of jobs here? (e.g., 2*lead times??) 
    results_list = run_parallel(save_lead_times, lead_times_itr, n_jobs=njobs)

    #print (results_list[0]) 

    #print (results_list[0].shape) 

    #test = results_list[0][:,:,0].reshape(int(nny*nnx), -1)
    #print (test[1001]) 
    #quit() 

    #print (len(results_list) ) 
    #print (results_list) 
    #print (len(results_list[0]))
    
    #print (results_list[0][1])
    #print (results_list[0][2])

    #for a in range(len(results_list)):
    #    print (results_list[a][1])
    #    print (results_list[a][2])
    #    print ("-----") 

    #print (len(results_list)) 
    #print (results_list[0].shape) 


    #NOTE: results_list has length 18 (3 hazards x 6 lead times) --> hail1, wind1, torn1, hail2, wind2, torn2, etc. 
    #Each element of results_list has shape (dy,dx, 6), where the 6 is each PS element 

    #Convert to xarray, where each variable has a name and the dimensions are (nT, nY, nX), 
    #where nT is the number of lead times, nY is the number of y points, and nX is the number of x points 
    #What do we want?  -- each var has a name. Then [NT, NY, NX] 
    ps_xr = ps_to_xr(results_list, ps_variable_order, nWindows, nny, nnx, hazs)

    print (ps_xr) 
    quit() 


    #Return the ProbSevere xarray dataset with all the ProbSevere predictors (for each lead time and spatial point)
    return ps_xr

#Converts the list of probSevere fields to an xarray
#@ps_list is the incoming list of ps fields 
#@ps_var_order is a list of PS fields, corresponding to the last dimension of ps_list
#@nT is the number of time windows (i.e., n_windows) 
#@nY is the number of y points
#@nX is the number of x points 
def ps_to_xr(ps_list, ps_var_order, nT, nY, nX, Hazs):
    #Will hold the different hazards at the different lead times 
    hail_arrs = ps_list[0:18:3]
    wind_arrs = ps_list[1:18:3]
    torn_arrs = ps_list[2:18:3] 
    
    varnames = (["%s_%s" %(h,v) for h in Hazs for v in ps_var_order]) 

    nV = len(ps_var_order) #number of variables 
    nH = len(Hazs) #number of hazards 

    new_arr = np.zeros((nT, nY, nX, nV, nH)) #Intermediate array 

    #Have to do the remapping here. -- Put everything in one big array
    for t in range(nT):
        for v in range(len(ps_var_order)):
            new_arr[t,:,:,v,0] = hail_arrs[t][:,:,v]
            new_arr[t,:,:,v,1] = wind_arrs[t][:,:,v]
            new_arr[t,:,:,v,2] = torn_arrs[t][:,:,v]
            
    #Now, need to create x_array with dimensions (nT, nY, nX) 
    new_xr = xr.Dataset(data_vars=None, coords={"lead_time": (range(nT)), "y": (range(nY)), "x": (range(nX))})
    count = 0 
    for h in range(nH):
        for v in range(nV):
            varname = varnames[count]
            new_xr[varname] = (["lead_time", "y", "x"], new_arr[:,:,:,v,h])
            count += 1
    
    return new_xr


#============
#WoFS preprocessing methods 

#Finds and returns the initial start ID given a string of the initial time (init_string)
#and first valid time (valid_string) 
def find_start_id(init_string, valid_string, wofsInc):

    new_init = int(init_string)
    new_valid = int(valid_string)
    #First, convert to "sum" time if necessary
    if (new_init < 1200):
        new_init = new_init + 2400
    if (new_valid < 1200):
        new_valid = new_valid + 2400

    #Next, obtain hours and minutes for init and valid
    hhi = str(new_init)[0:2] #init hours
    mmi = str(new_init)[2:4] #init mins

    hhv = str(new_valid)[0:2] #valid hours
    mmv = str(new_valid)[2:4] #valid mins

    #Take subtraction 
    hh_diff = int(hhv) - int(hhi)
    mm_diff = int(mmv) - int(mmi)

    #convert to minutes 
    min_diff = to_minutes(hh_diff, mm_diff)

    return int(min_diff/wofsInc)

#Takes a time in hours and minutes and converts to minutes 
def to_minutes(hours_in, minutes_in):

    return hours_in*60 + minutes_in


def get_standard_wofs(iTime, TimeWindow, startValidInd, StartValids, EndValids, WofsIncrement, VarInd, WofsVars, AggMethods,\
        useDate, WofsDir, ens_vars, env_vars, svr_vars, dbz_thresh, Ny, Nx, Dm, indivMemVars):

    #print (startValidInd)
    #print (VarInd) 

    StartValid = StartValids[startValidInd]
    EndValid = EndValids[startValidInd]

    WofsVar = WofsVars[VarInd] 
    AggMethod = AggMethods[VarInd]

    #First, determine ftype 
    if (WofsVar in ens_vars):
        ftype = "ENS"
    elif (WofsVar in env_vars):
        ftype = "ENV"
    elif (WofsVar in svr_vars):
        ftype = "SVR"

    #Get some information related to the wofs file names 
    n_wofs_files = int(TimeWindow/WofsIncrement)+1.0
    curr_valids = get_curr_valids(float(StartValid), int(n_wofs_files), int(WofsIncrement))

    start_id = find_start_id(iTime, str(StartValid), WofsIncrement)

    curr_ids = np.arange(start_id, start_id+int(n_wofs_files))
    curr_ids = [str(c).zfill(2) for c in curr_ids]

    wofs_filenames = ["%s/wofs_%s_%s_%s_%s_%s.nc" %(WofsDir, ftype, curr_ids[d], useDate, iTime, curr_valids[d]) for d in range(int(n_wofs_files))] 


    #Start getting the temporal aggregation
    if (AggMethod == "max"):
        if (WofsVar == "comp_dz"):
            time_max = np.ones((Ny,Nx))*-9999999.0
        else:
            time_max = np.ones((Dm, Ny, Nx))*-9999999.0
    elif (AggMethod == "min"):
        if (WofsVar == "comp_dz"):
            time_max = np.ones((Ny,Nx))*9999999.0
        else:
            time_max = np.ones((Dm, Ny, Nx))*9999999.0

    for f in range(len(wofs_filenames)): #loop over time 
        filename = wofs_filenames[f]
        try:
            ds = nc.Dataset(filename)
        except FileNotFoundError:
            print ("%s not found. Moving on." %filename)
            continue

        if (WofsVar in indivMemVars):
            curr_var = ds["uh_2to5"][:] #Because there's not code that explicitly captures indiv members
        else: 
            curr_var = ds[WofsVar][:]

        #For composite reflectivity, we're interested in the fraction
        #of members exceeding the threshold 
        if (WofsVar == "comp_dz"):
            #First apply threshold and binarize
            #Divide by number of ensemble members (do in same step) 
            curr_var = np.where(curr_var >= dbz_thresh, 1/Dm, 0)
            #sum over ensemble members
            curr_var = np.sum(curr_var, axis=0)

        if (AggMethod == "max"):
            time_max = np.maximum(time_max, curr_var)
        elif (AggMethod == "min"):
            time_max = np.minimum(time_max, curr_var)


    #For Individual Member
    if (WofsVar in indivMemVars):
        if (WofsVar == "m1_uh_2to5"):
            mem_index = 0
        elif (WofsVar == "m2_uh_2to5"):            
            mem_index = 1
        elif (WofsVar == "m3_uh_2to5"):         
            mem_index = 2
        elif (WofsVar == "m4_uh_2to5"):
            mem_index = 3
        elif (WofsVar == "m5_uh_2to5"):
            mem_index = 4 
        elif (WofsVar == "m6_uh_2to5"):
            mem_index = 5 
        elif (WofsVar == "m7_uh_2to5"):
            mem_index = 6 
        elif (WofsVar == "m8_uh_2to5"):
            mem_index = 7 
        elif (WofsVar == "m9_uh_2to5"):
            mem_index = 8 
        elif (WofsVar == "m10_uh_2to5"):
            mem_index = 9 
        elif (WofsVar == "m11_uh_2to5"):
            mem_index = 10 
        elif (WofsVar == "m12_uh_2to5"):
            mem_index = 11 
        elif (WofsVar == "m13_uh_2to5"):
            mem_index = 12 
        elif (WofsVar == "m14_uh_2to5"):
            mem_index = 13 
        elif (WofsVar == "m15_uh_2to5"):
            mem_index = 14 
        elif (WofsVar == "m16_uh_2to5"):
            mem_index = 15 
        elif (WofsVar == "m17_uh_2to5"):
            mem_index = 16 
        elif (WofsVar == "m18_uh_2to5"):
            mem_index = 17  
        
        member = time_max[mem_index, :, :]
        member = np.ma.getdata(member)
        member = np.float32(member)
        return member


    #Old: For Ensemble Mean
    #Next, take ensemble mean -- for everything except composite reflectivity
    if (WofsVar == "comp_dz"):
        new_ens_mean = time_max
    else:
        new_ens_mean = np.mean(time_max, axis=0)

    #Get rid of the masking element of the array    
    new_ens_mean = np.ma.getdata(new_ens_mean)

    #TODO: Save data to .dat (or .csv) file 
    #Need to convert to float32 first
    new_ens_mean = np.float32(new_ens_mean)

    return new_ens_mean



#Necessary to get the inICs variable 
def make_inICs_plot(spatial_mask_wofs, rng_wofs, in_llats, in_llons):

    fig = plt.figure(figsize=(12,12), facecolor='whitesmoke')

    ax1 = fig.add_subplot(1,1,1)

    #fig = plt.figure(figsize = (12,12), facecolor = 'whitesmoke')
    Map = Basemap(projection='lcc', llcrnrlon=sw_lon,\
                llcrnrlat=sw_lat,urcrnrlon=ne_lon,urcrnrlat=ne_lat, \
                lat_1 = tlat1, lat_2 = tlat2, lon_0 = stlon, \
                resolution='l')

    #Map.drawcoastlines(linewidth=1.0,color='black')
    Map.drawstates(linewidth=1.0,color='black')
    Map.drawcountries(linewidth=1.5,color='black')

    #Plot each probSevere object
    #for o in range(len(psLats)):
    #    if (psProbs[o] > 0): #only plot non-zero torn prob PS objects
    #        plot_probSevere(psLats[o], psLons[o], Map) 

    rng_obs = np.arange(0., 2.)
    #rng_obs = rng_obs - 0.0000001
    #rng_obs = np.array([0., 0.5])

    my_cmap_wofs = 'Greys'

    filled_plt = Map.contourf(in_llons, in_llats, spatial_mask_wofs,rng_wofs, cmap=my_cmap_wofs, extend='max', latlon=True, alpha=1.0, zorder=1)

#Create a lookup table for the contour levels 
    lvl_lookup = dict(zip(filled_plt.collections, filled_plt.levels))

    # loop over collections (and polygons in each collection), store polygons in PolyList
    #(grabbed this code from internet) 
    PolyList=[]
    for col in filled_plt.collections:
        z=lvl_lookup[col] # the value of this level
        for contour_path in col.get_paths():
            # create the polygon for this level
            for ncp,cp in enumerate(contour_path.to_polygons()):
                lons = cp[:,0]
                lats = cp[:,1]
                #mapped_lons = [Map(lons[x], lats[x])[0] for x in range(len(lons))]
                #mapped_lats = [Map(lons[x], lats[x])[1] for x in range(len(lats))]


                new_shape = geometry.Polygon([(i[0], i[1]) for i in zip(lons,lats)]).buffer(0)
                #new_shape = geometry.Polygon([Map(i[0], i[1]) for i in zip(lons,lats)]).buffer(0)
                if ncp == 0:
                    poly = new_shape # first shape
                else:
                    poly = poly.difference(new_shape) # Remove the holes

                #print (poly) 

                PolyList.append({'poly':poly,'props':{'z': z}})

    return PolyList, Map 



#Maps a list of polygons to wofs grid 
#@poly_list is a list of polygons -- with coordinates in Map1 coordinates
#@in_lats is a list of lats -- not yet in Map1 coordinates
#@in_lons is a list of lons -- not yet in Map1 coordinates 
def shape_to_wofs(poly_list, in_lats, in_lons, nnx, nny, Map1):

    #First, create output wofs grid
    outgrid = np.zeros((nny, nnx))

    #How many polygons are there?
    n_poly = len(poly_list)

    #prep_poly_list = [prep(p) for p in poly_list]

    #First, convert list of polygons to a MultiPolygon
    multi = MultiPolygon(poly_list)

    multi = prep(multi)

    if (n_poly > 0):
        #Span the Wofs points, put in Map coordinates and see if any polygon contains the points
        for i in range(nnx):
            for j in range(nny):
                curr_lat = in_lats[j,i]
                curr_lon = in_lons[j,i]

                #print (map_lon, map_lat) 

                #point = Point(map_lon, map_lat) 
                #point = Point(curr_lon, curr_lat) 
                #point = Point(Map1(curr_lat, curr_lon))
                point = Point(Map1(curr_lon, curr_lat))
                #point = Point(curr_lon, curr_lat) 

                #Is this point contained in the MultiPolyon ?
                #if ( multi.contains(point)):
                #    outgrid[j,i] = 1.0 
                #if (point.within(multi)):
                if (multi.contains(point)):
                    outgrid[j,i] = 1.0



    return outgrid



def get_inICs(iTime, TimeWindow, FullTimeWindow, validInd, StartValids, EndValids, WofsIncrement,\
        useDate, WofsDir, dbz_thresh, Ny, Nx, Dm, WofsLats, WofsLons):

    #Set the start and end valid times based on validInd
    StartValid = StartValids[validInd]
    EndValid = EndValids[validInd] 


    #Constants for the method -- We'll always be dealing with the same one variable, so we can put this in
    WofsVar = "comp_dz" 
    AggMethod = "max"    
    ftype = "ENS" 


    #Get some information related to the wofs file names 
    n_wofs_files = int(TimeWindow/WofsIncrement)+1.0
    n_wofs_files_full = int(FullTimeWindow/WofsIncrement)+1.0 
    curr_valids = get_curr_valids(float(StartValid), int(n_wofs_files), int(WofsIncrement))
    curr_valids_full = get_curr_valids(float(iTime), int(n_wofs_files_full), int(WofsIncrement))

    start_id = find_start_id(iTime, str(StartValid), WofsIncrement)
    start_id_full = find_start_id(iTime, str(iTime), WofsIncrement)

    curr_ids = np.arange(start_id, start_id+int(n_wofs_files))
    curr_ids = [str(c).zfill(2) for c in curr_ids]

    curr_ids_full = np.arange(start_id_full, start_id_full+int(n_wofs_files_full))
    curr_ids_full = [str(c).zfill(2) for c in curr_ids_full]

    wofs_filenames = ["%s/wofs_%s_%s_%s_%s_%s.nc" %(WofsDir, ftype, curr_ids[d], useDate, iTime, curr_valids[d]) for d in range(int(n_wofs_files))]
    wofs_filenames_full = ["%s/wofs_%s_%s_%s_%s_%s.nc" %(WofsDir, ftype, curr_ids_full[d], useDate, iTime, curr_valids_full[d]) for d in range(int(n_wofs_files_full))]

    #First, check if all wofs files exist. If not, quit the program 
    for w in wofs_filenames_full:
        if (not os.path.exists(w)):
            #TODO: In the future, might not want to completely quit everything. Might use all zeros or something else. 
            print ("missing %s. Quitting (inICs) for this date." %w)
            pool.terminate() #Will probably crash, but that's ok because we need the processes to end somehow
            quit()
    
    #Need to get p40_fcst (p40 integrated over forecast period) and p40_full (p40 integrated over full period) 
    time_max_full = np.ones((ny,nx))*-9999999.0
    time_max_fcst = np.ones((ny,nx))*-9999999.0
    

    for f in range(len(wofs_filenames_full)): #loop over time 
        filename = wofs_filenames_full[f]
        try:
            ds = nc.Dataset(filename)
        except FileNotFoundError:
            print ("%s not found. Moving on." %filename)
            continue

        curr_var = ds[WofsVar][:]


        #Compute WoFS probability of >40dbz at initialization time 
        curr_var = np.where(curr_var >= dbz_thresh, 1/dm, 0)
        #sum over ensemble members
        curr_var = np.sum(curr_var, axis=0)

        if (f == 0 ): #save p40dbz at initialization time 
            #Compute WoFS probability of >40dbz at initialization time 
            p40_init = curr_var

        #Take temporal max over full period 
        time_max_full = np.maximum(time_max_full, curr_var)

        #Take temporal max over forecast period.  
        #Check if this is part of the specific forecast period 
        if (filename in wofs_filenames):
            time_max_fcst = np.maximum(time_max_fcst, curr_var)

        #N0w, let's get rid of the masking element of the arrays we care about 
    p40_init = np.ma.getdata(p40_init)
    p40_fcst = np.ma.getdata(time_max_fcst)
    p40_full = np.ma.getdata(time_max_full)

    #Try binary fields 
    #p40_init_bin = np.where(p40_init >= math.ceil((dm-1)/dm), 1, 0) 
    p40_init_bin = np.where(p40_init >= 1, 1, 0)
    p40_fcst_bin = np.where(p40_fcst >= 0.49, 1, 0)
    p40_full_bin = np.where(p40_full >= 0.49, 1, 0)


    toPlot = [p40_init_bin, p40_fcst_bin, p40_full_bin]

    rng_wofs_full = np.linspace(0,1,11)
    rng_wofs = np.array([0, 0.95])
    rngs_wofs = [rng_wofs, rng_wofs, rng_wofs]


    polygon_list = [] #Will be a list of lists 
    for p in range(len(toPlot)):

        pgon, map_obj = make_inICs_plot(toPlot[p], rngs_wofs[p], WofsLats, WofsLons)
        polygon_list.append(pgon)


    #what does polygon list contain? Let me see: 
    pgon_init = polygon_list[0]
    pgon_fcst = polygon_list[1]
    pgon_full = polygon_list[2]

    #Step 1: Return the full polygon when the full polygon overlaps with the initial polygon
    full_overlaps = []
    forecast_overlaps = []
    for pi in pgon_init:
        #Only care about those whose value is >= 0.95 
        if (pi['props']['z'] >= 0.95):
            #Span the full polygon objects (> 0.95 and test) 
            for pfl in pgon_full:
                if (pfl['props']['z'] >= 0.95):
                    if (pfl['poly'].contains(pi['poly'])):
                        full_overlaps.append(pfl['poly'])
                        #print ("True") 

    #Now, we just need to get the overlaps with the forecast period polygons
    for pfcst in pgon_fcst:
        if (pfcst['props']['z'] >= 0.95):
            for pfull in full_overlaps:
                if (pfull.contains(pfcst['poly'])):
                    forecast_overlaps.append(pfcst['poly'])

    #Now we need to place this polygon on the wofs grid 

    final_field = shape_to_wofs(forecast_overlaps, WofsLats, WofsLons, nx, ny, map_obj)

    #Finally, convert to float32 and save to .dat 
    final_field = np.float32(final_field)


    return final_field 

#Converts list of standard wofs variables (output from parallelization) to xarray
#with dimensions (nT, nY, nX) 
#@in_wofs_list is the incoming list of gridded wofs data --
# the list spans the variables and then the times
# (e.g., v1_t1, v2_t1, v3_t1, ..., v1_t2, v2_t2, ...)
#@variable_list is the list of wofs data, 
#@nT is number of time windows
#@nY is number of y points
#@nX is number of x points
def standard_wofs_to_xr(in_wofs_list, variable_list, nT, nY, nX):
    
    nV = len(variable_list) #number of variables 

    #Create intermediate array of (nT, nY, nX, nV) 
    new_arr = np.zeros((nT, nY, nX, nV))

    count = 0 
    for t in range(nT):
        for v in range(nV):
            new_arr[t,:,:,v] = in_wofs_list[count] 
            count += 1


    #Now, get xarray dataset with (nT, nY, nX)
    new_xr = xr.Dataset(data_vars=None, coords={"lead_time": (range(nT)), "y": (range(nY)), "x": (range(nX))})

    for v in range(nV):
        varname = variable_list[v] 
        new_xr[varname] = (["lead_time", "y", "x"], new_arr[:,:,:,v])
                

    #NOTE: We need to rename "comp_dz" to prob_40dbz
    new_xr = new_xr.rename_vars({'comp_dz': 'prob_40dbz'}) 

    return new_xr

#Converts list of inICs variables (output from parallelization -- list of diff lead times)
#to xarray with dimensions (nT, nY, nX) 
#@inICs_list is list of gridded wofs inICs data, with each element corresponding to a different
#lead time
#@nT is the number of time windows
#@nY is the number of y points
#@nX is the number of x points 
def inICs_to_xr(inICs_list, nT, nY, nX):
   
    #Convert inICs_list from a list to a 3-d array
    new_arr = np.zeros((nT, nY, nX)) 
    
    for t in range(nT):
        new_arr[t,:,:] = inICs_list[t]

    #Create new xarray with dimensions (nT, nY, nX) 
    new_xr = xr.Dataset(data_vars={"inICs": (("lead_time", "y", "x"),new_arr)}, coords={"lead_time": (range(nT)), "y": (range(nY)), "x": (range(nX))})
    
    return new_xr 

#Returns the list of 2-d, time-aggregated wofs fields used as predictors 
#NOTE: will do this for each time window and variable -- will parallelize up front 
#@wofsIncrement is the time between each new wofs summary file in minutes
def do_wofs_preprocessing(initTime, timeWindow, startValids, endValids, wofsIncrement, wofs_vars, agg_methods, \
        currDate, nextDate, wofsDir, nextDayInits, indivMemberVars, nny, nnx, ddm, fullTimeWindow, wofsLats, \
        wofsLons, maxCores, nWindows):

    #WoFS dbz thresh -- almost always set to 40
    dbzThresh = 40 #Will compute number of members exceeding this comp refl threshold

    #First, get use day
    if (initTime in nextDayInits):
        use_date = nextDate
    else:
        use_date = currDate


    #Will be able to delete this eventually. Need for determining which file a given variable is in
    ens_wofs_variables = ["ws_80", "dbz_1km", "wz_0to2_instant", "uh_0to2_instant",  "uh_2to5", "w_up",\
                     "w_1km", "w_down", "buoyancy", "div_10m", "10-500m_bulkshear", "ctt", "fed",\
                     "rh_avg", "okubo_weiss", "hail", "hailcast", "freezing_level", "comp_dz",\
                     "m1_uh_2to5", "m2_uh_2to5", "m3_uh_2to5", "m4_uh_2to5", "m5_uh_2to5", "m6_uh_2to5",\
                     "m7_uh_2to5", "m8_uh_2to5", "m9_uh_2to5", "m10_uh_2to5", "m11_uh_2to5", "m12_uh_2to5",\
                     "m13_uh_2to5", "m14_uh_2to5", "m15_uh_2to5", "m16_uh_2to5", "m17_uh_2to5", "m18_uh_2to5"]

    env_wofs_variables = ["mslp", "u_10", "v_10", "td_2", "t_2", "qv_2", "theta_e", "omega", "psfc", \
                            "pbl_mfc", "mid_level_lapse_rate", "low_level_lapse_rate" ]

    svr_wofs_variables = ["shear_u_0to1", "shear_v_0to1", "shear_u_0to3", "shear_v_0to3", "shear_u_0to6", "shear_v_0to6",\
                      "srh_0to500", "srh_0to1", "srh_0to3", "cape_sfc", "cin_sfc", "lcl_sfc", "lfc_sfc",\
                       "stp", "scp", "stp_srh0to500"]


    #TODO: Need to parallelize below methods. Also, somehow concatenate 

    #NOTE: Have to do below for each start valid 

    #First, get the standard variables -- and parallelize by variable and time window 
    #Will be parallelized; below is a representation of eventual parallelization

    #NOTE: Have to fix. startVlids and endValids will always be connected. So probably pass in an index 
    start_valids_inds = np.arange(len(startValids))
    vars_inds = np.arange(len(wofs_vars))

    #Get the standard wofs fields/variables 
    #Spans the variables first and then the time windows second  -- In to_iterator, the latest variables get iterated over first 
    args_itr_standard_wofs = to_iterator([initTime], [timeWindow], start_valids_inds, [startValids], [endValids], [wofsIncrement], \
                                vars_inds, [wofs_vars], [agg_methods], [use_date], [wofsDir], [ens_wofs_variables], [env_wofs_variables],\
                                [svr_wofs_variables], [dbzThresh], [nny], [nnx], [ddm], [indivMemberVars])

    #Feed these args into the function 
    standard_wofs_list = run_parallel(get_standard_wofs, args_itr_standard_wofs, n_jobs=maxCores)

    #TODO: Convert to xarray with 3 dimensions: (dT, dY, dX) 
    #in_wofs_list, variable_list, nT, nY, nX
    xr_standard_wofs = standard_wofs_to_xr(standard_wofs_list, wofs_vars, nWindows, nny, nnx) 


    #Get the inICs field 
    args_itr_inICs = to_iterator([initTime], [timeWindow], [fullTimeWindow], start_valids_inds, [startValids], [endValids], \
                                [wofsIncrement], [use_date], [wofsDir], [dbzThresh], [nny], [nnx], [ddm], [wofsLats], [wofsLons])

    if (maxCores <= len(start_valids_inds)):
        njobs = maxCores
    else:
        njobs = int(len(start_valids_inds))

    in_ICs_list = run_parallel(get_inICs, args_itr_inICs, n_jobs=njobs)
    
    #Now convert in_ICs_list to xarray
    xr_inICs = inICs_to_xr(in_ICs_list, nWindows, nny, nnx) 

    #Now merge our two xrarrays into a single wofs xarray
    xr_wofs = xr.merge([xr_standard_wofs, xr_inICs])

    #Return the xr Dataset with all wofs-related variables 
    return xr_wofs

#===========

#Below has to do with convolutions/manipulating the predictors 

#Returns binary grid of "footprints" defining the circular kernel 
#given the sizes of the square neighborhood (n_sizes), the radii of the neighborhoods (km), 
#and the grid spacing in km 
def get_footprints(n_sizes, radii_km, km_spacing):

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


#Adds the spatial convolutions to the xarray dataset (since these will be used as predictors)
#@in_ds: Incoming xarray dataframe with 3d predictor fields (nT, nY, nX) that we will
#apply the spatial convolutions to
#@nT: The number of lead times/time windows
#@nY: Number of y points
#@nX: Number of x points
#@footprint_type: "square" or "circle" depending on how the spatial convolution should be done
#@radii_km_pred: List of radii (in km) over which to compute the neighborhood
#@radii_pts_pred: List of radii (in points) over which to compute the neighborhood 
#@all_vars_names is a list of all predictor fields (excluding lat and lon)-- in the order
# that the RF was trained on
#@all_vars_methods is a list of convolution methods (e.g., "max", "min", "abs", "minbut") 
#telling how to do the convolution for each variable, excluding lat and lon, in the order that
# the RF was trained on. 
#@rf_sizes is the square "diameter" of wofs grid points to consider--will round up if fractional values 
#@spacing_km is the grid spacing of the wofs grid in km
#@single_pt_vars is a list of variables that should *NOT* have the spatial convolutions applied 
def add_convolutions(in_ds, nT, nY, nX, footprint_type, \
                            radii_km_pred, radii_pts_pred, all_vars_names, all_vars_methods, rf_sizes,\
                            spacing_km, single_pt_vars):

    #Obtain circular footprint if necessary
    if (footprint_type == "circle"):
        ciruclar_footprints = get_footprints(rf_sizes, radii_km_pred, spacing_km) 


    #TODO: Might have to rename initial variables (radius of 0) 

    #We will add the convolutions as a separate variable 
    #Each convolution will be named according to the km_radius
    #Starting at 1 because the first radius is always 0--i.e., no convolution, so we can skip
    for r in range(1,len(radii_km_pred)):
        km_radius = radii_km_pred[r]
        pt_radius = radii_pts_pred[r]
        #print (pt_radius) 
        if (footprint_type == "circle"):
            circular_footprint = circular_footprints[r]

        for v in range(len(all_vars_names)):
            var_name = all_vars_names[v]
            var_method = all_vars_methods[v] 

            #Only add the convolution if the variable is not in the single_pt_vars        
            if (var_name not in single_pt_vars):

                #Need to add the appropriate convolved field. 
                #To do this, first obtain the original field 
                orig_field = in_ds[var_name].values

                #Do the convolution
                if (var_method == "max"):
                    if (footprint_type == "circle"):
                        conv = maximum_filter(orig_field, footprint = circular_footprint)
                    elif (footprint_type == "square"):
                        #NOTE: Have to give tuple (0,radius, radius) to prevent the smoothing from happening in time 
                        conv = maximum_filter(orig_field, (0,pt_radius,pt_radius))
                elif (var_method == "min"):
                    if (footprint_type == "circle"):
                        conv = minimum_filter(orig_field, footprint=circular_footprint)
                    elif (footprint_type == "square"):
                        conv = minimum_filter(orig_field, (0,pt_radius,pt_radius))
                elif (var_method == "abs"):
                    if (footprint_type == "circle"):
                        conv_hi = maximum_filter(orig_field, footprint=circular_footprint)
                        conv_low = minimum_filter(orig_field, footprint=circular_footprint)
                    elif (footprint_type == "square"):
                        conv_hi = maximum_filter(orig_field, (0,pt_radius,pt_radius))
                        conv_low = minimum_filter(orig_field, (0,pt_radius,pt_radius))

                    #now, take the max absolute value between the two
                    conv = np.where(abs(conv_low) > abs(conv_hi), conv_low, conv_hi)

                elif (var_method == "minbut"): #Handle the case where we want to take min, but keep -1 if there is no meaningful min
                    #Can start by renaming the -1--assigning this a high value 
                    conv_hi = np.where(orig_field == -1, 999999.9, orig_field)

                    if (footprint_type == "circle"):
                        conv_low = minimum_filter(conv_hi, footprint=circular_footprint)
                    elif (footprint_type == "square"):
                        conv_low = minimum_filter(conv_hi, (0,pt_radius,pt_radius))


                    #Now reassign 999999.9 to -1
                    conv = np.where(conv_low==999999.9, -1, conv_low)

                #Now, add the convolved field as a new variable 
                new_name = "%s_r%s" %(var_name, km_radius)

                in_ds[new_name] = (["lead_time", "y", "x"], conv)
 


    return in_ds


#==================

#Scripts related to getting the predictions

#Returns a full list of the predictors used -- including lat, lon, and all 
#spatial convolutions. 
#@radii_km_pred: List of radii (in km) over which to compute the neighborhood
#@all_vars_names is a list of all predictor fields (excluding lat and lon and 
#excluding the spatial convolutions)-- in the order that the RF was trained on
#@single_pt_vars is a list of the predictor fields that did *not* use convolution
def get_predictor_list(radii_km_pred, all_fields_names, single_pt_vars):

    new_names = [] #Will hold the list of new names 

    for r in range(len(radii_km_pred)):
        radii_km = radii_km_pred[r]
        for v in range(len(all_fields_names)):
            curr_name = all_fields_names[v]
            if (radii_km == 0): #If no spatial smoothing, there's no "r0" appended
                new_names.append(curr_name)
            else: #for the larger neighborhoods
                if (curr_name not in single_pt_vars):
                    new_name = "%s_r%s" %(curr_name, radii_km)
                    new_names.append(new_name) 
        

    #At the end, we need to add lat and lon
    new_names.append("lat")
    new_names.append("lon")         

    return new_names 


# Handles the parallelization to get 
#the predictors in 1d, which is needed for giving the data to the RF 
#Returns a list of 2-d arrays of shape (total points, total vars), which can be fed to the RF. 
#Each element in the list corresponds to a successive time window 
#@all_predictors_ds is the xarray Dataset containing all of the predictor variables and their convolutions,
# but still in 3-d format (nT, nY, nX) 
#@lead_time_number is an integer (starting with 0) corresonding to the particular time window of interest
#@maxCores is the number of max_cores specified for multiprocessing
#@nY is number of y points
#@nX is number of x points 
#@predictor_list is the list of all predictors--including convolutions--in the training order 
def run_1d_predictors(all_predictors_ds, nWindows, maxCores,\
                        nY, nX, predictor_list):

    nV = len(predictor_list) #total number of variables that will go into RF 

    n_samples = int(nY*nX) #Total number of wofs points 

    time_index = np.arange(nWindows) 

    #Parallelize for each time window 
    args_itr_1d = to_iterator([all_predictors_ds], time_index, [nY], [nX], [nV], [predictor_list], [n_samples])

    if (maxCores <= len(time_index)):
        njobs = maxCores
    else:
        njobs = len(time_index)

    #A list of, technically 2-d fields of dimension (total points, number of variables). Each
    #element of the list is valid for each successive time window 
    one_d_predictor_list = run_parallel(extract_1d, args_itr_1d, n_jobs=njobs)

    return one_d_predictor_list


# Gets the predictors in 1d, which is needed for giving the data to the RF 
#@all_pred_ds is the xarray Dataset containing all of the predictor variables and their convolutions,
# but still in 3-d format (nT, nY, nX) 
#@time_ind is an integer (starting with 0) corresonding to the particular time window of interest
#@NY is number of y points
#@NX is number of x points 
#@NV is the number of variables in predictorList
#@predictorList is the list of all predictors--including convolutions--in the training order 
def extract_1d(all_pred_ds, time_ind, NY, NX, NV, predictorList, nSamples):
    
    #For the given lead time, for each variable, we need to convert the 2-d field to a 1-d list 
    one_d_predictors = np.zeros((NY, NX, NV))
    for v in range(NV):
        curr_var = predictorList[v]
        if (curr_var == "lat" or curr_var == "lon"): #These have no time dimension
            one_d_predictors[:,:,v] = all_pred_ds[curr_var].to_numpy()
        else: #All other variables do have a time dimension 
            one_d_predictors[:,:,v] = all_pred_ds[curr_var].sel(lead_time=time_ind).to_numpy()

    #Now, flatten
    one_d_predictors = one_d_predictors.reshape(nSamples, -1)

    return one_d_predictors


#Drives the parallelization of obtaining RF predictions for each hazard and lead time. 
#@OneD_pred_list is the list of, technically 2d, predictors. Shape of each element is 
#(total # points, total # vars). Each element corresponds to a different lead time/window 
#@maxCores is the specified maximum number of cores
#@nny is number of y points
#@nnx is number of x points 
#@hazs is list of hazard names
#@pkl_hail_list is list of hail pkl files (one for each lead time) corresponding to the trained RF
#@pkl_wind_list is list of wind pkl files (one for each lead time) corresponding to the trained RF
#@pkl_torn_list is list of torn pkl files (one for each lead time) corresponding to the trained RF
def make_predictions(OneD_pred_list, maxCores, nny, nnx, hazs, pkl_hail_list, pkl_wind_list, \
                        pkl_torn_list):


    nT = len(OneD_pred_list) #number of time windows 

    time_indices = np.arange(nT) 
    haz_indices = np.arange(len(hazs)) 
    #print (time_indices)
    #print (haz_indices) 
    #We will parallelize over lead times and hazards 

    #Create a pkl list corresponding to each hazard 
    pkl_list = [] 
    for h in hazs:
        if (h == "hail"):
            pkl_list.append(pkl_hail_list)
        elif (h == "wind"):
            pkl_list.append(pkl_wind_list)
        elif (h == "torn"):
            pkl_list.append(pkl_torn_list) 

    #TODO: Come back here
    args_itr_rf = to_iterator([pkl_list], [OneD_pred_list], haz_indices, time_indices, [nny], [nnx])

    #Set the number of jobs
    if (maxCores <= (int(nT*len(hazs)))):
        njobs = maxCores
    else: 
        njobs = int(nT*len(hazs))

    #Will first span the times, then span the hazards 
    two_d_probs_list = run_parallel(predictRF, args_itr_rf, n_jobs = njobs )
    #TODO: Will need to then put predictions back into 2d 

    #print (len(two_d_probs_list))
    #print (max(two_d_probs_list[0].flatten()), max(two_d_probs_list[10].flatten()))
    #OK: This is good: two_d_probs_list is different!!! And looks good 

    #Will create an xarray dataset of shape (nT, nY, nX) with probs for hail, wind, and tornadoes 
    xr_probs = probs_to_xr(two_d_probs_list, nT, nny, nnx, hazs)

    #TODO: Yes, maybe convert this list to xarray dataset and return that--then can use that for plotting. 

    return xr_probs

#Puts the list of output probabilities into an xarray dataset 
#@probs_list is a list of 2-d RF probabilties that loops most quickly over the lead times and then loops over hazards
#@NT is the number of time windows/lead times
#@NY is the number of y points
#@NX is the number of x points
#@hazard_list is a list of the hazard names
def probs_to_xr(probs_list, NT, NY, NX, hazard_list):
    
    NH = len(hazard_list) 
    #Need to put in intermediate array 
    new_arr = np.zeros((NT, NY, NX, NH))
    count = 0 
    for h in range(NH):
        for t in range(NT):
            new_arr[t,:,:,h] = probs_list[count]
            count += 1

    #Put into new Dataset
    new_xr = xr.Dataset(data_vars=None, coords={"lead_time": (range(NT)), "y": (range(NY)), "x": (range(NX))})
    for h in range(NH):
        Haz = hazard_list[h]
        varname = "%s_probs" %Haz
        new_xr[varname] = (["lead_time", "y", "x"], new_arr[:,:,:,h])


    return new_xr


#Actually runs the predictors through the RF for given hazard and lead time 
#Returns 
#@pklList is a list of all pkl files. List has one element for each hazard. And each element
# holds the number of lead times corresponding to the different time windows. 
#@predictor_set_times is a list of 2-d arrays (shape: (total points, total vars) ) to
# be fed into the RF. Each element in the list corresponds to one lead time.
#@hazInd is the integer corresponding to the number of hazard (starting at 0--and relating to
#the order given at the start of the script)
#@timeInd is the integer corresponding to the number of time window (starting at 0) 
def predictRF(pklList, predictor_set_times, hazInd, timeInd, nY, nX):

    predictor_set = predictor_set_times[timeInd]
    #print (predictor_set) 
    #print (predictor_set.shape)     

    #Get appropriate pkl file from list 
    pkl_file = pklList[hazInd][timeInd]
    #print (pkl_file) 

    clf = unpickle(pkl_file) 
    clf_probs = clf.predict_proba(predictor_set)[:,1]

    #Convert 1-d back to 2-d field 
    two_d_probs = clf_probs.reshape(nY, nX) 

    return two_d_probs 

#Loads the pickled file into an array
def unpickle(Filename):
    f = open(Filename, 'rb')
    new = pickle.load(f)
    f.close()

    return new

#writes the probs for each hazard and lead time into a text file 
#@probs_DS is an xarray dataset with variables hail_probs, wind_probs, torn_probs
#The dimensions of each variable are: (nT, nY, nX) 
#@hazards is a list of hazard names 
#@nT is number of time windows 
#@RF_outdir is the directory where to place the output 
#@obsRadius is the string corresponding to the observation radius (i.e., radius for
# which the probs are valid) 
#@wofs_init is string of wofs initialization time
#@start_valids_list is a list of start valid times (for each time window) 
#@end_valids_list is a list of end valid times (for each time window) 
def save_to_text(probs_DS, hazards, nT, RF_outdir, obsRadius, wofs_init, start_valids_list, end_valids_list):

    #Save each hazard and lead time as a text file 
    for h in range(len(hazards)):
        Haz = hazards[h]
        varname = "%s_probs" %Haz
        for t in range(nT):
            print (varname, t) 
            svalid = start_valids_list[t]
            evalid = end_valids_list[t]
            #time_str = str(t+1) #for saving to file 
            prob_field = probs_DS[varname].sel(lead_time=t).to_numpy()
            #print (prob_field) 
            #print (max(prob_field.flatten()))
            #print (prob_field.shape) 
            
            #Write out to file 
            np.savetxt("%s/%s_i%s_v%s-%s_r%s.txt" %(RF_outdir, varname, wofs_init, svalid, evalid, obsRadius), prob_field)
    

    return 


#================
#Save the 1-d predictors 

#Saves the 1-d predictors to file (also does the random sampling) 
def save_1d_predictors(ListOf1DPreds, nWindows, samp_rate, nY, nX, startValids, endValids, useDate, iTime, \
                        full_npy_outdir, dat_outdir):

    nTotal = int(nY*nX)
    nSamples = int(nTotal*samp_rate) 

    #TODO: Get random indices to sample
    if (samp_rate < 1.):
        rand_inds = np.random.choice(np.arange(nTotal), size=nSamples, replace=False)

    for t in range(nWindows): 
        full_predictions = ListOf1DPreds[t]
        startValid = startValids[t]
        endValid = endValids[t] 

        sampled_predictions = full_predictions[rand_inds, :] 

        #Convert to float 32
        full_predictions = np.float32(full_predictions)
        sampled_predictions = np.float32(sampled_predictions) 

        #Save to file 
        np.save("%s/wofs1d_%s_%s_v%s-%s.npy" %(full_npy_outdir, useDate, iTime, startValid, endValid), full_predictions)
        sampled_predictions.tofile("%s/wofs1d_%s_%s_v%s-%s.dat" %(dat_outdir, useDate, iTime, startValid, endValid)) 
        #Save rand_inds too
        np.save("%s/rand_inds_%s_%s_v%s-%s.npy" %(dat_outdir, useDate, iTime, startValid, endValid), rand_inds)
 

    return 



##########################
# User defined parameters
##########################

#Remove after debugging
#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

##############
#Constants 
##############

#########################
# What will be passed in
#########################

#date = "20190506"
#next_date = "20190507" 

date = "20210604"
next_date = "20210605" 

#time_window = 30 #in minutes 
#n_windows = 6

time_window = 30
n_windows = 6

spinup_minutes = 25 #Assume forecast starts 25 minutes after initialization for "spinup" 

max_cores = 30

sampling_rate = 0.1 #fraction of examples to use each day 

init_time = "0230"
start_valid = "0255"
ps_start = "0254" 

extrap_time = 181.0 #Time in minutes -- how far to extrapolate the probSevere probabilities 

next_day_inits = ["0000", "0030", "0100", "0130", "0200", "0230", "0300", "0330", "0400", "0430", "0500"]

#Get use_date 
if (init_time in next_day_inits):
    Use_Date = next_date
else:
    Use_Date = date 

#NOTE: Only for testing
template_dir = "/work/mflora/SummaryFiles/%s/%s" %(date, init_time)
template_file = "%s/wofs_ALL_00_%s_%s_%s.nc" %(template_dir, Use_Date, init_time, init_time)

ps_dir = "/work/eric.loken/wofs/probSevere"
wofs_dir = "/work/mflora/SummaryFiles/%s/%s" %(date, init_time)

wofs_fields_file = "standard_wofs_variables_v9.txt" 
wofs_methods_file = "standard_wofs_methods_v9.txt" 

#Holds all predictors/methods from WoFS and PS combined (and in the order used during training) 
all_fields_file = "all_fields_v9.txt"
all_methods_file = "all_methods_v9.txt" 

#Tells which fields do not use spatial convolution and are only recorded at the given point
single_pt_file = "single_point_fields_v9.txt" 

#outdir = "."

#outdir_full_npy = "/work/eric.loken/wofs/parallelized_v10/fcst/less_extrap/%s/full_npy" %time_window
#outdir_dat = "/work/eric.loken/wofs/parallelized_v10/fcst/less_extrap/%s/dat" %time_window 
outdir_full_npy = "/work/eric.loken/wofs/paper6/fcst/%s/full_npy" %time_window
outdir_dat = "/work/eric.loken/wofs/paper6/fcst/%s/dat" %time_window


#rf_outdir = "rf_probs/less_extrap" #where to put RF text files  -- NOT ACTUALLY USED in this script anymore 

#print (start_valids)
#print (orig_start_leads) 
#quit() 

##############
#Constants 
##############

dx_km = 3.0 #horizontal grid spacing in km 

ps_thresh = 0.01 #ps objects must have probs greater than or equal to this to be considered

dm = 18 #number of WoFS members 

min_radius = 1.5 #in km
#max_radius = 30.0 #in km -- max radius for PS objects -- was original 30.0 
max_radius = 1.5 

obs_radius = "30.0" 

wofs_increment = 5 #in minutes; time between each new wofs summary file

radius = max_radius


#Handle wofs stuff 

bottom_hour_inits = ["1730", "1830", "1930", "2030", "2130", "2230", "2330", "0030", "0130",\
                     "0230", "0330", "0430", "0530", "0630", "0730", "0830", "0930", "1030",\
                     "1130", "1230", "1330", "1430", "1530", "1630"]

#Bottom of the hour forecasts have to be cut 60 minutes short (they don't go out as far as 
#top of the hour forecasts). 
#Also need to update the start and end valids 
if (init_time in bottom_hour_inits):
    n_windows = n_windows - math.ceil(60/time_window)

#How long (since wofs initialization) until the very end of the forecast period? 
#25 minutes from initialization for spinup + number of time windows*forecast time window 
full_time_window = spinup_minutes + time_window*n_windows

orig_start_leads, orig_end_leads = find_start_end_leads(start_valid, time_window, n_windows)

#TODO: Need to get start_valids, end_valids from start_valid
#Equivalent to start_valids_multi
start_valids = get_curr_valids(float(start_valid), n_windows, time_window)

#Get initial end valid -- i.e., the second element of start_valids
if (n_windows == 0):
    print ("No valid windows for this period." ) 
    quit() 

if (len(start_valids) > 1):
    end_valid = start_valids[1]
else:
    end_valid = get_curr_valids(float(start_valid), 2, time_window)
    end_valid = end_valid[1] 

end_valids = get_curr_valids(float(end_valid), n_windows, time_window)

pkl_dir = "/work/eric.loken/wofs/realtime_like/pkl_files"
hail_pkl_filenames = ["%s/all_v9_hail_r%s_30min_chunks%s_1.pkl" %(pkl_dir, obs_radius, n) for n in np.arange(1,n_windows+1,1)] 
wind_pkl_filenames = ["%s/all_v9_wind_r%s_30min_chunks%s_1.pkl" %(pkl_dir, obs_radius, n) for n in np.arange(1,n_windows+1,1)] 
torn_pkl_filenames = ["%s/all_v9_torn_r%s_30min_chunks%s_1.pkl" %(pkl_dir, obs_radius, n) for n in np.arange(1,n_windows+1,1)] 

#Spatial neighborhood info stuff
conv_type = "square" #"square" or "circle" 
predictor_radii_km = [0.0, 15.0, 30.0, 45.0, 60.0] #how far to "look" spatially
#square "diameter" of wofs grid points to consider--will round up if fractional radius.
n_sizes_rf = [(round(p/dx_km)*2 + 1) for p in predictor_radii_km]
predictor_radii_pts = [math.ceil((p-dx_km/2)/dx_km) for p in predictor_radii_km]

max_x = math.ceil((radius - 1.5)/dx_km)
max_y = math.ceil((radius - 1.5)/dx_km)
hazards = ["hail", "wind", "torn"] 

indiv_member_vars = ["m%s_uh_2to5" %str(int(a)) for a in np.arange(1,19,1)]

####################

#Step 1a: Get grid stats from template file -- Need for both WoFS and PS, so do before looping 

ny, nx, wofs_lats, wofs_lons, tlat1, tlat2, stlon, sw_lat, ne_lat, sw_lon, ne_lon = find_grid_stats(template_file) 

#Step 1b: Read in wofs fields and wofs computation methods (e.g., "min" or "max") 
wofs_fields = np.genfromtxt(wofs_fields_file, dtype='str') 
comp_methods = np.genfromtxt(wofs_methods_file, dtype='str') 

all_fields = np.genfromtxt(all_fields_file, dtype='str') 
all_methods = np.genfromtxt(all_methods_file, dtype='str') 

single_pt_fields = np.genfromtxt(single_pt_file, dtype='str') 


#Step 2a: ProbSevere preprocessing. Get hail, wind, tornado 2-d grids (Could also define as a-c; 1 each for tornado/wind/hail??) 

#Gets the xarray (with dimensions (dT, dY, dX)) for all of the ProbSevere variables. 
#Obtain the ProbSevere xarray dataset with all relevant probSevere variables 
ps_xr_ds = do_probSevere_preprocessing(init_time, start_valids, end_valids, ps_start, date, next_date, next_day_inits, \
    ps_dir, hazards, extrap_time, ps_thresh, min_radius, max_radius, orig_start_leads, orig_end_leads, \
    wofs_lats, wofs_lons, ny, nx, max_cores, n_windows)


#So, ps_xr_ds is wrong right now. TODO -- Problem might be how I'm transferring to xarray
#print (ps_xr_ds)
#two_d = ps_xr_ds['hail_raw_probs'].sel(lead_time=0).to_numpy()
#print (two_d.reshape(int(ny*nx),-1)[1001])
#quit() 

#Step 2b: WoFS preprocessing. Get 2-d grids for all wofs variables -- temporal aggregation

#Obtain the wofs xarray dataset with all relevant wofs variables 
wofs_xr_ds = do_wofs_preprocessing(init_time, time_window, start_valids, end_valids, wofs_increment, wofs_fields,\
                    comp_methods, date, next_date, wofs_dir, next_day_inits, indiv_member_vars, ny, nx, dm, full_time_window, \
                    wofs_lats, wofs_lons, max_cores, n_windows)

#Step 3a: Concatenate ProbSevere and WoFS xarrays 
predictors_ds = xr.merge([ps_xr_ds, wofs_xr_ds])

#Step 3b: Add the lat and longitude as predictors 
predictors_ds['lat'] = (["y", "x"], wofs_lats)
predictors_ds['lon'] = (["y", "x"], wofs_lons)


#Step 4: Add convolution neighborhoods to xarray 

conv_predictors_ds = add_convolutions(predictors_ds, n_windows, ny, nx, conv_type, \
                            predictor_radii_km, predictor_radii_pts, all_fields, all_methods, \
                            n_sizes_rf, dx_km, single_pt_fields)

#all_pred_ds[curr_var].sel(lead_time=time_ind).to_numpy()

#print (conv_predictors_ds.data_vars) 
#Step 5: Convert to 1d (for each lead time and hazard) and feed into appropriate pkl to get prediction
#(should ultimately be able to parallelize)

#Need to get a list of all_predictors 
all_predictor_list = get_predictor_list(predictor_radii_km, all_fields, single_pt_fields)


list_of_1d_preds = run_1d_predictors(conv_predictors_ds, n_windows, max_cores, ny, nx, all_predictor_list)


#print predictors to file 
#(ListOf1DPreds, nWindows, samp_rate, nY, nX, startValids, endValids, useDate, iTime, \
#                        full_npy_outdir, dat_outdir)
save_1d_predictors(list_of_1d_preds, n_windows, sampling_rate, ny, nx, start_valids, end_valids, date, init_time,\
                    outdir_full_npy, outdir_dat) 

#time0 = list_of_1d_preds[0]

#for z in range(len(time0[0,:])):
#    cvar = all_predictor_list[z]
#    print (cvar) 
#    print (time0[1001,z])
#
#quit()  

#Step 6: Load RF; run variables through the RF  
#probs_ds = make_predictions(list_of_1d_preds, max_cores, ny, nx, hazards, hail_pkl_filenames, wind_pkl_filenames, \
#                torn_pkl_filenames)

#FUTURE: Will save these to netcdf. First, though, let's just save to a file and try to plot. See if they're equivalent to what we have. 
#save_to_text(probs_ds, hazards, n_windows, rf_outdir, obs_radius, init_time, start_valids, end_valids)


#Step 7: Save output files to netcdf. (Save predictor tables too, for future training???) 
#TODO: Write to netcdf. 

