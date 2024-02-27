import os
#import paramiko
#from scp import SCPClient
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import pyproj
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import AzimuthalEquidistantConversion
from shapely.ops import transform
from shapely.geometry import Point
import geopandas as gpd
import datetime
import math
from scipy.ndimage import maximum_filter, minimum_filter

def copy_torp(write_dir,usr, pwd):
    ssh_client=paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname='myrorss2',username=usr,password=pwd)
    sftp = ssh_client.open_sftp()

    for date in sftp.listdir('/work/thea.sandmael/radar'):
        sftp.chdir('/work/thea.sandmael/radar/' + date)
        for site in sftp.listdir('/work/thea.sandmael/radar/' + date):
            csv_dir = '/work/thea.sandmael/radar/' + date + '/' + site + '/netcdf/torp/TORPcsv/00.50/'
            try:
                sftp.chdir(csv_dir)
            except:
                continue

            for file in sftp.listdir():
                if os.path.exists(write_dir + date + '/'):
                    sftp.get(csv_dir + file, write_dir + date + '/' + file)
                else:
                    os.mkdir(write_dir + date + '/')
                    sftp.get(csv_dir + file, write_dir + date + '/' + file)

def geodesic_point_buffer(lon, lat, km):
    proj_crs = ProjectedCRS(conversion = AzimuthalEquidistantConversion(lat, lon))
    proj_wgs84 = pyproj.Proj('EPSG:4326')
    Trans = pyproj.Transformer.from_proj(proj_crs,proj_wgs84,always_xy=True).transform

    return transform(Trans, Point(0, 0).buffer(km * 1000))

def parse_date(string):
        '''The ID is in the form of (ID number)_YYYYMMDD-HHMMSS, this
        creates a datetime object to store the time'''
        
        dt_arr = string.split('-')
        date_str = dt_arr[0]
        time_str = dt_arr[1]
        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:])
        hour = int(time_str[0:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:])
        
        date_time = datetime.datetime(year, month, day, hour, minute, second)
        #if date_time.hour >= 0:
        #    date_time = date_time + datetime.timedelta(days = 1)
        
        return date_time

def get_init_time(string):
    dt = parse_date(string)
    curr_hour = dt.hour
    curr_min = dt.minute
    if curr_min <= 30:
        dt = dt.replace(minute = 30, second = 0)
    else:
        if curr_hour < 23:
            dt = dt.replace(hour = curr_hour + 1, minute = 0, second = 0)
        else:
            dt = dt.replace(hour = 0, minute = 0, second = 0)
    
    return dt

def haversine(lat1, lon1, lat2, lon2):
    r = 6371
    
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    lon1 = math.radians(lon1)
    lon2 = math.radians(lon2)
    
    dLat = (lat1 - lat2)/2
    dLon = (lon1 - lon2)/2
    
    avgLat = (lat1 + lat2)/2
    
    first = math.pow((math.sin(dLat)), 2)
    second = math.pow(1 - math.sin(dLat), 2)
    third = math.pow(math.sin(avgLat), 2)
    fourth = math.pow(math.sin(dLon), 2)
    
    dist = 2*r*math.asin(math.sqrt(first + ((second - third) * fourth)))
    
    return abs(dist)

def haversine_get_lon(lat, lon, x_dist):
    '''Gets a longitude given a point and a zonal distance (in km)'''
    
    r = 6371
    lat = math.radians(lat)
    dLat = 0
    
    first = math.pow((math.sin(x_dist/(2*r))), 2)
    second = math.pow((math.sin(dLat)), 2)
    third = 1 - math.pow((math.sin(dLat)), 2)
    fourth = math.pow((math.sin(lat)), 2)
    
    top = first - second
    bottom = third - fourth
    
    new_lon = (2*math.asin(math.sqrt(top/bottom))) * (180/math.pi)
    
    return lon + (new_lon * (abs(x_dist)/x_dist))

def haversine_get_lat(lat1, lon1, lon2, y_dist):
    '''Gets a latitude given a point and meridional distance (in km)'''
    
    upper_bound = 90
    lower_bound = 0
    
    lat2 = (upper_bound + lower_bound)/2
    
    dist = abs(haversine(lat1, lon1, lat2, lon2))
    if lat2 < lat1:
        dist = -dist
    
    while abs(dist - y_dist) > 0.001:
        if dist > y_dist:
            upper_bound = lat2
        else:
            lower_bound = lat2
        
        lat2 = (upper_bound + lower_bound)/2
        dist = abs(haversine(lat1, lon1, lat2, lon2))
        if lat2 < lat1:
            dist = -dist
        
    return lat2

def get_footprints(n_sizes, radii_km, km_spacing):
    '''Returns binary grid of "footprints" defining the circular kernel 
    given the sizes of the square neighborhood (n_sizes), the radii of the neighborhoods (km), 
    and the grid spacing in km'''
    
    grids = [] #Will hold a list of grids 
    for n in range(len(n_sizes)):
        n_size = n_sizes[n]
        spatial_radius = radii_km[n]
        grid = np.zeros((n_size, n_size))
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

def add_convolutions(var_method, var, footprint):
    
    if (var_method == "max"):
        conv = maximum_filter(var, footprint = footprint)
    elif (var_method == "min"):
        conv = minimum_filter(var, footprint= footprint)
    elif (var_method == "abs"):
        conv_hi = maximum_filter(var, footprint= footprint)
        conv_low = minimum_filter(var, footprint= footprint)

        #now, take the max absolute value between the two
        conv = np.where(abs(conv_low) > abs(conv_hi), conv_low, conv_hi)
    elif (var_method == "minbut"): #Handle the case where we want to take min, but keep -1 if there is no meaningful min
        #Can start by renaming the -1--assigning this a high value 
        conv_hi = np.where(var == -1, 999999.9, var)
        conv_low = minimum_filter(conv_hi, footprint= footprint)

        #Now reassign 999999.9 to -1
        conv = np.where(conv_low==999999.9, -1, conv_low)
 
    return conv