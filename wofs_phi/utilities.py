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
                if os.path.exists(write_dir):
                    sftp.get(csv_dir + file, write_dir + file)
                else:
                    os.mkdir(write_dir)
                    sftp.get(csv_dir + file, write_dir + file)

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
        if date_time.hour >= 0:
            date_time = date_time + datetime.timedelta(days = 1)
        
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