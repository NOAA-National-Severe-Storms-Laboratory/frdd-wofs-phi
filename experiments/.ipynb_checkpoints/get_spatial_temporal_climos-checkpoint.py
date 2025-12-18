import numpy as np
import sys
sys.path.append('../wofs_phi')
from wofs_phi import config as c
from wofs_phi import utilities
from sklearn.metrics import brier_score_loss as BS
import os
from shutil import copy
import shutil
import copy
import matplotlib.pyplot as plt
import math
import pandas as pd
import matplotlib.ticker as ticker
import datetime as dt
from wofs_phi.wofs_phi import Grid
from wofs_phi.wofs_phi import PS
import imageio.v3 as iio
import skexplain
from copy import deepcopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import skexplain
import pickle
from matplotlib import rc, font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from wofs.common.zarr import open_dataset
import matplotlib.colors as mpl_colors
import netCDF4 as ncdf
import statistics as stat
import netCDF4 as nc
from wofs_phi import multiprocessing_driver as md
import geopandas as gpd
from shapely import Polygon
from shapely import Point

class ConusGrid:
    
    def __init__(self, lats, lons, ny, nx):
        
        self.lats = lats
        self.lons = lons
        self.ny = ny
        self.nx = nx
        
        return

def load_times(date, start_times):
    
    dt_list = [0]*len(start_times)
    for i in range(len(start_times)):
        time = start_times[i]
        date_time_str = str(int(date)) + time
        date_time = dt.datetime.strptime(date_time_str, '%Y%m%d%H%M')
        dt_list[i] = date_time
    
    return np.array(dt_list)

def get_date_times(use_bottom_hours = False):
    
    dates = np.genfromtxt('/home/ryan.martz/python_packages/frdd-wofs-phi/wofs_phi/probSevere_dates.txt')
    if use_bottom_hours:
        start_times = ["1700", "1730", "1800", "1830", "1900", "1930", "2000",\
                       "2030", "2100", "2130", "2200", "2230", "2300", "2330",\
                       "0000", "0030", "0100", "0130", "0200", "0230", "0300",\
                       "0330", "0400", "0430", "0500"]
    else:
        start_times = ["1700", "1800", "1900", "2000", "2100", "2200", "2300",\
                       "0000", "0100", "0200", "0300", "0400", "0500", "0600",\
                       "0700", "0800"]
    
    date_times = []
    
    for date in dates:
        date_times.append(load_times(date, start_times))
    
    return date_times

def load_lsr_file(hazard, date, start_time, end_time):
    
    directory = '/work/eric.loken/wofs/2024_update/SFE2024/obs/full_npy'
    if hazard == 'tornado':
        file = '%s_reps1d_%s_v%s-%s_r39km.npy' %(hazard, date, start_time, end_time)
    elif hazard == 'hail':
        file = '%s_reps1d_%s_v%s-%s_r39km_20_min_buffer.npy' %(hazard, date, start_time, end_time)
    elif hazard == 'wind':
        file = '%s_reps1d_%s_v%s-%s_r375km_20_min_buffer.npy' %(hazard, date, start_time, end_time)
    
    lsrs_1d = np.load('%s/%s' %(directory, file))
    lsrs_2d = lsrs_1d.reshape(300,300)
    
    return lsrs_2d

def load_lsr_probs(hazard, date, start, lead):
    
    init_time = (start - dt.timedelta(minutes = lead)).strftime('%H%M')
    start_time = start.strftime('%H%M')
    end_time = (start + dt.timedelta(minutes = 60)).strftime('%H%M')
    
    directory = '/work/ryan.martz/wofs_phi_data/obs_train/test_fcsts/wofs_psv3_with_torp/%s/'\
    'wofslag_25/length_60/%s/%s' %(hazard, date, init_time)
    
    if hazard == 'hail' or hazard == 'tornado':
        file = 'wofs_psv3_with_torp_obs_trained_rf_%s_raw_probs_%s_init%s_v%s-%s_r39km.txt'\
        %(hazard, date, init_time, start_time, end_time)
    else:
        file = 'wofs_psv3_with_torp_obs_trained_rf_%s_raw_probs_%s_init%s_v%s-%s_r375km.txt'\
        %(hazard, date, init_time, start_time, end_time)
    
    lsr_probs_2d = np.genfromtxt('%s/%s' %(directory, file))
    
    return lsr_probs_2d

def load_warning_file(hazard, date, start_time, end_time):
    
    directory = '/work/ryan.martz/wofs_phi_data/training_data/warnings/'\
    'full_1d_warnings/length_60/%s' %(hazard)
    file = '%s_warnings_%s_v%s-%s_1d.npy' %(hazard, date, start_time, end_time)
    
    warnings_1d = np.load('%s/%s' %(directory, file))
    warnings_2d = warnings_1d.reshape(300,300)
    
    return warnings_2d

def load_warning_probs(hazard, date, start, lead):
    
    init_time = (start - dt.timedelta(minutes = lead)).strftime('%H%M')
    start_time = start.strftime('%H%M')
    end_time = (start + dt.timedelta(minutes = 60)).strftime('%H%M')
    
    directory = '/work/ryan.martz/wofs_phi_data/warnings_train/test_fcsts/wofs_psv3_with_torp/%s/'\
    'wofslag_25/length_60/%s/%s' %(hazard, date, init_time)
    
    file = 'wofs_psv3_with_torp_warnings_trained_rf_%s_raw_probs_%s_init%s_v%s-%s.txt'\
    %(hazard, date, init_time, start_time, end_time)
    
    warning_probs_2d = np.genfromtxt('%s/%s' %(directory, file))
    
    return warning_probs_2d

def check_predictor_file(date, start):
    
    directory = '/work/eric.loken/wofs/2024_update/SFE2024/fcst/full_npy_backup'
    does_predictor_file_exist_list = []
    start_time = start.strftime('%H%M')
    end_time = (start + dt.timedelta(minutes = 60)).strftime('%H%M')
    
    for lead in [30, 60, 90, 120, 150, 180]:
        init_time = (start - dt.timedelta(minutes = lead)).strftime('%H%M')
        ps_init_time = ((start - dt.timedelta(minutes = lead))\
                        + dt.timedelta(minutes = 24)).strftime('%H%M')
        
        file = 'wofs1d_psv3_with_torp_%s_%s_%s_v%s-%s.npy'\
        %(date, init_time, ps_init_time, start_time, end_time)
        
        does_predictor_file_exist_list.append(os.path.exists('%s/%s' %(directory, file)))
        
    does_predictor_file_exist_array = np.array(does_predictor_file_exist_list)
    does_predictor_file_exist = np.any(does_predictor_file_exist_array)
    
    return does_predictor_file_exist


def check_lsr_warning_predictor_file(hazard, date_time):
    
    date_str = date_time.strftime('%Y%m%d')
    
    start_time_str = date_time.strftime('%H%M')
    end_time_str = (date_time + dt.timedelta(minutes = 60)).strftime('%H%M')
    
    try:
        load_lsr_file(hazard, date_str, start_time_str, end_time_str)
        load_warning_file(hazard, date_str, start_time_str, end_time_str)
        return check_predictor_file(date_str, date_time)
    except:
        return False

def load_conus():
    lats = np.load('/work/ryan.martz/wofs_phi_data/experiments/conus_latitudes.npy')
    lons = ((np.load('/work/ryan.martz/wofs_phi_data/experiments/conus_longitudes.npy') - 180) % 360) - 180
    return lats, lons

def get_wofs_grid(date):
    
    wofs_path = '/work2/wof/SummaryFiles/%s/2200/' %(date)
    
    env_file = 'wofs_ENV_00_%s_2200_2200.nc' %(date)
    if os.path.exists('%s/%s' %(wofs_path, env_file)):
        return Grid.create_wofs_grid(wofs_path, env_file)
    
    all_file = 'wofs_ALL_00_%s_2200_2200.nc' %(date)
    if os.path.exists('%s/%s' %(wofs_path, all_file)):
        return Grid.create_wofs_grid(wofs_path, all_file)
    else:
        return a

def get_grid_location_on_conus(wofs_grid, target_lat, target_lon, conus_lats,\
                               conus_lons, lat_tolerance, lon_tolerance):
    
    
    conus_index_wofs_grid_rows = np.where((np.abs(conus_lats - target_lat) <= lat_tolerance)\
                                          & (np.abs(conus_lons - target_lon) <= lon_tolerance))[0]
    conus_index_wofs_grid_cols = np.where((np.abs(conus_lats - target_lat) <= lat_tolerance)\
                                          & (np.abs(conus_lons - target_lon) <= lon_tolerance))[1]
    
    min_dist = 9999
    conus_row = -1
    conus_col = -1
    if len(conus_index_wofs_grid_rows) > 1:
        for i in range(len(conus_index_wofs_grid_rows)):
            test_lat = conus_lats[conus_index_wofs_grid_rows[i], conus_index_wofs_grid_cols[i]]
            test_lon = conus_lons[conus_index_wofs_grid_rows[i], conus_index_wofs_grid_cols[i]]
            dist = utilities.haversine(test_lat, test_lon, target_lat, target_lon)
            if dist < min_dist:
                min_dist = dist
                conus_row = conus_index_wofs_grid_rows[i]
                conus_col = conus_index_wofs_grid_cols[i]
    elif len(conus_index_wofs_grid_rows) == 1:
        conus_row = conus_index_wofs_grid_rows[0]
        conus_col = conus_index_wofs_grid_cols[0]
    elif len(conus_index_wofs_grid_rows) == 0:
        print(z)
    
    return conus_row, conus_col

def map_events_day_to_conus(date_times, hazard, climo_type):
    
    conus_lats, conus_lons = load_conus()
    conus = np.zeros(conus_lats.shape)
    date_str = date_times[0].strftime('%Y%m%d')
    wofs_grid = get_wofs_grid(date_str)
    
    lat_tolerance = 0.015
    lon_tolerance = 0.03
    
    #sw_row, sw_col = get_grid_location_on_conus(wofs_grid, wofs_grid.sw_lat.data, wofs_grid.sw_lon.data,\
    #                                            conus_lats, conus_lons, lat_tolerance, lon_tolerance)
    
    for time in date_times:
        if not (check_lsr_warning_predictor_file(hazard, time)):
            continue
        
        start_time_str = time.strftime('%H%M')
        end_time_str = (time + dt.timedelta(minutes = 60)).strftime('%H%M')
        
        if climo_type == 'lsrs':
            obs = load_lsr_file(hazard, date_str, start_time_str, end_time_str)
        elif climo_type == 'warnings':
            obs = load_warning_file(hazard, date_str, start_time_str, end_time_str)
        
        conus = map_grid_to_conus(obs, wofs_grid, conus, conus_lats,\
                                 conus_lons, lat_tolerance, lon_tolerance)
    
    #fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    #ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=1)
    #ax.contour(conus_lons, conus_lats, conus)
    #fig.savefig('test.png')
        
    return conus

def map_probs_day_to_conus(date_times, hazard, climo_type, lead):
    
    conus_lats, conus_lons = load_conus()
    conus = np.zeros(conus_lats.shape)
    date_str = date_times[0].strftime('%Y%m%d')
    wofs_grid = get_wofs_grid(date_str)
    
    lat_tolerance = 0.015
    lon_tolerance = 0.03
    
    #sw_row, sw_col = get_grid_location_on_conus(wofs_grid, wofs_grid.sw_lat.data, wofs_grid.sw_lon.data,\
    #                                            conus_lats, conus_lons, lat_tolerance, lon_tolerance)
    
    for time in date_times:
        
        start = time + dt.timedelta(minutes = lead)
        start_str = start.strftime('%H%M')
        end_str = (start + dt.timedelta(minutes = 60)).strftime('%H%M')
        
        try:
            lsr_probs = load_lsr_probs(hazard, date_str, start, lead)
            warning_probs = load_warning_probs(hazard, date_str, start, lead)
            lsrs = load_lsr_file(hazard, date_str, start_str, end_str)
            warnings = load_warning_file(hazard, date_str, start_str, end_str)
            if climo_type == 'lsrs':
                probs = lsr_probs
            elif climo_type == 'warnings':
                probs = warning_probs
        except:
            continue
        
        conus = map_grid_to_conus(probs, wofs_grid, conus, conus_lats,\
                                 conus_lons, lat_tolerance, lon_tolerance)
    
    #fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    #ax.add_feature(cfeature.STATES, edgecolor='black', linewidth=1)
    #ax.contour(conus_lons, conus_lats, conus)
    #fig.savefig('test.png')
        
    return conus

def map_grid_to_conus(grid, wofs_grid, conus, conus_lats, conus_lons, lat_tolerance, lon_tolerance):
    
    rows = np.where(grid >= 0.01)[0]
    cols = np.where(grid >= 0.01)[1]
    
    for i in range(len(rows)):
        r = rows[i]
        c = cols[i]
        lat = wofs_grid.lats.data[r,c]
        lon = wofs_grid.lons.data[r,c]
        conus_row, conus_col = get_grid_location_on_conus(wofs_grid, lat, lon, conus_lats,\
                                                          conus_lons, lat_tolerance, lon_tolerance)
        conus[conus_row, conus_col] += grid[r,c]
    
    return conus

def map_wofs_grid_to_conus(date, conus_lats, conus_lons, conus):
    
    date_str = str(int(date))
    wofs_grid = get_wofs_grid(date_str)
    
    wofs_sw = Point(wofs_grid.lons.data[0,0], wofs_grid.lats.data[0,0])
    wofs_nw = Point(wofs_grid.lons.data[-1,0], wofs_grid.lats.data[-1,0])
    wofs_ne = Point(wofs_grid.lons.data[-1,-1], wofs_grid.lats.data[-1,-1])
    wofs_se = Point(wofs_grid.lons.data[0,-1], wofs_grid.lats.data[0,-1])
    wofs_box = Polygon([wofs_sw, wofs_nw, wofs_ne, wofs_se])
    
    wofs_gdf = gpd.GeoDataFrame([{'geometry': wofs_box}], crs="EPSG:4326")
    
    conus_grid_ny = conus_lats.shape[0]
    conus_grid_nx = conus_lats.shape[1]
    conus_grid = ConusGrid(conus_lats, conus_lons, conus_grid_ny, conus_grid_nx)
    
    conus_gdf = PS.get_wofs_gdf(conus_grid)
    
    overlap_gdf = gpd.sjoin(conus_gdf, wofs_gdf, how="inner", predicate="intersects")
    
    for i in range(len(overlap_gdf.wofs_i.values)):
        conus[overlap_gdf.wofs_j.values[i],overlap_gdf.wofs_i.values[i]] += 1
    
    return conus

def aggregate_events_day_by_time(date_times, hazard, climo_type):
    
    date_str = date_times[0].strftime('%Y%m%d')
    
    event_count = np.zeros(date_times.shape)
    for i in range(len(date_times)):
        time = date_times[i]
        if not (check_lsr_warning_predictor_file(hazard, time)):
            event_count[i] = -1
            continue
        
        start_time_str = time.strftime('%H%M')
        end_time_str = (time + dt.timedelta(minutes = 60)).strftime('%H%M')
        
        if climo_type == 'lsrs':
            obs = load_lsr_file(hazard, date_str, start_time_str, end_time_str)
        elif climo_type == 'warnings':
            obs = load_warning_file(hazard, date_str, start_time_str, end_time_str)
        
        event_count[i] = np.sum(obs)
    
    return event_count

def aggregate_probs_day_by_time(date_times, hazard, climo_type, lead):
    
    date_str = date_times[0].strftime('%Y%m%d')
    
    event_count = np.zeros(((date_times.shape[0]+3)*2,))
    for i in range(len(date_times)):
        time = date_times[i]
        
        start = time + dt.timedelta(minutes = lead)
        start_str = start.strftime('%H%M')
        end_str = (start + dt.timedelta(minutes = 60)).strftime('%H%M')
        
        try:
            lsr_probs = load_lsr_probs(hazard, date_str, start, lead)
            warning_probs = load_warning_probs(hazard, date_str, start, lead)
            lsrs = load_lsr_file(hazard, date_str, start_str, end_str)
            warnings = load_warning_file(hazard, date_str, start_str, end_str)
            if climo_type == 'lsrs':
                probs = lsr_probs
            elif climo_type == 'warnings':
                probs = warning_probs
        except:
            try:
                event_count[int((i*2)+(lead/30))] = -1
            except:
                pass
            continue
        
        try:
            event_count[int((i*2)+(lead/30))] = np.sum(probs)
        except:
            pass
    
    return event_count

def aggregate_probs_by_day(date_times, hazard, climo_type, lead):
    
    date_str = date_times[0].strftime('%Y%m%d')
    day_all_probs = []
    
    for init_time in date_times:
        
        start = init_time + dt.timedelta(minutes = lead)
        start_str = start.strftime('%H%M')
        end_str = (start + dt.timedelta(minutes = 60)).strftime('%H%M')
        
        try:
            lsr_probs = load_lsr_probs(hazard, date_str, start, lead)
            warning_probs = load_warning_probs(hazard, date_str, start, lead)
            lsrs = load_lsr_file(hazard, date_str, start_str, end_str)
            warnings = load_warning_file(hazard, date_str, start_str, end_str)
            if climo_type == 'lsrs':
                day_all_probs.extend(lsr_probs.reshape((90000,)))
            elif climo_type == 'warnings':
                day_all_probs.extend(warning_probs.reshape((90000,)))
        except:
            continue
        
    return day_all_probs

def get_wofs_spatial_climo():
    
    save_dir = '/work/ryan.martz/wofs_phi_data/experiments/climatologies/spatial_climatology'
    save_file = 'wofs_grids_spatial_climo_conus.npy'
    dates = np.genfromtxt('/home/ryan.martz/python_packages/frdd-wofs-phi/wofs_phi/probSevere_dates.txt')
    conus_lats, conus_lons = load_conus()
    conus = np.zeros(conus_lats.shape)
    
    iterator = md.to_iterator(dates, [conus_lats], [conus_lons], [conus])
    results = md.run_parallel(map_wofs_grid_to_conus, iterator, nprocs_to_use = 20,\
                                   description = 'Mapping WoFS Grids to CONUS')
    
    for result in results:
        conus += result
    
    utilities.save_data(save_dir, save_file, conus, 'npy')
    
    return
    
def get_events_spatial_climos():
    
    hazards = ['hail', 'wind', 'tornado']
    climo_types = ['lsrs', 'warnings']
    save_dir = '/work/ryan.martz/wofs_phi_data/experiments/climatologies/spatial_climatology'
    date_times_by_date = get_date_times()
    
    for hazard in hazards:
        for climo_type in climo_types:
            iterator = md.to_iterator(date_times_by_date, [hazard], [climo_type])
            results = md.run_parallel(map_events_day_to_conus, iterator, nprocs_to_use = 20,\
                                   description = 'Mapping %s %s to conus grid' %(hazard, climo_type))
            total_conus = np.zeros(results[0].shape)
            for conus in results:
                total_conus += conus
            
            save_file = '%s_%s_spatial_climo_conus.npy' %(hazard, climo_type)
            utilities.save_data(save_dir, save_file, total_conus, 'npy')
    
    return

def get_probs_spatial_climos(use_bottom_hours = False):
    
    hazards = ['hail', 'wind', 'tornado']
    climo_types = ['lsrs', 'warnings']
    leads = [150, 180]
    save_dir = '/work/ryan.martz/wofs_phi_data/experiments/climatologies/spatial_climatology'
    date_times_by_date = get_date_times(use_bottom_hours = use_bottom_hours)
    
    for lead in leads:
        print('Lead Time: %s min.' %(lead))
        for hazard in hazards:
            for climo_type in climo_types:
            
                
                if use_bottom_hours:
                    save_file = '%s_%s_probs_spatial_climo_conus_%s-%smin.npy' %(hazard, climo_type, lead-30, (lead-30)+60)
                else:
                    save_file = '%s_%s_probs_spatial_climo_conus_%s-%smin_top_hour_inits_only.npy' %(hazard, climo_type, lead-30, (lead-30)+60)
                #if os.path.exists('%s/%s' %(save_dir, save_file)):
                #    continue
                
                iterator = md.to_iterator(date_times_by_date, [hazard], [climo_type], [lead])
                results = md.run_parallel(map_probs_day_to_conus, iterator, nprocs_to_use = 20,\
                                       description = 'Mapping %s %s probs to conus grid' %(hazard, climo_type))
                total_conus = np.zeros(results[0].shape)
                for conus in results:
                    total_conus += conus
                
                utilities.save_data(save_dir, save_file, total_conus, 'npy')
    
    return

def get_temporal_event_climos(use_bottom_hours = False):
    
    hazards = ['hail', 'wind', 'tornado']
    climo_types = ['lsrs', 'warnings']
    save_dir = '/work/ryan.martz/wofs_phi_data/experiments/climatologies/temporal_climatology'
    date_times_by_date = get_date_times(use_bottom_hours = use_bottom_hours)
    
    for hazard in hazards:
        for climo_type in climo_types:
            iterator = md.to_iterator(date_times_by_date, [hazard], [climo_type])
            results = md.run_parallel(aggregate_events_day_by_time, iterator, nprocs_to_use = 20,\
                                      description = 'Aggregating %s %s by time' %(hazard, climo_type))
            
            all_times = np.zeros(results[0].shape)
            all_grid_points = np.zeros(results[0].shape)
            for day in results:
                all_times += day
                all_grid_points[day >= 0] += 90000
            
            events_save_file = '%s_%s_temporal_climo.npy' %(hazard, climo_type)
            sample_size_file = '%s_%s_temporal_sample_size.npy' %(hazard, climo_type)
            grid_points_file = '%s_%s_temporal_climo_gridpoints.npy' %(hazard, climo_type)
            utilities.save_data(save_dir, save_file, all_times, 'npy')
            utilities.save_data(save_dir, grid_points_file, all_grid_points, 'npy')
            utilities.save_data(save_dir, sample_size_file, all_sample_sizes, 'npy')
    
    return

def get_temporal_prob_climos(use_bottom_hours = False):
    
    hazards = ['hail', 'wind', 'tornado']
    climo_types = ['lsrs', 'warnings']
    leads = [30, 60, 90, 120, 150, 180]
    save_dir = '/work/ryan.martz/wofs_phi_data/experiments/climatologies/temporal_climatology'
    date_times_by_date = get_date_times(use_bottom_hours = use_bottom_hours)
    
    for hazard in hazards:
        for climo_type in climo_types:
            for lead in leads:
                
                iterator = md.to_iterator(date_times_by_date, [hazard], [climo_type], [lead])
                results = md.run_parallel(aggregate_probs_day_by_time, iterator, nprocs_to_use = 20,\
                                       description = 'Aggregating %s %s probs by time' %(hazard, climo_type))

                all_times = np.zeros(results[0].shape)
                all_grid_points = np.zeros(results[0].shape)
                for day in results:
                    all_times += day
                    all_grid_points[day >= 0] += 90000
                
                if use_bottom_hours:
                    save_file = '%s_%s_probs_temporal_climo_%s-%smin.npy'\
                    %(hazard, climo_type, lead-30, (lead-30)+60)
                    grid_points_file = '%s_%s_temporal_climo_%s-%smin_gridpoints.npy'\
                    %(hazard, climo_type, lead-30, (lead-30)+60)
                    
                else:
                    save_file = '%s_%s_probs_temporal_climo_%s-%smin_top_hour_inits_only.npy'\
                    %(hazard, climo_type, lead-30, (lead-30)+60)
                    grid_points_file = '%s_%s_temporal_climo_%s-%smin_gridpoints_top_hour_inits_only.npy'\
                    %(hazard, climo_type, lead-30, (lead-30)+60)
                
                utilities.save_data(save_dir, save_file, all_times, 'npy')
                utilities.save_data(save_dir, grid_points_file, all_grid_points, 'npy')
    
    return

def get_all_probs(use_bottom_hours = False):
    
    hazards = ['hail', 'wind', 'tornado']
    climo_types = ['lsrs', 'warnings']
    leads = [30, 60, 90, 120, 150, 180]
    save_dir = '/work/ryan.martz/wofs_phi_data/experiments/climatologies/all_probs'
    date_times_by_date = get_date_times(use_bottom_hours = use_bottom_hours)
    
    for hazard in hazards:
        for climo_type in climo_types:
            for lead in leads:
                
                all_probs = []
                iterator = md.to_iterator(date_times_by_date, [hazard], [climo_type], [lead])
                results = md.run_parallel(aggregate_probs_by_day, iterator, nprocs_to_use = 20,\
                                       description = 'Aggregating %s %s probs by time' %(hazard, climo_type))
                
                for day_probs in results:
                    all_probs.extend(day_probs)
                all_probs = np.array(all_probs)
                
                if use_bottom_hours:
                    save_file = '%s_%s_all_probs_%s-%smin.npy'\
                    %(hazard, climo_type, lead-30, (lead-30)+60)
                    
                else:
                    save_file = '%s_%s_all_probs_%s-%smin_top_hour_inits_only.npy'\
                    %(hazard, climo_type, lead-30, (lead-30)+60)
                
                utilities.save_data(save_dir, save_file, all_probs, 'npy')
                    
    
    return

def main():
    
    #get_wofs_spatial_climo()
    #get_events_spatial_climos()
    #get_temporal_event_climos()
    #get_temporal_prob_climos()
    #get_all_probs()
    get_probs_spatial_climos()
    
    return


if (__name__ == '__main__'):

    main()