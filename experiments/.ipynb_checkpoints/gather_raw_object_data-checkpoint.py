import numpy as np
import sys
sys.path.append('../wofs_phi')
from wofs_phi import config as c
from wofs_phi import utilities
from sklearn.metrics import brier_score_loss as BS
import os
from shutil import copy
import matplotlib.pyplot as plt
import math
import pandas as pd
import matplotlib.ticker as ticker
import datetime as dt

def get_raw_object_skill():
    
    ps_versions = [2]
    leads = [30, 60, 90, 120, 150, 180]
    lengths = [60]
    
    print('gathering objects')
    
    for ps_version in ps_versions:
        for lead in leads:
            for length in lengths:
                print('psv%s, window: %s-%smin' %(ps_version, lead, lead+length))

                all_torp_probs = []
                all_torp_reflectivity = []
                all_ps2_tor_probs = []
                all_ps2_wind_probs = []
                all_ps2_hail_probs = []
                all_ps3_tor_probs = []
                all_ps3_wind_probs = []
                all_ps3_hail_probs = []

                all_tor_15_obs_and_warnings = []
                all_tor_39_obs_and_warnings = []
                all_wind_15_obs_and_warnings = []
                all_wind_39_obs_and_warnings = []
                all_hail_15_obs_and_warnings = []
                all_hail_39_obs_and_warnings = []

                all_tor_15_obs = []
                all_tor_39_obs = []
                all_wind_15_obs = []
                all_wind_39_obs = []
                all_hail_15_obs = []
                all_hail_39_obs = []

                for file in os.listdir('/work/eric.loken/wofs/2024_update/SFE2024/fcst/full_npy_backup'):
                    search_tag = 'psv%s_with_torp' %(ps_version)
                    if (search_tag in file) and (not ('filtered' in file)):
                        date = file.split('_')[4]
                        init = file.split('_')[5]
                        valid = file.split('_')[7].split('.')[0]
                        start = valid[1:5]
                        end = valid[6:]

                        init_dt = dt.datetime(1970,1,1,int(init[0:2]), int(init[2:]))
                        if init_dt.hour < 12:
                            init_dt += dt.timedelta(days = 1)
                        start_dt = dt.datetime(1970,1,1,int(start[0:2]), int(start[2:]))
                        if start_dt.hour < 12:
                            start_dt += dt.timedelta(days = 1)
                        end_dt = dt.datetime(1970,1,1,int(end[0:2]), int(end[2:]))
                        if end_dt.hour < 12:
                            end_dt += dt.timedelta(days = 1)

                        dur = int((end_dt - start_dt).seconds/60)
                        true_lead = int((start_dt - init_dt).seconds/60)

                        last_updated = os.path.getmtime('/work/eric.loken/wofs/2024_update/SFE2024/fcst/full_npy_backup/%s' %(file))
                        year_updated = dt.datetime.fromtimestamp(last_updated).year

                        if (not dur == length) or (not true_lead == lead):# or (not year_updated == 2025):
                            continue

                        try:
                            full_file_ps2 = np.load('/work/eric.loken/wofs/2024_update/SFE2024/fcst/full_npy_backup/%s'\
                                                    %(file))
                            full_file_ps3 = np.load('/work/eric.loken/wofs/2024_update/SFE2024/fcst/full_npy_backup/%s'\
                                                    %(file.replace('psv2', 'psv3')))
                        except:
                            continue

                        torp_probs = full_file_ps3[:,269]
                        torp_reflectivity = full_file_ps3[:,309]
                        tor_ps2_probs = full_file_ps2[:,54]
                        wind_ps2_probs = full_file_ps2[:,60]
                        hail_ps2_probs = full_file_ps2[:,66]
                        tor_ps3_probs = full_file_ps3[:,54]
                        wind_ps3_probs = full_file_ps3[:,60]
                        hail_ps3_probs = full_file_ps3[:,66]

                        obs_and_warnings_tor_dir = '/work/ryan.martz/wofs_phi_data/training_data/obs_and_warnings/full_1d_obs_and_warnings/length_60/tornado'
                        obs_and_warnings_tor_15_file = 'tornado_obs_and_warnings_%s_%s_r15km_1d.npy' %(date, valid)
                        obs_and_warnings_tor_39_file = 'tornado_obs_and_warnings_%s_%s_r39km_1d.npy' %(date, valid)
                        obs_and_warnings_wind_dir = '/work/ryan.martz/wofs_phi_data/training_data/obs_and_warnings/full_1d_obs_and_warnings/length_60/wind'
                        obs_and_warnings_wind_15_file = 'wind_obs_and_warnings_%s_%s_r15km_1d.npy' %(date, valid)
                        obs_and_warnings_wind_39_file = 'wind_obs_and_warnings_%s_%s_r39km_1d.npy' %(date, valid)
                        obs_and_warnings_hail_dir = '/work/ryan.martz/wofs_phi_data/training_data/obs_and_warnings/full_1d_obs_and_warnings/length_60/hail'
                        obs_and_warnings_hail_15_file = 'hail_obs_and_warnings_%s_%s_r15km_1d.npy' %(date, valid)
                        obs_and_warnings_hail_39_file = 'hail_obs_and_warnings_%s_%s_r39km_1d.npy' %(date, valid)

                        obs_dir = '/work/eric.loken/wofs/2024_update/SFE2024/obs/full_npy'
                        obs_tor_15_file = 'tornado_reps1d_%s_%s_r15km.npy' %(date, valid)
                        obs_wind_15_file = 'wind_reps1d_%s_%s_r15km.npy' %(date, valid)
                        obs_hail_15_file = 'hail_reps1d_%s_%s_r15km.npy' %(date, valid)
                        obs_tor_39_file = 'tornado_reps1d_%s_%s_r39km.npy' %(date, valid)
                        obs_wind_39_file = 'wind_reps1d_%s_%s_r39km.npy' %(date, valid)
                        obs_hail_39_file = 'hail_reps1d_%s_%s_r39km.npy' %(date, valid)

                        try:
                            tor_15_obs_and_warnings = np.load('%s/%s' %(obs_and_warnings_tor_dir, obs_and_warnings_tor_15_file))
                            tor_39_obs_and_warnings = np.load('%s/%s' %(obs_and_warnings_tor_dir, obs_and_warnings_tor_39_file))
                            wind_15_obs_and_warnings = np.load('%s/%s' %(obs_and_warnings_wind_dir, obs_and_warnings_wind_15_file))
                            wind_39_obs_and_warnings = np.load('%s/%s' %(obs_and_warnings_wind_dir, obs_and_warnings_wind_39_file))
                            hail_15_obs_and_warnings = np.load('%s/%s' %(obs_and_warnings_hail_dir, obs_and_warnings_hail_15_file))
                            hail_39_obs_and_warnings = np.load('%s/%s' %(obs_and_warnings_hail_dir, obs_and_warnings_hail_39_file))

                            tor_15_obs = np.load('%s/%s' %(obs_dir, obs_tor_15_file))
                            tor_39_obs = np.load('%s/%s' %(obs_dir, obs_tor_39_file))
                            wind_15_obs = np.load('%s/%s' %(obs_dir, obs_wind_15_file))
                            wind_39_obs = np.load('%s/%s' %(obs_dir, obs_wind_39_file))
                            hail_15_obs = np.load('%s/%s' %(obs_dir, obs_hail_15_file))
                            hail_39_obs = np.load('%s/%s' %(obs_dir, obs_hail_39_file))
                        except:
                            continue

                        all_torp_probs.extend(torp_probs)
                        all_torp_reflectivity.extend(torp_reflectivity)
                        all_ps2_tor_probs.extend(tor_ps2_probs)
                        all_ps2_wind_probs.extend(wind_ps2_probs)
                        all_ps2_hail_probs.extend(hail_ps2_probs)
                        all_ps3_tor_probs.extend(tor_ps3_probs)
                        all_ps3_wind_probs.extend(wind_ps3_probs)
                        all_ps3_hail_probs.extend(hail_ps3_probs)

                        all_tor_15_obs_and_warnings.extend(tor_15_obs_and_warnings.reshape(90000,))
                        all_tor_39_obs_and_warnings.extend(tor_39_obs_and_warnings.reshape(90000,))
                        all_wind_15_obs_and_warnings.extend(wind_15_obs_and_warnings.reshape(90000,))
                        all_wind_39_obs_and_warnings.extend(wind_39_obs_and_warnings.reshape(90000,))
                        all_hail_15_obs_and_warnings.extend(hail_15_obs_and_warnings.reshape(90000,))
                        all_hail_39_obs_and_warnings.extend(hail_39_obs_and_warnings.reshape(90000,))

                        all_tor_15_obs.extend(tor_15_obs.reshape(90000,))
                        all_tor_39_obs.extend(tor_39_obs.reshape(90000,))
                        all_wind_15_obs.extend(wind_15_obs.reshape(90000,))
                        all_wind_39_obs.extend(wind_39_obs.reshape(90000,))
                        all_hail_15_obs.extend(hail_15_obs.reshape(90000,))
                        all_hail_39_obs.extend(hail_39_obs.reshape(90000,))

                all_tor_15_obs_and_warnings = np.array(all_tor_15_obs_and_warnings)
                all_tor_39_obs_and_warnings = np.array(all_tor_39_obs_and_warnings)
                all_wind_15_obs_and_warnings = np.array(all_wind_15_obs_and_warnings)
                all_wind_39_obs_and_warnings = np.array(all_wind_39_obs_and_warnings)
                all_hail_15_obs_and_warnings = np.array(all_hail_15_obs_and_warnings)
                all_hail_39_obs_and_warnings = np.array(all_hail_39_obs_and_warnings)

                all_tor_15_obs = np.array(all_tor_15_obs)
                all_tor_39_obs = np.array(all_tor_39_obs)
                all_wind_15_obs = np.array(all_wind_15_obs)
                all_wind_39_obs = np.array(all_wind_39_obs)
                all_hail_15_obs = np.array(all_hail_15_obs)
                all_hail_39_obs = np.array(all_hail_39_obs)

                all_ps2_tor_probs = np.array(all_ps2_tor_probs)
                all_ps2_wind_probs = np.array(all_ps2_wind_probs)
                all_ps2_hail_probs = np.array(all_ps2_hail_probs)
                all_torp_probs = np.array(all_torp_probs)
                all_torp_reflectivity = np.array(all_torp_reflectivity)
                all_ps3_tor_probs = np.array(all_ps3_tor_probs)
                all_ps3_wind_probs = np.array(all_ps3_wind_probs)
                all_ps3_hail_probs = np.array(all_ps3_hail_probs)

                object_save_dir = '/work/ryan.martz/wofs_phi_data/experiments/torp_ps_raw_evals/object_probs'
                tor_ps2_file = 'all_ps2_tornado_raw_object_probs_%s-%smin.npy' %(lead, lead+length)
                wind_ps2_file = 'all_ps2_wind_raw_object_probs_%s-%smin.npy' %(lead, lead+length)
                hail_ps2_file = 'all_ps2_hail_raw_object_probs_%s-%smin.npy' %(lead, lead+length)
                tor_ps3_file = 'all_ps3_tornado_raw_object_probs_%s-%smin.npy' %(lead, lead+length)
                wind_ps3_file = 'all_ps3_wind_raw_object_probs_%s-%smin.npy' %(lead, lead+length)
                hail_ps3_file = 'all_ps3_hail_raw_object_probs_%s-%smin.npy' %(lead, lead+length)
                torp_file = 'all_torp_raw_object_probs_%s-%smin.npy' %(lead, lead+length)
                torp_ref_file = 'all_torp_raw_object_reflectivities_%s-%smin.npy' %(lead, lead+length)

                obs_save_dir = '/work/ryan.martz/wofs_phi_data/experiments/torp_ps_raw_evals/obs'
                tor_15_obs_and_warnings_file = 'all_tornado_obs_and_warnings_r15km_%s-%smin.npy' %(lead, lead+length)
                tor_39_obs_and_warnings_file = 'all_tornado_obs_and_warnings_r39km_%s-%smin.npy' %(lead, lead+length)
                wind_15_obs_and_warnings_file = 'all_wind_obs_and_warnings_r15km_%s-%smin.npy' %(lead, lead+length)
                wind_39_obs_and_warnings_file = 'all_wind_obs_and_warnings_r39km_%s-%smin.npy' %(lead, lead+length)
                hail_15_obs_and_warnings_file = 'all_hail_obs_and_warnings_r15km_%s-%smin.npy' %(lead, lead+length)
                hail_39_obs_and_warnings_file = 'all_hail_obs_and_warnings_r39km_%s-%smin.npy' %(lead, lead+length)

                tor_15_obs_file = 'all_tornado_obs_r15km_%s-%smin.npy' %(lead, lead+length)
                tor_39_obs_file = 'all_tornado_obs_r39km_%s-%smin.npy' %(lead, lead+length)
                wind_15_obs_file = 'all_wind_obs_r15km_%s-%smin.npy' %(lead, lead+length)
                wind_39_obs_file = 'all_wind_obs_r39km_%s-%smin.npy' %(lead, lead+length)
                hail_15_obs_file = 'all_hail_obs_r15km_%s-%smin.npy' %(lead, lead+length)
                hail_39_obs_file = 'all_hail_obs_r39km_%s-%smin.npy' %(lead, lead+length)

                utilities.save_data(object_save_dir, tor_ps2_file, all_ps2_tor_probs, 'npy')
                utilities.save_data(object_save_dir, wind_ps2_file, all_ps2_wind_probs, 'npy')
                utilities.save_data(object_save_dir, hail_ps2_file, all_ps2_hail_probs, 'npy')
                utilities.save_data(object_save_dir, tor_ps3_file, all_ps3_tor_probs, 'npy')
                utilities.save_data(object_save_dir, wind_ps3_file, all_ps3_wind_probs, 'npy')
                utilities.save_data(object_save_dir, hail_ps3_file, all_ps3_hail_probs, 'npy')
                utilities.save_data(object_save_dir, torp_file, all_torp_probs, 'npy')
                utilities.save_data(object_save_dir, torp_ref_file, all_torp_reflectivity, 'npy')

                utilities.save_data(obs_save_dir, tor_15_obs_and_warnings_file, all_tor_15_obs_and_warnings, 'npy')
                utilities.save_data(obs_save_dir, tor_39_obs_and_warnings_file, all_tor_39_obs_and_warnings, 'npy')
                utilities.save_data(obs_save_dir, wind_15_obs_and_warnings_file, all_wind_15_obs_and_warnings, 'npy')
                utilities.save_data(obs_save_dir, wind_39_obs_and_warnings_file, all_wind_39_obs_and_warnings, 'npy')
                utilities.save_data(obs_save_dir, hail_15_obs_and_warnings_file, all_hail_15_obs_and_warnings, 'npy')
                utilities.save_data(obs_save_dir, hail_39_obs_and_warnings_file, all_hail_39_obs_and_warnings, 'npy')

                utilities.save_data(obs_save_dir, tor_15_obs_file, all_tor_15_obs, 'npy')
                utilities.save_data(obs_save_dir, tor_39_obs_file, all_tor_39_obs, 'npy')
                utilities.save_data(obs_save_dir, wind_15_obs_file, all_wind_15_obs, 'npy')
                utilities.save_data(obs_save_dir, wind_39_obs_file, all_wind_39_obs, 'npy')
                utilities.save_data(obs_save_dir, hail_15_obs_file, all_hail_15_obs, 'npy')
                utilities.save_data(obs_save_dir, hail_39_obs_file, all_hail_39_obs, 'npy')
    
    return
    
    hazards = ['hail', 'wind', 'tornado']
    radii = [15, 39]
    ver_types = ['obs', 'obs_and_warnings']
    
    object_dir = '/work/ryan.martz/wofs_phi_data/experiments/torp_ps_raw_evals/object_probs'
    obs_dir = '/work/ryan.martz/wofs_phi_data/experiments/torp_ps_raw_evals/obs'
    metrics_dir = '/work/ryan.martz/wofs_phi_data/experiments/torp_ps_raw_evals/metrics'
    
    print('getting metrics')
    
    for lead in leads:
        for length in lengths:
            for hazard in hazards:
                for radius in radii:
                    for ver_type in ver_types:
                        obs_file = 'all_%s_%s_r%skm_%s-%smin.npy' %(hazard, ver_type, radius, lead, lead+length)
                        all_obs = np.load('%s/%s' %(obs_dir, obs_file))
                        climo = np.zeros(all_obs.shape)
                        climo = climo + np.mean(all_obs)
                        climo_bs = np.sum(np.power(climo - all_obs, 2))

                        ps2_probs_file = 'all_ps2_%s_raw_object_probs_%s-%smin.npy' %(hazard, lead, lead+length)
                        ps3_probs_file = 'all_ps3_%s_raw_object_probs_%s-%smin.npy' %(hazard, lead, lead+length)
                        torp_probs_file = 'all_torp_raw_object_probs_%s-%smin.npy' %(lead, lead+length)

                        all_ps2_probs = np.load('%s/%s' %(object_dir, ps2_probs_file))
                        all_ps3_probs = np.load('%s/%s' %(object_dir, ps3_probs_file))
                        all_torp_probs = np.load('%s/%s' %(object_dir, torp_probs_file))

                        torp_bss_arr = []
                        ps2_bss_arr = []
                        ps3_bss_arr = []

                        filter_levels = np.arange(0,1.01,0.05)

                        for p in filter_levels:

                            use_torp_probs = all_torp_probs
                            use_torp_probs[all_torp_probs <= p] = 0

                            use_ps2_probs = all_ps2_probs
                            use_ps2_probs[all_ps2_probs <= p] = 0

                            use_ps3_probs = all_ps3_probs
                            use_ps3_probs[all_ps3_probs <= p] = 0

                            torp_bs = np.sum(np.power(use_torp_probs - all_obs, 2))
                            ps2_bs = np.sum(np.power(use_ps2_probs - all_obs, 2))
                            ps3_bs = np.sum(np.power(use_ps3_probs - all_obs, 2))

                            torp_bss_arr.append((torp_bs - climo_bs)/(-climo_bs))
                            ps2_bss_arr.append((ps2_bs - climo_bs)/(-climo_bs))
                            ps3_bss_arr.append((ps3_bs - climo_bs)/(-climo_bs))

                        torp_bss_file = 'torp_bss_%s_%s_r%skm_%s-%smin.npy' %(hazard, ver_type, radius, lead, lead+length)
                        ps2_bss_file = 'ps2_bss_%s_%s_r%skm_%s-%smin.npy' %(hazard, ver_type, radius, lead, lead+length)
                        ps3_bss_file = 'ps3_bss_%s_%s_r%skm_%s-%smin.npy' %(hazard, ver_type, radius, lead, lead+length)
                        utilities.save_data(metrics_dir, torp_bss_file, torp_bss_arr, 'npy')
                        utilities.save_data(metrics_dir, ps2_bss_file, ps2_bss_arr, 'npy')
                        utilities.save_data(metrics_dir, ps3_bss_file, ps3_bss_arr, 'npy')
    
    return

def get_torp_ps_correlations():
    
    ps_versions = [2]
    leads = [30, 60, 90, 120, 150, 180]
    lengths = [60]

    for ps_version in ps_versions:
        for lead in leads:
            for length in lengths:
                print('psv%s, window: %s-%smin' %(ps_version, lead, lead+length))

                i = 0

                for file in os.listdir('/work/eric.loken/wofs/2024_update/SFE2024/fcst/full_npy_backup'):
                    search_tag = 'psv%s_with_torp' %(ps_version)
                    if (search_tag in file) and (not ('filtered' in file)):
                        date = file.split('_')[4]
                        init = file.split('_')[5]
                        valid = file.split('_')[7].split('.')[0]
                        start = valid[1:5]
                        end = valid[6:]
                        
                        init_dt = dt.datetime(1970,1,1,int(init[0:2]), int(init[2:]))
                        if init_dt.hour < 12:
                            init_dt += dt.timedelta(days = 1)
                        start_dt = dt.datetime(1970,1,1,int(start[0:2]), int(start[2:]))
                        if start_dt.hour < 12:
                            start_dt += dt.timedelta(days = 1)
                        end_dt = dt.datetime(1970,1,1,int(end[0:2]), int(end[2:]))
                        if end_dt.hour < 12:
                            end_dt += dt.timedelta(days = 1)

                        dur = int((end_dt - start_dt).seconds/60)
                        true_lead = int((start_dt - init_dt).seconds/60)

                        last_updated = os.path.getmtime('/work/eric.loken/wofs/2024_update/SFE2024/fcst/full_npy_backup/%s' %(file))
                        year_updated = dt.datetime.fromtimestamp(last_updated).year

                        if (not dur == length) or (not true_lead == lead):# or (not year_updated == 2025):
                            continue

                        try:
                            full_file = np.load('/work/eric.loken/wofs/2024_update/SFE2024/fcst/full_npy_backup/%s' %(file))
                            if ps_version == 2:
                                if not file.replace('psv2', 'psv3')\
                                in os.listdir('/work/eric.loken/wofs/2024_update/SFE2024/fcst/full_npy_backup'):
                                    continue
                        except:
                            continue

                        torp_probs = full_file[:,269:274]
                        tor_ps_probs = full_file[:,np.array([54,59,102,107,150,155,198,203,246,251])]
                        if i == 0:
                            all_torp_probs = torp_probs
                            all_ps_tor_probs = tor_ps_probs
                            i = 1
                        else:
                            all_torp_probs = np.append(all_torp_probs, torp_probs, axis = 0)
                            all_ps_tor_probs = np.append(all_ps_tor_probs, tor_ps_probs, axis = 0)

                all_ps_tor_probs = np.array(all_ps_tor_probs)
                all_torp_probs = np.array(all_torp_probs)

                object_save_dir = '/work/ryan.martz/wofs_phi_data/experiments/torp_ps_raw_evals/object_probs'
                tor_ps_file = 'all_ps%s_tornado_probs_%s-%smin.npy' %(ps_version, lead, lead+length)
                torp_file = 'all_torp_probs_%s-%smin.npy' %(lead, lead+length)

                utilities.save_data(object_save_dir, tor_ps_file, all_ps_tor_probs, 'npy')
                utilities.save_data(object_save_dir, torp_file, all_torp_probs, 'npy')
    
    return

def main():
    
    get_raw_object_skill()
    #get_torp_ps_correlations()
    
    return

if (__name__ == '__main__'):

    main()