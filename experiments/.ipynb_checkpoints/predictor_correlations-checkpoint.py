import numpy as np
import math
import os
import multiprocessing as mp
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
import datetime as dt
from shutil import copy
import sys
sys.path.append('../wofs_phi')
from wofs_phi import utilities
from wofs_phi import multiprocessing_driver as md
from . import get_ti_csv as ti_funcs
from . import performance_stats as p_stats

def load_times(date, start_times):
    
    dt_list = [0]*len(start_times)
    for i in range(len(start_times)):
        time = start_times[i]
        date_time_str = str(int(date)) + time
        date_time = dt.datetime.strptime(date_time_str, '%Y%m%d%H%M')
        dt_list[i] = date_time
    
    return np.array(dt_list)

def get_date_times(use_bottom_hours = True):
    
    dates = np.genfromtxt('/home/ryan.martz/python_packages/'\
                          'frdd-wofs-phi/wofs_phi/probSevere_dates.txt')
    if use_bottom_hours:
        start_times = ["1700", "1730", "1800", "1830", "1900", "1930", "2000",\
                       "2030", "2100", "2130", "2200", "2230", "2300", "2330",\
                       "0000", "0030", "0100", "0130", "0200", "0230", "0300",\
                       "0330", "0400", "0430", "0500"]
    else:
        start_times = ["1700", "1800", "1900", "2000", "2100", "2200", "2300",\
                      "0000", "0100", "0200", "0300", "0400", "0500"]
    
    date_times = []
    
    for date in dates:
        date_times.extend(load_times(date, start_times))
    
    return date_times

def load_lsrs(hazard, date, start_time, end_time):
    
    directory = '/work/eric.loken/wofs/2024_update/SFE2024/obs/full_npy'
    if hazard == 'tornado':
        file = '%s_reps1d_%s_v%s-%s_r39km.npy' %(hazard, date, start_time, end_time)
    elif hazard == 'hail':
        file = '%s_reps1d_%s_v%s-%s_r39km_20_min_buffer.npy' %(hazard, date, start_time, end_time)
    elif hazard == 'wind':
        file = '%s_reps1d_%s_v%s-%s_r375km_20_min_buffer.npy' %(hazard, date, start_time, end_time)
    
    lsrs_1d = np.load('%s/%s' %(directory, file))
    
    return lsrs_1d.reshape((90000,))

def load_warnings(hazard, date, start_time, end_time):
    
    directory = '/work/ryan.martz/wofs_phi_data/training_data/warnings/'\
    'full_1d_warnings/length_60/%s' %(hazard)
    file = '%s_warnings_%s_v%s-%s_1d.npy' %(hazard, date, start_time, end_time)
    
    warnings_1d = np.load('%s/%s' %(directory, file))
    
    return warnings_1d.reshape((90000,))

def load_predictors(date, start, lead, pred_indices):
    
    directory = '/work/eric.loken/wofs/2024_update/SFE2024/fcst/full_npy'
    
    start_time = start.strftime('%H%M')
    end_time = (start + dt.timedelta(minutes = 60)).strftime('%H%M')
    
    init_time = (start - dt.timedelta(minutes = lead)).strftime('%H%M')
    ps_init_time = ((start - dt.timedelta(minutes = lead))\
                    + dt.timedelta(minutes = 24)).strftime('%H%M')
    
    file = 'wofs1d_psv3_with_torp_%s_%s_%s_v%s-%s.npy'\
    %(date, init_time, ps_init_time, start_time, end_time)
    
    preds = np.load('%s/%s' %(directory, file))
    return_preds = preds[:,pred_indices]
    
    return return_preds

def load_lsr_probs(date, start, lead, hazard):
    
    start_time = start.strftime('%H%M')
    end_time = (start + dt.timedelta(minutes = 60)).strftime('%H%M')
    
    init_time = (start - dt.timedelta(minutes = lead)).strftime('%H%M')
    
    directory = '/work/ryan.martz/wofs_phi_data/obs_train/test_fcsts/wofs_psv3_with_torp/'\
    '%s/wofslag_25/length_60/%s/%s/' %(hazard, date, init_time)
    
    if hazard == 'tornado' or hazard == 'hail':
        file = 'wofs_psv3_with_torp_obs_trained_rf_%s_raw_probs_%s_init%s_v%s-%s_r39km.txt'\
        %(hazard, date, init_time, start_time, end_time)
    else:
        file = 'wofs_psv3_with_torp_obs_trained_rf_%s_raw_probs_%s_init%s_v%s-%s_r375km.txt'\
        %(hazard, date, init_time, start_time, end_time)
    
    lsr_probs = np.genfromtxt('%s/%s' %(directory, file))
    lsr_probs = lsr_probs.reshape((90000,))
    
    return lsr_probs

def load_warning_probs(date, start, lead, hazard):
    
    start_time = start.strftime('%H%M')
    end_time = (start + dt.timedelta(minutes = 60)).strftime('%H%M')
    
    init_time = (start - dt.timedelta(minutes = lead)).strftime('%H%M')
    
    directory = '/work/ryan.martz/wofs_phi_data/warnings_train/test_fcsts/wofs_psv3_with_torp/'\
    '%s/wofslag_25/length_60/%s/%s/' %(hazard, date, init_time)
    
    file = 'wofs_psv3_with_torp_warnings_trained_rf_%s_raw_probs_%s_init%s_v%s-%s.txt'\
    %(hazard, date, init_time, start_time, end_time)
    
    warning_probs = np.genfromtxt('%s/%s' %(directory, file))
    warning_probs = warning_probs.reshape((90000,))
    
    return warning_probs

def load_lsr_warning_predictor_predictions(hazard, date_time, lead, pred_indices):
    
    date_str = date_time.strftime('%Y%m%d')
    
    start_time_str = date_time.strftime('%H%M')
    end_time_str = (date_time + dt.timedelta(minutes = 60)).strftime('%H%M')
    
    lsrs = load_lsrs(hazard, date_str, start_time_str, end_time_str)
    warnings = load_warnings(hazard, date_str, start_time_str, end_time_str)
    preds = load_predictors(date_str, date_time, lead, pred_indices)
    lsr_probs = load_lsr_probs(date_str, date_time, lead, hazard)
    warning_probs = load_warning_probs(date_str, date_time, lead, hazard)
        
    return lsrs, warnings, preds, lsr_probs, warning_probs

def get_abs_correlation(a, b):
    
    cc_arr = np.corrcoef(a, b)
    cc = cc_arr[0, 1]
    abs_cc = np.abs(cc)
    
    return abs_cc

def load_single_correlation_dataset(date_times, hazard, lead, pred_indices, plvs):
    
    dataset_lsrs = []
    dataset_warnings = []
    dataset_lsr_probs = []
    dataset_warning_probs = []
    first = True
    
    lsrs_save_dir = '/work/ryan.martz/wofs_phi_data/experiments/wofs_psv3_with_torp_trained/'\
    'obs/length_60/correlations/%s' %(hazard)
    warnings_save_dir = '/work/ryan.martz/wofs_phi_data/experiments/wofs_psv3_with_torp_trained/'\
    'warnings/length_60/correlations/%s' %(hazard)
    
    if hazard == 'hail' or hazard == 'tornado':
        lsrs_save_file = 'all_%s_obs_r39km_%s-%smin.npy' %(hazard, lead, lead+60)
        lsr_probs_save_file = 'wofs_psv3_with_torp_%s_obs_r39km_trained_probs_%s-%smin.npy'\
        %(hazard, lead, lead+60)
    else:
        lsrs_save_file = 'all_%s_obs_r375km_%s-%smin.npy' %(hazard, lead, lead+60)
        lsr_probs_save_file = 'wofs_psv3_with_torp_%s_obs_r375km_trained_probs_%s-%smin.npy'\
        %(hazard, lead, lead+60)
    
    preds_save_file = 'all_selected_preds_%s-%smin.npy' %(lead, lead+60)
    
    warnings_save_file = 'all_%s_warnings_%s-%smin.npy' %(hazard, lead, lead+60)
    warning_probs_save_file = 'wofs_psv3_with_torp_%s_warnings_trained_probs_%s-%smin.npy'\
    %(hazard, lead, lead+60)
    
    try:
        dataset_lsrs = np.load('%s/%s' %(lsrs_save_dir, lsrs_save_file))
        dataset_lsr_probs = np.load('%s/%s' %(lsrs_save_dir, lsr_probs_save_file))
        dataset_preds = np.load('%s/%s' %(lsrs_save_dir, preds_save_file))
        dataset_warnings = np.load('%s/%s' %(warnings_save_dir, warnings_save_file))
        dataset_warning_probs = np.load('%s/%s' %(warnings_save_dir, warning_probs_save_file))
    except:
        print('re-loading lsrs/warnings/preds/predictions')
        for date_time in date_times:
            try:
                lsrs, warnings, preds, lsr_probs, warning_probs =\
                load_lsr_warning_predictor_predictions(hazard, date_time,\
                                                       lead, pred_indices)
            except:
                continue

            if first:
                dataset_preds = preds
                first = False
            else:
                dataset_preds = np.append(dataset_preds, preds, axis = 0)

            dataset_lsrs.extend(lsrs)
            dataset_warnings.extend(warnings)
            dataset_lsr_probs.extend(lsr_probs)
            dataset_warning_probs.extend(warning_probs)

        dataset_lsrs = np.array(dataset_lsrs)
        dataset_lsr_probs = np.array(dataset_lsr_probs)
        utilities.save_data(lsrs_save_dir, lsrs_save_file, dataset_lsrs, 'npy')
        utilities.save_data(lsrs_save_dir, preds_save_file, dataset_preds, 'npy')
        utilities.save_data(lsrs_save_dir, lsr_probs_save_file, dataset_lsr_probs, 'npy')

        dataset_warnings = np.array(dataset_warnings)
        dataset_warning_probs = np.array(dataset_warning_probs)
        utilities.save_data(warnings_save_dir, warnings_save_file, dataset_warnings, 'npy')
        utilities.save_data(warnings_save_dir, preds_save_file, dataset_preds, 'npy')
        utilities.save_data(warnings_save_dir, warning_probs_save_file, dataset_warning_probs, 'npy')
    
    lsr_cc_list = []
    warning_cc_list = []
    lsr_probs_cc_list = []
    warning_probs_cc_list = []
    
    plv_order = []
    for i in range(len(pred_indices)):
        plv_order.append(plvs[i])
        preds = dataset_preds[:,i]
        
        lsr_cc_list.append(get_abs_correlation(preds, dataset_lsrs))
        warning_cc_list.append(get_abs_correlation(preds, dataset_warnings))
        lsr_probs_cc_list.append(get_abs_correlation(preds, dataset_lsr_probs))
        warning_probs_cc_list.append(get_abs_correlation(preds, dataset_warning_probs))
    
    lsr_cc_df = pd.DataFrame({'Predictor': plv_order, 'Correlation': lsr_cc_list})
    warning_cc_df = pd.DataFrame({'Predictor': plv_order, 'Correlation': warning_cc_list})
    lsr_probs_cc_df = pd.DataFrame({'Predictor': plv_order, 'Correlation': lsr_probs_cc_list})
    warning_probs_cc_df = pd.DataFrame({'Predictor': plv_order, 'Correlation': warning_probs_cc_list})
    
    lsr_correlation_file = 'selected_predictor_%s_lsr_correlations_%s-%smin.csv'\
    %(hazard, lead, lead+60)
    lsr_probs_correlation_file = 'selected_predictor_%s_lsr_probs_correlations_%s-%smin.csv'\
    %(hazard, lead, lead+60)
    
    warning_correlation_file = 'selected_predictor_%s_warning_correlations_%s-%smin.csv'\
    %(hazard, lead, lead+60)
    warning_probs_correlation_file = 'selected_predictor_%s_warning_probs_correlations_%s-%smin.csv'\
    %(hazard, lead, lead+60)
    
    utilities.save_data(lsrs_save_dir, lsr_correlation_file, lsr_cc_df, 'csv')
    utilities.save_data(lsrs_save_dir, lsr_probs_correlation_file, lsr_probs_cc_df, 'csv')
    
    utilities.save_data(warnings_save_dir, warning_correlation_file, warning_cc_df, 'csv')
    utilities.save_data(warnings_save_dir, warning_probs_correlation_file, warning_probs_cc_df, 'csv')
    
    return

def map_plvs(plvs):
    
    for i in range(len(plvs)):
        plv = plvs[i]
        
        if 'Hail PS Smooth' in plv:
            plv = 'Hail PS Prob'
        elif 'Wind PS Smooth' in plv:
            plv = 'Wind PS Prob'
        elif 'Tornado PS Smooth' in plv:
            plv = 'Tor PS Prob'
        elif 'Lightning Flash Extent' in plv:
            plv = 'Lightning FED'
        elif '0-2km UH' in plv:
            plv = '0-2km UH'
        elif 'Vertical Velocity' in plv:
            plv = 'Updraft Speed'
        elif 'TORP Probability' in plv:
            plv = 'TORP Prob'
        elif 'TORP Max Reflectivity' in plv:
            plv = 'TORP dBZ'
        elif 'TORP Max Abs Velocity' in plv:
            plv = 'TORP Velocity'
        
        plvs[i] = plv
    
    return plvs

def save_correlation_data(hazards, leads):
    
    date_times = get_date_times()
    
    var_dict = pd.read_csv('/work/ryan.martz/wofs_phi_data/experiments/vars_dict.csv')
    all_plvs = np.array(var_dict.Plain_Language_Variable)
    
    wofs_var_indices = np.array([25, 26, 27, 120, 123, 124, 125, 127, 264, 265, 268])
    ps_var_indices = np.array([155, 161, 167])
    torp_var_indices = np.array([271, 311, 326])
    
    wofs_ps_indices = np.append(wofs_var_indices, ps_var_indices)
    var_indices = np.append(wofs_ps_indices, torp_var_indices)
    plvs = all_plvs[var_indices]
    
    plvs = map_plvs(plvs)
    
    iterator = md.to_iterator([date_times], hazards, leads, [var_indices], [plvs])
    results = md.run_parallel(load_single_correlation_dataset, iterator, nprocs_to_use = 18,\
                              description = 'Saving Correlation Data')
    
    return

def main():
    
    hazards = ['hail', 'wind', 'tornado']
    leads = [30, 60, 90, 120, 150, 180]
    
    save_correlation_data(hazards, leads)
    
    return

if (__name__ == '__main__'):

    main()