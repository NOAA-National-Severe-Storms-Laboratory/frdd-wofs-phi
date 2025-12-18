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

import skexplain
#import plotting_config
import itertools

def transfer_models(home_dir, model_type, hazard, wofs_lag, length, lead, train_type):
    
    if hazard == 'hail':
        train_r = '39'
        buffer_str = '_20_min_buffer'
    elif hazard == 'wind':
        train_r = '375'
        buffer_str = '_20_min_buffer'
    elif hazard == 'tornado':
        train_r = '39'
        buffer_str = ''
    
    ver_type = train_type
    train_dir = '/work/ryan.martz/wofs_phi_data/%s_train/models/%s' %(train_type, model_type)
    ver_r = train_r
    
    skills_dir, __, __, __, skills_fname = p_stats.get_bss_fname_dir(home_dir, hazard, train_r,\
                                                                     ver_r, length, train_type,\
                                                                     ver_type, model_type)
    
    try:
        skill_by_lead_time_fold = np.load('%s/%s' %(skills_dir, skills_fname))
    except:
        return
    
    if (length + lead > 240) or ((train_type == 'obs') and (length == 120)):
        return
    
    all_leads = np.array([30, 60, 90, 120, 150, 180])
    i = np.where(all_leads == lead)[0][0]
    ideal_fold = np.where(skill_by_lead_time_fold[i,:] == max(skill_by_lead_time_fold[i,:]))[0][0]
    
    curr_model_dir = '%s/%s/wofslag_%s/length_%s' %(train_dir, hazard, wofs_lag, length)
    paste_model_dir = '/work/ryan.martz/wofs_phi_data/experiments/'\
    '%s_trained/%s/length_%s/ale_curves/%s' %(model_type, train_type, length, hazard)
    
    if train_type == 'obs' or train_type == 'obs_and_warnings':
        model_fname = '%s_%s_trained_wofsphi_%s_%smin_window%s-%s_r%skm_fold%s.pkl'\
        %(model_type, train_type, hazard, length, lead, lead+length, train_r, ideal_fold)
    else:
        model_fname = '%s_%s_trained_wofsphi_%s_%smin_window%s-%s_fold%s.pkl'\
        %(model_type, train_type, hazard, length, lead, lead+length, ideal_fold)
    
    curr_model_fname = '%s/%s' %(curr_model_dir, model_fname)
    paste_model_fname = '%s/%s' %(paste_model_dir, model_fname)
    
    sub_dirs = paste_model_dir.split('/')
    full_dir = ''
    for i in range(len(sub_dirs)):
        sub_dir = sub_dirs[i]
        full_dir = full_dir + '/' + sub_dir
        if not os.path.exists(full_dir):
            try:
                os.mkdir(full_dir)
            except:
                continue
                        
    copy(curr_model_fname, paste_model_fname)
    
    return

def load_model(home_dir, model_type, hazard, length, lead, train_type, wofs_lag = 25):
    
    if hazard == 'hail':
        radius = '39'
        buffer_str = '_20_min_buffer'
    elif hazard == 'wind':
        radius = '375'
        buffer_str = '_20_min_buffer'
    elif hazard == 'tornado':
        radius = '39'
        buffer_str = ''
    
    model_dir = '/work/ryan.martz/wofs_phi_data/experiments/'\
    '%s_trained/%s/length_%s/ale_curves/%s' %(model_type, train_type, length, hazard)
    
    for fold in range(0,5):
        if train_type == 'obs' or train_type == 'obs_and_warnings':
            model_fname = '%s_%s_trained_wofsphi_%s_%smin_window%s-%s_r%skm_fold%s.pkl'\
            %(model_type, train_type, hazard, length, lead, lead+length, radius, fold)
        else:
            model_fname = '%s_%s_trained_wofsphi_%s_%smin_window%s-%s_fold%s.pkl'\
            %(model_type, train_type, hazard, length, lead, lead+length, fold)
        
        if os.path.exists('%s/%s' %(model_dir, model_fname)):
            break
    
    if not (os.path.exists('%s/%s' %(model_dir, model_fname))):
        return False
    
    model = ti_funcs.unpickle('%s/%s' %(model_dir, model_fname))
    fold = int(model_fname.split('.')[0][-1])
    
    return model, fold

def load_predictors_events(model_type, optimal_fold, init_times, lead, length,\
                           n_variables, train_type, hazard, torp_ale,\
                           wofslag = 25, sampled = True):
    
    if hazard == 'hail':
        radius = '39'
        buffer_str = '_20_min_buffer'
    elif hazard == 'wind':
        radius = '375'
        buffer_str = '_20_min_buffer'
    elif hazard == 'tornado':
        radius = '39'
        buffer_str = ''
    
    if not sampled:
        save_dir ='/work/ryan.martz/wofs_phi_data/experiments/%s_trained/%s/length_%s'\
        '/ale_curves/%s' %(model_type, train_type, length, hazard)
        save_pred_file = 'all_100_percent_predictor_samples_%s_%s_%s_%s-%smin.npy'\
        %(hazard, model_type, train_type, lead, lead+length)
        save_obs_file = 'all_100_percent_obs_samples_%s_%s_%s_%s-%smin.npy'\
        %(hazard, model_type, train_type, lead, lead+length)
        
        try:
            all_preds = np.load('%s/%s' %(save_dir, save_pred_file))
            all_obs = np.load('%s/%s' %(save_dir, save_obs_file))
            
            if torp_ale:
                torp_probs = all_preds[:,271]
            
                non_zero_indices = np.where(torp_probs > 0)[0]
                zero_indices = np.where(torp_probs == 0)[0]
            
                rand_inds = np.random.choice(np.arange(len(zero_indices)),\
                                             size=int(0.25*len(zero_indices)),\
                                             replace=False)
                full_sample = np.append(non_zero_indices, rand_inds)
            
                all_preds = all_preds[full_sample, :]
                all_obs = all_obs[full_sample]
            
            return all_preds, all_obs
        except:
            pass
    
    if sampled:
        obs_and_warnings_dir = '/work/ryan.martz/wofs_phi_data/training_data/obs_and_warnings/'\
        'sampled_1d_obs_and_warnings/length_%s/wofs_lead_%s/%s/' %(length, lead, hazard)
        
        warnings_dir = '/work/ryan.martz/wofs_phi_data/training_data/warnings/'\
        'sampled_1d_warnings/length_%s/wofs_lead_%s/%s/' %(length, lead, hazard)
        
        obs_dir = '/work/eric.loken/wofs/2024_update/SFE2024/obs/dat_new_backup'
    else:
        obs_and_warnings_dir = '/work/ryan.martz/wofs_phi_data/training_data/obs_and_warnings/'\
        'full_1d_obs_and_warnings/length_%s/%s' %(length, hazard)
        
        warnings_dir = '/work/ryan.martz/wofs_phi_data/training_data/warnings/'\
        'full_1d_warnings/length_%s/%s' %(length, hazard)
        
        obs_dir = '/work/eric.loken/wofs/2024_update/SFE2024/obs/full_npy'
    
    dates = np.genfromtxt("/home/ryan.martz/python_packages/"\
                          "frdd-wofs-phi/wofs_phi/probSevere_dates.txt",\
                          dtype='str')
    
    ___, test_dates = ti_funcs.get_chunk_splits(dates, num_folds = 5, subsets = 2)
    #___, ___, test_dates = ti_funcs.get_chunk_splits(dates, num_folds = 5, subsets = 3)
    
    use_dates = test_dates[optimal_fold]
    
    load_dir = '/work/ryan.martz/wofs_phi_data/experiments/'\
    '%s_trained/%s/length_%s/ale_curves/%s'\
    %(model_type, train_type, length, hazard)
    
    if sampled:
        preds_file = '%s/%s_%s_trained_training_fdata_%s_%s-%smin_r%skm.dat'\
        %(load_dir, model_type, train_type, hazard, lead,\
          lead + length, radius)
        events_file = '%s/%s_%s_trained_training_odata_%s_%s-%smin_r%skm.dat'\
        %(load_dir, model_type, train_type, hazard, lead,\
          lead + length, radius)
        
        fcst_dir = '/work/eric.loken/wofs/2024_update/SFE2024/fcst/dat_backup'
        
    else:
        all_preds = np.zeros((1,1))
        all_obs = np.zeros((1,1))
        
        fcst_dir = '/work/eric.loken/wofs/2024_update/SFE2024/fcst/full_npy_backup'
    
    for date in use_dates:
        for init_time in init_times:
            init = dt.datetime(1970, 1, 1, int(init_time[0:2]), int(init_time[2:]))
            start_valid = (init + dt.timedelta(seconds = lead*60)).strftime('%H%M')
            ps_init = init + dt.timedelta(seconds = wofslag*60)
            if ps_init.minute % 2 == 1:
                ps_init = ps_init - dt.timedelta(seconds = 60)
            ps_init = ps_init.strftime('%H%M')
            end_valid = (init + dt.timedelta(seconds=(lead+length)*60)).strftime('%H%M')
            #print (date, init_time, start_valid, end_valid)
            
            if model_type == 'wofs_psv2_no_torp':
                if sampled:
                    test_x_file = "%s/wofs1d_%s_%s_%s_v%s-%s.dat"\
                    %(fcst_dir, date, init_time, ps_init, start_valid,\
                      end_valid)
                else:
                    test_x_file = "%s/wofs1d_%s_%s_%s_v%s-%s.npy"\
                    %(fcst_dir, date, init_time, ps_init, start_valid,\
                      end_valid)
            else:
                if sampled:
                    test_x_file = "%s/wofs1d_%s_%s_%s_%s_v%s-%s.dat"\
                    %(fcst_dir, model_type[5:], date, init_time,\
                      ps_init, start_valid, end_valid)
                else:
                    test_x_file = "%s/wofs1d_%s_%s_%s_%s_v%s-%s.npy"\
                    %(fcst_dir, model_type[5:], date, init_time,\
                      ps_init, start_valid, end_valid)
            if sampled:
                if model_type == 'wofs_psv2_no_torp':
                    obs_and_warnings_file = '%s/sampled_%s_obs_and_warnings'\
                    '_%s_%s_%s_v%s-%s_r%skm%s.dat'\
                    %(obs_and_warnings_dir, hazard, date, init_time,\
                      ps_init, start_valid, end_valid, radius, buffer_str)
                    
                    warnings_file = '%s/sampled_%s_warnings_%s_%s_%s_v%s-%s.dat'\
                    %(warnings_dir, hazard, date, init_time,\
                      ps_init, start_valid, end_valid)
                    
                    obs_file = "%s/%s_reps1d_%s_%s_%s_v%s-%s_r%skm%s.dat"\
                    %(obs_dir, hazard, date, init_time, ps_init,\
                      start_valid, end_valid, radius, buffer_str)
                else:
                    obs_and_warnings_file = '%s/%s_sampled_%s_obs_and_warnings'\
                    '_%s_%s_%s_v%s-%s_r%skm%s.dat'\
                    %(obs_and_warnings_dir, model_type[5:], hazard, date, init_time,\
                      ps_init, start_valid, end_valid, radius, buffer_str)
                    
                    obs_file = "%s/%s_%s_reps1d_%s_%s_%s_v%s-%s_r%skm%s.dat"\
                    %(obs_dir, model_type[5:], hazard, date,\
                      init_time, ps_init, start_valid, end_valid,\
                      radius, buffer_str)
                    
                    warnings_file = '%s/%s_sampled_%s_warnings_%s_%s_%s_v%s-%s.dat'\
                    %(warnings_dir, model_type[5:], hazard, date, init_time,\
                      ps_init, start_valid, end_valid)
            else:
                obs_and_warnings_file = "%s/%s_obs_and_warnings_%s_v%s-%s_r%skm%s_1d.npy"\
                %(obs_and_warnings_dir, hazard, date, start_valid,
                  end_valid, radius, buffer_str)
                
                warnings_file = "%s/%s_warnings_%s_v%s-%s_1d.npy"\
                %(warnings_dir, hazard, date, start_valid,
                  end_valid)
                
                obs_file = "%s/%s_reps1d_%s_v%s-%s_r%skm%s.npy"\
                %(obs_dir, hazard, date, start_valid, end_valid,\
                  radius, buffer_str)
            
            #try loading 1-d obs and 1-d forecast files 
            #if both fcst and obs files are found, proceed
            
            if not (os.path.exists(test_x_file) and os.path.exists(obs_file)\
                    and os.path.exists(warnings_file)):
                continue
            
            if train_type == 'warnings':
                obs_file = warnings_file
            elif train_type == 'obs_and_warnings':
                obs_file = obs_and_warnings_file
            
            if sampled:
                os.system('cat %s >> %s' %(test_x_file, preds_file))
                os.system('cat %s >> %s' %(obs_file, events_file))
            else:
                preds = np.load(test_x_file)
                obs = np.load(obs_file)
                
                #rand_inds = np.random.choice(np.arange(preds.shape[0]),\
                #size=90000, replace=False)
                #preds = preds[rand_inds,:]
                #obs = obs[rand_inds]
                
                if all_preds.shape[0] == 1 and all_preds.shape[1] == 1:
                    all_preds = preds
                    all_obs = obs
                else:
                    all_preds = np.append(all_preds, preds, axis = 0)
                    all_obs = np.append(all_obs, obs)
    
    if not sampled:
        np.save('%s/%s' %(save_dir, save_pred_file), all_preds)
        np.save('%s/%s' %(save_dir, save_obs_file), all_obs)
        
        if torp_ale:
            torp_probs = all_preds[:,271]

            non_zero_indices = np.where(torp_probs > 0)[0]
            zero_indices = np.where(torp_probs == 0)[0]

            rand_inds = np.random.choice(np.arange(len(zero_indices)),\
                                         size=int(0.25*len(zero_indices)),\
                                         replace=False)
            full_sample = np.append(non_zero_indices, rand_inds)

            all_preds = all_preds[full_sample, :]
            all_obs = all_obs[full_sample]

        return all_preds, all_obs
            
    else:
        all_preds = ti_funcs.read_binary(preds_file, n_variables, False)\
        #already float32... or should be probably not though because reasons
        all_obs = ti_funcs.read_obs_binary(events_file)
    
        os.system('rm %s' %(preds_file))
        os.system('rm %s' %(events_file))
        
        return all_preds, all_obs

def do_ale(home_dir, model_type, hazard, length, lead, train_type, init_times, torp_ale,\
          wofslag = 25, sampled = True):
    
    if hazard == 'hail':
        radius = '39'
        buffer_str = '_20_min_buffer'
    elif hazard == 'wind':
        radius = '375'
        buffer_str = '_20_min_buffer'
    elif hazard == 'tornado':
        radius = '39'
        buffer_str = ''
    
    ale_save_dir = '/work/ryan.martz/wofs_phi_data/experiments/%s_trained/%s/length_%s/ale_curves/%s'\
    %(model_type, train_type, length, hazard)
    
    if torp_ale:
        ale_save_file = 'highest_skill_model_ale_curves_%s_%s_%s_trained_%s-%smin_r%skm_torp_only.nc'\
        %(model_type, hazard, train_type, lead, lead+length, radius)
    else:
        ale_save_file = 'highest_skill_model_ale_curves_%s_%s_%s_trained_%s-%smin_r%skm.nc'\
        %(model_type, hazard, train_type, lead, lead+length, radius)
    
    if train_type == 'warnings':
        prob_save_file = '%s_%s_%s_trained_%s-%smin_probabilities.npy'\
        %(model_type, hazard, train_type, lead, lead+length)
    else:
        prob_save_file = '%s_%s_%s_trained_%s-%smin_r%skm_probabilities.npy'\
        %(model_type, hazard, train_type, lead, lead+length, radius)
    
    #if os.path.exists('%s/%s' %(ale_save_dir, ale_save_file)):
    #    return
    
    wofs_indices = np.concatenate((np.arange(0,54), np.arange(72,102), np.arange(120,150),\
                                   np.arange(168,198), np.arange(216,246), np.arange(266,269)))
    ps_indices = np.concatenate((np.arange(54,72), np.arange(102,120), np.arange(150,168),\
                                 np.arange(198,216), np.arange(246,264)))
    torp_indices = np.arange(269,374)
    torp_radar_indices = np.arange(289,369)
    
    var_dir = '/home/ryan.martz/python_packages/frdd-wofs-phi/wofs_phi'
    variables = np.genfromtxt("%s/rf_variables.txt" %(var_dir), dtype='str')
    var_dict = pd.read_csv('/work/ryan.martz/wofs_phi_data/experiments/vars_dict.csv')
    all_plvs = np.array(var_dict.Plain_Language_Variable)
    
    wofs_var_indices = np.array([25, 26, 27, 120, 123, 124, 125, 127, 264, 265, 268])
    ps_var_indices = np.array([155, 161, 167])
    torp_var_indices = np.array([271, 311, 326])
    
    n_variables = 374
    wofs_ps_indices = np.append(wofs_var_indices, ps_var_indices)
    var_indices = np.append(wofs_ps_indices, torp_var_indices)
    plvs = all_plvs[var_indices]
    
    try:
        model, fold = load_model(home_dir, model_type, hazard, length, lead, train_type)
    except:
        do_loading = load_model(home_dir, model_type, hazard, length, lead, train_type)
        
        if do_loading == False:
            return
    
    if torp_ale:
        print('loading predictors')
        all_preds, all_obs = load_predictors_events(model_type, fold, init_times, lead,\
                                                    length, n_variables, train_type,\
                                                    hazard, torp_ale, wofslag = 25, sampled = False)
    else:
        print('loading predictors')
        all_preds, all_obs = load_predictors_events(model_type, fold, init_times, lead,\
                                                    length, n_variables, train_type,\
                                                    hazard, torp_ale, wofslag = 25, sampled = sampled)
    print('predictors loaded')
    
    #all_preds = all_preds[:,torp_var_indices]
    if torp_ale:
        plvs = all_plvs[torp_var_indices]
        #plvs = all_plvs[torp_indices]
    
    model_name = '%s_%s_%s_trained_%s-%smin_r%skm' %(model_type, hazard, train_type, lead,\
                                                     lead+length, radius)
    
    print('doing ALE')
    explainer = skexplain.ExplainToolkit((model_name, model),X=all_preds,\
                                         y=all_obs, feature_names = all_plvs)
    ale_probs = model.predict_proba(all_preds)
    ale_1d_ds = explainer.ale(features=list(plvs))\
    #, n_bootstrap=2,subsample=10000, n_jobs=1, n_bins=20)
    utilities.save_data(ale_save_dir, ale_save_file, ale_1d_ds, 'nc')
    utilities.save_data(ale_save_dir, prob_save_file, ale_probs, 'npy')
    
    return

def main():
    
    init_times = ["1700", "1730", "1800", "1830", "1900", "1930", "2000", "2030", "2100",\
                  "2130", "2200", "2230", "2300", "2330", "0000", "0030", "0100", "0130",\
                  "0200", "0230", "0300", "0330", "0400", "0430", "0500", "0530", "0600",\
                  "0630", "0700", "0730", "0800", "0830", "0900", "0930", "1000", "1030",\
                  "1100", "1130"]
    home_dir = '/work/ryan.martz/wofs_phi_data/experiments'
    hazards = ['hail', 'wind', 'tornado']
    wofs_lag = 25
    
    #radii = [39]#, 15]
    train_types = ['obs', 'warnings']#, 'warnings']
    leads = [30, 90, 150]
    lengths = [60]
    model_types = ['wofs_psv3_with_torp']
    torp_ale = [True, False]
    
    #print('transferring models')
    #iterator = md.to_iterator([home_dir], model_types, hazards, [wofs_lag],\
    #                          lengths, leads, train_types)
    #results = md.run_parallel(transfer_models, iterator, nprocs_to_use = 20,\
    #                          description = 'Transferring Models')
    
    print('starting ALE')
    
    iterator = md.to_iterator([home_dir], model_types, hazards, lengths, leads, train_types,\
                              [init_times], torp_ale)
    results = md.run_parallel(do_ale, iterator, nprocs_to_use = 15,\
                                           description = 'Computing ALE')
    
    print('done')
    return


if (__name__ == '__main__'):

    main()