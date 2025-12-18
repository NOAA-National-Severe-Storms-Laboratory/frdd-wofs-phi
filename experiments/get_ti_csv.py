#===================================
# This file makes the TI csv files 
# necessary for TI analysis
#===================================


import numpy as np
import math
import os
import multiprocessing as mp
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from treeinterpreter import treeinterpreter as ti
import datetime as dt
import sys
sys.path.append('../wofs_phi')
from wofs_phi import utilities
from wofs_phi import multiprocessing_driver as md
from . import find_mean_normalized_ti_weight_contingency as cont_ti
import warnings

np.set_printoptions(threshold=sys.maxsize)


#User defined parameters
def initialize():

    #Lead time window 
    #1= 0-30, 2= 30-60, 3= 60-90, 4=90-120, 5= 120-150, 6= 150-180
    leads = [30, 60, 90, 120, 150, 180]
    hazards = ['hail', 'wind']#, 'tornado']
    radii = [39]#, 15]
    #init_times = np.genfromtxt("init_times.txt", dtype='str')
    init_times = ["1700", "1730", "1800", "1830", "1900", "1930", "2000", "2030", "2100",\
                        "2130", "2200", "2230", "2300", "2330", "0000", "0030", "0100", "0130",\
                        "0200", "0230", "0300", "0330", "0400", "0430", "0500", "0530", "0600",\
                        "0630", "0700", "0730", "0800", "0830", "0900", "0930", "1000", "1030",\
                        "1100", "1130"]
    #start_valids = np.genfromtxt("start_valids_w%s.txt" %window, dtype='str')
    #end_valids = np.genfromtxt("end_valids_w%s.txt" %window, dtype='str')
    lengths = [60]#, 120]
    train_types = ['obs', 'warnings']#, 'obs_and_warnings']
    wofslag = 25
    sampled = True
    used_srs = False
    
    model_types = ['wofs_psv3_with_torp']
    
    dates = np.genfromtxt("/home/ryan.martz/python_packages/frdd-wofs-phi/wofs_phi/probSevere_dates.txt",\
                          dtype='str')

    fcst_dat_dir = "/work/eric.loken/wofs/2024_update/SFE2024/fcst/dat_backup" #dat directory
    #fcst_full_npy_dir = "/work/eric.loken/wofs/paper6/fcst/30/full_npy" #full npy directory
    
    #fcst_dir = '/work/eric.loken/wofs/2024_update/SFE2024/fcst/full_npy'
    
    #Probably not needed right now. 
    #ps_variables = np.genfromtxt("rf_ps_variables_v9.txt", dtype='str') 
    #wofs_variables = np.genfromtxt("rf_wofs_variables_v9.txt", dtype='str')  

    #Kfold information
    total_folds = 5

    return leads, lengths, hazards, radii, init_times, dates, fcst_dat_dir,\
                total_folds, train_types, model_types, wofslag, sampled, used_srs

def read_binary(infile, nvars, header):
    #@infile is the filename of unformatted binary file
    #Returns a numpy array of data
    #nvars is the number of RF variables 
    #@header is a binary variable -- True if contains a 1-elmnt header/footer at 
    #beginning and end of file; else False


    f = open ( infile , 'rb' )
    arr = np.fromfile ( f , dtype = np.float32 , count = -1 )
    f.close()

    if (header == True):
        arr = arr[1:-1]


    #print(arr.shape)
    #Rebroadcast the data into 2-d array with proper format
    arr.shape = (-1, nvars)
    return arr


def read_obs_binary(infile):
    #@infile is the filename of unformatted binary file
    #Returns a numpy array of data

    f = open ( infile , 'rb' )
    arr = np.fromfile ( f , dtype = np.float32 , count = -1 )
    f.close()

    return arr

def getDateChunk(in_dates,k, curr_chunk,ndays1,ndays2,nChunks1,nChunks2, daySplit):
# in_dates is the incoming array of dates (an array of strings)
# k is the total number of chunks
# curr_chunk is the current chunk for testing
# ndays1 is the number of days in chunk/group 1
# ndays2 is the number of days in chunk/group 2
# nChunks1 is the number of chunks in group1
# nChunks2 is the number of chunks in group2
# daySplit is the list of day indices telling how in_dates should be split

#Split the dates array into chunks. The first 4 chunks have 29 days;
#The last 3 chunks have 30 days

#Obtain testing, validation, and training dates for current fold.
#How this works: testing is the current "chunk", 
#validation is the next "chunk", and training comprises all other chunks

    date_chunks = np.array_split(in_dates, daySplit)

    test_date = date_chunks[curr_chunk]

    #if the last chunk is the testing chunk
    if (curr_chunk == k-1):
        #val_date = date_chunks[0]
        train_date = np.concatenate(date_chunks[0:curr_chunk],axis=0)

    #if first chunk is the testing chunk
    elif (curr_chunk == 0):
        train_date = np.concatenate(date_chunks[1:],axis=0)

    else:
        #val_date = date_chunks[curr_chunk + 1]
        train_date1 = date_chunks[:curr_chunk]
        train_date2 = date_chunks[curr_chunk + 1:]
        train_date = train_date1 + train_date2
        train_date = np.concatenate((train_date), axis=0)


    return test_date, train_date

#Loads the pickled file into an array
def unpickle(filename):
    f = open(filename, 'rb')
    new = pickle.load(f)
    f.close()

    return new

def get_chunk_splits(dates, num_folds, subsets):
    '''Does the k fold logic. The class stores a list of dates to train on, and this samples it into training
    testing, validation for each fold. This allows us to just use the fold number as an index to each of the
    lists to get the list of dates for the current fold for testing, validation, and training.'''
    
    num_dates = len(dates)
    indices = np.arange(num_dates)
    splits = np.array_split(indices, num_folds)
    chunk_splits = []
    for i in range(len(splits)):
        split = splits[i]
        chunk_splits.append(split[0])
    chunk_splits.append(num_dates)
    if subsets == 3:
        date_test_folds = []
    date_val_folds = []
    date_train_folds = []
    if num_folds == 1:
        if subsets == 3:
            date_test_folds = [dates]
        date_val_folds = [dates]
        date_train_folds = [dates]
        return
    if subsets == 3:
        for i in range(num_folds):
            if i == 0:
                date_test_folds.append(dates[chunk_splits[i]:chunk_splits[i+1]])
                date_val_folds.append(dates[chunk_splits[-2]:chunk_splits[-1]])
                date_train_folds.append(dates[chunk_splits[i+1]:chunk_splits[i+1+(num_folds-2)]])
            elif i == num_folds - 1:
                date_test_folds.append(dates[chunk_splits[i]:chunk_splits[-1]])
                date_val_folds.append(dates[chunk_splits[i-1]:chunk_splits[i]])
                date_train_folds.append(dates[chunk_splits[0]:chunk_splits[num_folds-2]])
            elif (i+1+(num_folds-2)) > (num_folds):
                date_test_folds.append(dates[chunk_splits[i]:chunk_splits[i+1]])
                date_val_folds.append(dates[chunk_splits[i-1]:chunk_splits[i]])
                train_folds = dates[chunk_splits[i+1]:chunk_splits[-1]]
                train_folds.extend(dates[chunk_splits[0]:chunk_splits[i-1]])
                date_train_folds.append(train_folds)
            else:
                date_test_folds.append(dates[chunk_splits[i]:chunk_splits[i+1]])
                date_val_folds.append(dates[chunk_splits[i-1]:chunk_splits[i]])
                date_train_folds.append(dates[chunk_splits[i+1]:chunk_splits[i+1+(num_folds-2)]])
    elif subsets == 2:
        for i in range(num_folds):
            if i == 0:
                date_train_folds.append(dates[chunk_splits[0]:chunk_splits[num_folds-1]])
                date_val_folds.append(dates[chunk_splits[num_folds-1]:chunk_splits[num_folds]])
            else:
                train_folds = list(dates[chunk_splits[i]:chunk_splits[num_folds]])
                train_folds.extend(dates[chunk_splits[0]:chunk_splits[i-1]])
                train_folds = np.array(train_folds)
                date_train_folds.append(train_folds)
                date_val_folds.append(dates[chunk_splits[i-1]:chunk_splits[i]])
    
    if subsets == 3:
        return date_train_folds, date_val_folds, date_test_folds
    elif subsets == 2:
        return date_train_folds, date_val_folds

#Will obtain the list which shows how the data should be split
#(given uneven chunks)
def get_split_info(sampPerChunk1, sampPerChunk2,n1,n2):
#sampPerChunk1 is the samples_per_chunk1 variable (user defined)
#sampPerChunk2 is the samples_per_chunk2 variable
#n1 is the number of chunks in group 1
#n2 is the number of chunks in group 2
    chunk_list = []

    #first chunk
    for c in range(1,n1+1):
        chunk_list.append(c*sampPerChunk1)

    if (n2 > 0): #make sure we indeed have two chunks; if not, the follwoing code doesn't make sense
    #second chunk
        for c in range(1,n2): # only n2 instead of n2+1 b/c of how np.array_split works
            chunk_list.append(n1*sampPerChunk1+c*sampPerChunk2)

    return chunk_list

def do_kfold(lead, length, hazard, train_type, init_times, dates,\
             fcst_dir, wofslag, total_folds, model_type, sampled,\
             used_srs):
    
    if lead + length > 240:
        return
    
    if hazard == 'hail':
        radius = '39'
        buffer_str = '_20_min_buffer'
    elif hazard == 'wind':
        radius = '375'
        buffer_str = '_20_min_buffer'
    elif hazard == 'tornado':
        radius = '39'
        buffer_str = ''
    
    var_dir = '/home/ryan.martz/python_packages/frdd-wofs-phi/wofs_phi'
    variables = np.genfromtxt("%s/rf_variables.txt" %(var_dir), dtype='str')
    
    if 'with_torp' in model_type:
        torp_var_dir = '/work/ryan.martz/wofs_phi_data/training_data/predictors'
        torp_variables = np.genfromtxt("%s/torp_predictors.txt" %(torp_var_dir), dtype='str')
        
        if 'p_only' in model_type:
            non_radar_torp_variables = np.append(torp_variables[0:20], torp_variables[-5:])
            variables = np.append(variables, non_radar_torp_variables)
        else:
            variables = np.append(variables, torp_variables)
        
    if 'NoTor' in model_type:
        ps_tor_indices = np.concatenate((np.arange(54,60), np.arange(102,108), np.arange(150,156),\
                                         np.arange(198,204), np.arange(246,252)))
        variables = np.delete(variables, ps_tor_indices)
    elif 'psv0' in model_type:
        ps_indices = np.concatenate((np.arange(54,72), np.arange(102,120), np.arange(150,168),\
                                     np.arange(198,216), np.arange(246,264)))
        variables = np.delete(variables, ps_indices)
    
    if used_srs:
        train_date_folds, val_date_folds, test_date_folds =\
        get_chunk_splits(dates, total_folds, subsets = 3)
    else:
        train_date_folds, test_date_folds = get_chunk_splits(dates, total_folds, subsets = 2)
    pkl_dir = "/work/ryan.martz/wofs_phi_data/%s_train/models/%s/%s/wofslag_%s/length_%s"\
    %(train_type, model_type, hazard, wofslag, length)
    
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
        'full_1dwarnings/length_%s/%s' %(length, hazard)
        obs_dir = '/work/eric.loken/wofs/2024_update/SFE2024/obs/full_npy'
    
    n_variables = len(variables)

    #span the number of folds
    #span the dates (in given fold) 
    #span the init times/valid times
    #See if the data's there for that init/valid time
    #if so, load the RF, get the predictions from the RF, 
    #and get the TI contribution
    #Add the obs, probs, bias, and save to file

    start = True
    
    full_outdir = '/work/ryan.martz/wofs_phi_data/experiments/'\
    '%s_trained/%s/length_%s/tree_interpreter/%s'\
    %(model_type, train_type, length, hazard)
    
    full_outfile = 'sampled_aggregate_ti_data_%s_%s_%s_trained_%s-%smin_r%skm.csv'\
    %(model_type, hazard, train_type, lead, lead+length, radius)
    
    if os.path.exists('fake_file.txt'):
    #if os.path.exists('%s/%s' %(full_outdir, full_outfile)):
        cont_ti_outfile = "%s/%s_ti_summary_%s_%s_trained_%s-%smin_r%skm_contingency.csv"\
        %(full_outdir, model_type, hazard, train_type, lead, lead+length, radius)
        
        if os.path.exists(cont_ti_outfile):
            return
        
        headers = np.append(variables, ['obs', 'probs', 'bias'])
        csv_file = '%s/%s' %(full_outdir, full_outfile)
        full_df, top_n_vars, obs_df = cont_ti.read_df(headers, variables, csv_file)
        cont_ti.find_top_predictors(full_df, top_n_vars, obs_df, cont_ti_outfile)
        
        return
    
    print('%s %s-%smin, %s %skm %s trained'\
          %(model_type, lead, lead+length, hazard, radius, train_type))
    
    for k in range(total_folds):
        
        #print("fold %s" %str(k))
        
        #Get the appropriate test/train dates 
        test_dates = test_date_folds[k]
        
        #Get the appropriate pkl file
        if train_type == 'obs' or train_type == 'obs_and_warnings':
            pkl_file = "%s_%s_trained_wofsphi_%s_%smin_window%s-%s_r%skm_fold%s.pkl"\
            %(model_type, train_type, hazard, length, lead, lead + length, radius, k)
        else:
            pkl_file = "%s_%s_trained_wofsphi_%s_%smin_window%s-%s_fold%s.pkl"\
            %(model_type, train_type, hazard, length, lead, lead + length, k)
        
        if not os.path.exists('%s/%s' %(pkl_dir, pkl_file)):
            return
        
        #Load RF
        clf = unpickle('%s/%s' %(pkl_dir, pkl_file))
        
        for d in range(len(test_dates)):
            date = test_dates[d] 
            
            for v in range(len(init_times)):
                init_time = init_times[v]
                init = dt.datetime(1970, 1, 1, int(init_time[0:2]), int(init_time[2:]))
                start_valid = (init + dt.timedelta(seconds = lead*60)).strftime('%H%M')
                ps_init = init + dt.timedelta(seconds = wofslag*60)
                if ps_init.minute % 2 == 1:
                    ps_init = ps_init - dt.timedelta(seconds = 60)
                ps_init = ps_init.strftime('%H%M')
                end_valid = (init + dt.timedelta(seconds=(lead+length)*60)).strftime('%H%M')
                #print(date, init_time, start_valid, end_valid)
                
                if model_type == 'wofs_psv2_no_torp':
                    if sampled:
                        test_x_file = "%s/wofs1d_%s_%s_%s_v%s-%s.dat"\
                        %(fcst_dir, date, init_time, ps_init, start_valid, end_valid)
                    else:
                        test_x_file = "%s/wofs1d_%s_%s_%s_v%s-%s.npy"\
                        %(fcst_dir, date, init_time, ps_init, start_valid, end_valid)
                else:
                    if sampled:
                        test_x_file = "%s/wofs1d_%s_%s_%s_%s_v%s-%s.dat"\
                        %(fcst_dir, model_type[5:], date, init_time,\
                          ps_init, start_valid, end_valid)
                    else:
                        test_x_file = "%s/wofs1d_%s_%s_%s_%s_v%s-%s.npy"\
                        %(fcst_dir, model_type[5:], date, init_time,ps_init,\
                          start_valid, end_valid)
                    
                if sampled:
                    if model_type == 'wofs_psv2_no_torp':
                        obs_and_warnings_file = '%s/sampled_%s_obs_and_warnings'\
                        '_%s_%s_%s_v%s-%s_r%skm%s.dat'\
                        %(obs_and_warnings_dir, hazard, date, init_time, ps_init,\
                          start_valid, end_valid, radius, buffer_str)
                        warnings_file = '%s/sampled_%s_warnings_%s_%s_%s_v%s-%s.dat'\
                        %(warnings_dir, hazard, date, init_time, ps_init, start_valid, end_valid)
                        obs_file = "%s/%s_reps1d_%s_%s_%s_v%s-%s_r%skm%s.dat"\
                        %(obs_dir, hazard, date, init_time, ps_init, start_valid,\
                          end_valid, radius, buffer_str)
                    else:
                        obs_and_warnings_file = '%s/%s_sampled_%s_obs_and_warnings_%s_%s'\
                        '_%s_v%s-%s_r%skm%s.dat'\
                        %(obs_and_warnings_dir, model_type[5:], hazard, date, init_time,\
                          ps_init, start_valid, end_valid, radius, buffer_str)
                        warnings_file = '%s/%s_sampled_%s_warnings_%s_%s_%s_v%s-%s.dat'\
                        %(warnings_dir, model_type[5:], hazard, date, init_time,\
                          ps_init, start_valid, end_valid)
                        obs_file = "%s/%s_%s_reps1d_%s_%s_%s_v%s-%s_r%skm%s.dat"\
                        %(obs_dir, model_type[5:], hazard, date, init_time, ps_init, start_valid,\
                          end_valid, radius, buffer_str)
                else:
                    obs_and_warnings_file = "%s/%s_obs_and_warnings_%s_v%s-%s_r%skm%s_1d.npy"\
                    %(obs_and_warnings_dir, hazard, train_type, date, start_valid,\
                      end_valid, radius, buffer_str)
                    warnings_file = "%s/%s_warnings_%s_v%s-%s_1d.npy"\
                    %(warnings_dir, hazard, train_type, date, start_valid, end_valid)
                    obs_file = "%s/%s_reps1d_%s_v%s-%s_r%skm%s.npy"\
                    %(obs_dir, hazard, date, start_valid, end_valid, radius, buffer_str)
                
                #try loading 1-d obs and 1-d forecast files 
                #if both fcst and obs files are found, proceed
                try: 
                    if sampled:
                        test_x = read_binary(test_x_file, n_variables, False)
                    else:
                        test_x = np.load(test_x_file)
                
                except FileNotFoundError:
                    #print("File not found: %s" %(test_x_file))
                    continue
                
                try:
                    if sampled:
                        #one_d_obs_and_warnings = read_obs_binary(obs_and_warnings_file)
                        one_d_warnings = read_obs_binary(warnings_file)
                        one_d_obs = read_obs_binary(obs_file)
                    else:
                        #one_d_obs_and_warnings = np.load(obs_and_warnings_file)
                        one_d_warnings = np.load(warnings_file)
                        one_d_obs = np.load(obs_file)
                    
                except FileNotFoundError:
                    #print("File not found: %s, %s" %(obs_file, warnings_file))
                    continue
                
                if train_type == 'warnings':
                    one_d_obs = one_d_warnings
                elif train_type == 'obs_and_warnings':
                    one_d_obs = one_d_obs_and_warnings
                
                #Get ti stuff
                outdir = '/work/ryan.martz/wofs_phi_data/experiments/'\
                '%s_trained/%s/length_%s/tree_interpreter/%s/%s/%s'\
                %(model_type, train_type, length, hazard, date, init_time)
                
                ti_df = get_ti_df(clf, np.float32(test_x), np.float32(one_d_obs),\
                                  variables, outdir, date, init_time, start_valid,\
                                  end_valid, radius, hazard, length, sampled,\
                                  train_type)
                
                if start:
                    full_df = ti_df
                    start = False
                else:
                    full_df = pd.concat([full_df, ti_df], ignore_index=True)
    
    utilities.save_data(full_outdir, full_outfile, full_df, 'csv')
    
    obs_df = pd.DataFrame(full_df['obs'], columns = ['obs'])
    full_df = full_df[variables]
    top_n_vars = len(full_df.iloc[0,:])
    
    cont_ti_outfile = "%s/%s_ti_summary_%s_%s_trained_%s-%smin_r%skm_contingency.csv"\
    %(full_outdir, model_type, hazard, train_type, lead, lead+length, radius)
    
    cont_ti.find_top_predictors(full_df, top_n_vars, obs_df, cont_ti_outfile)
        

    return


def append_csvs(lead, length, hazard, train_type, radii, init_times, dates, model_type):

    for radius in radii:
        i = 0
        for d in range(len(dates)):
            date = dates[d]
            for v in range(len(init_times)):
                init_time = init_times[v]
                init = dt.datetime(1970, 1, 1, int(init_time[0:2]), int(init_time[2:]))
                start_valid = (init + dt.timedelta(seconds = lead*60)).strftime('%H%M')
                end_valid = (init + dt.timedelta(seconds=(lead+length)*60)).strftime('%H%M')
                #print(date, init_time, start_valid, end_valid)
                
                outdir = '/work/ryan.martz/wofs_phi_data/experiments/'\
                '%s_trained/%s/length_%s/tree_interpreter/%s/%s/%s'\
                %(model_type, train_type, length, hazard, date, init_time)
                
                df_infile = '%s_%s_%s_v%s-%s_r%skm_interp_w%s.csv'\
                %(date, hazard, init_time, start_valid, end_valid, radius, length)
                
                try:
                    df = pd.read_csv('%s/%s' %(outdir, df_infile))
                except FileNotFoundError:
                    print("File not found: %s" %(df_infile))
                    continue
                
                if i == 0:
                    full_df = df
                    i = 1
                    continue
                
                pd.concat([full_df, df])
                
        full_df_outdir = '/work/ryan.martz/wofs_phi_data/experiments/%s_trained/%s/'\
        'length_%s/tree_interpreter/%s' %(model_type, train_type, length, hazard)
        full_df_outfile = '%s_%s_%s_r%skm_%s-%smin_raw_ti_data.csv'\
        %(model_type, train_type, hazard, radius, lead, lead+length)

        utilities.save_data(full_df_outdir, full_df_outfile, full_df, 'csv')
        del full_df
                
    return

#Saves and returns Tree interpreter data for a given day/init time/start valid (in dataframe format)
#based on clf model, set of 1d predictors, and 1d obs, variable names, as well as outdirectory and relevant
#descriptive parameters
def get_ti_df(clf_model, predictors_1d, obs_1d, var_names, df_outdir, fdate, initTime,\
              startValid, endValid, rad, haz, timeWindow, sampled, train_type):
    
    if sampled:
        df_outfile = 'sampled_ti_data_%s_%s_%s_trained_%s_v%s-%s_r%skm.csv'\
        %(fdate, haz, train_type, initTime, startValid, endValid, rad)
    else:
        df_outfile = '%s_%s_%s_%s_v%s-%s_r%skm_interp_w%s.csv'\
        %(fdate, haz, initTime, startValid, endValid, rad, timeWindow)
    
    #if os.path.exists('%s/%s' %(df_outdir, df_outfile)):
    #    return 0
    
    #Obtain probabilities 
    clf_probs = clf_model.predict_proba(predictors_1d)[:,1]

    #Get TI stuff 
    prediction, bias, contributions = ti.predict(clf_model, predictors_1d)
    interp = contributions[:,:,1] #only care about 'yes' prediction class
    
    #Create a dataframe to hold interpretability scores probs, and obs 
    df = pd.DataFrame(data=interp, columns=var_names)

    #Add the observations and probs 
    df['obs'] = obs_1d
    df['probs'] = clf_probs
    df['bias'] = bias[:,1]

    #Save df to file
    utilities.save_data(df_outdir, df_outfile, df, 'csv')

    return df


def main():
    
    warnings.filterwarnings('ignore')
    
    #Get input values
    leads, lengths, hazards, radii, init_times, dates, fcst_dir,\
    total_folds, train_types, model_types, wofslag, sampled, used_srs = initialize()
    
    iterator = md.to_iterator(leads, lengths, hazards, train_types, [init_times],\
                              [dates], [fcst_dir], [wofslag], [total_folds],\
                              model_types, [sampled], [used_srs])
    results = md.run_parallel(do_kfold, iterator, nprocs_to_use = 10,\
                                           description = 'Making TI CSVs')
    
    #for length in lengths:
    #    for lead in leads:
    #        if lead + length > 240:
    #            continue
    #        for hazard in hazards:
    #            for train_type in train_types:
    #                
    #                #Get day chunk list -- shows how the data should be split into chunks by date
    #                #day_chunk_list = get_split_info(n_days_per_fold1, n_days_per_fold2, k1, k2)
    #                
    #                
    #                    
    #                #Do the kfold stuff and get the csv files 
    #                do_kfold(lead, length, hazard, train_type, radii, init_times,\
    #                         dates, variables,\
    #                         fcst_dir, wofslag,\
    #                         total_folds, model_type, sampled)
    #                
    #                #append_csvs(lead, length, hazard, train_type, radii, init_times, dates, model_type)


    return 

if (__name__ == '__main__'):

    main()