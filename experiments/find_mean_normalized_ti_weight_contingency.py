#################################################
# Finds the mean normalized TI weight (independent
# of sign/directionality) for each predictor, 
# plus climo 
#
# NOTE: This script takes things a step farther: 
# breaks down RTI contributions based on contingency 
# table elements:
#   Hit is when probabilities are increased correctly
#   Miss is when probs are decreased incorrectly 
#   False alarm is when probs are increased incorrectly
#   Cneg is when probs are decreased correctly 
#################################################

##################
# Imports
##################

import numpy as np 
import os 
import pickle
import pandas as pd 
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import netCDF4 as nc
import sys
sys.path.append('../wofs_phi')
from wofs_phi import utilities
from wofs_phi import multiprocessing_driver as md
import multiprocessing as mp
####################

#User defined parameters/inputs 
def initialize(hazard, radius, train_type, lead, length, model_type,\
              radar_data, filtered_torp, top_n_vars):
    
    ti_csv_dir = '/work/ryan.martz/wofs_phi_data/experiments/'\
    '%s_trained/%s/length_%s/tree_interpreter/%s'\
    %(model_type, train_type, length, hazard)
    
    #csv_file = "%s/test_interp_w1.csv" %ti_csv_dir
    #csv_file = "%s/20190506_hail_2300_v2325-2355_r39.0km_interp_w1.csv" %ti_csv_dir
    #csv_file = "%s/short_test_interp_w1.csv" %ti_csv_dir
    csv_file = '%s/sampled_aggregate_ti_data_%s_%s_%s_trained_%s-%smin_r%skm.csv'\
    %(ti_csv_dir, model_type, hazard, train_type, lead, lead+length, radius)

    ti_outfile = "%s/%s_ti_summary_%s_%s_trained_%s-%smin_r%skm_contingency.csv"\
    %(ti_csv_dir, model_type, hazard, train_type, lead, lead+length, radius)
    #ti_outfile = "test_ti.csv"
    
    var_dir = '/home/ryan.martz/python_packages/frdd-wofs-phi/wofs_phi'
    retain_ti_variables = np.genfromtxt("%s/rf_variables.txt" %(var_dir), dtype='str')
    
    if 'with_torp' in model_type:
        torp_var_dir = '/work/ryan.martz/wofs_phi_data/training_data/predictors'
        torp_variables = np.genfromtxt("%s/torp_predictors.txt" %(torp_var_dir), dtype='str')
        if radar_data:
            retain_ti_variables = np.append(retain_ti_variables, torp_variables)
        else:
            non_radar_torp_variables = np.append(torp_variables[0:20], torp_variables[-5:])
            retain_ti_variables = np.append(retain_ti_variables, non_radar_torp_variables)
    if 'NoTor' in model_type:
        ps_tor_indices = np.concatenate((np.arange(54,60), np.arange(102,108), np.arange(150,156),\
                                         np.arange(198,204), np.arange(246,252)))
        retain_ti_variables = np.delete(retain_ti_variables, ps_tor_indices)
    elif 'psv0' in model_type:
        ps_indices = np.concatenate((np.arange(54,72), np.arange(102,120), np.arange(150,168),\
                                     np.arange(198,216), np.arange(246,264)))
        retain_ti_variables = np.delete(retain_ti_variables, ps_indices)
    
    ti_variables = np.append(retain_ti_variables, ['obs', 'probs', 'bias'])
    
    return top_n_vars, ti_variables, retain_ti_variables,  csv_file, ti_outfile


#Reads dataframe 
#Returns the dataframe containing the ti values as well as the number of variables in the df 
#interp_headers is a list of header names corresonding to the interp file 
#retain_headers is a list of header names to retain (e.g., no obs or probs)
#interp_ffile is the csv file with the ti values 

def read_df(interp_headers, retain_headers, interp_ffile):

    #Read in the titles/headers

    #Read in csv data
    df = pd.read_csv(interp_ffile, dtype=np.float32)

    #Keep a df for obs
    df_obs = pd.DataFrame(df['obs'], columns = ['obs'])

    #Get rid of the extraneous columns (e.g., obs, probs) 
    df = df[retain_headers]
    n = len(df.iloc[0,:])

    return df, n, df_obs


#OLD: 
#Finds the top N predictors  -- Returns the normalized dataframe (all points; relative to normal variable) with and without directionality preserved, 
# as well as a dataframe showing the fraction of importance values from the other variables (no directionality), 
# and finally the top predictors' name and mean normalized values (relative to "normal" predictor 1/total) 

#NOTE: Am modifying this function a bit. 
#@df is incoming dataframe of interpretability values -- now excludes 'bias', 'probs', and 'obs'
#@obs_df is the df containing the obs
#@total_vars is the number of total variables in the data frame (i.e., number of predictors) 
#@topN is the number of top predictors to retain/return
#@prob_thresh gives the probability threshold of points to consider (i.e., only consider points at or above @prob_thresh when 
#determining global importance of variables) 
def find_top_predictors(df, topN, obs_df,ti_outfile_name):
    '''Update: Returns the dataframe of Relative Tree Interpreter importances, broken down 
        into overall, hits, misses, false alarms, correct negatives, and weighted hits, 
        weighted misses, weighted false alarms, and wighted correct negatives. The weighted
        values are weighted by the overall RTI values.
    '''

    #Find out the overall 'weight' of the contribution
    copy_df = copy.deepcopy(df) 
    abs_df = copy_df.abs()

    #Compute the (absolute value) sum of each row 
    abs_df['sum'] = abs_df.sum(axis=1)

    #Add this absolute value sum to the original df 
    copy_df['sum'] = abs_df['sum'] 

    #Divide by the absolute value sum of each row to get a normalized df 
    copy_df = copy_df.divide(copy_df['sum'], axis=0)

    abs_copy_df = copy_df.abs()

    overall_mean_rti = abs_copy_df.mean(axis=0)

    #Add a row giving the absolute value sum for each row 

    #Absolute value row sum 
    col_abs_sum = abs_copy_df.sum(axis=0)

    #Now, break into contingency table elements 

    #Compute contingency table elements 
    yes_obs_df = copy_df.where(obs_df['obs'] == 1, other=0.0)

    no_obs_df = copy_df.where(obs_df['obs'] == 0, other=0.0)

    hits_df = yes_obs_df.where(yes_obs_df > 0.0, other=0.0)

    misses_df = yes_obs_df.where(yes_obs_df < 0.0, other=0.0) 

    false_alarm_df = no_obs_df.where(no_obs_df > 0.0, other=0.0)

    cneg_df = no_obs_df.where(no_obs_df < 0.0, other=0.0) 

    #Take absolute value of each 
    hits_df = hits_df.abs()
    misses_df = misses_df.abs()
    false_alarm_df = false_alarm_df.abs()
    cneg_df = cneg_df.abs() 


    #Now: What we really want is the sum of each column divided by the col_abs_sum 
    hits_df = hits_df.sum(axis=0)
    misses_df = misses_df.sum(axis=0)
    false_alarm_df = false_alarm_df.sum(axis=0)
    cneg_df = cneg_df.sum(axis=0)


    #Now, divide by col_abs_sum
    hits_df = hits_df.divide(col_abs_sum, axis=0)
    misses_df = misses_df.divide(col_abs_sum, axis=0)
    false_alarm_df = false_alarm_df.divide(col_abs_sum, axis=0)
    cneg_df = cneg_df.divide(col_abs_sum, axis=0) 

    #Now, put all of these in a single dataframe 
    final_df = pd.concat([hits_df, misses_df, false_alarm_df, cneg_df], axis=1)
    final_df.columns = ['hits', 'misses', 'false', 'cnegs']

    #Add relevant columns 
    final_df['good'] = final_df['hits'] + final_df['cnegs']
    final_df['bad'] = final_df['misses'] + final_df['false']
    final_df['overall'] = overall_mean_rti


    final_df['weighted_hits'] = final_df['hits']*final_df['overall']
    final_df['weighted_misses'] = final_df['misses']*final_df['overall']
    final_df['weighted_false'] = final_df['false']*final_df['overall']
    final_df['weighted_cnegs'] = final_df['cnegs']*final_df['overall']

    #Exclude the "sum" field---has no meaning
    final_df = final_df.iloc[:-1, :]


    #Save to file 
    final_df.to_csv(ti_outfile_name, header=True, index=True)


    return

def do_summarizing(hazard, radius, train_type, lead, length, model_type,\
                   radar_data, filtered_torp, top_n_vars):
    
    top_n_vars, ti_variables, retain_ti_variables, csv_file, ti_outfile =\
    initialize(hazard, radius, train_type, lead, length, model_type,\
               radar_data, filtered_torp, top_n_vars)
    
    if (not os.path.exists(csv_file)):# or os.path.exists(ti_outfile):
        return
    
    #Read in the TI values df
    #interp_headers, retain_headers, interp_ffile
    print ("reading") 
    interp_df, nvars, obs_df = read_df(ti_variables, retain_ti_variables, csv_file)
   

    #Get rid of the 'bias' column in interp_df
    #interp_df.drop(columns=['bias'], inplace=True)

    print ("data read") 
    #normalized_df, top_var_names, top_normalized_ti, top_pos_ti, top_neg_ti = find_top_predictors(interp_df, top_n_vars, obs_df, ti_outfile)
    ti_data = find_top_predictors(interp_df, top_n_vars, obs_df, ti_outfile)
    
    return

def main():
    
    hazards = ['tornado', 'wind', 'hail']
    radii = [15, 39]
    train_types = ['obs']#, 'obs_and_warnings']
    lengths = [60]#, 120]
    leads = [30, 60, 90, 120, 150, 180]
    
    ps_version = 2
    include_torp_in_predictors = False
    radar_data = True
    filtered_torp = False
    model_type = ''
    top_n_vars = 50 #Number of top variables to report
    
    iterator = md.to_iterator(hazards, radii, train_types, leads, lengths, model_types,\
                              [radar_data], [filtered_torp], [top_n_vars])
    results = md.run_parallel(do_summarizing, iterator, nprocs_to_use = 3,\
                                           description = 'Making TI CSVs')
    
    #for hazard in hazards:
    #    for radius in radii:
    #        for train_type in train_types:
    #            for length in lengths:
    #                for lead in leads:
    #                    do_summarizing(hazard, radius, train_type, lead, length, ps_version, include_torp_in_predictors,\
    #                          radar_data, filtered_torp, top_n_vars)
    
    #print (top_var_names)

    #print (top_normalized_ti) 


    return


if (__name__ == '__main__'):

    main()




