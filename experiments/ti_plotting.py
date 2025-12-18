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
import multiprocessing as mp
from wofs_phi import multiprocessing_driver as md

var_df = pd.read_csv('/work/ryan.martz/wofs_phi_data/experiments/vars_dict.csv')
var_dict = dict(zip(var_df.Variable, var_df.Plain_Language_Variable))
wofs_indices = np.concatenate((np.arange(0,54), np.arange(72,102), np.arange(120,150),\
                               np.arange(168,198), np.arange(216,246), np.arange(266,269)))
ps_indices = np.concatenate((np.arange(54,72), np.arange(102,120), np.arange(150,168),\
                             np.arange(198,216), np.arange(246,264)))
torp_indices = np.arange(269,374)
torp_nr_indices = np.append(np.arange(269,289), np.arange(369,374))
lat_lon_indices = np.array([264, 265])

def plot_ti(topx, leads, model, train_type, hazard, length, radius):
    
    fig_save_dir = '/work/ryan.martz/wofs_phi_data/experiments/%s_trained/%s/length_%s/tree_interpreter/%s'\
    %(model, train_type, length, hazard)
    fig_save_file = 'sampled_aggregate_ti_data_%s_%s_%s_trained_r%skm.png'\
    %(model, hazard, train_type, radius)
    
    #if os.path.exists('%s/%s' %(fig_save_dir, fig_save_file)):
    #    return
    
    fig, axs = plt.subplots(2,3)
    for i in range(len(leads)):
        lead = leads[i]
        if lead + length > 240:
            continue

        fdir = '/work/ryan.martz/wofs_phi_data/experiments/%s_trained/%s/length_%s/tree_interpreter/%s'\
        %(model, train_type, length, hazard)
        fname = 'sampled_aggregate_ti_data_%s_%s_%s_trained_%s-%smin_r%skm.csv'\
        %(model, hazard, train_type, lead, lead+length, radius)
        try:
            ti_df = pd.read_csv('%s/%s' %(fdir, fname))
        except:
            return

        predictors = np.abs(ti_df.iloc[:,1:-3])
        pred_sums = np.sum(predictors, axis = 1)
        rel_contribs = predictors.div(pred_sums, axis=0)
        avg_rel_contribs = np.mean(rel_contribs, axis = 0)
        top_vars = avg_rel_contribs.sort_values()[-topx:]
        colors = []
        ticks = []
        for column in top_vars.index:
            if column in np.array(var_df.Variable):
                ticks.append(var_dict[column])
                index = np.where(np.array(var_df.Variable) == column)[0][0]
                if index in wofs_indices:
                    colors.append('r')
                elif index in ps_indices:
                    colors.append('b')
                elif index in lat_lon_indices:
                    colors.append('k')
                else:
                    colors.append('c')

        top_vars.plot.barh(ax=fig.axes[i], color = colors)
        fig.patch.set_facecolor('xkcd:light gray')
        fig.axes[i].set_title('%s - %s Minutes' %(lead-30, lead+length-30), fontsize = 25, fontweight = 'bold')
        fig.axes[i].set_yticklabels(ticks, fontweight = 'bold')
        for ytick, color in zip(fig.axes[i].get_yticklabels(), colors):
            ytick.set_color(color)
        fig.axes[i].set_xlim([0,0.07])

    fig.set_size_inches(20, 10)
    fig.tight_layout()
    
    utilities.save_data(fig_save_dir, fig_save_file, fig, 'png')
    
    return

def main():
    
    topx = 20
    
    hazards = ['wind', 'tornado', 'hail']
    lengths = [60, 120]
    models = ['wofs_psv2_no_torp', 'wofs_psv2_with_torp', 'wofs_psv3_no_torp', 'wofs_psv3_with_torp',\
             'wofs_psv3_with_torp_filtered', 'wofs_psv3_with_torp_filtered_p_only']
    train_types = ['obs', 'obs_and_warnings']
    radii = [15, 39]
    leads = np.array([30, 60, 90, 120, 150, 180])
    topx = 20
    
    iterator = md.to_iterator([topx], [leads], models, train_types, hazards, lengths, radii)
    results = md.run_parallel(plot_ti, iterator, nprocs_to_use = int(0.6*mp.cpu_count()),\
                                           description = 'Plotting Aggregate TI')
    
    #for length in lengths:
    #    print('Length: %s min' %(length))
    #    for radius in radii:
    #        print('Radius: %s km' %(radius))
    #        for model in models:
    #            print('Model: %s' %(model))
    #            for train_type in train_types:
    #                print('Train Type: %s' %(train_type))
    #                for hazard in hazards:
    #                    print('Hazard: %s' %(hazard))
    #                    plot_ti(topx, leads, model, train_type, hazard, length, radius)
    
    return

if (__name__ == '__main__'):

    main()