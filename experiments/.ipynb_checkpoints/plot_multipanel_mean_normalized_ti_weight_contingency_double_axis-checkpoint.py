############################################################
# This script makes the bar plots based on numbers 
# in the csv files (from find_mean_normalized_ti_weight.py
#
# Additional NOTES: This file was created during first 
# round of revisions. Goal will be to plot the true positives
# and false negatives on a separate axis as the true negatives
# and false positives to better display the 4 contingency 
# elements 
############################################################

##################
# Imports
##################

import numpy as np
import os
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import netCDF4 as nc
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import ScalarFormatter
import sys
sys.path.append('../wofs_phi')
from wofs_phi import utilities
from wofs_phi import multiprocessing_driver as md
import multiprocessing as mp
import imageio.v3 as iio

####################

#User
def initialize(ps_version, include_torp_in_predictors, radar_data, filtered_torp, hazard, radius, train_type, length):
    
    var_df = pd.read_csv('/work/ryan.martz/wofs_phi_data/experiments/vars_dict.csv')
    var_dict = dict(zip(var_df.Variable, var_df.Plain_Language_Variable))
    plot_variables = np.array(var_df.Plain_Language_Variable)
    raw_variables = np.array(var_df.Variable)
    
    wofs_indices = np.concatenate((np.arange(0,54), np.arange(72,102), np.arange(120,150),\
                                   np.arange(168,198), np.arange(216,246), np.arange(266,269)))
    ps_indices = np.concatenate((np.arange(54,72), np.arange(102,120), np.arange(150,168),\
                                 np.arange(198,216), np.arange(246,264)))
    torp_indices = np.arange(269,374)
    torp_radar_indices = np.arange(289,369)
    if (ps_version == 2) and (not include_torp_in_predictors):
        model_type = 'wofs_psv2_no_torp'
        use_plot_variables = np.delete(plot_variables, torp_indices)
    elif (ps_version == 2) and (include_torp_in_predictors):
        model_type = 'wofs_psv2_with_torp'
        use_plot_variables = plot_variables
    elif (ps_version == 3) and (not include_torp_in_predictors):
        model_type = 'wofs_psv3_no_torp'
        use_plot_variables = np.delete(plot_variables, torp_indices)
    elif (ps_version == 3) and (include_torp_in_predictors):
        if radar_data and (not filtered_torp):
            model_type = 'wofs_psv3_with_torp'
            use_plot_variables = plot_variables
        elif radar_data and filtered_torp:
            model_type = 'wofs_psv3_with_torp_filtered'
            use_plot_variables = plot_variables
        elif (not radar_data) and filtered_torp:
            model_type = 'wofs_psv3_with_torp_filtered_p_only'
            torp_indices = np.append(np.arange(269,289), np.arange(369,374))
            use_plot_variables = np.delete(plot_variables, torp_radar_indices)
        elif (not radar_data) and (not filtered_torp):
            model_type = 'wofs_psv3_with_torp_p_only'
            torp_indices = np.append(np.arange(269,289), np.arange(369,374))
            use_plot_variables = np.delete(plot_variables, torp_radar_indices)
        else:
            return ''
    elif (ps_version == 0) and (include_torp_in_predictors):
        model_type = 'wofs_psv0_with_torp'
        use_plot_variables = np.delete(plot_variables, ps_indices)
    else:
        return ''
        
    if length == 60 or length == 30:
        leads = [30, 60, 90, 120, 150, 180]
        titles = ["0-60 Minutes", "30-90 Minutes", "60-120 Minutes",\
                    "90-150 Minutes", "120-180 Minutes", "150-210 Minutes"]
        panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        show_legends = [False, False, False, False, False, True]
    elif length == 120:
        leads = [30, 60, 90, 120]
        titles = ["0-120 Minutes", "30-150 Minutes", "60-180 Minutes",\
                    "90-210 Minutes"]
        panel_labels = ['(a)', '(b)', '(c)', '(d)']
        show_legends = [False, False, False, True]
    
    ti_csv_dir = '/work/ryan.martz/wofs_phi_data/experiments/'\
    '%s_trained/%s/length_%s/tree_interpreter/%s'\
    %(model_type, train_type, length, hazard)
    
    csv_names = ['%s/%s_ti_summary_%s_%s_trained_%s-%smin_r%skm_contingency.csv'\
    %(ti_csv_dir, model_type, hazard, train_type, lead, lead+length, radius) for lead in leads]
    
    wofs_variables = plot_variables[wofs_indices]
    ps_variables = plot_variables[ps_indices]
    torp_variables = plot_variables[torp_indices]

    wofs_color = 'chocolate'
    ps_color = 'darkmagenta'
    torp_color = 'deepskyblue'
    other_color = 'black'
    
    grouped_names_dict = {'80m Wind': ["ws_80", "ws_80_r15.0", "ws_80_r30.0", "ws_80_r45.0", "ws_80_r60.0"], \
        '1km dBZ': ["dbz_1km", "dbz_1km_r15.0", "dbz_1km_r30.0", "dbz_1km_r45.0", "dbz_1km_r60.0"],\
        '0-2km Vertical Vorticity': ["wz_0to2_instant", "wz_0to2_instant_r15.0", "wz_0to2_instant_r30.0", \
                                "wz_0to2_instant_r45.0", "wz_0to2_instant_r60.0"],\
        '0-2km Updraft Helicity': ["uh_0to2_instant", "uh_0to2_instant_r15.0", "uh_0to2_instant_r30.0",\
                                    "uh_0to2_instant_r45.0", "uh_0to2_instant_r60.0"],\
        '2-5km Updraft Helicity': ["uh_2to5", "m1_uh_2to5", "m2_uh_2to5", "m3_uh_2to5",\
                                    "m4_uh_2to5", "m5_uh_2to5" ,"m6_uh_2to5", "m7_uh_2to5",\
                                    "m8_uh_2to5", "m9_uh_2to5", "m10_uh_2to5", "m11_uh_2to5",\
                                    "m12_uh_2to5", "m13_uh_2to5", "m14_uh_2to5", "m15_uh_2to5",\
                                    "m16_uh_2to5", "m17_uh_2to5", "m18_uh_2to5",\
    
                                    "uh_2to5_r15.0", "m1_uh_2to5_r15.0", "m2_uh_2to5_r15.0", "m3_uh_2to5_r15.0",\
                                    "m4_uh_2to5_r15.0", "m5_uh_2to5_r15.0" ,"m6_uh_2to5_r15.0", "m7_uh_2to5_r15.0",\
                                    "m8_uh_2to5_r15.0", "m9_uh_2to5_r15.0", "m10_uh_2to5_r15.0", "m11_uh_2to5_r15.0",\
                                    "m12_uh_2to5_r15.0", "m13_uh_2to5_r15.0", "m14_uh_2to5_r15.0", "m15_uh_2to5_r15.0",\
                                    "m16_uh_2to5_r15.0", "m17_uh_2to5_r15.0", "m18_uh_2to5_r15.0",\

                                    "uh_2to5_r30.0", "m1_uh_2to5_r30.0", "m2_uh_2to5_r30.0", "m3_uh_2to5_r30.0",\
                                    "m4_uh_2to5_r30.0", "m5_uh_2to5_r30.0" ,"m6_uh_2to5_r30.0", "m7_uh_2to5_r30.0",\
                                    "m8_uh_2to5_r30.0", "m9_uh_2to5_r30.0", "m10_uh_2to5_r30.0", "m11_uh_2to5_r30.0",\
                                    "m12_uh_2to5_r30.0", "m13_uh_2to5_r30.0", "m14_uh_2to5_r30.0", "m15_uh_2to5_r30.0",\
                                    "m16_uh_2to5_r30.0", "m17_uh_2to5_r30.0", "m18_uh_2to5_r30.0",\

                                    "uh_2to5_r45.0", "m1_uh_2to5_r45.0", "m2_uh_2to5_r45.0", "m3_uh_2to5_r45.0",\
                                    "m4_uh_2to5_r45.0", "m5_uh_2to5_r45.0" ,"m6_uh_2to5_r45.0", "m7_uh_2to5_r45.0",\
                                    "m8_uh_2to5_r45.0", "m9_uh_2to5_r45.0", "m10_uh_2to5_r45.0", "m11_uh_2to5_r45.0",\
                                    "m12_uh_2to5_r45.0", "m13_uh_2to5_r45.0", "m14_uh_2to5_r45.0", "m15_uh_2to5_r45.0",\
                                    "m16_uh_2to5_r45.0", "m17_uh_2to5_r45.0", "m18_uh_2to5_r45.0",\

                                    "uh_2to5_r60.0", "m1_uh_2to5_r60.0", "m2_uh_2to5_r60.0", "m3_uh_2to5_r60.0",\
                                    "m4_uh_2to5_r60.0", "m5_uh_2to5_r60.0" ,"m6_uh_2to5_r60.0", "m7_uh_2to5_r60.0",\
                                    "m8_uh_2to5_r60.0", "m9_uh_2to5_r60.0", "m10_uh_2to5_r60.0", "m11_uh_2to5_r60.0",\
                                    "m12_uh_2to5_r60.0", "m13_uh_2to5_r60.0", "m14_uh_2to5_r60.0", "m15_uh_2to5_r60.0",\
                                    "m16_uh_2to5_r60.0", "m17_uh_2to5_r60.0", "m18_uh_2to5_r60.0"],\
            
        'Updraft Speed': ["w_up", "w_1km", "w_up_r15.0", "w_1km_r15.0", "w_up_r30.0", "w_1km_r30.0",\
                            "w_up_r45.0", "w_1km_r45.0",  "w_up_r60.0", "w_1km_r60.0"],\


        '10-500m Bulk Shear': ["10-500m_bulkshear"], \
        'Flash Extent Density': ["fed", "fed_r15.0", "fed_r30.0", "fed_r45.0", "fed_r60.0"], \
        '10m Wind': ["u_10", "v_10"],\
        '2m Temperature': ["t_2"],\
        '2m Dewpoint': ["td_2"],\

        'Mid-level Lapse Rate': ["mid_level_lapse_rate"],\
        'Low-level Lapse Rate': ["low_level_lapse_rate"],\
        '0-1km Shear Components': ["shear_u_0to1", "shear_v_0to1"],\
        '0-3km Shear Components': ["shear_u_0to3", "shear_v_0to3"],\
        '0-6km Shear Components': ["shear_u_0to6", "shear_v_0to6"],\
        'SRH': ["srh_0to500", "srh_0to1", "srh_0to3"],\
        'SBCAPE': ["cape_sfc"],\
        'STP': ["stp", "stp_srh0to500"],\
        'SCP': ["scp"],\
        'Downdraft Speed': ["w_down", "w_down_r15.0", "w_down_r30.0", "w_down_r45.0", "w_down_r60.0"],\
        'Cloud Top Temperature': ["ctt"],\
        'Surface Pressure': ["mslp", "mslp_r15.0", "mslp_r30.0", "mslp_r45.0", "mslp_r60.0", \
                             "psfc", "psfc_r15.0", "psfc_r30.0", "psfc_r45.0", "psfc_r60.0"],\
        'LCL Height': ["lcl_sfc"],\
        'Hailcast Hail': ["hail"],\
        'Freezing Level': ["freezing_level"],\
        'WoFS probability of 40dBZ': ["prob_40dbz", "prob_40dbz_r15.0", "prob_40dbz_r30.0",\
                                        "prob_40dbz_r45.0", "prob_40dbz_r60.0"],\
        'In ICs': ["inICs"],\
        'PS3 Tornado Probability': ['tornado_raw_probs', 'tornado_raw_probs_r15.0',\
                                    'tornado_raw_probs_r30.0', 'tornado_raw_probs_r45.0',\
                                    'tornado_raw_probs_r60.0', 'tornado_smoothed_probs',\
                                    'tornado_smoothed_probs_r15.0', 'tornado_smoothed_probs_r30.0',\
                                    'tornado_smoothed_probs_r45.0', 'tornado_smoothed_probs_r60.0'],\
        'PS3 Tornado 14 Min. Change': ["tornado_changes14", "tornado_changes14_r15.0",\
                                      "tornado_changes14_r30.0", "tornado_changes14_r45.0",\
                                      "tornado_changes14_r60.0"],\
        'PS3 Tornado 30 Min. Change': ["tornado_changes30", "tornado_changes30_r15.0",\
                                        "tornado_changes30_r30.0", "tornado_changes30_r45.0",
                                        "tornado_changes30_r60.0"],\
        'PS3 Age': ["tornado_ages", "wind_ages", "hail_ages",\
                    "tornado_ages_r15.0", "wind_ages_r15.0", "hail_ages_r15.0",\
                    "tornado_ages_r30.0", "wind_ages_r30.0", "hail_ages_r30.0",\
                    "tornado_ages_r45.0", "wind_ages_r45.0", "hail_ages_r45.0",\
                    "tornado_ages_r60.0", "wind_ages_r60.0", "hail_ages_r60.0"],\

        'PS3 Lead Time': ["tornado_leads", "wind_leads", "hail_leads",\
                        "tornado_leads_r15.0", "wind_leads_r15.0", "hail_leads_r15.0",\
                        "tornado_leads_r30.0", "wind_leads_r30.0", "hail_leads_r30.0",\
                        "tornado_leads_r45.0", "wind_leads_r45.0", "hail_leads_r45.0",\
                        "tornado_leads_r60.0", "wind_leads_r60.0", "hail_leads_r60.0"],\
        'PS3 Wind Probability': ['wind_raw_probs', 'wind_raw_probs_r15.0',\
                                    'wind_raw_probs_r30.0', 'wind_raw_probs_r45.0',\
                                    'wind_raw_probs_r60.0', 'wind_smoothed_probs',\
                                    'wind_smoothed_probs_r15.0', 'wind_smoothed_probs_r30.0',\
                                    'wind_smoothed_probs_r45.0', 'wind_smoothed_probs_r60.0'],\

        'PS3 Wind 14 Min. Change': ["wind_changes14", "wind_changes14_r15.0",\
                                      "wind_changes14_r30.0", "wind_changes14_r45.0",\
                                      "wind_changes14_r60.0"],\

        'PS3 Wind 30 Min. Change': ["wind_changes30", "wind_changes30_r15.0",\
                                        "wind_changes30_r30.0", "wind_changes30_r45.0",
                                        "wind_changes30_r60.0"],\

        'PS3 Hail Probability': ['hail_raw_probs', 'hail_raw_probs_r15.0',\
                                    'hail_raw_probs_r30.0', 'hail_raw_probs_r45.0',\
                                    'hail_raw_probs_r60.0', 'hail_smoothed_probs',\
                                    'hail_smoothed_probs_r15.0', 'hail_smoothed_probs_r30.0',\
                                    'hail_smoothed_probs_r45.0', 'hail_smoothed_probs_r60.0'],\

        'PS3 Hail 14 Min. Change': ["hail_changes14", "hail_changes14_r15.0",\
                                      "hail_changes14_r30.0", "hail_changes14_r45.0",\
                                      "hail_changes14_r60.0"],\

        'PS3 Hail 30 Min. Change': ["hail_changes30", "hail_changes30_r15.0",\
                                        "hail_changes30_r30.0", "hail_changes30_r45.0",
                                        "hail_changes30_r60.0"],\

        'Location (Lat/Lon)': ["lat", "lon"],\
        
        'TORP Probability': ['torp_prob', 'torp_prob_max_15km', 'torp_prob_max_30km', 'torp_prob_max_45km', 'torp_prob_max_60km'],\
        
        'TORP Age': ['torp_age', 'torp_age_max_15km', 'torp_age_max_30km', 'torp_age_max_45km', 'torp_age_max_60km'],\
        
        'TORP 5 Min Change': ['torp_p_change_5_min', 'torp_p_change_5_min_max_15km', 'torp_p_change_5_min_max_30km',\
                              'torp_p_change_5_min_max_45km', 'torp_p_change_5_min_max_60km'],\
                          
        'TORP 10 Min Change': ['torp_p_change_10_min', 'torp_p_change_10_min_max_15km', 'torp_p_change_10_min_max_30km',\
                              'torp_p_change_10_min_max_45km', 'torp_p_change_10_min_max_60km'],\
                         
        'AzShear Max': ['azshear_max', 'azshear_max_max_15km', 'azshear_max_max_30km', 'azshear_max_max_45km', 'azshear_max_max_60km'],\
                          
        'AzShear Max Trend': ['azshear_max_trend', 'azshear_max_trend_max_15km', 'azshear_max_trend_max_30km',\
                              'azshear_max_trend_max_45km', 'azshear_max_trend_max_60km'],\
                          
        'DivShear Min': ['divshear_min', 'divshear_min_min_15km', 'divshear_min_min_30km',\
                              'divshear_min_min_45km', 'divshear_min_min_60km'],\
                          
        'DivShear Min Trend': ['divshear_min_trend', 'divshear_min_trend_min_15km', 'divshear_min_trend_min_30km',\
                              'divshear_min_trend_min_45km', 'divshear_min_trend_min_60km'],\
                          
        'Reflectivity Max': ['reflectivity_max', 'reflectivity_max_max_15km', 'reflectivity_max_max_30km',\
                             'reflectivity_max_max_45km', 'reflectivity_max_max_60km'],\
                         
        'Reflectivity Min': ['reflectivity_min_min', 'reflectivity_min_min_15km', 'reflectivity_min_min_30km',\
                             'reflectivity_min_min_45km', 'reflectivity_min_min_60km'],\
                         
        'Spectrum Width Max': ['spectrumwidth_max', 'spectrumwidth_max_15km', 'spectrumwidth_max_30km',\
                               'spectrumwidth_max_45km', 'spectrumwidth_max_60km'],
                          
        'Spectrum Width Max Trend': ['spectrumwidth_max_trend', 'spectrumwidth_max_trend_15km', 'spectrumwidth_max_trend_30km',\
                               'spectrumwidth_max_trend_45km', 'spectrumwidth_max_trend_60km'],
                          
        'Max Absolute Velocity': ['absolute_velocity_max', 'absolute_velocity_max_max_15km', 'absolute_velocity_max_max_30km',\
                                  'absolute_velocity_max_max_45km', 'absolute_velocity_max_max_60km'],
                          
        'Max Absolute Velocity Trend': ['absolute_velocity_max_trend', 'absolute_velocity_max_trend_max_15km',\
                                        'absolute_velocity_max_trend_max_30km', 'absolute_velocity_max_trend_max_45km',\
                                        'absolute_velocity_max_trend_max_60km'],
                          
        'Rotational Velocity': ['vrot', 'vrot_max_15km', 'vrot_max_30km', 'vrot_max_45km', 'vrot_max_60km'],
                          
        'Rotational Velocity Distance': ['vrot_distance', 'vrot_distance_min_15km', 'vrot_distance_min_30km',\
                                        'vrot_distance_min_45km', 'vrot_distance_min_60km'],
                          
        'CC Min': ['cc_min', 'cc_min_15km', 'cc_min_30km', 'cc_min_45km', 'cc_min_60km'],\
                          
        'CC Max': ['cc_max', 'cc_max_15km', 'cc_max_30km', 'cc_max_45km', 'cc_max_60km'],\
                          
        'PhiDP Min': ['phidp_min', 'phidp_min_15km', 'phidp_min_30km', 'phidp_min_45km', 'phidp_min_60km'],\
                          
        'PhiDP Max': ['phidp_max', 'phidp_max_15km', 'phidp_max_30km', 'phidp_max_45km', 'phidp_max_60km'],\
                          
        'TORP Extrapolation': ['extrapolation_minutes', 'extrapolation_minutes_min_15km', 'extrapolation_minutes_min_30km',\
                               'extrapolation_minutes_min_45km', 'extrapolation_minutes_min_60km'],\
        'All WoFS': raw_variables[wofs_indices], 'All PS3': raw_variables[ps_indices], 'All TORP': raw_variables[torp_indices]
                         }

    grouped_colors_dict = {'80m Wind': wofs_color, '1km dBZ': wofs_color, '0-2km Vertical Vorticity': wofs_color,\
                                 '0-2km Updraft Helicity': wofs_color, '2-5km Updraft Helicity': wofs_color,\
                                 'Updraft Speed': wofs_color, '10-500m Bulk Shear': wofs_color,\
                                 'Flash Extent Density': wofs_color, '10m Wind': wofs_color, \
                                 '2m Temperature': wofs_color, '2m Dewpoint': wofs_color, \
                                 'Mid-level Lapse Rate': wofs_color, 'Low-level Lapse Rate': wofs_color,\
                                 '0-1km Shear Components': wofs_color, '0-3km Shear Components': wofs_color,\
                                 '0-6km Shear Components': wofs_color, 'SRH': wofs_color, 'SBCAPE': wofs_color,\
                                 'STP': wofs_color, 'SCP': wofs_color, 'Downdraft Speed': wofs_color, \
                                 'Cloud Top Temperature': wofs_color, 'Surface Pressure': wofs_color, \
                                 'LCL Height': wofs_color, 'Hailcast Hail': wofs_color, 'Freezing Level': wofs_color,\
                                 'WoFS probability of 40dBZ': wofs_color, 'In ICs': wofs_color, \
                                 'PS3 Tornado Probability': ps_color, 'PS3 Tornado 14 Min. Change': ps_color, \
                                 'PS3 Tornado 30 Min. Change': ps_color, 'PS3 Age': ps_color, 'PS3 Lead Time': ps_color,\
                                 'PS3 Wind Probability': ps_color, 'PS3 Wind 14 Min. Change': ps_color, \
                                 'PS3 Wind 30 Min. Change': ps_color, 'PS3 Hail Probability': ps_color,\
                                 'PS3 Hail 14 Min. Change': ps_color, 'PS3 Hail 30 Min. Change': ps_color,\
                                 'Location (Lat/Lon)': other_color, 'TORP Probability': torp_color,\
                          'TORP Age': torp_color, 'TORP 5 Min Change': torp_color,\
                           'TORP 10 Min Change': torp_color, 'AzShear Max': torp_color,\
                          'AzShear Max Trend': torp_color, 'DivShear Min': torp_color,\
                          'DivShear Min Trend': torp_color, 'Reflectivity Max': torp_color,\
                          'Reflectivity Min': torp_color, 'Spectrum Width Max': torp_color,\
                          'Spectrum Width Max Trend': torp_color, 'Max Absolute Velocity': torp_color,\
                          'Max Absolute Velocity Trend': torp_color, 'Rotational Velocity': torp_color,\
                          'Rotational Velocity Distance': torp_color, 'CC Min': torp_color,\
                          'CC Max': torp_color, 'PhiDP Min': torp_color, 'PhiDP Max': torp_color,\
                          'TORP Extrapolation': torp_color, 'All WoFS': wofs_color,\
                          'All PS3': ps_color, 'All TORP': torp_color}

    return hazard, radius, leads, model_type, titles, panel_labels, show_legends, csv_names,\
wofs_variables, ps_variables,torp_variables, use_plot_variables, wofs_color, ps_color,\
torp_color, other_color, grouped_names_dict, grouped_colors_dict

def make_ti_plot_contingency(hits, misses, false, cnegs, names, name_colors, title, showLegend, \
                                panelLabel, ax1):

    '''Makes the TI plot broken into contingency table elements. hits, misses, false, and cnegs
            are really the weighted hits, misses, false alarms, and correct negatives. Otherwise,
            everything else is the same as the original make_ti_plot method
    '''
    
    ys = np.arange(len(names))
    xs = np.array([0, 0.2, 0.4, 0.6, 0.8, 1]) 

    #Goal: Want to plot good_weights and bad_weights on top of each other 
    #Will plot good weights then bad weights 
  
    #Order: hits, cnegs, misses, false alarms

    #ax1.barh(ys, hits, color='royalblue', left=xs[0], label='Hits', edgecolor='black')
    #ax1.barh(ys, cnegs, color='lightsteelblue', left=xs[0]+hits, label='Correct Negatives', edgecolor='black')
    #ax1.barh(ys, misses, color='brown', left=xs[0]+hits+cnegs, label='Misses', edgecolor='black')
    #ax1.barh(ys, false, color='lightcoral', left=xs[0]+hits+cnegs+misses, label='False Alarms', edgecolor='black')
    
    ax1.barh(ys, hits, color='royalblue', left=xs[0], label='Hits', edgecolor='royalblue')
    ax1.barh(ys, cnegs, color='lightsteelblue', left=xs[0]+hits, label='Correct Negatives', edgecolor='lightsteelblue')
    ax1.barh(ys, misses, color='brown', left=xs[0]+hits+cnegs, label='Misses', edgecolor='brown')
    ax1.barh(ys, false, color='lightcoral', left=xs[0]+hits+cnegs+misses, label='False Alarms', edgecolor='lightcoral')


    ax1.set_yticks(ys)
    ax1.set_yticklabels(names, fontsize=7, fontweight='bold')
    ax1.invert_yaxis()
    ax1.set_xticks(xs)
    #ax1.set_xlabel("Avg. Fractional \nImpact on Probs.", fontsize=15, fontweight='bold') 
    #ax1.set_xlabel("Avg. Proportional \nContribution to Probs.", fontsize=15, fontweight='bold') 
    ax1.set_xlabel("RTI Value (Mean Proportional\nContribution to Probs.)", fontsize=15, fontweight='bold') 
    ax1.set_xticklabels(xs, fontsize=12)
    ax1.set_title(title, fontsize=18, fontweight='bold')
    ax1.set_xlim(xs[0]-0.0005, xs[-1]+0.0005)
    ax1.set_ylim(ys[-1]+0.5, ys[0]-0.5)

    #Change the label colors 
    for ytick, color in zip(ax1.get_yticklabels(), name_colors):
        ytick.set_color(color) 

    #Write the panel label 
    ax1.text(0.94, 0.9, panelLabel, horizontalalignment='center', verticalalignment='center',\
                color='black', fontsize=14, fontweight='bold', transform=ax1.transAxes) 

    #Add the legend
    if (showLegend == True):
        ax1.legend(loc='lower right', prop={'size': 6})
    
    
    
    return ax1


def make_ti_plot_contingency_dual_axes(hits, misses, false, cnegs, total, names, name_colors, title, showLegend, \
                                panelLabel, haz, ax1, differences = False):

    '''Makes the TI plot broken into contingency table elements. hits, misses, false, and cnegs
            are really the weighted hits, misses, false alarms, and correct negatives. 
            Additionally, hits and misses are plotted on a different axis than false alarms and
            correct negatives--that is the main difference from the make_ti_plot_contingency method. 
    '''
    
    ys = np.arange(len(names))
    xs = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
    #xs2 = np.linspace(0.0, 0.0007, 8) #xs for the twin axis 
    
    xs2 = np.array([0, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004])
    
    if not (haz == 'tornado'):
        xs2 = xs2*6
        
    if differences:
        xs2 = np.array([-0.01, -0.008, -0.006, -0.004, -0.002, 0, 0.002, 0.004, 0.006, 0.008, 0.01])
        xs = np.array([-0.1, -0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.1])

    #Hard
    #x2_labels = ['0.0', '1e-4', '2e-4', '3e-4', '4e-4', '5e-4', '6e-4', '7e-4'] 
    

    #Goal: Want to plot good_weights and bad_weights on top of each other 
    #Will plot good weights then bad weights 
  
    #Order: hits, cnegs, misses, false alarms

    #Create twin axis
    ax2 = ax1.twiny() 
   
    #Idea: Maybe still plot the original, but then also plot only the hits and misses on a separate axis but using 
    # a smaller bar height. See what that looks like. 
    
    if differences:
        #original:
        for i in range(len(names)):
            total_pos = 0
            total_neg = 0
            if i == 0:
                if hits[i] < 0:
                    ax1.barh(ys[i], hits[i], color='royalblue', left=0, label='Hits', edgecolor='royalblue')
                    total_neg += hits[i]
                else:
                    ax1.barh(ys[i], hits[i], color='royalblue', left=0, label='Hits', edgecolor='royalblue')
                    total_pos += hits[i]
                if cnegs[i] < 0:
                    ax1.barh(ys[i], cnegs[i], color='lightsteelblue', left=total_neg, label='Correct Negatives', edgecolor='lightsteelblue')
                    total_neg += cnegs[i]
                else:
                    ax1.barh(ys[i], cnegs[i], color='lightsteelblue', left=total_pos, label='Correct Negatives', edgecolor='lightsteelblue')
                    total_pos += cnegs[i]

                if misses[i] < 0:
                    ax1.barh(ys[i], misses[i], color='brown', left=total_neg, label='Misses', edgecolor='brown')
                    total_neg += misses[i]
                else:
                    ax1.barh(ys[i], misses[i], color='brown', left=total_neg, label='Misses', edgecolor='brown')
                    total_pos += misses[i]

                if false[i] < 0:
                    ax1.barh(ys[i], false[i], color='lightcoral', left=total_neg, label='False Alarms', edgecolor='lightcoral')
                    total_neg += misses[i]
                else:
                    ax1.barh(ys[i], false[i], color='lightcoral', left=total_neg, label='False Alarms', edgecolor='lightcoral')
                    total_pos += misses[i]
                ax1.barh(ys[i], total[i], color='k', left=0, label='Total', edgecolor='k', height = 0.4)
            else:
                if hits[i] < 0:
                    ax1.barh(ys[i], hits[i], color='royalblue', left=0, edgecolor='royalblue')
                    total_neg += hits[i]
                else:
                    ax1.barh(ys[i], hits[i], color='royalblue', left=0, edgecolor='royalblue')
                    total_pos += hits[i]
                if cnegs[i] < 0:
                    ax1.barh(ys[i], cnegs[i], color='lightsteelblue', left=total_neg, edgecolor='lightsteelblue')
                    total_neg += cnegs[i]
                else:
                    ax1.barh(ys[i], cnegs[i], color='lightsteelblue', left=total_pos, edgecolor='lightsteelblue')
                    total_pos += cnegs[i]

                if misses[i] < 0:
                    ax1.barh(ys[i], misses[i], color='brown', left=total_neg, edgecolor='brown')
                    total_neg += misses[i]
                else:
                    ax1.barh(ys[i], misses[i], color='brown', left=total_neg, edgecolor='brown')
                    total_pos += misses[i]

                if false[i] < 0:
                    ax1.barh(ys[i], false[i], color='lightcoral', left=total_neg, edgecolor='lightcoral')
                    total_neg += misses[i]
                else:
                    ax1.barh(ys[i], false[i], color='lightcoral', left=total_neg, edgecolor='lightcoral')
                    total_pos += misses[i]
                ax1.barh(ys[i], total[i], color='k', left=0, edgecolor='k', height = 0.4)

            #New on different axis
            total_neg = 0
            total_pos = 0
            
            if hits[i] < 0:
                ax2.barh(ys[i], hits[i], color='royalblue', left=total_neg, label='Hits', edgecolor='royalblue', height = 0.5)
                total_neg += hits[i]
            else:
                ax2.barh(ys[i], hits[i], color='royalblue', left=total_pos, label='Hits', edgecolor='royalblue', height = 0.5)
                total_pos += hits[i]
            
            if misses[i] < 0:
                ax2.barh(ys[i], misses[i], color='royalblue', left=total_neg, label='Misses', edgecolor='brown', height = 0.5)
                total_neg += misses[i]
            else:
                ax2.barh(ys[i], misses[i], color='brown', left=total_pos, label='Misses', edgecolor='brown', height = 0.5)
                total_pos += misses[i]
            
    else:
        #original: 
        ax1.barh(ys, hits, color='royalblue', left=xs[0], label='Hits', edgecolor='royalblue')
        ax1.barh(ys, cnegs, color='lightsteelblue', left=xs[0]+hits, label='Correct Negatives', edgecolor='lightsteelblue')
        ax1.barh(ys, misses, color='brown', left=xs[0]+hits+cnegs, label='Misses', edgecolor='brown')
        ax1.barh(ys, false, color='lightcoral', left=xs[0]+hits+cnegs+misses, label='False Alarms', edgecolor='lightcoral')

        #New on different axis
        ax2.barh(ys, hits, color='royalblue', left=xs2[0], label='Hits', edgecolor='royalblue', height=0.5)
        ax2.barh(ys, misses, color='brown', left=xs2[0]+hits, label='Misses', edgecolor='brown',height=0.5) 

    ax1.set_yticks(ys)
    if differences:
        ax1.set_yticklabels(names, fontsize=7, fontweight='bold')
    else:
        ax1.set_yticklabels(names, fontsize=7, fontweight='bold')
    ax1.invert_yaxis()
    ax1.set_xticks(xs)
    #ax1.set_xlabel("Avg. Fractional \nImpact on Probs.", fontsize=15, fontweight='bold') 
    #ax1.set_xlabel("Avg. Proportional \nContribution to Probs.", fontsize=15, fontweight='bold') 
    ax1.set_xlabel("RTI Value (Mean Proportional\nContribution to Probs.)", fontsize=15, fontweight='bold') 
    ax1.set_xticklabels(xs, fontsize=10)
    ax1.set_title(title, fontsize=20, fontweight='bold')
    if differences:
        ax1.plot([0, 0], [200, -200], linewidth = 0.5)
    ax1.set_xlim(xs[0]-0.0005, xs[-1]+0.0005)
    ax1.set_ylim(ys[-1]+0.5, ys[0]-0.5)

    ax2.set_xticks(xs2)
    #ax2.set_xticklabels(x2_labels, fontsize=12)
    ax2.set_xticklabels(xs2, fontsize=10)
    ax2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False) 
    #plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax2.set_xlim(xs2[0], xs2[-1])

    #Change the label colors
    for ytick, color in zip(ax1.get_yticklabels(), name_colors):
        ytick.set_color(color)

    #Write the panel label 
    ax1.text(0.94, 0.9, panelLabel, horizontalalignment='center', verticalalignment='center',\
                color='black', fontsize=14, fontweight='bold', transform=ax1.transAxes) 


    #Add the legend 
    if (showLegend == True):
        if differences:
            ax1.legend(loc='lower right', prop={'size': 10})
        else:
            ax1.legend(loc='lower right', prop={'size': 8})



    return ax1

#returns list of colors with wofs variables in one color, ps variables in
#another, and neither in a third 
def get_colors_for_names(names, wofs_list, ps_list, wofsColor, psColor, otherColor):

    color_list = [] 
    for name in names:
        if (name in wofs_list):
            color_list.append(wofsColor)
        elif (name in ps_list):
            color_list.append(psColor)
        else: 
            color_list.append(otherColor)

    return color_list

def assemble_groups(in_df, groups_dict): 
    #Converts the individual-predictor RTI dataframe into a grouped_predictor RTI dataframe,
    #where individual predictors in @in_df are aggregated based on the groups in 
    #@groups_dict
    #@in_df is the individual-predictor RTI dataframe
    #@groups_dict is the dictionary where the keys are the groups names and the values are
        #lists of individual predictors making up each group 

    df_list = [] #Will hold the different dfs, which will be combined later.
    
    for group_name, subset_vars in groups_dict.items(): 

        #Get a subset of data based on the variable (i.e., predictor) name 
        mask = in_df['variable'].isin(subset_vars)
        subset_df = in_df[mask]

        #Sum this along the rows
        sum_subset_df = subset_df.sum(axis=0)
        
        #Rename the variable 
        sum_subset_df['variable'] = group_name

        df_list.append(sum_subset_df)

    #Concatenate all dataframes in df_list
    final_df = pd.concat(df_list, axis=1)

    #Take the transpose so the metrics are the columns 
    final_df = final_df.T

    return final_df


def get_colors_from_dict(group_name_list, colors_dict):
    #Gets a list of colors for each group name in @group_name_list (list of group names in order)
    #based on @colors_dict, a dictionary of colors where the keys are the group names and the values
    #are the colors

    colors_list = [colors_dict[g] for g in group_name_list]

    return colors_list


#good_weights and bad_weights are np arrays giving the good/bad weight contribution, respectively
def make_ti_plot(good_weights, bad_weights, names, name_colors, title, showLegend, panelLabel, ax1):

    ys = np.arange(len(names))
    #xs = np.linspace(0.0, 0.50, 11) #Might change and/or pass in later
    #xs = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]) 
    #xs = np.linspace(0.0, 0.10, 11) 
    #xs = np.linspace(0.0, 0.06, 7) 
    xs = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])

    #Goal: Want to plot good_weights and bad_weights on top of each other 
    #Will plot good weights then bad weights 
   

    #First, plot the bad weights
    ax1.barh(ys, good_weights, color='blue', left=xs[0], label='Correct Sign', edgecolor='black') 
    #ax2.barh(ys[0], good_weights[0], color='blue', left=xs[0], label='Correct Sign', edgecolor='black') 
    
    #Next, plot the good weights on top
    ax1.barh(ys, bad_weights, color='red', left=good_weights, label='Incorrect Sign', edgecolor='black') 
    #ax2.barh(ys[0], bad_weights[0], color='red', left=good_weights, label='Incorrect Sign', edgecolor='black') 

    ax1.set_yticks(ys)
    ax1.set_yticklabels(names, fontsize=7, fontweight='bold')
    ax1.invert_yaxis()
    ax1.set_xticks(xs)
    #ax1.set_xlabel("Avg. Fractional \nImpact on Probs.", fontsize=15, fontweight='bold') 
    ax1.set_xlabel("Avg. Proportional \nContribution to Probs.", fontsize=15, fontweight='bold') 
    ax1.set_xticklabels(xs, fontsize=12)
    ax1.set_title(title, fontsize=20, fontweight='bold')
    ax1.set_xlim(xs[0]-0.0005, xs[-1]+0.0005)
    ax1.set_ylim(ys[-1]+0.5, ys[0]-0.5)

    #Change the label colors 
    for ytick, color in zip(ax1.get_yticklabels(), name_colors):
        ytick.set_color(color) 

    #Write the panel label 
    ax1.text(0.94, 0.9, panelLabel, horizontalalignment='center', verticalalignment='center',\
                color='black', fontsize=14, fontweight='bold', transform=ax1.transAxes) 

    #Add the legend 
    if (showLegend == True):
        ax1.legend(loc='lower right', fontsize=12) 

    return ax1

def make_comparison_ti_plot(rti_vals1, rti_vals2, names, name_colors, title,\
                            showLegend, panelLabel, ax1):

    ys = np.arange(len(names))
    #xs = np.linspace(0.0, 0.50, 11) #Might change and/or pass in later
    #xs = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]) 
    #xs = np.linspace(0.0, 0.10, 11) 
    #xs = np.linspace(0.0, 0.06, 7) 
    xs = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])

    #Goal: Want to plot good_weights and bad_weights on top of each other 
    #Will plot good weights then bad weights 
   

    #First, plot the bad weights
    ax1.barh(ys-0.15, rti_vals1, color='blue', left=xs[0], label='Warnings', edgecolor='black', height = 0.3) 
    #ax2.barh(ys[0], good_weights[0], color='blue', left=xs[0], label='Correct Sign', edgecolor='black') 
    
    #Next, plot the good weights on top
    ax1.barh(ys+0.15, rti_vals2, color='yellow', left=xs[0], label='LSRs', edgecolor='black', height = 0.3) 
    #ax2.barh(ys[0], bad_weights[0], color='red', left=good_weights, label='Incorrect Sign', edgecolor='black') 

    ax1.set_yticks(ys)
    ax1.set_yticklabels(names, fontsize=7, fontweight='bold')
    ax1.invert_yaxis()
    ax1.set_xticks(xs)
    #ax1.set_xlabel("Avg. Fractional \nImpact on Probs.", fontsize=15, fontweight='bold') 
    ax1.set_xlabel("Avg. Proportional \nContribution to Probs.", fontsize=15, fontweight='bold') 
    ax1.set_xticklabels(xs, fontsize=12)
    ax1.set_title(title, fontsize=20, fontweight='bold')
    ax1.set_xlim(xs[0]-0.0005, xs[-1]+0.0005)
    ax1.set_ylim(ys[-1]+0.5, ys[0]-0.5)

    #Change the label colors 
    for ytick, color in zip(ax1.get_yticklabels(), name_colors):
        ytick.set_color(color) 

    #Write the panel label 
    ax1.text(0.03, 1.05, panelLabel, horizontalalignment='center', verticalalignment='center',\
                color='black', fontsize=14, fontweight='bold', transform=ax1.transAxes)

    #Add the legend 
    if (showLegend == True):
        ax1.legend(loc='lower right', fontsize=12) 

    return ax1

#returns sorted data array by given variable
#Only return first ntop variables 
def sort_ti(in_df, sort_var, ntop):

    return in_df.sort_values(by=sort_var, ascending=False).iloc[0:ntop]

def sort_ti_abs_val(in_df, sort_var, ntop):
    
    return in_df.sort_values(by=sort_var, key = abs, ascending=False).iloc[0:ntop]

#returns list of colors with wofs variables in one color, ps variables in
#another, and neither in a third 
def get_colors_for_names(names, wofs_list, ps_list, torp_list, wofsColor, psColor, torpColor, otherColor):

    color_list = []
    var_df = pd.read_csv('/work/ryan.martz/wofs_phi_data/experiments/vars_dict.csv')
    var_dict = dict(zip(var_df.Variable, var_df.Plain_Language_Variable))
    for raw_name in names:
        name = var_dict[raw_name]
        if (name in wofs_list):
            color_list.append(wofsColor)
        elif (name in ps_list):
            color_list.append(psColor)
        elif (name in torp_list):
            color_list.append(torpColor)
        else: 
            color_list.append(otherColor)

    return color_list

def do_plotting(ps_version, include_torp_in_predictors, radar_data, filtered_torp, hazard, radius, train_type,\
                length, plot_type, n_top, var_agg_tag):
    
    try:
        hazard, radius, leads, model, titles, panel_labels, show_legends, csv_names, wofs_variables, ps_variables, \
        torp_variables, plot_variables, wofs_color, ps_color, torp_color, other_color, grouped_names_dict,\
        grouped_colors_dict = initialize(ps_version, include_torp_in_predictors, radar_data, filtered_torp,\
                                         hazard, radius, train_type, length)
    except:
        result = initialize(ps_version, include_torp_in_predictors, radar_data, filtered_torp, hazard, radius, train_type, length)
        if result == '':
            print('bad init')
            return
    
    fig = plt.figure(figsize=(12,12), facecolor='whitesmoke')
    
    file_exists = False
    for l in range(len(leads)):
        csv_name = csv_names[l]
        title = titles[l]
        panel_label = panel_labels[l] 
        show_legend = show_legends[l]
        ax = fig.add_subplot(3,2,l+1)

        #Read CSV file 
        #Probably due to rounding error. Probably will need to read in np.float32 
        if not os.path.exists(csv_name):
            continue
        file_exists = True
        data = pd.read_csv(csv_name) 

        #Rename first column 
        data.rename(columns={"Unnamed: 0": "variable"}, inplace=True)

        #Add column of plot variables
        #data['plot_variable'] = plot_variables
        
        data = assemble_groups(data, grouped_names_dict)

        #data['new_overall'] = data['good'] + data['bad'] #Not sure this is the best solution, but it does guarantee good and bad will sum to 1
        data['new_overall'] = data['weighted_hits'] + data['weighted_misses'] + data['weighted_false'] + data['weighted_cnegs']

        #Get sorted data
        if plot_type == 'regular':
            sort_data = sort_ti(data, 'new_overall', n_top)
        elif plot_type == 'hits_misses':
            sort_data = sort_ti(data, 'weighted_hits', n_top)
        
        #plot_names = sort_data['plot_variable'].to_list()
        plot_names = sort_data['variable']

        #Get list of colors corresponding to names 
        #colors_for_names = get_colors_for_names(sort_data['variable'].to_list(), wofs_variables, \
        #                        ps_variables, torp_variables, wofs_color, ps_color, torp_color, other_color)
        colors_for_names = get_colors_from_dict(sort_data['variable'].to_list(), grouped_colors_dict)
        
        if plot_type == 'regular':
            ax = make_ti_plot_contingency_dual_axes(sort_data['weighted_hits'].to_numpy(), sort_data['weighted_misses'].to_numpy(),\
                        sort_data['weighted_false'].to_numpy(), sort_data['weighted_cnegs'].to_numpy(), sort_data['new_overall'].to_numpy(), \
                        plot_names, colors_for_names, title, \
                        show_legend, panel_label, hazard, ax)
            fig_save_file = 'aggregate_ti_data_%s_%s_%s_trained_r%skm%s.png'\
            %(model, hazard, train_type, radius, var_agg_tag)
        elif plot_type == 'hits_misses':
            ax = make_ti_plot_contingency_dual_axes(sort_data['weighted_hits'].to_numpy(), sort_data['weighted_misses'].to_numpy(),\
                        sort_data['weighted_false'].to_numpy(), sort_data['weighted_cnegs'].to_numpy(), sort_data['new_overall'].to_numpy(), \
                        plot_names, colors_for_names, title, \
                        show_legend, panel_label, hazard, ax)
            fig_save_file = 'hit_miss_aggregate_ti_data_%s_%s_%s_trained_r%skm%s.png'\
            %(model, hazard, train_type, radius, var_agg_tag)


        #Only label x axis if it's the last 2 subplots 
        if (l < 4):
            ax.set_xlabel("")
    
    if not file_exists:
        return
    fig.subplots_adjust(wspace=0.65, hspace=0.45)
    #fig.savefig("new_overall_ti_barplots_%s_r%skm_multipanel.png" %(hazard, radius), bbox_inches='tight', dpi=300)
    #fig.savefig("test_barplots_contingency_new.png", bbox_inches='tight', dpi=300)
    #fig.savefig("new_overall_ti_barplots_%s_r%skm_multipanel_contingency.png" %(hazard, radius), bbox_inches='tight', dpi=300)
    fig.suptitle('%s' %(hazard.capitalize()), fontsize = 30, fontweight = 'bold')
    fig.tight_layout()
    fig_save_dir = '/work/ryan.martz/wofs_phi_data/experiments/%s_trained/%s/length_%s/tree_interpreter/%s'\
    %(model, train_type, length, hazard)
    fig.savefig("%s/%s" %(fig_save_dir, fig_save_file), bbox_inches='tight', dpi=300)
    return

def do_plotting_gifs(ps_version, include_torp_in_predictors, radar_data, filtered_torp, hazard, radius, train_type,\
                length, plot_type, n_top, var_agg_tag):
    
    try:
        hazard, radius, leads, model, titles, panel_labels, show_legends, csv_names, wofs_variables, ps_variables, \
        torp_variables, plot_variables, wofs_color, ps_color, torp_color, other_color, grouped_names_dict,\
        grouped_colors_dict = initialize(ps_version, include_torp_in_predictors, radar_data, filtered_torp,\
                                         hazard, radius, train_type, length)
    except:
        result = initialize(ps_version, include_torp_in_predictors, radar_data, filtered_torp, hazard, radius, train_type, length)
        if result == '':
            return
    
    #leads = [30, 60, 90, 120]#, 150, 180]
    #titles = ["0-60 Minutes", "30-90 Minutes", "60-120 Minutes",\
    #          "90-150 Minutes"]#, "120-180 Minutes", "150-210 Minutes"]
    
    file_exists = False
    gif_files = []
    for l in range(len(leads)):
        lead = leads[l]
        fig, ax = plt.subplots(1,1,figsize=(6,6), facecolor='whitesmoke')
        csv_name = csv_names[l]
        title = titles[l]
        panel_label = ''
        show_legend = True

        #Read CSV file 
        #Probably due to rounding error. Probably will need to read in np.float32 
        if not os.path.exists(csv_name):
            continue
        file_exists = True
        data = pd.read_csv(csv_name) 

        #Rename first column 
        data.rename(columns={"Unnamed: 0": "variable"}, inplace=True)

        #Add column of plot variables
        #data['plot_variable'] = plot_variables
        
        data = assemble_groups(data, grouped_names_dict)

        #data['new_overall'] = data['good'] + data['bad'] #Not sure this is the best solution, but it does guarantee good and bad will sum to 1
        data['new_overall'] = data['weighted_hits'] + data['weighted_misses'] + data['weighted_false'] + data['weighted_cnegs']

        #Get sorted data
        if plot_type == 'regular':
            sort_data = sort_ti(data, 'new_overall', n_top)
        elif plot_type == 'hits_misses':
            sort_data = sort_ti(data, 'weighted_hits', n_top)
        
        #plot_names = sort_data['plot_variable'].to_list()
        plot_names = sort_data['variable']

        #Get list of colors corresponding to names 
        #colors_for_names = get_colors_for_names(sort_data['variable'].to_list(), wofs_variables, \
        #                        ps_variables, torp_variables, wofs_color, ps_color, torp_color, other_color)
        colors_for_names = get_colors_from_dict(sort_data['variable'].to_list(), grouped_colors_dict)
        
        if plot_type == 'regular':
            ax = make_ti_plot_contingency_dual_axes(sort_data['weighted_hits'].to_numpy(), sort_data['weighted_misses'].to_numpy(),\
                        sort_data['weighted_false'].to_numpy(), sort_data['weighted_cnegs'].to_numpy(), sort_data['new_overall'].to_numpy(), \
                        plot_names, colors_for_names, title, \
                        show_legend, panel_label, hazard, ax)
            fig_save_file = 'aggregate_ti_data_%s_%s_%s_trained_r%skm%s.png'\
            %(model, hazard, train_type, radius, var_agg_tag)
        elif plot_type == 'hits_misses':
            ax = make_ti_plot_contingency_dual_axes(sort_data['weighted_hits'].to_numpy(), sort_data['weighted_misses'].to_numpy(),\
                        sort_data['weighted_false'].to_numpy(), sort_data['weighted_cnegs'].to_numpy(), sort_data['new_overall'].to_numpy(), \
                        plot_names, colors_for_names, title, \
                        show_legend, panel_label, hazard, ax)
            fig_save_file = 'hit_miss_aggregate_ti_data_%s-%smin_%s_%s_%s_trained_r%skm%s.png'\
            %(lead-30, lead+length-30, model, hazard, train_type, radius, var_agg_tag)
            if length == 120:
                gif_save_file = 'hit_miss_aggregate_ti_data_%s_%s_%s_trained_r%skm%s_2hr.gif'\
                %(model, hazard, train_type, radius, var_agg_tag)
            else:
                gif_save_file = 'hit_miss_aggregate_ti_data_%s_%s_%s_trained_r%skm%s.gif'\
                %(model, hazard, train_type, radius, var_agg_tag)


        if not file_exists:
            return
        fig.suptitle('%s' %(hazard.capitalize()), fontsize = 30, fontweight = 'bold')
        fig.tight_layout()
        fig_save_dir = '/work/ryan.martz/wofs_phi_data/experiments/Thesis_Figs/RTI_GIFs'
        fig.savefig("%s/%s" %(fig_save_dir, fig_save_file), bbox_inches='tight')
        gif_files.append("%s/%s" %(fig_save_dir, fig_save_file))
    
    images = []
    for filename in gif_files:
        images.append(iio.imread(filename))
    iio.imwrite('%s/%s' %(fig_save_dir, gif_save_file), images, duration = 1500, loop = 0)
    
    return

def do_plotting_differences(ps_versions, include_torp_in_predictors,\
                            radar_datas, filtered_torps, hazard, radii,\
                            train_types, lengths, plot_type, n_top, var_agg_tag):
    
    try:
        hazard, radius1, leads, model1, titles, panel_labels, show_legends, csv_names1, wofs_variables, ps_variables, \
        torp_variables, plot_variables, wofs_color, ps_color, torp_color, other_color, grouped_names_dict,\
        grouped_colors_dict = initialize(ps_versions[0], include_torp_in_predictors[0], radar_datas[0], filtered_torps[0],\
                                         hazard, radii[0], train_types[0], lengths[0])
        
        hazard, radius2, leads, model2, titles, panel_labels, show_legends, csv_names2, wofs_variables, ps_variables, \
        torp_variables, plot_variables, wofs_color, ps_color, torp_color, other_color, grouped_names_dict,\
        grouped_colors_dict = initialize(ps_versions[1], include_torp_in_predictors[1], radar_datas[1], filtered_torps[1],\
                                         hazard, radii[1], train_types[1], lengths[1])
    except:
        result = initialize(ps_versions, include_torp_in_predictors, radar_datas, filtered_torps, hazard, radii, train_types, lengths)
        if result == '':
            print(result)
            return
        else:
            print(result)
            return
    
    fig = plt.figure(figsize=(12,12), facecolor='whitesmoke')
    
    file_exists = False
    for l in range(len(leads)):
        csv_name1 = csv_names1[l]
        csv_name2 = csv_names2[l]
        title = titles[l]
        panel_label = panel_labels[l] 
        show_legend = show_legends[l]
        ax = fig.add_subplot(3,2,l+1)

        #Read CSV file 
        #Probably due to rounding error. Probably will need to read in np.float32 
        if not (os.path.exists(csv_name1) and os.path.exists(csv_name2)):
            continue
        file_exists = True
        data1 = pd.read_csv(csv_name1)
        data2 = pd.read_csv(csv_name2)
        
        #Rename first column 
        data1.rename(columns={"Unnamed: 0": "variable"}, inplace=True)
        data2.rename(columns={"Unnamed: 0": "variable"}, inplace=True)

        #Add column of plot variables
        #data['plot_variable'] = plot_variables
        
        data1 = assemble_groups(data1, grouped_names_dict)
        data1['new_overall'] = data1['weighted_hits'] + data1['weighted_misses'] + data1['weighted_false'] + data1['weighted_cnegs']
        data2 = assemble_groups(data2, grouped_names_dict)
        data2['new_overall'] = data2['weighted_hits'] + data2['weighted_misses'] + data2['weighted_false'] + data2['weighted_cnegs']
        
        data = pd.DataFrame()
        data['new_overall'] = data1['new_overall'] - data2['new_overall']
        data['weighted_hits'] = data1['weighted_hits'] - data2['weighted_hits']
        data['weighted_misses'] = data1['weighted_misses'] - data2['weighted_misses']
        data['weighted_false'] = data1['weighted_false'] - data2['weighted_false']
        data['weighted_cnegs'] = data1['weighted_cnegs'] - data2['weighted_cnegs']
        
        data['variable'] = data1['variable']
        
        #Get sorted data
        if plot_type == 'regular':
            sort_data = sort_ti_abs_val(data, 'new_overall', n_top)
        elif plot_type == 'hits_misses':
            sort_data = sort_ti_abs_val(data, 'weighted_hits', n_top)
        
        #plot_names = sort_data['plot_variable'].to_list()
        plot_names = sort_data['variable']

        #Get list of colors corresponding to names 
        #colors_for_names = get_colors_for_names(sort_data['variable'].to_list(), wofs_variables, \
        #                        ps_variables, torp_variables, wofs_color, ps_color, torp_color, other_color)
        colors_for_names = get_colors_from_dict(sort_data['variable'].to_list(), grouped_colors_dict)
        
        if plot_type == 'regular':
            ax = make_ti_plot_contingency_dual_axes(sort_data['weighted_hits'].to_numpy(), sort_data['weighted_misses'].to_numpy(),\
                        sort_data['weighted_false'].to_numpy(), sort_data['weighted_cnegs'].to_numpy(), sort_data['new_overall'].to_numpy(),\
                        plot_names, colors_for_names, title, show_legend, panel_label, hazard, ax, True)
            fig_save_file = 'aggregate_ti_data_%s_%s_%s_r%skm_trained_minus_%s_%s_r%skm_trained%s.png'\
            %(hazard, model1, train_types[0], radius1, model2, train_types[1], radius2, var_agg_tag)
        elif plot_type == 'hits_misses':
            ax = make_ti_plot_contingency_dual_axes(sort_data['weighted_hits'].to_numpy(), sort_data['weighted_misses'].to_numpy(),\
                        sort_data['weighted_false'].to_numpy(), sort_data['weighted_cnegs'].to_numpy(), sort_data['new_overall'].to_numpy(),\
                        plot_names, colors_for_names, title, show_legend, panel_label, hazard, ax, True)
            fig_save_file = 'hit_miss_aggregate_ti_data_%s_%s_%s_r%skm_trained_minus_%s_%s_r%skm_trained%s.png'\
            %(hazard, model1, train_types[0], radius1, model2, train_types[1], radius2, var_agg_tag)


        #Only label x axis if it's the last 2 subplots 
        if (l < 4):
            ax.set_xlabel("")
    
    if not file_exists:
        return
    fig.subplots_adjust(wspace=0.65, hspace=0.45)
    #fig.savefig("new_overall_ti_barplots_%s_r%skm_multipanel.png" %(hazard, radius), bbox_inches='tight', dpi=300)
    #fig.savefig("test_barplots_contingency_new.png", bbox_inches='tight', dpi=300)
    #fig.savefig("new_overall_ti_barplots_%s_r%skm_multipanel_contingency.png" %(hazard, radius), bbox_inches='tight', dpi=300)
    fig_save_dir = '/work/ryan.martz/wofs_phi_data/experiments/%s_trained/%s/length_%s/tree_interpreter/%s'\
    %(model1, train_types[0], lengths[0], hazard)
    fig.suptitle('%s' %(hazard.capitalize()), fontsize = 24, fontweight = 'bold')
    fig.tight_layout()
    fig.savefig("%s/%s" %(fig_save_dir, fig_save_file), bbox_inches='tight', dpi=300)
    
    return

def do_plotting_lsr_warning_comparison(ps_versions, include_torp_in_predictors,\
                                       radar_datas, filtered_torps, hazard,\
                                       lengths, n_top):
    
    if hazard == 'hail' or hazard == 'tornado':
        radius = 39
    else:
        radius = 375
    
    try:
        hazard, radius, leads, model1, titles, panel_labels, show_legends, warning_csv_names, wofs_variables, ps_variables, \
        torp_variables, plot_variables, wofs_color, ps_color, torp_color, other_color, grouped_names_dict,\
        grouped_colors_dict = initialize(ps_versions[0], include_torp_in_predictors[0], radar_datas[0], filtered_torps[0],\
                                         hazard, radius, 'warnings', lengths[0])
        
        hazard, radius, leads, model2, titles, panel_labels, show_legends, obs_csv_names, wofs_variables, ps_variables, \
        torp_variables, plot_variables, wofs_color, ps_color, torp_color, other_color, grouped_names_dict,\
        grouped_colors_dict = initialize(ps_versions[1], include_torp_in_predictors[1], radar_datas[1], filtered_torps[1],\
                                         hazard, radius, 'obs', lengths[1])
    except:
        result = initialize(ps_versions, include_torp_in_predictors, radar_datas, filtered_torps, hazard, radii, train_types, lengths)
        if result == '':
            print(result)
            return
        else:
            print(result)
            return
    
    fig = plt.figure(figsize=(12,12), facecolor='whitesmoke')
    
    file_exists = False
    for l in range(len(leads)):
        warning_csv_name = warning_csv_names[l]
        obs_csv_name = obs_csv_names[l]
        title = titles[l]
        panel_label = panel_labels[l] 
        show_legend = show_legends[l]
        ax = fig.add_subplot(3,2,l+1)

        #Read CSV file 
        #Probably due to rounding error. Probably will need to read in np.float32 
        if not (os.path.exists(warning_csv_name) and os.path.exists(obs_csv_name)):
            continue
        file_exists = True
        warning_rti = pd.read_csv(warning_csv_name)
        obs_rti = pd.read_csv(obs_csv_name)
        
        #Rename first column 
        warning_rti.rename(columns={"Unnamed: 0": "variable"}, inplace=True)
        obs_rti.rename(columns={"Unnamed: 0": "variable"}, inplace=True)

        #Add column of plot variables
        #data['plot_variable'] = plot_variables
        
        warning_rti = assemble_groups(warning_rti, grouped_names_dict)
        warning_rti['new_overall'] = warning_rti['weighted_hits'] +\
        warning_rti['weighted_misses'] + warning_rti['weighted_false'] +\
        warning_rti['weighted_cnegs']
        
        obs_rti = assemble_groups(obs_rti, grouped_names_dict)
        obs_rti['new_overall'] = obs_rti['weighted_hits'] +\
        obs_rti['weighted_misses'] + obs_rti['weighted_false'] +\
        obs_rti['weighted_cnegs']
        
        sorted_warning_rti = sort_ti_abs_val(warning_rti, 'new_overall', np.array(warning_rti['new_overall']).shape[0])
        warning_vars = np.array(sorted_warning_rti['variable'])
        warning_rti_vals = np.array(sorted_warning_rti['new_overall'])
        
        sorted_obs_rti = sort_ti_abs_val(obs_rti, 'new_overall', np.array(obs_rti['new_overall']).shape[0])
        obs_vars = np.array(sorted_obs_rti['variable'])
        obs_rti_vals = np.array(sorted_obs_rti['new_overall'])
        
        rti_vars = []
        plot_warning_rti_vals = np.zeros((n_top,))
        plot_obs_rti_vals = np.zeros((n_top,))
        
        for i in range(n_top):
            if warning_rti_vals[0] >= obs_rti_vals[0]:
                rti_vars.append(warning_vars[0])
                plot_warning_rti_vals[i] = warning_rti_vals[0]
                
                obs_index = np.where(obs_vars == warning_vars[0])[0][0]
                plot_obs_rti_vals[i] = obs_rti_vals[obs_index]
                
                warning_vars = np.delete(warning_vars, 0)
                warning_rti_vals = np.delete(warning_rti_vals, 0)
                obs_vars = np.delete(obs_vars, obs_index)
                obs_rti_vals = np.delete(obs_rti_vals, obs_index)
            else:
                rti_vars.append(obs_vars[0])
                plot_obs_rti_vals[i] = obs_rti_vals[0]
                
                warning_index = np.where(warning_vars == obs_vars[0])[0][0]
                plot_warning_rti_vals[i] = warning_rti_vals[warning_index]
                
                obs_vars = np.delete(obs_vars, 0)
                obs_rti_vals = np.delete(obs_rti_vals, 0)
                warning_vars = np.delete(warning_vars, warning_index)
                warning_rti_vals = np.delete(warning_rti_vals, warning_index)

        #Get list of colors corresponding to names 
        #colors_for_names = get_colors_for_names(sort_data['variable'].to_list(), wofs_variables, \
        #                        ps_variables, torp_variables, wofs_color, ps_color, torp_color, other_color)
        colors_for_names = get_colors_from_dict(rti_vars, grouped_colors_dict)
        
        ax = make_comparison_ti_plot(plot_warning_rti_vals, plot_obs_rti_vals,\
                                     rti_vars, colors_for_names, title,\
                                     show_legend, panel_label, ax)
        fig_save_file = 'aggregate_ti_data_%s_%s_warnings_trained_vs_obs_r%skm_trained.png'\
        %(model1, hazard, radius)

        #Only label x axis if it's the last 2 subplots 
        if (l < 4):
            ax.set_xlabel("")
    
    if not file_exists:
        return
    fig.subplots_adjust(wspace=0.65, hspace=0.45)
    #fig.savefig("new_overall_ti_barplots_%s_r%skm_multipanel.png" %(hazard, radius), bbox_inches='tight', dpi=300)
    #fig.savefig("test_barplots_contingency_new.png", bbox_inches='tight', dpi=300)
    #fig.savefig("new_overall_ti_barplots_%s_r%skm_multipanel_contingency.png" %(hazard, radius), bbox_inches='tight', dpi=300)
    fig_save_dir = '/work/ryan.martz/wofs_phi_data/experiments/Paper_Figs/RTI_Plots'
    fig.suptitle('%s' %(hazard.capitalize()), fontsize = 24, fontweight = 'bold')
    fig.tight_layout()
    fig.savefig("%s/%s" %(fig_save_dir, fig_save_file), bbox_inches='tight', dpi=300)
    
    return

def main():
    
    hazards = ['hail', 'tornado', 'wind']
    all_train_types = ['warnings', 'obs']
    all_ps_versions = [3, 3]#[0, 2, 3]
    all_lengths = [60, 60]#, 120]
    all_include_torp_in_predictors = [True, True]#, False]
    all_radar_datas = [True, True]#, False]
    all_filtered_torps = [False, False]
    #plot_types = ['regular', 'hits_misses']#'regular', 'hits_misses']
    n_top = 20
    var_agg_tags = ['_aggregated_vars']
    
    iterator = md.to_iterator([all_ps_versions], [all_include_torp_in_predictors], [all_radar_datas],\
                              [all_filtered_torps], hazards, [all_lengths], [n_top])
    results = md.run_parallel(do_plotting_lsr_warning_comparison, iterator, nprocs_to_use = 3,\
                                           description = 'Plotting TI Comparisons')
    
    
    #all_radiis = [39]
    #all_ps_versions = [3]
    #all_lengths = [60]
    #all_include_torp_in_predictors = [True]
    #all_radar_datas = [True]
    #all_filtered_torps = [False]
    #
    #iterator = md.to_iterator(all_ps_versions, all_include_torp_in_predictors, all_radar_datas,\
    #                          all_filtered_torps, hazards, all_radiis, all_train_types, all_lengths,\
    #                          plot_types, [n_top], var_agg_tags)
    #results = md.run_parallel(do_plotting, iterator, nprocs_to_use = 20,\
    #                                       description = 'Plotting TI regular')
    
    #ps_versions = [2]#, 2]
    #include_torp_in_predictors = [False]#, True]
    #radar_datas = [True]#, True]
    #filtered_torps = [False]#, False]
    #lengths = [60]#, 60]
    #train_types = ['obs']#['obs_and_warnings', 'obs']
    #radii = [15, 39]
    #var_agg_tag = '_aggregated_vars'
    #n_top = 10
    # 
    #for hazard in hazards:
    #    for plot_type in plot_types:
    #        do_plotting_differences(ps_versions, include_torp_in_predictors, radar_datas,\
    #                                filtered_torps, hazard, radii, train_types, lengths,\
    #                                plot_type, n_top, var_agg_tag)
    
    
    return 



if (__name__ == '__main__'):

    main()


