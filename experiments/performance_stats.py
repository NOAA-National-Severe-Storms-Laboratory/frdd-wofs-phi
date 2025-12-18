import numpy as np
import sys
sys.path.insert(0, '../wofs_phi')
from wofs_phi import config as c
from wofs_phi import utilities
from sklearn.metrics import brier_score_loss as BS
import os
from shutil import copy
import time
import datetime as dt
import random
from wofs_phi import multiprocessing_driver as md

class model_stats:
    
    def __init__(self, hazard, wofs_spinup_time, forecast_length, wofs_lead_time, train_radius,\
                 ver_radius, train_type, ver_type, model_type, n_folds, use_avg_srs, use_any_srs,\
                 dates, buffer_str):
        self.hazard = hazard
        self.wofs_spinup_time = wofs_spinup_time
        self.forecast_length = forecast_length
        self.train_radius = train_radius
        self.ver_radius = ver_radius
        self.lead_time = wofs_lead_time
        self.train_type = train_type
        self.ver_type = ver_type
        self.train_test_dir = '/work/ryan.martz/wofs_phi_data/%s_train/test_fcsts/%s' %(train_type, model_type)
        self.ver_test_dir = '/work/ryan.martz/wofs_phi_data/%s_train/test_fcsts/%s' %(ver_type, model_type)
        self.model_type = model_type
        self.n_folds = n_folds
        self.dates = dates
        self.use_avg_srs = use_avg_srs
        self.use_any_srs = use_any_srs
        self.buffer_str = buffer_str
    
    def get_all_test_sr_events_fname_dir(self):
        start_min = self.lead_time
        end_min = self.lead_time + self.forecast_length
        train_save_dir = '%s/%s/wofslag_%s/length_%s'\
        %(self.train_test_dir, self.hazard, self.wofs_spinup_time, self.forecast_length)
        ver_save_dir = '%s/%s/wofslag_%s/length_%s'\
        %(self.ver_test_dir, self.hazard, self.wofs_spinup_time, self.forecast_length)
        if self.use_avg_srs:
            all_srs_fname = '%s_%s_r%skm_trained_all_%s_avg_sr_probs_%s-%smin.npy'\
            %(self.model_type, self.train_type, self.train_radius, self.hazard, start_min, end_min)
        else:
            all_srs_fname = '%s_%s_r%skm_trained_all_%s_sr_probs_%s-%smin.npy'\
            %(self.model_type, self.train_type, self.train_radius, self.hazard, start_min, end_min)
        all_probs_fname = '%s_%s_r%skm_trained_%s_r%skm_verified_all_%s_test_raw_probs_%s-%smin.npy'\
        %(self.model_type, self.train_type, self.train_radius, self.ver_type, self.ver_radius,\
          self.hazard, start_min, end_min)
        all_events_fname = '%s_all_%s_%s_r%skm_trained_%s_r%skm_verified_%s-%smin.npy'\
        %(self.model_type, self.hazard, self.train_type, self.train_radius, self.ver_type, self.ver_radius,\
          self.lead_time, self.lead_time + self.forecast_length)
        
        if self.use_any_srs:
            return train_save_dir, ver_save_dir, all_srs_fname, all_probs_fname, all_events_fname
        else:
            return train_save_dir, ver_save_dir, all_probs_fname, all_events_fname
    
    def get_all_test_sr_events_fname_dir_by_fold(self, fold):
        start_min = self.lead_time
        end_min = self.lead_time + self.forecast_length
        train_save_dir = '%s/%s/wofslag_%s/length_%s/all_raw_probs_fold%s'\
        %(self.train_test_dir, self.hazard, self.wofs_spinup_time,\
          self.forecast_length, fold)
        ver_save_dir = '%s/%s/wofslag_%s/length_%s/all_raw_probs_fold%s'\
        %(self.ver_test_dir, self.hazard, self.wofs_spinup_time,\
          self.forecast_length, fold)
        if self.train_type == 'warnings':
            if self.use_any_srs:
                if self.use_avg_srs:
                    all_srs_fname = '%s_%s_trained_all_%s_avg_sr_probs_%s-%smin_fold%s.npy'\
                    %(self.model_type, self.train_type, self.hazard, start_min, end_min, fold)
                else:
                    all_srs_fname = '%s_%s_trained_all_%s_sr_probs_%s-%smin_fold%s.npy'\
                    %(self.model_type, self.train_type, self.hazard, start_min, end_min, fold)
            all_probs_fname = '%s_%s_trained_all_rf_%s_raw_probs_spinup%smin_length%s_min%s-%s_fold%s.npy'\
            %(self.model_type, self.train_type, self.hazard, self.wofs_spinup_time, self.forecast_length,\
              start_min, end_min, fold)
        else:
            if self.use_any_srs:
                if self.use_avg_srs:
                    all_srs_fname = '%s_%s_r%skm_trained_all_%s_avg_sr_probs_%s-%smin_fold%s.npy'\
                    %(self.model_type, self.train_type, self.train_radius, self.hazard,\
                      start_min, end_min, fold)
                else:
                    all_srs_fname = '%s_%s_r%skm_trained_all_%s_sr_probs_%s-%smin_fold%s.npy'\
                    %(self.model_type, self.train_type, self.train_radius, self.hazard,\
                      start_min, end_min, fold)
            all_probs_fname = '%s_%s_trained_all_rf_%s_raw_probs_spinup%smin_length%s_min%s-%s_r%skm_fold%s.npy'\
            %(self.model_type, self.train_type, self.hazard, self.wofs_spinup_time, self.forecast_length,\
              start_min, end_min, self.train_radius, fold)
        if self.ver_type == 'warnings':
            all_events_fname = '%s_%s_trained_all_%s_events_spinup%smin_length%s_min%s-%s_fold%s.npy'\
            %(self.model_type, self.train_type, self.hazard, self.wofs_spinup_time, self.forecast_length,\
              start_min, end_min, fold)
        else:
            all_events_fname = '%s_%s_trained_all_%s_events_spinup%smin_length%s_min%s-%s_r%skm_fold%s.npy'\
            %(self.model_type, self.train_type, self.hazard, self.wofs_spinup_time, self.forecast_length,\
              start_min, end_min, self.ver_radius, fold)
        
        if self.use_any_srs:
            return train_save_dir, ver_save_dir, all_srs_fname, all_probs_fname, all_events_fname
        else:
            return train_save_dir, ver_save_dir, all_probs_fname, all_events_fname

    def get_bootstrap_bss_fnames(self, home_dir):
        
        save_dir = '%s/%s_trained/length_%s/%s' %(home_dir, self.train_type, self.forecast_length, self.hazard)
        
        prob_bootstrap_bss_save_fname = '%s_%s_%skm_trained_%s_%skm_verified_%s_%s-%smin'\
        '_daily_bss_from_raw_probs.npy'\
        %(self.model_type, self.train_type, self.train_radius,\
          self.ver_type, self.ver_radius, self.hazard, self.lead_time,\
          self.lead_time+self.forecast_length)
        
        sr_bootstrap_bss_save_fname = '%s_%s_%skm_trained_%s_%skm_verified_%s_%s-%smin'\
        '_daily_bss_from_avg_sr_map.npy'\
        %(self.model_type, self.train_type, self.train_radius,\
          self.ver_type, self.ver_radius, self.hazard, self.lead_time,\
          self.lead_time+self.forecast_length)
        
        return save_dir, prob_bootstrap_bss_save_fname, sr_bootstrap_bss_save_fname
    
    def get_bootstrap_rel_fnames(self, home_dir):
        
        save_dir = '%s/%s_trained/length_%s/%s' %(home_dir, self.train_type, self.forecast_length, self.hazard)
        
        prob_bootstrap_rel_save_fname = '%s_%s_%skm_trained_%s_%skm_verified_%s_%s-%smin'\
        '_daily_rel_from_raw_probs.npy'\
        %(self.model_type, self.train_type, self.train_radius,\
          self.ver_type, self.ver_radius, self.hazard, self.lead_time,\
          self.lead_time+self.forecast_length)
        
        sr_bootstrap_rel_save_fname = '%s_%s_%skm_trained_%s_%skm_verified_%s_%s-%smin'\
        '_daily_rel_from_avg_sr_map.npy'\
        %(self.model_type, self.train_type, self.train_radius,\
          self.ver_type, self.ver_radius, self.hazard, self.lead_time,\
          self.lead_time+self.forecast_length)
        
        return save_dir, prob_bootstrap_rel_save_fname, sr_bootstrap_rel_save_fname
    
    def get_bootstrap_pd_fnames(self, home_dir):
        
        save_dir = '%s/%s_trained/length_%s/%s' %(home_dir, self.train_type, self.forecast_length, self.hazard)
        
        prob_bootstrap_pd_save_fname = '%s_%s_%skm_trained_%s_%skm_verified_%s_%s-%smin'\
        '_daily_pd_from_raw_probs.npy'\
        %(self.model_type, self.train_type, self.train_radius,\
          self.ver_type, self.ver_radius, self.hazard, self.lead_time,\
          self.lead_time+self.forecast_length)
        
        sr_bootstrap_pd_save_fname = '%s_%s_%skm_trained_%s_%skm_verified_%s_%s-%smin'\
        '_daily_pd_from_avg_sr_map.npy'\
        %(self.model_type, self.train_type, self.train_radius,\
          self.ver_type, self.ver_radius, self.hazard, self.lead_time,\
          self.lead_time+self.forecast_length)
        
        return save_dir, prob_bootstrap_pd_save_fname, sr_bootstrap_pd_save_fname
    
    def get_bootstrap_summary_bss_fnames(self, home_dir, CI):
        
        save_dir = '%s/%s_trained/length_%s/%s' %(home_dir, self.train_type, self.forecast_length, self.hazard)
        
        prob_bootstrap_bss_save_fname = '%s_%s_%skm_trained_%s_%skm_verified_%s_%smin'\
        '_bootstrapped_bss_from_raw_probs_by_lead_time_%s_percent_CI.npy'\
        %(self.model_type, self.train_type, self.train_radius,\
          self.ver_type, self.ver_radius, self.hazard, self.forecast_length, CI)
        
        sr_bootstrap_bss_save_fname = '%s_%s_%skm_trained_%s_%skm_verified_%s_%smin_'\
        'bootstrapped_bss_from_avg_sr_map_by_lead_time_%s_percent_CI.npy'\
        %(self.model_type, self.train_type, self.train_radius,\
          self.ver_type, self.ver_radius, self.hazard, self.forecast_length, CI)
        
        return save_dir, prob_bootstrap_bss_save_fname, sr_bootstrap_bss_save_fname
    
    def get_bootstrap_summary_rel_fnames(self, home_dir, CI):
        
        save_dir = '%s/%s_trained/length_%s/%s' %(home_dir, self.train_type, self.forecast_length, self.hazard)
        
        bootstrap_rel_save_fname = '%s_%s_%skm_trained_%s_%skm_verified_%s_%smin'\
        '_bootstrapped_rel_%s-%smin_%s_percent_CI.npy'\
        %(self.model_type, self.train_type, self.train_radius,\
          self.ver_type, self.ver_radius, self.hazard, self.forecast_length,\
          self.lead_time, self.lead_time+self.forecast_length, CI)
        
        return save_dir, bootstrap_rel_save_fname
    
    def get_bootstrap_summary_pd_fnames(self, home_dir, CI):
        
        save_dir = '%s/%s_trained/length_%s/%s' %(home_dir, self.train_type, self.forecast_length, self.hazard)
        
        bootstrap_pd_save_fname = '%s_%s_%skm_trained_%s_%skm_verified_%s_%smin'\
        '_bootstrapped_pd_%s-%smin_%s_percent_CI.npy'\
        %(self.model_type, self.train_type, self.train_radius,\
          self.ver_type, self.ver_radius, self.hazard, self.forecast_length,\
          self.lead_time, self.lead_time+self.forecast_length, CI)
        
        return save_dir, bootstrap_pd_save_fname
    
    def get_bootstrapping_bs_data(self, n_boot, home_dir, use_any_srs):
        
        test_dir = '%s/%s/wofslag_%s/length_%s'\
        %(self.train_test_dir, self.hazard, self.wofs_spinup_time, self.forecast_length)
        
        daily_prob_bs_fname = '%s_%s_r%skm_trained_%s_r%skm_verified_all_%s_test_daily_prob_bs_%s-%smin.npy'\
        %(self.model_type, self.train_type, self.train_radius, self.ver_type, self.ver_radius, self.hazard,\
          self.lead_time, self.lead_time + self.forecast_length)
        
        daily_sr_bs_fname = '%s_%s_r%skm_trained_%s_r%skm_verified_all_%s_test_daily_sr_bs_%s-%smin.npy'\
        %(self.model_type, self.train_type, self.train_radius, self.ver_type, self.ver_radius, self.hazard,\
          self.lead_time, self.lead_time + self.forecast_length)
        
        daily_climo_fname = '%s_all_%s_%s_r%skm_trained_%s_r%skm_verified_%s-%smin_daily_climos.npy'\
        %(self.model_type, self.hazard, self.train_type, self.train_radius, self.ver_type, self.ver_radius, self.lead_time,\
          self.lead_time + self.forecast_length)
        
        save_dir, prob_bootstrap_bss_save_fname, sr_bootstrap_bss_save_fname = self.get_bootstrap_bss_fnames(home_dir)
        
        all_prob_bs = np.load('%s/%s' %(test_dir, daily_prob_bs_fname))
        if use_any_srs:
            all_sr_bs = np.load('%s/%s' %(test_dir, daily_sr_bs_fname))
        else:
            all_sr_bs = []
        all_climo = np.load('%s/%s' %(test_dir, daily_climo_fname))
        
        bootstrap_prob_bss = []
        bootstrap_sr_bss = []
        
        iterator = md.to_iterator(np.arange(n_boot), [all_prob_bs], [all_sr_bs], [all_climo], [use_any_srs])
        results = md.run_parallel(bootstrap, iterator, nprocs_to_use = 40,\
                                  description = 'Bootstrapping')
        
        if use_any_srs:
            for result in results:
                bootstrap_prob_bss.append(result[0])
                bootstrap_sr_bss.append(result[1])
        else:
            for result in results:
                bootstrap_prob_bss.append(result)
        
        bootstrap_prob_bss = np.array(bootstrap_prob_bss)
        if use_any_srs:
            bootstrap_sr_bss = np.array(bootstrap_sr_bss)
        
        if use_any_srs:
            utilities.save_data(save_dir, sr_bootstrap_bss_save_fname, bootstrap_sr_bss, 'npy')
        utilities.save_data(save_dir, prob_bootstrap_bss_save_fname, bootstrap_prob_bss, 'npy')
        
        if use_any_srs:
            return bootstrap_prob_bss, bootstrap_sr_bss
        else:
            return bootstrap_prob_bss
    
    def get_bootstrapping_rel_data(self, n_boot, home_dir, use_any_srs, points):
        
        nbins = len(points)
        
        test_dir = '%s/%s/wofslag_%s/length_%s'\
        %(self.train_test_dir, self.hazard, self.wofs_spinup_time, self.forecast_length)
        
        daily_prob_rel_fname = '%s_%s_r%skm_trained_%s_r%skm_verified_all_%s_test_daily_prob_rel_%s-%smin.npy'\
        %(self.model_type, self.train_type, self.train_radius, self.ver_type, self.ver_radius, self.hazard,\
          self.lead_time, self.lead_time + self.forecast_length)
        
        daily_prob_fcst_freq_fname = '%s_%s_r%skm_trained_%s_r%skm_verified_all_%s_test_daily_prob_fcst_freq_%s-%smin.npy'\
        %(self.model_type, self.train_type, self.train_radius, self.ver_type, self.ver_radius, self.hazard,\
          self.lead_time, self.lead_time + self.forecast_length)
        
        daily_sr_rel_fname = '%s_%s_r%skm_trained_%s_r%skm_verified_all_%s_test_daily_sr_rel_%s-%smin.npy'\
        %(self.model_type, self.train_type, self.train_radius, self.ver_type, self.ver_radius, self.hazard,\
          self.lead_time, self.lead_time + self.forecast_length)
        
        daily_sr_fcst_freq_fname = '%s_%s_r%skm_trained_%s_r%skm_verified_all_%s_test_daily_sr_fcst_freq_%s-%smin.npy'\
        %(self.model_type, self.train_type, self.train_radius, self.ver_type, self.ver_radius, self.hazard,\
          self.lead_time, self.lead_time + self.forecast_length)
        
        daily_climo_fname = '%s_all_%s_%s_r%skm_trained_%s_r%skm_verified_%s-%smin_daily_climos.npy'\
        %(self.model_type, self.hazard, self.train_type, self.train_radius, self.ver_type, self.ver_radius, self.lead_time,\
          self.lead_time + self.forecast_length)
        
        save_dir, prob_bootstrap_rel_save_fname, sr_bootstrap_rel_save_fname = self.get_bootstrap_rel_fnames(home_dir)
        
        all_prob_rel = np.load('%s/%s' %(test_dir, daily_prob_rel_fname))
        all_prob_fcst_freq = np.load('%s/%s' %(test_dir, daily_prob_fcst_freq_fname))
        if use_any_srs:
            all_sr_rel = np.load('%s/%s' %(test_dir, daily_sr_rel_fname))
            all_sr_fcst_freq = np.load('%s/%s' %(test_dir, daily_sr_fcst_freq_fname))
        else:
            all_sr_rel = []
            all_sr_fcst_freq = []
        
        all_climo = np.load('%s/%s' %(test_dir, daily_climo_fname))
        
        bootstrap_probs = np.zeros((((2*nbins)+2),n_boot))
        bootstrap_srs = np.zeros((((2*nbins)+2),n_boot))
        
        iterator = md.to_iterator(np.arange(n_boot), [all_prob_rel], [all_prob_fcst_freq],\
                                  [all_sr_rel], [all_sr_fcst_freq], [all_climo], [use_any_srs])
        results = md.run_parallel(bootstrap_reliability, iterator, nprocs_to_use = 40,\
                                  description = 'Bootstrapping')
        
        #Daily Reliability Arrays:
        #1. Reliability Scores
        #2. Climatologies
        #3:x(3+len(bins)-1). fcst frequency (one row per bin)
        #x:y(x+len(bins)-1). event climos (one row per bin)
        
        x = nbins + 2
        y = nbins + x
        
        for i in range(len(results)):
            r = results[i]
            prob_rel = r[0]
            prob_fcst_freq = r[1]
            climo = r[2]
            
            prob_rel_score = 0
            for j in range(len(points)):
                fk = points[j]
                x_bar_k = prob_rel[j]
                nk = prob_fcst_freq[j]
                prob_rel_score += (nk*((fk-x_bar_k)**2))
            N = np.sum(prob_fcst_freq)
            prob_rel_score = (prob_rel_score/N)[0]
            
            bootstrap_probs[0,i] = prob_rel_score
            bootstrap_probs[1,i] = climo
            bootstrap_probs[2:x,i] = prob_rel.reshape((prob_rel.shape[0],))
            bootstrap_probs[x:y,i] = prob_fcst_freq.reshape((prob_fcst_freq.shape[0],))
        
            if use_any_srs:
                sr_rel = r[3]
                sr_fcst_freq = r[4]
                
                sr_rel_score = 0
                for j in range(len(points)):
                    fk = points[j]
                    x_bar_k = sr_rel[j]
                    nk = sr_fcst_freq[j]
                    sr_rel_score += (nk*((fk-x_bar_k)**2))
                N = np.sum(sr_fcst_freq)
                sr_rel_score = sr_rel_score/N

                bootstrap_srs[0,i] = sr_rel_score
                bootstrap_srs[1,i] = climos
                bootstrap_srs[2:x,i] = sr_rel
                bootstrap_srs[x:y,i] = sr_fcst_freq
        
        if use_any_srs:
            utilities.save_data(save_dir, sr_bootstrap_rel_save_fname, bootstrap_srs, 'npy')
        utilities.save_data(save_dir, prob_bootstrap_rel_save_fname, bootstrap_probs, 'npy')
        
        if use_any_srs:
            return bootstrap_probs, bootstrap_srs
        else:
            return bootstrap_probs
    
    def get_bootstrapping_pd_data(self, n_boot, home_dir, use_any_srs, thresholds):
        
        nthresh = len(thresholds)
        
        test_dir = '%s/%s/wofslag_%s/length_%s'\
        %(self.train_test_dir, self.hazard, self.wofs_spinup_time, self.forecast_length)
        
        daily_prob_event_freq_fname = '%s_%s_r%skm_trained_%s_r%skm_verified_all_%s_test_daily_prob_event_freq_%s-%smin.npy'\
        %(self.model_type, self.train_type, self.train_radius, self.ver_type, self.ver_radius, self.hazard,\
          self.lead_time, self.lead_time + self.forecast_length)
        
        daily_prob_total_events_fname = '%s_%s_r%skm_trained_%s_r%skm_verified_all_%s_test_daily_prob_total_events_%s-%smin.npy'\
        %(self.model_type, self.train_type, self.train_radius, self.ver_type, self.ver_radius, self.hazard,\
          self.lead_time, self.lead_time + self.forecast_length)
        
        daily_prob_fcst_freq_pd_fname = '%s_%s_r%skm_trained_%s_r%skm_verified_all_%s_test_daily_prob_fcst_freq_pd_%s-%smin.npy'\
        %(self.model_type, self.train_type, self.train_radius, self.ver_type, self.ver_radius, self.hazard,\
          self.lead_time, self.lead_time + self.forecast_length)
        
        daily_sr_event_freq_fname = '%s_%s_r%skm_trained_%s_r%skm_verified_all_%s_test_daily_sr_event_freq_%s-%smin.npy'\
        %(self.model_type, self.train_type, self.train_radius, self.ver_type, self.ver_radius, self.hazard,\
          self.lead_time, self.lead_time + self.forecast_length)
        
        daily_sr_total_events_fname = '%s_%s_r%skm_trained_%s_r%skm_verified_all_%s_test_daily_sr_total_events_%s-%smin.npy'\
        %(self.model_type, self.train_type, self.train_radius, self.ver_type, self.ver_radius, self.hazard,\
          self.lead_time, self.lead_time + self.forecast_length)
        
        daily_sr_fcst_freq_pd_fname = '%s_%s_r%skm_trained_%s_r%skm_verified_all_%s_test_daily_sr_fcst_freq_pd_%s-%smin.npy'\
        %(self.model_type, self.train_type, self.train_radius, self.ver_type, self.ver_radius, self.hazard,\
          self.lead_time, self.lead_time + self.forecast_length)
        
        save_dir, prob_bootstrap_pd_save_fname, sr_bootstrap_pd_save_fname = self.get_bootstrap_pd_fnames(home_dir)
        
        all_prob_event_freq = np.load('%s/%s' %(test_dir, daily_prob_event_freq_fname))
        all_prob_total_events = np.load('%s/%s' %(test_dir, daily_prob_total_events_fname))
        all_prob_fcst_freq_pd = np.load('%s/%s' %(test_dir, daily_prob_fcst_freq_pd_fname))
        if use_any_srs:
            all_sr_event_freq = np.load('%s/%s' %(test_dir, daily_sr_event_freq_fname))
            all_sr_total_events = np.load('%s/%s' %(test_dir, daily_sr_total_events_fname))
            all_sr_fcst_freq_pd = np.load('%s/%s' %(test_dir, daily_sr_fcst_freq_pd_fname))
        else:
            all_sr_event_freq = []
            all_sr_total_events = []
            all_sr_fcst_freq_pd = []
        
        bootstrap_probs = np.zeros(((2*nthresh),n_boot))
        bootstrap_srs = np.zeros(((2*nthresh),n_boot))
        
        iterator = md.to_iterator(np.arange(n_boot), [all_prob_event_freq], [all_prob_fcst_freq_pd],\
                                  [all_prob_total_events], [all_sr_event_freq],\
                                  [all_sr_total_events], [use_any_srs])
        results = md.run_parallel(bootstrap_pd, iterator, nprocs_to_use = 40,\
                                  description = 'Bootstrapping PD Data')
        
        #Daily PD Arrays:
        #1:x. PODs (one row per threshold)
        #x:y. SRs (one row per threshold)
        
        x = nthresh
        y = nthresh + x
        
        for i in range(len(results)):
            r = results[i]
            prob_pod = r[0]
            prob_sr = r[1]
            bootstrap_probs[0:x,i] = prob_pod.reshape((len(prob_pod),))
            bootstrap_probs[x:y,i] = prob_sr.reshape((len(prob_sr),))
        
            if use_any_srs:
                sr_pod = r[2]
                sr_sr = r[3]

                bootstrap_srs[0:x,i] = sr_pod.reshape((len(sr_pod),))
                bootstrap_srs[x:y,i] = sr_sr.reshape((len(sr_sr),))
        
        if use_any_srs:
            utilities.save_data(save_dir, sr_bootstrap_pd_save_fname, bootstrap_srs, 'npy')
        utilities.save_data(save_dir, prob_bootstrap_pd_save_fname, bootstrap_probs, 'npy')
        
        if use_any_srs:
            return bootstrap_probs, bootstraps_srs
        else:
            return bootstrap_probs
        
        return
    
    def load_all_test_srs_probs_events(self, fcst_dir, ver_dir, all_srs_fname, all_probs_fname, all_events_fname):
        if self.use_any_srs:
            all_srs = np.load('%s/%s' %(fcst_dir, all_srs_fname))
        all_probs = np.load('%s/%s' %(fcst_dir, all_probs_fname))
        all_events = np.load('%s/%s' %(ver_dir, all_events_fname))
        if self.use_any_srs:
            return all_srs, all_probs, all_events
        else:
            return all_probs, all_events
    
    def get_all_test_srs_probs_events(self):
        if self.use_any_srs:
            all_fcst_dir, all_ver_dir, all_srs_fname, all_probs_fname, all_events_fname = self.get_all_test_sr_events_fname_dir()
            all_srs, all_probs, all_events = self.load_all_test_srs_probs_events(all_fcst_dir, all_ver_dir,\
                                                                                 all_srs_fname, all_probs_fname, all_events_fname)
            return all_srs, all_probs, all_events
        else:
            all_fcst_dir, all_ver_dir, all_probs_fname, all_events_fname = self.get_all_test_sr_events_fname_dir()
            all_probs, all_events = self.load_all_test_srs_probs_events(all_fcst_dir, all_ver_dir, '', all_probs_fname, all_events_fname)
            return all_probs, all_events
    
    def save_test_srs_probs_events(self, inits):
        
        if self.hazard == 'tornado':
            pd_thresholds = np.array([0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        else:
            pd_thresholds = pd_thresholds = np.arange(0, 1.01, 0.1)
        rel_bins = np.arange(0, 1.01, 0.1)
        
        all_srs = []
        all_probs = []
        all_events = []
        
        all_daily_sr_bs = []
        all_daily_prob_bs = []
        all_daily_climo = []
        
        all_daily_prob_rel = np.zeros((len(rel_bins)-1,1))
        all_daily_prob_fcst_freq = np.zeros((len(rel_bins)-1,1))
        all_daily_prob_event_freq = np.zeros((len(pd_thresholds), 1))
        all_daily_prob_total_events = np.zeros((len(pd_thresholds), 1))
        all_daily_prob_fcst_freq_pd = np.zeros((len(pd_thresholds), 1))
        
        all_daily_sr_rel = np.zeros((len(rel_bins)-1,1))
        all_daily_sr_fcst_freq = np.zeros((len(rel_bins)-1,1))
        all_daily_sr_event_freq = np.zeros((len(pd_thresholds), 1))
        all_daily_sr_total_events = np.zeros((len(pd_thresholds), 1))
        all_daily_sr_fcst_freq_pd = np.zeros((len(pd_thresholds), 1))
        
        first_grid = True
        
        for date in self.dates:
            for init in inits:
                if self.forecast_length > 30:
                    obs_path = '/work/ryan.martz/wofs_phi_data/obs_train/test_fcsts/%s/%s/wofslag_25/length_%s/%s/%s'\
                    %(self.model_type, self.hazard, self.forecast_length, date, init)
                    warnings_path = '/work/ryan.martz/wofs_phi_data/warnings_train/test_fcsts/%s/%s/wofslag_25/length_%s/%s/%s'\
                    %(self.model_type, self.hazard, self.forecast_length, date, init)
                    obs_and_warnings_path = '/work/ryan.martz/wofs_phi_data/obs_and_warnings_train/test_fcsts/%s/%s/wofslag_25'\
                    '/length_%s/%s/%s'\
                    %(self.model_type, self.hazard, self.forecast_length, date, init)
                else:
                    obs_path = '/work/eric.loken/wofs/paper6/rf_probs/30'
                    warnings_path = '/work/ryan.martz/wofs_phi_data/warnings_train/test_fcsts/%s/%s/wofslag_25/length_%s/%s/%s'\
                    %(self.model_type, self.hazard, self.forecast_length, date, init)
                    obs_and_warnings_path = '/work/ryan.martz/wofs_phi_data/obs_and_warnings_train/test_fcsts/%s/%s/wofslag_25'\
                    '/length_%s/%s/%s'\
                    %(self.model_type, self.hazard, self.forecast_length, date, init)
                
                init_time = dt.datetime.strptime(init, '%H%M')
                if self.forecast_length > 30:
                    start = init_time + dt.timedelta(seconds = 60*self.lead_time)
                    end = start + dt.timedelta(seconds = 60*self.forecast_length)
                else:
                    start = init_time + dt.timedelta(seconds = 60*(self.lead_time-5))
                    end = start + dt.timedelta(seconds = 60*self.forecast_length)
                start_str = start.strftime('%H%M')
                end_str = end.strftime('%H%M')
                
                if self.use_avg_srs:
                    if 'obs' in self.train_type:
                        sr_file = '%s_%s_trained_rf_%s_avg_sr_%s_init%s_v%s-%s_r%skm.txt'\
                        %(self.model_type, self.train_type, self.hazard, date, init, start_str,\
                          end_str, self.train_radius)
                    else:
                        sr_file = '%s_%s_trained_rf_%s_avg_sr_%s_init%s_v%s-%s.txt'\
                        %(self.model_type, self.train_type, self.hazard, date, init, start_str,\
                          end_str)
                    save_sr_fname = '%s_%s_r%skm_trained_all_%s_avg_sr_probs_%s-%smin.npy'\
                    %(self.model_type, self.train_type, self.train_radius, self.hazard,\
                      self.lead_time, self.lead_time + self.forecast_length)
                    save_daily_sr_bs_fname = '%s_%s_r%skm_trained_all_%s_test_daily_sr_bs_%s-%smin.npy'\
                    %(self.model_type, self.train_type, self.train_radius, self.hazard,\
                      self.lead_time, self.lead_time + self.forecast_length)
                    save_daily_sr_rel_fname = '%s_%s_r%skm_trained_all_%s_test_daily_sr_rel_%s-%smin.npy'\
                    %(self.model_type, self.train_type, self.train_radius, self.hazard,\
                      self.lead_time, self.lead_time + self.forecast_length)
                    save_daily_sr_fcst_freq_fname = '%s_%s_r%skm_trained_all_%s_test_daily_sr_fcst_freq_%s-%smin.npy'\
                    %(self.model_type, self.train_type, self.train_radius, self.hazard,\
                      self.lead_time, self.lead_time + self.forecast_length)
                    save_daily_sr_fcst_freq_pd_fname = '%s_%s_r%skm_trained_all_%s_test_daily_sr_fcst_freq_pd_%s-%smin.npy'\
                    %(self.model_type, self.train_type, self.train_radius, self.hazard,\
                      self.lead_time, self.lead_time + self.forecast_length)
                    save_daily_sr_event_freq_fname = '%s_%s_r%skm_trained_all_%s_test_daily_sr_event_freq_%s-%smin.npy'\
                    %(self.model_type, self.train_type, self.train_radius, self.hazard,\
                      self.lead_time, self.lead_time + self.forecast_length)
                    save_daily_sr_total_events_fname = '%s_%s_r%skm_trained_all_%s_test_daily_sr_total_events_%s-%smin.npy'\
                    %(self.model_type, self.train_type, self.train_radius, self.hazard,\
                      self.lead_time, self.lead_time + self.forecast_length)
                else:
                    sr_file = '%s_%s_trained_rf_%s_sr_probs_%s_init%s_v%s-%s_r%skm.txt'\
                    %(self.model_type, self.train_type, self.hazard, date, init,\
                      start_str, end_str, self.train_radius)
                    save_sr_fname = '%s_%s_r%skm_trained_all_%s_sr_probs_%s-%smin.npy'\
                    %(self.model_type, self.train_type, self.train_radius, self.hazard,\
                      self.lead_time, self.lead_time + self.forecast_length)
                
                if self.forecast_length > 30:
                    obs_raw_probs_file = '%s_obs_trained_rf_%s_raw_probs_%s_init%s_v%s-%s_r%skm.txt'\
                    %(self.model_type, self.hazard, date, init, start_str, end_str, self.train_radius)
                    warnings_raw_probs_file = '%s_warnings_trained_rf_%s_raw_probs_%s_init%s_v%s-%s.txt'\
                    %(self.model_type, self.hazard, date, init, start_str, end_str)
                    obs_and_warnings_raw_probs_file = '%s_obs_and_warnings_trained_rf_%s_raw_probs_%s_init%s_v%s-%s.txt'\
                    %(self.model_type, self.hazard, date, init, start_str, end_str)
                    
                    save_probs_fname = '%s_%s_r%skm_trained_%s_r%skm_verified_all_%s_test_raw_probs_%s-%smin.npy'\
                    %(self.model_type, self.train_type, self.train_radius, self.ver_type, self.ver_radius,\
                      self.hazard, self.lead_time, self.lead_time+self.forecast_length)
                    save_daily_prob_bs_fname = '%s_%s_r%skm_trained_%s_r%skm_verified_all_%s_test_daily_prob_bs_%s-%smin.npy'\
                    %(self.model_type, self.train_type, self.train_radius, self.ver_type, self.ver_radius, self.hazard,\
                      self.lead_time, self.lead_time + self.forecast_length)
                    save_daily_prob_rel_fname = '%s_%s_r%skm_trained_%s_r%skm_verified_all_%s_test_daily_prob_rel_%s-%smin.npy'\
                    %(self.model_type, self.train_type, self.train_radius, self.ver_type, self.ver_radius, self.hazard,\
                      self.lead_time, self.lead_time + self.forecast_length)
                    save_daily_prob_fcst_freq_fname = '%s_%s_r%skm_trained_%s_r%skm_verified_all_%s_test_daily_prob_'\
                    'fcst_freq_%s-%smin.npy' %(self.model_type, self.train_type, self.train_radius,\
                                               self.ver_type, self.ver_radius, self.hazard,\
                                               self.lead_time, self.lead_time + self.forecast_length)
                    save_daily_prob_fcst_freq_pd_fname = '%s_%s_r%skm_trained_%s_r%skm_verified_all_%s_test_daily_prob_'\
                    'fcst_freq_pd_%s-%smin.npy' %(self.model_type, self.train_type, self.train_radius,\
                                                  self.ver_type, self.ver_radius, self.hazard,\
                                                  self.lead_time, self.lead_time + self.forecast_length)
                    save_daily_prob_event_freq_fname = '%s_%s_r%skm_trained_%s_r%skm_verified_all_%s_test_daily_prob_event_freq'\
                    '_%s-%smin.npy' %(self.model_type, self.train_type, self.train_radius, self.ver_type, self.ver_radius,\
                                      self.hazard, self.lead_time, self.lead_time + self.forecast_length)
                    save_daily_prob_total_events_fname = '%s_%s_r%skm_trained_%s_r%skm_verified_all_%s_test_daily_prob_total_'\
                    'events_%s-%smin.npy' %(self.model_type, self.train_type, self.train_radius, self.ver_type, self.ver_radius,\
                                            self.hazard, self.lead_time, self.lead_time + self.forecast_length)
                else:
                    if self.hazard == 'tornado':
                        raw_probs_file = 'all_rf_probs_torn_%s_%s_v%s-%s_r%.1fkm.txt'\
                        %(date, init, start_str, end_str, self.train_radius)
                    else:
                        raw_probs_file = 'all_rf_probs_%s_%s_%s_v%s-%s_r%.1fkm.txt'\
                        %(self.hazard, date, init, start_str, end_str, self.train_radius)
                    
                    save_probs_fname = '%s_%s_r%skm_trained_all_%s_test_raw_probs_%s-%smin.npy'\
                    %(self.model_type, self.train_type, self.train_radius, self.hazard,\
                      self.lead_time, self.lead_time + self.forecast_length)
                
                if self.forecast_length > 30:
                    obs_dir = '/work/eric.loken/wofs/2024_update/SFE2024/obs/full_npy'
                    obs_file = '%s_reps1d_%s_v%s-%s_r%skm%s.npy'\
                    %(self.hazard, date, start_str, end_str, self.ver_radius, self.buffer_str)
                else:
                    obs_dir = '/work/eric.loken/wofs/2024_update/SFE2024/obs/full_npy'
                    if self.hazard == 'tornado':
                        obs_file = 'torn_reps1d_%s_v%s-%s_r%.1fkm.npy' %(date, start_str, end_str, self.ver_radius)
                    else:
                        obs_file = '%s_reps1d_%s_v%s-%s_r%.1fkm.npy' %(self.hazard, date, start_str, end_str, self.ver_radius)
                    
                obs_and_warnings_dir = '/work/ryan.martz/wofs_phi_data/training_data/obs_and_warnings/full_1d_obs_and_warnings/length_%s/%s'\
                %(self.forecast_length, self.hazard)
                warnings_dir = '/work/ryan.martz/wofs_phi_data/training_data/warnings/full_1d_warnings/length_%s/%s'\
                %(self.forecast_length, self.hazard)
                obs_and_warnings_file = '%s_obs_and_warnings_%s_v%s-%s_r%skm_1d.npy'\
                %(self.hazard, date, start_str, end_str, self.ver_radius)
                warnings_file = '%s_warnings_%s_v%s-%s_1d.npy'\
                %(self.hazard, date, start_str, end_str)
                    
                save_events_fname = '%s_all_%s_%s_r%skm_trained_%s_r%skm_verified_%s-%smin.npy'\
                %(self.model_type, self.hazard, self.train_type, self.train_radius, self.ver_type, self.ver_radius,\
                  self.lead_time, self.lead_time + self.forecast_length)
                
                save_daily_climo_fname = '%s_all_%s_%s_r%skm_trained_%s_r%skm_verified_%s-%smin_daily_climos.npy'\
                %(self.model_type, self.hazard, self.train_type, self.train_radius, self.ver_type, self.ver_radius, self.lead_time,\
                  self.lead_time + self.forecast_length)
                
                try:
                    obs_probs_2d = np.genfromtxt('%s/%s' %(obs_path, obs_raw_probs_file))
                    warnings_probs_2d = np.genfromtxt('%s/%s' %(warnings_path, warnings_raw_probs_file))
                    #obs_and_warnings_probs_2d = np.genfromtxt('%s/%s' %(obs_and_warnings_path, obs_and_warnings_raw_probs_file))
                    if self.train_type == 'obs':
                        probs = obs_probs_2d.reshape(90000,)
                    elif self.train_type == 'warnings':
                        probs = warnings_probs_2d.reshape(90000,)
                    elif self.train_type == 'obs_and_warnings':
                        probs = obs_and_warnings_probs_2d.reshape(90000,)
                    
                    obs = np.load('%s/%s' %(obs_dir, obs_file))
                    warnings = np.load('%s/%s' %(warnings_dir, warnings_file))
                    #obs_and_warnings = np.load('%s/%s' %(obs_and_warnings_dir, obs_and_warnings_file))
                    if self.ver_type == 'obs':
                        events = obs.reshape(90000,)
                    elif self.ver_type == 'warnings':
                        events = warnings.reshape(90000)
                    elif self.ver_type == 'obs_and_warnings':
                        events = obs_and_warnings.reshape(90000,)
                except:
                    continue
                
                if self.use_any_srs:
                    srs_2d = np.genfromtxt('%s/%s' %(fcst_path, sr_file))
                    srs = srs_2d.reshape(90000,)
                    all_srs.extend(srs)
                    sr_bs = BS(events, srs)
                    all_daily_sr_bs.append(sr_bs)
                    sr_rel, sr_fcst_freq, sr_fcst_freq_pd = self.get_grid_rel(srs, events, rel_bins)
                    if first_grid:
                        all_daily_sr_rel[:,0] = sr_rel
                        all_daily_sr_fcst_freq[:,0] = sr_fcst_freq
                        all_daily_sr_event_freq[:,0] = sr_event_freq
                        all_daily_sr_total_events[:,0] = sr_total_events
                        all_daily_sr_fcst_freq_pd[:,0] = sr_fcst_freq_pd
                    else:
                        all_daily_sr_rel = np.append(all_daily_sr_rel, sr_rel, axis = 1)
                        all_daily_sr_fcst_freq = np.append(all_daily_sr_fcst_freq, sr_fcst_freq, axis = 1)
                        all_daily_sr_event_freq = np.append(all_daily_sr_event_freq, sr_event_freq, axis = 1)
                        all_daily_sr_total_events = np.append(all_daily_sr_fcst_freq, sr_total_events, axis = 1)
                        all_daily_sr_fcst_freq_pd = np.append(all_daily_sr_fcst_freq, sr_fcst_freq_pd, axis = 1)
                
                all_probs.extend(probs)
                all_events.extend(events)
                prob_bs = BS(events, probs)
                all_daily_prob_bs.append(prob_bs)
                climo = np.mean(events)
                all_daily_climo.append(climo)
                
                prob_rel, prob_fcst_freq = self.get_grid_rel(probs, events, rel_bins)
                prob_event_freq, prob_total_events, prob_fcst_freq_pd = self.get_grid_pd(probs, events, pd_thresholds)
                if first_grid:
                    all_daily_prob_rel[:,0] = prob_rel.reshape((prob_rel.shape[0],))
                    all_daily_prob_fcst_freq[:,0] = prob_fcst_freq.reshape((prob_fcst_freq.shape[0],))
                    all_daily_prob_event_freq[:,0] = prob_event_freq.reshape((prob_event_freq.shape[0],))
                    all_daily_prob_total_events[:,0] = prob_total_events.reshape((prob_total_events.shape[0],))
                    all_daily_prob_fcst_freq_pd[:,0] = prob_fcst_freq_pd.reshape((prob_fcst_freq_pd.shape[0],))
                    first_grid = False
                else:
                    all_daily_prob_rel = np.append(all_daily_prob_rel, prob_rel, axis = 1)
                    all_daily_prob_fcst_freq = np.append(all_daily_prob_fcst_freq, prob_fcst_freq, axis = 1)
                    all_daily_prob_event_freq = np.append(all_daily_prob_event_freq, prob_event_freq, axis = 1)
                    all_daily_prob_total_events = np.append(all_daily_prob_total_events,\
                                                                     prob_total_events, axis = 1)
                    all_daily_prob_fcst_freq_pd = np.append(all_daily_prob_fcst_freq_pd, prob_fcst_freq_pd, axis = 1)
                
                #except:
                #    continue
                
                
        
        save_dir = '/work/ryan.martz/wofs_phi_data/%s_train/test_fcsts/%s/%s/wofslag_%s/length_%s'\
        %(self.train_type, self.model_type, self.hazard, self.wofs_spinup_time, self.forecast_length)
        if self.use_any_srs:
            print(len(all_srs), len(all_probs), len(all_events))
        else:
            print(len(all_probs), len(all_events))
        
        if self.use_any_srs:
            utilities.save_data(save_dir, save_sr_fname, all_srs, 'npy')
            utilities.save_data(save_dir, save_daily_sr_bs_fname, all_daily_sr_bs, 'npy')
            utilities.save_data(save_dir, save_daily_sr_rel_fname, all_daily_sr_rel, 'npy')
            utilities.save_data(save_dir, save_daily_sr_fcst_freq_fname, all_daily_sr_fcst_freq, 'npy')
            utilities.save_data(save_dir, save_daily_sr_event_freq_fname, all_daily_sr_event_freq, 'npy')
            utilities.save_data(save_dir, save_daily_sr_total_events_fname, all_daily_sr_total_events, 'npy')
            utilities.save_data(save_dir, save_daily_sr_fcst_freq_pd_fname, all_daily_sr_fcst_freq_pd, 'npy')
        utilities.save_data(save_dir, save_probs_fname, all_probs, 'npy')
        utilities.save_data(save_dir, save_daily_prob_bs_fname, all_daily_prob_bs, 'npy')
        utilities.save_data(save_dir, save_daily_prob_rel_fname, all_daily_prob_rel, 'npy')
        utilities.save_data(save_dir, save_daily_prob_fcst_freq_fname, all_daily_prob_fcst_freq, 'npy')
        utilities.save_data(save_dir, save_daily_prob_event_freq_fname, all_daily_prob_event_freq, 'npy')
        utilities.save_data(save_dir, save_daily_prob_total_events_fname, all_daily_prob_total_events, 'npy')
        utilities.save_data(save_dir, save_daily_prob_fcst_freq_pd_fname, all_daily_prob_fcst_freq_pd, 'npy')
        utilities.save_data(save_dir, save_daily_climo_fname, all_daily_climo, 'npy')
        utilities.save_data(save_dir, save_events_fname, all_events, 'npy')
        
        return
    
    def get_grid_rel(self, fcsts, events, bins):
        grid_rel = np.zeros((len(bins)-1,1))
        grid_fcst_freq = np.zeros((len(bins)-1,1))
        for i in range(len(bins)-1):
            if i == len(bins) - 2:
                bin_events = events[(fcsts >= bins[i]) & (fcsts <= bins[i+1])]
            else:
                bin_events = events[(fcsts >= bins[i]) & (fcsts < bins[i+1])]
            grid_rel[i,0] = np.sum(bin_events)
            grid_fcst_freq[i,0] = len(bin_events)
        
        return grid_rel, grid_fcst_freq
    
    def get_grid_pd(self, fcsts, events, thresholds):
        
        grid_event_freq = np.zeros((len(thresholds),1))
        grid_fcst_freq_pd = np.zeros((len(thresholds),1))
        grid_total_events = np.zeros((len(thresholds),1))
        
        for i in range(len(thresholds)):
            t = thresholds[i]
            grid_event_freq[i] = np.sum(events[fcsts >= t])
            grid_total_events[i] = np.sum(events)
            grid_fcst_freq_pd[i] = len(fcsts[fcsts >= t])
        
        return grid_event_freq, grid_total_events, grid_fcst_freq_pd
        
    def save_test_srs_probs_events_by_fold(self, inits):
        for fold in range(self.n_folds):
            fold_dates = self.date_test_folds[fold]
            all_srs = []
            all_probs = []
            all_events = []
            for date in fold_dates:
                for init in inits:
                    fcst_path = '/work/ryan.martz/wofs_phi_data/%s_train/test_fcsts/%s/%s/wofslag_25/length_%s/%s/%s'\
                    %(self.train_type, self.model_type, self.hazard, self.forecast_length, date, init)
                    time = dt.datetime.strptime(init, '%H%M')
                    start = time + dt.timedelta(seconds = 60*self.lead_time)
                    end = start + dt.timedelta(seconds = 60*self.forecast_length)
                    start_str = start.strftime('%H%M')
                    end_str = end.strftime('%H%M')

                    if self.use_avg_srs:
                        sr_file = '%s_%s_trained_rf_%s_avg_sr_probs_%s_init%s_v%s-%s_r%skm.txt'\
                        %(self.model_type, self.train_type, self.hazard, date, init, start_str,\
                          end_str, self.train_radius)
                        save_sr_fname = '%s_%s_trained_all_rf_%s_avg_sr_spinup%smin_length%smin_%s-%s_r%skm_fold%s.npy'\
                        %(self.model_type, self.ver_type, self.hazard, self.wofs_spinup_time, self.forecast_length,\
                          start_str, end_str, self.ver_radius, fold)
                    else:
                        sr_file = '%s_%s_trained_rf_%s_sr_probs_%s_init%s_v%s-%s_r%skm.txt'\
                        %(self.model_type, self.train_type, self.hazard, date, init, start_str,\
                          end_str, self.train_radius)
                        save_sr_fname = '%s_%s_trained_all_rf_%s_sr_spinup%smin_length%smin_%s-%s_r%skm_fold%s.npy'\
                        %(self.model_type, self.ver_type, self.hazard, self.wofs_spinup_time, self.forecast_length,\
                          start_str, end_str, self.ver_radius, fold)

                    if self.forecast_length > 30:
                        lsr_dir = '/work/eric.loken/wofs/2024_update/SFE2024/obs/full_npy'
                        lsr_file = '%s_reps1d_%s_v%s-%s_r%skm%s.npy'\
                        %(self.hazard, date, start_str, end_str, self.ver_radius, self.buffer_str)
                        
                        obsw_dir = '/work/ryan.martz/wofs_phi_data/training_data/obs_and_warnings/'\
                        'full_1d_obs_and_warnings/length_%s/%s' %(self.forecast_length, self.hazard)
                        obsw_file = '%s_obs_and_warnings_%s_v%s-%s_r%skm_1d.npy'\
                        %(self.hazard, date, start_str, end_str, self.ver_radius)
                        
                        warn_dir = '/work/ryan.martz/wofs_phi_data/training_data/warnings/'\
                        'full_1d_warnings/length_%s/%s' %(self.forecast_length, self.hazard)
                        warn_file = '%s_warnings_%s_v%s-%s_1d.npy' %(self.hazard, date, start_str, end_str)
                        
                        lsr_prob_dir = '/work/ryan.martz/wofs_phi_data/obs_train/test_fcsts/%s/%s/'\
                        'wofslag_25/length_%s/%s/%s' %(self.model_type, self.hazard, self.forecast_length,\
                                                       date, init)
                        lsr_prob_file = '%s_obs_trained_rf_%s_raw_probs_%s_init%s_v%s-%s_r%skm.txt'\
                        %(self.model_type, self.hazard, date, init, start_str,\
                          end_str, self.train_radius)
                        obsw_prob_dir = '/work/ryan.martz/wofs_phi_data/obs_and_warnings_train/test_fcsts/%s/%s/'\
                        'wofslag_25/length_%s/%s/%s' %(self.model_type, self.hazard, self.forecast_length,\
                                                       date, init)
                        obsw_prob_file = '%s_obs_and_warnings_trained_rf_%s_raw_probs_%s_init%s_v%s-%s_r%skm.txt'\
                        %(self.model_type, self.hazard, date, init, start_str,\
                          end_str, self.train_radius)
                        warn_prob_dir = '/work/ryan.martz/wofs_phi_data/warnings_train/test_fcsts/%s/%s/'\
                        'wofslag_25/length_%s/%s/%s' %(self.model_type, self.hazard, self.forecast_length,\
                                                       date, init)
                        warn_prob_file = '%s_warnings_trained_rf_%s_raw_probs_%s_init%s_v%s-%s.txt'\
                        %(self.model_type, self.hazard, date, init, start_str, end_str)
                        
                        if (not os.path.exists('%s/%s' %(lsr_dir, lsr_file))) or\
                        (not os.path.exists('%s/%s' %(warn_dir, warn_file))) or\
                        (not os.path.exists('%s/%s' %(lsr_prob_dir, lsr_prob_file))) or\
                        (not os.path.exists('%s/%s' %(warn_prob_dir, warn_prob_file))):
                            continue
                        
                        if self.ver_type == 'obs':
                            events_dir = lsr_dir
                            events_file = lsr_file
                            fcst_path = lsr_prob_dir
                            raw_probs_file = lsr_prob_file
                            save_probs_fname = '%s_%s_trained_all_rf_%s_raw_probs_spinup%smin_'\
                            'length%s_min%s-%s_r%skm_fold%s.npy' %(self.model_type, self.ver_type,\
                                                                   self.hazard, self.wofs_spinup_time,\
                                                                   self.forecast_length, self.lead_time,\
                                                                   self.lead_time + self.forecast_length,\
                                                                   self.ver_radius, fold)
                            save_events_fname = '%s_%s_trained_all_%s_events_spinup%smin_'\
                            'length%s_min%s-%s_r%skm_fold%s.npy' %(self.model_type, self.ver_type,\
                                                                   self.hazard, self.wofs_spinup_time,\
                                                                   self.forecast_length, self.lead_time,\
                                                                   self.lead_time + self.forecast_length,\
                                                                   self.ver_radius, fold)
                        elif self.ver_type == 'obs_and_warnings':
                            events_dir = obsw_dir
                            events_file = obsw_file
                            fcst_path = obsw_prob_dir
                            raw_probs_file = obsw_prob_file
                        elif self.ver_type == 'warnings':
                            events_dir = warn_dir
                            events_file = warn_file
                            fcst_path = warn_prob_dir
                            raw_probs_file = warn_prob_file
                            save_probs_fname = '%s_%s_trained_all_rf_%s_raw_probs_spinup%smin_'\
                            'length%s_min%s-%s_fold%s.npy' %(self.model_type, self.ver_type,\
                                                             self.hazard, self.wofs_spinup_time,\
                                                             self.forecast_length, self.lead_time,\
                                                             self.lead_time + self.forecast_length, fold)
                            save_events_fname = '%s_%s_trained_all_%s_events_spinup%smin_'\
                            'length%s_min%s-%s_fold%s.npy' %(self.model_type, self.ver_type,\
                                                             self.hazard, self.wofs_spinup_time,\
                                                             self.forecast_length, self.lead_time,\
                                                             self.lead_time + self.forecast_length, fold)
                    
                    elif self.ver_type == 'obs':
                        events_dir = '/work/eric.loken/wofs/paper6/obs/full_npy'
                        if self.hazard == 'tornado':
                            events_file = 'torn_reps1d_%s_v%s-%s_r%.1fkm.npy' %(date, start_str, end_str, self.ver_radius)
                        else:
                            events_file = '%s_reps1d_%s_v%s-%s_r%.1fkm.npy' %(self.hazard, date, start_str, end_str, self.ver_radius)
                    
                    try:
                        #srs_2d = np.genfromtxt('%s/%s' %(fcst_path, sr_file))
                        #srs = srs_2d.reshape(90000,)
                        probs_2d = np.genfromtxt('%s/%s' %(fcst_path, raw_probs_file))
                        probs = probs_2d.reshape(90000,)
                        events = np.load('%s/%s' %(events_dir, events_file))
                        events = events.reshape(90000,)
                    except:
                        continue

                    #all_srs.extend(srs)
                    all_probs.extend(probs)
                    all_events.extend(events)

            save_dir = '/work/ryan.martz/wofs_phi_data/%s_train/test_fcsts/%s/%s/wofslag_%s/length_%s/all_raw_probs_fold%s'\
            %(self.train_type, self.model_type, self.hazard, self.wofs_spinup_time, self.forecast_length, fold)
            print('fold ', fold, ': ', len(all_probs), len(all_events))
            #utilities.save_data(save_dir, save_sr_fname, all_srs, 'npy')
            utilities.save_data(save_dir, save_probs_fname, all_probs, 'npy')
            utilities.save_data(save_dir, save_events_fname, all_events, 'npy')
        
        return

    
    def calc_reliability(self, bins):
        'bins should be given in decimal values'
        
        if self.use_any_srs:
            all_srs, all_probs, all_events = self.get_all_test_srs_probs_events()
            print(len(all_srs), len(all_probs), len(all_events))
        else:
            all_probs, all_events = self.get_all_test_srs_probs_events()
            print(len(all_probs), len(all_events))
        rel_sum_srs = 0
        rel_sum_probs = 0
        rel_climos_srs = []
        rel_climos_probs = []
        fcst_frequencies_srs = []
        fcst_frequencies_probs = []
        for i in range(len(bins)-1):
            bin_start = bins[i]
            bin_end = bins[i+1]
            if self.use_any_srs:
                events_srs = all_events[np.where((all_srs >= bin_start) & (all_srs < bin_end))]
                rel_srs = all_srs[np.where((all_srs >= bin_start) & (all_srs < bin_end))]
                if events_srs.size == 0:
                    rel_climos_srs.append(-1)
                else:
                    rel_climos_srs.append(np.mean(events_srs))
                    rel_sum_srs += np.sum(np.power(rel_srs - events_srs, 2))
                fcst_frequencies_srs.append(rel_srs.size)
                
            events_probs = all_events[np.where((all_probs >= bin_start) & (all_probs < bin_end))]
            events_probs = np.reshape(events_probs, (len(events_probs),))
            rel_probs = all_probs[np.where((all_probs >= bin_start) & (all_probs < bin_end))]
            if events_probs.size == 0:
                rel_climos_probs.append(-1)
            else:
                rel_climos_probs.append(np.mean(events_probs))
                diffs = rel_probs - events_probs
                rel_sum_probs += np.sum(np.power(diffs, 2))
            
            fcst_frequencies_probs.append(rel_probs.size)
        
        N = all_probs.size
        if self.use_any_srs:
            rel_srs = rel_sum_srs/N
        rel_probs = rel_sum_probs/N
        
        ver_climo = np.mean(all_events)
        
        if self.use_any_srs:
            return fcst_frequencies_srs, fcst_frequencies_probs, rel_srs, rel_probs, rel_climos_srs, rel_climos_probs, ver_climo
        else:
            return fcst_frequencies_probs, rel_probs, rel_climos_probs, ver_climo
            
    def calc_pod_sr(self, thresholds):
        if self.use_any_srs:
            all_srs, all_probs, all_events = self.get_all_test_srs_probs_events()
        else:
            all_probs, all_events = self.get_all_test_srs_probs_events()
        pods_sr = []
        pods_probs = []
        srs_sr = []
        srs_probs = []
        thresholds_sr = []
        thresholds_probs = []
        for t in thresholds:
            if self.use_any_srs:
                yes_fcst_events_srs = all_events[np.where(all_srs >= t)]
                no_fcst_events_srs = all_events[np.where(all_srs < t)]
            
                a_srs = np.sum(yes_fcst_events_srs)
                b_srs = yes_fcst_events_srs.size - a_srs
                c_srs = np.sum(no_fcst_events_srs)
                d_srs = no_fcst_events_srs.size - c_srs
            
                if (a_srs + c_srs > 0) and (a_srs + b_srs > 0):

                    pod_srs = a_srs/(a_srs+c_srs)
                    sr_srs = a_srs/(a_srs+b_srs)

                    thresholds_sr.append(t)
                    srs_sr.append(sr_srs)
                    pods_sr.append(pod_srs)

                else:
                    thresholds_sr.append(-1)
                    srs_sr.append(-1)
                    pods_sr.append(-1)
            
            yes_fcst_events_probs = all_events[np.where(all_probs >= t)]
            no_fcst_events_probs = all_events[np.where(all_probs < t)]
            
            a_probs = np.sum(yes_fcst_events_probs)
            b_probs = yes_fcst_events_probs.size - a_probs
            c_probs = np.sum(no_fcst_events_probs)
            d_probs = no_fcst_events_probs.size - c_probs
            
            if (a_probs + c_probs > 0) and (a_probs + b_probs > 0):
                pod_probs = a_probs/(a_probs+c_probs)
                sr_probs = a_probs/(a_probs+b_probs)

                thresholds_probs.append(t)
                pods_probs.append(pod_probs)
                srs_probs.append(sr_probs)
            else:
                thresholds_probs.append(-1)
                pods_probs.append(-1)
                srs_probs.append(-1)
        if self.use_any_srs:
            return thresholds_sr, pods_sr, srs_sr, thresholds_probs, pods_probs, srs_probs
        else:
            return thresholds_probs, pods_probs, srs_probs
    
    def gen_brier_skill_scores(self):
        
        fcst_dir, ver_dir, all_probs_fname, all_events_fname =\
        self.get_all_test_sr_events_fname_dir()
        all_probs, all_events = self.load_all_test_srs_probs_events(fcst_dir, ver_dir, '',\
                                                                    all_probs_fname, all_events_fname)
        climo = np.mean(all_events)
        
        if self.use_any_srs:
            sr_skill_by_fold = []
            probs_skill_by_fold = []
            all_srs = []
            for fold in range(self.n_folds):
                fcst_dir, ver_dir, all_srs_fname, all_probs_fname, all_events_fname =\
                self.get_all_test_sr_events_fname_dir_by_fold(fold)
                all_srs_fold, all_probs_fold, all_events_fold =\
                self.load_all_test_srs_probs_events(fcst_dir, ver_dir, all_srs_fname,\
                                                    all_probs_fname, all_events_fname)

                #probs_less_10_indices = np.where(all_probs_fold < 0.1)
                #all_probs_fold[probs_less_10_indices] = 0
                #all_srs_fold[probs_less_10_indices] = 0

                climo_fcst_fold = np.ones(all_events_fold.shape)*climo

                bs_sr_fold = BS(all_events_fold, all_srs_fold)
                bs_probs_fold = BS(all_events_fold, all_probs_fold)
                bs_climo_fold = BS(all_events_fold, climo_fcst_fold)

                bss_sr_fold = (bs_sr_fold - bs_climo_fold)/(-bs_climo_fold)
                bss_probs_fold = (bs_probs_fold - bs_climo_fold)/(-bs_climo_fold)

                sr_skill_by_fold.append(bss_sr_fold)
                probs_skill_by_fold.append(bss_probs_fold)

                all_srs.extend(all_srs_fold)
        else:
            sr_skill_by_fold = []
            probs_skill_by_fold = []
            for fold in range(self.n_folds):
                fcst_dir_folds, ver_dir_folds, all_probs_fname_folds, all_events_fname_folds =\
                self.get_all_test_sr_events_fname_dir_by_fold(fold)
                all_probs_fold, all_events_fold =\
                self.load_all_test_srs_probs_events(fcst_dir_folds, ver_dir_folds, '',\
                                                    all_probs_fname_folds, all_events_fname_folds)
                
                climo_fcst_fold = np.ones(all_events_fold.shape)*climo
                bs_probs_fold = BS(all_events_fold, all_probs_fold)
                bs_climo_fold = BS(all_events_fold, climo_fcst_fold)
                bss_probs_fold = (bs_probs_fold - bs_climo_fold)/(-bs_climo_fold)
                probs_skill_by_fold.append(bss_probs_fold)
        
        num_fcsts = len(all_probs)
        climo_fcst = np.ones(num_fcsts)*climo
        
        bs_probs = BS(all_events, all_probs)
        bs_climo = BS(all_events, climo_fcst)
        bss_probs = (bs_probs - bs_climo)/(-bs_climo)
        if self.use_any_srs:    
            bs_sr = BS(all_events, all_srs)
            bss_sr = (bs_sr - bs_climo)/(-bs_climo)
        
        if self.use_any_srs:
            return bss_sr, sr_skill_by_fold, bss_probs, probs_skill_by_fold
        else:
            return bss_probs, probs_skill_by_fold
    
    def set_chunk_splits(self):
        '''Does the k fold logic. The class stores a list of dates to train on, and this samples it into training, testing, validation for each fold.
        This allows us to just use the fold number as an index to each of the lists to get the list of dates for the current fold for testing, validation,
        and training.'''
        
        num_dates = len(self.dates)
        indices = np.arange(num_dates)
        splits = np.array_split(indices, self.n_folds)
        chunk_splits = []
        for i in range(len(splits)):
            split = splits[i]
            chunk_splits.append(split[0])
        chunk_splits.append(num_dates)
        self.date_test_folds = []
        self.date_val_folds = []
        self.date_train_folds = []
        if self.n_folds == 1:
            self.date_test_folds = [self.dates]
            self.date_val_folds = [self.dates]
            self.date_train_folds = [self.dates]
            return
        for i in range(self.n_folds):
            if i == 0:
                self.date_test_folds.append(self.dates[chunk_splits[i]:chunk_splits[i+1]])
                self.date_val_folds.append(self.dates[chunk_splits[-2]:chunk_splits[-1]])
                self.date_train_folds.append(self.dates[chunk_splits[i+1]:chunk_splits[i+1+(self.n_folds-2)]])
            elif i == self.n_folds - 1:
                self.date_test_folds.append(self.dates[chunk_splits[i]:chunk_splits[-1]])
                self.date_val_folds.append(self.dates[chunk_splits[i-1]:chunk_splits[i]])
                self.date_train_folds.append(self.dates[chunk_splits[0]:chunk_splits[self.n_folds-2]])
            elif (i+1+(self.n_folds-2)) > (self.n_folds):
                self.date_test_folds.append(self.dates[chunk_splits[i]:chunk_splits[i+1]])
                self.date_val_folds.append(self.dates[chunk_splits[i-1]:chunk_splits[i]])
                train_folds = list(self.dates[chunk_splits[i+1]:chunk_splits[-1]])
                train_folds.extend(list(self.dates[chunk_splits[0]:chunk_splits[i-1]]))
                self.date_train_folds.append(train_folds)
            else:
                self.date_test_folds.append(self.dates[chunk_splits[i]:chunk_splits[i+1]])
                self.date_val_folds.append(self.dates[chunk_splits[i-1]:chunk_splits[i]])
                self.date_train_folds.append(self.dates[chunk_splits[i+1]:chunk_splits[i+1+(self.n_folds-2)]])
        
        return

def bootstrap(n_boot, all_prob_bs, all_sr_bs, all_climo, use_any_srs):
    
    selections = random.choices(range(len(all_prob_bs)), k = len(all_prob_bs))
    
    sampled_prob_bs = all_prob_bs[selections]
    if use_any_srs:
        sampled_sr_bs = all_sr_bs[selections]
    sampled_climo = all_climo[selections]
    
    sampled_climo_bs = np.zeros(len(sampled_climo))
    climo_fcst = np.zeros(90000) + np.mean(sampled_climo)
    
    for j in range(len(sampled_climo)):
        day_events = np.append(np.ones(int(round(sampled_climo[j]*90000,3))),\
                               np.zeros(90000-int(round(sampled_climo[j]*90000,3))))
        sampled_climo_bs[j] = BS(day_events, climo_fcst)
    
    prob_bss = np.mean(sampled_prob_bs - sampled_climo_bs)/np.mean(-sampled_climo_bs)
    if use_any_srs:
        sr_bss = np.mean(sampled_sr_bs - sampled_climo_bs)/np.mean(-sampled_climo_bs)
        return prob_bss, sr_bss
    else:
        return prob_bss
    

def bootstrap_reliability(n_boot, all_prob_rel, all_prob_fcst_freq,\
                          all_sr_rel, all_sr_fcst_freq, all_climo, use_any_srs):
    
    selections = random.choices(range(len(all_prob_rel[0,:])), k = len(all_prob_rel[0,:]))
    
    sampled_prob_rel = all_prob_rel[:,selections]
    sampled_prob_fcst_freq = all_prob_fcst_freq[:,selections]
    if use_any_srs:
        sampled_sr_rel = all_sr_rel[:,selections]
        sampled_sr_fcst_freq = all_sr_fcst_freq[:,selections]
    sampled_climo = all_climo[selections]
    
    climo = np.mean(sampled_climo)
    
    prob_fcst_freq = np.sum(sampled_prob_fcst_freq, axis = 1)
    prob_event_instances = np.sum(sampled_prob_rel, axis = 1)
    prob_rel = np.zeros((len(prob_fcst_freq), 1))
    for i in range(len(prob_fcst_freq)):
        if prob_fcst_freq[i] == 0:
            prob_rel[i] = 0
        else:
            prob_rel[i] = prob_event_instances[i]/prob_fcst_freq[i]
    
    if use_any_srs:
        sr_fcst_freq = np.sum(sampled_sr_fcst_freq, axis = 1)
        sr_event_instances = np.sum(sampled_sr_rel, axis = 1)
        sr_rel = sr_event_instances/sr_fcst_freq
    
    if use_any_srs:
        return prob_rel, prob_fcst_freq, climo, sr_rel, sr_fcst_freq
    else:
        return prob_rel, prob_fcst_freq, climo

def bootstrap_pd(n_boot, all_prob_event_freq, all_prob_fcst_freq_pd, all_prob_total_events,\
                 all_sr_event_freq, all_sr_total_events, use_any_srs):
    
    selections = random.choices(range(len(all_prob_event_freq[0,:])), k = len(all_prob_event_freq[0,:]))
    
    sampled_prob_event_freq = all_prob_event_freq[:,selections]
    sampled_prob_fcst_freq = all_prob_fcst_freq_pd[:,selections]
    sampled_prob_total_events = all_prob_total_events[:,selections]
    if use_any_srs:
        sampled_sr_event_freq = all_sr_event_freq[:,selections]
        sampled_sr_total_events = all_sr_total_events[:,selections]
        sampled_prob_fcst_freq = all_sr_fcst_freq_pd[:,selections]
    
    prob_fcst_freq = np.sum(sampled_prob_fcst_freq, axis = 1)
    prob_total_events = np.sum(sampled_prob_total_events, axis = 1)
    prob_event_instances = np.sum(sampled_prob_event_freq, axis = 1)
    
    prob_pod = np.zeros((len(prob_event_instances), 1))
    prob_sr = np.zeros((len(prob_event_instances), 1))
    for i in range(len(prob_event_instances)):
        if prob_total_events[i] == 0:
            prob_pod[i] = 0
        else:
            prob_pod[i] = prob_event_instances[i]/prob_total_events[i]
        
        if prob_fcst_freq[i] == 0:
            prob_sr[i] = 0
        else:
            prob_sr[i] = prob_event_instances[i]/prob_fcst_freq[i]
    
    if use_any_srs:
        sr_fcst_freq_pd = np.sum(sampled_sr_fcst_freq, axis = 1)
        sr_total_events = np.sum(sampled_sr_total_events, axis = 1)
        sr_event_instances = np.sum(sampled_sr_event_freq, axis = 1)
        sr_pod = sr_total_events/sr_event_instances
        sr_sr = sr_event_instances/sr_fcst_freq
    
    if use_any_srs:
        return prob_pod, prob_sr, sr_pod, sr_sr
    else:
        return prob_pod, prob_sr

def get_bss_fname_dir(home_dir, hazard, train_radius, ver_radius, forecast_length, train_type, ver_type, model_type, use_avg_srs = False):
    if train_type == 'warnings' and ver_type == 'warnings':
        save_dir = '%s/%s_trained/length_%s/%s' %(home_dir, train_type, forecast_length, hazard)
        if use_avg_srs:
            overall_sr_file = '%s_%s_trained_%s_verified_%s_%smin_bss_from_avg_sr_map_by_lead_time.npy'\
            %(model_type, train_type, ver_type, hazard, forecast_length)
            sr_by_fold_file = '%s_%s_trained_%s_verified_%s_%smin_bss_from_avg_sr_map_by_lead_time_and_fold.npy'\
            %(model_type, train_type, ver_type, hazard, forecast_length)
        else:
            overall_sr_file = '%s_%s_trained_%s_verified_%s_%smin_bss_from_sr_map_by_lead_time.npy'\
            %(model_type, train_type, ver_type, hazard, forecast_length)
            sr_by_fold_file = '%s_%s_trained_%s_verified_%s_%smin_bss_from_sr_map_by_lead_time_and_fold.npy'\
            %(model_type, train_type, ver_type, hazard, forecast_length)
        
        overall_probs_file = '%s_%s_trained_%s_verified_%s_%smin_bss_from_raw_probs_by_lead_time.npy'\
        %(model_type, train_type, ver_type, hazard, forecast_length)
        probs_by_fold_file = '%s_%s_trained_%s_verified_%s_%smin_bss_from_raw_probs_by_lead_time_and_fold.npy'\
        %(model_type, train_type, ver_type, hazard, forecast_length)
    
    elif train_type == 'warnings':
        save_dir = '%s/%s_trained/length_%s/%s' %(home_dir, train_type, forecast_length, hazard)
        if use_avg_srs:
            overall_sr_file = '%s_%s_trained_%s_%skm_verified_%s_%smin_bss_from_avg_sr_map_by_lead_time.npy'\
            %(model_type, train_type, ver_type, ver_radius, hazard, forecast_length)
            sr_by_fold_file = '%s_%s_trained_%s_%skm_verified_%s_%smin_%skm_bss_from_avg_sr_map_by_lead_time_and_fold.npy'\
            %(model_type, train_type, ver_type, ver_radius, hazard, forecast_length)
        else:
            overall_sr_file = '%s_%s_trained_%s_%skm_verified_%s_%smin_bss_from_sr_map_by_lead_time.npy'\
            %(model_type, train_type, ver_type, ver_radius, hazard, forecast_length)
            sr_by_fold_file = '%s_%s_trained_%s_%skm_verified_%s_%smin_%skm_bss_from_sr_map_by_lead_time_and_fold.npy'\
            %(model_type, train_type, ver_type, ver_radius, hazard, forecast_length)
        
        overall_probs_file = '%s_%s_trained_%s_%skm_verified_%s_%smin_%skm_bss_from_raw_probs_by_lead_time.npy'\
        %(model_type, train_type, ver_type, ver_radius, hazard, forecast_length)
        probs_by_fold_file = '%s_%s_trained_%s_%skm_verified_%s_%smin_%skm_bss_from_raw_probs_by_lead_time_and_fold.npy'\
        %(model_type, train_type, ver_type, ver_radius, hazard, forecast_length)
    
    elif ver_type == 'warnings':
        save_dir = '%s/%s_trained/length_%s/%s' %(home_dir, train_type, forecast_length, hazard)
        if use_avg_srs:
            overall_sr_file = '%s_%s_%skm_trained_%s_verified_%s_%smin_bss_from_avg_sr_map_by_lead_time.npy'\
            %(model_type, train_type, train_radius, ver_type, ver_radius, hazard, forecast_length)
            sr_by_fold_file = '%s_%s_%skm_trained_%s_verified_%s_%smin_%skm_bss_from_avg_sr_map_by_lead_time_and_fold.npy'\
            %(model_type, train_type, train_radius, ver_type, ver_radius, hazard, forecast_length)
        else:
            overall_sr_file = '%s_%s_%skm_trained_%s_verified_%s_%smin_bss_from_sr_map_by_lead_time.npy'\
            %(model_type, train_type, train_radius, ver_type, ver_radius, hazard, forecast_length)
            sr_by_fold_file = '%s_%s_%skm_trained_%s_verified_%s_%smin_%skm_bss_from_sr_map_by_lead_time_and_fold.npy'\
            %(model_type, train_type, train_radius, ver_type, ver_radius, hazard, forecast_length)
        
        overall_probs_file = '%s_%s_%skm_trained_%s_verified_%s_%smin_%skm_bss_from_raw_probs_by_lead_time.npy'\
        %(model_type, train_type, train_radius, ver_type, ver_radius, hazard, forecast_length)
        probs_by_fold_file = '%s_%s_%skm_trained_%s_verified_%s_%smin_%skm_bss_from_raw_probs_by_lead_time_and_fold.npy'\
        %(model_type, train_type, train_radius, ver_type, ver_radius, hazard, forecast_length)
    
    else:
        save_dir = '%s/%s_trained/length_%s/%s' %(home_dir, train_type, forecast_length, hazard)
        if use_avg_srs:
            overall_sr_file = '%s_%s_%skm_trained_%s_%skm_verified_%s_%smin_bss_from_avg_sr_map_by_lead_time.npy'\
            %(model_type, train_type, train_radius, ver_type, ver_radius, hazard, forecast_length)
            sr_by_fold_file = '%s_%s_%skm_trained_%s_%skm_verified_%s_%smin_bss_from_avg_sr_map_by_lead_time_and_fold.npy'\
            %(model_type, train_type, train_radius, ver_type, ver_radius, hazard, forecast_length)
        else:
            overall_sr_file = '%s_%s_%skm_trained_%s_%skm_verified_%s_%smin_bss_from_sr_map_by_lead_time.npy'\
            %(model_type, train_type, train_radius, ver_type, ver_radius, hazard, forecast_length)
            sr_by_fold_file = '%s_%s_%skm_trained_%s_%skm_verified_%s_%smin_bss_from_sr_map_by_lead_time_and_fold.npy'\
            %(model_type, train_type, train_radius, ver_type, ver_radius, hazard, forecast_length)
        
        overall_probs_file = '%s_%s_%skm_trained_%s_%skm_verified_%s_%smin_bss_from_raw_probs_by_lead_time.npy'\
        %(model_type, train_type, train_radius, ver_type, ver_radius, hazard, forecast_length)
        probs_by_fold_file = '%s_%s_%skm_trained_%s_%skm_verified_%s_%smin_bss_from_raw_probs_by_lead_time_and_fold.npy'\
        %(model_type, train_type, train_radius, ver_type, ver_radius, hazard, forecast_length)
    
    return save_dir, overall_sr_file, overall_probs_file, sr_by_fold_file, probs_by_fold_file

def get_bss_start_i(home_dir, hazard, train_radius, ver_radius, forecast_length, train_type,\
                    ver_type, model_type, use_any_srs, use_avg_srs = False):
    
    save_dir, overall_sr_file, overall_probs_file, sr_by_fold_file, probs_by_fold_file =\
    get_bss_fname_dir(home_dir, hazard, train_radius, ver_radius, forecast_length, train_type,\
                      ver_type, model_type, use_avg_srs = use_avg_srs)
    
    if use_any_srs:
        if not (os.path.isfile('%s/%s' %(save_dir, overall_sr_file)) and\
                os.path.isfile('%s/%s' %(save_dir, overall_probs_file)) and\
                os.path.isfile('%s/%s' %(save_dir, sr_by_fold_file)) and\
                os.path.isfile('%s/%s' %(save_dir, probs_by_fold_file))):
            return 0
    else:
        if not (os.path.isfile('%s/%s' %(save_dir, overall_probs_file)) and\
                os.path.isfile('%s/%s' %(save_dir, probs_by_fold_file))):
            return 0
    
    npy = np.load('%s/%s' %(save_dir, overall_probs_file))
    return npy.size

def get_done_skills(home_dir, hazard, train_radius, ver_radius, forecast_length, train_type,\
                    ver_type, model_type, use_avg_srs = True, use_any_srs = True):
    
    save_dir, overall_sr_file, overall_probs_file, sr_by_fold_file, probs_by_fold_file =\
    get_bss_fname_dir(home_dir, hazard, train_radius, ver_radius, forecast_length,\
                      train_type, ver_type, model_type, use_avg_srs = use_avg_srs)
    
    if use_any_srs:
        overall_srs = np.load('%s/%s' %(save_dir, overall_sr_file))
        sr_by_fold = np.load('%s/%s' %(save_dir, sr_by_fold_file))
    
    probs_by_fold = np.load('%s/%s' %(save_dir, probs_by_fold_file))
    overall_probs = np.load('%s/%s' %(save_dir, overall_probs_file))
    
    if use_any_srs:
        return overall_srs, overall_probs, sr_by_fold, probs_by_fold
    else:
        return overall_probs, probs_by_fold

def get_rel_file(home_dir, model_type, train_type, train_radius, ver_type, ver_radius, hazard, forecast_length, lead, m):
    
    save_dir = '%s/%s_trained/length_%s/%s' %(home_dir, train_type, forecast_length, hazard)
    
    if train_type == 'warnings' and ver_type == 'warnings':
        if m.use_avg_srs:
            rel_fname = '%s_%s_trained_%s_verified_%s_%smin_reliability_data_avg_srs_%s-%s.npy'\
            %(model_type, train_type, ver_type, hazard, forecast_length, lead, lead+forecast_length)
        else:
            rel_fname = '%s_%s_trained_%s_verified_%s_%smin_reliability_data_%s-%s.npy'\
            %(model_type, train_type, ver_type, hazard, forecast_length, lead, lead+forecast_length)
    elif train_type == 'warnings':
        if m.use_avg_srs:
            rel_fname = '%s_%s_trained_%s_%skm_verified_%s_%smin_reliability_data_avg_srs_%s-%s.npy'\
            %(model_type, train_type, ver_type, ver_radius, hazard, forecast_length, lead,\
              lead+forecast_length)
        else:
            rel_fname = '%s_%s_trained_%s_%skm_verified_%s_%smin_reliability_data_%s-%s.npy'\
            %(model_type, train_type, ver_type, ver_radius, hazard, forecast_length,\
              lead, lead+forecast_length)
    elif ver_type == 'warnings':
        if m.use_avg_srs:
            rel_fname = '%s_%s_%skm_trained_%s_verified_%s_%smin_reliability_data_avg_srs_%s-%s.npy'\
            %(model_type, train_type, train_radius, ver_type, hazard, forecast_length, lead,\
              lead+forecast_length)
        else:
            rel_fname = '%s_%s_%skm_trained_%s_verified_%s_%smin_reliability_data_%s-%s.npy'\
            %(model_type, train_type, train_radius, ver_type, hazard, forecast_length, lead,\
              lead+forecast_length)
    else:
        if m.use_avg_srs:
            rel_fname = '%s_%s_%skm_trained_%s_%skm_verified_%s_%smin_reliability_data_avg_srs_%s-%s.npy'\
            %(model_type, train_type, train_radius, ver_type, ver_radius, hazard, forecast_length, lead,\
              lead+forecast_length)
        else:
            rel_fname = '%s_%s_%skm_trained_%s_%skm_verified_%s_%smin_reliability_data_%s-%s.npy'\
            %(model_type, train_type, train_radius, ver_type, ver_radius, hazard, forecast_length,\
              lead, lead+forecast_length)
    
    return save_dir, rel_fname

def get_pd_file(home_dir, model_type, train_type, train_radius, ver_type, ver_radius, hazard, forecast_length, lead, m):
    save_dir = '%s/%s_trained/length_%s/%s' %(home_dir, train_type, forecast_length, hazard)
    if train_type == 'warnings' and ver_type == 'warnings':
        if m.use_avg_srs:
            pd_fname = '%s_%s_trained_%s_verified_%s_%smin_performance_diagram_data_avg_srs_%s-%s.npy'\
            %(model_type, train_type, ver_type, hazard, forecast_length, lead, lead+forecast_length)
        else:
            pd_fname = '%s_%s_trained_%s_verified_%s_%smin_performance_diagram_data_%s-%s.npy'\
            %(model_type, train_type, ver_type, hazard, forecast_length, lead, lead+forecast_length)
    elif train_type == 'warnings':
        if m.use_avg_srs:
            pd_fname = '%s_%s_trained_%s_%skm_verified_%s_%smin_performance_diagram_data_avg_srs_%s-%s.npy'\
            %(model_type, train_type, ver_type, ver_radius, hazard, forecast_length, lead, lead+forecast_length)
        else:
            pd_fname = '%s_%s_trained_%s_%skm_verified_%s_%smin_performance_diagram_data_%s-%s.npy'\
            %(model_type, train_type, ver_type, ver_radius, hazard, forecast_length, lead, lead+forecast_length)
    elif ver_type == 'warnings':
        if m.use_avg_srs:
            pd_fname = '%s_%s_%skm_trained_%s_verified_%s_%smin_performance_diagram_data_avg_srs_%s-%s.npy'\
            %(model_type, train_type, train_radius, ver_type, hazard, forecast_length, lead, lead+forecast_length)
        else:
            pd_fname = '%s_%s_%skm_trained_%s_verified_%s_%smin_performance_diagram_data_%s-%s.npy'\
            %(model_type, train_type, train_radius, ver_type, hazard, forecast_length, lead, lead+forecast_length)
    else:
        if m.use_avg_srs:
            pd_fname = '%s_%s_%skm_trained_%s_%skm_verified_%s_%smin_performance_diagram_data_avg_srs_%s-%s.npy'\
            %(model_type, train_type, train_radius, ver_type, ver_radius, hazard, forecast_length, lead, lead+forecast_length)
        else:
            pd_fname = '%s_%s_%skm_trained_%s_%skm_verified_%s_%smin_performance_diagram_data_%s-%s.npy'\
            %(model_type, train_type, train_radius, ver_type, ver_radius, hazard, forecast_length,\
              lead, lead+forecast_length)
    
    return save_dir, pd_fname

def gather_data(hazards, wofs_spinup_time, forecast_lengths, leads, train_types,\
                ver_types, model_type, num_folds, use_avg_srs, use_any_srs, dates):
    for forecast_length in forecast_lengths:
        print(forecast_length)
        for hazard in hazards:
            print(hazard)
            for lead in leads:
                print(lead)
                if lead + forecast_length > 240:
                    continue
                for i in range(len(train_types)):
                    for j in range(len(ver_types)):
                        train_type = train_types[i]
                        ver_type = ver_types[j]
                        print(train_type, ver_type)
                        inits = ["1700", "1730", "1800", "1830", "1900", "1930",\
                                 "2000", "2030", "2100", "2130", "2200", "2230",\
                                 "2300", "2330", "0000", "0030", "0100", "0130",\
                                 "0200", "0230", "0300", "0330", "0400", "0430",\
                                 "0500"]
                        if hazard == 'hail':
                            train_radius = '39'
                            buffer_str = '_20_min_buffer'
                        elif hazard == 'wind':
                            train_radius = '375'
                            buffer_str = '_20_min_buffer'
                        elif hazard == 'tornado':
                            train_radius = '39'
                            buffer_str = ''
                        ver_radius = train_radius
                        dates = np.genfromtxt('/home/ryan.martz/python_packages/frdd-wofs-'\
                                              'phi/wofs_phi/probSevere_dates.txt').astype(int).astype(str)
                        m = model_stats(hazard, wofs_spinup_time, forecast_length, lead,\
                                        train_radius, ver_radius, train_type, ver_type,\
                                        model_type, num_folds, use_avg_srs, use_any_srs,\
                                        dates, buffer_str)
                        m.save_test_srs_probs_events(inits)
    return

def check_predictor_file(date_str, start_str, end_str):
    
    directory = '/work/eric.loken/wofs/2024_update/SFE2024/fcst/full_npy'
    does_predictor_file_exist_list = []
    start_time = dt.datetime.strptime(date_str + start_str, '%Y%m%d%H%M')
    
    for lead in [30, 60, 90, 120, 150, 180]:
        init_str = (start_time - dt.timedelta(minutes = lead)).strftime('%H%M')
        ps_init_str = ((start_time - dt.timedelta(minutes = lead))\
                        + dt.timedelta(minutes = 24)).strftime('%H%M')
        
        file = 'wofs1d_psv3_with_torp_%s_%s_%s_v%s-%s.npy'\
        %(date_str, init_str, ps_init_str, start_str, end_str)
        
        does_predictor_file_exist_list.append(os.path.exists('%s/%s' %(directory, file)))
    
    does_predictor_file_exist_array = np.array(does_predictor_file_exist_list)
    does_predictor_file_exist = np.any(does_predictor_file_exist_array)
    
    return does_predictor_file_exist

def get_warning_lsr_climo_date(hazard, start_end, date_str, radius, buffer_str):
    
    start_str = start_end[0]
    end_str = start_end[1]
    
    warnings_file = '/work/ryan.martz/wofs_phi_data/training_data/warnings/full_1d_warnings/length_60/%s/%s_warnings_%s_v%s-%s_1d.npy'\
    %(hazard, hazard, date_str, start_str, end_str)
    
    lsr_file = '/work/eric.loken/wofs/2024_update/SFE2024/obs/full_npy/%s_reps1d_%s_v%s-%s_r%skm%s.npy'\
    %(hazard, date_str, start_str, end_str, radius, buffer_str)
    
    if (not os.path.exists(warnings_file)) or (not os.path.exists(lsr_file))\
    or (not check_predictor_file(date_str, start_str, end_str)):
        return -1, -1
    
    warnings = np.load(warnings_file)
    lsrs = np.load(lsr_file)
    
    warning_climo = np.mean(warnings)
    lsr_climo = np.mean(lsrs)
    
    return lsr_climo, warning_climo

def get_warning_lsr_climo_full(boot, hazard, radius):
    
    load_dir = '/work/ryan.martz/wofs_phi_data/training_data/warnings'
    load_warnings_file = '%s_warning_fcst_climos.npy' %(hazard)
    load_lsr_file = '%s_%s_lsr_fcst_climos.npy' %(hazard, radius)
    
    warnings_climos = np.load('%s/%s' %(load_dir, load_warnings_file))
    lsr_climos = np.load('%s/%s' %(load_dir, load_lsr_file))
    
    if not (len(warnings_climos) == len(lsr_climos)):
        print('BADNESS 10000')
    
    rand_inds = np.random.choice(np.arange(len(warnings_climos)), size=len(warnings_climos), replace=True)
    
    rand_warnings = warnings_climos[rand_inds]
    rand_lsrs = lsr_climos[rand_inds]
    
    rand_warning_climos = np.mean(rand_warnings)
    rand_lsr_climos = np.mean(rand_lsrs)
    
    return rand_lsr_climos, rand_warning_climos

def aggregate_warnings_lsrs(hazard, start_ends, dates, radius, buffer_str):
    
    iterator = md.to_iterator([hazard], start_ends, dates, [radius], [buffer_str])
    results = md.run_parallel(get_warning_lsr_climo_date, iterator, nprocs_to_use = 40,\
                              description = 'Gathering Event Climos')
    
    warning_climos = np.zeros(len(results))
    lsr_climos = np.zeros(len(results))
    
    for i in range(len(results)):
        warning_climos[i] = results[i][1]
        lsr_climos[i] = results[i][0]
    
    warning_climos = warning_climos[warning_climos >= 0]
    lsr_climos = lsr_climos[lsr_climos >= 0]
    
    save_dir = '/work/ryan.martz/wofs_phi_data/training_data/warnings'
    save_lsr_file = '%s_%s_lsr_fcst_climos.npy' %(hazard, radius)
    save_warning_file = '%s_warning_fcst_climos.npy' %(hazard)
    
    utilities.save_data(save_dir, save_lsr_file, lsr_climos, 'npy')
    utilities.save_data(save_dir, save_warning_file, warning_climos, 'npy')
    
    return

def bootstrap_warning_lsr_climo(hazard, radius, buffer_str, nBoot = 10000):
    
    dates = np.genfromtxt('/home/ryan.martz/python_packages/frdd-wofs-'\
                          'phi/wofs_phi/probSevere_dates.txt').astype(int).astype(str)
    
    start_ends = [["1730", "1830"], ["1800", "1900"], ["1830", "1930"], ["1900", "2000"], ["1930", "2030"], ["2000", "2100"],\
                  ["2030", "2130"], ["2100", "2200"], ["2130", "2230"], ["2200", "2300"], ["2230", "2230"], ["2300", "0000"],\
                  ["2330", "0030"], ["0000", "0100"], ["0030", "0130"], ["0100", "0200"], ["0130", "0230"], ["0200", "0300"],\
                  ["0230", "0330"], ["0300", "0400"], ["0330", "0430"], ["0400", "0500"], ["0430", "0530"], ["0500", "0600"]]
    
    load_dir = '/work/ryan.martz/wofs_phi_data/training_data/warnings'
    load_warnings_file = '%s_warning_fcst_climos.npy' %(hazard)
    load_lsr_file = '%s_%s_lsr_fcst_climos.npy' %(hazard, radius)
    
    #try:
    #    warnings_climos = np.load('%s/%s' %(load_dir, load_warnings_file))
    #    lsr_climos = np.load('%s/%s' %(load_dir, load_lsr_file))
    #except:
    aggregate_warnings_lsrs(hazard, start_ends, dates, radius, buffer_str)
    
    iterator = md.to_iterator(np.arange(nBoot), [hazard], [radius])
    results = md.run_parallel(get_warning_lsr_climo_full, iterator, nprocs_to_use = 40,\
                              description = 'Bootstrapping Event Climos')
    
    boot_warning_climos = np.zeros(nBoot)
    boot_lsr_climos = np.zeros(nBoot)
    for i in range(len(results)):
        boot_warning_climos[i] = results[i][1]
        boot_lsr_climos[i] = results[i][0]
    
    save_boot_file_warnings = '%s_warning_fcst_climos_bootstrapped.npy' %(hazard)
    save_boot_file_lsrs = '%s_%skm_lsr_fcst_climos_bootstrapped.npy' %(hazard, radius)
    
    utilities.save_data(load_dir, save_boot_file_lsrs, boot_lsr_climos, 'npy')
    utilities.save_data(load_dir, save_boot_file_warnings, boot_warning_climos, 'npy')
    
    return

def gather_fold_data(hazards, wofs_spinup_time, forecast_lengths, leads, train_types,\
                     ver_types, model_type, num_folds, use_avg_srs, use_any_srs, dates):
    for forecast_length in forecast_lengths:
        print(forecast_length)
        for hazard in hazards:
            print(hazard)
            for lead in leads:
                print(lead)
                if lead + forecast_length > 240:
                    continue
                for i in range(len(train_types)):
                    train_type = train_types[i]
                    ver_type = ver_types[i]
                    print(train_type, ver_type)
                    inits = ["1700", "1730", "1800", "1830", "1900",\
                             "1930", "2000", "2030", "2100", "2130",\
                             "2200", "2230", "2300", "2330", "0000",\
                             "0030", "0100", "0130", "0200", "0230",\
                             "0300", "0330", "0400", "0430", "0500"]
                    if hazard == 'hail':
                        train_radius = '39'
                        buffer_str = '_20_min_buffer'
                    elif hazard == 'wind':
                        train_radius = '375'
                        buffer_str = '_20_min_buffer'
                    elif hazard == 'tornado':
                        train_radius = 39
                        buffer_str = ''
                    ver_radius = train_radius
                    m = model_stats(hazard, wofs_spinup_time, forecast_length,\
                                    lead, train_radius, ver_radius, train_type,\
                                    ver_type, model_type, num_folds, use_avg_srs,\
                                    use_any_srs, dates, buffer_str)
                    m.set_chunk_splits()
                    m.save_test_srs_probs_events_by_fold(inits)
    
    return

def calc_bss(home_dir, num_folds, hazards, wofs_spinup_time, forecast_lengths, leads,\
             train_types, ver_types, model_type, use_avg_srs, use_any_srs, start_new, dates):
    print('bss calc')
    for forecast_length in forecast_lengths:
        print(forecast_length)
        for train_type in train_types:
            print(train_type)
            ver_type = train_type
            for hazard in hazards:
                print(hazard)
                sr_skill_by_lead = np.zeros(len(leads))
                probs_skill_by_lead = np.zeros(len(leads))
                sr_skill_by_lead_fold = np.zeros((len(leads), num_folds))
                probs_skill_by_lead_fold = np.zeros((len(leads), num_folds))
                
                if hazard == 'hail':
                    train_radius = '39'
                    buffer_str = '_20_min_buffer'
                elif hazard == 'wind':
                    train_radius = '375'
                    buffer_str = '_20_min_buffer'
                elif hazard == 'tornado':
                    train_radius = 39
                    buffer_str = ''
                ver_radius = train_radius
                
                if start_new:
                    start_i = 0
                else:
                    start_i = get_bss_start_i(home_dir, hazard, train_radius,\
                                              ver_radius, forecast_length, train_type,\
                                              ver_type, model_type, use_any_srs, use_avg_srs)
                if not (start_i == 0):
                    if use_any_srs:
                        sr_skill_by_lead[0:start_i], probs_skill_by_lead[0:start_i],\
                        sr_skill_by_lead_fold[0:start_i,:], probs_skill_by_lead_fold[0:start_i,:] = \
                        get_done_skills(home_dir, hazard, train_radius, ver_radius,\
                                        forecast_length, train_type, ver_type, model_type,\
                                        use_avg_srs, use_any_srs)
                    else:
                        probs_skill_by_lead[0:start_i], probs_skill_by_lead_fold[0:start_i,:] = \
                        get_done_skills(home_dir, hazard, train_radius, ver_radius,\
                                        forecast_length, train_type, ver_type, model_type,\
                                        use_avg_srs, use_any_srs)
                for i in range(start_i, len(leads)):
                    lead = leads[i]
                    print(lead)
                    if forecast_length + lead > 240:
                        continue
                    m = model_stats(hazard, wofs_spinup_time, forecast_length,\
                                    lead, train_radius, ver_radius, train_type,\
                                    ver_type, model_type, num_folds, use_avg_srs,\
                                    use_any_srs, dates, buffer_str)
                    if use_any_srs:
                        try:
                            bss_sr, sr_skill_by_fold, bss_probs, probs_skill_by_fold =\
                            m.gen_brier_skill_scores()
                        except:
                            continue
                        sr_skill_by_lead_fold[i,:] = sr_skill_by_fold
                        sr_skill_by_lead[i] = bss_sr
                    else:
                        bss_probs, probs_skill_by_fold = m.gen_brier_skill_scores()
                    
                    probs_skill_by_lead[i] = bss_probs
                    probs_skill_by_lead_fold[i,:] = probs_skill_by_fold
                
                if np.all(probs_skill_by_lead == np.zeros(len(leads))):
                    continue
                
                save_dir, overall_sr_file, overall_probs_file, sr_by_fold_file,\
                probs_by_fold_file = get_bss_fname_dir(home_dir, hazard, train_radius,\
                                                       ver_radius, forecast_length,\
                                                       train_type, ver_type, model_type,\
                                                       use_avg_srs)
                if use_any_srs:
                    utilities.save_data(save_dir, overall_sr_file, sr_skill_by_lead, 'npy')
                    utilities.save_data(save_dir, sr_by_fold_file, sr_skill_by_lead_fold, 'npy')
                utilities.save_data(save_dir, probs_by_fold_file, probs_skill_by_lead_fold, 'npy')
                utilities.save_data(save_dir, overall_probs_file, probs_skill_by_lead, 'npy')
        
    return

def calc_bootstrap_bss(home_dir, num_folds, hazards, wofs_spinup_time, forecast_lengths, leads,\
                       train_types, ver_types, model_type, use_avg_srs, use_any_srs, start_new, dates,\
                       n_boot, redo_bootstrap, CI):
    
    print('bootstrap bss calc')
    
    for forecast_length in forecast_lengths:
        print(forecast_length)
        for train_type in train_types:
            print('trained: ', train_type)
            for ver_type in ver_types:
                print('trained: ', ver_type)
                for hazard in hazards:
                    print(hazard)
                    if hazard == 'hail':
                        train_radius = '39'
                        buffer_str = '_20_min_buffer'
                    elif hazard == 'wind':
                        train_radius = '375'
                        buffer_str = '_20_min_buffer'
                    elif hazard == 'tornado':
                        train_radius = 39
                        buffer_str = ''
                    ver_radius = train_radius
                    sr_skill_by_lead = np.zeros((3, len(leads)))
                    prob_skill_by_lead = np.zeros((3, len(leads)))
                    for i in range(len(leads)):
                        lead = leads[i]
                        print(lead)
                        if (forecast_length + lead > 240) or ((train_type == 'obs') and (forecast_length == 120)):
                            continue
                        
                        m = model_stats(hazard, wofs_spinup_time, forecast_length,\
                                        lead, train_radius, ver_radius, train_type,\
                                        ver_type, model_type, num_folds, use_avg_srs,\
                                        use_any_srs, dates, buffer_str)
                        
                        save_dir, prob_daily_bss_save_fname, sr_daily_bss_save_fname =\
                        m.get_bootstrap_bss_fnames(home_dir)
                        
                        if use_any_srs:
                            if (redo_bootstrap) or (not (os.path.exists('%s/%s' %(save_dir, prob_daily_bss_save_fname))\
                                                          and (os.path.exists('%s/%s' %(save_dir, sr_daily_bss_save_fname))))):
                                daily_prob_bss, daily_sr_bss = m.get_bootstrapping_bs_data(n_boot, home_dir, use_any_srs)
                            
                            else:
                                daily_prob_bss = np.load('%s/%s' %(save_dir, prob_daily_bss_save_fname))
                                daily_sr_bss = np.load('%s/%s' %(save_dir, sr_daily_bss_save_fname))
                        else:
                            if (redo_bootstrap) or (not (os.path.exists('%s/%s' %(save_dir, prob_daily_bss_save_fname)))):
                                daily_prob_bss = m.get_bootstrapping_bs_data(n_boot, home_dir, use_any_srs)
                            
                            else:
                                daily_prob_bss = np.load('%s/%s' %(save_dir, prob_daily_bss_save_fname))
                            
                        upper_bound = 100 - ((100-CI)/2)
                        lower_bound = (100 - CI)/2
                        
                        if use_any_srs:
                            sr_skill_by_lead[0,i] = np.percentile(daily_sr_bss, upper_bound)
                            sr_skill_by_lead[1,i] = np.mean(daily_sr_bss)
                            sr_skill_by_lead[2,i] = np.percentile(daily_sr_bss, lower_bound)
                        else:
                            sr_skill_by_lead[0:2,i] -= 1
                        
                        prob_skill_by_lead[0,i] = np.percentile(daily_prob_bss, upper_bound)
                        prob_skill_by_lead[1,i] = np.mean(daily_prob_bss)
                        prob_skill_by_lead[2,i] = np.percentile(daily_prob_bss, lower_bound)
                        
                    save_dir, prob_bootstrap_bss_save_fname, sr_bootstrap_bss_save_fname =\
                    m.get_bootstrap_summary_bss_fnames(home_dir, CI)
                    
                    utilities.save_data(save_dir, prob_bootstrap_bss_save_fname, prob_skill_by_lead, 'npy')
                    utilities.save_data(save_dir, sr_bootstrap_bss_save_fname, sr_skill_by_lead, 'npy')
                        
    return

def calc_bootstrap_reliability(home_dir, num_folds, hazards, wofs_spinup_time,\
                               forecast_lengths, leads, train_types, ver_types, model_type,\
                               use_avg_srs, use_any_srs, dates, redo_bootstrap, CI, n_boot):
    
    ########################## Calc Reliability ##########################
    '''Documentation on Reliability Summary Files (Content by Row):
    1. Reliability Score from SR forecasts (2.5, mean, 97.5)
    2. Reliability Score from raw prob forecasts (2.5, mean, 97.5)
    3. Overall climo of forecasted event (2.5, mean, 97.5)
    4. The point forecasts to use for plotting (ex: 0.05, 0.15, 0.25, etc. for 0-0.1, 0.1-0.2, 0.2-0.3, etc.
    5. The event climatology given such a forecast from the SRs (2.5)
    6. The event climatology given such a forecast from the SRs (mean)
    7. The event climatology given such a forecast from the SRs (97.5)
    8. The frequency of forecasts in each range for SR probs (2.5)
    9. The frequency of forecasts in each range for SR probs (mean)
    10. The frequency of forecasts in each range for SR probs (97.5)
    11. The event climatology given such a forecast from the raw probs (2.5)
    12. The event climatology given such a forecast from the raw probs (mean)
    13. The event climatology given such a forecast from the raw probs (97.5)
    14. The frequency of forecasts in each range for raw probs (2.5)
    15. The frequency of forecasts in each range for raw probs (mean)
    16. The frequency of forecasts in each range for raw probs (97.5)'''
    
    print('bootstrapping rel data:')
    for forecast_length in forecast_lengths:
        print(forecast_length)
        for train_type in train_types:
            print('trained: ', train_type)
            for ver_type in ver_types:
                print('verified: ', ver_type)
                for hazard in hazards:
                    print(hazard)
                    if hazard == 'hail':
                        train_radius = '39'
                        buffer_str = '_20_min_buffer'
                    elif hazard == 'wind':
                        train_radius = '375'
                        buffer_str = '_20_min_buffer'
                    elif hazard == 'tornado':
                        train_radius = 39
                        buffer_str = ''
                    ver_radius = train_radius
                    for i in range(len(leads)):
                        lead = leads[i]
                        print(lead)
                        if (forecast_length + lead > 240) or ((train_type == 'obs') and (forecast_length == 120)):
                            continue
                        m = model_stats(hazard, wofs_spinup_time, forecast_length, lead,\
                                        train_radius, ver_radius, train_type, ver_type, model_type, num_folds,
                                        use_avg_srs, use_any_srs, dates, buffer_str)
                        points = np.arange(0.05, 1, 0.1)
                        bins = np.arange(0, 1.01, 0.1)
                        nbins = len(bins)-1
                        
                        save_dir, prob_daily_rel_save_fname, sr_daily_rel_save_fname =\
                        m.get_bootstrap_rel_fnames(home_dir)
                        
                        if use_any_srs:
                            if (redo_bootstrap) or (not (os.path.exists('%s/%s' %(save_dir, prob_daily_rel_save_fname))\
                                                         and (os.path.exists('%s/%s' %(save_dir, sr_daily_rel_save_fname))))):
                                daily_prob_rel, daily_sr_rel = m.get_bootstrapping_rel_data(n_boot, home_dir, use_any_srs, points)
                            else:
                                daily_prob_rel = np.load('%s/%s' %(save_dir, prob_daily_rel_save_fname))
                                daily_sr_rel = np.load('%s/%s' %(save_dir, sr_daily_rel_save_fname))
                        else:
                            if (redo_bootstrap) or (not (os.path.exists('%s/%s' %(save_dir, prob_daily_rel_save_fname)))):
                                 daily_prob_rel = m.get_bootstrapping_rel_data(n_boot, home_dir, use_any_srs, points)
                            else:
                                daily_prob_rel = np.load('%s/%s' %(save_dir, prob_daily_rel_save_fname))
                        
                        #Daily Reliability Arrays:
                        #1. Rel Scores
                        #2. Climatologies
                        #3:x(4+len(bins)-1). fcst frequency (one row per bin)
                        #x:y(x+len(bins)-1). event climos (one row per bin)
                        
                        x = len(bins)-1 + 2
                        y = len(bins)-1 + x
                        
                        rel_summary_data = np.zeros((16, points.size))
                        
                        upper_bound = 100 - ((100-CI)/2)
                        lower_bound = (100 - CI)/2
                        
                        if use_any_srs:
                            rel_summary_data[0,0] = np.percentile(daily_sr_rel[0,:], lower_bound)
                            rel_summary_data[0,1] = np.mean(daily_sr_rel[0,:])
                            rel_summary_data[0,2] = np.percentile(daily_sr_rel[0,:], upper_bound)
                            
                            rel_summary_data[4,:] = np.percentile(daily_sr_rel[2:x,:], lower_bound, axis=1)
                            rel_summary_data[5,:] = np.mean(daily_sr_rel[2:x,:], axis=1)
                            rel_summary_data[6,:] = np.percentile(daily_sr_rel[2:x,:], upper_bound, axis=1)
                            
                            rel_summary_data[7,:] = np.percentile(daily_sr_rel[x:y,:], lower_bound, axis=1)
                            rel_summary_data[8,:] = np.mean(daily_sr_rel[x:y,:], axis=1)
                            rel_summary_data[9,:] = np.percentile(daily_sr_rel[x:y,:], upper_bound, axis=1)
                        else:
                            rel_summary_data[0,0:3] -= 1
                            rel_summary_data[4:10,:] -= 1
                        
                        rel_summary_data[1,0] = np.percentile(daily_prob_rel[0,:], lower_bound)
                        rel_summary_data[1,1] = np.mean(daily_prob_rel[0,:])
                        rel_summary_data[1,2] = np.percentile(daily_prob_rel[0,:], upper_bound)
                        
                        rel_summary_data[2,0] = np.percentile(daily_prob_rel[1,:], lower_bound)
                        rel_summary_data[2,1] = np.mean(daily_prob_rel[1,:])
                        rel_summary_data[2,2] = np.percentile(daily_prob_rel[1,:], upper_bound)
                        
                        rel_summary_data[3,:] = points
                        
                        rel_summary_data[10,:] = np.percentile(daily_prob_rel[2:x,:], lower_bound, axis=1)
                        rel_summary_data[11,:] = np.mean(daily_prob_rel[2:x,:], axis=1)
                        rel_summary_data[12,:] = np.percentile(daily_prob_rel[2:x,:], upper_bound, axis=1)
                        
                        rel_summary_data[13,:] = np.percentile(daily_prob_rel[x:y,:], lower_bound, axis=1)
                        rel_summary_data[14,:] = np.mean(daily_prob_rel[x:y,:], axis=1)
                        rel_summary_data[15,:] = np.percentile(daily_prob_rel[x:y,:], upper_bound, axis=1)
                        
                        save_dir, bootstrap_rel_save_fname = m.get_bootstrap_summary_rel_fnames(home_dir, CI)
                        
                        utilities.save_data(save_dir, bootstrap_rel_save_fname, rel_summary_data, 'npy')
                        time.sleep(2)
    
    return

def calc_bootstrap_pd(home_dir, num_folds, hazards, wofs_spinup_time,\
                               forecast_lengths, leads, train_types, ver_types, model_type,\
                               use_avg_srs, use_any_srs, dates, redo_bootstrap, CI, n_boot):
    
    ########################## Calc PD Stats ##########################
    '''Documentation on PD Summary Files (Content by Row):
    1. The point thresholds to use for plotting (ex: 0.05, 0.15, 0.25)
    2. The POD for each threshold for SR forecasts (2.5)
    3. The POD for each threshold for SR forecasts (mean)
    4. The POD for each threshold for SR forecasts (97.5)
    5. The SR for each threshold for SR forecasts (2.5)
    6. The SR for each threshold for SR forecasts (mean)
    7. The SR for each threshold for SR forecasts (97.5)
    8. The POD for each threshold for prob forecasts (2.5)
    9. The POD for each threshold for prob forecasts (mean)
    10. The POD for each threshold for prob forecasts (97.5)
    11. The SR for each threshold for prob forecasts (2.5)
    12. The SR for each threshold for prob forecasts (mean)
    13. The SR for each threshold for prob forecasts (97.5)'''
    
    print('bootstrapping pd data:')
    for forecast_length in forecast_lengths:
        print(forecast_length)
        for train_type in train_types:
            print('trained: ', train_type)
            for ver_type in ver_types:
                print('verified: ', ver_type)
                for hazard in hazards:
                    print(hazard)
                    if hazard == 'hail':
                        train_radius = '39'
                        buffer_str = '_20_min_buffer'
                    elif hazard == 'wind':
                        train_radius = '375'
                        buffer_str = '_20_min_buffer'
                    elif hazard == 'tornado':
                        train_radius = 39
                        buffer_str = ''
                    ver_radius = train_radius
                    for i in range(len(leads)):
                        lead = leads[i]
                        print(lead)
                        if (forecast_length + lead > 240) or ((train_type == 'obs') and (forecast_length == 120)):
                            continue
                        m = model_stats(hazard, wofs_spinup_time, forecast_length, lead,\
                                        train_radius, ver_radius, train_type, ver_type, model_type, num_folds,
                                        use_avg_srs, use_any_srs, dates, buffer_str)
                        points = np.arange(0.05, 1, 0.1)
                        bins = np.arange(0, 1.01, 0.1)
                        nbins = len(bins)-1
                        
                        if hazard == 'tornado':
                            thresholds = np.array([0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,\
                                                   0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
                        else:
                            thresholds = bins
                        
                        nthresh = len(thresholds)
                        
                        save_dir, prob_daily_pd_save_fname, sr_daily_pd_save_fname =\
                        m.get_bootstrap_pd_fnames(home_dir)
                        
                        if use_any_srs:
                            if (redo_bootstrap) or (not (os.path.exists('%s/%s' %(save_dir, prob_daily_pd_save_fname))\
                                                         and (os.path.exists('%s/%s' %(save_dir, sr_daily_pd_save_fname))))):
                                daily_prob_pd, daily_sr_pd = m.get_bootstrapping_pd_data(n_boot, home_dir, use_any_srs, thresholds)
                            else:
                                daily_prob_pd = np.load('%s/%s' %(save_dir, prob_daily_pd_save_fname))
                                daily_sr_pd = np.load('%s/%s' %(save_dir, sr_daily_pd_save_fname))
                        else:
                            if (redo_bootstrap) or (not (os.path.exists('%s/%s' %(save_dir, prob_daily_pd_save_fname)))):
                                daily_prob_pd = m.get_bootstrapping_pd_data(n_boot, home_dir, use_any_srs, thresholds)
                            else:
                                daily_prob_pd = np.load('%s/%s' %(save_dir, prob_daily_pd_save_fname))
                        
                        #Daily PD Arrays:
                        #1:x. PODs (one row per threshold)
                        #x:y. SRs (one row per threshold)
                        
                        x = nthresh
                        y = nthresh + x
                        
                        pd_summary_data = np.zeros((13, thresholds.size))
                        pd_summary_data[0,:] = thresholds
                        
                        upper_bound = 100 - ((100-CI)/2)
                        lower_bound = (100 - CI)/2
                        
                        if use_any_srs:
                            pd_summary_data[1,:] = np.percentile(daily_sr_pd[0:x,:], lower_bound, axis=1)
                            pd_summary_data[2,:] = np.mean(daily_sr_pd[0:x,:], axis=1)
                            pd_summary_data[3,:] = np.percentile(daily_sr_pd[0:x,:], upper_bound, axis=1)
                            
                            pd_summary_data[4,:] = np.percentile(daily_sr_pd[x:y,:], lower_bound, axis=1)
                            pd_summary_data[5,:] = np.mean(daily_sr_pd[x:y,:], axis=1)
                            pd_summary_data[6,:] = np.percentile(daily_sr_pd[x:y,:], upper_bound, axis=1)
                        else:
                            pd_summary_data[1:7,:] -= 1
                        
                        pd_summary_data[7,:] = np.percentile(daily_prob_pd[0:x,:], lower_bound, axis=1)
                        pd_summary_data[8,:] = np.mean(daily_prob_pd[0:x,:], axis=1)
                        pd_summary_data[9,:] = np.percentile(daily_prob_pd[0:x,:], upper_bound, axis=1)
                        
                        pd_summary_data[10,:] = np.percentile(daily_prob_pd[x:y,:], lower_bound, axis=1)
                        pd_summary_data[11,:] = np.mean(daily_prob_pd[x:y,:], axis=1)
                        pd_summary_data[12,:] = np.percentile(daily_prob_pd[x:y,:], upper_bound, axis=1)
                        
                        save_dir, bootstrap_pd_save_fname = m.get_bootstrap_summary_pd_fnames(home_dir, CI)
                        
                        utilities.save_data(save_dir, bootstrap_pd_save_fname, pd_summary_data, 'npy')
                        time.sleep(2)
    
    return

def calc_reliability(home_dir, num_folds, hazards, wofs_spinup_time, forecast_lengths, leads,\
                     train_types, ver_types, model_type, use_avg_srs, use_any_srs, dates):
    
    ########################## Calc Reliability ##########################
    '''Documentation on Reliability Summary Files (Content by Row):
    1. Reliability Score from SR forecasts, Reliability Score from raw prob forecasts, Overall climo of forecasted event, 0s the rest of the row
    2. The point forecasts to use for plotting (ex: 0.05, 0.15, 0.25, etc. for 0-0.1, 0.1-0.2, 0.2-0.3, etc.
    3. The frequency of forecasts in each range for SR probs
    4. The event climatology given such a forecast from the SRs
    5. The frequency of forecasts in each range for raw probs
    6. The event climatology given such a forecast from the raw probs'''
    print('rel data:')
    for forecast_length in forecast_lengths:
        print(forecast_length)
        for train_type in train_types:
            print('trained: ', train_type)
            for ver_type in ver_types:
                print('verified: ', ver_type)
                for hazard in hazards:
                    print(hazard)
                    if hazard == 'hail':
                        train_radius = '39'
                        buffer_str = '_20_min_buffer'
                    elif hazard == 'wind':
                        train_radius = '375'
                        buffer_str = '_20_min_buffer'
                    elif hazard == 'tornado':
                        train_radius = 39
                        buffer_str = ''
                    ver_radius = train_radius
                    for i in range(len(leads)):
                        lead = leads[i]
                        print(lead)
                        if (forecast_length + lead > 240) or ((train_type == 'obs') and (forecast_length == 120)):
                            continue
                        m = model_stats(hazard, wofs_spinup_time, forecast_length, lead, train_radius,\
                                        ver_radius, train_type, ver_type, model_type, num_folds,
                                        use_avg_srs, use_any_srs, dates, buffer_str)
                        points = np.arange(0.05, 1, 0.1)
                        bins = np.arange(0, 1.01, 0.1)
                        try:
                            if use_any_srs:
                                fcst_frequencies_srs, fcst_frequencies_probs, rel_srs, rel_probs,\
                                rel_climos_srs, rel_climos_probs, ver_climo = m.calc_reliability(bins)
                            else:
                                fcst_frequencies_probs, rel_probs, rel_climos_probs, ver_climo = m.calc_reliability(bins)
                                rel_srs = -1
                                fcst_frequencies_srs = np.zeros((1, points.size)) - 1
                                rel_climos_srs = np.zeros((1, points.size)) - 1
                        except:
                            print('failed on: ', forecast_length, train_type, ver_type)
                            continue
                        rel_summary_data = np.zeros((6, points.size))
                        rel_summary_data[0,0] = rel_srs
                        rel_summary_data[0,1] = rel_probs
                        rel_summary_data[0,2] = ver_climo
                        rel_summary_data[1,:] = points
                        rel_summary_data[2,:] = fcst_frequencies_srs
                        rel_summary_data[3,:] = rel_climos_srs
                        rel_summary_data[4,:] = fcst_frequencies_probs
                        rel_summary_data[5,:] = rel_climos_probs
                        
                        save_dir, rel_fname = get_rel_file(home_dir, model_type, train_type, train_radius,\
                                                           ver_type, ver_radius, hazard, forecast_length, lead, m)
                        
                        utilities.save_data(save_dir, rel_fname, rel_summary_data, 'npy')
                        time.sleep(2)
    
    return
                    
def calc_pd(home_dir, num_folds, hazards, wofs_spinup_time, forecast_lengths, leads,\
            train_types, ver_types, model_type, use_avg_srs, use_any_srs, dates):
    
    ########################## Calc Performance Diagram Stats ##########################
    '''Performance Diagram Summary Data Files by Row:
    1. Forecast Thresholds for SR data
    2. POD using SR probs
    3. SR using SR probs
    4. Forecast Thresholds for raw probs
    5. POD using raw probs
    6. SR using raw probs'''
    print('pd stats:')
    for forecast_length in forecast_lengths:
        print(forecast_length)
        for train_type in train_types:
            print('trained: ', train_type)
            for ver_type in ver_types:
                print('verified: ', ver_type)
                for hazard in hazards:
                    print(hazard)
                    if hazard == 'hail':
                        train_radius = '39'
                        buffer_str = '_20_min_buffer'
                    elif hazard == 'wind':
                        train_radius = '375'
                        buffer_str = '_20_min_buffer'
                    elif hazard == 'tornado':
                        train_radius = 39
                        buffer_str = ''
                    ver_radius = train_radius
                    for i in range(len(leads)):
                        lead = leads[i]
                        print(lead)
                        if (forecast_length + lead > 240) or ((train_type == 'obs') and (forecast_length == 120)):
                            continue
                        m = model_stats(hazard, wofs_spinup_time, forecast_length, lead, train_radius, ver_radius,\
                                        train_type, ver_type, model_type, num_folds, 
                                        use_avg_srs, use_any_srs, dates, buffer_str)
                        if hazard == 'tornado':
                            thresholds = np.array([0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
                        else:
                            thresholds = np.arange(0, 1.01, 0.1)
                        
                        try:
                            if use_any_srs:
                                thresholds_sr, pods_sr, srs_sr, thresholds_probs, pods_probs, srs_probs = m.calc_pod_sr(thresholds)
                            else:
                                thresholds_probs, pods_probs, srs_probs = m.calc_pod_sr(thresholds)
                                thresholds_sr = np.zeros((1, thresholds.size)) - 1
                                pods_sr = np.zeros((1, thresholds.size)) - 1
                                srs_sr = np.zeros((1, thresholds.size)) - 1
                        except:
                            print(forecast_length, train_type, ver_type)
                            continue
                        pd_summary_data = np.zeros((6, thresholds.size))
                        pd_summary_data[0,:] = thresholds_sr
                        pd_summary_data[1,:] = pods_sr
                        pd_summary_data[2,:] = srs_sr
                        pd_summary_data[3,:] = thresholds_probs
                        pd_summary_data[4,:] = pods_probs
                        pd_summary_data[5,:] = srs_probs
                        
                        save_dir, pd_fname = get_pd_file(home_dir, model_type, train_type,\
                                                         train_radius, ver_type, ver_radius, hazard, forecast_length, lead, m)

                        utilities.save_data(save_dir, pd_fname, pd_summary_data, 'npy')
                        
                        time.sleep(2)
    
    return

def transfer_models(home_dir, model_type, hazards, train_radii, wofs_lag, lengths, leads, train_types, use_avg_srs):
    sr_paste_dir = '/work/eric.loken/wofs/2024_update/SFE2024/sr_csv/latest'
    model_paste_dir = '/work/eric.loken/wofs/2024_update/SFE2024/rf_models/latest'
    for length in lengths:
        for train_type in train_types:
            ver_type = train_type
            validation_dir = '/work/ryan.martz/wofs_phi_data/%s_train/validation_fcsts/%s' %(train_type, model_type)
            train_dir = '/work/ryan.martz/wofs_phi_data/%s_train/models/%s' %(train_type, model_type)
            for haz in hazards:
                for train_r in train_radii:
                    ver_r = train_r
                    
                    sr_skills_dir, __, __, sr_skills_fname, __ = get_bss_fname_dir(home_dir, haz, train_r,\
                                                                                   ver_r, length, train_type,\
                                                                                   ver_type, model_type, use_avg_srs)
                    try:
                        sr_skill_by_lead_time_fold = np.load('%s/%s' %(sr_skills_dir, sr_skills_fname))
                    except:
                        continue
                    
                    for i in range(len(leads)):
                        lead = leads[i]
                        
                        if lead+length > 240:
                            continue

                        ideal_fold = np.where(sr_skill_by_lead_time_fold[i,:] == max(sr_skill_by_lead_time_fold[i,:]))[0][0]
                        
                        if use_avg_srs:
                            sr_dir = '%s/%s/wofslag_%s/length_%s' %(validation_dir, haz, wofs_lag, length)
                        else:
                            sr_dir = '%s/%s/wofslag_%s/length_%s/all_raw_probs_fold%s' %(validation_dir, haz, wofs_lag,\
                                                                                         length, ideal_fold)

                        model_dir = '%s/%s/wofslag_%s/length_%s' %(train_dir, haz, wofs_lag, length)
                        if train_type == 'warnings':
                            if use_avg_srs:
                                sr_fname = '%s_%s_trained_avg_sr_map_%s_spinup%smin_length%smin_%s-%s.csv'\
                                %(model_type, train_type, haz, wofs_lag, length, lead, lead+length)
                            else:
                                sr_fname = '%s_%s_trained_sr_map_%s_spinup%smin_length%smin_%s-%s_fold%s.csv'\
                                %(model_type, train_type, haz, wofs_lag, length, lead, lead+length, ideal_fold)
                                
                            sr_paste_name = '%s_%s-%smin_%s_sr_map.csv' %(train_type, lead, lead+length, haz)
                        else:
                            if use_avg_srs:
                                sr_fname = '%s_%s_trained_avg_sr_map_%s_spinup%smin_length%smin_%s-%s_r%skm.csv'\
                                %(model_type, train_type, haz, wofs_lag, length, lead, lead+length, train_r)
                            else:
                                sr_fname = '%s_%s_trained_sr_map_%s_spinup%smin_length%smin_%s-%s_r%skm_fold%s.csv'\
                                %(model_type, train_type, haz, wofs_lag, length, lead, lead+length, train_r, ideal_fold)
                            sr_paste_name = '%s_%skm_%s-%smin_%s_sr_map.csv' %(train_type, train_r,\
                                                                                  lead, lead+length, haz)
                            
                        model_fname = '%s_%s_trained_wofsphi_%s_%smin_window%s-%s_r%skm_fold%s.pkl'\
                        %(model_type, train_type, haz, length, lead, lead+length, train_r, ideal_fold)
                        model_paste_fname = '%s_trained_wofsphi_%s_%smin_window%s-%s_r%skm.pkl'\
                        %(train_type, haz, length, lead, lead+length, train_r)
                        
                        full_sr_fname = '%s/%s' %(sr_dir, sr_fname)
                        full_model_fname = '%s/%s' %(model_dir, model_fname)
                        
                        copy(full_sr_fname, '%s/%s' %(sr_paste_dir, sr_paste_name))
                        copy(full_model_fname, '%s/%s' %(model_paste_dir, model_paste_fname))
    
    return
                    
                    
def remove_old_data(model_type, lengths, train_types, hazards):
    for length in lengths:
        for train_type in train_types:
            for haz in hazards:
                direc_a = '/work/ryan.martz/wofs_phi_data/%s_train/test_fcsts/%s/%s/wofslag_25/length_%s'\
                %(train_type, model_type, haz, length)
                
                for file in os.listdir(direc_a):
                    if '.npy' in file:
                        os.remove('%s/%s' %(direc_a, file))
                
                direc_b = '/work/ryan.martz/wofs_phi_data/experiments/%s_trained/length_%s/%s'\
                %(train_type, length, haz)
                
                for file in os.listdir(direc_b):
                    if '.npy' in file:
                        os.remove('%s/%s' %(direc_b, file))
                
                #for date in os.listdir(direc):
                #    if '.npy' in date or 'fold' in date:
                #        continue
                #    for init in os.listdir('%s/%s' %(direc, date)):
                #        final_dir = '%s/%s/%s' %(direc, date, init)
                #        for file in os.listdir(final_dir):
                #            creation_time = dt.datetime.fromtimestamp(os.stat('%s/%s' %(final_dir, file)).st_mtime)
                #            if creation_time.month < 10:
                #                try:
                #                    os.remove('%s/%s' %(final_dir, file))
                #                except:
                #                    continue
    return

def main():
    
    home_dir = '/work/ryan.martz/wofs_phi_data/experiments'
    dates = np.genfromtxt('/home/ryan.martz/python_packages/frdd-wofs-phi/wofs_phi/probSevere_dates.txt').astype(int).astype(str)
    
    num_folds = 5
    hazards = ['hail', 'wind']
    #radii = [37, 375, 38]#, 39]
    wofs_spinup_time = 25
    forecast_lengths = [60]#, 120]
    leads = [30, 60, 90, 120, 150, 180]
    train_types = ['obs', 'warnings']
    ver_types = ['obs', 'warnings']
    ps_version = 3 #i.e., which probSevere version is being used 2 for v2, 3 for v3
    ps_flag = '' #NoTor
    include_torp_in_predictors = True
    radar_data = True
    filtered_torp = False
    n_boot = 10000
    CI = 95 #confidence interval percent
    
    start_new = True
    do_remove_old_data = False
    do_gather_data = False
    do_fold_gathering = False
    do_calc_bss = False
    do_bootstrap_bss = True
    do_bootstrap_rel = True
    do_bootstrap_pd = True
    redo_bootstrapping = False
    do_calc_rel = False
    do_calc_pd = False
    do_transfer_models = False
    
    use_avg_srs = True
    use_any_srs = False
    
    num_wofs_vars = 177
    num_misc_vars = 2
    num_ps_vars = 0
    if ps_version > 0 and ps_flag == 'NoTor':
        num_ps_vars -= 30
    num_torp_vars = 0

    if (ps_version == 2) and (not include_torp_in_predictors):
        model_type = 'wofs_psv2%s_no_torp' %(ps_flag)
        num_ps_vars += 90
    elif (ps_version == 2) and (include_torp_in_predictors):
        model_type = 'wofs_psv2%s_with_torp' %(ps_flag)
        num_torp_vars = 105
        num_ps_vars += 90
    elif (ps_version == 3) and (not include_torp_in_predictors):
        model_type = 'wofs_psv3%s_no_torp' %(ps_flag)
        num_ps_vars += 90
    elif (ps_version == 3) and (include_torp_in_predictors):
        if radar_data and (not filtered_torp):
            model_type = 'wofs_psv3%s_with_torp' %(ps_flag)
            num_torp_vars = 105
            num_ps_vars += 90
        elif radar_data and filtered_torp:
            model_type = 'wofs_psv3%s_with_torp_filtered' %(ps_flag)
            num_torp_vars = 105
            num_ps_vars += 90
        elif (not radar_data) and filtered_torp:
            model_type = 'wofs_psv3%s_with_torp_filtered_p_only' %(ps_flag)
            num_torp_vars = 25
            num_ps_vars += 90
        elif (not radar_data) and (not filtered_torp):
            model_type = 'wofs_psv3%s_with_torp_p_only' %(ps_flag)
            num_torp_vars = 25
            num_ps_vars += 90
    elif (ps_version == 0) and (include_torp_in_predictors):
        num_torp_vars = 105
        model_type = 'wofs_psv0_with_torp'
    
    if do_remove_old_data:
        remove_old_data(model_type, forecast_lengths, train_types, hazards)
    if do_gather_data:
        print('gathering data')
        gather_data(hazards, wofs_spinup_time, forecast_lengths, leads,\
                    train_types, ver_types, model_type, num_folds, use_avg_srs,\
                    use_any_srs, dates)
    if do_fold_gathering:
        print('gathering fold data')
        gather_fold_data(hazards, wofs_spinup_time, forecast_lengths,\
                         leads, train_types, ver_types, model_type, num_folds,\
                         use_avg_srs, use_any_srs, dates)
    if do_calc_bss:
        calc_bss(home_dir, num_folds, hazards, wofs_spinup_time, forecast_lengths,\
                 leads, train_types, ver_types, model_type, use_avg_srs, use_any_srs,\
                 start_new, dates)
        print('bss calc done')
    if do_bootstrap_bss:
        print('bootstrapping bss')
        calc_bootstrap_bss(home_dir, num_folds, hazards, wofs_spinup_time, forecast_lengths,\
                 leads, train_types, ver_types, model_type, use_avg_srs, use_any_srs,\
                 start_new, dates, n_boot, redo_bootstrapping, CI)
        print('bootstrapping bss done')
    if do_calc_rel:
        calc_reliability(home_dir, num_folds, hazards, wofs_spinup_time,\
                         forecast_lengths, leads, train_types, ver_types, model_type,\
                         use_avg_srs, use_any_srs, dates)
        print('reliability calc done')
    if do_bootstrap_rel:
        print('bootstrapping reliability')
        calc_bootstrap_reliability(home_dir, num_folds, hazards, wofs_spinup_time,\
                               forecast_lengths, leads, train_types, ver_types, model_type,\
                               use_avg_srs, use_any_srs, dates, redo_bootstrapping, CI, n_boot)
        print('bootstrapping reliabilty done')
    if do_calc_pd:
        calc_pd(home_dir, num_folds, hazards, wofs_spinup_time, forecast_lengths,\
                leads, train_types, ver_types, model_type, use_avg_srs, use_any_srs, dates)
        print('pd calc done')
    if do_bootstrap_pd:
        print('bootstrapping PD')
        calc_bootstrap_pd(home_dir, num_folds, hazards, wofs_spinup_time,\
                               forecast_lengths, leads, train_types, ver_types, model_type,\
                               use_avg_srs, use_any_srs, dates, redo_bootstrapping, CI, n_boot)
        print('bootstrapping PD done')
    if do_transfer_models:
        print('transferring models')
        transfer_models(home_dir, model_type, hazards, radii, wofs_spinup_time,\
                        forecast_lengths, leads, train_types, use_avg_srs)
        print('models transferred')
    
    #print('aggregating warnings/lsrs')
    #hazards = ['wind']
    #radii = [375]#, 40, 41, 42]
    #for h in hazards:
    #    if hazard == 'hail':
    #        radius = '39'
    #        buffer_str = '_20_min_buffer'
    #    elif hazard == 'wind':
    #        radius = '375'
    #        buffer_str = '_20_min_buffer'
    #    elif hazard == 'tornado':
    #        radius = '39'
    #        buffer_str = ''
    #    bootstrap_warning_lsr_climo(h, radius, buffer_str, nBoot = 10000)
    
    
    print('done')
    
    return


if (__name__ == '__main__'):

    main()