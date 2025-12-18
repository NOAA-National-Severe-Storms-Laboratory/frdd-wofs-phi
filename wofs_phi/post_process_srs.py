import pandas as pd
import numpy as np
from . import config as c


def get_avg_sr_fname_dir(haz, train_type, lead, r):
    validation_dir = '/work/ryan.martz/wofs_phi_data/%s_train/validation_fcsts/%s' %(train_type, c.model_type)
    sr_dir = '%s/%s/wofslag_%s/length_%s' %(validation_dir, haz, c.wofs_spinup_time, c.forecast_length)
    sr_fname = '%s_%s_trained_avg_sr_map_%s_spinup%smin_length%smin_%s-%s_r%skm.csv' %(c.model_type, train_type, haz, c.wofs_spinup_time, c.forecast_length, lead,
                                                                                       lead+c.forecast_length, r)
    return sr_dir, sr_fname

def main():
    ################## Post Process LSR + Warnings based on lead time ##################
    train_type = 'obs_and_warnings'
    for r in c.train_radii:
        for haz in c.train_hazards:
            for i in range(1, len(c.train_lead_times)):

                lead_curr = c.train_lead_times[i]
                sr_dir_curr, sr_fname_curr = get_avg_sr_fname_dir(haz, train_type, lead_curr, r)
                sr_map_curr = pd.read_csv('%s/%s' %(sr_dir_curr, sr_fname_curr))
                srs_curr = np.array(sr_map_curr.SR)
                probs_curr = np.array(sr_map_curr.raw_prob)

                lead_prev = c.train_lead_times[i-1]
                sr_dir_prev, sr_fname_prev = get_avg_sr_fname_dir(haz, train_type, lead_prev, r)
                sr_map_prev = pd.read_csv('%s/%s' %(sr_dir_prev, sr_fname_prev))
                srs_prev = np.array(sr_map_prev.SR)

                for j in range(len(srs_curr)):
                    sr_curr = float(srs_curr[j])
                    sr_prev = float(srs_prev[j])
                    if sr_curr > sr_prev:
                        srs_curr[j] = sr_prev

                sr_map_curr = pd.DataFrame({'raw_prob': probs_curr, 'SR': srs_curr})
                sr_map_curr.to_csv('%s/%s' %(sr_dir_curr, sr_fname_curr))
    
    print('done with obs+warnings')
    
    ################## Post Process LSR Only based on lead time ##################
    train_type = 'obs'
    for r in c.train_radii:
        for haz in c.train_hazards:
            for i in range(1, len(c.train_lead_times)):

                lead_curr = c.train_lead_times[i]
                sr_dir_curr, sr_fname_curr = get_avg_sr_fname_dir(haz, train_type, lead_curr, r)
                sr_map_curr = pd.read_csv('%s/%s' %(sr_dir_curr, sr_fname_curr))
                srs_curr = np.array(sr_map_curr.SR)
                probs_curr = np.array(sr_map_curr.raw_prob)

                lead_prev = c.train_lead_times[i-1]
                sr_dir_prev, sr_fname_prev = get_avg_sr_fname_dir(haz, train_type, lead_prev, r)
                sr_map_prev = pd.read_csv('%s/%s' %(sr_dir_prev, sr_fname_prev))
                srs_prev = np.array(sr_map_prev.SR)

                for j in range(len(srs_curr)):
                    sr_curr = float(srs_curr[j])
                    sr_prev = float(srs_prev[j])
                    if sr_curr > sr_prev:
                        srs_curr[j] = sr_prev

                sr_map_curr = pd.DataFrame({'raw_prob': probs_curr, 'SR': srs_curr})
                sr_map_curr.to_csv('%s/%s' %(sr_dir_curr, sr_fname_curr))
    
    print('done with obs only')
    
    ################## Post Process LSR only to max out at LSR + Warning Threshold ##################
    
    for r in c.train_radii:
        for haz in c.train_hazards:
            for lead in c.train_lead_times:

                obs_sr_dir, obs_sr_fname = get_avg_sr_fname_dir(haz, 'obs', lead, r)
                obs_sr_map = pd.read_csv('%s/%s' %(obs_sr_dir, obs_sr_fname))
                obs_srs = np.array(obs_sr_map.SR)
                obs_probs = np.array(obs_sr_map.raw_prob)

                obsw_sr_dir, obsw_sr_fname = get_avg_sr_fname_dir(haz, 'obs_and_warnings', lead, r)
                obsw_sr_map = pd.read_csv('%s/%s' %(obsw_sr_dir, obsw_sr_fname))
                obsw_srs = np.array(obsw_sr_map.SR)

                for j in range(len(obs_srs)):
                    obs_sr = float(obs_srs[j])
                    obsw_sr = float(obsw_srs[j])
                    if obs_sr > obsw_sr:
                        obs_srs[j] = obsw_sr
                    

                obs_sr_map = pd.DataFrame({'raw_prob': obs_probs, 'SR': obs_srs})
                obs_sr_map.to_csv('%s/%s' %(obs_sr_dir, obs_sr_fname))
    
    print('done')
            


if (__name__ == '__main__'):

    main()