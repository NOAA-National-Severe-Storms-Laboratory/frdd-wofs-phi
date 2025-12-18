import shutil
import os
from wofs_phi import multiprocessing_driver as md
import datetime as dt

def rm_ps2(file):
    
    path = '/work/eric.loken/wofs/2024_update/SFE2024/fcst/full_npy_backup'
    #ts = os.path.getmtime('%s/%s' %(path, file))
    #ts = os.path.getmtime(file)
    #mdt = dt.datetime.fromtimestamp(ts)
    #test_dt = dt.datetime(2025, 2, 20, 8)
    #if mdt < test_dt:
    #print(file)
    os.remove('%s/%s' %(path, file))
    
    return

def rm_ps2_experiments():
    
    print('removing TI experiments')
    path = '/work/ryan.martz/wofs_phi_data/experiments'
    for direc in os.listdir(path):
        if 'wofs_psv2' in direc:
            shutil.rmtree('%s/%s' %(path, direc))
    
    print('removing models')
    path = '/work/ryan.martz/wofs_phi_data/obs_train/models'
    for direc in os.listdir(path):
        if 'wofs_psv2' in direc:
            shutil.rmtree('%s/%s' %(path, direc))
    
    print('removing test fcsts')
    path = '/work/ryan.martz/wofs_phi_data/obs_train/test_fcsts'
    for direc in os.listdir(path):
        if 'wofs_psv2' in direc:
            shutil.rmtree('%s/%s' %(path, direc))
    
    print('removing validation fcsts')
    path = '/work/ryan.martz/wofs_phi_data/obs_train/validation_fcsts'
    for direc in os.listdir(path):
        if 'wofs_psv2' in direc:
            shutil.rmtree('%s/%s' %(path, direc))
    
    return

def main():
    
    #base_path = '/work/ryan.martz/wofs_phi_data/training_data/obs_and_warnings/sampled_1d_obs_and_warnings'
    #lengths = [60, 120]
    #leads = [30, 60, 90, 120, 150, 180]
    #hazards = ['hail', 'wind', 'tornado']
    files = []
    #for length in lengths:
    #    for lead in leads:
    #        if lead+length > 240:
    #            continue
    #        for hazard in hazards:
    #            path = '%s/length_%s/wofs_lead_%s/%s' %(base_path, length, lead, hazard)
    #            for file in os.listdir(path):
    #                if  (('20230615' in file) or ('20230324' in file) or ('20240313' in file)\
    #                    or ('20230511' in file) or ('20240521' in file)):
    #                    continue
    #                files.append('%s/%s' %(path, file))
    path = '/work/eric.loken/wofs/2024_update/SFE2024/fcst/full_npy_backup'
    #path = '/work/eric.loken/wofs/2024_update/SFE2024/obs/dat_new_backup'
    for file in os.listdir(path):
        if ('wofs1d_2' in file) or ('rand_inds_2' in file) or ('no_torp' in file)\
        or ('old_eric' in file) or ('with_nan' in file):# or ('' in file):
            continue
        files.append(file)
    iterator = md.to_iterator(files)
    results = md.run_parallel(rm_ps2, iterator, nprocs_to_use = 30,\
                           description = 'Removing Sampled Fcsts and Rand Inds')
    
    #rm_ps2_experiments()
    
    return


if (__name__ == '__main__'):

    main()