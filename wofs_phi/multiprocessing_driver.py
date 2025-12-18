#import sys
#_wofsphi = '/home/ryan.martz/python_packages/frdd-wofs-phi'
#sys.path.insert(0, _wofsphi)

import multiprocessing as mp
import itertools
from multiprocessing.pool import Pool
from tqdm import tqdm  
import traceback
from collections import ChainMap
import warnings
import copy

import os
import numpy as np

from . import config as c

class LogExceptions(object):
    def __init__(self, func):
        self.func = func

    def error(self, msg, *args):
        """ Shortcut to multiprocessing's logger """
        return mp.get_logger().error(msg, *args)
    
    def __call__(self, *args, **kwargs):
        try:
            result = self.func(*args, **kwargs)
                    
        except Exception as e:
            # Here we add some debugging help. If multiprocessing's
            # debugging is on, it will arrange to log the traceback
            print(traceback.format_exc())
            self.error(traceback.format_exc())
            # Re-raise the original exception so the Pool worker can
            # clean up
            raise

        # It was fine, give a normal answer
        return result

def to_iterator(*lists):
    """
    turn list
    """
    return itertools.product(*lists)

def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)

def run_parallel(
    func,
    args_iterator,
    nprocs_to_use,
    description=None,
    kwargs={}, 
):
    """
    Runs a series of python scripts in parallel. Scripts uses the tqdm to create a
    progress bar.
    Args:
    -------------------------
        func : callable
            python function, the function to be parallelized; can be a function which issues a series of python scripts
        args_iterator :  iterable, list,
            python iterator, the arguments of func to be iterated over
                             it can be the iterator itself or a series of list
        nprocs_to_use : int or float,
            if int, taken as the literal number of processors to use
            if float (between 0 and 1), taken as the percentage of available processors to use
        kwargs : dict
            keyword arguments to be passed to the func
    """
    iter_copy = copy.copy(args_iterator)
    
    total = len(list(iter_copy))
    pbar = tqdm(total=total, desc=description)
    results = [] 
    def update(*a):
        # This is called whenever a process returns a result.
        # results is modified only by the main process, not by the pool workers. 
        pbar.update()
    
    if 0 <= nprocs_to_use < 1:
        nprocs_to_use = int(nprocs_to_use * mp.cpu_count())
    else:
        nprocs_to_use = int(nprocs_to_use)

    if nprocs_to_use > mp.cpu_count():
        raise ValueError(
            f"User requested {nprocs_to_use} processors, but system only has {mp.cpu_count()}!"
        )
        
    pool = Pool(processes=nprocs_to_use)
    ps = []
    for args in args_iterator:
        if isinstance(args, str):
            args = (args,)
         
        p = pool.apply_async(LogExceptions(func), args=args, callback=update)
        ps.append(p)
        
    pool.close()
    pool.join()

    results = [p.get() for p in ps]
    
    return results

def do_torp_transpose():
    
    raw_pred_files = os.listdir(c.train_fcst_full_npy_dir)
    pred_files = []
    for file in raw_pred_files:
        if 'with_torp' in file:
            pred_files.append(file)
    new_torp_dir = ['/work/eric.loken/wofs/2024_update/SFE2024/fcst/full_npy']
    
    iterator = to_iterator(pred_files, new_torp_dir)
    results = run_parallel(transpose_torps.transpose_torp, iterator, nprocs_to_use = int(0.6*mp.cpu_count()), description = 'Transposing TORP Data')
    
    return

def do_torp_prob_filter():
    
    filter_level = 0.4
    
    raw_pred_files = os.listdir(c.train_fcst_full_npy_dir)
    pred_files = []
    for file in raw_pred_files:
        if (('20230615' in file) or ('20230324' in file) or ('20240313' in file)\
                or ('20230511' in file) or ('20240521' in file)):
                continue
        if 'psv3_with_torp' in file:
            pred_files.append(file)
    
    iterator = to_iterator(pred_files, filter_level)
    results = run_parallel(transpose_torps.filter_torp, iterator, nprocs_to_use = int(0.6*mp.cpu_count()),\
                           description = 'Filtering by Probability')
    
    
    return


def do_torp_radar_filter():
    
    raw_pred_files = os.listdir(c.train_fcst_full_npy_dir)
    pred_files = []
    for file in raw_pred_files:
        if (('20230615' in file) or ('20230324' in file) or ('20240313' in file)\
                or ('20230511' in file) or ('20240521' in file)):
                continue
        if 'psv3_with_torp' in file:
            pred_files.append(file)
    
    iterator = to_iterator(pred_files)
    results = run_parallel(transpose_torps.remove_radar_from_torp, iterator, nprocs_to_use = int(0.6*mp.cpu_count()),\
                           description = 'Removing Radar Data')
    
    return

def train_wofs_phi():
    print(c.model_type)
    dates = [list(np.genfromtxt('wofs_phi/probSevere_dates.txt').astype(int).astype(str))]
    if c.train_mode == "train" or c.train_mode == "validate" or c.train_mode == "make_maps":
        iterator = to_iterator(dates, c.forecast_lengths, c.train_lead_times, c.train_hazards,\
                               c.train_radii, c.train_types)
        results = run_parallel(train, iterator, nprocs_to_use = int(0.6*mp.cpu_count()),\
                               description = 'Training/Validating WoFS-PHI')
        for length in c.forecast_lengths:
            for train_type in c.train_types:
                MLTrainer.postprocess_sr_map_by_lead_time(train_type, length)
        if 'obs' in c.train_types:
            for length in c.forecast_lengths:
                if ('obs' in c.train_types) and ('obs_and_warnings' in c.train_types):
                    MLTrainer.postprocess_sr_map_by_obs_limiting(length)
    print('starting test predictions')
    iterator = to_iterator(dates, c.forecast_lengths, c.train_lead_times, c.train_hazards,\
                           c.train_radii, c.train_types)
    results = run_parallel(test, iterator, nprocs_to_use = int(0.6*mp.cpu_count()),\
                           description = 'Making Test Predictions for WoFS-PHI')
    return

def test_sizes():
    
    raw_pred_files = os.listdir(c.train_fcst_dat_dir)
    pred_files = []
    for file in raw_pred_files:
        if (('20230615' in file) or ('20230324' in file) or ('20240313' in file)\
            or ('20230511' in file) or ('20240521' in file)):
            continue
        if ('with_torp' in file) and (not 'rand_inds' in file):
            pred_files.append(file)
    print(len(pred_files))
    iterator = to_iterator(pred_files)
    results = run_parallel(transpose_torps.test_size, iterator, nprocs_to_use = int(0.6*mp.cpu_count()),\
                           description = 'Checking TORP File Sizes')
    
    return

def main():
    
    train_wofs_phi()
    #do_torp_transpose()
    #regen_torp()
    #test_sizes()
    #do_torp_radar_filter()
    
    return

if (__name__ == '__main__'):

    main()