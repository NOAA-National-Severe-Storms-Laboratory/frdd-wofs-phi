import numpy as np
import os
import sys
_wofsphi = '/home/ryan.martz/python_packages/frdd-wofs-phi'
sys.path.insert(0, _wofsphi)
from . import config as c
import datetime
from . import utilities
import sys
sys.path.append('../wofs_phi')
from wofs_phi import utilities
from wofs_phi import multiprocessing_driver as md
import multiprocessing as mp

def redo_min_convs(pred_file):
    
    var_method = 'min'
    km_spacing = c.dx_km
    radii_km = c.torp_conv_dists
    n_sizes = []
    for i in range(len(radii_km)):
        r = radii_km[i]
        n_sizes.append(int(((r/km_spacing)*2)+3))
    conv_footprints = utilities.get_footprints(n_sizes, radii_km, km_spacing)
    total_grid_points = 90000
    
    ti_c = os.path.getctime('%s/%s' %(c.train_fcst_full_npy_dir, pred_file))
    c_ti = time.ctime(ti_c)
    obj = time.strptime(c_ti)
    dt = datetime.datetime(obj.tm_year, obj.tm_mon, obj.tm_mday, obj.tm_hour, obj.tm_min)
    cutoff_dt = datetime.datetime(2025, 2, 26, 14)
    
    preds = np.load('%s/%s' %(c.train_fcst_full_npy_dir, pred_file))
    
    redo_inds = [369]#[299, 304, 339, 344, 354, 364, 369]
    if max(preds[:,369]) < 9999:
        return
    for i in redo_inds:
        try:
            redo_preds = preds[:,i]
        except:
            print(i, pred_file)
        if i == 369:
            redo_preds[redo_preds == -1] = 9999
        else:
            redo_preds[redo_preds == 0] = 9999
        
        array_2d = redo_preds.reshape((300,300))
        array_1d = array_2d.reshape((total_grid_points,))
        
        array_2d_15km = utilities.add_convolutions(var_method, array_2d, conv_footprints[0])
        array_1d_15km = array_2d_15km.reshape((total_grid_points,))
        array_2d_30km = utilities.add_convolutions(var_method, array_2d, conv_footprints[1])
        array_1d_30km = array_2d_30km.reshape((total_grid_points,))
        array_2d_45km = utilities.add_convolutions(var_method, array_2d, conv_footprints[2])
        array_1d_45km = array_2d_45km.reshape((total_grid_points,))
        array_2d_60km = utilities.add_convolutions(var_method, array_2d, conv_footprints[3])
        array_1d_60km = array_2d_60km.reshape((total_grid_points,))
        
        preds[:,i] = array_1d
        #preds[:,i+1] = array_1d_15km
        #preds[:,i+2] = array_1d_30km
        #preds[:,i+3] = array_1d_45km
        #preds[:,i+4] = array_1d_60km
    
    np.save('%s/%s' %(c.train_fcst_full_npy_dir, pred_file), preds)
    sample_new_torp_file(pred_file)
    
    return

def transpose_torp(old_pred_file, old_torp_dir):
    
    new_pred_file = old_pred_file.replace('psv3_with_torp', 'psv3_no_torp')
    print(new_pred_file)
    
    if not (os.path.exists('%s/%s' %(old_torp_dir, old_pred_file)) and os.path.exists('%s/%s' %(c.train_fcst_full_npy_dir, new_pred_file))):
        return
    
    if os.path.exists('%s/%s' %(c.train_fcst_full_npy_dir, old_pred_file)):
        return
    
    #last_updated = os.path.getmtime('%s/%s' %(c.train_fcst_full_npy_dir, new_pred_file))
    #year_updated = datetime.datetime.fromtimestamp(last_updated).year
    #if year_updated == 2025:
    #    return
    new_preds = np.load('%s/%s' %(c.train_fcst_full_npy_dir, new_pred_file))
    old_preds = np.load('%s/%s' %(old_torp_dir, old_pred_file))
    
    new_new_preds = np.zeros((90000,374))
    new_new_preds[:,0:269] = new_preds[:,0:269]
    
    for i in range(269,374):
        to_transpose = old_preds[:,i].reshape(300,300)
        transposed = to_transpose.transpose()
        new_new_preds[:,i] = transposed.reshape(90000,)
    
    np.save('%s/%s' %(c.train_fcst_full_npy_dir, old_pred_file), new_new_preds)
    
    return

def remove_ps(pred_file):
    
    preds_without_ps_fname = pred_file.replace('psv3', 'psv0')
    if preds_without_ps_fname == pred_file:
        return
    
    if not os.path.exists('%s/%s' %(c.train_fcst_full_npy_dir, pred_file)):
        return
    ps_indices = np.concatenate((np.arange(54,72), np.arange(102,120), np.arange(150,168),\
                                 np.arange(198,216), np.arange(246,264)))
    #ps_tor_indices = np.concatenate((np.arange(54,60), np.arange(102,108), np.arange(150,156),\
    #                             np.arange(198,204), np.arange(246,252)))
    preds_with_ps = np.load('%s/%s' %(c.train_fcst_full_npy_dir, pred_file))
    preds_without_ps = np.delete(preds_with_ps, ps_indices, axis = 1)
    #preds_without_tor_ps = np.delete(preds_with_ps, ps_tor_indices, axis = 1)
    
    #preds_without_tor_ps_fname = pred_file.replace('psv2', 'psv2NoTor')
    
    np.save('%s/%s' %(c.train_fcst_full_npy_dir, preds_without_ps_fname), preds_without_ps)
    sample_new_torp_file(preds_without_ps_fname)
    #np.save('%s/%s' %(c.train_fcst_full_npy_dir, preds_without_tor_ps_fname), preds_without_tor_ps)
    
    return

def merge_torp(with_torp_file):
    
    no_torp_file = with_torp_file.replace('wofs1d_psv3_with_torp', 'wofs1d')
    new_preds_fname = no_torp_file.replace('wofs1d', 'wofs1d_psv2_with_torp')
    
    if (new_preds_fname == no_torp_file) or (new_preds_fname == with_torp_file):
        return
    
    if (not os.path.exists('%s/%s' %(c.train_fcst_full_npy_dir, no_torp_file)))\
    or (not os.path.exists('%s/%s' %(c.train_fcst_full_npy_dir, with_torp_file))):
        return
    
    ps_tor_indices = np.concatenate((np.arange(54,60), np.arange(102,108), np.arange(150,156),\
                                     np.arange(198,204), np.arange(246,252)))
    
    preds = np.load('%s/%s' %(c.train_fcst_full_npy_dir, no_torp_file))
    with_torp_preds = np.load('%s/%s' %(c.train_fcst_full_npy_dir, with_torp_file))
    torp_preds = with_torp_preds[:,269:]
    
    new_preds = np.zeros((90000,374))
    new_preds[:,0:269] = preds
    new_preds[:,269:] = torp_preds
    
    #new_preds_no_psTor = np.delete(new_preds, ps_tor_indices, axis = 1)
    
    #new_preds_no_psTor_fname = no_torp_file.replace('wofs1d', 'wofs1d_psv2NoTor_with_torp')
    
    #np.save('%s/%s' %(c.train_fcst_full_npy_dir, new_preds_no_psTor_fname), new_preds_no_psTor)
    np.save('%s/%s' %(c.train_fcst_full_npy_dir, new_preds_fname), new_preds)
    sample_new_torp_file(new_preds_fname)
    
    return

def remove_torp(pred_file):
    
    if (not (os.path.exists('%s/%s' %(c.train_fcst_full_npy_dir, pred_file))))\
    or (not ('psv3_with_torp' in pred_file)) or (('p_only' in pred_file) or ('filtered' in pred_file)):
        return
    if (('20230615' in pred_file) or ('20230324' in pred_file) or ('20240313' in pred_file)\
                or ('20230511' in pred_file) or ('20240521' in pred_file)):
        return
    
    if ('psv3_with_torp' in pred_file) and (not ('filtered' in pred_file)):
        preds_without_torp_fname = pred_file.replace('psv3_with_torp', 'psv3_no_torp')
    else:
        return
    
    if preds_without_torp_fname == pred_file:
        return
    
    torp_indices = np.arange(269,374)
    preds_with_torp = np.load('%s/%s' %(c.train_fcst_full_npy_dir, pred_file))
    preds_without_torp = np.delete(preds_with_torp, torp_indices, axis = 1)
    
    np.save('%s/%s' %(c.train_fcst_full_npy_dir, preds_without_torp_fname), preds_without_torp)
    sample_new_torp_file(preds_without_torp_fname)
    
    return

def filter_torp(pred_file, filter_level = 0.4):
    
    pred_file_arr = pred_file.split('_')
    model_type = pred_file_arr[1] + '_with_torp_filtered'
    date_time_info = '%s_%s_%s_%s' %(pred_file_arr[-4], pred_file_arr[-3], pred_file_arr[-2], pred_file_arr[-1])
    filtered_pred_file = 'wofs1d_%s_%s' %(model_type, date_time_info)
    
    if filtered_pred_file == pred_file:
        return
    
    preds = np.load('%s/%s' %(c.train_fcst_full_npy_dir, pred_file))
    
    min_conv_inds = np.array([299, 304, 339, 344, 354, 364, 369])
    inds = np.arange(269,374,5)
    
    for conv in [269, 270, 271, 272, 273]:
        raw_probs = preds[:,conv]
        del_indices = np.where(raw_probs < filter_level)
        for i in inds:
            if ((i >= 274) and (i <= 278)):
                preds[del_indices,i] = -1
            elif i in min_conv_inds:
                preds[del_indices,i] = 9999
            else:
                preds[del_indices,i] = 0
        
        inds += 1
        min_conv_inds += 1
            
    
    utilities.save_data(c.train_fcst_full_npy_dir, filtered_pred_file, preds, 'npy')
    sample_new_torp_file(filtered_pred_file)
    
    return filtered_pred_file


def remove_radar_from_torp(pred_file):
    
    #if not ('filtered' in pred_file):
    #    pred_file = filter_torp(pred_file)
    
    pred_file_arr = pred_file.split('_')
    model_type = pred_file_arr[1] + '_with_torp_p_only'
    date_time_info = '%s_%s_%s_%s' %(pred_file_arr[-4], pred_file_arr[-3], pred_file_arr[-2], pred_file_arr[-1])
    p_only_pred_file = 'wofs1d_%s_%s' %(model_type, date_time_info)
    
    if p_only_pred_file == pred_file:
        return
    
    preds = np.load('%s/%s' %(c.train_fcst_full_npy_dir, pred_file))
    new_preds = np.zeros((preds.shape[0], 294))
    new_preds[:,0:289] = preds[:,0:289]
    new_preds[:,289:] = preds[:,369:]
    
    utilities.save_data(c.train_fcst_full_npy_dir, p_only_pred_file, new_preds, 'npy')
    sample_new_torp_file(p_only_pred_file)
    
    return

def sample_new_torp_file(full_npy):
    
    preds = np.load('%s/%s' %(c.train_fcst_full_npy_dir, full_npy))
    
    dat_filename = full_npy.split('.')[0] + '.dat'
    rand_inds_filename = full_npy.replace('wofs1d', 'rand_inds')
    
    rand_inds = np.random.choice(np.arange(preds.shape[0]), size=int(preds.shape[0]*0.1), replace=False)
    
    dat_data = preds[rand_inds,:]
    
    full_predictions = np.float32(preds)
    sampled_predictions = np.float32(dat_data)
    
    #utilities.save_data(c.train_fcst_full_npy_dir, full_npy, full_predictions, 'npy')
    utilities.save_data(c.train_fcst_dat_dir, dat_filename, sampled_predictions, 'dat')
    utilities.save_data(c.train_fcst_dat_dir, rand_inds_filename, rand_inds, 'npy')
    
    return

def read_binary(infile, header = False):
    '''read in the .dat forecast files'''
    
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


    #print arr.shape
    #Rebroadcast the data into 2-d array with proper format
    try:
        arr.shape = (-1, c.num_training_vars)
    except:
        return arr
    return arr

def test_size(file):
    
    total_file = '%s/%s' %(c.train_fcst_dat_dir, file)
    arr = read_binary(total_file)
    if (not (arr.shape == (9000,374))):
        #os.remove('%s/%s' %(c.train_fcst_dat_dir, file))
        rand_inds_filename = file.replace('wofs1d', 'rand_inds')
        rand_inds_filename = rand_inds_filename.replace('.dat', '.npy')
        #os.remove('%s/%s' %(c.train_fcst_dat_dir, rand_inds_filename))
        print(file, arr.shape)
    
    return

def main():
    
    #old_torp_dir = '/work/eric.loken/wofs/2024_update/SFE2024/fcst/full_npy'
    #for pred_file in os.listdir(c.train_fcst_full_npy_dir):
    #    if 'with_torp' in pred_file:
    #        transpose_torp(pred_file, new_torp_dir)
    #        print(pred_file)
    
    pred_files = os.listdir(c.train_fcst_full_npy_dir)
    #pred_files = os.listdir(old_torp_dir)
    use_pred_files = []
    for file in pred_files:
        if 'psv3_with_torp' in file and\
        (not (('20230615' in file) or ('20230324' in file) or ('20240313' in file) or ('20230511' in file)\
              or ('20240521' in file))) and (not ('filtered' in file)) and (not ('p_only' in file)):
            use_pred_files.append(file)
    
    iterator = md.to_iterator(use_pred_files)
    results = md.run_parallel(merge_torp, iterator, nprocs_to_use = 10,\
                                           description = 'Making PS2 with TORP')
    ##remove_ps
    #merge_torp
    #filter_torp
    #remove_radar_from_torp
    
    return

if (__name__ == '__main__'):

    main()