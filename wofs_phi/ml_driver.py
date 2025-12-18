#===================================================
# Script that will drive the ML training and testing
#===================================================

#from . import wofs_phi
#from . import config as c
#from . import predictor_extractor as pex
import wofs_phi
import config as c
import predictor_extractor as pex

import sys
sys.path.append('../wofs_phi')
from wofs_phi import utilities
#from wofs_phi import multiprocessing_driver as md
import multiprocessing as mp

import os.path
from itertools import compress
import warnings
import datetime
import time

import math
import numpy as np
from datetime import timedelta
import copy



#Could also call this, e.g., FileObtainer or something. 
class MLDriver:
    ''' Handles the driving of the ML for training or testing.
        i.e., Sets the relevant filenames, etc.
    '''

    def __init__(self, pre00z_date, time_window, wofs_init, wofs_lead_time,\
                    wofs_path, ps_path, wofs_files, ps_files, torp_files, ps_init):
        ''' @pre00z_date is the date for the case before 00z (string format 
                YYYYMMDD) 
            @time_window is the forecast time window in minutes
            @wofs_init is the wofs initialization time 
                (string: HHMM; e.g., "2200")
            @wofs_lead time is the time difference (in minutes) between the 
                wofs intialization time and the start of the forecast
                valid period 
            @wofs_path is the path to the wofs files 
            @ps_path is the path to the ProbSevere files
            @wofs_files is the list of wofs files (i.e., spanning the forecast
                valid period 
            @ps_files is the list of probSevere files (starting with the most 
                recent and going back 3 hours in time)
            @torp_files is the list of torp files (all within last 3 hours - unorganized)
            @ps_init is the ProbSevere intiailization time 
        '''

        self.pre00z_date = pre00z_date
        self.time_window = time_window
        self.wofs_init = wofs_init
        self.wofs_lead_time = wofs_lead_time
        self.wofs_path = wofs_path
        self.ps_path = ps_path
        self.wofs_files = wofs_files
        self.ps_files = ps_files
        self.torp_files = torp_files
        self.ps_init = ps_init

        return 

    @classmethod 
    def start_driver(cls, before00zDate, timeWindow, wofsInitTime, wofsLeadTime, psPath,\
                current_mode, training = False):
        '''Creates an MLDriver object given a date (from before 00z; @before00zDate), 
            time window (in minutes; @timeWindow), 
            wofs initialization time (string (YYYYMMDD); @wofsInitTime), 
            wofs lead time (in minutes; @wofsLeadTime), and probSevere path (@psPath)
            @Returns MLDriver object
            @current_mode is the string of the mode we're in: 
                "forecast" for forecast mode, 
                "warning" for warning mode  

        '''
        
        #Find wofs path 
        wofsPath = MLDriver.find_wofs_path(before00zDate, wofsInitTime) 

        #Find wofs files 
        wofs_file_list = MLDriver.find_wofs_file_list(timeWindow, wofsInitTime, wofsLeadTime,\
                                before00zDate)

        #Find ps files
        ps_file_list = MLDriver.find_ps_files_from_first_wofs(wofs_file_list[0], current_mode)
        
        #Find torp files
        if c.include_torp_in_predictors:
            torp_file_list = wofs_phi.TORP.get_torp_files(wofs_file_list[0], training = training)
        else:
            torp_file_list = []

        #Find the ps init time 
        ps_iTime, __ = wofs_phi.ForecastSpecs.find_ps_date_time(ps_file_list[0], c.ps_version)
        
        if c.ps_version == 0:
            ps_file_list = []

        #Create MLDriver object
        obj = MLDriver(before00zDate, timeWindow, wofsInitTime, wofsLeadTime, wofsPath,\
                psPath, wofs_file_list, ps_file_list, torp_file_list, ps_iTime)

        return obj



    @staticmethod
    def get_info_for_warning_mode(dt_ps_start, pre00z_date):

        '''This method will create an MLDriver object based on a 
            datetime object corresponding with real-time, i.e., 
            the ProbSevere start time.

            This method gathers the information needed to drive 
            warning mode. Namely, 
            @Returns wofs initialization time string and wofs lead time
        '''

        #Get valid period (i.e., start time + ps_spinup (5 min))
        start_valid_dt = dt_ps_start + timedelta(minutes=c.ps_spinup_time)

        #Get wofs initialization time 
        wofs_init_dt, wofs_init_str = MLDriver.find_wofs_init_time(\
                dt_ps_start, pre00z_date)
        if wofs_init_dt == '0' and wofs_init_str == '0':
            return '0', 0

        #Get wofs lead time -- i.e., difference between start of the valid period 
        #and the initialization time 
    
        leadTime = wofs_phi.ForecastSpecs.subtract_dt(start_valid_dt, wofs_init_dt, True)


        return wofs_init_str, leadTime


    @staticmethod 
    def find_wofs_init_time(current_dt, before00z_date):
        ''' Finds the wofs initialization time based on...
            @current_dt is the datetime object corresponding
                to the current date/time.
            @before00z_date is the 8-character string (YYYYMMDD) corresponding
                to the date before 00z. 
            @Returns a datetime object and a 4-character string corresponding
                to the wofs initialization time that should be used. 
        '''

        #Need to Get a list of datetime objects corresponding to when we 
        #switch: i.e., wofs initialization times + wofs displacement. 

        #Create list of wofs_init_datetime objects 
        wofs_init_dts = MLDriver.get_wofs_init_dts(before00z_date, \
                            c.all_wofs_init_times, c.wofs_time_between_runs)


        #dts indicating when the given init time can be used 
        increment_dts = [w + timedelta(minutes=c.wofs_spinup_time) \
                            for w in wofs_init_dts]
        
        #Compare current_dt to each element in increment_dts 
        is_current_time_greater = [(current_dt >= i) for i in increment_dts]

        #If there aren't any times that are greater, then we have to stop
        #the code/can't proceed
        if (sum(is_current_time_greater) <= 0):
            print ("We don't have enough data to use warning mode yet") 
            #quit()
            return '0', '0'
    

        #If there's at least one true: We want the datetime object correponding to
        #the index of the last True in is_current_time_greater

        #apply the boolean list to the wofs_init_dts to get the "possible" list--
        #of which we're interested in the last element
        possible_dt_to_use = list(compress(wofs_init_dts, is_current_time_greater))
        good_wofs_init_dt = possible_dt_to_use[-1]


        #Convert this to string as well 
        __, i_time_str = wofs_phi.ForecastSpecs.dattime_to_str(good_wofs_init_dt) 

        return good_wofs_init_dt, i_time_str


    @staticmethod
    def get_wofs_init_dts(date_pre_00z, init_time_list, wofs_update_freq):
        ''' Computes a list of datetime objects based on...
            @date_pre_00z is the before00z date (str; YYYYMMDD)
            @init_time_list is a list of 4-character strings (HHMM) corresponding to 
                all wofs initialization times (in chronological order, starting with 
                e.g., "1700")
            @wofs_update_freq is the time between wofs initializations in 
                minutes (currently, 30) 
        '''

        dts_wofs_init = [] #Will hold the datetime objects


        for d in range(len(init_time_list)):
            wfs_init_time = init_time_list[d]
            if (d == 0): #i.e., for first init time, get dt object
                dt_obj = wofs_phi.ForecastSpecs.str_to_dattime(wfs_init_time, date_pre_00z)   
                #If it's after 00z, then have to increment by a day---even though
                #this should basically never be the case
                if (wfs_init_time in c.next_day_inits):
                    dt_obj += timedelta(days=1) 

            else: 
                dt_obj += timedelta(minutes=wofs_update_freq) 
                

       
            #Append to list 
            dts_wofs_init.append(dt_obj) 
 

        return dts_wofs_init


    @staticmethod
    def find_ps_files_from_first_wofs(first_wofs_file, mode_str):
        ''' Finds the set of ProbSevere files given the first wofs
            file.

            NOTE: This will depend if we are in forecast or warning
            mode.
            In forecast mode (at least for training), the first
                probSevere file will be from wofs initialization
                time plus wofs spinup time 
            In warning mode, the first probSevere file will be the
                most recent PS file relative to the start of the 
                valid period 

            @first_wofs_file is the first/earliest wofs file of the 
                valid period 

            @mode_str is the string corresponding to which mode we're in;
                "forecast" for forecast mode ; "warning" for warning mode
        '''
        
        #First, need to get the datetime object associated with the
        #first wofs file


        if (mode_str == "warning"):
            wofs_time, wofs_date = wofs_phi.ForecastSpecs.find_date_time_from_wofs(\
                first_wofs_file, "forecast") 
        elif (mode_str == "forecast"):
            wofs_time, wofs_date = wofs_phi.ForecastSpecs.find_date_time_from_wofs(\
                first_wofs_file, "init") #But then we still need to add the spinup in this case

        wofs_dt = wofs_phi.ForecastSpecs.str_to_dattime(wofs_time, wofs_date) 

        #We need to add the spinup time if we're in forecast mode
        if (mode_str == "forecast"):
            wofs_dt += timedelta(minutes=c.wofs_spinup_time) 

        #For warning mode, we need to remove time to account for ps spinup
        elif (mode_str == "warning"):
            wofs_dt -= timedelta(minutes=c.ps_spinup_time) 

        #Now, find initial probSevere datetime
        first_ps_dt = MLDriver.find_first_ps_datetime_from_wofs_datetime(wofs_dt)

        #Get the list of probSevere datetime files
        ps_dt_list = MLDriver.get_ps_datetimes(first_ps_dt) 
        
        ps_filenames = MLDriver.get_ps_names_from_dt_list(ps_dt_list)
        
        return ps_filenames


    @staticmethod
    def get_ps_names_from_dt_list(dt_list):
        '''Returns a list of probSevere filenames from datetime list'''
        
        ps_names = []
        
        for l in range(len(dt_list)):
            dt = dt_list[l]
            
            #get date and time from datetime object
            date_str, time_str = wofs_phi.ForecastSpecs.dattime_to_str(dt)
            year = date_str[0:4]
            
            if (c.ps_version == 2) or (c.ps_version == 0):
                
                #TODO: Might need to add capability to check for each second
                #for real-time. Might also not be a problem in real time.
                
                ps_name = "%s/%s/PROBSEVERE_good_motions/SSEC_AWIPS_PROBSEVERE_%s_%s00.json"\
                %(year, date_str, date_str, time_str)# - training/realtime
                
            elif (c.ps_version == 3):
                ps_name = "%s/%s/PROBSEVERE-V3/SSEC_AWIPS_PROBSEVERE-V3_%s_%s00.json" %(year, date_str, date_str, time_str)
                
                #if dt.hour < 12:
                #    dir_str, time_str = wofs_phi.ForecastSpecs.dattime_to_str(dt - datetime.timedelta(days = 1))
                #else:
                #    dir_str = date_str
                #ps_name = "%s/SSEC_AWIPS_PROBSEVERE-V3_%s_%s00.json" %(dir_str, date_str, time_str)

            ps_names.append(ps_name)


        return ps_names

    @staticmethod
    def get_ps_datetimes(first_dt):
        '''Returns a list of datetime objects corresponding to the
            set of ProbSevere files we'd like to grab, given the 
            first datetime object (i.e., datetime object of the 
            first ProbSevere file)
        '''

        ps_datetimes = [] 
        
        times = np.arange(0,c.ps_time_to_go_back+c.ps_update_rate, c.ps_update_rate)
        for t in range(len(times)):
            time = times[t]
            if (t == 0):
                #Add first dt object
                ps_datetimes.append(first_dt)
            elif (t == 1):
                dt = first_dt - timedelta(minutes=c.ps_update_rate)
                ps_datetimes.append(dt) 
            else:
                dt -= timedelta(minutes=c.ps_update_rate) 
                ps_datetimes.append(dt) 

        return ps_datetimes

    
    @staticmethod
    def find_first_ps_datetime_from_wofs_datetime(dt_wofs):
        '''Finds the first ProbSevere datetime object based on 
            the first wofs datetime object -- based on the 
            principle that ProbSevere files are generated every
            2 minutes (on the even minutes)
            @dt_wofs is the wofs datetime object corresponding to
                either the start of the valid period (in warning
                mode) or the wofs initialization time + spinup time
                (in forecast mode) 
            @Returns datetime object corresponding to the first
                ProbSevere time
        '''


        #See if we're at an even minute. If so, dt_wofs is the 
        #first probSevere dt. Else, we need to subtract 1 minute

        #Get time string from datetime object
        __, time_str = wofs_phi.ForecastSpecs.dattime_to_str(dt_wofs)

        #Convert to integer
        time_int = int(time_str) 
        
        #Is this integer odd? Subtract 1 minute
        if (time_int%2 == 1):
            ps_dt = dt_wofs - timedelta(minutes=1) 
        else:
            ps_dt = dt_wofs


        return ps_dt


    @staticmethod
    def find_wofs_path(before_00z_date, wofs_initialization_time):

        ''' Returns the path to the wofs files 
            @before_00z_date is the date of the case before 00z
            @wofs_initialization_time is the 4-character string corresponding
                to the wofs initialization time
        '''

        wofs_direc = "%s/%s/%s" %(c.wofs_base_path, before_00z_date, wofs_initialization_time)

        return wofs_direc

    @staticmethod
    def find_wofs_file_list(forecastWindow, wofsInit, wofs_lead_time_min, dateBefore00z):
        '''Finds the list of wofs filenames.
            @forecastWindow is the length of the forecast window in minutes
            @wofsInit is the wofs initialization time ("HHMM") 
            @wofs_lead_time_min is the wofs lead time in minutes 
                i.e., difference between wofs initialization and start of valid period 
            @dateBefore00z is the 8-character string before 00z (YYYYMMDD) 

        '''
    
        filenames = [] #will hold the list of wofs filenames to return

        #Find wofs indices
        indices = MLDriver.find_wofs_indices(wofs_lead_time_min, forecastWindow)

        #Get new WofsFile object
        for i in range(len(indices)):
            #Each index will be a new WofsFile object
            wofs_file_obj = WofsFile(indices[i], wofsInit, dateBefore00z)
            filename = wofs_file_obj.get_name()
            filenames.append(filename) 
        

        return filenames


    @staticmethod
    def find_wofs_indices(wofs_lead_time, wofs_time_window):
        ''' Finds the wofs indices associated with the wofs files
            @wofs_lead_time is the time in minutes between wofs_initialization
                time and the start of the valid period
            @wofs_time_window is the length of the forecast valid period in minutes

        '''
       

        #Compute number of time intervals within the forecast valid period 
        #n_time_intervals = math.floor(wofs_time_window/c.wofs_update_rate) 
        n_time_intervals = MLDriver.compute_num_wofs_intervals(wofs_time_window,\
                                    c.wofs_update_rate) 


        #Find initial index 
        initial_index = MLDriver.compute_num_wofs_intervals(wofs_lead_time, c.wofs_update_rate)

        #Find final index 
        final_index = initial_index + n_time_intervals

        #Find array 
        ind_array = np.linspace(initial_index, final_index, n_time_intervals +1) 

        #Convert to string 
        str_ind_array = MLDriver.ind_to_str_array(ind_array)

        return str_ind_array 

    @staticmethod 
    def ind_to_str_array(in_arr):

        '''Converts an array of integers into an array of strings with 
            a padded zero if needed'''

        new_arr = ["%s" %str(int(a)).zfill(2) for a in in_arr]
        
        return new_arr


    @staticmethod
    def compute_num_wofs_intervals(window, update_rate):
        '''Computes the index or number of time intervals of a given time
            window (@window; in minutes), based on a given wofs update
            frequency in minutes (@update_rate; e.g., 5 means a new wofs 
            file is generated every 5 minutes)
        '''


        return math.floor(window/update_rate)


class WofsFile:

    ''' Handles the characteristics of a wofs file'''

    def __init__(self, index, init_time, date_before_00z):
        '''@index is the 2-character string associated with the 
                index of the wofs file, e.g., "05"
            @init_time is the 4-character string of wofs initialization
                time
        '''

        self.index = index
        self.init_time = init_time
        self.date_before_00z = date_before_00z

        return 


    def get_date(self):
        '''Returns the date (actual--that updates after 00z) associated
            with the given file'''

        #get datetime object object for the initialization time 
        wofs_init_dt = WofsFile.get_wofs_dt(self.init_time, self.date_before_00z)

        wofs_date, wofs_i_time_str = wofs_phi.ForecastSpecs.dattime_to_str(wofs_init_dt) 
        

        return wofs_date

    
    def get_valid_time(self):
        
        #Get datetime object for the initialization time 
        wofs_init_dt = WofsFile.get_wofs_dt(self.init_time, self.date_before_00z)
        
        #Convert index into a displacement time -- i.e., 
        #time to add to the wofs initialization time 
        displacement_time = float(self.index)*c.wofs_update_rate

        wofs_valid_dt = wofs_init_dt + timedelta(minutes=displacement_time) 

        #Convert to string date and time 
        valid_date, valid_time_str = wofs_phi.ForecastSpecs.dattime_to_str(wofs_valid_dt) 

        
        return valid_time_str 


    def get_name(self): 
        #wofs_ALL_05_${date}_0200_0225.nc" 

        file_date = self.get_date()
        valid_time = self.get_valid_time()
        filename = "wofs_ALL_%s_%s_%s_%s.nc" \
                %(self.index, file_date, self.init_time, valid_time)

        return filename


    @staticmethod
    def get_wofs_dt(time_string, before_00z_date_string):
        #Returns datetime object for wofs file given a 4-character time 
        #e.g., 2300 (@time_string) and the date before 00z (@before_00z_date_string; 
        #in format YYYYMMDD) 

        #First, obtain the datetime object 
        dt_obj = wofs_phi.ForecastSpecs.str_to_dattime(time_string, before_00z_date_string)

        #If it's after 00z, add 1 day to the above datetime object

        if (time_string in c.next_day_inits):
            dt_obj += timedelta(days=1) 


        return dt_obj


#For forecast mode 
def create_forecast_mode_training(train_types):
    ''' Creates training files'''

    modes = ["forecast"]
    trainings = [True]

    windows = [60]#,120] #Focus on 60 minute windows 
    
    date_file = (c.base_path/"probSevere_dates.txt")
    #date_file = "first_obs_dates.txt" 

    dates = read_txt(date_file) 

    torpFiles = [] 

    report_radii = [15] #Only really used to check if reports file 
                        #already exists
    
    training_init_times = ["1700", "1730", "1800", "1830", "1900",\
        "1930", "2000", "2030", "2100", "2130", "2200", "2230", \
        "2300", "2330", "0000", "0030", "0100", "0130", "0200"]

    lead_times = [30, 60, 90, 120, 150, 180] #These are wofs lead times, btw
    
    #lead_times = [30]
    #dates = ['20190430']
    #training_init_times = ['0000']

    #Get the data
    with Pool() as p:
        p.starmap(gen_preds, ((date, init_time, window, lead_time, mode, training, report_radius)\
                              for date in dates for init_time in training_init_times\
                              for window in windows for lead_time in lead_times for mode in modes\
                              for training in trainings for report_radius in report_radii))
    
    #for window in windows:
    #    for lead_time in lead_times:
    #        for date in dates:
    #            for init_time in training_init_times:  

    return

def regen_pred_file(file):
    
    mode = "forecast"
    training = True
    report_radius = 15
    
    file_split = file.split('_')
    date = file_split[4]
    init_time = file_split[5]
    valid_split = file_split[7].split('.')[0]
    start = valid_split[1:].split('-')[0]
    end = valid_split[1:].split('-')[1]
    
    init_dt = datetime.datetime(1970, 1, 1, int(init_time[0:2]), int(init_time[2:]))
    start_dt = datetime.datetime(1970, 1, 1, int(start[0:2]), int(start[2:]))
    end_dt = datetime.datetime(1970, 1, 1, int(end[0:2]), int(end[2:]))
    
    window = int((end_dt - start_dt).seconds/60)
    lead_time = int((start_dt - init_dt).seconds/60)
    
    gen_preds(date, init_time, window, lead_time, mode, training, report_radius)
    
    return

def gen_preds(date, init_time, window, lead_time, mode, training, report_radius):
    
    #print (date, init_time, 'wofs lead: ' + str(lead_time) + ' min', 'duration: ' + str(window) + ' min')
    if ((window + lead_time > 180) and (init_time[2:] == '30')) or (window + lead_time > 240):
        return
    mld = MLDriver.start_driver(date, window, init_time, lead_time, c.ps_dir,\
                                mode, training = training)
    #Use this to drive the forecast
    ml = wofs_phi.MLGenerator(mld.wofs_files, mld.ps_files, mld.ps_path,\
                     mld.wofs_path, mld.torp_files, c.nc_outdir, mode, c.train_types)

    #Check to make sure wofs files exist; if so we can generate.
    proceed_wofs = does_wofs_exist(mld.wofs_path, mld.wofs_files[0], init_time, lead_time)
    
    #proceed_torp = does_torp_exist(mld.torp_files, date, init_time, lead_time)
    
    #if not (c.ps_version == 0):
    #    proceed_ps = does_ps_exist(mld.ps_path, mld.ps_files[0])
    #else:
    #    proceed_ps = False
    
    #try:
    #    already_done = does_full_npy_exist(date, init_time, mld.ps_init,\
    #                        mld.wofs_files[0], mld.wofs_files[-1], \
    #                        c.train_fcst_full_npy_dir)
    #except:
    #    already_done = False

    already_done_dat = does_dat_exist(date, init_time, mld.ps_init,\
                        mld.wofs_files[0], mld.wofs_files[-1], \
                        c.train_fcst_dat_dir)
    
    already_done_reps = does_reps_file_exist(date, mld.wofs_files[0], \
                                             mld.wofs_files[-1], c.train_obs_full_npy_dir, \
                                             report_radius)

    #Note: Can also check to make sure we don't already have a npy file
    #if (proceed_wofs == True and proceed_ps == True and already_done_reps == False):
    #if (proceed_wofs == True and proceed_ps == True and already_done == False):
    #if (not already_done_reps) and (proceed_wofs) and (proceed_ps): #reports only generation - set the generate forecasts to false in config

    #    ml.generate()
    #if (already_done) and (not already_done_dat):
    #    return
    #    print (date, init_time, 'wofs lead: ' + str(lead_time) + ' min', 'duration: ' + str(window) + ' min')
    #    create_new_rand_inds(date, init_time, mld.ps_init,\
    #                    mld.wofs_path, mld.wofs_files[0], mld.wofs_files[-1],\
    #                    c.train_fcst_dat_dir, c.train_fcst_full_npy_dir, ml)
    if proceed_wofs and already_done_dat:
        ml.generate()
    
    #if already_done and (not proceed_torp):
    #    start_valid, __ = wofs_phi.ForecastSpecs.find_date_time_from_wofs(mld.wofs_files[0], "forecast")
    #    
    #    end_valid, __ = wofs_phi.ForecastSpecs.find_date_time_from_wofs(mld.wofs_files[-1], "forecast") 
    #    model_type = c.model_type[5:]
    #    filename = "%s/wofs1d_%s_%s_%s_%s_v%s-%s.npy" %(c.train_fcst_full_npy_dir, model_type, date, init_time, \
    #                mld.ps_init, start_valid, end_valid)
    #    print(filename)
    #    os.remove(filename)
    
    #if (not (proceed_wofs and proceed_ps)) and (already_done or already_done_dat):
    #    del_preds(date, init_time, mld.ps_init,\
    #                    mld.wofs_path, mld.wofs_files[0], mld.wofs_files[-1],\
    #                    c.train_fcst_dat_dir, c.train_fcst_full_npy_dir)
        

    #elif (proceed_wofs and proceed_ps and ((date == "20190725") or (date == "20190705"))) and ((lead_time == 180) and init_time == "2300"):
    #    ml.generate()

    #if (already_done == True and already_done_dat == False):
    #    create_new_rand_inds(date, init_time, mld.ps_init,\
    #                    mld.wofs_path, mld.wofs_files[0], mld.wofs_files[-1], \
    #                    c.train_fcst_dat_dir, c.train_fcst_full_npy_dir)


    #if (proceed_wofs == True and proceed_ps == True and \
    #    already_done_reps == False):
    #
    #    #Create a report object and generate/save the 
    #    #reports grid 
    #        pass
    
    return

def del_preds(date_before_00z, wofs_initTime, ps_initTime, wofs_path, first_wofs_file,\
                         last_wofs_file, dat_path, npy_path):
    
    start_valid, __ = wofs_phi.ForecastSpecs.find_date_time_from_wofs(first_wofs_file, "forecast")

    end_valid, __ = wofs_phi.ForecastSpecs.find_date_time_from_wofs(last_wofs_file, "forecast") 
    
    if c.model_type == 'wofs_psv2_no_torp':
        npy_filename = "wofs1d_%s_%s_%s_v%s-%s.npy" %(date_before_00z, wofs_initTime, \
                    ps_initTime, start_valid, end_valid)
    else:
        model_type = c.model_type[5:]
        npy_filename = "wofs1d_%s_%s_%s_%s_v%s-%s.npy" %(model_type, date_before_00z, wofs_initTime, \
                    ps_initTime, start_valid, end_valid)
        
    if c.model_type == 'psv2_no_torp':
        dat_filename = "wofs1d_%s_%s_%s_v%s-%s.dat" %(date_before_00z, wofs_initTime, \
                    ps_initTime, start_valid, end_valid)
        rand_inds_filename = 'rand_inds_%s_%s_%s_v%s-%s.npy' %(date_before_00z, wofs_initTime, ps_initTime, start_valid, end_valid)
    else:
        model_type = c.model_type[5:]
        dat_filename = "wofs1d_%s_%s_%s_%s_v%s-%s.dat" %(model_type, date_before_00z, wofs_initTime, \
                    ps_initTime, start_valid, end_valid)
        rand_inds_filename = 'rand_inds_%s_%s_%s_%s_v%s-%s.npy' %(model_type, date_before_00z, wofs_initTime, ps_initTime, start_valid,
                                                                                       end_valid)
    
    try:
        os.remove('%s/%s' %(npy_path, npy_filename))
    except:
        pass
    try:
        os.remove('%s/%s' %(dat_path, dat_filename))
    except:
        pass
    try:
        os.remove('%s/%s' %(dat_path, rand_inds_filename))
    except:
        pass
    
    
    return

def create_new_rand_inds(date_before_00z, wofs_initTime, ps_initTime, wofs_path, first_wofs_file,\
                         last_wofs_file, dat_path, npy_path, ml):
    
    start_valid, __ = wofs_phi.ForecastSpecs.find_date_time_from_wofs(first_wofs_file, "forecast")

    end_valid, __ = wofs_phi.ForecastSpecs.find_date_time_from_wofs(last_wofs_file, "forecast") 
    
    if c.model_type == 'wofs_psv2_no_torp':
        npy_filename = "wofs1d_%s_%s_%s_v%s-%s.npy" %(date_before_00z, wofs_initTime, \
                    ps_initTime, start_valid, end_valid)
    else:
        model_type = c.model_type[5:]
        npy_filename = "wofs1d_%s_%s_%s_%s_v%s-%s.npy" %(model_type, date_before_00z, wofs_initTime, \
                    ps_initTime, start_valid, end_valid)
    
    try:
        full_pred_array = np.load('%s/%s' %(npy_path, npy_filename))
    except:
        print('regenerating: ' + npy_filename)
        ml.generate()
        return
    
    sample_rate = 0.1
    g = wofs_phi.Grid.create_wofs_grid(wofs_path, first_wofs_file)
    
    if c.model_type == 'wofs_psv2_no_torp':
        dat_filename = "wofs1d_%s_%s_%s_v%s-%s.dat" %(date_before_00z, wofs_initTime, \
                    ps_initTime, start_valid, end_valid)
        rand_inds_filename = 'rand_inds_%s_%s_%s_v%s-%s.npy' %(date_before_00z, wofs_initTime, ps_initTime, start_valid, end_valid)
    else:
        model_type = c.model_type[5:]
        dat_filename = "wofs1d_%s_%s_%s_%s_v%s-%s.dat" %(model_type, date_before_00z, wofs_initTime, \
                    ps_initTime, start_valid, end_valid)
        rand_inds_filename = 'rand_inds_%s_%s_%s_%s_v%s-%s.npy' %(model_type, date_before_00z, wofs_initTime, ps_initTime, start_valid,
                                                                                       end_valid)
        
    pex.save_predictors(full_pred_array, sample_rate, g, npy_path, npy_filename, dat_path, dat_filename, rand_inds_filename)
    
    return

def create_warning_mode_training(train_types):
    '''Will obtain the proper files, etc. when we're in warning mode.
        NOTE: In real time, Warning mode will be driven purely by the actual time.
        Similarly, in training mode, we will loop over a series of start times 
    '''

    mode = "warning" 

    window = 60 #Focus on 60 minute windows 
    date_file = (c.base_path/"probSevere_dates.txt")
    dates = read_txt(date_file)
    report_radius = 39 #in km 


    start_times = ["1735", "1805", "1835", "1905", "1935", "2005", "2035", "2105",\
                    "2135", "2205", "2235", "2305", "2335", "0005", "0035", "0105",\
                    "0135", "0205", "0235"]

    #Maybe for training in warning mode I'll pick a time at the top
    #of the hour, or something 

    dates = ["20190506"] 
    #start_times = ["2205", "2235", "2305", "2335", "0005"]
    #start_times = ["2200", "2230", "2300", "2330", "0000"] 
    #dates = ["20200507"]
    #start_times = ["2335", "2340", "2345", "2350"] 
    start_times = ["2300", "2305", "2310", "2315"] 

    #NOTE: date is the before-00z date 
    for d in range(len(dates)):
        date = dates[d] 
        for s in range(len(start_times)): 
            start_time = start_times[s] 
            #Create datetime object 
            dt = wofs_phi.ForecastSpecs.str_to_dattime(start_time, date) 

            #Need to increment this if start time is in the next day times
            if (start_time in c.next_day_times):
                dt += timedelta(days=1) 

            #Get the necessary info to start the driver 
            init_time, lead_time = MLDriver.get_info_for_warning_mode(\
                    dt, date) 

            print (date, start_time, init_time, lead_time) 

            #find start and end valid times 
            

            #Now, start MLDriver object 
            mld = MLDriver.start_driver(date, window, init_time, lead_time, c.ps_dir,\
                    mode)
    
            #Use this to drive the forecast 
            ml = wofs_phi.MLGenerator(mld.wofs_files, mld.ps_files, mld.ps_path,\
                        mld.wofs_path, mld.torp_files, c.nc_outdir, mode, train_types)

            #Check to make sure wofs files exist; if so we can generate.
            proceed_wofs = does_wofs_exist(mld.wofs_path, mld.wofs_files[0])

            proceed_ps = does_ps_exist(mld.ps_path, mld.ps_files[0]) 

            already_done = does_full_npy_exist(date, init_time, mld.ps_init,\
                                    mld.wofs_files[0], mld.wofs_files[-1], \
                                    c.train_fcst_full_npy_dir)

            already_done_reps = does_reps_file_exist(date, mld.wofs_files[0], \
                                mld.wofs_files[-1], c.train_obs_full_npy_dir, \
                                report_radius)
            

            #Note: Can also check to make sure we don't already have a npy file 
            if (proceed_wofs == True and proceed_ps == True and already_done == False):
            #if (proceed_wofs == True and proceed_ps == True):

                ml.generate()
            

            if (proceed_wofs == True and proceed_ps == True and \
                    already_done_reps == False):
            
                #Create a report object and generate/save the 
                #reports grid 
                pass

    return

def create_W2W_predictors(train_types):
    ''' Creates training files'''

    mode = "forecast" 

    window = 120 #Focus on 60 minute windows 
    
    #dates = ['20240313', '20230511', '20230324', '20230615', '20240521']
    dates = ['20240521']

    torpFiles = []

    report_radius = 15 #Only really used to check if reports file 
                        #already exists
    
    start = "1700" #first init time to use
    end = "1725" #last init time used
    training_start_times = get_w2w_init_times(start, end)

    lead_times = [0, 30, 60, 90]
    #lead_times = [0]
    #lead_times = [0, 30, 60, 90, 120, 150]
    #lead_times = [0, 30, 60]
    #lead_times = [90, 120, 150]

    #Get the data
    for lead_time in lead_times:
        for date in dates:
            for start_time in training_start_times:
                c.wofs_spinup_time = 0
                dt = wofs_phi.ForecastSpecs.str_to_dattime(start_time, date) 

                #Need to increment this if start time is in the next day times
                if (start_time in c.next_day_times):
                    dt += timedelta(days=1) 

                #Get the necessary info to start the driver 
                init_time, true_lead_time = MLDriver.get_info_for_warning_mode(\
                        dt, date)
                if (init_time == '0' and true_lead_time == 0):
                    continue
                
                true_lead_time += lead_time
                
                if (window + true_lead_time > 180) and (init_time[2:] == '30'):
                    init_time = init_time[0:2] + '00'
                    c.wofs_spinup_time = ((true_lead_time - c.ps_spinup_time) + 30) - lead_time
                    c.plot_forecasts = False
                    true_lead_time += 30
                else:
                    c.wofs_spinup_time = (true_lead_time - 5) - lead_time
                    c.plot_forecasts = True
                
                print (date, init_time, c.wofs_spinup_time, true_lead_time)
                
                if c.ps_version == 2:
                    ps2_dir = "/work/eric.loken/wofs/w2w_2024_ps2"
                    mld = MLDriver.start_driver(date, window, init_time, true_lead_time, ps2_dir,\
                                        mode)
                elif c.ps_version == 3:
                    mld = MLDriver.start_driver(date, window, init_time, true_lead_time, c.ps_dir,\
                                        mode)

                #Use this to drive the forecast 
                ml = wofs_phi.MLGenerator(mld.wofs_files, mld.ps_files, mld.ps_path,\
                        mld.wofs_path, mld.torp_files, c.nc_outdir, mode, train_types)
                
                forecast_specs = wofs_phi.ForecastSpecs.create_forecast_specs(mld.ps_files, mld.wofs_files, c.all_fields_file, c.all_methods_file, c.single_pt_file)

                #Check to make sure wofs files exist; if so we can generate. 
                #proceed_wofs = does_wofs_exist(mld.wofs_path, mld.wofs_files[0], init_time, true_lead_time) 

                #proceed_ps = does_ps_exist(mld.ps_path, mld.ps_files[0])

                #already_done = does_full_npy_exist(date, init_time, mld.ps_init,\
                #                    mld.wofs_files[0], mld.wofs_files[-1], \
                #                    c.train_fcst_full_npy_dir)
                
                #already_done_dat = does_dat_exist(date, init_time, mld.ps_init,\
                #                    mld.wofs_files[0], mld.wofs_files[-1], \
                #                    c.train_fcst_dat_dir)

                already_done_reps = does_reps_file_exist(date, mld.wofs_files[0], \
                                mld.wofs_files[-1], c.train_obs_full_npy_dir, \
                                report_radius)
                
                #Note: Can also check to make sure we don't already have a npy file
                #if (proceed_wofs == True and proceed_ps == True and already_done_reps == False):
                
                #if (proceed_wofs == True and proceed_ps == True and already_done == False):
                if (already_done_reps == False): #reports only generation - set the generate forecasts to false in config

                    ml.generate()
                    
                
                #elif (proceed_wofs and proceed_ps and ((date == "20190725") or (date == "20190705"))) and ((lead_time == 180) and init_time == "2300"):
                #    ml.generate()
                
                #if (already_done == True and already_done_dat == False):
                #    create_new_rand_inds(date, init_time, mld.ps_init,\
                #                    mld.wofs_path, mld.wofs_files[0], mld.wofs_files[-1], \
                #                    c.train_fcst_dat_dir, c.train_fcst_full_npy_dir)


                #if (proceed_wofs == True and proceed_ps == True and \
                #    already_done_reps == False):
                #
                #    #Create a report object and generate/save the 
                #    #reports grid 
                #        pass

    return

def get_w2w_init_times(start, end):
    times = []
    time = start
    start_int = int(start)
    if (int(end) < 1200) and (start_int >= 1200):
        end_test = int(end) + 2400
    else:
        end_test = int(end)
    while ((int(time) <= int(end)) and ((int(time) >= 1200))) or (int(time) <= end_test):
        time = str(time)
        time_min = time[-2:]
        if int(time_min) < 60:
            if int(time) >= 2400:
                append_time = int(time) - 2400
            else:
                append_time = int(time)
            times.append("%04d" % append_time)
        time = int(time) + 5
    
    return times

def does_full_npy_exist(date_before_00z, wofs_initTime, ps_initTime,\
            first_wofs_file, last_wofs_file, npy_path):
    '''Checks if we have the full npy training file'''
    
    #First, need to compute start and end valid times from first and 
    #last wofs files 
    #wofs1d_20190430_1900_v2000-2100.npy


    exists = False
    
    start_valid, __ = wofs_phi.ForecastSpecs.find_date_time_from_wofs(first_wofs_file, "forecast")

    end_valid, __ = wofs_phi.ForecastSpecs.find_date_time_from_wofs(last_wofs_file, "forecast") 
    
    if c.model_type == 'wofs_psv2_no_torp':
        filename = "%s/wofs1d_%s_%s_%s_v%s-%s.npy" %(npy_path, date_before_00z, wofs_initTime, \
                    ps_initTime, start_valid, end_valid)
    else:
        model_type = c.model_type[5:]
        filename = "%s/wofs1d_%s_%s_%s_%s_v%s-%s.npy" %(npy_path, model_type, date_before_00z, wofs_initTime, \
                    ps_initTime, start_valid, end_valid)
    
    if (os.path.isfile(filename)):
        exists = True
        #ti_c = os.path.getctime(filename)
        #c_ti = time.ctime(ti_c)
        #obj = time.strptime(c_ti)
        #dt = datetime.datetime(obj.tm_year, obj.tm_mon, obj.tm_mday, obj.tm_hour, obj.tm_min)
        #cutoff_dt = datetime.datetime(2025, 2, 23)
        #if dt > cutoff_dt:
        #    exists = True

    return exists

def does_dat_exist(date_before_00z, wofs_initTime, ps_initTime, first_wofs_file, last_wofs_file, npy_path):
    exists = False
    
    start_valid, __ = wofs_phi.ForecastSpecs.find_date_time_from_wofs(first_wofs_file, "forecast")

    end_valid, __ = wofs_phi.ForecastSpecs.find_date_time_from_wofs(last_wofs_file, "forecast") 
    
    if c.model_type == 'wofs_psv2_no_torp':
        filename = "%s/wofs1d_%s_%s_%s_v%s-%s.dat" %(npy_path, date_before_00z, wofs_initTime, \
                    ps_initTime, start_valid, end_valid)
    else:
        model_type = c.model_type[5:]
        filename = "%s/wofs1d_%s_%s_%s_%s_v%s-%s.dat" %(npy_path, model_type, date_before_00z, wofs_initTime, \
                    ps_initTime, start_valid, end_valid)
    
    if (os.path.isfile(filename)):
        exists = True
    #    ti_c = os.path.getctime(filename)
    #    c_ti = time.ctime(ti_c)
    #    obj = time.strptime(c_ti)
    #    dt = datetime.datetime(obj.tm_year, obj.tm_mon, obj.tm_mday, obj.tm_hour, obj.tm_min)
    #    cutoff_dt = datetime.datetime(2024, 4, 8)
    #    if dt > cutoff_dt:
    #        exists = True

    return exists


def does_reps_file_exist(date_before_00z, first_wofs_file, last_wofs_file, npy_path, \
                         radius):
    '''Checks if we have the full npy reps file'''

    exists = False
    
    start_valid, __ = wofs_phi.ForecastSpecs.find_date_time_from_wofs(first_wofs_file, "forecast")

    end_valid, __ = wofs_phi.ForecastSpecs.find_date_time_from_wofs(last_wofs_file, "forecast")
    
    filenames = ["%s/%s_reps1d_%s_v%s-%s_r%skm.npy" %(npy_path, h,\
                    date_before_00z, start_valid, end_valid, str(radius)) \
                    for h in c.final_hazards]

    exist_arr = [os.path.isfile(f) for f in filenames]
    
    if (sum(exist_arr) == len(c.final_hazards)):
        exists = True 

    return exists

def does_torp_exist(torp_files, date, init, lead):
    
    for file in torp_files:
        date_time = file.split('_')[-4]
        false_time = date_time.split('/')[-1]
        time = false_time.split('-')[1]
        dt = datetime.datetime(int(date[0:4]), int(date[4:6]), int(date[6:]), int(time[0:2]), int(time[2:4]))
        start_dt = datetime.datetime(int(date[0:4]), int(date[4:6]), int(date[6:]), int(init[0:2]), int(init[2:4]))\
        + timedelta(seconds = 30*60)
        
        if dt.hour < 12:
            dt += timedelta(days = 1)
        if start_dt.hour < 12:
            start_dt += timedelta(days = 1)
        
        if ((start_dt - dt).seconds <= (35*60)) and ((start_dt - dt).seconds >= (5*60)):
            return True
    
    return False

def does_wofs_exist(wofs_direc, wofs_ALL_file, init_time, lead_time):
    '''Method to test if the first wofs file exists.
        Returns True if so, else Returns False
        @wofs_direc is the path to the wofs files
        @wofs_ALL_file is the first wofs file (ALL format)
    '''

    exists = False #Assume file doesn't exist

    fullFile = "%s/%s" %(wofs_direc, wofs_ALL_file)
    legacyNames = wofs_phi.WoFS_Agg.get_legacy_filenames("mslp", [wofs_ALL_file]) 
    legacyName = legacyNames[0]    

    fullLegacyFile = "%s/%s" %(wofs_direc, legacyName)
    
    if (os.path.isfile(fullFile) or os.path.isfile(fullLegacyFile)):
        exists = True
    if (init_time[2:] == "30") and (lead_time > 120):
        exists = False

    return exists


def does_ps_exist(ps_direc, curr_ps_file):
    '''Method to test if the current probSevere file exists. 
        @Returns True if so, else Returns False.
        @ps_direc is the path tot he probSevere files 
        @curr_ps_file is the current ProbSevere file name
    '''

    exists = False #Assume file doesn't exist
    fullFile = "%s/%s" %(ps_direc, curr_ps_file)
    if (os.path.isfile(fullFile)):
        exists = True


    return exists


def read_txt(txt_file):
    '''Reads in a given text file to string list'''

    return np.genfromtxt(txt_file, dtype='str') 


def main():
    '''Main method'''
    
    warnings.filterwarnings('ignore')
    
    #SET mode here 
    mode_to_generate = "forecast"
    #mode_to_generate = "warning"
    #mode_to_generate = "w2w"
    
    #SET train type here 
    #options: "obs", "warnings", or "obs_and_warnings"
    #train_types = ["obs_and_warnings"] #only need it set to one when generating predictors
    training = True
    report_radii = [36]
    date_file = (c.base_path/"probSevere_dates.txt")
    dates = read_txt(date_file)
    windows = [60]#, 120]
    init_times = copy.deepcopy(c.all_wofs_init_times)
    lead_times = [30, 60, 90, 120, 150, 180]
    
    #new_dates = ['20200429', '20200430', '20200501', '20200504', '20200505', '20200506',\
    #         '20200626', '20200627', '20200630', '20200701', '20200824', '20200825',\
    #         '20200826', '20200909', '20200910']
    #dates = list(dates)
    #for d in new_dates:
    #    if not (d in dates):
    #        dates.append(d)
    #dates = np.array(dates)
    #dates = ['20190501']
    #lead_times = [30]#, 60, 90, 120, 150, 180]
    #init_times = ['0500']
    #windows = [60]
    
    iterator = md.to_iterator(dates, init_times, windows, lead_times,\
                              [mode_to_generate], [training], report_radii)
    results = md.run_parallel(gen_preds, iterator, nprocs_to_use = 30,\
                              description = 'Generating New Obs')
    
    #if (mode_to_generate == "forecast"):
    #    create_forecast_mode_training(train_types)
    
    #elif (mode_to_generate == "warning"):
    #    create_warning_mode_training(train_types)
    
    #elif (mode_to_generate == "w2w"):
    #    create_W2W_predictors(train_types)

    #date = "20190430" #date before 00z 
    #window = 60
    #init_time = "0100"
    #lead_time = 25

    #torpFiles = [] #NOTE: Will probably eventually want to include torp files 
    #as part of this 

    #Create MLDriver object 
    #mld = MLDriver.start_driver(date, window, init_time, lead_time, c.ps_dir) 

    #NOTE: Can now use this to drive the forecast 
    #ml = wofs_phi.MLGenerator(mld.wofs_files, mld.ps_files, mld.ps_path,\
    #            mld.wofs_path, torpFiles, c.nc_outdir)

    #ml.generate()
    #Get wofs_files 
    
    print('done')

    return 




if (__name__ == '__main__'):

    main() 

