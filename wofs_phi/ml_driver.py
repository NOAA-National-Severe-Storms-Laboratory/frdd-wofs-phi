#===================================================
# Script that will drive the ML training and testing
#===================================================

from wofs_phi import * 
import config as c
import os.path



#Could also call this, e.g., FileObtainer or something. 
class MLDriver:
    ''' Handles the driving of the ML for training or testing.
        i.e., Sets the relevant filenames, etc.
    '''

    def __init__(self, pre00z_date, time_window, wofs_init, wofs_lead_time,\
                    wofs_path, ps_path, wofs_files, ps_files):
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
        '''

        self.pre00z_date = pre00z_date
        self.time_window = time_window
        self.wofs_init = wofs_init
        self.wofs_lead_time = wofs_lead_time
        self.wofs_path = wofs_path
        self.ps_path = ps_path
        self.wofs_files = wofs_files
        self.ps_files = ps_files 

        return 

    @classmethod 
    def start_driver(cls, before00zDate, timeWindow, wofsInitTime, wofsLeadTime, psPath):
        '''Creates an MLDriver object given a date (from before 00z; @before00zDate), 
            time window (in minutes; @timeWindow), 
            wofs initialization time (string (YYYYMMDD); @wofsInitTime), 
            wofs lead time (in minutes; @wofsLeadTime), and probSevere path (@psPath)
            @Returns MLDriver object 

        '''

        #Find wofs path 
        wofsPath = MLDriver.find_wofs_path(before00zDate, wofsInitTime) 

        #Find wofs files 
        wofs_file_list = MLDriver.find_wofs_file_list(timeWindow, wofsInitTime, wofsLeadTime,\
                                before00zDate) 

        #Find ps files 
        ps_file_list = MLDriver.find_ps_files_from_first_wofs(wofs_file_list[0])

        #Create MLDriver object
        obj = MLDriver(before00zDate, timeWindow, wofsInitTime, wofsLeadTime, wofsPath,\
                c.ps_dir, wofs_file_list, ps_file_list) 


        return obj


    @staticmethod
    def find_ps_files_from_first_wofs(first_wofs_file):
        ''' Finds the set of ProbSevere files given the first wofs
            file
        '''
        
        #First, need to get the datetime object associated with the
        #first wofs file
        wofs_time, wofs_date = ForecastSpecs.find_date_time_from_wofs(\
                first_wofs_file, "forecast") 


        wofs_dt = ForecastSpecs.str_to_dattime(wofs_time, wofs_date) 

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
            
            if (c.ps_version == 2):
                #get date and time from datetime object
                date_str, time_str = ForecastSpecs.dattime_to_str(dt)
                
                #TODO: Might need to add capability to check for each second
                #for real-time. Might also not be a problem in real time. 
                ps_name = "MRMS_EXP_PROBSEVERE_%s.%s00.json" %(date_str, time_str)
                
            
                
            #TODO: Need to implement capabilities for version 3 
            elif (c.ps_version == 3):

                pass    
    

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
            @Returns datetime object corresponding to the first
                ProbSevere time
        '''

        
        #See if we're at an even minute. If so, dt_wofs is the 
        #first probSevere dt. Else, we need to subtract 1 minute

        #Get time string from datetime object
        __, time_str = ForecastSpecs.dattime_to_str(dt_wofs)

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

        wofs_date, wofs_i_time_str = ForecastSpecs.dattime_to_str(wofs_init_dt) 
        

        return wofs_date

    
    def get_valid_time(self):
        
        #Get datetime object for the initialization time 
        wofs_init_dt = WofsFile.get_wofs_dt(self.init_time, self.date_before_00z)
        
        #Convert index into a displacement time -- i.e., 
        #time to add to the wofs initialization time 
        displacement_time = float(self.index)*c.wofs_update_rate

        wofs_valid_dt = wofs_init_dt + timedelta(minutes=displacement_time) 

        #Convert to string date and time 
        valid_date, valid_time_str = ForecastSpecs.dattime_to_str(wofs_valid_dt) 

        
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
        dt_obj = ForecastSpecs.str_to_dattime(time_string, before_00z_date_string)

        #If it's after 00z, add 1 day to the above datetime object

        if (time_string in c.next_day_inits):
            dt_obj += timedelta(days=1) 


        return dt_obj


def create_training():
    ''' Creates training files'''

    window = 60 #Focus on 60 minute windows 
    
    date_file = "probSevere_dates.txt"

    dates = read_txt(date_file) 

    torpFiles = [] 

    training_init_times = ["1700", "1730", "1800", "1830", "1900",\
        "1930", "2000", "2030", "2100", "2130", "2200", "2230", \
        "2300", "2330", "0000", "0030", "0100", "0130", "0200"]

    #These are the most important right now: 
    #60 and 120 lead times are for forecast mode in SFE;
    #15 min lead time will be used for warning mode
    #lead_times = [60, 120, 15, 180] #These are the most important right now
    lead_times = [60]

    #Get the data
    for lead_time in lead_times:
        for date in dates:
            for init_time in training_init_times:
                print (date, init_time, lead_time) 
                mld = MLDriver.start_driver(date, window, init_time, lead_time, c.ps_dir)

                #Use this to drive the forecast 
                ml = MLGenerator(mld.wofs_files, mld.ps_files, mld.ps_path,\
                        mld.wofs_path, torpFiles, c.nc_outdir)

                #Check to make sure wofs files exist; if so we can generate. 
                proceed = does_wofs_exist(mld.wofs_path, mld.wofs_files[0]) 
                if (proceed == True):
            
                    ml.generate()

                
        
 
    return 


def does_wofs_exist(wofs_direc, wofs_ALL_file):
    '''Method to test if the first wofs file exists.
        Returns True if so, else Returns False
        @wofs_direc is the path to the wofs files
        @wofs_ALL_file is the first wofs file (ALL format)
    '''

    exists = False #Assume file doesn't exist

    fullFile = "%s/%s" %(wofs_direc, wofs_ALL_file) 
    legacyNames = WoFS_Agg.get_legacy_filenames("mslp", [wofs_ALL_file]) 
    legacyName = legacyNames[0]    

    fullLegacyFile = "%s/%s" %(wofs_direc, legacyName)
    
    if (os.path.isfile(fullFile) or os.path.isfile(fullLegacyFile)):
        exists = True
        

    return exists


def read_txt(txt_file):
    '''Reads in a given text file to string list'''

    return np.genfromtxt(txt_file, dtype='str') 


def main():
    '''Main method'''

    create_training()

    #date = "20190430" #date before 00z 
    #window = 60
    #init_time = "0100"
    #lead_time = 25

    #torpFiles = [] #NOTE: Will probably eventually want to include torp files 
    #as part of this 

    #Create MLDriver object 
    #mld = MLDriver.start_driver(date, window, init_time, lead_time, c.ps_dir) 

    #NOTE: Can now use this to drive the forecast 
    #ml = MLGenerator(mld.wofs_files, mld.ps_files, mld.ps_path,\
    #            mld.wofs_path, torpFiles, c.nc_outdir)

    #ml.generate()
    #Get wofs_files 

    return 




if (__name__ == '__main__'):

    main() 

