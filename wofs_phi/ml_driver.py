#===================================================
# Script that will drive the ML training and testing
#===================================================

from wofs_phi import * 
import config as c



#Could also call this, e.g., FileObtainer or something. 
class MLDriver:
    ''' Handles the driving of the ML for training or testing.
        i.e., Sets the relevant filenames, etc.
    '''

    def __init__(self, pre00z_date, time_window, wofs_init, wofs_lead_time):
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

        print (wofs_file_list) 

        #Find ps files 


        #Create MLDriver object


        #Return MLDriver object 
        

        return 

    
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
        
        #Find wofs times (string)
    
        #Find wofs dates (string) 

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

def main():
    '''Main method'''

    date = "20190430"
    window = 60
    init_time = "2300"
    lead_time = 25
    ps_direc = "/work/eric.loken/wofs/probSevere"

    #Create MLDriver object 
    ml_driver = MLDriver.start_driver(date, window, init_time, lead_time, ps_direc) 
    
    #Get wofs_files 

    return 




if (__name__ == '__main__'):

    main() 

