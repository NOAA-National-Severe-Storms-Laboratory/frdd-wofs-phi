#========================================================
# This script drives the real-time generation of wofs-phi
# in forecast mode 
#========================================================

from wofs_phi import *
import config as c
import os.path
from itertools import compress
from ml_driver import MLDriver, WofsFile
import datetime as dt 
import copy 
import math 


class StartRT:

    '''Starts the realtime ML generation.'''

    #Constants
    WINDOW = 60 #Time window in minutes (i.e., length of forecast valid period) 

    WOFS_INIT_FREQ = 30 #Wofs is initialized every 30 minutes 

    #Wofs lead times at the start of the valid period we're interested in
    WOFS_LEAD_TIMES = [30, 60, 90, 120, 150, 180] 

    FIRST_WOFS_INIT = "1700" 

    WARNING_MODE_UPDATE_FREQ = 5 #How often is warning mode updated? 

    #Maps current time str to wofs init str
    #TODO: But have I already done this for warning mode? 
    #
    #NOTE: Might not make this a class constant
    #WOFS_INIT_DICT = create_wofs_init_dict("1700", "0900", c.wofs_spinup_time)


    def __init__(self, wofs_init_time_dt, ps_init_time_dt, start_valid_time_dt, \
                    end_valid_time_dt, current_time_dt, first_wofs_init_dt, \
                    date_before_00z, date_after_00z, wofs_files, ps_files, proceed):

        '''Initializes a StartRT object. 
            @wofs_init_time_dt is datetime object of wofs initialization time
            @ps_init_time_dt is datetime object of probsevere initialization time
                to use
            @start_valid_time_dt is datetime object of forecast start valid time
            @end_vaid_time_dt is the datetime object of forecast end valid time 
            @current_time_dt is the current datetime object
            @first_wofs_init_dt is the datetime object corresponding to the first
                wofs initialization time for the current situation (i.e., 1700Z in
                most cases). 
            @date_before_00z is the 8-char string date before 00z 
            @date_after_00z is the 8-char string date after 00z 
            @wofs_files is the list of wofs files to use to drive the ML generation
            @ps_files is the list of ProbSevere files to use to drive the ML generation
            @proceed is a boolean variable; True if we have all the files we need available 
                to make the prediction; else False 
        '''

        
        self.wofs_init_time_dt = wofs_init_time_dt 
        self.ps_init_time_dt = ps_init_time_dt 
        self.start_valid_time_dt = start_valid_time_dt
        self.end_valid_time_dt = end_valid_time_dt  

        self.current_time_dt = current_time_dt 

        self.first_wofs_init_dt = first_wofs_init_dt 

        self.date_before_00z = date_before_00z
        self.date_after_00z = date_after_00z

        self.wofs_files = wofs_files
        self.ps_files = ps_files 

        self.proceed = proceed



    @classmethod
    def start_rt_from_dt(cls, current_datetime, wofs_lead_time):
        '''Creates/returns a StartRT object from the current datetime (and list of 
            class constants
            @current_datetime is the datetime object corresponding to the current time.     
            @wofs_lead_time is the time (in minutes) between the wofs initialization time
                and the start of the valid period (for forecast mode) we're interested in
        '''

        #Initialize a new StartRT object 
        start_rt = StartRT(current_datetime, current_datetime, current_datetime, \
                            current_datetime, current_datetime, current_datetime, \
                            "00000000", "00000000", [], [], False)


        #Set the date before_00z 
        start_rt.set_date_before_after_00z() 

        #Set the first wofs initialization time 
        start_rt.set_first_wofs_init_time() 

        #Obtain wofs initialization time 
        start_rt.set_wofs_init_time()

        #Obtain PS initialization time 
        start_rt.set_ps_init_time() 


        #NOTE: These will depend on if we're in forecast mode or warning mode 
        #Obtain start valid time 
        start_rt.set_start_valid(wofs_lead_time)

        #Obtain end valid time 
        start_rt.set_end_valid() 


        #Obtain list of wofs files to use
        start_rt.set_wofs_list()


        print (start_rt.wofs_init_time_dt)
        print (start_rt.start_valid_time_dt)
        print (start_rt.end_valid_time_dt) 
        print (start_rt.wofs_files) 


        #Obtain list of probsevere files to use 
        start_rt.set_ps_list()

        print (start_rt.ps_files) 


        #Check if wofs/ps files exist         



        return 

    
    def set_proceed(self):

        return 


    def set_ps_list(self):
        #TODO: I don't know that we'll use this method 

        self.ps_files = MLDriver.find_ps_files_from_first_wofs(self.wofs_files[0])  

        return 


    def set_wofs_list(self):

        __, wofs_init_time = ForecastSpecs.dattime_to_str(self.wofs_init_time_dt) 

        wofs_lead_time_min = ForecastSpecs.subtract_dt(self.start_valid_time_dt, \
                    self.wofs_init_time_dt, True) 

        self.wofs_files = MLDriver.find_wofs_file_list(self.WINDOW, wofs_init_time, \
                wofs_lead_time_min, self.date_before_00z)

        return 


    def set_end_valid(self):
        '''Sets the end valid dt by simply stepping forward in time
            by the forecast window'''

        self.end_valid_time_dt = self.start_valid_time_dt + dt.timedelta(minutes=self.WINDOW)

        return 

    
    def set_start_valid(self, wofs_lead_time=30):
        '''Sets the start valid time; this will depend on if we're in forecast or warning mode
            In warning mode: Is the current time divisible by 5? Then add 5 minutes (ps spinup time) 
                Otherwise, look at the remainder: If 1, then add 4 minutes. If 2, then add 3 minutes, 
                If 3, then add 7 minutes; if 4, then add 6 minutes 

            In forecast mode: add 30 minutes to the wofs initialization time + add'l wofs lead time

            @wofs_lead_time is the time (in minuts) from the most 
                recent wofs initialization to the start of 
                the valid period--only applicable for forecast mode, since we'll be wanting to 
                get different wofs lead times. default is 30 (for 30 minutes). 

        '''

        if (c.mode == "warning"):
            #Find out how far away the current time is from 5 minutes 
            str_date, str_time = ForecastSpecs.dattime_to_str(self.current_time_dt) 

            int_time = int(str_time) 
            #5 is the frequency of warning mode updates 
            remainder = int_time % WARNING_MODE_UPDATE_FREQ
        
            #Step ahead to next 5 minutes if early enough
            if (remainder < math.ceil(WARNING_MODE_UPDATE_FREQ/2)): 
                self.start_valid_time_dt = self.current_time_dt + \
                                        dt.timedelta(minutes=WARNING_MODE_UPDATE_FREQ-remainder) 

            #Otherwise, have to step ahead a bit farther 
            else: 
                self.start_valid_time_dt = self.current_time_dt + \
                    dt.timedelta(minutes=WARNING_MODE_UPDATE_FREQ-remainder+WARNING_MODE_UPDATE_FREQ)
                    

        elif (c.mode == "forecast"):
            self.start_valid_time_dt = self.wofs_init_time_dt + dt.timedelta(minutes=wofs_lead_time) 



        return 


    def set_ps_init_time(self):

        '''Sets the probSevere initialization time. We want to use the most recent ps file 
            available. Keep in mind the PS files are generally output every 2 minutes, 
            coming available on the even minutes.'''



        ps_dt = copy.deepcopy(self.current_time_dt) 

        date_str, time_str = ForecastSpecs.dattime_to_str(self.current_time_dt) 

        #Convert time string to integer (so we can test if it's even) 
        time_int = int(time_str) 
        
        #If this is odd, then substract 1 minute 
        if (time_int%2 == 1):  
            ps_dt -= dt.timedelta(minutes=1) 


        #Make the assignment 
        self.ps_init_time_dt = ps_dt 
         

        return 


    def set_wofs_init_time(self):   
        '''Sets the wofs initialization time (primarily based on the current datetime)''' 

        #Take the subtraction (in minutes) of the current time and the first wofs initialization time 
        time_since_first_initialization = ForecastSpecs.subtract_dt(self.current_time_dt, self.first_wofs_init_dt, True)

        
        num_inits = (time_since_first_initialization-c.wofs_spinup_time)//self.WOFS_INIT_FREQ
    
        #If we haven't yet exceeded the first spinup time yet, I suppose the thing to do would be to assign the init time
        #as the very first wofs initialization time 

        if (num_inits < 0):
            curr_wofs_init = self.FIRST_WOFS_INIT 

        else:  
            curr_wofs_init = c.all_wofs_init_times[num_inits]

        #Get datetime object representation 
        if (self.get_curr_date_str() == self.date_before_00z):

            curr_wofs_init_dt = ForecastSpecs.str_to_dattime(curr_wofs_init, self.date_before_00z)

        else: 
            curr_wofs_init_dt = ForecastSpecs.str_to_dattime(curr_wofs_init, self.date_after_00z) 


        #Set the attribute 
        self.wofs_init_time_dt = curr_wofs_init_dt 


        return 

    def set_first_wofs_init_time(self):

        '''Sets the first/earliest wofs initialization time'''

        #set the attribute 
        self.first_wofs_init_dt = ForecastSpecs.str_to_dattime(\
                                    self.FIRST_WOFS_INIT, self.date_before_00z) 


        return 

    def set_date_before_after_00z(self):
        date_str, time_str = ForecastSpecs.dattime_to_str(self.current_time_dt)


        date_before_00z = copy.deepcopy(date_str)
        date_after_00z = copy.deepcopy(date_str) 

        if (time_str[0] == "0" or time_str[0:1] == "10" or time_str[0:1] == "11"):
            #In this case, we're after 00z, so we have to go back 1 day
            before_00z_dt = self.current_time_dt - dt.timedelta(days=1)

            date_before_00z, __ = ForecastSpecs.dattime_to_str(before_00z_dt)

        else: 
            after_00z_dt = self.current_time_dt + dt.timedelta(days=1)
            date_after_00z, __ = ForecastSpecs.dattime_to_str(after_00z_dt) 


        #Set the before 00z date and after 00z date 

        #self.before_00z_date = date_before_00z
        #self.after_00z_date = date_after_00z
        self.date_before_00z  = date_before_00z
        self.date_after_00z = date_after_00z


        return 

    def get_curr_date_str(self):
        '''Returns the 8-char string date of the current time'''

        date_str, time_str = ForecastSpecs.dattime_to_str(self.current_time_dt)


        return date_str 



def main():

    #Get current datetime object 
    #curr_datetime  = dt.datetime.now()
    curr_datetime = dt.datetime.utcnow() 

    #lead_times = [30, 60, 90, 120, 150, 180] 
    lead_times = [30]

    for lead_time in lead_times:
        StartRT.start_rt_from_dt(curr_datetime, lead_time)




    return 


if (__name__ == '__main__'):

    main() 

