#===========================================================
# This module contains a number of utility/helper functions
# that are used in WoFS-PHI development 
#===========================================================

import numpy as np 
import copy 
from datetime import datetime, timedelta
import datetime as dt 


#TODO: Not done yet. Also may need to rework. 
def find_date_time_from_wofs(wofs_file, fcst_type): 

    """ Finds/returns the string time, string date, 
        and datetime object corresponding to either
        the wofs initialization time or wofs valid time associated with the given
        wofs summary file. NOTE: Assumes the format similar to the following: 
        "wofs_ALL_${index}_${initialization_date}_${initialization_time}_${end_valid_time}.nc"

        @wofs_file : str : WoFS summary filename 
        @fcst_type : str : Tells what type of forecast this is (i.e., which
            time to return). "valid" means it will return the (end) valid time; 
            "init" means it will retrun the initialization time. 

        NOTE: The date returned will be the date associated with the valid time, 
        NOT the initialization time. As a result, the date string returned might
        differ from the date in the wofs file name. 
    """ 


    #Split the string based on underscores 
    split_str = wofs_file.split("_")

    #Find initialization time
    initTime = split_str[4]

    validTime = split_str[5] #In this case, it'll be the 5th element of the wofs file string

    #Have to remove the .nc
    validTime = validTime.split(".")[0]


    date = split_str[3] #date listed in wofs file 

    #New approach will be to generate 2 datetime objects (1 for init time and 1 for end valid
    #time). If end valid time is earlier than initial time, then increment end valid time by
    #1 day. 

    dt_initial = datetime.strptime(f"{date} {initTime}", "%Y%m%d %H%M") 
    dt_valid = datetime.strptime(f"{date} {validTime}", "%Y%m%d %H%M")

    if (dt_initial > dt_valid):
        dt_valid += timedelta(days=1) 


    if (fcst_type == "valid"): 

        time = copy.deepcopy(validTime) 
        date = dt_valid.strftime("%Y%m%d") 
        use_datetime = copy.deepcopy(dt_valid) 

    elif (fcst_type == "init"): 

        time = copy.deepcopy(initTime)
        date = dt_initial.strftime("%Y%m%d") 
        use_datetime = copy.deepcopy(dt_initial) 

    return time, date, use_datetime

    #OLD code: 
    ##NOTE: Let's modify the below code to be cleaner. 

    ##Increment the date listed if the init time is in the previous day but the
    ##valid time is in the next day -- but only if we're concerned with returning
    ##the forecast date/time and not the initialization date/time 
    #if ((fcst_type=="forecast") and (validTime in c.next_day_times)\
    #        and (initTime not in c.next_day_inits)):
    #    dt = ForecastSpecs.str_to_dattime(validTime, date)
    #    dt += timedelta(days=1)
    #    date, __ = ForecastSpecs.dattime_to_str(dt)

    #if (fcst_type == "forecast"):
    #    time = validTime
    #elif (fcst_type == "init"):
    #    time = initTime
    #
    #    return time, date



    return 
