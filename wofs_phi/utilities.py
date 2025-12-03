#===========================================================
# This module contains a number of utility/helper functions
# that are used in WoFS-PHI development 
#===========================================================

import numpy as np 
import copy 
from datetime import datetime, timedelta
import datetime as dt 
import json 


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


def get_date_before_00z(wofsInitTimeDT, jsonFileName):
    """Returns the 8-char string (YYYYMMDD) corresponding to the forecast date
        before 00z, assuming that times between 0000 and 1155 belong to the "next day"
        after 00z. This should be most important/useful/relevant for training, which
        has been done on "standard" wofs runs.

        @wofsInitTimeDT : datetime obj : Wofs initialization time
        @jsonFileName : str : Name of .json config file used 
    """

    #Get the list of next day times from the json config file 

    config_data = read_json(jsonFileName) 
    next_day_times = config_data['next_day_times'] 

    date_string, time_string = dattime_to_str(wofsInitTimeDT)

    #If the time string is in the next_day times, then we need to find
    #the date string from the day before
    
    if (time_string in next_day_times): 
        
        day_before_dt = wofsInitTimeDT - timedelta(days=1) 

        #Get updated date string

        date_string, time_string = dattime_to_str(day_before_dt) 


    return date_string


def subtract_dt(dt1, dt2, inMinutes): 

    """ Returns the difference (in datetime format) resulting from 
        dt1 - dt2. 
        @dt1 : datetime object 
        @dt2 : datetime object
        @inMinutes : boolean : If True, returns the subtraction in minutes, 
            If False, returns a timedelta object
    """


    difference = dt1 - dt2 

    if (inMinutes == True): 
        difference = timedelta_to_min(difference) 
    
    return difference

def timedelta_to_min(in_dt): 
    """Converts the timedelta object to minutes. Returns the number of minutes. 
        @in_dt : timedelta object : To be converted to minutes
    """

    minutes = int(in_dt.total_seconds()/60)

    return minutes


def dattime_to_str(dt_obj): 
    """Returns 8-char date string (YYYYMMDD) and 4-char time string (HHMM) 
        based on datetime object. 
        @dt_obj : datetime object : datetime object to convert to string
    """

    new_date_string = dt_obj.strftime("%Y%m%d") 
    new_time_string = dt_obj.strftime("%H%M") 

    return new_date_string, new_time_string 


def read_json(jsonFilename): 
    """ Reads in json file. Returns json data.
        @jsonFilename : str : .json filename to read in
    """
    
    with open(jsonFilename, 'r') as file:
        data = json.load(file)

    return data


def find_ps_date_time(ps_file, jsonConfigFilename): 
    """ Finds/Returns the (string) probSevere initialization time and date 
        strings from the ProbSevere file 
        @ps_file : str : Name of the ProbSevere file 
        @jsonConfigFilename : str : Name of the .json config file 
    """

    #Obtain ProbSevere version (ps_version) from the .json config file
    config_data = read_json(jsonConfigFilename) 

    ps_version = config_data['ps_version']

   
    m = re.match('.*_(?P<date>\d{8})_(?P<time>\d{4})\d{2}.json$', ps_file)

    if m != None:
        return (m.group('time'), m.group('date'))
    
    if (ps_version == 2):
        split_str = ps_file.split(".")
        time = split_str[1]

        #now get the date
        date_split = ps_file.split("_")[3]

        #get *only* the date (first 8 characters)
        date = date_split[0:8]

    #NOTE: Haven't debugged this yet 
    elif (ps_version == 3):

        split_str = ps_file.split("_")
        time = split_str[4]
        time = time.split(".")[0]

        #Now get the date 
        date = split_str[3]

    #Now, remove the seconds 
    time = time[0:4]


    return time, date


def str_to_dattime(string_time, string_date): 

    """Returns a datetime object based on a string time and string date. 
        @string_time : 4-char string : Time, Format HHMM, e.g., "0025"
        @string_date : 8-char string : Date, Format YYYYMMDD, e.g., "20190504"
    """

    #combine string date and time 
    full_string = f"{string_date}{string_time}"

    dt_obj = datetime.strptime(full_string, "%Y%m%d%H%M")


    return dt_obj



