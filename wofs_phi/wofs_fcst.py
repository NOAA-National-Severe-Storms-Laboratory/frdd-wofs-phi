#==================================================================
# This module handles the processing related to the WoFS forecasts
# (i.e., temporal aggregation, etc.). 
#==================================================================

import numpy as np 
import copy 
import utilities as utils


class Wofs: 

    """Will handle code related to basic reading of individual WoFS 
        files (i.e., at a given time step)
    """ 
    
    #Legacy file naming conventions 
    ENS_VARS = ["ws_80", "dbz_1km", "wz_0to2_instant", "uh_0to2_instant",  "uh_2to5", "w_up",\
                     "w_1km", "w_down", "buoyancy", "div_10m", "10-500m_bulkshear", "ctt", "fed",\
                     "rh_avg", "okubo_weiss", "hail", "hailcast", "freezing_level", "comp_dz"]

    ENV_VARS = ["mslp", "u_10", "v_10", "td_2", "t_2", "qv_2", "theta_e", "omega", "psfc", \
                            "pbl_mfc", "mid_level_lapse_rate", "low_level_lapse_rate" ]

    SVR_VARS = ["shear_u_0to1", "shear_v_0to1", "shear_u_0to3", "shear_v_0to3", "shear_u_0to6", "shear_v_0to6",\
                      "srh_0to500", "srh_0to1", "srh_0to3", "cape_sfc", "cin_sfc", "lcl_sfc", "lfc_sfc",\
                       "stp", "scp", "stp_srh0to500"]
    


    def __init__(self): 

        return 




class Wofs_List: 

    """Container class of multiple WoFS objects. Useful for 
        performing operations like temporal aggregation. 
    """


    def __init__(self): 
    
        return 


#Helper functions 
def get_legacy_filenames(wofs_var_name, wofs_files_list):
    """ Returns a list of wofs filenames in legacy format. 
        e.g., replacing the "ALL" with "ENS" or "ENV", etc. 
        @wofs_var_name : string wofs variable name
        @wofs_files_list : list of wofs filenames 
            (with the ALL convention) 
    """

    #Default
    new_names = copy.deepcopy(wofs_files_list)

    if (wofs_var_name in Wofs.ENS_VARS):
        new_names = [s.replace("ALL", "ENS") for s in wofs_files_list]

    elif (wofs_var_name in Wofs.ENV_VARS):
        new_names = [s.replace("ALL", "ENV") for s in wofs_files_list]

    elif (wofs_var_name in Wofs.SVR_VARS):
        new_names = [s.replace("ALL", "SVR") for s in wofs_files_list]

    return new_names
