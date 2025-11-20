#===========================================================
# This module contains a number of utility/helper functions
# that are used in WoFS-PHI development 
#===========================================================

import numpy as np 
import copy 


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
 
