#====================================================
# This script generates a .json config file based on 
# user specified parameters. 
# The output .json file can then be used as an input
# when running the wofs_phi code. 
#====================================================

import numpy as np
import json 


def save_dict_to_json(in_dict, output_filename): 
    """Saves a dictionary to output json file. 
        @in_dict : Python dictionary to convert to json"""

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(in_dict, f, indent=4)
    

    return 


def main(): 

    #=================================================================
    #Parameters to set, which will be incorporated into the json file:
    #=================================================================
    

    json_output_dir = "/home/eric.loken/python_packages/frdd-wofs-phi/wofs_phi/config_json_files"

    #Specify output name here: 
    outfile_name = "config_training.json"

    full_outfile = f"{json_output_dir}/{outfile_name}"


    #========================================================
    # Set variables for dictionary/json config file 
    #========================================================

    generate_forecasts = True #Generates the predictors array if True
    generate_reports = True #Generates the reports file if True 
    save_npy = True #Tells whether or not to save the npy predictor files 
    save_ncdf = True #Tells whether or not to create/save the ncdf (realtime) files
    plot_forecasts = True #Tells whether or not to create the .png files for wofs viewer

    nc_outdir = "." #Where to place the final netcdf files #Needed for real time
    #nc_outdir = "/home/eric.loken/python_packages/frdd-wofs-phi/wofs_phi/ncdf"

    #If True, use the ALL naming convention (will be true on cloud) 
    #If False, use the legacy naming convention (e.g., ENS, ENV, SVR, etc.) 
    use_ALL_files = True
    

    #=========================================================

    #Create the dictionary and save to file 

    json_dict = {"generate_forecasts": generate_forecasts, \
                "generate_reports": generate_reports, "save_npy": save_npy, \
                "save_ncdf": save_ncdf, "plot_forecasts": plot_forecasts, \
                "nc_outdir": nc_outdir, "use_ALL_files": use_ALL_files}


    #Save to file 
    save_dict_to_json(json_dict, full_outfile) 


    return 


if (__name__ == '__main__'): 


    main() 

