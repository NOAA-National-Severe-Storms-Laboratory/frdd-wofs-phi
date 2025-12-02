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

def get_methods_dict_from_files(filePath, allFieldsFilename, allMethodsFilename): 
    """ Returns a dictionary where the keys are the fields names (in abbreviated form)
        and the values are the convolution method names (e.g., "max", "min", "abs", or
        "minbut") 
        @filePath : str : Directory to where the all_fields and all_methods text files are.
        @allFieldsFilename : str : Name of the text file containing all the predictor fields
            (in abbreviated format) 
        @allMethodsFilename : str : Name of the text fiile containing the convolution 
            methods corresponding to each predictor field. 
    """

    #Read in the all fields and all methods files 
    all_fields = np.genfromtxt(f"{filePath}/{allFieldsFilename}", dtype='str')
    all_methods = np.genfromtxt(f"{filePath}/{allMethodsFilename}", dtype='str') 

    #Create dictionary 
    fields_methods_dict = dict(zip(all_fields, all_methods))

    return fields_methods_dict 


def get_list_from_file(filePath, fileName): 
    """ Returns a list given a file path and file name.
        @filePath : str : Path to text file 
        @fileName : str : Name of text file to read in
    """

    new_list = np.genfromtxt(f"{filePath}/{fileName}", dtype='str')

    new_list = list(new_list) 


    return new_list 


def main(): 

    #=================================================================
    #Parameters to set, which will be incorporated into the json file:
    #=================================================================
    

    json_output_dir = "/home/eric.loken/python_packages/frdd-wofs-phi/wofs_phi/config_json_files"

    #Specify output name here: 
    outfile_name = "config_training.json"

    full_outfile = f"{json_output_dir}/{outfile_name}"

    txt_file_dir = "../txt_files" #Relative path 

    all_fields_file = "all_fields.txt"
    all_methods_file = "all_methods.txt"
    singlePtFile = "single_point_fields.txt" 


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
    

    #TODO: Implement these methods
    methods_dict = get_methods_dict_from_files(txt_file_dir, all_fields_file, all_methods_file)
    single_point_fields = get_list_from_file(txt_file_dir, singlePtFile) 

    #=========================================================

    #Create the dictionary and save to file 

    json_dict = {"generate_forecasts": generate_forecasts, \
                "generate_reports": generate_reports, "save_npy": save_npy, \
                "save_ncdf": save_ncdf, "plot_forecasts": plot_forecasts, \
                "nc_outdir": nc_outdir, "use_ALL_files": use_ALL_files, \
                "fields_methods_dict": methods_dict, "single_point_fields": single_point_fields}


    #Save to file 
    save_dict_to_json(json_dict, full_outfile) 


    return 


if (__name__ == '__main__'): 


    main() 

