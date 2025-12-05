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
    save_predictors_to_npy = True #Tells whether or not to save the npy predictor files 
    save_predictions_to_ncdf = True #Tells whether or not to create/save the ncdf (realtime) files
    plot_forecasts = True #Tells whether or not to create the .png files for wofs viewer
    include_torp_in_predictors = False #Tells whether or not to include TORP as predictors

    nc_outdir = "." #Where to place the final netcdf files #Needed for real time
    #nc_outdir = "/home/eric.loken/python_packages/frdd-wofs-phi/wofs_phi/ncdf"

    #If True, use the ALL naming convention (will be true on cloud) 
    #If False, use the legacy naming convention (e.g., ENS, ENV, SVR, etc.) 
    use_ALL_files = True

    ps_version = 2 #Which version of ProbSevere to use

    dx_km = 3.0 #Horizontal grid spacing of wofs in km 

    ps_thresh = 0.01 #ps objects must have probs greater than or equal to this amount to be considered

    #Maximum amount of PS extrapolation time (used for setting min and max radius) 
    max_ps_extrap_time = 241.0 

    min_radius = 1.5 #in km (for probsevere objects) 

    max_radius = 1.5 #in km 

    #Amount of time (in minutes) to go back (relative to first PS file) 
    #ps_time_to_go_back = 180.0

    #nan_replace_value = 0.0 #Replace nans in wofs files with this value 
    
    #conv_type = "circle" #"circle" or "square" -- Tells how to do the convolutions

    #How far to "look" spatially in km for predictors 
    #predictor_radii_km = [0.0, 15.0, 30.0, 45.0, 60.0] 
    
    #TODO: Implement these methods
    methods_dict = get_methods_dict_from_files(txt_file_dir, all_fields_file, all_methods_file)
    single_point_fields = get_list_from_file(txt_file_dir, singlePtFile) 

    #Assumes times between 0000 and 1155 UTC were started the previous day. However, 
    #this list will be most useful/relevant for training. Actual start/end valid dates
    #are determined based on the WoFS summary file names and should not be dependent on
    #this list. 
    next_day_times = ["0000", "0005", "0010", "0015", "0020", "0025",\
                    "0030", "0035", "0040","0045", "0050", "0055",\
                    "0100", "0105", "0110", "0115", "0120", "0125",\
                    "0130", "0135", "0140","0145", "0150", "0155",\
                    "0200", "0205", "0210", "0215", "0220", "0225",\
                    "0230", "0235", "0240","0245", "0250", "0255",\
                    "0300", "0305", "0310", "0315", "0320", "0325",\
                    "0330", "0335", "0340","0345", "0350", "0355",\
                    "0400", "0405", "0410", "0415", "0420", "0425",\
                    "0430", "0435", "0440","0445", "0450", "0455",\
                    "0500", "0505", "0510", "0515", "0520", "0525",\
                    "0530", "0535", "0540","0545", "0550", "0555",\
                    "0600", "0605", "0610", "0615", "0620", "0625",\
                    "0630", "0635", "0640","0645", "0650", "0655",\
                    "0700", "0705", "0710", "0715", "0720", "0725",\
                    "0730", "0735", "0740","0745", "0750", "0755",\
                    "0800", "0805", "0810", "0815", "0820", "0825",\
                    "0830", "0835", "0840","0845", "0850", "0855",\
                    "0900", "0905", "0910", "0915", "0920", "0925",\
                    "0930", "0935", "0940","0945", "0950", "0955",\
                    "1000", "1005", "1010", "1015", "1020", "1025",\
                    "1030", "1035", "1040","1045", "1050", "1055", \
                    "1100", "1105", "1110", "1115", "1120", "1125",\
                    "1130", "1135", "1140", "1145", "1150", "1155"]

    #=========================================================

    #Create the dictionary and save to file 

    json_dict = {"generate_forecasts": generate_forecasts, \
                "generate_reports": generate_reports, \
                "save_predictors_to_npy": save_predictors_to_npy, \
                "save_predictions_to_ncdf": save_predictions_to_ncdf,\
                "plot_forecasts": plot_forecasts, \
                "include_torp_in_predictors":include_torp_in_predictors,\
                "nc_outdir": nc_outdir, "use_ALL_files": use_ALL_files, \
                "fields_methods_dict": methods_dict, "single_point_fields": single_point_fields,\
                "next_day_times":next_day_times, "ps_version": ps_version,\
                "dx_km":dx_km, "ps_thresh": ps_thresh, "max_ps_extrap_time":max_ps_extrap_time,\
                "min_radius":min_radius, "max_radius":max_radius}


    #Save to file 
    save_dict_to_json(json_dict, full_outfile) 


    return 


if (__name__ == '__main__'): 


    main() 

